"""Freeze synthetic sixteen-shard panels and reject stale audit artifacts."""

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from experiments.phadf.build_dataset import freeze_panels
from experiments.phadf.inventory import build_inventory


def _write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def _counts(number):
    return {
        "accepted_targets": number,
        "source_annotations": number,
        "complete_eligible_candidates": number,
        "complete_eligible_validation_failed": 0,
    }


class PanelAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="phadf_panels_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "dataset"
        (self.root / "data/train").mkdir(parents=True)
        (self.root / "data/split").mkdir(parents=True)
        entries = [
            ("g0", "train", "CCATGAAATAATTTT"),
            ("g1", "val", "GGATGCCCTAATTTT"),
            ("g2", "val", "CCATGGGATAAGGGG"),
        ]
        (self.root / "data/train/training_set.fasta").write_text("".join(
            f">{name}\n{sequence[:7]}\n{sequence[7:]}\n"
            for name, _, sequence in entries
        ))
        (self.root / "data/split/manifest.csv").write_text(
            "id,source,split,length_bp\n" + "".join(
                f"{name},TEST,{split},{len(sequence)}\n"
                for name, split, sequence in entries
            )
        )
        build_inventory(self.root, self.output)
        inventory_path = self.output / "genomes.sqlite"
        self.inventory_hash = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
        with sqlite3.connect(inventory_path) as connection:
            connection.row_factory = sqlite3.Row
            genomes = {row["genome_idx"]: dict(row) for row in connection.execute("SELECT * FROM genomes")}
        (self.output / "targets").mkdir()
        for index in range(16):
            path = self.output / "targets" / f"shard_{index:04d}.sqlite"
            with sqlite3.connect(path) as connection:
                connection.execute("""CREATE TABLE targets (
                    target_id TEXT PRIMARY KEY, genome_idx INTEGER, genome_id TEXT,
                    gene_id TEXT, split TEXT, gene_start INTEGER, gene_end INTEGER,
                    strand INTEGER, genetic_code INTEGER, target_start INTEGER,
                    target_end INTEGER, target_kind TEXT, protein_sha256 TEXT,
                    gene_length INTEGER, target_length INTEGER, genome_length INTEGER,
                    source TEXT)""")
                if index < 3:
                    genome = genomes[index]
                    connection.execute("INSERT INTO targets VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                        f"target_{index}", index, genome["genome_id"], f"g{index}_1",
                        genome["split"], 2, 11, 1, 11, 2, 11, "whole_cds",
                        hashlib.sha256(b"fixture_protein").hexdigest(), 9, 9,
                        genome["length_bp"], "TEST",
                    ))
            audit = {
                "status": "complete", "seed": 20261103,
                "inventory_sha256": self.inventory_hash,
                "target_database_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "counts": _counts(int(index < 3)), "excluded": {},
            }
            _write_json(path.with_suffix(".audit.json"), audit)
        self.annotation_path = self.output / "ANNOTATION_REPORT.json"
        _write_json(self.annotation_path, {
            "status": "complete", "seed": 20261103,
            "inventory_sha256": self.inventory_hash,
            "counts": _counts(3), "excluded": {},
        })

    def _replace_report(self, **updates):
        report = json.loads(self.annotation_path.read_text())
        report.update(updates)
        _write_json(self.annotation_path, report)

    def _assert_failure(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            freeze_panels(self.root, self.output)
        self.assertFalse((self.output / "ACCEPTANCE.json").exists())

    def test_freeze_complete_sixteen_shard_fixture(self):
        report = freeze_panels(self.root, self.output)
        self.assertEqual(report["status"], "offline_dataset_ready_training_not_started")
        self.assertEqual(report["targets_by_split"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual(len(report["indexes"]), 16)
        self.assertEqual(report["panel_counts"], {
            "pilot_train": 1, "pilot_val": 1, "pilot_test": 1,
            "generation_val": 1, "generation_test": 1,
        })
        for split in ("train", "val", "test"):
            rows = (self.output / "panels" / f"pilot_{split}.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 1)
            target = json.loads(rows[0])
            self.assertEqual(target["split"], split)
            self.assertEqual(target["target_end"] - target["target_start"], 9)
        self.assertEqual(json.loads((self.output / "ACCEPTANCE.json").read_text()), report)

    def test_stale_inventory_hash_is_rejected(self):
        self._replace_report(inventory_sha256="0" * 64)
        self._assert_failure("different inventory")

    def test_changed_target_database_is_rejected(self):
        path = self.output / "targets/shard_0000.sqlite"
        with sqlite3.connect(path) as connection:
            connection.execute("UPDATE targets SET target_length=8")
        self._assert_failure("database changed after audit")

    def test_inconsistent_report_accounting_is_rejected(self):
        counts = _counts(3)
        counts["source_annotations"] = 4
        self._replace_report(counts=counts)
        self._assert_failure("input/output accounting mismatch")

    def test_consistent_but_wrong_aggregate_count_is_rejected(self):
        self._replace_report(counts=_counts(4))
        self._assert_failure("Target count differs from aggregate")

    def test_missing_shard_is_rejected(self):
        (self.output / "targets/shard_0015.sqlite").unlink()
        self._assert_failure("exact sixteen annotation shards")

    def test_failed_shard_audit_is_rejected(self):
        path = self.output / "targets/shard_0000.audit.json"
        audit = json.loads(path.read_text())
        audit["status"] = "failed"
        _write_json(path, audit)
        self._assert_failure("Stale or incomplete annotation shard")

    def test_inconsistent_seed_is_rejected(self):
        self._replace_report(seed=123)
        self._assert_failure("inconsistent split seed")


if __name__ == "__main__":
    unittest.main()
