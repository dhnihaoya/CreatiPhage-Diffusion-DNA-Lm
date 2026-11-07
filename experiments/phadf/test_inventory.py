"""Exercise inventory isolation and FASTA offsets using synthetic records."""

import csv
import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from experiments.phadf.inventory import build_inventory


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="phadf_inventory_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "data/train").mkdir(parents=True)
        (self.root / "data/split").mkdir(parents=True)
        self.fasta = self.root / "data/train/training_set.fasta"
        self.manifest = self.root / "data/split/manifest.csv"
        self.entries = [
            ("a", "train", "ACGTTT"),
            ("b", "train", "AAACGT"),
            ("c|provirus_1_6", "train", "AGAGAG"),
            ("c|provirus_8_13", "val", "TCTCTA"),
            ("d", "val", "ACACAC"),
            ("e", "val", "AGCTTA"),
            ("f", "val", "AATTCC"),
            ("g", "val", "GGTTAA"),
            ("h", "train", "ACNGTT"),
            ("i", "train", "AACCTT"),
        ]
        self.fasta.write_bytes(b"".join(
            (">" + name + " description\n" + sequence[:3] + "\n" + sequence[3:] + "\n").encode()
            for name, _, sequence in self.entries
        ))
        self._write_manifest()

    def _write_manifest(self, rows=None):
        if rows is None:
            rows = [(name, "TEST", split, len(sequence)) for name, split, sequence in self.entries]
        with self.manifest.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "source", "split", "length_bp"])
            writer.writerows(rows)

    def _build(self, output_name="out"):
        output = self.root / output_name
        report = build_inventory(self.root, output)
        connection = sqlite3.connect(output / "genomes.sqlite")
        try:
            connection.row_factory = sqlite3.Row
            rows = {row["genome_id"]: dict(row) for row in connection.execute("SELECT * FROM genomes")}
        finally:
            connection.close()
        return report, rows

    def test_exact_rc_representative_and_alias(self):
        _, rows = self._build()
        self.assertNotEqual(rows["a"]["forward_sha256"], rows["b"]["forward_sha256"])
        self.assertEqual(rows["a"]["canonical_sha256"], rows["b"]["canonical_sha256"])
        self.assertEqual(rows["a"]["eligible"], 1)
        self.assertEqual(rows["b"]["representative_id"], "a")
        self.assertEqual(rows["b"]["eligible"], 0)
        self.assertIn("exact_rc_alias", rows["b"]["exclusion"])

    def test_same_base_cross_original_split_is_quarantined(self):
        report, rows = self._build()
        training = rows["c|provirus_1_6"]
        validation = rows["c|provirus_8_13"]
        self.assertNotEqual(training["canonical_sha256"], validation["canonical_sha256"])
        self.assertEqual(training["isolation_group"], validation["isolation_group"])
        self.assertEqual(training["split"], "train")
        self.assertEqual(validation["original_split"], "val")
        self.assertEqual(validation["split"], "quarantine")
        self.assertEqual(validation["eligible"], 0)
        self.assertIn("original_val_component_has_train", validation["exclusion"])
        self.assertTrue(all(value == 0 for value in report["leakage_checks"].values()))

    def test_non_acgt_context_is_excluded(self):
        _, rows = self._build()
        self.assertEqual(rows["h"]["ambiguous_count"], 1)
        self.assertEqual(rows["h"]["eligible"], 0)
        self.assertEqual(rows["h"]["exclusion"], "non_acgt_context")
        self.assertTrue(all(row["ambiguous_count"] == 0 for row in rows.values() if row["eligible"]))

    def test_byte_offsets_restore_multiline_source_and_hashes(self):
        report, rows = self._build()
        with self.fasta.open("rb") as handle:
            for index, (name, _, reference) in enumerate(self.entries):
                row = rows[name]
                self.assertEqual(row["genome_idx"], index)
                handle.seek(row["seq_offset"])
                raw = handle.read(row["seq_end"] - row["seq_offset"])
                self.assertNotIn(b">", raw)
                sequence = b"".join(raw.split()).upper()
                self.assertEqual(sequence.decode(), reference)
                self.assertEqual(hashlib.sha256(sequence).hexdigest(), row["forward_sha256"])
        self.assertEqual(report["fasta_sha256"], hashlib.sha256(self.fasta.read_bytes()).hexdigest())
        self.assertEqual(report["manifest_sha256"], hashlib.sha256(self.manifest.read_bytes()).hexdigest())

    def test_determinism_and_heldout_provenance(self):
        first_report, first_rows = self._build("first")
        second_report, second_rows = self._build("second")
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(first_report, second_report)
        self.assertEqual(first_report["eligible_split_counts"], {"train": 3, "val": 2, "test": 2})
        for row in first_rows.values():
            if row["eligible"] and row["split"] in {"val", "test"}:
                self.assertEqual(row["original_split"], "val")
        self.assertEqual(first_report["near_homology_isolation"], "not_audited")

    def test_duplicate_manifest_identifier_is_rejected(self):
        with self.manifest.open("a") as handle:
            handle.write("a,TEST,train,6\n")
        with self.assertRaisesRegex(ValueError, "Duplicate manifest identifier"):
            self._build()

    def test_manifest_length_mismatch_is_rejected(self):
        rows = [(name, "TEST", split, len(sequence)) for name, split, sequence in self.entries]
        rows[0] = ("a", "TEST", "train", 7)
        self._write_manifest(rows)
        with self.assertRaisesRegex(ValueError, "Manifest length mismatch"):
            self._build()

    def test_manifest_missing_fasta_record_is_rejected(self):
        with self.manifest.open("a") as handle:
            handle.write("missing,TEST,val,6\n")
        with self.assertRaisesRegex(ValueError, "FASTA missing"):
            self._build()

    def test_fasta_record_absent_from_manifest_is_rejected(self):
        with self.fasta.open("ab") as handle:
            handle.write(b">extra\nACGTAC\n")
        with self.assertRaisesRegex(ValueError, "unmanifested FASTA identifier"):
            self._build()


if __name__ == "__main__":
    unittest.main()
