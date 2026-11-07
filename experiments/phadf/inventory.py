"""Build an audited sequence inventory without duplicating the source FASTA."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


_COMPLEMENT = bytes.maketrans(b"ACGTNRYKMSWBDHVU", b"TGCANYRMKSWVHDBA")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fasta_records(path: Path, digest):
    """Yield normalized sequence bytes and seekable raw sequence-line ranges."""
    with path.open("rb") as handle:
        name, start, pieces = None, None, []
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if line:
                digest.update(line)
            if not line or line.startswith(b">"):
                if name is not None:
                    sequence = b"".join(pieces).upper()
                    if not sequence:
                        raise ValueError(f"Empty FASTA record: {name}")
                    yield name, sequence, start, line_start
                if not line:
                    break
                fields = line[1:].strip().split()
                if not fields:
                    raise ValueError("Empty FASTA identifier")
                name = fields[0].decode("utf-8")
                start, pieces = handle.tell(), []
            elif name is None:
                if line.strip():
                    raise ValueError("Sequence before first FASTA header")
            else:
                pieces.append(b"".join(line.split()))


def _length_stats(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "total_bp": 0}
    n = len(values)
    return {
        "count": n, "total_bp": sum(values), "min": values[0],
        "median": (values[(n - 1) // 2] + values[n // 2]) / 2,
        "p90": values[round((n - 1) * 0.9)],
        "p99": values[round((n - 1) * 0.99)], "max": values[-1],
        "mean": sum(values) / n,
    }


def build_inventory(repo_root: Path, output_dir: Path, seed: int = 20261103) -> dict:
    """Audit FASTA records and freeze exact/RC plus source-accession isolation."""
    repo_root, output_dir = Path(repo_root), Path(output_dir)
    fasta = repo_root / "data/train/training_set.fasta"
    manifest = repo_root / "data/split/manifest.csv"
    originals = {}
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["id"]
            if name in originals:
                raise ValueError(f"Duplicate manifest identifier: {name}")
            if row["split"] not in {"train", "val"}:
                raise ValueError(f"Unexpected original split: {row['split']}")
            originals[name] = row
    records, seen, by_hash, by_base = [], set(), defaultdict(list), defaultdict(list)
    fasta_digest = hashlib.sha256()
    for index, (name, sequence, offset, end) in enumerate(_fasta_records(fasta, fasta_digest)):
        if name in seen or name not in originals:
            raise ValueError(f"Duplicate or unmanifested FASTA identifier: {name}")
        original = originals[name]
        if len(sequence) != int(original["length_bp"]):
            raise ValueError(f"Manifest length mismatch: {name}")
        seen.add(name)
        canonical = min(sequence, sequence.translate(_COMPLEMENT)[::-1])
        canonical_hash = hashlib.sha256(canonical).hexdigest()
        base = name.split("|", 1)[0]
        record = {
            "genome_idx": index, "genome_id": name, "source": original["source"],
            "original_split": original["split"], "length_bp": len(sequence),
            "seq_offset": offset, "seq_end": end,
            "forward_sha256": hashlib.sha256(sequence).hexdigest(),
            "canonical_sha256": canonical_hash, "base_accession": base,
            "ambiguous_count": len(sequence) - sum(sequence.count(base) for base in (b"A", b"C", b"G", b"T")),
        }
        records.append(record)
        by_hash[canonical_hash].append(index)
        by_base[base].append(index)
    if seen != originals.keys():
        raise ValueError(f"FASTA missing {len(originals.keys() - seen)} manifest records")

    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for groups in (by_hash, by_base):
        for members in groups.values():
            for member in members[1:]:
                parent[find(member)] = find(members[0])
    components = defaultdict(list)
    for index in range(len(records)):
        components[find(index)].append(index)
    representatives = {
        key: min(members, key=lambda index: records[index]["genome_id"])
        for key, members in by_hash.items()
    }
    strata, component_split = defaultdict(list), {}
    mixed_components = 0
    for root, members in components.items():
        ids = sorted(records[index]["genome_id"] for index in members)
        group = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        for index in members:
            records[index]["isolation_group"] = group
        original_splits = {records[index]["original_split"] for index in members}
        if "train" in original_splits:
            component_split[root] = "train"
            mixed_components += len(original_splits) > 1
        else:
            source = min(records[index]["source"] for index in members)
            strata[source].append((group, root))
    rng = random.Random(seed)
    for source, entries in sorted(strata.items()):
        entries.sort()
        rng.shuffle(entries)
        midpoint = len(entries) // 2
        for i, (_, root) in enumerate(entries):
            component_split[root] = "val" if i < midpoint else "test"
    for index, record in enumerate(records):
        representative = records[representatives[record["canonical_sha256"]]]["genome_id"]
        record["representative_id"] = representative
        split = component_split[find(index)]
        reasons = []
        if split == "train" and record["original_split"] == "val":
            split = "quarantine"
            reasons.append("original_val_component_has_train")
        if representative != record["genome_id"]:
            reasons.append("exact_rc_alias")
        if record["ambiguous_count"]:
            reasons.append("non_acgt_context")
        record.update(split=split, eligible=int(not reasons), exclusion=";".join(reasons))

    eligible = [record for record in records if record["eligible"]]
    group_splits, hash_splits = defaultdict(set), defaultdict(set)
    for record in eligible:
        group_splits[record["isolation_group"]].add(record["split"])
        hash_splits[record["canonical_sha256"]].add(record["split"])
    leakage = {
        "eligible_isolation_groups_cross_split": sum(len(splits) > 1 for splits in group_splits.values()),
        "eligible_canonical_hashes_cross_split": sum(len(splits) > 1 for splits in hash_splits.values()),
        "eligible_heldout_with_original_train": sum(record["original_split"] == "train" for record in eligible if record["split"] in {"val", "test"}),
    }
    if any(leakage.values()):
        raise AssertionError(f"Inventory isolation failed: {leakage}")
    metadata = {
        "schema_version": 1,
        "sequence_source": "原始 FASTA 完整记录；含组装 contig 和 provirus 区段，不保证生物学完整基因组。",
        "offset_convention": "seq_offset/seq_end 为原始 FASTA 的字节半开区间，包含序列行及换行、不含 header；读取后去空白并转大写。",
        "hash_convention": "forward_sha256 对去空白大写序列计算；canonical_sha256 对正向与反向互补中字典序较小者计算，非 ACGT 上下文全部排除。",
        "representative_policy": "每个 exact/RC 组选择 genome_id 字典序最小记录；其余为 alias，不参与训练评测。",
        "split_policy": "exact/RC 与相同 base_accession 合并连通分量；含原 train 分量中的原 val 记录隔离，纯原 val 分量按来源分层随机等分。",
        "source_stratification": "分量来源混合时取来源字符串字典序最小值；各来源 val 取 floor(n/2)，剩余分给 test。",
        "pretraining_exposure": "train 来自原 AR train；val/test 来自原 AR val，已用于历史基座验证，不是从未接触的外部测试。",
        "near_homology_isolation": "not_audited",
        "column_notes": {
            "genome_idx": "原 FASTA 顺序，0 起，排除记录也保留索引。",
            "genome_id": "原 FASTA header 的首个空白分隔字段，唯一。",
            "source": "原 manifest 来源。", "original_split": "原 AR 的 train/val 身份。",
            "length_bp": "去空白大写后序列长度。", "ambiguous_count": "非 A/C/G/T 字符数。",
            "base_accession": "genome_id 第一个竖线前的来源记录 ID。",
            "split": "新 train/val/test/quarantine；必须同时过滤 eligible=1。",
            "eligible": "1 为可用代表记录，0 为排除；尚未应用基因注释质量筛选。",
            "exclusion": "分号分隔的全部排除原因，空串代表可用。",
            "representative_id": "该 exact/RC 组字典序首个 ID；alias 坐标不可直接迁移到代表序列。",
            "isolation_group": "同一来源或 exact/RC 关联分量全部排序 ID 的 SHA256。",
        },
    }
    report = {
        "schema_version": 1, "seed": seed, "fasta_path": fasta.relative_to(repo_root).as_posix(),
        "manifest_path": manifest.relative_to(repo_root).as_posix(), "fasta_sha256": fasta_digest.hexdigest(),
        "manifest_sha256": _sha256(manifest), "record_count": len(records),
        "canonical_group_count": len(by_hash), "isolation_group_count": len(components),
        "mixed_original_split_components": mixed_components,
        "original_split_counts": dict(Counter(record["original_split"] for record in records)),
        "inventory_split_counts": dict(Counter(record["split"] for record in records)),
        "eligible_split_counts": dict(Counter(record["split"] for record in eligible)),
        "exclusion_reason_counts": dict(Counter(reason for record in records for reason in record["exclusion"].split(";") if reason)),
        "all_record_context_lengths": _length_stats(record["length_bp"] for record in records),
        "eligible_context_lengths_by_split": {split: _length_stats(record["length_bp"] for record in eligible if record["split"] == split) for split in ("train", "val", "test")},
        "leakage_checks": leakage, "exact_rc_isolated": True,
        "near_homology_isolation": "not_audited",
        "pretrain_val_exposure": metadata["pretraining_exposure"], "metadata": metadata,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    database_tmp = output_dir / "genomes.sqlite.tmp"
    if database_tmp.exists():
        raise FileExistsError(f"Refusing to overwrite unfinished inventory: {database_tmp}")
    connection = sqlite3.connect(database_tmp)
    try:
        connection.execute("""CREATE TABLE genomes (
            genome_idx INTEGER PRIMARY KEY, genome_id TEXT UNIQUE NOT NULL,
            source TEXT, original_split TEXT, length_bp INTEGER,
            seq_offset INTEGER, seq_end INTEGER, forward_sha256 TEXT,
            canonical_sha256 TEXT, base_accession TEXT, split TEXT,
            eligible INTEGER, exclusion TEXT, representative_id TEXT,
            isolation_group TEXT, ambiguous_count INTEGER)""")
        columns = list(records[0]) if records else []
        if columns:
            connection.executemany(
                f"INSERT INTO genomes ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                ([record[column] for column in columns] for record in records),
            )
        connection.execute("CREATE INDEX genomes_split_eligible ON genomes(split, eligible)")
        connection.execute("CREATE INDEX genomes_isolation ON genomes(isolation_group)")
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", ((key, json.dumps(value, ensure_ascii=False)) for key, value in metadata.items()))
        connection.commit()
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("SQLite integrity check failed")
    finally:
        connection.close()
    report_tmp = output_dir / "INVENTORY_REPORT.json.tmp"
    report_tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    os.replace(database_tmp, output_dir / "genomes.sqlite")
    os.replace(report_tmp, output_dir / "INVENTORY_REPORT.json")
    return report
