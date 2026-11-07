"""Build and accept an offline natural-gene infilling dataset; never train."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sqlite3

from .data import GeneDatasetIndex
from .inventory import build_inventory


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_panels(repo_root: Path, output_dir: Path, seed=20261103):
    """Freeze diagnostic panels without replacing the full held-out pools."""
    inventory = json.loads((output_dir / 'INVENTORY_REPORT.json').read_text())
    annotation = json.loads((output_dir / 'ANNOTATION_REPORT.json').read_text())
    inventory_sha = sha256(output_dir / 'genomes.sqlite')
    if annotation['status'] != 'complete' or annotation['seed'] != seed or inventory['seed'] != seed:
        raise ValueError('Incomplete annotation build or inconsistent split seed')
    if annotation['inventory_sha256'] != inventory_sha:
        raise ValueError('Annotation build used a different inventory')
    def verify_counts(counts, excluded):
        accepted = counts.get('accepted_targets', 0)
        if accepted + sum(excluded.values()) != counts.get('source_annotations', 0):
            raise ValueError('Annotation input/output accounting mismatch')
        if accepted + counts.get('complete_eligible_validation_failed', 0) != counts.get('complete_eligible_candidates', 0):
            raise ValueError('Complete candidate accounting mismatch')
    verify_counts(annotation['counts'], annotation['excluded'])
    with sqlite3.connect(f"file:{output_dir / 'genomes.sqlite'}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        genomes = {row['genome_idx']: dict(row) for row in connection.execute('SELECT * FROM genomes WHERE eligible=1')}
    by_split = defaultdict(list)
    genome_database, target_counts, counts_by_kind = {}, Counter(), Counter()
    indexes = []
    paths = sorted((output_dir / 'targets').glob('shard_*.sqlite'))
    if [path.name for path in paths] != [f'shard_{i:04d}.sqlite' for i in range(16)]:
        raise ValueError('Expected the exact sixteen annotation shards')
    for path in paths:
        relative = str(path.relative_to(output_dir))
        audit = json.loads(path.with_suffix('.audit.json').read_text())
        database_sha = sha256(path)
        if audit['status'] != 'complete' or audit['seed'] != seed or audit['inventory_sha256'] != inventory_sha:
            raise ValueError(f'Stale or incomplete annotation shard: {path}')
        if audit['target_database_sha256'] != database_sha:
            raise ValueError(f'Annotation database changed after audit: {path}')
        verify_counts(audit['counts'], audit['excluded'])
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as connection:
            if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError(f'Invalid database: {path}')
            invalid = connection.execute('''SELECT COUNT(*) FROM targets WHERE
                gene_start<0 OR gene_end<=gene_start OR target_start<gene_start OR
                target_end>gene_end OR target_end<=target_start OR target_end-target_start>3000 OR
                strand NOT IN (-1,1) OR split NOT IN ('train','val','test') OR
                target_kind NOT IN ('whole_cds','partial_cds_window') OR
                (target_kind='whole_cds' AND (target_start!=gene_start OR target_end!=gene_end)) OR
                (target_kind='partial_cds_window' AND (gene_end-gene_start<=3000 OR target_end-target_start!=3000))''').fetchone()[0]
            if invalid:
                raise ValueError(f'Invalid target intervals in {path}: {invalid}')
            if connection.execute('SELECT COUNT(*) FROM targets').fetchone()[0] != audit['counts']['accepted_targets']:
                raise ValueError(f'Target count differs from shard audit: {path}')
            for idx, split, maximum, minimum, count in connection.execute('SELECT genome_idx,split,MAX(gene_end),MIN(gene_start),COUNT(*) FROM targets GROUP BY genome_idx,split'):
                if idx in genome_database or idx not in genomes:
                    raise ValueError(f'Duplicate or ineligible source index: {idx}')
                genome = genomes[idx]
                if split != genome['split'] or maximum > genome['length_bp'] or minimum < 0:
                    raise ValueError(f'Target bounds/split mismatch: {idx}')
                genome_database[idx] = relative
                by_split[split].append(idx)
                target_counts[split] += count
            for split, kind, count in connection.execute('SELECT split,target_kind,COUNT(*) FROM targets GROUP BY split,target_kind'):
                counts_by_kind[f'{split}/{kind}'] += count
        indexes.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': database_sha})
    if sum(target_counts.values()) != annotation['counts']['accepted_targets']:
        raise ValueError('Target count differs from aggregate annotation report')
    if any(not by_split[split] for split in ('train', 'val', 'test')):
        raise ValueError('An entire dataset split is empty')
    panels = {}
    for split, limit in (('train', 2048), ('val', 128), ('test', 128)):
        ids = sorted(by_split[split], key=lambda idx: hashlib.sha256(f'{seed}/{split}/{genomes[idx]["genome_id"]}'.encode()).digest())[:limit]
        rows = []
        for idx in ids:
            relative = genome_database[idx]
            with sqlite3.connect(f'file:{output_dir / relative}?mode=ro', uri=True) as connection:
                connection.row_factory = sqlite3.Row
                candidates = list(connection.execute('SELECT * FROM targets WHERE genome_idx=? ORDER BY target_id', (idx,)))
            whole = [row for row in candidates if row['target_kind'] == 'whole_cds']
            candidates = whole or candidates
            rng = random.Random(f'{seed}/target/{genomes[idx]["genome_id"]}')
            selected = dict(candidates[rng.randrange(len(candidates))])
            selected.update(target_db=relative, context_length_bp=genomes[idx]['length_bp'])
            rows.append(selected)
        panels[f'pilot_{split}'] = rows
    panels['generation_val'] = panels['pilot_val'][:16]
    panels['generation_test'] = panels['pilot_test'][:32]
    directory = output_dir / 'panels'
    directory.mkdir(exist_ok=True)
    validation_counts = Counter()
    with GeneDatasetIndex(repo_root, output_dir) as reader:
        for name, rows in panels.items():
            temporary = directory / f'{name}.jsonl.tmp'
            with temporary.open('w') as handle:
                for row in rows:
                    example = reader.read(row['target_db'], row['target_id'])
                    if example['target']['target_end'] - example['target']['target_start'] != row['target_end'] - row['target_start']:
                        raise AssertionError('Panel did not round-trip')
                    validation_counts[name] += 1
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            temporary.replace(directory / f'{name}.jsonl')
    report = {
        'status': 'offline_dataset_ready_training_not_started', 'seed': seed,
        'coordinate_convention': '0-based half-open; natural source orientation',
        'context_policy': 'full_original_fasta_record_no_context_truncation',
        'target_policy': 'partial=00; <=3000bp whole CDS; >3000bp one seeded contiguous 3000bp window',
        'no_lower_target_length_limit': True,
        'training_noise': 'not_frozen; offline index stores clean references and target intervals only',
        'genomes_with_targets_by_split': {key: len(value) for key, value in by_split.items()},
        'targets_by_split': dict(target_counts), 'targets_by_split_and_kind': dict(counts_by_kind),
        'eligible_genomes_without_targets': len(genomes) - len(genome_database),
        'full_source_length_max': max(genomes[idx]['length_bp'] for idx in genome_database),
        'panel_counts': dict(validation_counts),
        'panel_policy': 'seeded genome sampling without replacement, one uniform whole-CDS target per genome (partial fallback); generation panels are nested subsets',
        'panel_purpose': 'small diagnostic subsets; full val/test pools remain available; not a standalone preprint benchmark',
        'panel_context_total_bp': {name: sum(row['context_length_bp'] for row in rows) for name, rows in panels.items()},
        'panel_target_total_bp': {name: sum(row['target_end'] - row['target_start'] for row in rows) for name, rows in panels.items()},
        'leakage_checks': inventory['leakage_checks'],
        'near_homology_isolation': 'not_audited',
        'heldout_exposure': 'original AR validation; never original AR train, but historically used for validation',
        'fasta_sha256': inventory['fasta_sha256'], 'manifest_sha256': inventory['manifest_sha256'],
        'annotation_report_sha256': sha256(output_dir / 'ANNOTATION_REPORT.json'),
        'genomes_database_sha256': inventory_sha,
        'indexes': indexes,
    }
    atomic_json(output_dir / 'ACCEPTANCE.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('inventory', 'annotations', 'panels', 'all'), default='all')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261103)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    previous = args.output / 'ACCEPTANCE.json'
    if previous.exists():
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        previous.replace(args.output / f'ACCEPTANCE.superseded.{stamp}.json')
    if args.stage in ('inventory', 'all'):
        print(json.dumps(build_inventory(args.repo_root, args.output, args.seed), ensure_ascii=False), flush=True)
    if args.stage in ('annotations', 'all'):
        from .annotations import build_annotations
        print(json.dumps(build_annotations(args.repo_root, args.output, args.seed, args.workers), ensure_ascii=False), flush=True)
    if args.stage in ('panels', 'all'):
        print(json.dumps(freeze_panels(args.repo_root, args.output, args.seed), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
