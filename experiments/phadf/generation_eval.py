"""Iterative whole-CDS and gap restoration on a local validation panel."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch

from experiments.phadf.data import GeneDatasetIndex, read_panel, window_bounds
from experiments.phadf.model import BASES, DiffusionPhageSeek, confidence_restore, encode
from experiments.phadf.panel_eval import _autocast, coding_direction
from experiments.phadf.protein_metrics import score_generated

SEEDS = (2801, 2802, 2803, 2804)
CONDITIONS = (
    ('whole_argmax_s32', None, 'argmax', 32, (SEEDS[0],)),
    ('gap12_categorical_s32', 12, 'categorical', 32, SEEDS),
    ('gap24_categorical_s32', 24, 'categorical', 32, SEEDS),
    ('gap48_categorical_s32', 48, 'categorical', 32, SEEDS),
)
METRICS = ('internal_stop', 'aa_identity', 'hole_nt_accuracy', 'distinct_codons', 'top_codon_share')


def gap_interval(rel_start, rel_end, strand, gap):
    """Return a codon-aligned central gap in the coding direction."""
    length = rel_end - rel_start
    if gap % 3 or length % 3 or gap >= length - 6:
        raise ValueError('Gap must be codon aligned and leave the start and stop codons visible')
    offset = 3 * ((length // 3 - gap // 3) // 2)
    if strand == 1:
        return rel_start + offset, rel_start + offset + gap
    return rel_end - offset - gap, rel_end - offset


def tasks(rows):
    return [dict(row=row, condition=name, gap=gap, decoding=decoding, steps=steps, seed=seed)
            for row in rows for name, gap, decoding, steps, seeds in CONDITIONS for seed in seeds]


def run_task(model, reader, task, device, radius):
    row = task['row']
    record = reader.read(row['target_db'], row['target_id'])
    target = record['target']
    start, end = target['target_start'], target['target_end']
    left, right = window_bounds(len(record['sequence']), start, end, radius)
    sequence = record['sequence'][left:right]
    rel_start, rel_end = start - left, end - left
    strand, code = target['strand'], target['genetic_code']
    hole = (rel_start, rel_end) if task['gap'] is None else gap_interval(
        rel_start, rel_end, strand, task['gap'])
    clean = encode(sequence, device)
    with _autocast(device):
        filled = confidence_restore(model, clean, *hole, steps=task['steps'], seed=task['seed'],
                                    decoding=task['decoding'])
    written = ''.join(BASES[value - 1] for value in filled[0, rel_start + 1:rel_end + 1].tolist())
    reference = sequence[rel_start:rel_end]
    scored = score_generated(coding_direction(written, strand), coding_direction(reference, strand), code)
    inside = slice(hole[0] - rel_start, hole[1] - rel_start)
    hole_matches = sum(a == b for a, b in zip(written[inside], reference[inside]))
    return dict(target_id=row['target_id'], condition=task['condition'], seed=task['seed'],
                hole_nt_accuracy=hole_matches / (hole[1] - hole[0]),
                aa_identity=scored.get('aa_identity'),
                internal_stop=scored['orf'].get('internal_stop'),
                distinct_codons=scored['diversity']['distinct_codons'],
                top_codon_share=scored['diversity']['top_codon_share'])


def evaluate(model, reader, rows, device, radius, rank=0, world=1):
    """Evaluate this rank's share of the task list."""
    was_training = model.training
    model.eval()
    try:
        return [run_task(model, reader, task, device, radius)
                for index, task in enumerate(tasks(rows)) if index % world == rank]
    finally:
        model.train(was_training)


def per_cds(results, condition, key):
    """Average replicate runs within each CDS before panel aggregation."""
    values = defaultdict(list)
    for row in results:
        if row['condition'] == condition and row.get(key) is not None:
            values[row['target_id']].append(float(row[key]))
    return {target: sum(items) / len(items) for target, items in values.items()}


def aggregate(results):
    out = {}
    for name, *_ in CONDITIONS:
        runs = [row for row in results if row['condition'] == name]
        if not runs:
            continue
        item = dict(n_cds=len({row['target_id'] for row in runs}), n_runs=len(runs))
        for key in METRICS:
            cds = per_cds(results, name, key)
            item[key if key != 'internal_stop' else 'internal_stop_rate'] = (
                sum(cds.values()) / len(cds) if cds else None)
        out[name] = item
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--dataset-dir', type=Path, default=Path('data/cpdd_dataset'))
    parser.add_argument('--val-panel', type=Path,
                        default=Path('data/cpdd_dataset/panels/pilot_val.jsonl'))
    parser.add_argument('--radius', type=int, default=16384)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args(argv)
    device = torch.device('cuda' if args.device == 'cuda' or
                          (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    model, state = DiffusionPhageSeek.from_checkpoint(args.checkpoint)
    model.to(device)
    reader = GeneDatasetIndex(args.repo_root, args.dataset_dir)
    try:
        results = evaluate(model, reader, list(read_panel(args.val_panel)), device, args.radius, 0, 1)
    finally:
        reader.close()
    summary = dict(checkpoint=str(args.checkpoint), step=state.get('step'), radius=args.radius,
                   generation=aggregate(results), tasks=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary['generation'], indent=1))


if __name__ == '__main__':
    main()
