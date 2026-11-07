"""Restore a fixed-length DNA interval with a local CPDD checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phadf.model import BASES, DiffusionPhageSeek, confidence_restore, encode


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True,
                        help='Local checkpoint, release directory, or Hugging Face model ID')
    parser.add_argument('--sequence', required=True, help='Complete ACGT context, without FASTA headers')
    parser.add_argument('--start', type=int, required=True, help='Zero-based target start')
    parser.add_argument('--end', type=int, required=True, help='Exclusive target end')
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2801)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--decoding', choices=('categorical', 'argmax'), default='categorical')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path, help='Optional JSON output path')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    sequence = ''.join(args.sequence.split()).upper()
    if not sequence or set(sequence) - set('ACGT'):
        raise ValueError('The demo accepts non-empty ACGT sequences only')
    if not 0 <= args.start < args.end <= len(sequence):
        raise ValueError('Target must satisfy 0 <= start < end <= sequence length')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')

    device = torch.device(args.device)
    model, state = DiffusionPhageSeek.from_checkpoint(args.checkpoint)
    model.to(device)
    clean = encode(sequence, device)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        restored = confidence_restore(
            model, clean, args.start, args.end, steps=args.steps, seed=args.seed,
            temperature=args.temperature, decoding=args.decoding)

    left = slice(0, args.start + 1)
    right = slice(args.end + 1, clean.shape[1])
    if not torch.equal(clean[:, left], restored[:, left]) or not torch.equal(clean[:, right], restored[:, right]):
        raise RuntimeError('Restoration changed visible conditioning bases')
    restored_sequence = ''.join(BASES[value - 1] for value in restored[0, 1:-1].tolist())
    target = sequence[args.start:args.end]
    prediction = restored_sequence[args.start:args.end]
    correct = sum(a == b for a, b in zip(target, prediction))
    result = dict(
        checkpoint=str(args.checkpoint), step=state.get('step'), device=str(device),
        decoding=args.decoding, steps=args.steps, seed=args.seed,
        temperature=args.temperature, target_start=args.start, target_end=args.end,
        original=sequence, restored=restored_sequence, target_original=target,
        target_restored=prediction, target_accuracy=correct / len(target),
        visible_unchanged=True)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
