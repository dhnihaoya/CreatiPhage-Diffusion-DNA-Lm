"""Generate a short DNA sequence with the released CreatiPhage AR checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phageseek.model import PhageSeek


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True,
                        help='Local checkpoint, release directory, or Hugging Face model ID')
    parser.add_argument('--primer', default='ATGCGT', help='ACGT primer; BOS is added automatically')
    parser.add_argument('--length', type=int, default=16, help='Total token length, including BOS and primer')
    parser.add_argument('--temperature', type=float, default=0.95)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2801)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    primer = ''.join(args.primer.split()).upper()
    if not primer or set(primer) - set('ACGT'):
        raise ValueError('The primer must contain ACGT only')
    if args.length <= len(primer) + 1:
        raise ValueError('--length must exceed BOS plus the primer length')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')

    model, state = PhageSeek.from_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    model.to(device).eval()
    lookup = {'A': 1, 'T': 2, 'C': 3, 'G': 4}
    ids = torch.tensor([[0] + [lookup[base] for base in primer]], dtype=torch.long, device=device)
    torch.manual_seed(args.seed)
    generated = model.generate(
        prime=ids, seq_len=args.length, temperature=args.temperature,
        top_k=args.top_k, use_cache=True, eos_id=5)
    tokens = generated[0].tolist()
    reverse = {1: 'A', 2: 'T', 3: 'C', 4: 'G', 5: '#', 0: '*'}
    sequence = ''.join(reverse.get(token, '?') for token in tokens[1:])
    result = dict(checkpoint=str(args.checkpoint), step=state.get('step'), device=str(device),
                  primer=primer, requested_total_tokens=args.length, total_tokens=len(tokens),
                  temperature=args.temperature, top_k=args.top_k, seed=args.seed,
                  sequence=sequence, continuation=sequence[len(primer):],
                  special_tokens='The initial BOS is omitted; # is EOS and * is PAD after EOS')
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
