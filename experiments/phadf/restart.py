"""CPDD restart trainer. Formal entry accepts only the AR base; resume accepts only this recipe.

Runs under torchrun with one sample per rank. Every rank advances the same serial
sampler/branch/mask streams and keeps its own slice, so the RNG state is identical on
all ranks and rank 0's checkpoint resumes the whole run.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
from torch.nn import functional as F

from experiments.phadf.amino_acid import amino_acid_loss
from experiments.phadf.data import GeneDatasetIndex, SourceSampler, read_panel
from experiments.phadf import generation_eval
from experiments.phadf.engram_adapt import checkpoint_kind
from experiments.phadf.grouped_lr import (
    DEFAULT_MIN_RATIO, DEFAULT_REWARMUP_STEPS, DEFAULT_WARMUP, apply_lrs, build_optimizer)
from experiments.phadf.model import DiffusionPhageSeek
from experiments.phadf.panel_eval import score_panel
from experiments.phadf.parallel import Parallel
from experiments.phadf.recipe import (
    MASK_FRACTION, STAGE_RADIUS, SUPERVISION_BUDGET, build_supervised_window, move_example,
    radius_for_stage)
from experiments.phadf.corruption import (
    SUBSTITUTION_RANGE, BranchSampler, draw_rate, draw_substitution)

RECIPE = 'cpdd_restart_v4'
RESUMABLE = ('cpdd_restart_v3', RECIPE)
DEFAULT_CLIP = 5.0
DEFAULT_AA_WEIGHT = 1.0
DEFAULT_LONG_PROBABILITY = 0.15


def assert_training_source(state, resume):
    kind = checkpoint_kind(state['model_state'])
    if resume:
        if state.get('run_config', {}).get('recipe') not in RESUMABLE or kind != 'restart':
            raise ValueError('Resume only a checkpoint written by the CPDD restart recipe')
        return
    if kind != 'ar':
        raise ValueError('Formal training only accepts the AR base checkpoint')


def _autocast(device):
    if device.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


def _loss_and_logits(model, example, aa_weight):
    ids, clean, positions = example['ids'], example['clean'], example['positions']
    row = model.logits(ids)[0, :, 1:5]
    selected = row[positions].float()
    truth = clean[0, positions] - 1
    nt = F.cross_entropy(selected, truth)
    correct = int((selected.argmax(-1) == truth).sum())
    aa_value, codons = None, 0
    loss = nt
    if aa_weight > 0:
        aa, codons = amino_acid_loss(row, clean[0], example)
        if codons:
            loss = loss + aa_weight * aa
            aa_value = float(aa.detach())
    return loss, dict(nt_loss=float(nt.detach()), correct=correct, supervised=int(truth.numel()),
                      aa_loss=aa_value, aa_supervised=codons), row


def combined_loss(model, example, aa_weight):
    """One forward, two terms: per-base CE plus the amino-acid NLL on whole CDS.

    The AA term is normalised per base so ``aa_weight`` compares like with like.
    """
    loss, stats, _ = _loss_and_logits(model, example, aa_weight)
    return loss, stats


def backward_example(model, example, aa_weight=0.0):
    """Accumulate this rank's gradients; returns the local scalar statistics."""
    device = example['ids'].device
    with _autocast(device):
        loss, stats = combined_loss(model, example, aa_weight)
    loss.backward()
    stats['loss'] = float(loss.detach())
    return stats


@dataclass(frozen=True)
class BatchPlan:
    """Everything about an update that every rank must agree on before it builds anything."""
    radius: int | None
    seed: int
    budget: int = SUPERVISION_BUDGET
    mask_fraction: float = MASK_FRACTION
    long_probability: float = DEFAULT_LONG_PROBABILITY
    substitution_range: tuple = SUBSTITUTION_RANGE

    def __post_init__(self):
        if not 0 <= self.long_probability <= 1:
            raise ValueError('Whole-source probability must be in [0, 1]')


def sample_rng(seed, step, index):
    """Derive an independent, resumable noise stream from step and rank."""
    digest = hashlib.sha256(f'{seed}/{step}/{index}'.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], 'big'))


def next_batch(DP, sampler, reader, plan, branch_sampler, mask_rng, device, step):
    """Draw one synchronized update and materialise only this rank's example."""
    branch = branch_sampler.next()
    rate = draw_rate(branch, mask_rng)
    substitution = draw_substitution(mask_rng, plan.substitution_range)
    whole_source = mask_rng.random() < plan.long_probability
    # Advance the source stream once per rank so the genomes stay disjoint within an
    # update and rank 0's checkpoint still describes the whole run; only this rank reads.
    picks = [sampler.next() for _ in range(DP.world)]
    database, target_id = picks[DP.rank]
    record = reader.read(database, target_id)
    primary = record['target']
    cursor = sampler.connections[database].execute(
        'SELECT * FROM targets WHERE genome_idx=? AND split=?', (primary['genome_idx'], 'train'))
    columns = [item[0] for item in cursor.description]
    siblings = [dict(zip(columns, row)) for row in cursor]
    example = build_supervised_window(
        record['sequence'], primary, siblings, None if whole_source else plan.radius,
        sample_rng(plan.seed, step, DP.rank), branch, rate, plan.mask_fraction,
        substitution=substitution, budget=plan.budget)
    draw = dict(branch=branch, rate=rate, substitution=substitution, whole_source=whole_source)
    return move_example(example, device), draw


def _load_state(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def execute(args):
    DP = Parallel(cpu=args.device == 'cpu', expect_world=args.expect_world)
    output = Path(args.output)
    if DP.is_root:
        output.mkdir(parents=True, exist_ok=True)
    DP.barrier()
    resume_path = Path(args.resume) if args.resume else None
    source_path = resume_path or Path(args.init)
    state = _load_state(source_path)
    assert_training_source(state, resume=resume_path is not None)
    device = DP.device
    model, _ = DiffusionPhageSeek.from_checkpoint(source_path)
    model.set_gradient_checkpointing(True, free=args.ckpt_free)
    model.to(device).train()
    optimizer = build_optimizer(model, args.peak, args.weight_decay, args.warmup)
    step = 0
    if resume_path is not None:
        optimizer.load_state_dict(state['optimizer_state'])
        step = int(state['step'])

    excluded = json.loads(Path(args.exclude_genomes_file).read_text()) if args.exclude_genomes_file else []
    sampler = SourceSampler(args.dataset_dir, args.seed, excluded)
    branch_sampler = BranchSampler(args.seed + 1)
    mask_rng = random.Random(args.seed + 2)
    migrating = resume_path is not None and state.get('run_config', {}).get('recipe') != RECIPE
    if resume_path is not None:
        sampler.load_state_dict(state['sampler_state'])
        branch_sampler.load_state_dict(state['branch_sampler_state'],
                                       accept_policy_change=migrating)
        mask_rng.setstate(state['mask_rng_state'])
    del state

    radius = radius_for_stage(args.stage)
    plan = BatchPlan(radius=radius, seed=args.seed, budget=args.supervision_budget,
                     mask_fraction=args.mask_fraction,
                     long_probability=args.long_probability,
                     substitution_range=(0.0, args.substitution_max))
    reader = GeneDatasetIndex(Path(args.repo_root), Path(args.dataset_dir))
    val_rows = list(read_panel(args.val_panel)) if args.val_panel else []
    metrics_path = Path(args.metrics) if args.metrics else None
    if metrics_path is not None and DP.is_root and resume_path is None:
        metrics_path.write_text('')
    generation_path = output / 'generation.jsonl'

    def generation_round():
        local = generation_eval.evaluate(model, reader, val_rows, device, radius, DP.rank, DP.world)
        merged = [item for part in DP.gather(local) for item in part]
        if DP.is_root:
            with generation_path.open('a') as handle:
                handle.write(json.dumps(dict(step=step, generation=generation_eval.aggregate(merged),
                                             tasks=merged)) + '\n')
        DP.barrier()

    recent_norms = []
    started = time.time()
    try:
        if args.gen_eval_at_start and val_rows:
            generation_round()
        while step < args.max_steps:
            if DP.any_true(bool(args.stop_file and Path(args.stop_file).exists())):
                break
            lrs = apply_lrs(optimizer, step, args.peak, args.warmup, args.decay_start,
                            args.decay_steps, DEFAULT_MIN_RATIO, args.switch_step, args.rewarmup_steps)
            optimizer.zero_grad(set_to_none=True)
            example, draw = next_batch(
                DP, sampler, reader, plan, branch_sampler, mask_rng, device, step)
            # Use the long-context activation-checkpoint budget for whole-source updates.
            model.set_gradient_checkpointing(
                True, free=args.ckpt_free_long if draw['whole_source'] else args.ckpt_free)
            stats = backward_example(model, example, args.aa_weight)
            if not math.isfinite(stats['loss']):
                raise FloatingPointError('Nonfinite training loss; optimizer not advanced')
            scale, total_supervised = DP.weight_masked_gradients(model, stats['supervised'])
            DP.average_gradients(model)
            # clip_grad_norm_ already returns the PRE-clip total norm, so there is no
            # reason to walk every parameter a second time to measure it.
            unclipped = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip,
                                                             error_if_nonfinite=True))
            optimizer.step()
            step += 1
            recent_norms.append(unclipped)
            if len(recent_norms) > 500:
                recent_norms.pop(0)

            if DP.is_root:
                row = dict(step=step, loss=stats['loss'], nt_loss=stats['nt_loss'],
                           accuracy=stats['correct'] / stats['supervised'],
                           branch=draw['branch'], rate=draw['rate'],
                           substitution=draw['substitution'],
                           substituted=example['substituted'],
                           substituted_correct=example['substituted_correct'],
                           whole_source=draw['whole_source'],
                           supervised_local=stats['supervised'],
                           supervised_total=total_supervised, targets=example['supervised_targets'],
                           window_tokens=example['window_tokens'],
                           mask_share=stats['supervised'] / example['window_tokens'],
                           grad_norm=unclipped,
                           clipped=bool(unclipped > args.clip), weight_scale=scale,
                           lr_adapter=lrs['adapter'], seconds=time.time() - started)
                if stats['aa_loss'] is not None:
                    row['aa_loss'] = stats['aa_loss']
                    row['aa_supervised'] = stats['aa_supervised']
                if step % args.diagnostic_steps == 0 and len(recent_norms) >= 20:
                    ordered = sorted(recent_norms)
                    row['grad_norm_p50'] = statistics.median(ordered)
                    row['grad_norm_p95'] = ordered[int(0.95 * (len(ordered) - 1))]
                    row['clip_fraction_recent'] = sum(1 for v in recent_norms if v > args.clip) / len(recent_norms)
            else:
                row = None

            if args.val_panel and args.val_every and step % args.val_every == 0:
                if DP.is_root:
                    short = score_panel(model, reader, val_rows, device, radius)
                    row['validation_short'] = short
                    if args.long_val_every and step % args.long_val_every == 0 and radius is not None:
                        long = score_panel(model, reader, val_rows, device, None)
                        row['validation_long'] = long
                        row['long_short_gap_full_ce'] = long['1.0']['ce'] - short['1.0']['ce']
                        row['long_short_gap_full_aa_identity'] = (
                            long['1.0'].get('aa_identity', 0.0) - short['1.0'].get('aa_identity', 0.0))
                DP.barrier()
            if DP.is_root and row is not None and metrics_path is not None:
                with metrics_path.open('a') as handle:
                    handle.write(json.dumps(row) + '\n')
            if val_rows and args.gen_eval_every and step % args.gen_eval_every == 0:
                generation_round()
            if step % args.checkpoint_every == 0 or step >= args.max_steps:
                if DP.is_root:
                    for path in checkpoint_paths(output, step, args.keep_steps):
                        _write_checkpoint(path, model, optimizer, sampler,
                                          branch_sampler, mask_rng, step, args)
                DP.barrier()
    finally:
        reader.close()
        DP.shutdown()
    return step


def checkpoint_paths(output, step, keep_steps):
    paths = [output / 'latest.pt']
    if step in keep_steps:
        paths.append(output / f'step_{step:05d}.pt')
    return paths


def parse_keep_steps(value):
    try:
        steps = frozenset(int(item) for item in value.split(',') if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError('keep-steps must be comma-separated integers') from error
    if not steps or any(step < 1 for step in steps):
        raise argparse.ArgumentTypeError('keep-steps must contain positive integers')
    return steps


def _write_checkpoint(path, model, optimizer, sampler, branch_sampler, mask_rng, step, args):
    payload = dict(
        model_state=model.state_dict(), model_config=model.config.to_dict(),
        optimizer_state=optimizer.state_dict(), sampler_state=sampler.state_dict(),
        branch_sampler_state=branch_sampler.state_dict(), mask_rng_state=mask_rng.getstate(),
        step=step, run_config=dict(
            recipe=RECIPE, stage=args.stage, peak=args.peak, warmup=args.warmup,
            decay_start=args.decay_start, decay_steps=args.decay_steps,
            switch_step=args.switch_step, rewarmup_steps=args.rewarmup_steps,
            max_steps=args.max_steps, radius=STAGE_RADIUS[args.stage], clip=args.clip,
            mask_fraction=args.mask_fraction, supervision_budget=args.supervision_budget,
            long_probability=args.long_probability, substitution_max=args.substitution_max,
            ckpt_free=args.ckpt_free, ckpt_free_long=args.ckpt_free_long,
            keep_steps=sorted(args.keep_steps), seed=args.seed, aa_weight=args.aa_weight))
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=Path.cwd())
    parser.add_argument('--exclude-genomes-file', type=Path)
    parser.add_argument('--val-panel', type=Path)
    parser.add_argument('--stage', choices=tuple(STAGE_RADIUS), default='B')
    parser.add_argument('--peak', type=float, default=5e-5)
    parser.add_argument('--warmup', type=int, default=DEFAULT_WARMUP)
    parser.add_argument('--decay-start', type=int, default=None,
                        help='WSD stays flat until this step; omit to keep the stable phase open')
    parser.add_argument('--decay-steps', type=int, default=None)
    parser.add_argument('--switch-step', type=int, default=None,
                        help='step at which the stage radius changed, triggers a short re-warmup')
    parser.add_argument('--rewarmup-steps', type=int, default=DEFAULT_REWARMUP_STEPS)
    parser.add_argument('--clip', type=float, default=DEFAULT_CLIP)
    parser.add_argument('--aa-weight', type=float, default=DEFAULT_AA_WEIGHT,
                        help='weight on the amino-acid NLL; 0 disables it')
    parser.add_argument('--mask-fraction', type=float, default=MASK_FRACTION,
                        help='cap on the share of the window that may be corrupted')
    parser.add_argument('--supervision-budget', type=int, default=SUPERVISION_BUDGET,
                        help='supervised bases per sample, held constant as the window grows')
    parser.add_argument('--long-probability', type=float, default=DEFAULT_LONG_PROBABILITY,
                        help='share of updates that use the whole source instead of the stage crop')
    parser.add_argument('--substitution-max', type=float, default=SUBSTITUTION_RANGE[1],
                        help='upper end of the per-update uniform-noise share; 0 disables revision training')
    parser.add_argument('--ckpt-free', type=int, default=8)
    parser.add_argument('--ckpt-free-long', type=int, default=4,
                        help='activation-checkpoint budget on whole-source steps')
    parser.add_argument('--max-steps', type=int, default=20000)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=20261104)
    parser.add_argument('--val-every', type=int, default=500)
    parser.add_argument('--long-val-every', type=int, default=2500)
    parser.add_argument('--gen-eval-every', type=int, default=0,
                        help='multi-step generation on the val panel every N steps, sharded over ranks; 0 disables')
    parser.add_argument('--gen-eval-at-start', action='store_true',
                        help='also run the generation evaluation once before the first update')
    parser.add_argument('--diagnostic-steps', type=int, default=50)
    parser.add_argument('--checkpoint-every', type=int, default=250)
    parser.add_argument('--keep-steps', type=parse_keep_steps, default=frozenset(),
                        help='also retain step_XXXXX.pt at these comma-separated steps')
    parser.add_argument('--expect-world', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--metrics', type=Path)
    parser.add_argument('--stop-file', type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if bool(args.init) == bool(args.resume):
        raise SystemExit('Pass exactly one of --init (AR base) or --resume (restart checkpoint)')
    return execute(args)


if __name__ == '__main__':
    main()
