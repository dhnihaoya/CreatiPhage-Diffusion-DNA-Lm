"""Synchronous data-parallel helpers for the CPDD restart.

Manual all-reduce rather than DistributedDataParallel: every rank sees a different
window length, so DDP's bucketing and unused-parameter detection would fight the
variable shapes. Each rank advances the SAME serial sampler/noise streams and keeps
only its own slice, which keeps the RNG state identical everywhere and makes a
checkpoint from rank 0 sufficient to resume the whole run.
"""
from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist

BUCKET_BYTES = 64 * 1024 ** 2


class Parallel:
    def __init__(self, cpu=False, expect_world=None):
        self.rank = int(os.environ.get('RANK', '0'))
        self.world = int(os.environ.get('WORLD_SIZE', '1'))
        self.local = int(os.environ.get('LOCAL_RANK', '0'))
        self.distributed = self.world > 1 or 'RANK' in os.environ
        self.device = torch.device('cpu' if cpu else f'cuda:{self.local}')
        if not cpu:
            torch.cuda.set_device(self.device)
        if expect_world is not None and self.world != expect_world:
            raise ValueError(f'Expected world size {expect_world}, got {self.world}')
        if self.distributed:
            dist.init_process_group('gloo' if cpu else 'nccl',
                                    timeout=datetime.timedelta(hours=2))

    @property
    def is_root(self):
        return self.rank == 0

    def barrier(self):
        if self.distributed:
            dist.barrier()

    def gather(self, value):
        if not self.distributed:
            return [value]
        result = [None] * self.world
        dist.all_gather_object(result, value)
        return result

    def any_true(self, value):
        if not self.distributed:
            return bool(value)
        flag = torch.tensor(int(bool(value)), device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    def weight_masked_gradients(self, model, local_tokens):
        """Turn per-rank mean gradients into a global masked-token mean."""
        if not self.distributed:
            return 1.0, int(local_tokens)
        counts = torch.tensor([local_tokens, int(local_tokens <= 0)],
                              dtype=torch.int64, device=self.device)
        dist.all_reduce(counts)
        total, invalid = counts.tolist()
        if invalid or total <= 0:
            raise ValueError('Every sample must supervise at least one masked token')
        # average_gradients divides by world size after this weighted sum.
        scale = self.world * local_tokens / total
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
        return scale, total

    def average_gradients(self, model):
        if not self.distributed:
            return
        parameters = list(model.parameters())
        present = torch.tensor([p.grad is not None for p in parameters],
                               dtype=torch.int32, device=self.device)
        dist.all_reduce(present)
        if any(count not in (0, self.world) for count in present.tolist()):
            raise RuntimeError('Ranks disagree on gradient presence')
        bucket, size = [], 0

        def flush():
            nonlocal size
            if not bucket:
                return
            flat = torch.cat([grad.reshape(-1) for grad in bucket])
            dist.all_reduce(flat)
            flat.div_(self.world)
            offset = 0
            for grad in bucket:
                grad.copy_(flat[offset:offset + grad.numel()].view_as(grad))
                offset += grad.numel()
            bucket.clear()
            size = 0

        for parameter in parameters:
            grad = parameter.grad
            if grad is None:
                continue
            if bucket and (size + grad.numel() * grad.element_size() > BUCKET_BYTES
                           or bucket[0].dtype != grad.dtype):
                flush()
            bucket.append(grad)
            size += grad.numel() * grad.element_size()
        flush()

    def shutdown(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()
