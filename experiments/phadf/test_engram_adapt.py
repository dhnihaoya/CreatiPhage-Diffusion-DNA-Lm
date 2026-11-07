"""CPU regressions for the diffusion Engram adapter."""
import copy
import unittest

import torch

from experiments.phageseek.modules import EngramMemory
from experiments.phadf.engram_adapt import (
    MASK, install_diffusion_engram, masked_engram_forward, radix5_ngram, transplant_conv_weight,
)


class EngramProbeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        torch.set_num_threads(1)
        self.original = EngramMemory(
            vocab_size=6, pad_id=0, slots=96, memory_dim=8,
            model_dim=8, ks=(3, 6, 12), heads_per_order=2,
            layer_idx=1, zero_init_residual=False, direct_threshold=65536)
        self.memory = copy.deepcopy(self.original)
        install_diffusion_engram(self.memory)

    def adapted(self, hidden, ids):
        return masked_engram_forward(self.memory, hidden, ids)

    def test_atcg_rows_are_copied_and_mask_patterns_differ(self):
        table = self.memory.memory_tables[0]
        torch.testing.assert_close(table.atcg, self.original.memory_tables[0].weight[:-1], rtol=0, atol=0)
        self.assertEqual(tuple(table.mask_rows.shape), (5 ** 3 - 4 ** 3, 8))
        ids = torch.tensor([[6, 6, 6, 1, 6, 6]])
        codes = radix5_ngram(self.memory, ids, 3)
        self.assertEqual(int(codes[0, 2]), 4 + 4 * 5 + 4 * 25)
        self.assertNotEqual(int(codes[0, 2]), int(codes[0, 5]))
        self.assertEqual(int(table.group[int(codes[0, 2])]), 1)
        self.assertEqual(int(table.group[0]), 0)

    def test_full_mask_memory_is_nonzero_and_trains_the_mask_row(self):
        with torch.no_grad():
            self.memory.short_conv.conv.weight.zero_()
            self.memory.memory_tables[0].mask_rows.fill_(1)
            self.memory.value_proj.weight.zero_()
            self.memory.value_proj.weight[0, 0] = 1
            self.memory.value_proj.bias.zero_()
            self.memory.key_proj.weight.zero_()
            self.memory.key_proj.bias.zero_()
        ids = torch.full((1, 20), MASK, dtype=torch.long)
        result = self.adapted(torch.randn(1, 20, 8), ids)
        self.assertGreater(float(result.detach().abs().sum()), 0)
        result.sum().backward()
        self.assertGreater(float(self.memory.value_proj.bias.grad.abs().sum()), 0)
        self.assertGreater(float(self.memory.memory_tables[0].mask_rows.grad.abs().sum()), 0)
        self.assertEqual(float(self.memory.memory_tables[0].atcg.grad.abs().sum()), 0)

    def test_future_tap_changes_the_current_position(self):
        conv = self.memory.short_conv
        with torch.no_grad():
            conv.conv.weight.zero_()
            conv.norm.weight.fill_(1)
            conv.conv.weight[0, 0, 4] = 1
        x = torch.zeros(1, 80, conv.conv.weight.shape[0])
        x[0, 40 + conv.dilation, 0] = 1
        moved = conv(x)[0, 40, 0]
        self.assertNotEqual(float(moved.detach()), 0)
        with torch.no_grad():
            conv.conv.weight.zero_()
            causal = torch.arange(4, dtype=conv.conv.weight.dtype).view(1, 1, 4)
            conv.conv.weight[:1].copy_(transplant_conv_weight(causal))
        past = torch.zeros_like(x)
        past[0, 40 + conv.dilation, 0] = 1
        self.assertEqual(float(conv(past)[0, 40].detach().abs().sum()), 0)
        past[0, 40 - conv.dilation, 0] = 1
        self.assertGreater(float(conv(past)[0, 40].detach().abs().sum()), 0)

    def test_hidden_answer_changes_do_not_change_corrupted_input_output(self):
        clean1 = torch.tensor([[0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5]])
        clean2 = clean1.clone()
        clean2[:, 4:10] = torch.tensor([[2, 3, 4, 1, 2, 3]])
        mask = torch.zeros_like(clean1, dtype=torch.bool)
        mask[:, 4:10] = True
        ids1, ids2 = clean1.masked_fill(mask, MASK), clean2.masked_fill(mask, MASK)
        hidden = torch.randn(1, clean1.shape[1], 8)
        out1, out2 = self.adapted(hidden, ids1), self.adapted(hidden, ids2)
        self.assertFalse(torch.equal(clean1, clean2))
        torch.testing.assert_close(out1, out2, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
