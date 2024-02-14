# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, List

import torch

import llm_transparency_tool.routes.contributions as contributions


class TestContributions(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

        self.eps = 1e-4

        # It may be useful to run the test on GPU in case there are any issues with
        # creating temporary tensors on another device. But turn this off by default.
        self.test_on_gpu = False

        self.device = "cuda" if self.test_on_gpu else "cpu"

        self.batch = 4
        self.tokens = 5
        self.heads = 6
        self.d_model = 10

        self.decomposed_attn = torch.rand(
            self.batch,
            self.tokens,
            self.tokens,
            self.heads,
            self.d_model,
            device=self.device,
        )
        self.mlp_out = torch.rand(
            self.batch, self.tokens, self.d_model, device=self.device
        )
        self.resid_pre = torch.rand(
            self.batch, self.tokens, self.d_model, device=self.device
        )
        self.resid_mid = torch.rand(
            self.batch, self.tokens, self.d_model, device=self.device
        )
        self.resid_post = torch.rand(
            self.batch, self.tokens, self.d_model, device=self.device
        )

    def _assert_tensor_eq(self, t: torch.Tensor, expected: List[Any]):
        self.assertTrue(
            torch.isclose(t, torch.Tensor(expected), atol=self.eps).all(),
            t,
        )

    def test_mlp_contributions(self):
        mlp_out = torch.tensor([[[1.0, 1.0]]])
        resid_mid = torch.tensor([[[0.0, 0.0]]])
        resid_post = torch.tensor([[[1.0, 1.0]]])

        c_mlp, c_residual = contributions.get_mlp_contributions(
            resid_mid, resid_post, mlp_out
        )
        self.assertAlmostEqual(c_mlp.item(), 1.0, delta=self.eps)
        self.assertAlmostEqual(c_residual.item(), 0.0, delta=self.eps)

    def test_decomposed_attn_contributions(self):
        resid_pre = torch.tensor([[[2.0, 1.0]]])
        resid_mid = torch.tensor([[[2.0, 2.0]]])
        decomposed_attn = torch.tensor(
            [
                [
                    [
                        [
                            [1.0, 1.0],
                            [-1.0, 0.0],
                        ]
                    ]
                ]
            ]
        )

        c_attn, c_residual = contributions.get_attention_contributions(
            resid_pre, resid_mid, decomposed_attn, distance_norm=2
        )
        self._assert_tensor_eq(c_attn, [[[[0.43613, 0]]]])
        self.assertAlmostEqual(c_residual.item(), 0.56387, delta=self.eps)

    def test_decomposed_mlp_contributions(self):
        pre = torch.tensor([10.0, 10.0])
        post = torch.tensor([-10.0, 10.0])
        neuron_impacts = torch.tensor(
            [
                [0.0, 1.0],
                [1.0, 0.0],
                [-21.0, -1.0],
            ]
        )
        c_mlp, c_residual = contributions.get_decomposed_mlp_contributions(
            pre, post, neuron_impacts, distance_norm=2
        )
        # A bit counter-intuitive, but the only vector pointing from 0 towards the
        # output is the first one.
        self._assert_tensor_eq(c_mlp, [1, 0, 0])
        self.assertAlmostEqual(c_residual, 0, delta=self.eps)

    def test_decomposed_mlp_contributions_single_direction(self):
        pre = torch.tensor([1.0, 1.0])
        post = torch.tensor([4.0, 4.0])
        neuron_impacts = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 2.0],
            ]
        )
        c_mlp, c_residual = contributions.get_decomposed_mlp_contributions(
            pre, post, neuron_impacts, distance_norm=2
        )
        self._assert_tensor_eq(c_mlp, [0.25, 0.5])
        self.assertAlmostEqual(c_residual, 0.25, delta=self.eps)

    def test_attention_contributions_shape(self):
        c_attn, c_residual = contributions.get_attention_contributions(
            self.resid_pre, self.resid_mid, self.decomposed_attn
        )
        self.assertEqual(
            list(c_attn.shape), [self.batch, self.tokens, self.tokens, self.heads]
        )
        self.assertEqual(list(c_residual.shape), [self.batch, self.tokens])

    def test_mlp_contributions_shape(self):
        c_mlp, c_residual = contributions.get_mlp_contributions(
            self.resid_mid, self.resid_post, self.mlp_out
        )
        self.assertEqual(list(c_mlp.shape), [self.batch, self.tokens])
        self.assertEqual(list(c_residual.shape), [self.batch, self.tokens])

    def test_renormalizing_threshold(self):
        c_blocks = torch.Tensor([[0.05, 0.15], [0.05, 0.05]])
        c_residual = torch.Tensor([0.8, 0.9])
        norm_blocks, norm_residual = contributions.apply_threshold_and_renormalize(
            0.1, c_blocks, c_residual
        )
        self._assert_tensor_eq(norm_blocks, [[0.0, 0.157894], [0.0, 0.0]])
        self._assert_tensor_eq(norm_residual, [0.842105, 1.0])
