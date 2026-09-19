from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import TokenBlockDataset  # noqa: E402
from evaluation.oracle import evaluate_layer_oracle  # noqa: E402
from evaluation.temperature_scaling import fit_temperature, scale_probabilities  # noqa: E402
from moe.model import TinyMoELanguageModel  # noqa: E402
from moe.adaptive_routing import RoutingPolicy, forward_with_routing_policy  # noqa: E402
from moe.compute_gate import (  # noqa: E402
    GateFeatureSpec,
    LinearComputeGate,
    forward_with_compute_gate,
    gate_parameter_count,
)
from moe.router_baseline import TopKRouter  # noqa: E402


class TopKRouterTest(unittest.TestCase):
    def test_probabilities_and_assignments_are_valid(self) -> None:
        router = TopKRouter(model_dim=8, num_experts=4, top_k=2)
        output = router(torch.randn(11, 8))
        torch.testing.assert_close(
            output.probabilities.sum(dim=-1), torch.ones(11)
        )
        torch.testing.assert_close(
            output.expert_weights.sum(dim=-1), torch.ones(11)
        )
        self.assertEqual(int(output.utilization.sum()), 22)
        self.assertTrue(torch.isfinite(output.auxiliary_loss))


class ModelTest(unittest.TestCase):
    @staticmethod
    def _tiny_model() -> TinyMoELanguageModel:
        return TinyMoELanguageModel(
            vocab_size=256,
            sequence_length=16,
            num_layers=2,
            model_dim=32,
            num_heads=4,
            dense_hidden_dim=64,
            expert_hidden_dim=64,
            num_experts=4,
            top_k=2,
            moe_layers=[1],
            dropout=0.0,
            load_balance_weight=0.01,
            router_z_loss_weight=0.001,
        )

    def test_forward_backward_and_utilization(self) -> None:
        model = self._tiny_model()
        inputs = torch.randint(0, 256, (2, 16))
        output = model(inputs, inputs)
        self.assertEqual(tuple(output.logits.shape), (2, 16, 256))
        self.assertEqual(tuple(output.expert_utilization.shape), (1, 4))
        self.assertEqual(int(output.expert_utilization.sum()), 64)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        router_gradient = model.blocks[1].feed_forward.router.projection.weight.grad
        self.assertIsNotNone(router_gradient)
        self.assertGreater(float(router_gradient.abs().sum()), 0.0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_fp16_sparse_combine(self) -> None:
        model = self._tiny_model().cuda()
        inputs = torch.randint(0, 256, (2, 16), device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(inputs, inputs)
        self.assertEqual(output.logits.device.type, "cuda")
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()


class TokenDatasetTest(unittest.TestCase):
    def test_next_token_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.npy"
            np.save(path, np.arange(21, dtype=np.uint8))
            dataset = TokenBlockDataset(path, sequence_length=5)
            inputs, targets = dataset[0]
            self.assertEqual(inputs.tolist(), [0, 1, 2, 3, 4])
            self.assertEqual(targets.tolist(), [1, 2, 3, 4, 5])


class OracleInterventionTest(unittest.TestCase):
    def test_reconstructed_baseline_matches_normal_forward(self) -> None:
        model = TinyMoELanguageModel(
            vocab_size=256,
            sequence_length=8,
            num_layers=2,
            model_dim=32,
            num_heads=4,
            dense_hidden_dim=64,
            expert_hidden_dim=64,
            num_experts=4,
            top_k=2,
            moe_layers=[0, 1],
            dropout=0.0,
            load_balance_weight=0.01,
            router_z_loss_weight=0.001,
        ).eval()
        inputs = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))
        with torch.inference_mode():
            normal = model(inputs, targets)
            normal_losses = F.cross_entropy(
                normal.logits.reshape(-1, 256), targets.reshape(-1), reduction="none"
            )
            for layer in [0, 1]:
                oracle = evaluate_layer_oracle(
                    model, inputs, targets, layer, intervention_chunk_size=3
                )
                torch.testing.assert_close(
                    oracle.baseline_top2_losses,
                    normal_losses,
                    rtol=1e-5,
                    atol=1e-5,
                )
                self.assertEqual(tuple(oracle.expert_losses.shape), (16, 4))
                self.assertTrue(torch.isfinite(oracle.expert_losses).all())


class PolicyForwardTest(unittest.TestCase):
    @staticmethod
    def _model() -> TinyMoELanguageModel:
        return TinyMoELanguageModel(
            vocab_size=256,
            sequence_length=8,
            num_layers=2,
            model_dim=32,
            num_heads=4,
            dense_hidden_dim=64,
            expert_hidden_dim=64,
            num_experts=4,
            top_k=2,
            moe_layers=[0, 1],
            dropout=0.0,
            load_balance_weight=0.01,
            router_z_loss_weight=0.001,
        ).eval()

    def test_fixed_top2_matches_normal_model(self) -> None:
        model = self._model()
        inputs = torch.randint(0, 256, (2, 8))
        targets = torch.randint(0, 256, (2, 8))
        with torch.inference_mode():
            normal = model(inputs, targets)
            controlled = forward_with_routing_policy(
                model, inputs, targets, RoutingPolicy("top2", fixed_k=2)
            )
        torch.testing.assert_close(controlled.logits, normal.logits)
        self.assertEqual(controlled.expert_assignments, 64)
        self.assertEqual(controlled.routing_decisions, 32)

    def test_top1_ranking_is_temperature_invariant(self) -> None:
        model = self._model()
        inputs = torch.randint(0, 256, (2, 8))
        policy_a = RoutingPolicy("top1-a", fixed_k=1)
        policy_b = RoutingPolicy("top1-b", fixed_k=1, temperatures={0: 3.0, 1: 0.4})
        with torch.inference_mode():
            output_a = forward_with_routing_policy(model, inputs, None, policy_a)
            output_b = forward_with_routing_policy(model, inputs, None, policy_b)
        torch.testing.assert_close(output_a.logits, output_b.logits)
        self.assertEqual(output_a.expert_assignments, 32)


class TemperatureScalingTest(unittest.TestCase):
    def test_temperature_fit_reduces_hard_label_nll(self) -> None:
        probabilities = np.array(
            [
                [0.95, 0.03, 0.01, 0.01],
                [0.90, 0.05, 0.03, 0.02],
                [0.80, 0.10, 0.05, 0.05],
                [0.70, 0.10, 0.10, 0.10],
            ],
            dtype=np.float64,
        )
        targets = np.array([1, 0, 2, 0], dtype=np.int64)
        fitted = fit_temperature(probabilities, hard_targets=targets)
        self.assertGreater(fitted["temperature"], 1.0)
        self.assertLess(fitted["objective_after"], fitted["objective_before"])
        scaled = scale_probabilities(probabilities, fitted["temperature"])
        np.testing.assert_allclose(scaled.sum(axis=1), 1.0)


class ComputeGateTest(unittest.TestCase):
    def test_extreme_gate_decisions_match_fixed_k_controls(self) -> None:
        model = PolicyForwardTest._model()
        inputs = torch.randint(0, 256, (2, 8))
        spec = GateFeatureSpec(True, 32, (0, 1))
        gate = LinearComputeGate(spec.input_dim).eval()
        torch.nn.init.zeros_(gate.output.weight)
        mean = torch.zeros(spec.input_dim)
        std = torch.ones(spec.input_dim)
        with torch.inference_mode():
            gate.output.bias.fill_(1.0)
            learned_top2 = forward_with_compute_gate(
                model, inputs, None, gate, spec, mean, std, threshold=0.0
            )
            fixed_top2 = forward_with_routing_policy(
                model, inputs, None, RoutingPolicy("top2", fixed_k=2)
            )
            gate.output.bias.fill_(-1.0)
            learned_top1 = forward_with_compute_gate(
                model, inputs, None, gate, spec, mean, std, threshold=0.0
            )
            fixed_top1 = forward_with_routing_policy(
                model, inputs, None, RoutingPolicy("top1", fixed_k=1)
            )
        torch.testing.assert_close(learned_top2.logits, fixed_top2.logits)
        torch.testing.assert_close(learned_top1.logits, fixed_top1.logits)
        self.assertEqual(gate_parameter_count(gate), spec.input_dim + 1)


if __name__ == "__main__":
    unittest.main()
