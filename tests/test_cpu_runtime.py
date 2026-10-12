"""Small real CPU tests; no AF3 data, model weights, native build or GPU needed."""

import importlib
import pkgutil
import subprocess
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from xfold import fastnn
from xfold.fastnn import config
from xfold.fastnn.gated_linear_unit import GLUWeightCache
from xfold.nn.cache import InputCache, input_cache_scope
from xfold.nn.diffusion_transformer import DiffusionTransition
from xfold.nn.pairformer import PairformerBlock
from xfold.nn.primitives import Transition
from xfold.nn.triangle_multiplication import TriangleMultiplication


class CPURuntimeTests(unittest.TestCase):
    def setUp(self):
        self.settings = {name: value for name, value in vars(config).items()
                         if not name.startswith('_')}
        config.layer_norm_implementation = 'torch'
        config.gated_linear_unit_implementation = 'torch'
        config.dot_product_attention_implementation = 'torch'
        config.triangle_multiplication_implementation = 'torch'
        config.input_cache_enabled = False
        config.packed_weight_cache_enabled = False
        torch.manual_seed(42)

    def tearDown(self):
        for name, value in self.settings.items():
            setattr(config, name, value)

    def test_core_modules_import_without_optional_accelerators(self):
        import xfold
        names = []
        for info in pkgutil.walk_packages(xfold.__path__, 'xfold.'):
            if '._triton_' not in info.name:
                importlib.import_module(info.name)
                names.append(info.name)
        self.assertIn('xfold.alphafold3', names)
        self.assertIn('xfold.params', names)
        self.assertNotIn('triton', sys.modules)
        self.assertNotIn('intel_extension_for_pytorch', sys.modules)
        self.assertNotIn('oneccl_bindings_for_pytorch', sys.modules)

    def test_layer_norm_matches_torch_with_affine_and_noncontiguous_inputs(self):
        for affine, bias in ((True, True), (True, False), (False, False)):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(affine=affine, bias=bias, dtype=dtype):
                    layer = fastnn.LayerNorm(7, elementwise_affine=affine, bias=bias).eval()
                    x = torch.randn(2, 7, 3).transpose(1, 2).to(dtype)
                    with torch.inference_mode():
                        expected = F.layer_norm(x, (7,), layer.weight, layer.bias, layer.eps)
                        torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)

    def test_glu_fp32_bf16_and_odd_feature_dimensions(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype), torch.inference_mode():
                x = torch.randn(2, 3, 7).to(dtype)
                weight = torch.randn(10, 7).T.to(dtype)
                a, b = (x @ weight).chunk(2, dim=-1)
                torch.testing.assert_close(fastnn.gated_linear_unit(x, weight), F.silu(a) * b)

    def test_attention_matches_independent_einsum_reference(self):
        for dtype in (torch.float32, torch.bfloat16):
            for batched in (True, False):
                for mask_rank in (1, 2, 4):
                    with self.subTest(dtype=dtype, batched=batched, mask_rank=mask_rank), torch.inference_mode():
                        q, k, v = [torch.randn(*shape).to(dtype) for shape in
                                   ((1, 2, 3, 7), (1, 2, 5, 7), (1, 2, 5, 4))]
                        mask = torch.tensor([1, 1, 0, 1, 0], dtype=torch.bool)
                        expanded = mask[None, None, None, :]
                        passed = mask if mask_rank == 1 else mask[None] if mask_rank == 2 else expanded
                        bias = torch.randn(1, 2, 3, 5).to(dtype)
                        logits = torch.einsum('bhqd,bhkd->bhqk', q * 7 ** -0.5, k) + bias
                        logits.masked_fill_(~expanded, -1e9)
                        expected = torch.einsum('bhqk,bhkd->bhqd', logits.softmax(-1), v)
                        if not batched:
                            q, k, v, bias = q[0], k[0], v[0], bias[0]
                            expected = expected[0]
                        result = fastnn.dot_product_attention(q, k, v, passed, bias)
                        torch.testing.assert_close(result, expected, rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
                                                   atol=0.02 if dtype == torch.bfloat16 else 1e-6)

    def test_transition_and_conditioned_diffusion_transition_execute_in_both_precisions(self):
        for bf16 in (False, True):
            with self.subTest(bf16=bf16), torch.inference_mode(), torch.amp.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                x = torch.randn(2, 3, 8)
                for layer in (Transition(8), DiffusionTransition(8, 4),
                              DiffusionTransition(8, 4, use_single_cond=True)):
                    layer.eval()
                    result = layer(x, torch.randn(2, 3, 4)) if getattr(layer, 'use_single_cond', False) else layer(x)
                    self.assertEqual(result.shape, x.shape)
                    self.assertTrue(torch.isfinite(result).all())
                    self.assertEqual(result.dtype, torch.bfloat16 if bf16 else torch.float32)
                    self.assertTrue(all(p.dtype == torch.float32 for p in layer.parameters()))

    def test_triangle_directions_match_explicit_channel_contractions(self):
        for outgoing in (False, True):
            with self.subTest(outgoing=outgoing), torch.inference_mode():
                layer = TriangleMultiplication(8, _outgoing=outgoing).eval()
                pair, mask = torch.randn(3, 3, 8), torch.eye(3)
                normalized = layer.left_norm_input(pair)
                projected = layer.projection(normalized) * mask[..., None] * layer.gate(normalized).sigmoid()
                a, b = projected[..., 0::2], projected[..., 1::2]
                product = torch.stack([a[..., c] @ b[..., c].T if outgoing else
                                       b[..., c].T @ a[..., c] for c in range(8)], dim=-1)
                expected = layer.output_projection(layer.center_norm(product)) * layer.gating_linear(normalized).sigmoid()
                torch.testing.assert_close(layer(pair, mask), expected)

    def test_pairformer_block_runs_end_to_end_with_small_tensors(self):
        for bf16 in (False, True):
            with self.subTest(bf16=bf16), torch.inference_mode(), torch.amp.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
                layer = PairformerBlock(n_heads=2, c_pair=8, c_single=8, n_heads_pair=2).eval()
                pair, single = layer(torch.randn(3, 3, 8), torch.ones(3, 3),
                                     torch.randn(3, 8), torch.ones(3))
                self.assertEqual(pair.shape, (3, 3, 8))
                self.assertEqual(single.shape, (3, 8))
                self.assertTrue(torch.isfinite(pair).all() and torch.isfinite(single).all())

    def test_input_cache_reuses_and_invalidates_features_and_weights(self):
        config.input_cache_enabled = True
        owner, cache, feature = nn.Linear(3, 3).eval(), InputCache(), torch.randn(2, 3)
        calls = []
        def factory():
            calls.append(1)
            return owner(feature)
        with torch.no_grad(), input_cache_scope():
            first = cache.get('x', (feature,), owner, factory)
            self.assertIs(first, cache.get('x', (feature,), owner, factory))
            feature.add_(1)
            second = cache.get('x', (feature,), owner, factory)
            self.assertIsNot(first, second)
            owner.weight.add_(1)
            self.assertIsNot(second, cache.get('x', (feature,), owner, factory))
            self.assertEqual(len(calls), 3)
        self.assertFalse(cache._entries)

    def test_cache_copy_scope_budget_training_and_exception_cleanup(self):
        config.input_cache_enabled = True
        owner, cache, feature = nn.Linear(3, 3).eval(), InputCache(), torch.randn(2, 3)
        calls = []
        def factory():
            calls.append(1)
            return feature * 2
        with torch.no_grad(), input_cache_scope():
            result = cache.get('x', (feature,), owner, factory, copy=torch.clone)
            result.zero_()
            torch.testing.assert_close(cache.get('x', (feature,), owner, factory, copy=torch.clone), feature * 2)
            self.assertEqual(len(calls), 1)
        config.input_cache_max_bytes = 1
        with torch.no_grad(), input_cache_scope():
            cache.get('x', (feature,), owner, factory)
            cache.get('x', (feature,), owner, factory)
            self.assertFalse(cache._entries)
        config.input_cache_max_bytes = 4096
        with self.assertRaisesRegex(RuntimeError, 'test exception'), torch.no_grad(), input_cache_scope():
            cache.get('x', (feature,), owner, factory)
            raise RuntimeError('test exception')
        self.assertFalse(cache._entries)
        owner.train()
        with torch.no_grad(), input_cache_scope():
            cache.get('x', (feature,), owner, factory)
            self.assertFalse(cache._entries)

    def test_inference_features_cache_but_versionless_weights_do_not(self):
        config.input_cache_enabled = True
        owner = nn.Linear(3, 3).eval()
        cache, calls = InputCache(), []
        with torch.inference_mode(), input_cache_scope():
            feature = torch.ones(2, 3)
            def factory():
                calls.append(1)
                return feature * 2
            self.assertIs(cache.get('x', (feature,), owner, factory), cache.get('x', (feature,), owner, factory))
            self.assertEqual(len(calls), 1)
            versionless_owner = nn.Linear(3, 3).eval()
            cache.get('y', (feature,), versionless_owner, factory)
            cache.get('y', (feature,), versionless_owner, factory)
            self.assertEqual(len(calls), 3)

    def test_weight_cache_reuses_mutation_checked_transposes_and_dtypes(self):
        weight = nn.Parameter(torch.randn(7, 5))
        cache = GLUWeightCache()
        with torch.no_grad():
            first = cache.prepare(weight.T, torch.float32)
            self.assertIs(first, cache.prepare(weight.T, torch.float32))
            weight.add_(1)
            updated = cache.prepare(weight.T, torch.float32)
            self.assertIsNot(first, updated)
            torch.testing.assert_close(updated, weight.T)
            bf16 = cache.prepare(weight.T, torch.bfloat16)
            self.assertEqual(bf16.dtype, torch.bfloat16)
        with torch.inference_mode():
            unversioned = torch.randn(5, 7)
            cache.prepare(unversioned, torch.float32)
            self.assertIsNone(cache._key)

    def test_optional_backend_guards_fail_clearly_without_loading_accelerators(self):
        from xfold.fastnn.native import check_cpu_inference, compute_dtype
        with self.assertRaisesRegex(RuntimeError, 'inference only'):
            check_cpu_inference(torch.randn(2, requires_grad=True))
        with self.assertRaisesRegex(ValueError, 'dtypes must match'):
            compute_dtype(torch.ones(2), torch.ones(2, dtype=torch.bfloat16))
        config.gated_linear_unit_implementation = 'triton'
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            fastnn.gated_linear_unit(torch.randn(2, 3), torch.randn(3, 4))
        config.gated_linear_unit_implementation = 'cpp'
        # Reject before native loading even on a cluster with af3_kernels installed.
        with torch.enable_grad(), self.assertRaisesRegex(RuntimeError, 'inference only'):
            fastnn.gated_linear_unit(torch.randn(2, 3, requires_grad=True), torch.randn(3, 4))
        config.layer_norm_implementation = 'ipex'
        with self.assertRaisesRegex(RuntimeError, 'eval inference'):
            fastnn.LayerNorm(3)(torch.randn(2, 3))

    def test_thread_configuration_in_a_fresh_subprocess(self):
        code = """import torch
from xfold.cpu_runtime import configure_cpu_threads
first = 1 if torch.get_num_interop_threads() != 1 else 2
configure_cpu_threads(2, first)
configure_cpu_threads(2, first)
assert torch.get_num_threads() == 2
assert torch.get_num_interop_threads() == first
try:
    configure_cpu_threads(2, first + 1)
except RuntimeError as error:
    assert 'fresh process' in str(error)
else:
    raise AssertionError('Used inter-op pool was resized')
"""
        checked = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
        self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == '__main__':
    unittest.main()
