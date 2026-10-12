"""Stage 4 source contracts; no PyTorch import, native build, or kernel execution."""

import ast
import unittest

from test_cpu_static import ROOT, calls, keywords, parse


class GLUStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.glu = parse('xfold/fastnn/gated_linear_unit.py')
        cls.runner = parse('run_alphafold.py')
        cls.native = (ROOT / 'af3_kernels/src/csrc/gated_linear_unit.cpp').read_text()

    def function(self, name):
        return next(node for node in self.glu.body if getattr(node, 'name', None) == name)

    def runner_init(self):
        model_runner = next(node for node in self.runner.body if getattr(node, 'name', None) == 'ModelRunner')
        return next(node for node in model_runner.body if getattr(node, 'name', None) == '__init__')

    def test_backend_is_explicit_and_defaults_preserve_baselines(self):
        flag = next(call for call in calls(self.runner, 'flags.DEFINE_enum')
                    if ast.literal_eval(call.args[0]) == 'glu_backend')
        self.assertEqual(ast.literal_eval(flag.args[1]), 'auto')
        self.assertEqual(ast.literal_eval(flag.args[2]), ['auto', 'torch', 'cpp'])
        init = self.runner_init()
        defaults = dict(zip([arg.arg for arg in init.args.args[-len(init.args.defaults):]], init.args.defaults))
        self.assertEqual(ast.literal_eval(defaults['glu_backend']), 'auto')
        constructor = calls(self.runner, 'ModelRunner')[0]
        self.assertEqual(keywords(constructor)['glu_backend'], '_GLU_BACKEND.value')
        config = parse('xfold/fastnn/config.py')
        default = next(node for node in config.body if isinstance(node, ast.Assign)
                       and ast.unparse(node.targets[0]) == 'gated_linear_unit_implementation')
        self.assertEqual(ast.literal_eval(default.value), 'torch')

    def test_cpp_preflight_is_cpu_only_and_precedes_weights(self):
        init = self.runner_init()
        branch = next(node for node in init.body if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "glu_backend == 'cpp'")
        cpu_guard = next(node for node in branch.body if isinstance(node, ast.If))
        self.assertEqual(ast.unparse(cpu_guard.test), "self._device.type != 'cpu'")
        self.assertTrue(any(isinstance(node, ast.Raise) for node in cpu_guard.body))
        load = calls(branch, 'load_cpp_glu')[0]
        self.assertLess(cpu_guard.lineno, load.lineno)
        self.assertLess(load.lineno, calls(init, 'AlphaFold3')[0].lineno)
        self.assertLess(load.lineno, calls(init, 'import_jax_weights_')[0].lineno)

    def test_original_pytorch_glu_is_unchanged(self):
        expected = ast.parse('''
def gated_linear_unit_torch(x, weight):
    y = torch.matmul(x, weight)
    a, b = torch.chunk(y, 2, dim=-1)
    out = torch.nn.functional.silu(a) * b
    return out
''').body[0]
        self.assertEqual(ast.dump(self.function('gated_linear_unit_torch')), ast.dump(expected))

    def test_cpp_selection_does_not_silently_fallback(self):
        dispatch = self.function('gated_linear_unit')
        branch = next(node for node in ast.walk(dispatch) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "fastnn_config.gated_linear_unit_implementation == 'cpp'")
        self.assertEqual(len(calls(branch, 'gated_linear_unit_cpp')), 1)
        bridge = self.function('gated_linear_unit_cpp')
        self.assertFalse(calls(bridge, 'gated_linear_unit_torch'))
        self.assertFalse(any(isinstance(node, ast.Try) for node in ast.walk(bridge)))
        loader = self.function('load_cpp_glu')
        self.assertIn("hasattr(extension, 'gated_linear_unit')", ast.unparse(loader))

    def test_cpu_autocast_is_queried_before_disabled_native_region(self):
        bridge = self.function('gated_linear_unit_cpp')
        dtype = next(node for node in bridge.body if isinstance(node, ast.Assign)
                     and ast.unparse(node.targets[0]) == 'dtype')
        self.assertEqual(ast.unparse(dtype.value),
                         "torch.get_autocast_dtype('cpu') if torch.is_autocast_enabled('cpu') else x.dtype")
        region = next(node for node in bridge.body if isinstance(node, ast.With))
        self.assertLess(dtype.lineno, region.lineno)
        context = region.items[0].context_expr
        self.assertEqual(keywords(context), {'device_type': "'cpu'", 'enabled': 'False'})
        self.assertEqual(len(calls(region, 'extension.gated_linear_unit')), 2)
        self.assertIn('x.detach().resolve_conj().resolve_neg().to(dtype=dtype).contiguous()', ast.unparse(region))
        self.assertIn('weight_cache.prepare(weight, dtype)', ast.unparse(region))
        self.assertFalse(calls(bridge, 'copy_'))

    def test_training_cuda_and_dtype_mismatch_rejected_before_native_call(self):
        bridge = self.function('gated_linear_unit_cpp')
        guards = {ast.unparse(node.test): node for node in bridge.body if isinstance(node, ast.If)}
        expected = (
            "x.device.type != 'cpu' or weight.device.type != 'cpu'",
            'torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)',
            "not torch.is_autocast_enabled('cpu') and weight.dtype != x.dtype",
            'dtype not in (torch.float32, torch.bfloat16)',
        )
        for condition in expected:
            with self.subTest(condition=condition):
                self.assertTrue(any(isinstance(node, ast.Raise) for node in guards[condition].body))
                self.assertLess(guards[condition].lineno, calls(bridge, 'load_cpp_glu')[0].lineno)

    def test_cache_invalidation_covers_mutation_storage_shape_and_precision(self):
        cache = self.function('GLUWeightCache')
        prepare = next(node for node in cache.body if getattr(node, 'name', None) == 'prepare')
        key = next(node for node in prepare.body if isinstance(node, ast.Assign)
                   and ast.unparse(node.targets[0]) == 'key')
        self.assertEqual([ast.unparse(item) for item in key.value.elts], [
            'weight.data_ptr()', 'version', 'tuple(weight.shape)', 'tuple(weight.stride())',
            'weight.dtype', 'weight.device', 'dtype',
        ])
        self.assertIn('version = weight._version', ast.unparse(prepare))
        self.assertIn('self._source = weight', ast.unparse(prepare))
        self.assertIn('if self._key != key:', ast.unparse(prepare))
        self.assertFalse(calls(prepare, 'register_buffer'))
        self.assertFalse(calls(prepare, 'register_parameter'))

    def test_unversioned_inference_weights_are_not_cached(self):
        cache = self.function('GLUWeightCache')
        prepare = next(node for node in cache.body if getattr(node, 'name', None) == 'prepare')
        handler = next(node for node in prepare.body if isinstance(node, ast.Try)).handlers[0]
        self.assertEqual(ast.unparse(handler.type), 'RuntimeError')
        self.assertIn('self._key = self._source = self._weight = self._packed = None', ast.unparse(handler))
        self.assertEqual(ast.unparse(handler.body[-1]),
                         'return weight.detach().to(dtype=dtype).contiguous()')

    def test_both_transitions_share_bridge_and_keep_other_operators(self):
        cases = (
            ('xfold/nn/primitives.py', 'Transition', 'self.input_layer_norm', 'self.transition2'),
            ('xfold/nn/diffusion_transformer.py', 'DiffusionTransition',
             'self.adaptive_layernorm', 'self.adaptive_zero_init'),
        )
        for path, name, norm, final in cases:
            with self.subTest(name=name):
                tree = parse(path)
                transition = next(node for node in tree.body if getattr(node, 'name', None) == name)
                init = next(node for node in transition.body if getattr(node, 'name', None) == '__init__')
                forward = next(node for node in transition.body if getattr(node, 'name', None) == 'forward')
                self.assertEqual(len(calls(init, 'GLUWeightCache')), 1)
                glu = calls(forward, 'fastnn.gated_linear_unit')[0]
                self.assertEqual([ast.unparse(arg) for arg in glu.args], ['x', 'self.transition1.weight.T'])
                self.assertEqual(keywords(glu), {'weight_cache': 'self._glu_weight_cache'})
                self.assertLess(calls(forward, norm)[0].lineno, glu.lineno)
                self.assertGreater(calls(forward, final)[0].lineno, glu.lineno)
                self.assertFalse(calls(forward, 'af3_kernels.gated_linear_unit'))

    def test_native_guards_shape_restoration_and_bf16_packing(self):
        for fragment in (
            'af3_detail::check_tensor(x)', 'af3_detail::check_tensor(weight)',
            'x.dim() >= 1 && weight.dim() == 2',
            'x.is_contiguous() && weight.is_contiguous()',
            'x.scalar_type() == weight.scalar_type()',
            'weight.size(0) == static_cast<int64_t>(k) + (weight_packed ? k % 2 : 0)',
            'weight_stride % 2 == 0', 'xsmm_size(m_block)', 'xsmm_size(n_block)',
            'auto shape = x.sizes().vec()', 'shape.back() = n', 'if (m == 0) return output',
            'static_cast<int64_t>(k) + k % 2', 'vnni_repack_tensor(right, 2, 0)',
            'LIBXSMM_GEMM_FLAG_VNNI_A', 'LIBXSMM_GEMM_FLAG_BETA_0',
            'dtype, dtype, dtype, LIBXSMM_DATATYPE_F32',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.native)

    def test_tail_descriptors_and_thread_scratch_are_set_up_before_parallel_work(self):
        for fragment in (
            '(m - 1) % mb + 1', '(n - 1) % nb + 1',
            'widths[w], heights[h], k, weight_stride, k, nb',
            'weight + col * pack', 'weight + (n + col) * pack',
            'omp_get_thread_num()', 'r < heights[hi]', 'c < widths[wi]',
            'out[(row + r) * n + col + c]', 'const T activated = T(a * sigmoid)',
        ):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.native)
        parallel = self.native.index('#pragma omp parallel for')
        self.assertLess(self.native.index('TORCH_CHECK(kernels[h][w] != nullptr'), parallel)
        self.assertLess(self.native.index('auto scratch ='), parallel)
        self.assertNotIn('TORCH_CHECK', self.native[parallel:self.native.index('}  // namespace')])


if __name__ == '__main__':
    unittest.main()
