"""Stages 5/6 source contracts; no tensor execution or native compilation."""

import ast
import unittest

from test_cpu_static import ROOT, calls, keywords, parse


def definition(tree, name):
    return next(node for node in tree.body if getattr(node, 'name', None) == name)


class TriangleAttentionStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = parse('run_alphafold.py')
        cls.native = parse('xfold/fastnn/native.py')
        cls.attention = parse('xfold/fastnn/attention.py')
        cls.triangle = definition(parse('xfold/nn/triangle_multiplication.py'), 'TriangleMultiplication')
        cls.grid = definition(parse('xfold/nn/attention.py'), 'GridSelfAttention')
        cls.self_attention = definition(parse('xfold/nn/diffusion_transformer.py'), 'SelfAttention')
        cls.matrix_cpp = (ROOT / 'af3_kernels/src/csrc/matrix.cpp').read_text()
        cls.triangle_cpp = (ROOT / 'af3_kernels/src/csrc/triangle_multiplication.cpp').read_text()
        cls.attention_cpp = (ROOT / 'af3_kernels/src/csrc/dot_product_attention.cpp').read_text()

    def test_explicit_flags_default_to_previous_selection_and_reach_runner(self):
        runner = definition(self.runner, 'ModelRunner')
        init = definition(runner, '__init__')
        defaults = dict(zip([arg.arg for arg in init.args.args[-len(init.args.defaults):]], init.args.defaults))
        main_call = calls(self.runner, 'ModelRunner')[0]
        for backend in ('triangle', 'attention'):
            flag = next(call for call in calls(self.runner, 'flags.DEFINE_enum')
                        if ast.literal_eval(call.args[0]) == f'{backend}_backend')
            self.assertEqual(ast.literal_eval(flag.args[1]), 'auto')
            self.assertEqual(ast.literal_eval(flag.args[2]), ['auto', 'torch', 'cpp'])
            self.assertEqual(ast.literal_eval(defaults[f'{backend}_backend']), 'auto')
            self.assertEqual(keywords(main_call)[f'{backend}_backend'], f'_{backend.upper()}_BACKEND.value')
        assignment = next(node for node in init.body if isinstance(node, ast.Assign)
                          and ast.unparse(node.targets[0]) == 'fastnn_config.triangle_multiplication_implementation')
        self.assertEqual(ast.unparse(assignment.value), "'torch' if triangle_backend == 'auto' else triangle_backend")
        self.assertLess(assignment.lineno, calls(init, 'AlphaFold3')[0].lineno)

    def test_cpu_only_operator_preflight_precedes_model_loading(self):
        init = definition(definition(self.runner, 'ModelRunner'), '__init__')
        loop = next(node for node in init.body if isinstance(node, ast.For))
        rows = loop.iter.elts
        self.assertEqual([ast.literal_eval(row.elts[2]) for row in rows],
                         [('triangle_multiplication',), ('linear', 'dot_product_attention')])
        cpp = next(node for node in loop.body if isinstance(node, ast.If)
                   and ast.unparse(node.test) == "backend == 'cpp'")
        cpu_guard = cpp.body[0]
        self.assertEqual(ast.unparse(cpu_guard.test), "self._device.type != 'cpu'")
        self.assertTrue(any(isinstance(node, ast.Raise) for node in cpu_guard.body))
        self.assertLess(cpu_guard.lineno, calls(cpp, 'load_native_operator')[0].lineno)
        self.assertLess(loop.end_lineno, calls(init, 'import_jax_weights_')[0].lineno)

    def test_native_helpers_preserve_precision_and_inference_contract(self):
        check = definition(self.native, 'check_cpu_inference')
        self.assertIn('torch.is_grad_enabled() and tensor.requires_grad', ast.unparse(check))
        dtype = definition(self.native, 'compute_dtype')
        self.assertIn("torch.get_autocast_dtype('cpu')", ast.unparse(dtype))
        self.assertIn('weight.dtype != dtype', ast.unparse(dtype))
        linear = definition(self.native, 'linear_cpp')
        region = next(node for node in linear.body if isinstance(node, ast.With))
        self.assertEqual(keywords(region.items[0].context_expr), {'device_type': "'cpu'", 'enabled': 'False'})
        self.assertIn('weight_cache.prepare(weight, dtype)', ast.unparse(region))
        self.assertIn('prepare_tensor(layer.bias, dtype)', ast.unparse(region))
        self.assertFalse(calls(linear, 'torch.set_default_dtype'))

    def test_triangle_switch_is_runtime_and_keeps_parameter_modules(self):
        forward = definition(self.triangle, 'forward')
        self.assertEqual(len(calls(forward, 'self._forward_cpp')), 1)
        self.assertEqual(len(calls(forward, 'self._forward_torch')), 1)
        self.assertTrue(any(isinstance(node, ast.Raise) for node in forward.body))
        init = definition(self.triangle, '__init__')
        self.assertEqual(len(calls(init, 'nn.Linear')), 4)
        self.assertEqual(len(calls(init, 'fastnn.LayerNorm')), 2)
        cpp = definition(self.triangle, '_forward_cpp')
        self.assertIn('self.left_norm_input.eps', ast.unparse(cpp))
        self.assertIn('self.center_norm.eps', ast.unparse(cpp))
        self.assertIn("self.equation == 'cik,cjk->cij'", ast.unparse(cpp))
        self.assertFalse(calls(cpp, 'self._forward_torch'))

    def test_triangle_interleaving_directions_and_residual_ownership(self):
        for fragment in (
            'projected.reshape({c, 2, n, n})', 'interleaved.select(1, 0)', 'interleaved.select(1, 1)',
            'outgoing ? a : b.transpose(1, 2).contiguous()',
            'outgoing ? b.transpose(1, 2).contiguous() : a',
            'return result.view({n, n, c})', 'pair.size(0) == pair.size(1)',
            'projected.mul_(mask->unsqueeze(0))', 'mask->expand({n, n})',
        ):
            self.assertIn(fragment, self.triangle_cpp)
        self.assertNotIn('pair.add_', self.triangle_cpp)
        self.assertNotIn('pair +=', self.triangle_cpp)
        pairformer = (ROOT / 'xfold/nn/pairformer.py').read_text()
        self.assertIn('pair += self.triangle_multiplication_outgoing', pairformer)
        self.assertIn('pair += self.triangle_multiplication_incoming', pairformer)

    def test_triangle_normalization_is_not_globally_cast_to_bf16(self):
        cpp = definition(self.triangle, '_forward_cpp')
        self.assertIn('prepare_tensor(pair)', ast.unparse(cpp))
        self.assertNotIn('prepare_tensor(pair, dtype)', ast.unparse(cpp))
        self.assertIn('at::layer_norm(pair, {c}, left_weight, left_bias, left_eps', self.triangle_cpp)
        self.assertIn('at::layer_norm(contracted, {c}, center_weight, center_bias, center_eps', self.triangle_cpp)

    def test_native_linear_fuses_bias_before_output_rounding(self):
        self.assertIn('LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32', self.matrix_cpp)
        bias = self.matrix_cpp.index('value += static_cast<float>(bias_data[col + c])')
        store = self.matrix_cpp.index('= T(value);')
        self.assertLess(bias, store)
        self.assertIn('static_cast<int64_t>(k) + k % 2', self.matrix_cpp)
        self.assertIn('batches * inner', self.matrix_cpp)
        self.assertIn('if (batches == 0 || m == 0) return output', self.matrix_cpp)

    def test_attention_scales_before_autocast_conversion(self):
        bridge = definition(self.attention, 'dot_product_attention_cpp')
        region = next(node for node in bridge.body if isinstance(node, ast.With))
        scaled = next(node for node in region.body if isinstance(node, ast.Assign)
                      and ast.unparse(node.targets[0]) == 'scaled_q')
        native_call = calls(region, 'extension.dot_product_attention')[0]
        self.assertEqual(ast.unparse(scaled.value), 'q.detach() * q.size(-1) ** (-0.5)')
        self.assertEqual(ast.unparse(native_call.args[0]), 'prepare_tensor(scaled_q, dtype)')
        self.assertEqual(keywords(native_call)['q_pre_scaled'], 'True')
        self.assertFalse(calls(bridge, 'dot_product_attention_torch'))

    def test_cpp_attention_selected_before_triton_fallback(self):
        dispatch = definition(self.attention, 'dot_product_attention')
        cpp = next(node for node in ast.walk(dispatch) if isinstance(node, ast.If)
                   and ast.unparse(node.test) == "fastnn_config.dot_product_attention_implementation == 'cpp'")
        self.assertEqual(len(calls(cpp, 'dot_product_attention_cpp')), 1)
        self.assertIn('q = q.unsqueeze(0)', ast.unparse(dispatch))
        self.assertIn('out = out.squeeze(0)', ast.unparse(dispatch))

    def test_grid_and_self_attention_use_native_projections_with_original_outputs(self):
        for cls, method, output in (
            (self.grid, '_attention_cpp', 'linear_cpp'),
            (self.self_attention, '_forward_cpp', 'self.adaptive_zero_init'),
        ):
            cpp = definition(cls, method)
            self.assertEqual(len(calls(cpp, 'fastnn.dot_product_attention')), 1)
            self.assertIn("('q_projection', 'k_projection', 'v_projection')", ast.unparse(cpp))
            self.assertTrue(calls(cpp, 'linear_cpp'))
            self.assertTrue(calls(cpp, output))
            self.assertTrue(calls(cpp, 'torch.sigmoid'))
        grid_forward = definition(self.grid, 'forward')
        self.assertEqual(len(calls(grid_forward, 'self.act_norm')), 1)
        self.assertEqual(len(calls(grid_forward, 'pair.permute')), 2)
        self.assertIn('self.adaptive_layernorm', ast.unparse(definition(self.self_attention, '_forward_cpp')))

    def test_mask_bias_broadcast_and_nan_semantics_are_explicit(self):
        for fragment in (
            'bool_mask.to(at::kBool).unsqueeze(1).unsqueeze(1)',
            'bool_mask.expand({batches, heads, queries, key_count})',
            'bias->to(at::kFloat).expand({batches, heads, queries, key_count})',
            'T(-1e9f)', 'T(value + bias_data[offset])',
            'std::isnan(value)', 'quiet_NaN()', 'T(exps[j] / sum)',
            'j = key_count; j < padded_keys',
        ):
            self.assertIn(fragment, self.attention_cpp)
        self.assertNotIn('nan_to_num', self.attention_cpp)
        self.assertNotIn('masked_fill_', self.attention_cpp)

    def test_attention_scratch_is_bounded_by_query_block_per_thread(self):
        self.assertIn('{threads, qb, padded_keys}', self.attention_cpp)
        self.assertIn('{threads, key_count}', self.attention_cpp)
        self.assertIn('q.size(3) == k.size(3) && k.size(2) == v.size(2)', self.attention_cpp)
        self.assertIn('static_cast<int64_t>(d) + d % 2', self.attention_cpp)
        self.assertIn('static_cast<int64_t>(nk) + nk % 2', self.attention_cpp)
        self.assertIn('key_count == 0', self.attention_cpp)
        self.assertIn('output.zero_()', self.attention_cpp)

    def test_parallel_kernels_validate_and_allocate_on_calling_thread(self):
        for code, guard in ((self.matrix_cpp, 'TORCH_CHECK(kernels'),
                            (self.attention_cpp, 'TORCH_CHECK(qk')):
            start = code.index('#pragma omp parallel for')
            end = code.index('}  // namespace')
            self.assertLess(code.index('auto scratch ='), start)
            self.assertLess(code.index(guard), start)
            self.assertNotIn('TORCH_CHECK', code[start:end])
            self.assertNotIn('at::empty', code[start:end])


if __name__ == '__main__':
    unittest.main()
