"""Stage 7 source contracts; no dependencies, tensors, or model execution."""

import ast
import unittest

from test_cpu_static import ROOT, calls, keywords, parse
from test_triangle_attention_static import definition


class CacheStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = parse('run_alphafold.py')
        cls.cache = parse('xfold/nn/cache.py')
        cls.input_cache = definition(cls.cache, 'InputCache')
        cls.glu = parse('xfold/fastnn/gated_linear_unit.py')
        cls.weight_cache = definition(cls.glu, 'GLUWeightCache')
        cls.model = parse('xfold/alphafold3.py')
        cls.bindings = (ROOT / 'af3_kernels/src/csrc/bindings.cpp').read_text()

    def test_cache_flags_are_explicit_and_off_by_default(self):
        flag = next(call for call in calls(self.runner, 'flags.DEFINE_enum')
                    if ast.literal_eval(call.args[0]) == 'cache_mode')
        self.assertEqual(ast.literal_eval(flag.args[1]), 'off')
        self.assertEqual(ast.literal_eval(flag.args[2]), ['off', 'weights', 'inputs', 'all'])
        init = definition(definition(self.runner, 'ModelRunner'), '__init__')
        defaults = dict(zip([arg.arg for arg in init.args.args[-len(init.args.defaults):]], init.args.defaults))
        self.assertEqual(ast.literal_eval(defaults['cache_mode']), 'off')
        self.assertEqual(ast.literal_eval(defaults['input_cache_max_mb']), 512)
        main_call = calls(self.runner, 'ModelRunner')[0]
        self.assertEqual(keywords(main_call)['cache_mode'], '_CACHE_MODE.value')
        self.assertEqual(keywords(main_call)['input_cache_max_mb'], '_INPUT_CACHE_MAX_MB.value')
        configuration = parse('xfold/fastnn/config.py')
        for name in ('input_cache_enabled', 'packed_weight_cache_enabled'):
            node = next(n for n in configuration.body if isinstance(n, ast.Assign)
                        and ast.unparse(n.targets[0]) == name)
            self.assertIs(ast.literal_eval(node.value), False)

    def test_budget_validation_and_global_reset_precede_model_creation(self):
        init = definition(definition(self.runner, 'ModelRunner'), '__init__')
        creation = calls(init, 'AlphaFold3')[0]
        guard = next(n for n in init.body if isinstance(n, ast.If)
                     and 'input_cache_max_mb < 0' in ast.unparse(n.test))
        self.assertTrue(any(isinstance(n, ast.Raise) for n in guard.body))
        self.assertLess(guard.lineno, creation.lineno)
        expected = {
            'input_cache_enabled': "cache_mode in ('inputs', 'all')",
            'packed_weight_cache_enabled': "cache_mode in ('weights', 'all')",
            'input_cache_max_bytes': 'input_cache_max_mb * 1024 * 1024',
        }
        for name, value in expected.items():
            node = next(n for n in init.body if isinstance(n, ast.Assign)
                        and ast.unparse(n.targets[0]) == f'fastnn_config.{name}')
            self.assertEqual(ast.unparse(node.value), value)
            self.assertLess(node.lineno, creation.lineno)

    def test_old_packing_api_fails_early_only_for_requested_native_caches(self):
        init = definition(definition(self.runner, 'ModelRunner'), '__init__')
        branch = next(n for n in init.body if isinstance(n, ast.If)
                      and "'cpp' in (glu_backend, triangle_backend, attention_backend)" in ast.unparse(n.test))
        self.assertIn("cache_mode in ('weights', 'all')", ast.unparse(branch.test))
        self.assertIn("getattr(extension, 'packed_weight_api', 0) != 1", ast.unparse(branch))
        self.assertLess(branch.end_lineno, calls(init, 'AlphaFold3')[0].lineno)
        self.assertIn('module.attr("packed_weight_api") = 1', self.bindings)

    def test_scope_is_unique_and_exception_safe(self):
        scope = definition(self.cache, 'input_cache_scope')
        self.assertEqual(len(calls(scope, '_Scope')), 1)
        self.assertEqual(len(calls(scope, '_ACTIVE_SCOPE.set')), 1)
        guarded = next(n for n in scope.body if isinstance(n, ast.Try))
        self.assertTrue(any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Yield) for n in guarded.body))
        cleanup = ast.Module(body=guarded.finalbody, type_ignores=[])
        self.assertEqual(len(calls(cleanup, 'cache.clear')), 1)
        self.assertEqual(len(calls(cleanup, '_ACTIVE_SCOPE.reset')), 1)

    def test_every_model_call_resets_before_after_and_between_phases(self):
        model = definition(self.model, 'AlphaFold3')
        forward = definition(model, 'forward')
        self.assertEqual(ast.unparse(forward.body[0]), 'self.reset_input_cache()')
        guarded = next(n for n in forward.body if isinstance(n, ast.Try))
        self.assertIn('with input_cache_scope():', ast.unparse(guarded))
        self.assertEqual(ast.unparse(guarded.finalbody[0]), 'self.reset_input_cache()')
        pipeline = definition(model, '_forward_impl')
        diffusion = calls(pipeline, 'self._sample_diffusion')[0]
        resets = calls(pipeline, 'self.reset_input_cache')
        self.assertEqual(len(resets), 2)
        self.assertLess(resets[0].lineno, diffusion.lineno)
        self.assertGreater(resets[1].lineno, diffusion.lineno)
        reset = definition(self.cache, 'reset_input_cache')
        self.assertIn("isinstance(cache, InputCache)", ast.unparse(reset))
        self.assertNotIn('_packed', ast.unparse(reset))

    def test_training_gradients_and_unscoped_calls_never_reuse_inputs(self):
        get = definition(self.input_cache, 'get')
        guard = next(n for n in get.body if isinstance(n, ast.If))
        self.assertEqual(ast.unparse(guard.test),
                         'not config.input_cache_enabled or scope is None or owner.training or torch.is_grad_enabled()')
        self.assertEqual(ast.unparse(guard.body[-1]), 'return factory()')
        self.assertEqual(len(calls(guard, 'self.clear')), 1)

    def test_keys_track_identity_mutation_layout_precision_and_module_settings(self):
        signature = definition(self.cache, '_signature')
        for value in ('id(value)', 'value.data_ptr()', 'value._version', 'tuple(value.shape)',
                      'tuple(value.stride())', 'value.dtype', 'value.device', 'dataclasses.fields(value)'):
            self.assertIn(value, ast.unparse(signature))
        get = definition(self.input_cache, 'get')
        for value in ("torch.is_autocast_enabled('cpu')", "torch.is_autocast_enabled('cuda')",
                      'allow_inference=False', "('eps', 'normalized_shape')", 'config.packed_weight_cache_enabled',
                      '(key, value, dependencies, parameters, scope, size)'):
            self.assertIn(value, ast.unparse(get))
        self.assertIn('key = (scope,', ast.unparse(get))

    def test_versionless_features_require_scope_but_versionless_weights_bypass_reuse(self):
        signature = definition(self.cache, '_signature')
        handler = next(n for n in ast.walk(signature) if isinstance(n, ast.ExceptHandler))
        self.assertIn('not allow_inference or not value.is_inference()', ast.unparse(handler))
        self.assertIn('raise _Uncacheable', ast.unparse(handler))
        get = definition(self.input_cache, 'get')
        self.assertIn('_signature(parameters, allow_inference=False)', ast.unparse(get))
        self.assertIn('scope is None', ast.unparse(get))

    def test_storage_budget_and_replacement_release_are_explicit(self):
        size = definition(self.cache, '_retained_bytes')
        self.assertIn('item.untyped_storage()', ast.unparse(size))
        self.assertIn('storage.nbytes()', ast.unparse(size))
        get = definition(self.input_cache, 'get')
        guard = next(n for n in get.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'size > scope.limit - scope.used')
        self.assertEqual(ast.unparse(guard.body[-1]), 'return value')
        self.assertIn('scope.used += size', ast.unparse(get))
        discard = definition(self.input_cache, '_discard')
        self.assertIn('entry[4].used -= entry[5]', ast.unparse(discard))

    def test_recycle_and_conditioning_mutable_outputs_are_copied(self):
        evo = definition(self.model, 'Evoformer')
        sequence = definition(evo, '_seq_pair_embedding')
        self.assertEqual(keywords(calls(sequence, 'self._input_cache.get')[0])['copy'], 'clone_tensors')
        forward = definition(evo, 'forward')
        single = next(call for call in calls(forward, 'self._input_cache.get')
                      if ast.literal_eval(call.args[0]) == 'single')
        self.assertEqual(keywords(single)['copy'], 'clone_tensors')
        head = definition(parse('xfold/nn/diffusion_head.py'), 'DiffusionHead')
        conditioning = definition(head, '_conditioning')
        self.assertEqual(keywords(calls(conditioning, 'self._input_cache.get')[0])['copy'],
                         'lambda value: (value[0].clone(), value[1])')
        static = definition(head, '_static_conditioning')
        self.assertNotIn('noise_level', ast.unparse(static))
        self.assertNotIn('fourier_embeddings', ast.unparse(static))
        self.assertEqual(len(calls(conditioning, 'self.fourier_embeddings')), 1)

    def test_atom_conditioning_does_not_cache_noisy_positions(self):
        encoder = definition(parse('xfold/nn/atom_cross_attention.py'), 'AtomCrossAttEncoder')
        static = definition(encoder, '_static_conditioning')
        self.assertNotIn('token_atoms_act', ast.unparse(static))
        self.assertFalse(calls(static, 'self.atom_positions_to_features'))
        forward = definition(encoder, 'forward')
        self.assertTrue(calls(forward, 'self.atom_positions_to_features'))
        self.assertTrue(calls(forward, 'self.atom_transformer_encoder'))
        dependencies = calls(forward, 'self._input_cache.get')[0].args[1]
        self.assertIn('trunk_single_cond', ast.unparse(dependencies))
        self.assertIn('trunk_pair_cond', ast.unparse(dependencies))
        self.assertNotIn('token_atoms_act', ast.unparse(dependencies))

    def test_pair_logits_cache_dependencies_include_conditioning_and_heads(self):
        tree = parse('xfold/nn/diffusion_transformer.py')
        for name in ('DiffusionTransformer', 'DiffusionCrossAttTransformer'):
            forward = definition(definition(tree, name), 'forward')
            cached = next(call for call in calls(forward, 'self._input_cache.get')
                          if 'pair_logits' in ast.unparse(call.args[0]))
            self.assertEqual(ast.unparse(cached.args[1]), '(pair_cond, self.num_head)')
            self.assertNotIn('queries_act', ast.unparse(cached.args[1]))
            self.assertNotIn('single_cond', ast.unparse(cached.args[1]))

    def test_packed_weight_cache_pads_and_invalidates_with_the_source_key(self):
        packed = definition(self.weight_cache, 'prepare_packed')
        self.assertIn('dtype != torch.bfloat16', ast.unparse(packed))
        self.assertIn('matrix = self.prepare(weight, dtype)', ast.unparse(packed))
        self.assertIn('if self._packed is not None:', ast.unparse(packed))
        self.assertIn('matrix.size(0) + 1', ast.unparse(packed))
        self.assertIn('extension.vnni_repack_tensor(matrix, 2, 0)', ast.unparse(packed))
        store = next(n for n in packed.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'self._key is not None')
        self.assertIn('self._packed = packed', ast.unparse(store))
        prepare = definition(self.weight_cache, 'prepare')
        branch = next(n for n in prepare.body if isinstance(n, ast.If))
        self.assertEqual(ast.unparse(branch.test), 'self._key != key')
        self.assertIn('self._packed = None', ast.unparse(branch))

    def test_native_packed_branches_validate_bf16_and_skip_double_packing(self):
        for path, flag in (
            ('gated_linear_unit.cpp', 'weight_packed'), ('matrix.cpp', 'b_packed'),
            ('triangle_multiplication.cpp', 'weights_packed'),
        ):
            source = (ROOT / 'af3_kernels/src/csrc' / path).read_text()
            self.assertIn(f'!{flag} ||', source)
            self.assertIn(f'{flag} ? ', source)
            if path != 'triangle_multiplication.cpp':
                start = source.index(f'if (!{flag})')
                end = source.index('\n        }', start)
                self.assertIn('vnni_repack_tensor(', source[start:end])
        triangle = (ROOT / 'af3_kernels/src/csrc/triangle_multiplication.cpp').read_text()
        self.assertIn('tiled_matmul(left, right, c10::nullopt, block_size, block_size)', triangle)
        self.assertNotIn('block_size, block_size, weights_packed)\n                         .permute', triangle)


if __name__ == '__main__':
    unittest.main()
