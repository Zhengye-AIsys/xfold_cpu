"""Static precision checks; no tensors, weights, or inference are executed."""

import ast
import unittest

from test_cpu_static import ROOT, calls, keywords, parse


class BF16StaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = parse('run_alphafold.py')
        cls.model_runner = next(
            node for node in cls.runner.body
            if isinstance(node, ast.ClassDef) and node.name == 'ModelRunner'
        )

    def method(self, name):
        return next(
            node for node in self.model_runner.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )

    def test_precision_defaults_to_fp32_and_bf16_is_explicit(self):
        flag = next(
            call for call in calls(self.runner, 'flags.DEFINE_enum')
            if ast.literal_eval(call.args[0]) == 'cpu_precision'
        )
        self.assertEqual(ast.literal_eval(flag.args[1]), 'fp32')
        self.assertEqual(ast.literal_eval(flag.args[2]), ['fp32', 'bf16'])
        init = self.method('__init__')
        defaults = dict(zip(
            [arg.arg for arg in init.args.args[-len(init.args.defaults):]],
            init.args.defaults,
        ))
        self.assertEqual(ast.literal_eval(defaults['cpu_precision']), 'fp32')
        main = next(node for node in self.runner.body if getattr(node, 'name', None) == 'main')
        constructor = calls(main, 'ModelRunner')
        self.assertEqual(len(constructor), 1)
        self.assertEqual(keywords(constructor[0])['cpu_precision'], '_CPU_PRECISION.value')

    def test_precision_is_validated_before_model_creation(self):
        init = self.method('__init__')
        guards = [node for node in init.body if isinstance(node, ast.If)]
        supported = next(
            node for node in guards
            if ast.unparse(node.test) == "cpu_precision not in ('fp32', 'bf16')"
        )
        cpu_only = next(
            node for node in guards
            if ast.unparse(node.test) == "self._device.type != 'cpu' and cpu_precision != 'fp32'"
        )
        creation = calls(init, 'AlphaFold3')[0]
        for guard in (supported, cpu_only):
            self.assertTrue(any(isinstance(node, ast.Raise) for node in guard.body))
            self.assertLess(guard.lineno, creation.lineno)

    def test_cpu_bf16_enables_autocast_and_fp32_disables_it(self):
        context = self.method('_inference_context')
        branch = next(node for node in context.body if isinstance(node, ast.If))
        self.assertEqual(ast.unparse(branch.test), "self._device.type == 'cpu'")
        autocast = calls(branch, 'torch.amp.autocast')
        self.assertEqual(len(autocast), 1)
        options = keywords(autocast[0])
        self.assertEqual(options['device_type'], "'cpu'")
        self.assertEqual(options['dtype'], 'torch.bfloat16')
        enabled = next(item.value for item in autocast[0].keywords if item.arg == 'enabled')
        self.assertIsInstance(enabled, ast.Compare)
        self.assertEqual(ast.unparse(enabled.left), 'self._cpu_precision')
        self.assertEqual(len(enabled.ops), 1)
        self.assertIsInstance(enabled.ops[0], ast.Eq)
        self.assertEqual(ast.literal_eval(enabled.comparators[0]), 'bf16')
        for precision, expected in (('fp32', False), ('bf16', True)):
            with self.subTest(precision=precision):
                self.assertEqual(precision == ast.literal_eval(enabled.comparators[0]), expected)

    def test_autocast_wraps_forward_and_ends_before_output_conversion(self):
        inference = self.method('run_inference')
        region = next(node for node in inference.body if isinstance(node, ast.With))
        self.assertEqual(ast.unparse(region.items[0].context_expr), 'self._inference_context()')
        self.assertEqual(len(calls(region, 'self._model')), 1)
        conversions = [
            node for node in ast.walk(inference) if isinstance(node, ast.Lambda)
            and isinstance(node.body, ast.IfExp)
            and ast.unparse(node.body.test) == 'x.dtype == torch.bfloat16'
        ]
        self.assertEqual(len(conversions), 1)
        conversion = conversions[0]
        self.assertEqual(keywords(conversion.body.body)['dtype'], 'torch.float32')
        self.assertEqual(ast.unparse(conversion.body.orelse), 'x')
        self.assertGreater(conversion.lineno, region.end_lineno)
        numpy_output = calls(inference, 'x.cpu().detach().numpy')
        self.assertEqual(len(numpy_output), 1)
        self.assertGreater(numpy_output[0].lineno, conversion.end_lineno)

    def test_weights_and_input_storage_stay_fp32_without_global_dtype_changes(self):
        init = self.method('__init__')
        model_to = calls(init, 'self._model.to')
        self.assertEqual(len(model_to), 1)
        self.assertEqual(keywords(model_to[0])['dtype'], 'torch.float32')
        inference = self.method('run_inference')
        input_cast = next(
            node for node in ast.walk(inference) if isinstance(node, ast.Lambda)
            and isinstance(node.body, ast.IfExp) and calls(node.body.test, 'x.is_floating_point')
        )
        self.assertEqual(keywords(input_cast.body.body)['dtype'], 'torch.float32')
        self.assertNotIn('dtype', keywords(input_cast.body.orelse))
        self.assertFalse(calls(self.runner, 'torch.set_default_dtype'))
        for node in ast.walk(self.runner):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotEqual(node.func.attr, 'bfloat16')
                if node.func.attr == 'to':
                    self.assertNotIn('torch.bfloat16', keywords(node).values())

    def test_cpu_operator_backend_does_not_depend_on_precision(self):
        init = self.method('__init__')
        backend = next(
            node for node in init.body if isinstance(node, ast.Assign)
            and any(ast.unparse(target) == 'implementation' for target in node.targets)
        )
        self.assertEqual(ast.literal_eval(backend.value.orelse), 'torch')
        self.assertNotIn('precision', ast.unparse(backend.value))

    def test_diffusion_schedule_and_initial_coordinates_have_no_bf16_casts(self):
        model = parse('xfold/alphafold3.py')
        sampling = next(
            node for node in ast.walk(model)
            if isinstance(node, ast.FunctionDef) and node.name == '_sample_diffusion'
        )
        for factory in ('torch.linspace', 'torch.randn'):
            tensor_calls = calls(sampling, factory)
            self.assertEqual(len(tensor_calls), 1)
            self.assertEqual(keywords(tensor_calls[0])['device'], 'device')
            self.assertNotIn('dtype', keywords(tensor_calls[0]))
        self.assertNotIn('bfloat16', ast.unparse(sampling))
        self.assertNotIn('set_default_dtype', (ROOT / 'run_alphafold.py').read_text())


if __name__ == '__main__':
    unittest.main()
