"""Source-level CPU checks; no model dependencies or inference are executed."""

import ast
import fnmatch
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


def parse(relative_path):
    return ast.parse((ROOT / relative_path).read_text(), filename=relative_path)


def calls(node, name):
    return [
        item for item in ast.walk(node)
        if isinstance(item, ast.Call) and ast.unparse(item.func) == name
    ]


def keywords(call):
    return {item.arg: ast.unparse(item.value) for item in call.keywords}


def eager_imports(node):
    """Visit import-time code, excluding deferred function and method bodies."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return
    if isinstance(node, ast.Import):
        yield from (item.name for item in node.names)
    elif isinstance(node, ast.ImportFrom):
        yield node.module or ''
        yield from (item.name for item in node.names)
    for child in ast.iter_child_nodes(node):
        yield from eager_imports(child)


class CPUStaticTests(unittest.TestCase):
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

    def test_all_python_sources_compile(self):
        paths = [ROOT / 'run_alphafold.py']
        paths.extend((ROOT / 'xfold').rglob('*.py'))
        paths.extend((ROOT / 'tests').rglob('*.py'))
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                compile(path.read_text(), str(path), 'exec')

    def test_pytorch_path_has_no_eager_gpu_dependencies(self):
        for path in (ROOT / 'xfold').rglob('*.py'):
            if path.name.startswith('_triton_'):
                continue
            with self.subTest(path=path.relative_to(ROOT)):
                for name in eager_imports(ast.parse(path.read_text())):
                    self.assertFalse(name.startswith(('triton', '_triton_')))
                    self.assertFalse(name.startswith(('af3_kernels', 'intel_extension_for_pytorch', 'oneccl')))

    def test_distribution_includes_model_and_lazy_kernel_packages(self):
        config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        discovery = config['tool']['setuptools']['packages']['find']
        self.assertTrue(discovery['namespaces'])
        for path in (ROOT / 'xfold').rglob('*.py'):
            package = '.'.join(path.parent.relative_to(ROOT).parts)
            with self.subTest(package=package):
                self.assertTrue(any(fnmatch.fnmatchcase(package, pattern) for pattern in discovery['include']))
                self.assertFalse(any(fnmatch.fnmatchcase(package, pattern) for pattern in discovery.get('exclude', [])))

    def test_deferred_triton_imports_resolve_to_source_modules(self):
        for path in (ROOT / 'xfold' / 'fastnn').glob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('_triton_'):
                    with self.subTest(path=path.name, module=node.module):
                        self.assertEqual(node.level, 1)
                        module = path.parent / (node.module + '.py')
                        self.assertTrue(module.is_file())
                        definitions = {
                            item.name for item in ast.parse(module.read_text()).body
                            if isinstance(item, (ast.FunctionDef, ast.ClassDef))
                        }
                        self.assertTrue({item.name for item in node.names}.issubset(definitions))

    def test_device_flag_and_auto_cpu_fallback(self):
        device_flags = [
            call for call in calls(self.runner, 'flags.DEFINE_enum')
            if ast.literal_eval(call.args[0]) == 'device'
        ]
        self.assertEqual(len(device_flags), 1)
        self.assertEqual(ast.literal_eval(device_flags[0].args[1]), 'auto')
        self.assertEqual(ast.literal_eval(device_flags[0].args[2]), ['auto', 'cpu', 'cuda'])
        main = next(node for node in self.runner.body if getattr(node, 'name', None) == 'main')
        selections = [
            node for node in ast.walk(main) if isinstance(node, ast.IfExp)
            and ast.unparse(node.test) == 'torch.cuda.is_available()'
        ]
        self.assertEqual(len(selections), 1)
        self.assertEqual(ast.literal_eval(selections[0].body), 'cuda')
        self.assertEqual(ast.literal_eval(selections[0].orelse), 'cpu')

    def test_cpu_backend_resets_all_operator_choices_before_model_creation(self):
        init = self.method('__init__')
        assignments = [node for node in init.body if isinstance(node, ast.Assign)]
        choice = next(
            node for node in assignments
            if any(ast.unparse(target) == 'implementation' for target in node.targets)
        )
        self.assertIsInstance(choice.value, ast.IfExp)
        self.assertEqual(ast.unparse(choice.value.test), "self._device.type == 'cuda' and _USE_FASTNN.value")
        self.assertEqual(ast.literal_eval(choice.value.body), 'triton')
        self.assertEqual(ast.literal_eval(choice.value.orelse), 'torch')
        model_creation = calls(init, 'AlphaFold3')[0]
        for backend in ('layer_norm', 'dot_product_attention', 'gated_linear_unit'):
            assignment = next(
                node for node in assignments
                if any(ast.unparse(target) == f'fastnn_config.{backend}_implementation' for target in node.targets)
            )
            self.assertEqual(ast.unparse(assignment.value), 'implementation')
            self.assertLess(assignment.lineno, model_creation.lineno)
        model_to = calls(init, 'self._model.to')
        self.assertEqual(len(model_to), 1)
        self.assertEqual(keywords(model_to[0])['dtype'], 'torch.float32')

    def test_cpu_float_features_preserve_integer_and_boolean_dtypes(self):
        inference = self.method('run_inference')
        conversions = [
            node for node in ast.walk(inference)
            if isinstance(node, ast.Lambda) and isinstance(node.body, ast.IfExp)
            and calls(node.body.test, 'x.is_floating_point')
        ]
        self.assertEqual(len(conversions), 1)
        conversion = conversions[0].body
        self.assertEqual(ast.unparse(conversion.test), "self._device.type == 'cpu' and x.is_floating_point()")
        self.assertEqual(keywords(conversion.body)['dtype'], 'torch.float32')
        self.assertNotIn('dtype', keywords(conversion.orelse))

    def test_autocast_and_synchronization_are_cuda_only(self):
        inference = self.method('run_inference')
        contexts = [
            node for node in ast.walk(inference) if isinstance(node, ast.IfExp)
            and calls(node.body, 'torch.amp.autocast')
        ]
        self.assertEqual(len(contexts), 1)
        self.assertEqual(ast.unparse(contexts[0].test), "self._device.type == 'cuda'")
        self.assertEqual(ast.unparse(contexts[0].orelse), 'nullcontext()')
        autocast = calls(self.runner, 'torch.amp.autocast')
        self.assertEqual(len(autocast), 1)
        self.assertEqual(keywords(autocast[0])['dtype'], 'torch.bfloat16')
        sync = self.method('synchronize')
        guard = next(node for node in sync.body if isinstance(node, ast.If))
        self.assertEqual(ast.unparse(guard.test), "self._device.type == 'cuda'")
        sync_calls = calls(self.runner, 'torch.cuda.synchronize')
        self.assertEqual(len(sync_calls), 1)
        self.assertIn(sync_calls[0], calls(guard, 'torch.cuda.synchronize'))
        self.assertEqual(ast.unparse(sync_calls[0].args[0]), 'self._device')
        self.assertEqual(len(calls(self.runner, 'model_runner.synchronize')), 2)

    def test_triton_modules_are_deferred_and_require_cuda(self):
        wrappers = (
            ('xfold/fastnn/attention.py', 'dot_product_attention_triton'),
            ('xfold/fastnn/gated_linear_unit.py', 'gated_linear_unit_triton'),
            ('xfold/fastnn/layer_norm.py', 'forward'),
        )
        for path, name in wrappers:
            with self.subTest(path=path):
                tree = parse(path)
                wrapper = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
                imports = [
                    node for node in ast.walk(wrapper) if isinstance(node, ast.ImportFrom)
                    and (node.module or '').startswith('_triton_')
                ]
                self.assertEqual(len(imports), 1)
                guards = [
                    node for node in ast.walk(wrapper) if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Compare)
                    and ast.unparse(node.test).endswith(".device.type != 'cuda'")
                    and any(isinstance(item, ast.Raise) for item in node.body)
                ]
                self.assertEqual(len(guards), 1)
                self.assertLess(guards[0].lineno, imports[0].lineno)

    def test_explicit_unavailable_cuda_is_rejected_before_loading_weights(self):
        init = self.method('__init__')
        guard = next(
            node for node in init.body if isinstance(node, ast.If)
            and ast.unparse(node.test) == "self._device.type == 'cuda' and (not torch.cuda.is_available())"
        )
        self.assertTrue(any(isinstance(node, ast.Raise) for node in guard.body))
        self.assertLess(guard.lineno, calls(init, 'import_jax_weights_')[0].lineno)


if __name__ == '__main__':
    unittest.main()
