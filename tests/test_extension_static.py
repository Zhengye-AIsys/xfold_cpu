"""Native infrastructure source checks; no dependencies, compilation, or JIT run."""

import ast
import hashlib
import json
from pathlib import Path
import re
import tomllib
import unittest

from test_cpu_static import ROOT, calls, eager_imports


EXTENSION = ROOT / 'af3_kernels'
PACKAGE = EXTENSION / 'src/af3_kernels'
VENDOR = EXTENSION / 'third_party/libxsmm'
REVISION = '0d1f61ed425aae8318fffafeea9834971a33aec7'


class ExtensionStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup = ast.parse((EXTENSION / 'setup.py').read_text())
        cls.package = ast.parse((PACKAGE / '__init__.py').read_text())
        cls.native = (EXTENSION / 'src/csrc/tools.cpp').read_text()
        cls.bindings = (EXTENSION / 'src/csrc/bindings.cpp').read_text()
        cls.build_support = ast.parse((EXTENSION / 'build_support.py').read_text())

    def test_extension_python_sources_compile(self):
        paths = [EXTENSION / 'setup.py', EXTENSION / 'build_support.py']
        paths.extend((EXTENSION / 'src').rglob('*.py'))
        for path in paths:
            with self.subTest(path=path.relative_to(EXTENSION)):
                compile(path.read_text(), str(path), 'exec')

    def test_optional_package_has_no_eager_torch_or_native_loading(self):
        imports = set(eager_imports(self.package))
        self.assertNotIn('torch', imports)
        self.assertNotIn('_C', imports)
        self.assertNotIn('af3_kernels._C', imports)
        module_calls = [node for node in self.package.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
        self.assertFalse(module_calls)
        self.assertFalse(calls(self.package, 'subprocess.check_call'))
        self.assertFalse(calls(self.package, 'torch.utils.cpp_extension.load'))
        loader = next(node for node in self.package.body if getattr(node, 'name', None) == 'load_extension')
        modules = [ast.literal_eval(call.args[0]) for call in calls(loader, 'import_module')]
        self.assertEqual(modules, ['torch', 'af3_kernels._C'])
        self.assertTrue(any(isinstance(node, ast.Try) for node in loader.body))

    def test_model_and_runner_never_import_the_optional_extension(self):
        paths = [ROOT / 'run_alphafold.py']
        paths.extend((ROOT / 'xfold').rglob('*.py'))
        for path in paths:
            with self.subTest(path=path.relative_to(ROOT)):
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        self.assertFalse(any(item.name.startswith('af3_kernels') for item in node.names))
                    elif isinstance(node, ast.ImportFrom):
                        self.assertFalse((node.module or '').startswith('af3_kernels'))

    def test_main_distribution_does_not_require_native_package(self):
        config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        self.assertFalse(any('af3' in dependency or 'libxsmm' in dependency for dependency in config['project']['dependencies']))
        self.assertEqual(config['tool']['setuptools']['packages']['find']['include'], ['xfold', 'xfold.*'])

    def test_native_package_metadata_and_source_distribution_inputs(self):
        config = tomllib.loads((EXTENSION / 'pyproject.toml').read_text())
        self.assertEqual(config['project']['name'], 'af3_kernels')
        self.assertEqual(config['tool']['setuptools']['packages']['find']['where'], ['src'])
        self.assertEqual(config['tool']['setuptools']['packages']['find']['include'], ['af3_kernels'])
        self.assertIn('licenses/*.txt', config['tool']['setuptools']['package-data']['af3_kernels'])
        self.assertFalse(any(name.startswith('torch') for name in config['build-system']['requires']))
        manifest = (EXTENSION / 'MANIFEST.in').read_text()
        for item in ('libxsmm.lock.json', 'build_support.py', 'third_party/libxsmm', '*.cpp', '*.h'):
            self.assertIn(item, manifest)

    def test_libxsmm_build_source_matches_pinned_manifest(self):
        manifest = json.loads((EXTENSION / 'libxsmm.lock.json').read_text())
        self.assertEqual(manifest['revision'], REVISION)
        self.assertEqual(manifest['repository'], 'https://github.com/libxsmm/libxsmm.git')
        self.assertGreater(len(manifest['files']), 200)
        for name, expected in manifest['files'].items():
            with self.subTest(path=name):
                path = VENDOR / name
                self.assertTrue(path.is_file())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
                self.assertNotIn('..', Path(name).parts)
        files = {str(path.relative_to(VENDOR)) for path in VENDOR.rglob('*') if path.is_file()}
        self.assertEqual(files, set(manifest['files']))
        self.assertFalse((VENDOR / '.git').exists())

    def test_build_validates_and_copies_sources_before_make(self):
        build = next(node for node in self.setup.body if isinstance(node, ast.ClassDef))
        run = next(node for node in build.body if getattr(node, 'name', None) == 'run')
        stage = calls(run, 'stage_libxsmm')
        make = calls(run, 'subprocess.check_call')
        self.assertEqual(len(stage), 1)
        self.assertEqual(len(make), 1)
        self.assertLess(stage[0].lineno, make[0].lineno)
        self.assertEqual(ast.unparse(stage[0].args[1]), 'self.build_temp')
        staging = next(node for node in self.build_support.body if getattr(node, 'name', None) == 'stage_libxsmm')
        validate = calls(staging, 'verify_libxsmm')
        fresh = calls(staging, 'tempfile.mkdtemp')
        copy = calls(staging, 'shutil.copy2')
        self.assertEqual(len(validate), 1)
        self.assertEqual(len(fresh), 1)
        self.assertEqual(len(copy), 1)
        self.assertLess(validate[0].lineno, fresh[0].lineno)
        self.assertLess(fresh[0].lineno, copy[0].lineno)
        self.assertEqual(ast.unparse(copy[0].args[0]), 'source / name')
        self.assertEqual(ast.unparse(copy[0].args[1]), 'target')
        loop = next(node for node in staging.body if isinstance(node, ast.For))
        self.assertEqual(ast.unparse(loop.iter), "libxsmm_manifest()['files'].items()")
        self.assertIn('workspace / name', ast.unparse(loop))
        self.assertIn('hashlib.sha256(target.read_bytes()).hexdigest() != expected', ast.unparse(loop))
        self.assertIn('build_root == source', ast.unparse(staging))
        self.assertFalse(calls(staging, 'shutil.copytree'))
        args = make[0].args[0]
        self.assertIsInstance(args, ast.List)
        literals = [ast.literal_eval(item) for item in args.elts if isinstance(item, ast.Constant)]
        for argument in ('make', '-C', 'STATIC=1', 'PIC=pic', 'BLAS=0', 'FORTRAN=0', 'lib/libxsmm.a', 'lib/libxsmmnoblas.a'):
            self.assertIn(argument, literals)
        self.assertIn('str(workspace)', ast.unparse(args))
        self.assertIn('extension.extra_objects', ast.unparse(run))

    def test_cpp_extension_has_matching_binding_and_openmp_link_flags(self):
        extension = calls(self.setup, 'CppExtension')
        self.assertEqual(len(extension), 1)
        self.assertEqual(ast.literal_eval(extension[0].args[0]), 'af3_kernels._C')
        options = {item.arg: item.value for item in extension[0].keywords}
        for name in ast.literal_eval(options['sources']):
            self.assertFalse(Path(name).is_absolute())
            self.assertTrue((EXTENSION / name).is_file())
        self.assertIn('-fopenmp', ast.literal_eval(options['extra_compile_args']))
        self.assertIn('-fopenmp', ast.literal_eval(options['extra_link_args']))
        self.assertIn('-march=native', ast.literal_eval(options['extra_compile_args']))
        self.assertIn('PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)', self.bindings)
        self.assertIn('platform.system()', ast.unparse(self.setup))
        self.assertIn("'Linux'", ast.unparse(self.setup))

    def test_python_native_tools_match_bindings(self):
        names = set(re.findall(r'module\.def\("([a-z_]+)"', self.bindings))
        self.assertEqual(names, {'runtime_info', 'pad_and_align_tensor', 'vnni_repack_tensor', 'gemm'})
        for name in names:
            function = next(node for node in self.package.body if getattr(node, 'name', None) == name)
            self.assertEqual(len(calls(function, f'load_extension().{name}')), 1)
        self.assertNotIn('gated_linear_unit', names)
        self.assertNotIn('triangle_multiplication', names)
        self.assertNotIn('attention', names)

    def test_native_libxsmm_api_names_exist_in_pinned_headers(self):
        code = self.native + self.bindings
        header = (VENDOR / 'include/libxsmm.h').read_text()
        typedefs = (VENDOR / 'include/libxsmm_typedefs.h').read_text()
        apis = set(re.findall(r'\b(libxsmm_[a-z_]+)\s*\(', code))
        for api in apis:
            with self.subTest(api=api):
                self.assertRegex(header, rf'\b{api}\s*\(')
        constants = set(re.findall(r'\bLIBXSMM_[A-Z0-9_]+\b', code))
        for name in constants:
            self.assertIn(name, typedefs)
        self.assertIn('LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32', self.native)
        self.assertIn('kernel != nullptr', self.native)

    def test_native_input_and_allocation_guards_are_present(self):
        for requirement in (
            'tensor.device().is_cpu()', 'tensor.layout() == at::kStrided',
            '!tensor.requires_grad()', '!tensor.is_conj()', '!tensor.is_neg()',
            'size >= 0', 'Tensor size overflow',
            'Tensor byte size overflow', 'posix_memalign', 'std::free(pointer)',
            'std::shared_ptr<void>', '[owner](void*)',
            'shape.size()', 'Padding cannot shrink dimensions',
            'tensor.is_contiguous()', 'rows % 2 == 0', 'cols % block == 0',
            'type == 2', 'a.scalar_type() == b.scalar_type()', 'a.size(1) == b.size(0)',
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, self.native)

    def test_new_source_has_no_reference_brand_copyright_claim(self):
        for path in (EXTENSION / 'src/csrc').glob('*'):
            with self.subTest(path=path.name):
                self.assertNotIn('Xflops', path.read_text())
                self.assertIn('SPDX-License-Identifier: BSD-3-Clause', path.read_text())

    def test_manifest_paths_cannot_escape_source_tree(self):
        manifest = next(node for node in self.build_support.body if getattr(node, 'name', None) == 'libxsmm_manifest')
        loop = next(node for node in manifest.body if isinstance(node, ast.For))
        guard = next(node for node in loop.body if isinstance(node, ast.If))
        self.assertEqual(ast.unparse(guard.test), "path.is_absolute() or '..' in path.parts")
        self.assertTrue(any(isinstance(node, ast.Raise) for node in guard.body))

    def test_license_notices_are_retained_and_packaged(self):
        self.assertIn('Copyright (c) 2025, Xflops', (EXTENSION / 'LICENSE').read_text())
        self.assertIn('Copyright (c) 2022, Intel Corporation', (EXTENSION / 'LICENSE').read_text())
        self.assertEqual((PACKAGE / 'licenses/LIBXSMM.txt').read_bytes(), (VENDOR / 'LICENSE.md').read_bytes())


if __name__ == '__main__':
    unittest.main()
