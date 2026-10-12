"""Stage 8 source/shell syntax contracts; no Torch/IPEX imports or execution."""

import ast
import subprocess
import unittest

from test_cpu_static import ROOT, calls, eager_imports, keywords, parse
from test_triangle_attention_static import definition


class RuntimeStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = parse('run_alphafold.py')
        cls.init = definition(definition(cls.runner, 'ModelRunner'), '__init__')
        cls.layer = definition(parse('xfold/fastnn/layer_norm.py'), 'LayerNorm')
        cls.bridge = parse('xfold/fastnn/_ipex_layer_norm.py')
        cls.runtime = parse('xfold/cpu_runtime.py')
        cls.launcher = (ROOT / 'scripts/run_cpu.sh').read_text()

    def test_layer_norm_choice_defaults_preserve_prior_cpu_cuda_behavior(self):
        flag = next(c for c in calls(self.runner, 'flags.DEFINE_enum')
                    if ast.literal_eval(c.args[0]) == 'layer_norm_backend')
        self.assertEqual(ast.literal_eval(flag.args[1]), 'auto')
        self.assertEqual(ast.literal_eval(flag.args[2]), ['auto', 'torch', 'ipex'])
        selected = next(n for n in self.init.body if isinstance(n, ast.Assign)
                        and ast.unparse(n.targets[0]) == 'fastnn_config.layer_norm_implementation')
        self.assertEqual(ast.unparse(selected.value),
                         "implementation if layer_norm_backend == 'auto' else layer_norm_backend")
        self.assertLess(selected.lineno, calls(self.init, 'AlphaFold3')[0].lineno)
        self.assertEqual(keywords(calls(self.runner, 'ModelRunner')[0])['layer_norm_backend'],
                         '_LAYER_NORM_BACKEND.value')

    def test_torch_layer_norm_retains_original_parameters_and_operation(self):
        forward = definition(self.layer, 'forward')
        branch = next(n for n in forward.body if isinstance(n, ast.If))
        self.assertEqual(ast.unparse(branch.test), "fastnn_config.layer_norm_implementation == 'torch'")
        operation = calls(branch.body[0], 'F.layer_norm')[0]
        self.assertEqual([ast.unparse(n) for n in operation.args],
                         ['input', 'self.normalized_shape', 'self.weight', 'self.bias', 'self.eps'])
        self.assertEqual(operation.keywords, [])
        init = definition(self.layer, '__init__')
        self.assertEqual(len(calls(init, 'nn.Parameter')), 2)
        self.assertFalse(calls(init, 'load_ipex_layer_norm'))
        self.assertFalse(calls(init, 'layer_norm_ipex'))

    def test_ipex_import_is_lazy_and_never_a_required_dependency(self):
        self.assertNotIn('intel_extension_for_pytorch', list(eager_imports(self.bridge)))
        load = definition(self.bridge, 'load_ipex_layer_norm')
        self.assertEqual(ast.literal_eval(calls(load, 'import_module')[0].args[0]),
                         'intel_extension_for_pytorch')
        self.assertIn('lru_cache(maxsize=1)', [ast.unparse(n) for n in load.decorator_list])
        self.assertNotIn('intel_extension_for_pytorch', (ROOT / 'pyproject.toml').read_text())
        for name in ('ipex.optimize', 'torch.compile', 'torch.jit.trace', 'torch.utils.cpp_extension.load'):
            self.assertFalse(calls(self.bridge, name))

    def test_ipex_preflight_rejects_non_cpu_before_loading_model(self):
        guard = next(n for n in self.init.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == "layer_norm_backend == 'ipex' and self._device.type != 'cpu'")
        self.assertTrue(any(isinstance(n, ast.Raise) for n in guard.body))
        load = calls(self.init, 'load_ipex_layer_norm')[0]
        self.assertLess(guard.lineno, load.lineno)
        self.assertLess(load.lineno, calls(self.init, 'AlphaFold3')[0].lineno)
        branch = next(n for n in self.init.body if isinstance(n, ast.If) and load in calls(n, 'load_ipex_layer_norm'))
        self.assertEqual(ast.unparse(branch.test), "layer_norm_backend == 'ipex'")

    def test_optional_loader_checks_platform_interface_and_import_failures(self):
        load = definition(self.bridge, 'load_ipex_layer_norm')
        platform_guard = load.body[1]
        self.assertIn("platform.system() != 'Linux'", ast.unparse(platform_guard))
        self.assertIn("('x86_64', 'amd64')", ast.unparse(platform_guard))
        self.assertLess(platform_guard.end_lineno, calls(load, 'import_module')[0].lineno)
        handler = next(n for n in ast.walk(load) if isinstance(n, ast.ExceptHandler))
        self.assertIn('SystemExit', ast.unparse(handler.type))
        self.assertIn('OSError', ast.unparse(handler.type))
        self.assertIn('raise RuntimeError', ast.unparse(handler))
        self.assertIn('not callable(operation)', ast.unparse(load))
        self.assertIn("'fast_layer_norm'", ast.unparse(load))

    def test_ipex_preserves_autocast_dtype_and_optional_affine_parameters(self):
        bridge = definition(self.bridge, 'layer_norm_ipex')
        operation = calls(bridge, 'operation')[0]
        self.assertEqual([ast.unparse(n) for n in operation.args],
                         ['input', 'normalized_shape', 'weight', 'bias', 'eps'])
        self.assertIn('if value is not None', ast.unparse(bridge))
        self.assertFalse(calls(bridge, 'torch.amp.autocast'))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr in ('to', 'detach', 'contiguous') for n in ast.walk(bridge)))
        self.assertFalse(calls(bridge, 'torch.set_default_dtype'))

    def test_ipex_direct_module_path_checks_training_gradients_device_and_dtype(self):
        forward = definition(self.layer, 'forward')
        branch = next(n for n in ast.walk(forward) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "fastnn_config.layer_norm_implementation == 'ipex'")
        self.assertIn('if self.training:', ast.unparse(branch))
        self.assertTrue(calls(branch, 'layer_norm_ipex'))
        bridge = definition(self.bridge, 'layer_norm_ipex')
        for contract in ("tensor.device.type != 'cpu'", 'tensor.layout != torch.strided',
                         '(torch.float32, torch.bfloat16)', 'torch.is_grad_enabled() and tensor.requires_grad'):
            self.assertIn(contract, ast.unparse(bridge))
        self.assertFalse(calls(bridge, 'F.layer_norm'))

    def test_backend_switch_is_part_of_input_cache_keys(self):
        cache = definition(definition(parse('xfold/nn/cache.py'), 'InputCache'), 'get')
        key = next(n for n in ast.walk(cache) if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == 'key')
        self.assertIn('config.layer_norm_implementation', ast.unparse(key.value))
        # Native Triangle continues to own its ATen normalizations, independently.
        native = (ROOT / 'af3_kernels/src/csrc/triangle_multiplication.cpp').read_text()
        self.assertEqual(native.count('at::layer_norm('), 2)
        self.assertNotIn('ipex', native)

    def test_thread_flags_default_to_unchanged_runtime_and_reach_runner(self):
        for name, flag_name in (('cpu_threads', '_CPU_THREADS'), ('cpu_interop_threads', '_CPU_INTEROP_THREADS')):
            flag = next(c for c in calls(self.runner, 'flags.DEFINE_integer')
                        if ast.literal_eval(c.args[0]) == name)
            self.assertIsNone(ast.literal_eval(flag.args[1]))
            self.assertEqual(keywords(calls(self.runner, 'ModelRunner')[0])[name], flag_name + '.value')
        defaults = dict(zip([a.arg for a in self.init.args.args[-len(self.init.args.defaults):]], self.init.args.defaults))
        self.assertIsNone(ast.literal_eval(defaults['cpu_threads']))
        self.assertIsNone(ast.literal_eval(defaults['cpu_interop_threads']))

    def test_thread_configuration_is_cpu_only_and_precedes_optional_loaders(self):
        configured = calls(self.init, 'configure_cpu_threads')[0]
        branch = next(n for n in self.init.body if isinstance(n, ast.If)
                      and configured in calls(n, 'configure_cpu_threads'))
        self.assertEqual(ast.unparse(branch.test), "self._device.type == 'cpu'")
        guard = next(n for n in self.init.body if isinstance(n, ast.If)
                     and 'cpu_threads is not None or cpu_interop_threads is not None' in ast.unparse(n.test))
        self.assertIn("self._device.type != 'cpu'", ast.unparse(guard.test))
        self.assertLess(configured.lineno, calls(self.init, 'load_ipex_layer_norm')[0].lineno)
        self.assertLess(configured.lineno, calls(self.init, 'load_cpp_glu')[0].lineno)
        self.assertLess(configured.lineno, calls(self.init, 'AlphaFold3')[0].lineno)

    def test_thread_validation_precedes_setters_and_preserves_none(self):
        configure = definition(self.runtime, 'configure_cpu_threads')
        validation = next(n for n in configure.body if isinstance(n, ast.For))
        self.assertIn('type(value) is not int', ast.unparse(validation))
        self.assertIn('0 < value <= 2147483647', ast.unparse(validation))
        for name in ('torch.set_num_threads', 'torch.set_num_interop_threads'):
            self.assertLess(validation.end_lineno, calls(configure, name)[0].lineno)
        guard = next(n for n in configure.body if isinstance(n, ast.If)
                     and ast.unparse(n.test) == 'num_threads is not None')
        self.assertEqual(len(calls(guard, 'torch.set_num_threads')), 1)
        self.assertFalse(any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript)
                             and ast.unparse(t.value) == 'os.environ' for t in n.targets)
                             for n in ast.walk(configure)))

    def test_interop_repeated_matching_settings_skip_setter_and_differing_requests_fail(self):
        configure = definition(self.runtime, 'configure_cpu_threads')
        guard = next(n for n in configure.body if isinstance(n, ast.If)
                     and 'num_interop_threads != torch.get_num_interop_threads()' in ast.unparse(n.test))
        self.assertIn('num_interop_threads is not None', ast.unparse(guard.test))
        self.assertEqual(len(calls(guard, 'torch.set_num_interop_threads')), 1)
        handler = next(n for n in ast.walk(guard) if isinstance(n, ast.ExceptHandler))
        self.assertIn('raise RuntimeError', ast.unparse(handler))
        self.assertIn('fresh process', ast.unparse(handler))
        self.assertLess(calls(configure, 'torch.set_num_interop_threads')[0].lineno,
                        calls(configure, 'torch.set_num_threads')[0].lineno)

    def test_launcher_and_configuration_have_valid_shell_syntax(self):
        for path in ('scripts/run_cpu.sh', 'scripts/cpu_env.sh.example'):
            with self.subTest(path=path):
                checked = subprocess.run(['bash', '-n', str(ROOT / path)], capture_output=True, text=True)
                self.assertEqual(checked.returncode, 0, checked.stderr)

    def test_launcher_keeps_defaults_and_exports_threads_before_python(self):
        for setting in ('memory_policy=${XFOLD_CPU_MEM_POLICY:-inherit}',
                        'affinity=${XFOLD_CPU_AFFINITY:-inherit}', 'threads=${XFOLD_CPU_THREADS:-}',
                        'interop_threads=${XFOLD_CPU_INTEROP_THREADS:-}'):
            self.assertIn(setting, self.launcher)
        for setting in ('export OMP_NUM_THREADS="$threads"', 'export MKL_NUM_THREADS="$threads"'):
            self.assertIn(setting, self.launcher)
            self.assertLess(self.launcher.index(setting), self.launcher.index('exec "${command[@]}"'))
        self.assertIn('positive_integer XFOLD_CPU_THREADS "$threads"', self.launcher)
        self.assertIn('--cpu_threads=$threads', self.launcher)
        self.assertIn('--cpu_interop_threads=$interop_threads', self.launcher)

    def test_launcher_has_explicit_cpu_numa_memory_and_affinity_policies(self):
        for setting in ('--physcpubind=$cpus', '--cpunodebind=$cpu_nodes', '--membind=$memory_nodes',
                        '--interleave=$memory_nodes', '--localalloc', 'export OMP_PLACES=cores',
                        'export OMP_PROC_BIND=close', 'export OMP_PROC_BIND=spread', 'export OMP_PROC_BIND=false'):
            self.assertIn(setting, self.launcher)
        self.assertIn('[[ -z "$cpus" || -z "$cpu_nodes" ]]', self.launcher)
        self.assertIn('command -v numactl', self.launcher)
        self.assertIn('KMP_AFFINITY/GOMP_CPU_AFFINITY', self.launcher)
        self.assertNotIn('--all', self.launcher)

    def test_launcher_uses_arrays_without_eval_and_preloads_only_explicit_paths(self):
        self.assertIn('command=("$python_binary"', self.launcher)
        self.assertIn('command=(numactl "${binding[@]}" -- "${command[@]}")', self.launcher)
        self.assertIn('exec "${command[@]}"', self.launcher)
        self.assertNotIn('eval ', self.launcher)
        self.assertNotIn('mpirun', self.launcher)
        self.assertNotIn('CONDA_PREFIX', self.launcher)
        self.assertIn('"${XFOLD_CPU_OMP_LIBRARY:-}" "${XFOLD_CPU_ALLOCATOR_LIBRARY:-}"', self.launcher)
        self.assertIn('"$library" == /* && -f "$library" && -r "$library"', self.launcher)
        self.assertIn('${LD_PRELOAD:+$LD_PRELOAD:}$library', self.launcher)

    def test_launcher_dry_run_exits_before_platform_checks_and_execution(self):
        printed = self.launcher.index("printf 'Command: '")
        dry = self.launcher.index('if [[ "$dry_run" == 1 ]]')
        self.assertLess(printed, dry)
        self.assertIn('exit 0', self.launcher[dry:self.launcher.index('uname -s')])
        self.assertLess(dry, self.launcher.index('exec "${command[@]}"'))
        self.assertIn('--device|-device|--cpu_threads|-cpu_threads', self.launcher)

    def test_launcher_rejects_both_abseil_spellings_and_nested_flagfile_overrides(self):
        # Check the actual case alternatives against =value and separate-value
        # spellings, without executing Bash, Abseil, Torch, or the launcher.
        import fnmatch
        case = self.launcher.split('case "$flag_name" in', 1)[1].split('esac', 1)[0]
        patterns = []
        for line in case.splitlines():
            line = line.strip()
            if line.startswith('--') and line.endswith(')'):
                patterns.extend(line[:-1].split('|'))
        self.assertIn('flag_name=${argument%%=*}', self.launcher)
        for flag in ('device', 'cpu_threads', 'cpu_interop_threads', 'flagfile'):
            for prefix in ('-', '--'):
                for suffix in ('', '=value'):
                    argument = prefix + flag + suffix
                    with self.subTest(argument=argument):
                        name = argument.split('=', 1)[0]
                        self.assertTrue(any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns))
        for argument in ('--cpu_precision=bf16', '--layer_norm_backend=ipex', '--output_dir=/a b'):
            self.assertFalse(any(fnmatch.fnmatchcase(argument.split('=', 1)[0], pattern) for pattern in patterns))
        self.assertIn('Flagfiles can override launcher-owned settings', case)


if __name__ == '__main__':
    unittest.main()
