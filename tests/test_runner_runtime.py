"""Runner entry/precision/preflight checks without AF3 classes or trained weights."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from absl import flags
from absl.testing import flagsaver
import torch

import run_alphafold as runner


ROOT = Path(__file__).resolve().parents[1]


class RunnerRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not flags.FLAGS.is_parsed():
            flags.FLAGS(['runtime-test'])

    def test_help_runs_without_upstream_alphafold(self):
        result = subprocess.run([sys.executable, str(ROOT/'run_alphafold.py'), '--helpshort'],
                                capture_output=True, text=True, timeout=30)
        # Abseil prints short help with exit 1 by convention.
        self.assertIn(result.returncode, (0, 1))
        self.assertIn('--distributed_mode:', result.stdout)
        self.assertIn('--cpu_precision:', result.stdout)
        self.assertNotIn('Traceback', result.stderr)

    def test_serial_input_path_xor_is_validated_before_loading_dependencies(self):
        for json_path, input_dir in ((None, None), ('input.json', 'inputs')):
            with flagsaver.flagsaver(json_path=json_path, input_dir=input_dir, distributed_mode='off'):
                with self.assertRaisesRegex(ValueError, 'Exactly one'):
                    runner.main([])

    def test_missing_upstream_dependencies_have_an_actionable_message(self):
        if importlib.util.find_spec('alphafold3') is not None:
            self.skipTest('Upstream AF3 present; absence test not applicable')
        with self.assertRaisesRegex(RuntimeError, 'AlphaFold3 dependencies are unavailable'):
            runner._load_alphafold_dependencies()

    def test_dependency_loader_binds_all_names_and_reuses_successful_load(self):
        # Module fixtures exercise import plumbing, not the external AF3 implementation.
        names = ('alphafold3', 'alphafold3.common', 'alphafold3.common.folding_input',
                 'alphafold3.constants', 'alphafold3.constants.chemical_components', 'alphafold3.cpp',
                 'alphafold3.data', 'alphafold3.data.featurisation', 'alphafold3.data.pipeline',
                 'alphafold3.model', 'alphafold3.model.features', 'alphafold3.model.post_processing',
                 'alphafold3.model.components', 'alphafold3.model.components.base_model',
                 'alphafold3.model.components.utils', 'alphafold3.model.diffusion',
                 'alphafold3.model.diffusion.model')
        modules = {name: ModuleType(name) for name in names}
        for name, module in modules.items():
            module.__path__ = []
            if '.' in name:
                parent, child = name.rsplit('.', 1)
                setattr(modules[parent], child, module)
        diffuser = object()
        modules['alphafold3.model.diffusion.model'].Diffuser = diffuser
        expected = {name.rsplit('.', 1)[-1]: modules[name] for name in (
            'alphafold3.common.folding_input', 'alphafold3.constants.chemical_components',
            'alphafold3.data.featurisation', 'alphafold3.data.pipeline',
            'alphafold3.model.features', 'alphafold3.model.post_processing',
            'alphafold3.model.components.base_model', 'alphafold3.model.components.utils')}
        expected.update(alphafold3=modules['alphafold3'], Diffuser=diffuser)
        with patch.dict(runner.__dict__, {'_AF3_DEPENDENCIES_LOADED': False}), patch.dict(sys.modules, modules):
            runner._load_alphafold_dependencies()
            self.assertTrue(runner._AF3_DEPENDENCIES_LOADED)
            for name, value in expected.items():
                self.assertIs(getattr(runner, name), value)
            with patch('builtins.__import__', side_effect=AssertionError('Repeated dependency import')):
                runner._load_alphafold_dependencies()

    def test_missing_upstream_dependencies_fail_before_large_model_allocation(self):
        if importlib.util.find_spec('alphafold3') is not None:
            self.skipTest('Upstream AF3 present; absence test not applicable')
        with patch.object(runner, 'AlphaFold3', side_effect=AssertionError('Unexpected model allocation')):
            with self.assertRaisesRegex(RuntimeError, 'AlphaFold3 dependencies are unavailable'):
                runner.ModelRunner(Path('.'), torch.device('cpu'))

    def test_cpu_precision_context_overrides_outer_bf16_and_preserves_fp32_storage(self):
        model_runner = object.__new__(runner.ModelRunner)
        model_runner._device = torch.device('cpu')
        x, weight = torch.randn(3, 7), torch.randn(7, 5)
        with torch.inference_mode(), torch.amp.autocast('cpu', dtype=torch.bfloat16):
            model_runner._cpu_precision = 'fp32'
            with model_runner._inference_context():
                self.assertFalse(torch.is_autocast_enabled('cpu'))
                self.assertEqual((x @ weight).dtype, torch.float32)
            model_runner._cpu_precision = 'bf16'
            with model_runner._inference_context():
                self.assertTrue(torch.is_autocast_enabled('cpu'))
                self.assertEqual((x @ weight).dtype, torch.bfloat16)
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual(weight.dtype, torch.float32)

    def test_invalid_sampling_configuration_fails_before_model_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            for steps in (0, -1, True):
                with self.assertRaisesRegex(ValueError, 'Diffusion steps'):
                    runner.ModelRunner(Path(directory), torch.device('cpu'), diffusion_steps=steps)
            with flagsaver.flagsaver(num_diffusion_samples=0), self.assertRaisesRegex(ValueError, 'Diffusion samples'):
                runner.ModelRunner(Path(directory), torch.device('cpu'))
            with self.assertRaisesRegex(ValueError, 'Sample RNG'):
                runner.ModelRunner(Path(directory), torch.device('cpu'), sample_rng='invalid')

    def test_unavailable_cuda_fails_without_allocating_model(self):
        if torch.cuda.is_available():
            self.skipTest('CUDA available')
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            runner.ModelRunner(Path('.'), torch.device('cuda'))


if __name__ == '__main__':
    unittest.main()
