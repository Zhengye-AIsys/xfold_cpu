"""Execute launcher dry-runs and rejection paths; never start MPI or inference."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherRuntimeTests(unittest.TestCase):
    def run_script(self, name, arguments=(), settings=None):
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(('XFOLD_CPU_', 'XFOLD_DIST_'))
                       and key not in ('MASTER_ADDR', 'MASTER_PORT', 'KMP_AFFINITY', 'GOMP_CPU_AFFINITY', 'LD_PRELOAD')}
        environment.update(settings or {})
        return subprocess.run(['bash', str(ROOT/'scripts'/name), '--dry-run', *arguments],
                              env=environment, capture_output=True, text=True, timeout=10)

    def test_cpu_thread_numa_settings_and_quoted_paths(self):
        result = self.run_script('run_cpu.sh', ['--json_path=/tmp/input with spaces.json'],
                                 {'XFOLD_CPU_THREADS':'2', 'XFOLD_CPU_INTEROP_THREADS':'1',
                                  'XFOLD_CPU_CPUS':'0-3', 'XFOLD_CPU_MEM_POLICY':'bind',
                                  'XFOLD_CPU_MEM_NODES':'0', 'XFOLD_CPU_PYTHON':'/missing/python'})
        self.assertEqual(result.returncode, 0, result.stderr)
        for text in ('OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2', '--physcpubind=0-3',
                     '--membind=0', 'input\\ with\\ spaces.json', '--cpu_interop_threads=1'):
            self.assertIn(text, result.stdout)

    def test_torchrun_dry_run_has_no_five_rank_cap(self):
        result = self.run_script('run_distributed.sh', ['--num_diffusion_samples=7'],
                                 {'XFOLD_DIST_PPN':'8', 'XFOLD_CPU_PYTHON':'/missing/python'})
        self.assertEqual(result.returncode, 0, result.stderr)
        for text in ('--nproc-per-node=8', '--max-restarts=0', '--no-python',
                     '--distributed_mode=samples', '--num_diffusion_samples=7'):
            self.assertIn(text, result.stdout)

    def test_mpi_launchers_keep_heterogeneous_hostfiles_without_forced_ppn(self):
        with tempfile.TemporaryDirectory() as directory:
            hostfile = Path(directory, 'hosts')
            hostfile.write_text('node01 slots=4\nnode02 slots=2\n')
            for launcher, host_option in (('openmpi','--hostfile'), ('intelmpi','-f')):
                result = self.run_script('run_distributed.sh', settings={
                    'XFOLD_DIST_LAUNCHER':launcher, 'XFOLD_DIST_WORLD_SIZE':'6',
                    'XFOLD_DIST_HOSTFILE':str(hostfile), 'MASTER_ADDR':'node01', 'MASTER_PORT':'29501',
                    'XFOLD_DIST_MODE':'tasks', 'XFOLD_DIST_TASK_GRANULARITY':'seed'})
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('mpirun -n 6', result.stdout)
                self.assertIn(host_option, result.stdout)
                self.assertNotIn('--map-by', result.stdout)
                self.assertNotIn('-ppn', result.stdout)

    def test_external_launch_uses_configuration_file_and_no_nested_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            configuration = Path(directory, 'environment.sh')
            configuration.write_text('XFOLD_DIST_LAUNCHER=external\nMASTER_ADDR=node01\nMASTER_PORT=29501\n')
            result = self.run_script('run_distributed.sh', settings={'XFOLD_DIST_ENV_FILE':str(configuration)})
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('run_cpu.sh', result.stdout)
            self.assertNotIn('torch.distributed.run', result.stdout)
            self.assertNotIn('mpirun', result.stdout)

    def test_invalid_counts_overrides_and_missing_endpoints_fail_before_execution(self):
        cases = (
            ('run_cpu.sh', (), {'XFOLD_CPU_THREADS':'0'}),
            ('run_cpu.sh', ('-device=cpu',), {}),
            ('run_cpu.sh', ('--flagfile=/tmp/unused',), {}),
            ('run_distributed.sh', (), {'XFOLD_DIST_PPN':'0'}),
            ('run_distributed.sh', ('-distributed_mode=tasks',), {}),
            ('run_distributed.sh', (), {'XFOLD_DIST_NNODES':'2'}),
            ('run_distributed.sh', (), {'XFOLD_DIST_LAUNCHER':'external'}),
            ('run_distributed.sh', (), {'XFOLD_DIST_SCHEDULE':'weighted', 'XFOLD_DIST_RANK_WEIGHTS':'1,-1'}),
        )
        for name, arguments, settings in cases:
            with self.subTest(name=name, arguments=arguments, settings=settings):
                result = self.run_script(name, arguments, settings)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
