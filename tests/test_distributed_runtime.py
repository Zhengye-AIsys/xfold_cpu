"""Real local Gloo transport and scheduling tests, without AF3 trained weights.

Linux is still required by the production launcher. The portable coordinator is
tested directly here on CPU; toy denoising/tasks test control flow, not AF3 accuracy.
"""

from dataclasses import dataclass
from datetime import timedelta
from contextlib import nullcontext, redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from xfold.alphafold3 import AlphaFold3
from xfold.distributed import (DistributedContext, distributed_scope,
                               parse_rank_weights, rank_environment, sample_seed, work_indices)


class ToyDenoiser(nn.Module):
    def forward(self, positions_noisy, **kwargs):
        return positions_noisy * 0.5


class SmallSampler(AlphaFold3):
    """Original sampling/denoising methods with a small deterministic test head."""
    def __init__(self, count, context=None):
        nn.Module.__init__(self)
        self.num_samples, self.diffusion_steps = count, 3
        self.gamma_0, self.gamma_min = 0.8, 1.0
        self.noise_scale, self.step_scale = 1.003, 1.5
        self.diffusion_head = ToyDenoiser()
        self.configure_sampling('indexed', context)
        self.eval()


@dataclass(frozen=True)
class ToyInput:
    name: str
    rng_seeds: tuple[int, ...]

    def sanitised_name(self):
        return self.name


def _transport_worker(rank, size, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=size,
                            timeout=timedelta(seconds=45))
    context = DistributedContext(torch, dist, 'samples', rank, size, rank, 'cyclic', (), 1024 * 1024)
    # Force multiple chunks even for very small fixtures.
    context.chunk_bytes = 23
    checks = []
    try:
        value = {'text': 'sample', 'features': np.arange(100, dtype=np.float32), 'empty': []}
        received = context.broadcast_object(value if rank == size - 1 else None, src=size - 1)
        assert received['text'] == value['text']
        np.testing.assert_array_equal(received['features'], value['features'])
        checks.append('object/numpy/chunked/non-root-source')

        gathered = context.gather_objects_to_root(None if rank == size - 1 else {'rank': rank, 'data': 'x' * (rank + 1) * 80})
        if rank == 0:
            assert gathered[-1] is None
            assert [item['rank'] for item in gathered[:-1]] == list(range(size - 1))
        else:
            assert gathered is None
        checks.append('unequal/idle/root-only-gather')

        for dtype in (torch.float32, torch.bfloat16, torch.int64, torch.bool):
            with torch.inference_mode():
                expected = torch.arange(35).reshape(5, 7).T.to(dtype)
                tensor = expected.clone() if rank == size - 1 else torch.zeros_like(expected)
                received = context.broadcast_tensor(tensor, src=size - 1)
                torch.testing.assert_close(received, expected, rtol=0, atol=0)
        with torch.inference_mode():
            embeddings = {'pair': torch.randn(3, 3, 8).bfloat16(), 'single': torch.randn(3, 8)} if rank == 0 else None
            received = context.broadcast_tensor_dict(embeddings)
            assert received['pair'].dtype == torch.bfloat16 and received['single'].shape == (3, 8)
            assert context.broadcast_tensor(torch.empty(0, 3)).numel() == 0
            assert context.broadcast_tensor(torch.tensor(7)).item() == 7
        checks.append('inference/strides/scalar/empty/FP32/BF16/embedding-transport')

        def fail():
            if rank == size - 1:
                raise ValueError('intentional compute error')
        try:
            context.collective_call(fail, 'injected computation')
        except RuntimeError as error:
            assert 'intentional compute error' in str(error) and f'rank {size - 1}' in str(error)
        else:
            raise AssertionError('Missing collective compute failure')
        checks.append('collective-compute-error')

        for value, expected in ((lambda: None, 'pickle'), ('x' * 1000, 'budget')):
            context.max_payload_bytes = 128
            try:
                context.broadcast_object(value if rank == 0 else None)
            except RuntimeError as error:
                assert ('local object' in str(error).lower()) if expected == 'pickle' else ('budget' in str(error))
            else:
                raise AssertionError('Missing serialization failure')
        context.max_payload_bytes = 1024 * 1024
        checks.append('serialization/budget-errors')

        allocation_guard = (patch('xfold.distributed.bytearray', side_effect=MemoryError('injected allocation'), create=True)
                            if rank == size - 1 else nullcontext())
        with allocation_guard:
            try:
                context.broadcast_object(value if rank == 0 else None)
            except RuntimeError as error:
                assert 'receive allocation failed' in str(error)
            else:
                raise AssertionError('Missing collective receive allocation failure')
        decode_guard = (patch('xfold.distributed.pickle.loads', side_effect=ValueError('injected decoding'))
                        if rank == size - 1 else nullcontext())
        with decode_guard:
            try:
                context.broadcast_object({'valid': True} if rank == 0 else None)
            except RuntimeError as error:
                assert 'object decoding' in str(error) and 'injected decoding' in str(error)
            else:
                raise AssertionError('Missing collective decoding failure')
        checks.append('receive-allocation/decoding-errors')

        # One rank fails byte-view preparation; peers must report it, not enter broadcast.
        unsupported = torch.ones(2, dtype=torch.complex64).conj() if rank == size - 1 else torch.ones(2)
        try:
            context.broadcast_tensor(unsupported)
        except RuntimeError as error:
            assert 'tensor preparation' in str(error)
        else:
            raise AssertionError('Missing byte-view failure')
        assert context.broadcast_object('still coordinated' if rank == 0 else None) == 'still coordinated'
        checks.append('tensor-preparation-error-and-subsequent-collective')

        batch = SimpleNamespace(predicted_structure_info=SimpleNamespace(atom_mask=torch.tensor([[1., 1.], [1., 0.]])))
        for schedule in ('cyclic', 'block', 'weighted'):
            context.schedule = schedule
            context.weights = tuple(0 if r == 0 else r for r in range(size)) if schedule == 'weighted' else ()
            for count in (1, 5, 7):
                with torch.inference_mode():
                    torch.manual_seed(127)
                    original_rng = torch.get_rng_state().clone()
                    baseline = SmallSampler(count)._sample_diffusion_indexed(batch, {})
                    torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
                    actual = SmallSampler(count, context)._sample_diffusion_indexed(batch, {})
                    torch.testing.assert_close(actual['atom_positions'], baseline['atom_positions'], rtol=0, atol=0)
                    torch.testing.assert_close(actual['mask'], baseline['mask'], rtol=0, atol=0)
                    torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
        checks.append('indexed-sampling-all-schedules/idle/remainders/RNG-state')

        # Run the real task coordinator with lightweight prediction/output fixtures.
        import run_alphafold as runner
        from absl import flags
        flags.FLAGS(['runtime-test'])
        flags.FLAGS['output_dir'].value = output
        inputs = [ToyInput('a', (11, 22, 33)), ToyInput('b', (44, 55))]
        local_jobs, written = [], {}
        def predict(item, unused_runner):
            local_jobs.extend((item.name, seed) for seed in item.rng_seeds)
            return [runner.ResultsForSeed(seed=seed, inference_results=[(item.name, seed, 0), (item.name, seed, 1)],
                                          full_fold_input=item) for seed in item.rng_seeds]
        def write(results, path, name):
            assert rank == 0 and name not in written
            assert [r.seed for r in results] == list(next(i.rng_seeds for i in inputs if i.name == name))
            written[name] = [(r.seed, r.inference_results) for r in results]
        def json_output(item, path):
            assert rank == 0
        context.mode = 'tasks'
        for schedule in ('cyclic', 'block', 'weighted'):
            context.schedule = schedule
            context.weights = tuple(0 if r == 0 else r for r in range(size)) if schedule == 'weighted' else ()
            for granularity in ('input', 'seed', 'auto'):
                flags.FLAGS['distributed_task_granularity'].value = granularity
                written.clear()
                local_jobs.clear()
                with patch.object(runner, 'predict_structure', predict), patch.object(runner, 'write_outputs', write), \
                     patch.object(runner, 'write_fold_input_json', json_output), redirect_stdout(io.StringIO()):
                    runner._distributed_tasks(inputs, object(), context)
                all_jobs = context.gather_objects_to_root(list(local_jobs))
                if rank == 0:
                    assert len(written) == 2
                    flattened = [job for worker_jobs in all_jobs for job in worker_jobs]
                    assert sorted(flattened) == sorted((i.name, seed) for i in inputs for seed in i.rng_seeds)
        def failed_output(*arguments):
            raise OSError('injected output failure')
        with patch.object(runner, 'predict_structure', predict), patch.object(runner, 'write_outputs', failed_output), \
             patch.object(runner, 'write_fold_input_json', json_output), redirect_stdout(io.StringIO()):
            try:
                runner._distributed_tasks(inputs, object(), context)
            except RuntimeError as error:
                assert 'task output and ranking' in str(error) and 'injected output failure' in str(error)
            else:
                raise AssertionError('Missing collective output failure')
        checks.append('real-task-coordinator/granularities/schedules/root-ranking')
        if rank == 0:
            Path(output, f'gloo-{size}-ranks.txt').write_text('\n'.join(checks) + '\n')
    finally:
        dist.destroy_process_group()


class DistributedRuntimeTests(unittest.TestCase):
    def test_partition_matrix_has_exact_disjoint_coverage(self):
        for size in (1, 2, 3, 5, 6, 8, 17):
            for total in (0, 1, 2, 5, 7, 13, 33):
                for schedule in ('cyclic', 'block', 'weighted'):
                    for weights in ((tuple(range(size)) if size > 1 else (1,)), tuple(1 for _ in range(size))):
                        active = weights if schedule == 'weighted' else ()
                        indices = [i for rank in range(size) for i in work_indices(total, rank, size, schedule, active)]
                        self.assertEqual(sorted(indices), list(range(total)))

    def test_invalid_partitions_and_weights_fail(self):
        for args in ((-1, 0, 1), (1, -1, 2), (1, 2, 2), (1, 0, 0), (True, 0, 1),
                     (1, 0, 2, 'weighted', (0, 0)), (1, 0, 2, 'weighted', (1,)),
                     (1, 0, 2, 'weighted', (1, -1)), (1, 0, 1, 'cyclic', (1,))):
            with self.subTest(args=args), self.assertRaises(ValueError):
                work_indices(*args)
        self.assertEqual(parse_rank_weights('0, 2, 3'), (0, 2, 3))
        for value in ('1,', '-1,2', '1.5,2', '1,,2'):
            with self.assertRaises(ValueError):
                parse_rank_weights(value)

    def test_rank_environments_are_normalized_and_partial_pairs_rejected(self):
        for environment, expected in (({}, (0, 1, 0)),
                                      ({'RANK': '3', 'WORLD_SIZE': '8', 'LOCAL_RANK': '1'}, (3, 8, 1)),
                                      ({'PMI_RANK': '2', 'PMI_SIZE': '7', 'MPI_LOCALRANKID': '1'}, (2, 7, 1)),
                                      ({'OMPI_COMM_WORLD_RANK': '6', 'OMPI_COMM_WORLD_SIZE': '8',
                                        'OMPI_COMM_WORLD_LOCAL_RANK': '2'}, (6, 8, 2)),
                                      ({'SLURM_PROCID': '5', 'SLURM_NTASKS': '9', 'SLURM_LOCALID': '0'}, (5, 9, 0))):
            with patch.dict(os.environ, environment, clear=True):
                self.assertEqual(rank_environment(), expected)
        for environment in ({'RANK': '1'}, {'PMI_SIZE': '5'}, {'RANK': '5', 'WORLD_SIZE': '2'}):
            with patch.dict(os.environ, environment, clear=True), self.assertRaises(ValueError):
                rank_environment()

    def test_serial_scope_does_not_load_group_or_validate_cluster_environment(self):
        with patch('xfold.distributed.import_module', side_effect=AssertionError('Unexpected import')):
            with distributed_scope('off') as context:
                self.assertIsNone(context)

    def test_sample_seed_is_stable_distinct_and_in_range(self):
        values = [sample_seed(42, i) for i in range(100)]
        self.assertEqual(len(set(values)), 100)
        self.assertTrue(all(0 <= value < 2 ** 63 for value in values))
        self.assertEqual(values, [sample_seed(42, i) for i in range(100)])
        self.assertNotEqual(values, [sample_seed(43, i) for i in range(100)])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'CPU Gloo unavailable')
    def test_real_gloo_transport_sampling_and_task_coordination(self):
        for size in (3, 6):
            with self.subTest(size=size), tempfile.TemporaryDirectory(prefix='xfold-gloo-') as directory:
                rendezvous = 'file://' + str(Path(directory, 'rendezvous'))
                mp.spawn(_transport_worker, args=(size, rendezvous, directory), nprocs=size, join=True)
                result = Path(directory, f'gloo-{size}-ranks.txt').read_text()
                self.assertIn('real-task-coordinator', result)
                self.assertIn('RNG-state', result)


if __name__ == '__main__':
    unittest.main()
