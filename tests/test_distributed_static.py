"""Stage 9 source contracts only; no planner/model/collective execution."""

import ast
import subprocess
import unittest

from test_cpu_static import ROOT, calls, eager_imports, keywords, parse
from test_triangle_attention_static import definition


class DistributedStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = parse('xfold/distributed.py')
        cls.context = definition(cls.runtime, 'DistributedContext')
        cls.runner = parse('run_alphafold.py')
        cls.model = definition(parse('xfold/alphafold3.py'), 'AlphaFold3')
        cls.launcher = (ROOT / 'scripts/run_distributed.sh').read_text()

    def test_distributed_modes_are_explicit_and_default_to_off(self):
        flag = next(c for c in calls(self.runner, 'flags.DEFINE_enum')
                    if ast.literal_eval(c.args[0]) == 'distributed_mode')
        self.assertEqual(ast.literal_eval(flag.args[1]), 'off')
        self.assertEqual(ast.literal_eval(flag.args[2]), ['off', 'samples', 'tasks'])
        main = definition(self.runner, 'main')
        branch = main.body[0]
        self.assertEqual(ast.unparse(branch.test), "_DISTRIBUTED_MODE.value != 'off'")
        self.assertEqual(len(calls(branch, 'distributed_scope')), 1)
        self.assertTrue(any(isinstance(n, ast.Return) for n in ast.walk(branch)))

    def test_off_mode_has_no_rank_environment_or_distributed_loading(self):
        scope = definition(self.runtime, 'distributed_scope')
        off = next(n for n in scope.body if isinstance(n, ast.If))
        self.assertEqual(ast.unparse(off.test), "mode == 'off'")
        self.assertIn('yield None', ast.unparse(off))
        self.assertIsInstance(off.body[-1], ast.Return)
        self.assertLess(off.end_lineno, calls(scope, 'rank_environment')[0].lineno)
        self.assertLess(off.end_lineno, calls(scope, 'import_module')[0].lineno)
        self.assertNotIn('torch', list(eager_imports(self.runtime)))
        self.assertNotIn('torch.distributed', list(eager_imports(self.runtime)))

    def test_ccl_is_lazy_optional_and_gloo_is_the_default(self):
        scope = definition(self.runtime, 'distributed_scope')
        branch = next(n for n in ast.walk(scope) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "backend == 'ccl'")
        self.assertEqual(ast.literal_eval(calls(branch, 'import_module')[0].args[0]), 'oneccl_bindings_for_pytorch')
        self.assertIn("('x86_64', 'amd64')", ast.unparse(branch))
        self.assertIn('SystemExit', ast.unparse(branch))
        self.assertNotIn('oneccl', (ROOT / 'pyproject.toml').read_text())
        backend = next(c for c in calls(self.runner, 'flags.DEFINE_enum')
                       if ast.literal_eval(c.args[0]) == 'distributed_backend')
        self.assertEqual(ast.literal_eval(backend.args[1]), 'gloo')

    def test_environment_supports_torchrun_mpi_and_slurm_without_guessing_rank(self):
        environment = definition(self.runtime, 'rank_environment')
        families = next(n for n in environment.body if isinstance(n, ast.Assign)
                        and ast.unparse(n.targets[0]) == 'families')
        values = ast.literal_eval(families.value)
        self.assertEqual([item[0] for item in values],
                         ['RANK', 'PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'SLURM_PROCID'])
        self.assertIn('must be supplied together', ast.unparse(environment))
        self.assertIn('0 <= rank < world_size', ast.unparse(environment))
        self.assertIn('local_rank < 0', ast.unparse(environment))

    def test_partitions_have_remainders_idle_ranks_and_no_fixed_sample_limits(self):
        planner = definition(self.runtime, 'work_indices')
        code = ast.unparse(planner)
        self.assertIn('range(rank, total, world_size)', code)
        self.assertIn('divmod(total, world_size)', code)
        self.assertIn('rank * quotient + min(rank, remainder)', code)
        self.assertIn('int(rank < remainder)', code)
        self.assertNotIn('assert ', code)
        self.assertNotIn('total < world_size', code)
        for n in ast.walk(planner):
            if isinstance(n, ast.Compare):
                self.assertFalse(any(isinstance(value, ast.Constant) and value.value in (5, 200)
                                     for value in [n.left, *n.comparators]))

    def test_weighted_ranges_have_prefix_boundaries_and_allow_zero_capacity(self):
        planner = definition(self.runtime, 'work_indices')
        code = ast.unparse(planner)
        self.assertIn('len(weights) != world_size', code)
        self.assertIn('w < 0', code)
        self.assertIn('sum(weights) <= 0', code)
        self.assertIn('total * sum(weights[:rank]) // weight_total', code)
        self.assertIn('total * sum(weights[:rank + 1]) // weight_total', code)
        parse_weights = definition(self.runtime, 'parse_rank_weights')
        self.assertIn('re.fullmatch', ast.unparse(parse_weights))

    def test_group_init_and_cleanup_share_try_finally_without_exit_barrier(self):
        scope = definition(self.runtime, 'distributed_scope')
        guarded = next(n for n in scope.body if isinstance(n, ast.Try))
        self.assertEqual(len(calls(guarded, 'dist.init_process_group')), 1)
        cleanup = ast.Module(body=guarded.finalbody, type_ignores=[])
        self.assertEqual(len(calls(cleanup, 'dist.destroy_process_group')), 1)
        self.assertFalse(calls(scope, 'dist.barrier'))
        self.assertIn('dist.is_initialized()', ast.unparse(scope))
        group = calls(scope, 'dist.init_process_group')[0]
        self.assertEqual(keywords(group)['init_method'], "'env://'")
        self.assertEqual(keywords(group)['timeout'], 'timedelta(seconds=timeout_seconds)')

    def test_control_collectives_use_explicit_cpu_int64_and_broadcast_raw_bytes(self):
        for c in calls(self.context, 'self.torch.tensor'):
            self.assertEqual(keywords(c)['device'], "'cpu'")
            self.assertEqual(keywords(c)['dtype'], 'self.torch.int64')
        tensor = definition(self.context, 'broadcast_tensor')
        self.assertIn('view(self.torch.uint8)', ast.unparse(tensor))
        self.assertIn('self.chunk_bytes', ast.unparse(tensor))
        self.assertFalse(calls(self.context, 'self.dist.gather_object'))
        self.assertFalse(calls(self.context, 'self.dist.broadcast_object_list'))

    def test_variable_objects_have_size_budget_allocation_agreement_and_bounded_discard(self):
        obj = definition(self.context, 'broadcast_object')
        code = ast.unparse(obj)
        self.assertIn('len(payload) > self.max_payload_bytes', code)
        self.assertIn('length if retain else min(length, self.chunk_bytes)', code)
        self.assertIn('offset = begin if self.rank == src or retain else 0', code)
        status = calls(obj, 'self.dist.all_reduce')[0]
        chunks = calls(obj, 'self.torch.frombuffer')[0]
        self.assertLess(status.lineno, chunks.lineno)
        self.assertIn("'object decoding'", code)
        self.assertIn('pickle.dump', code)

    def test_tensor_byte_view_preparation_agrees_before_payload_broadcast(self):
        tensor = definition(self.context, 'broadcast_tensor')
        prepare = definition(tensor, 'prepare')
        self.assertEqual(len(calls(prepare, 'tensor.contiguous')), 1)
        self.assertIn('raw = output.reshape(-1).view(self.torch.uint8)', ast.unparse(prepare))
        self.assertEqual(ast.unparse(prepare.body[-1]), 'return (output, raw)')
        guard = calls(tensor, 'self.collective_call')[0]
        broadcast = calls(tensor, 'self.dist.broadcast')[0]
        self.assertEqual(ast.unparse(guard.args[0]), 'prepare')
        self.assertLess(prepare.end_lineno, guard.lineno)
        self.assertLess(guard.lineno, broadcast.lineno)

    def test_python_failures_agree_before_next_phase_and_include_failing_rank(self):
        guarded = definition(self.context, 'collective_call')
        self.assertIn('except Exception as error:', ast.unparse(guarded))
        self.assertIn('self.dist.ReduceOp.MIN', ast.unparse(guarded))
        self.assertIn('failed_rank != self.world_size', ast.unparse(guarded))
        self.assertIn('src=failed_rank', ast.unparse(guarded))
        self.assertIn('raise RuntimeError', ast.unparse(guarded))
        gathered = definition(self.context, 'gather_objects_to_root')
        self.assertIn('range(self.world_size)', ast.unparse(gathered))
        self.assertIn('root_only=True', ast.unparse(gathered))

    def test_samples_mode_runs_trunk_heads_and_structure_extraction_only_on_root(self):
        forward = definition(self.model, '_forward_distributed')
        self.assertIn('self._compute_embeddings(batch) if context.rank == 0 else None', ast.unparse(forward))
        self.assertIn('self._compute_heads(batch, embeddings, samples) if context.rank == 0 else None', ast.unparse(forward))
        self.assertEqual(len(calls(forward, 'context.broadcast_tensor_dict')), 1)
        self.assertEqual(len(calls(forward, 'self.reset_input_cache')), 2)
        predict = definition(self.runner, 'predict_structure')
        self.assertIn('list(featurise()) if distributed_context.rank == 0 else None', ast.unparse(predict))
        self.assertIn('extract() if distributed_context.rank == 0 else []', ast.unparse(predict))

    def test_serial_and_parallel_paths_share_trunk_and_output_head_implementations(self):
        for name in ('_forward_impl', '_forward_distributed'):
            forward = definition(self.model, name)
            self.assertEqual(len(calls(forward, 'self._compute_embeddings')), 1)
            self.assertEqual(len(calls(forward, 'self._compute_heads')), 1)
            self.assertFalse(calls(forward, 'self.evoformer'))
            self.assertFalse(calls(forward, 'self.confidence_head'))

    def test_legacy_sampling_remains_default_and_parallel_comparison_uses_indexed_streams(self):
        init = definition(self.model, '__init__')
        self.assertIn("self._sample_rng = 'legacy'", ast.unparse(init))
        configure = definition(self.model, 'configure_sampling')
        self.assertIn("'indexed' if parallel_samples else 'legacy'", ast.unparse(configure))
        self.assertIn("parallel_samples and resolved != 'indexed'", ast.unparse(configure))
        sample = definition(self.model, '_sample_diffusion')
        self.assertEqual(ast.unparse(sample.body[1].test), "self._sample_rng == 'indexed'")
        self.assertEqual(len(calls(sample, 'torch.randn')), 1)
        self.assertEqual(len(calls(sample, 'self._apply_denoising_step')), 1)

    def test_indexed_rng_is_rank_independent_and_isolates_only_cpu_state(self):
        seed = definition(self.runtime, 'sample_seed')
        self.assertEqual([item.arg for item in seed.args.args], ['base_seed', 'sample_index'])
        sample = definition(self.model, '_sample_diffusion_indexed')
        self.assertEqual(keywords(calls(sample, 'torch.random.fork_rng')[0]), {'devices': '[]'})
        self.assertEqual(keywords(calls(sample, 'torch.Generator')[0]), {'device': "'cpu'"})
        self.assertEqual(len(calls(sample, 'torch.set_rng_state')), 1)
        self.assertFalse(calls(sample, 'torch.manual_seed'))
        self.assertIn('range(self.diffusion_steps)', ast.unparse(sample))
        self.assertIn('context.indices(self.num_samples, source)', ast.unparse(sample))

    def test_sample_allocations_and_computation_have_collective_error_guards(self):
        sample = definition(self.model, '_sample_diffusion_indexed')
        code = ast.unparse(sample)
        for phase in ('diffusion allocation', 'local diffusion samples', 'sample masks'):
            self.assertIn(repr(phase), code)
        inference = definition(definition(self.runner, 'ModelRunner'), 'run_inference')
        for phase in ('feature conversion', 'model identifier', 'result conversion'):
            self.assertIn(repr(phase), ast.unparse(inference))

    def test_tasks_support_input_seed_granularity_and_idle_workers_in_every_wave(self):
        tasks = definition(self.runner, '_distributed_tasks')
        code = ast.unparse(tasks)
        self.assertIn("'input' if len(fold_inputs) > 1 else 'seed'", code)
        self.assertIn('range(len(fold_input.rng_seeds))', code)
        self.assertIn('dataclasses.replace(original, rng_seeds=(original.rng_seeds[seed_idx],))', code)
        self.assertIn('local_jobs[wave] if wave < len(local_jobs) else None', code)
        self.assertIn('for wave in range(waves):', code)
        self.assertEqual(len(calls(tasks, 'context.gather_objects_to_root')), 1)
        self.assertEqual(len(calls(tasks, 'predict_structure')), 1)
        self.assertNotIn('distributed_context=', ast.unparse(calls(tasks, 'predict_structure')[0]))

    def test_global_outputs_retain_original_seed_order_and_have_one_writer(self):
        tasks = definition(self.runner, '_distributed_tasks')
        code = ast.unparse(tasks)
        self.assertIn('receive_and_write(received) if context.rank == 0 else None', code)
        self.assertIn('[accumulated[i] for i in range(len(original.rng_seeds))]', code)
        self.assertIn('ordinal in accumulated or input_idx in completed', code)
        self.assertIn('pending or len(completed) != len(fold_inputs)', code)
        processing = next(n for n in self.runner.body if isinstance(n, ast.FunctionDef)
                          and n.name == 'process_fold_input' and len(n.body) > 1)
        self.assertIn('write_fold_input_json(fold_input, output_dir) if distributed_context.rank == 0 else None', ast.unparse(processing))
        self.assertIn('write() if distributed_context.rank == 0 else None', ast.unparse(processing))

    def test_distributed_inputs_and_semantics_agree_while_thread_counts_may_differ(self):
        main = definition(self.runner, '_main_distributed')
        code = ast.unparse(main)
        self.assertIn('_RUN_DATA_PIPELINE.value or not _RUN_INFERENCE.value', code)
        self.assertIn('context.collective_call(check_settings', code)
        settings = next(n for n in main.body if isinstance(n, ast.Assign)
                        and ast.unparse(n.targets[0]) == 'settings')
        self.assertIn('context.weights', ast.unparse(settings))
        self.assertIn('_SAMPLE_RNG.value', ast.unparse(settings))
        self.assertNotIn('_CPU_THREADS', ast.unparse(settings))
        self.assertIn('len(set(names)) != len(names)', code)
        self.assertIn('len(set(item.rng_seeds)) != len(item.rng_seeds)', code)

    def test_sampling_counts_and_cpu_constraints_precede_model_allocation(self):
        init = definition(definition(self.runner, 'ModelRunner'), '__init__')
        creation = calls(init, 'AlphaFold3')[0]
        for text in ('diffusion_steps < 1', '_NUM_DIFFUSION_SAMPLES.value < 1',
                     "distributed_context is not None and self._device.type != 'cpu'"):
            guard = next(n for n in init.body if isinstance(n, ast.If) and text in ast.unparse(n.test))
            self.assertLess(guard.end_lineno, creation.lineno)
        self.assertEqual(keywords(creation)['diffusion_steps'], 'diffusion_steps')
        self.assertEqual(len(calls(init, 'self._model.configure_sampling')), 1)

    def test_launcher_syntax_and_dry_run_without_execution(self):
        for name in ('scripts/run_distributed.sh', 'scripts/distributed_env.sh.example'):
            checked = subprocess.run(['bash', '-n', str(ROOT / name)], text=True, capture_output=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn('if [[ "$dry_run" == 1 ]]; then exit 0; fi', self.launcher)
        self.assertLess(self.launcher.index('if [[ "$dry_run" == 1 ]]'), self.launcher.index('uname -s'))
        self.assertNotIn('eval ', self.launcher)

    def test_launchers_allow_arbitrary_world_sizes_heterogeneous_hostfiles_and_external_ranks(self):
        for text in ('torchrun)', 'openmpi|intelmpi)', 'external)', '--max-restarts=0 --no-python',
                     '--hostfile "$hostfile"', '-f "$hostfile"', '--bind-to none', '-genv I_MPI_PIN 0',
                     'XFOLD_DIST_WORLD_SIZE:-$((nnodes * ppn))', 'export MASTER_ADDR MASTER_PORT'):
            self.assertIn(text, self.launcher)
        self.assertIn('[[ -n "${XFOLD_DIST_PPN:-}" ]]', self.launcher)
        self.assertIn('--distributed_*|-distributed_*|--flagfile|-flagfile', self.launcher)
        self.assertIn('"$repo_dir/scripts/run_cpu.sh"', self.launcher)
        self.assertNotIn('<= 5', self.launcher)


if __name__ == '__main__':
    unittest.main()
