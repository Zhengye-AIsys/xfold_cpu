# Copyright 2024 xfold authors
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold 3 structure prediction script.

AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

To request access to the AlphaFold 3 model parameters, follow the process set
out at https://github.com/google-deepmind/alphafold3. You may only use these
if received directly from Google. Use is subject to terms of use available at
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
"""

from __future__ import annotations

import random
from collections.abc import Sequence
import csv
import dataclasses
import multiprocessing
import os
import pathlib
import shutil
import string
import textwrap
import time
from typing import overload

from absl import app
from absl import flags

import numpy as np
import torch
import torch.utils._pytree as pytree

from xfold.alphafold3 import AlphaFold3
from xfold.params import import_jax_weights_
from xfold.fastnn import config as fastnn_config


_HOME_DIR = pathlib.Path(os.environ.get('HOME'))
DEFAULT_MODEL_DIR = _HOME_DIR / 'models/model_103275239_1'
DEFAULT_DB_DIR = _HOME_DIR / 'public_databases'


# Input and output paths.
_JSON_PATH = flags.DEFINE_string(
    'json_path',
    None,
    'Path to the input JSON file.',
)
_INPUT_DIR = flags.DEFINE_string(
    'input_dir',
    None,
    'Path to the directory containing input JSON files.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'Path to a directory where the results will be saved.',
)

_MODEL_DIR = flags.DEFINE_string(
    'model_dir',
    DEFAULT_MODEL_DIR.as_posix(),
    'Path to the model to use for inference.',
)

# Control which stages to run.
_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    'run_data_pipeline',
    True,
    'Whether to run the data pipeline on the fold inputs.',
)
_RUN_INFERENCE = flags.DEFINE_bool(
    'run_inference',
    True,
    'Whether to run inference on the fold inputs.',
)

_DEVICE = flags.DEFINE_enum(
    'device',
    'auto',
    ['auto', 'cpu', 'cuda'],
    'Inference device. auto uses CUDA when available, otherwise CPU. '
    'CPU inference uses --cpu_precision and the GLU/Triangle/Attention backend flags.',
)

_CPU_PRECISION = flags.DEFINE_enum(
    'cpu_precision',
    'fp32',
    ['fp32', 'bf16'],
    'CPU inference precision. fp32 disables CPU autocast; bf16 enables '
    'CPU BF16 autocast while retaining FP32 weights and floating-point inputs. '
    'Select --device=cpu when comparing these modes.',
)

_LAYER_NORM_BACKEND = flags.DEFINE_enum(
    'layer_norm_backend', 'auto', ['auto', 'torch', 'ipex'],
    'LayerNorm backend. auto preserves PyTorch CPU / fastnn CUDA behavior; '
    'torch explicitly selects PyTorch; ipex selects optional Linux x86_64 '
    'CPU IPEX functional LayerNorm. Native Triangle internal norms remain ATen.',
)
_CPU_THREADS = flags.DEFINE_integer(
    'cpu_threads', None,
    'Positive PyTorch CPU intra-op thread count; unset preserves runtime defaults. '
    'Use scripts/run_cpu.sh to also set OpenMP/MKL before Python starts.',
)
_CPU_INTEROP_THREADS = flags.DEFINE_integer(
    'cpu_interop_threads', None,
    'Positive PyTorch CPU inter-op thread count; unset preserves runtime defaults. '
    'Set before model construction; changing an initialized pool needs a fresh process.',
)

_USE_FASTNN = flags.DEFINE_bool(
    'fastnn',
    True,
    'Whether to use Triton operators for CUDA inference. '
    'CPU native operators are selected separately with their backend flags.',
)

_GLU_BACKEND = flags.DEFINE_enum(
    'glu_backend',
    'auto',
    ['auto', 'torch', 'cpp'],
    'GLU backend. auto preserves PyTorch CPU / fastnn CUDA behavior; '
    'torch selects PyTorch GLU; cpp explicitly selects the optional CPU '
    'C++/LIBXSMM extension. LayerNorm and Attention are selected separately.',
)

_TRIANGLE_BACKEND = flags.DEFINE_enum(
    'triangle_backend', 'auto', ['auto', 'torch', 'cpp'],
    'Triangle Multiplication backend. auto/torch preserve PyTorch; '
    'cpp selects the optional CPU C++/LIBXSMM implementation.',
)
_ATTENTION_BACKEND = flags.DEFINE_enum(
    'attention_backend', 'auto', ['auto', 'torch', 'cpp'],
    'Grid/SelfAttention backend. auto preserves fastnn selection; torch uses '
    'PyTorch; cpp uses CPU native projections and dot-product attention. '
    'MSA/CrossAttention keep their existing implementation.',
)

_CACHE_MODE = flags.DEFINE_enum(
    'cache_mode', 'off', ['off', 'weights', 'inputs', 'all'],
    'Stage 7 inference caches: off preserves prior behavior; weights reuses '
    'BF16 native VNNI weights; inputs caches immutable per-example conditioning; '
    'all enables both. Cache choices are independent of operator backends.',
)
_INPUT_CACHE_MAX_MB = flags.DEFINE_integer(
    'input_cache_max_mb', 512,
    'Example-wide retained input-cache storage budget in MiB. 0 disables retention; '
    'oversized values are recomputed. Packed weight caches use separate storage.',
)

_DISTRIBUTED_MODE = flags.DEFINE_enum(
    'distributed_mode', 'off', ['off', 'samples', 'tasks'],
    'off preserves serial inference; samples parallelizes independent diffusion '
    'samples; tasks distributes inputs/seeds. Distributed modes require CPU '
    'inference with a completed input data pipeline.',
)
_DISTRIBUTED_BACKEND = flags.DEFINE_enum(
    'distributed_backend', 'gloo', ['gloo', 'ccl'],
    'CPU communication backend. ccl explicitly loads optional oneCCL bindings.',
)
_DISTRIBUTED_SCHEDULE = flags.DEFINE_enum(
    'distributed_schedule', 'cyclic', ['cyclic', 'block', 'weighted'],
    'Work assignment independent of physical nodes; no fixed rank/sample limit.',
)
_DISTRIBUTED_RANK_WEIGHTS = flags.DEFINE_string(
    'distributed_rank_weights', '',
    'Weighted schedule: one nonnegative integer per rank, with a positive total. '
    'Zero assigns no work to that rank; capacities refer to actual global ranks.',
)
_DISTRIBUTED_TASK_GRANULARITY = flags.DEFINE_enum(
    'distributed_task_granularity', 'auto', ['auto', 'input', 'seed'],
    'Task mode: auto distributes inputs when there are multiple inputs, otherwise '
    'seeds. input handles all seeds of one input; seed schedules input/seed pairs.',
)
_DISTRIBUTED_TIMEOUT = flags.DEFINE_integer(
    'distributed_timeout_seconds', 3600,
    'Process-group collective timeout; size this for the longest computation phase.',
)
_DISTRIBUTED_MAX_PAYLOAD = flags.DEFINE_integer(
    'distributed_max_payload_mb', 4096,
    'Maximum serialized object payload in MiB; tensor broadcasts use separate storage.',
)
_SAMPLE_RNG = flags.DEFINE_enum(
    'sample_rng', 'auto', ['auto', 'legacy', 'indexed'],
    'auto retains serial RNG except sample parallelism, which requires indexed '
    'CPU streams. Use indexed in serial mode for the matching parallel baseline.',
)
_DIFFUSION_STEPS = flags.DEFINE_integer(
    'diffusion_steps', 200, 'Positive diffusion step count; default preserves the original schedule.',
)

# Binary paths.
_JACKHMMER_BINARY_PATH = flags.DEFINE_string(
    'jackhmmer_binary_path',
    shutil.which('jackhmmer'),
    'Path to the Jackhmmer binary.',
)
_NHMMER_BINARY_PATH = flags.DEFINE_string(
    'nhmmer_binary_path',
    shutil.which('nhmmer'),
    'Path to the Nhmmer binary.',
)
_HMMALIGN_BINARY_PATH = flags.DEFINE_string(
    'hmmalign_binary_path',
    shutil.which('hmmalign'),
    'Path to the Hmmalign binary.',
)
_HMMSEARCH_BINARY_PATH = flags.DEFINE_string(
    'hmmsearch_binary_path',
    shutil.which('hmmsearch'),
    'Path to the Hmmsearch binary.',
)
_HMMBUILD_BINARY_PATH = flags.DEFINE_string(
    'hmmbuild_binary_path',
    shutil.which('hmmbuild'),
    'Path to the Hmmbuild binary.',
)

# Database paths.
_DB_DIR = flags.DEFINE_string(
    'db_dir',
    DEFAULT_DB_DIR.as_posix(),
    'Path to the directory containing the databases.',
)
_SMALL_BFD_DATABASE_PATH = flags.DEFINE_string(
    'small_bfd_database_path',
    '${DB_DIR}/bfd-first_non_consensus_sequences.fasta',
    'Small BFD database path, used for protein MSA search.',
)
_MGNIFY_DATABASE_PATH = flags.DEFINE_string(
    'mgnify_database_path',
    '${DB_DIR}/mgy_clusters_2022_05.fa',
    'Mgnify database path, used for protein MSA search.',
)
_UNIPROT_CLUSTER_ANNOT_DATABASE_PATH = flags.DEFINE_string(
    'uniprot_cluster_annot_database_path',
    '${DB_DIR}/uniprot_all_2021_04.fa',
    'UniProt database path, used for protein paired MSA search.',
)
_UNIREF90_DATABASE_PATH = flags.DEFINE_string(
    'uniref90_database_path',
    '${DB_DIR}/uniref90_2022_05.fa',
    'UniRef90 database path, used for MSA search. The MSA obtained by '
    'searching it is used to construct the profile for template search.',
)
_NTRNA_DATABASE_PATH = flags.DEFINE_string(
    'ntrna_database_path',
    '${DB_DIR}/nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta',
    'NT-RNA database path, used for RNA MSA search.',
)
_RFAM_DATABASE_PATH = flags.DEFINE_string(
    'rfam_database_path',
    '${DB_DIR}/rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta',
    'Rfam database path, used for RNA MSA search.',
)
_RNA_CENTRAL_DATABASE_PATH = flags.DEFINE_string(
    'rna_central_database_path',
    '${DB_DIR}/rnacentral_active_seq_id_90_cov_80_linclust.fasta',
    'RNAcentral database path, used for RNA MSA search.',
)
_PDB_DATABASE_PATH = flags.DEFINE_string(
    'pdb_database_path',
    '${DB_DIR}/pdb_2022_09_28_mmcif_files.tar',
    'PDB database directory with mmCIF files path, used for template search.',
)
_SEQRES_DATABASE_PATH = flags.DEFINE_string(
    'seqres_database_path',
    '${DB_DIR}/pdb_seqres_2022_09_28.fasta',
    'PDB sequence database path, used for template search.',
)

# Number of CPUs to use for MSA tools.
_JACKHMMER_N_CPU = flags.DEFINE_integer(
    'jackhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Jackhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)
_NHMMER_N_CPU = flags.DEFINE_integer(
    'nhmmer_n_cpu',
    min(multiprocessing.cpu_count(), 8),
    'Number of CPUs to use for Nhmmer. Default to min(cpu_count, 8). Going'
    ' beyond 8 CPUs provides very little additional speedup.',
)

_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer(
    'num_diffusion_samples',
    5,
    'Number of diffusion samples to generate.',
)


_AF3_DEPENDENCIES_LOADED = False


def _load_alphafold_dependencies():
    """Keep --help and local CPU tests independent of the upstream AF3 install."""
    global folding_input, chemical_components, alphafold3, featurisation, pipeline
    global features, post_processing, base_model, utils, Diffuser
    global _AF3_DEPENDENCIES_LOADED
    if _AF3_DEPENDENCIES_LOADED:
        return
    try:
        from alphafold3.common import folding_input
        from alphafold3.constants import chemical_components
        import alphafold3.cpp
        from alphafold3.data import featurisation, pipeline
        from alphafold3.model import features, post_processing
        from alphafold3.model.components import base_model, utils
        from alphafold3.model.diffusion.model import Diffuser
    except (ImportError, OSError) as error:
        raise RuntimeError(
            'AlphaFold3 dependencies are unavailable or incompatible. Prepare the '
            'upstream AlphaFold3 environment described in README.md before prediction. '
            f'Original import error: {error}'
        ) from error
    _AF3_DEPENDENCIES_LOADED = True


class ModelRunner:
    """Helper class to run structure prediction stages."""

    def __init__(
        self,
        model_dir: pathlib.Path,
        device: torch.device,
        cpu_precision: str = 'fp32',
        glu_backend: str = 'auto',
        triangle_backend: str = 'auto',
        attention_backend: str = 'auto',
        cache_mode: str = 'off',
        input_cache_max_mb: int = 512,
        layer_norm_backend: str = 'auto',
        cpu_threads: int | None = None,
        cpu_interop_threads: int | None = None,
        distributed_context=None,
        sample_rng: str = 'auto',
        diffusion_steps: int = 200,
    ):
        self._model_dir = model_dir
        self._device = torch.device(device)
        self._distributed_context = distributed_context
        if distributed_context is not None and self._device.type != 'cpu':
            raise ValueError('Distributed inference requires --device=cpu.')
        if sample_rng not in ('auto', 'legacy', 'indexed'):
            raise ValueError('Sample RNG must be auto, legacy, or indexed.')
        if distributed_context is not None and distributed_context.mode == 'samples' and sample_rng == 'legacy':
            raise ValueError('Sample parallelism requires --sample_rng=indexed or auto.')
        if sample_rng == 'indexed' and self._device.type != 'cpu':
            raise ValueError('Indexed sampling requires --device=cpu.')
        if type(diffusion_steps) is not int or diffusion_steps < 1:
            raise ValueError('Diffusion steps must be a positive integer.')
        if type(_NUM_DIFFUSION_SAMPLES.value) is not int or _NUM_DIFFUSION_SAMPLES.value < 1:
            raise ValueError('Diffusion samples must be a positive integer.')
        if self._device.type not in ('cpu', 'cuda'):
            raise ValueError('Inference supports CPU and CUDA devices only.')
        if self._device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA was requested but is unavailable. Use --device=cpu '
                'or --device=auto.'
            )
        if cpu_precision not in ('fp32', 'bf16'):
            raise ValueError('CPU precision must be fp32 or bf16.')
        if self._device.type != 'cpu' and cpu_precision != 'fp32':
            raise ValueError(
                '--cpu_precision=bf16 requires a CPU device. '
                'Use --device=cpu to select the CPU explicitly.'
            )
        self._cpu_precision = cpu_precision
        if cache_mode not in ('off', 'weights', 'inputs', 'all'):
            raise ValueError('Cache mode must be off, weights, inputs, or all.')
        if not isinstance(input_cache_max_mb, int) or input_cache_max_mb < 0:
            raise ValueError('Input cache budget must be a nonnegative integer MiB value.')

        if layer_norm_backend not in ('auto', 'torch', 'ipex'):
            raise ValueError('LayerNorm backend must be auto, torch, or ipex.')
        if self._device.type != 'cpu' and (cpu_threads is not None or cpu_interop_threads is not None):
            raise ValueError('CPU thread flags require --device=cpu.')
        if layer_norm_backend == 'ipex' and self._device.type != 'cpu':
            raise ValueError('--layer_norm_backend=ipex requires --device=cpu.')
        if self._device.type == 'cpu':
            from xfold.cpu_runtime import configure_cpu_threads
            configure_cpu_threads(cpu_threads, cpu_interop_threads)
        if layer_norm_backend == 'ipex':
            from xfold.fastnn._ipex_layer_norm import load_ipex_layer_norm
            load_ipex_layer_norm()

        if glu_backend not in ('auto', 'torch', 'cpp'):
            raise ValueError('GLU backend must be auto, torch, or cpp.')
        if glu_backend == 'cpp':
            if self._device.type != 'cpu':
                raise ValueError('--glu_backend=cpp requires --device=cpu.')
            from xfold.fastnn.gated_linear_unit import load_cpp_glu
            load_cpp_glu()

        for name, backend, operators in (
            ('triangle', triangle_backend, ('triangle_multiplication',)),
            ('attention', attention_backend, ('linear', 'dot_product_attention')),
        ):
            if backend not in ('auto', 'torch', 'cpp'):
                raise ValueError(f'{name} backend must be auto, torch, or cpp.')
            if backend == 'cpp':
                if self._device.type != 'cpu':
                    raise ValueError(f'--{name}_backend=cpp requires --device=cpu.')
                from xfold.fastnn.native import load_native_operator
                for operator in operators:
                    load_native_operator(operator)

        if cache_mode in ('weights', 'all') and 'cpp' in (glu_backend, triangle_backend, attention_backend):
            from xfold.fastnn.native import load_native_operator
            extension = load_native_operator('gated_linear_unit')
            if getattr(extension, 'packed_weight_api', 0) != 1:
                raise RuntimeError('Rebuild af3_kernels for the stage 7 packed-weight API.')

        fastnn_config.input_cache_enabled = cache_mode in ('inputs', 'all')
        fastnn_config.packed_weight_cache_enabled = cache_mode in ('weights', 'all')
        fastnn_config.input_cache_max_bytes = input_cache_max_mb * 1024 * 1024
        print(f'Cache mode: {cache_mode}; input cache budget: {input_cache_max_mb} MiB')

        implementation = (
            'triton' if self._device.type == 'cuda' and _USE_FASTNN.value
            else 'torch'
        )
        fastnn_config.layer_norm_implementation = (
            implementation if layer_norm_backend == 'auto' else layer_norm_backend)
        print(f'LayerNorm backend: {fastnn_config.layer_norm_implementation}')
        fastnn_config.dot_product_attention_implementation = (
            implementation if attention_backend == 'auto' else attention_backend)
        fastnn_config.gated_linear_unit_implementation = (
            implementation if glu_backend == 'auto' else glu_backend)
        print(f'GLU backend: {fastnn_config.gated_linear_unit_implementation}')
        fastnn_config.triangle_multiplication_implementation = (
            'torch' if triangle_backend == 'auto' else triangle_backend)
        print(f'Triangle backend: {fastnn_config.triangle_multiplication_implementation}; '
              f'Attention backend: {fastnn_config.dot_product_attention_implementation}')

        _load_alphafold_dependencies()
        self._model = AlphaFold3(num_samples=_NUM_DIFFUSION_SAMPLES.value, diffusion_steps=diffusion_steps)
        self._model.configure_sampling(sample_rng, distributed_context)
        self._model.eval()
        print('loading the model parameters...')
        import_jax_weights_(self._model, model_dir)

        self._model = self._model.to(
            device=self._device, dtype=torch.float32)
        if self._device.type == 'cpu':
            print(
                f'CPU precision: {self._cpu_precision}; '
                'FP32 weights and floating-point inputs; '
                f'BF16 autocast: {self._cpu_precision == "bf16"}'
            )

    def _inference_context(self):
        """Choose computation precision without casting stored weights/features."""
        if self._device.type == 'cpu':
            return torch.amp.autocast(
                device_type='cpu', dtype=torch.bfloat16,
                enabled=self._cpu_precision == 'bf16',
            )
        return torch.amp.autocast(
            device_type='cuda', dtype=torch.bfloat16)

    def synchronize(self) -> None:
        """Wait for inference work only when this runner uses CUDA."""
        if self._device.type == 'cuda':
            torch.cuda.synchronize(self._device)

    @torch.inference_mode()
    def run_inference(
        self, featurised_example: features.BatchDict
    ) -> base_model.ModelResult | None:
        """Computes a forward pass of the model on a featurised example."""
        def prepare_features(featurised_example=featurised_example):
            featurised_example = pytree.tree_map(
                torch.from_numpy, utils.remove_invalidly_typed_feats(
                    featurised_example)
            )
            featurised_example = pytree.tree_map_only(
                torch.Tensor,
                lambda x: x.to(device=self._device, dtype=torch.float32)
                if self._device.type == 'cpu' and x.is_floating_point()
                else x.to(device=self._device),
                featurised_example,
            )
            featurised_example['deletion_mean'] = featurised_example['deletion_mean'].to(
                dtype=torch.float32)
            return featurised_example
        context = self._distributed_context
        parallel = context is not None and context.mode == 'samples'
        featurised_example = (context.collective_call(prepare_features, 'feature conversion')
                              if parallel else prepare_features())

        with self._inference_context():
            result = self._model(featurised_example)
            def identify():
                if result is not None:
                    result['__identifier__'] = self._model.__identifier__.cpu().numpy()
            if parallel:
                context.collective_call(identify, 'model identifier')
            else:
                identify()

        def convert(result=result):
            if result is None:
                return None
            result = pytree.tree_map_only(
                torch.Tensor,
                lambda x: x.to(
                    dtype=torch.float32) if x.dtype == torch.bfloat16 else x,
                result,
            )
            result = pytree.tree_map_only(
                torch.Tensor, lambda x: x.cpu().detach().numpy(), result)
            result['__identifier__'] = result['__identifier__'].tobytes()
            return result
        return context.collective_call(convert, 'result conversion') if parallel else convert()

    def extract_structures(
        self,
        batch: features.BatchDict,
        result: base_model.ModelResult,
        target_name: str,
    ) -> list[base_model.InferenceResult]:
        """Generates structures from model outputs."""
        return list(
            Diffuser.get_inference_result(
                batch=batch, result=result, target_name=target_name
            )
        )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ResultsForSeed:
    """Stores the inference results (diffusion samples) for a single seed.

    Attributes:
      seed: The seed used to generate the samples.
      inference_results: The inference results, one per sample.
      full_fold_input: The fold input that must also include the results of
        running the data pipeline - MSA and templates.
    """

    seed: int
    inference_results: Sequence[base_model.InferenceResult]
    full_fold_input: folding_input.Input


def predict_structure(
    fold_input: folding_input.Input,
    model_runner: ModelRunner,
    buckets: Sequence[int] | None = None,
    distributed_context=None,
) -> Sequence[ResultsForSeed]:
    """Runs the full inference pipeline to predict structures for each seed."""

    _load_alphafold_dependencies()
    print(f'Featurising data for seeds {fold_input.rng_seeds}...')
    featurisation_start_time = time.time()
    def featurise():
        ccd = chemical_components.cached_ccd(user_ccd=fold_input.user_ccd)
        return featurisation.featurise_input(
            fold_input=fold_input, buckets=buckets, ccd=ccd, verbose=True)
    if distributed_context is None:
        featurised_examples = featurise()
    else:
        featurised_examples = distributed_context.collective_call(
            lambda: list(featurise()) if distributed_context.rank == 0 else None, 'rank 0 featurisation')
        featurised_examples = distributed_context.broadcast_object(featurised_examples)
    print(
        f'Featurising data for seeds {fold_input.rng_seeds} took '
        f' {time.time() - featurisation_start_time:.2f} seconds.'
    )
    all_inference_start_time = time.time()
    all_inference_results = []
    for seed, example in zip(fold_input.rng_seeds, featurised_examples):
        print(f'Running model inference for seed {seed}...')
        model_runner.synchronize()
        inference_start_time = time.time()

        # set the random seed for the model.
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        result = model_runner.run_inference(example)
        model_runner.synchronize()
        print(
            f'Running model inference for seed {seed} took '
            f' {time.time() - inference_start_time:.2f} seconds.'
        )
        print(
            f'Extracting output structures (one per sample) for seed {seed}...')
        extract_structures = time.time()
        def extract():
            if result is None:
                raise RuntimeError('The output rank did not receive a model result.')
            return model_runner.extract_structures(batch=example, result=result, target_name=fold_input.name)
        inference_results = (extract() if distributed_context is None else
                             distributed_context.collective_call(
                                 lambda: extract() if distributed_context.rank == 0 else [], 'structure extraction'))
        print(
            f'Extracting output structures (one per sample) for seed {seed} took '
            f' {time.time() - extract_structures:.2f} seconds.'
        )
        all_inference_results.append(
            ResultsForSeed(
                seed=seed,
                inference_results=inference_results,
                full_fold_input=fold_input,
            )
        )
        print(
            'Running model inference and extracting output structures for seed'
            f' {seed} took  {time.time() - inference_start_time:.2f} seconds.'
        )
    print(
        'Running model inference and extracting output structures for seeds'
        f' {fold_input.rng_seeds} took '
        f' {time.time() - all_inference_start_time:.2f} seconds.'
    )
    return all_inference_results


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> None:
    """Writes the input JSON to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(
            output_dir, f'{fold_input.sanitised_name()}_data.json'), 'wt'
    ) as f:
        f.write(fold_input.to_json())


def write_outputs(
    all_inference_results: Sequence[ResultsForSeed],
    output_dir: os.PathLike[str] | str,
    job_name: str,
) -> None:
    """Writes outputs to the specified output directory."""
    ranking_scores = []
    max_ranking_score = None
    max_ranking_result = None

    output_terms = (
        pathlib.Path(alphafold3.cpp.__file__).parent / 'OUTPUT_TERMS_OF_USE.md'
    ).read_text()

    os.makedirs(output_dir, exist_ok=True)
    for results_for_seed in all_inference_results:
        seed = results_for_seed.seed
        for sample_idx, result in enumerate(results_for_seed.inference_results):
            sample_dir = os.path.join(
                output_dir, f'seed-{seed}_sample-{sample_idx}')
            os.makedirs(sample_dir, exist_ok=True)
            post_processing.write_output(
                inference_result=result, output_dir=sample_dir
            )
            ranking_score = float(result.metadata['ranking_score'])
            ranking_scores.append((seed, sample_idx, ranking_score))
            if max_ranking_score is None or ranking_score > max_ranking_score:
                max_ranking_score = ranking_score
                max_ranking_result = result

    if max_ranking_result is not None:  # True iff ranking_scores non-empty.
        post_processing.write_output(
            inference_result=max_ranking_result,
            output_dir=output_dir,
            # The output terms of use are the same for all seeds/samples.
            terms_of_use=output_terms,
            name=job_name,
        )
        # Save csv of ranking scores with seeds and sample indices, to allow easier
        # comparison of ranking scores across different runs.
        with open(os.path.join(output_dir, 'ranking_scores.csv'), 'wt') as f:
            writer = csv.writer(f)
            writer.writerow(['seed', 'sample', 'ranking_score'])
            writer.writerows(ranking_scores)


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    distributed_context=None,
) -> folding_input.Input:
    ...


@overload
def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    distributed_context=None,
) -> Sequence[ResultsForSeed]:
    ...


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    model_runner: ModelRunner | None,
    output_dir: os.PathLike[str] | str,
    buckets: Sequence[int] | None = None,
    distributed_context=None,
) -> folding_input.Input | Sequence[ResultsForSeed]:
    """Runs data pipeline and/or inference on a single fold input.

    Args:
      fold_input: Fold input to process.
      data_pipeline_config: Data pipeline config to use. If None, skip the data
        pipeline.
      model_runner: Model runner to use. If None, skip inference.
      output_dir: Output directory to write to.
      buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
        of the model. If None, calculate the appropriate bucket size from the
        number of tokens. If not None, must be a sequence of at least one integer,
        in strictly increasing order. Will raise an error if the number of tokens
        is more than the largest bucket size.

    Returns:
      The processed fold input, or the inference results for each seed.

    Raises:
      ValueError: If the fold input has no chains.
    """
    print(f'Processing fold input {fold_input.name}')

    if not fold_input.chains:
        raise ValueError('Fold input has no chains.')

    _load_alphafold_dependencies()

    if data_pipeline_config is None:
        print('Skipping data pipeline...')
    else:
        print('Running data pipeline...')
        fold_input = pipeline.DataPipeline(
            data_pipeline_config).process(fold_input)

    print(f'Output directory: {output_dir}')
    print(f'Writing model input JSON to {output_dir}')
    if distributed_context is None:
        write_fold_input_json(fold_input, output_dir)
    else:
        distributed_context.collective_call(
            lambda: write_fold_input_json(fold_input, output_dir) if distributed_context.rank == 0 else None,
            'input JSON output')
    if model_runner is None:
        print('Skipping inference...')
        output = fold_input
    else:
        print(
            f'Predicting 3D structure for {fold_input.name} for seed(s)'
            f' {fold_input.rng_seeds}...'
        )
        all_inference_results = predict_structure(
            fold_input=fold_input,
            model_runner=model_runner,
            buckets=buckets,
            distributed_context=distributed_context,
        )
        print(
            f'Writing outputs for {fold_input.name} for seed(s)'
            f' {fold_input.rng_seeds}...'
        )
        def write():
            write_outputs(all_inference_results=all_inference_results, output_dir=output_dir,
                          job_name=fold_input.sanitised_name())
        if distributed_context is None:
            write()
        else:
            distributed_context.collective_call(
                lambda: write() if distributed_context.rank == 0 else None, 'prediction output')
        output = all_inference_results

    print(f'Done processing fold input {fold_input.name}.')
    return output


def _distributed_tasks(fold_inputs, model_runner, context):
    granularity = _DISTRIBUTED_TASK_GRANULARITY.value
    if granularity == 'auto':
        granularity = 'input' if len(fold_inputs) > 1 else 'seed'
    jobs = ([(input_idx, None) for input_idx in range(len(fold_inputs))] if granularity == 'input' else
            [(input_idx, seed_idx) for input_idx, fold_input in enumerate(fold_inputs)
             for seed_idx in range(len(fold_input.rng_seeds))])
    assignments = [context.indices(len(jobs), rank) for rank in range(context.world_size)]
    waves = max(len(indices) for indices in assignments)
    local_jobs = assignments[context.rank]
    print(f'Task plan: granularity={granularity}, jobs={len(jobs)}, '
          f'local_jobs={len(local_jobs)}, waves={waves}')
    pending, completed = {}, set()

    def compute(job_id):
        if job_id is None:
            return None
        input_idx, seed_idx = jobs[job_id]
        original = fold_inputs[input_idx]
        selected = original if seed_idx is None else dataclasses.replace(
            original, rng_seeds=(original.rng_seeds[seed_idx],))
        results = predict_structure(selected, model_runner)
        # Root already has the full input; transport structures only, preserving
        # original input/seed/sample order when writing global rankings.
        return job_id, [(item.seed, list(item.inference_results)) for item in results]

    def receive_and_write(received):
        for item in received:
            if item is None:
                continue
            job_id, results = item
            input_idx, seed_idx = jobs[job_id]
            original = fold_inputs[input_idx]
            expected = range(len(original.rng_seeds)) if seed_idx is None else (seed_idx,)
            if len(results) != len(expected):
                raise RuntimeError('Task returned an incorrect seed count.')
            accumulated = pending.setdefault(input_idx, {})
            for ordinal, (seed, inference_results) in zip(expected, results):
                if seed != original.rng_seeds[ordinal] or ordinal in accumulated or input_idx in completed:
                    raise RuntimeError('Task returned a duplicate or mismatched seed.')
                accumulated[ordinal] = ResultsForSeed(
                    seed=seed, inference_results=inference_results, full_fold_input=original)
            if len(accumulated) == len(original.rng_seeds):
                output_dir = os.path.join(_OUTPUT_DIR.value, original.sanitised_name())
                write_fold_input_json(original, output_dir)
                write_outputs([accumulated[i] for i in range(len(original.rng_seeds))],
                              output_dir, original.sanitised_name())
                del pending[input_idx]
                completed.add(input_idx)

    for wave in range(waves):
        job_id = local_jobs[wave] if wave < len(local_jobs) else None
        result = context.collective_call(lambda: compute(job_id), f'task wave {wave}')
        received = context.gather_objects_to_root(result)
        context.collective_call(
            lambda: receive_and_write(received) if context.rank == 0 else None, 'task output and ranking')

    def finish():
        if pending or len(completed) != len(fold_inputs):
            raise RuntimeError('Distributed tasks did not produce every input/seed result.')
    context.collective_call(lambda: finish() if context.rank == 0 else None, 'task completion')


def _print_model_notice():
    notice = textwrap.wrap(
        'Running AlphaFold 3. Please note that standard AlphaFold 3 model'
        ' parameters are only available under terms of use provided at'
        ' https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.'
        ' If you do not agree to these terms and are using AlphaFold 3 derived'
        ' model parameters, cancel execution of AlphaFold 3 inference with'
        ' CTRL-C, and do not use the model parameters.',
        break_long_words=False,
        break_on_hyphens=False,
        width=80,
    )
    print('\n'.join(notice))


def _main_distributed(context):
    def validate():
        if (_JSON_PATH.value is None) == (_INPUT_DIR.value is None):
            raise ValueError('Exactly one of --json_path or --input_dir must be specified.')
        if _RUN_DATA_PIPELINE.value or not _RUN_INFERENCE.value:
            raise ValueError('Distributed inference requires --run_data_pipeline=False and --run_inference=True.')
        if _DEVICE.value not in ('auto', 'cpu'):
            raise ValueError('Distributed inference requires a CPU device.')
    context.collective_call(validate, 'distributed flags')
    context.collective_call(_load_alphafold_dependencies, 'AlphaFold3 dependencies')

    # Rank-dependent thread/affinity settings may differ; model semantics and
    # collective ownership settings must agree before loading models or data.
    settings = (
        context.mode, context.schedule, context.weights, context.max_payload_bytes,
        _CPU_PRECISION.value, _GLU_BACKEND.value, _TRIANGLE_BACKEND.value,
        _ATTENTION_BACKEND.value, _LAYER_NORM_BACKEND.value, _CACHE_MODE.value,
        _INPUT_CACHE_MAX_MB.value, _NUM_DIFFUSION_SAMPLES.value, _DIFFUSION_STEPS.value,
        _SAMPLE_RNG.value, _DISTRIBUTED_TASK_GRANULARITY.value, _MODEL_DIR.value,
        _OUTPUT_DIR.value, _JSON_PATH.value, _INPUT_DIR.value,
    )
    root_settings = context.broadcast_object(settings if context.rank == 0 else None)
    def check_settings():
        if settings != root_settings:
            raise ValueError('Ranks must use matching input/model/operator/sampling/collective settings.')
    context.collective_call(check_settings, 'configuration agreement')

    def load_inputs():
        loaded = (folding_input.load_fold_inputs_from_dir(pathlib.Path(_INPUT_DIR.value))
                  if _INPUT_DIR.value is not None else
                  folding_input.load_fold_inputs_from_path(pathlib.Path(_JSON_PATH.value)))
        inputs = list(loaded)
        if not inputs:
            raise ValueError('No fold inputs were found.')
        names = [item.sanitised_name() for item in inputs]
        if len(set(names)) != len(names):
            raise ValueError('Distributed inputs must have distinct sanitized output names.')
        for item in inputs:
            if not item.chains or not item.rng_seeds or len(set(item.rng_seeds)) != len(item.rng_seeds):
                raise ValueError('Each distributed input needs chains and unique nonempty seeds.')
        os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
        return inputs
    fold_inputs = context.collective_call(
        lambda: load_inputs() if context.rank == 0 else None, 'input loading')
    fold_inputs = context.broadcast_object(fold_inputs)
    context.collective_call(lambda: _print_model_notice() if context.rank == 0 else None, 'model notice')

    def construct():
        return ModelRunner(
            model_dir=pathlib.Path(_MODEL_DIR.value), device=torch.device('cpu'),
            cpu_precision=_CPU_PRECISION.value, glu_backend=_GLU_BACKEND.value,
            triangle_backend=_TRIANGLE_BACKEND.value, attention_backend=_ATTENTION_BACKEND.value,
            cache_mode=_CACHE_MODE.value, input_cache_max_mb=_INPUT_CACHE_MAX_MB.value,
            layer_norm_backend=_LAYER_NORM_BACKEND.value, cpu_threads=_CPU_THREADS.value,
            cpu_interop_threads=_CPU_INTEROP_THREADS.value, distributed_context=context,
            sample_rng=_SAMPLE_RNG.value, diffusion_steps=_DIFFUSION_STEPS.value,
        )
    model_runner = context.collective_call(construct, 'model construction and weight loading')
    if context.mode == 'tasks':
        _distributed_tasks(fold_inputs, model_runner, context)
    else:
        for fold_input in fold_inputs:
            process_fold_input(
                fold_input=fold_input, data_pipeline_config=None, model_runner=model_runner,
                output_dir=os.path.join(_OUTPUT_DIR.value, fold_input.sanitised_name()),
                distributed_context=context,
            )
    print(f'Rank {context.rank}: completed {len(fold_inputs)} inputs.')


def main(_):
    if _DISTRIBUTED_MODE.value != 'off':
        from xfold.distributed import distributed_scope
        with distributed_scope(
            mode=_DISTRIBUTED_MODE.value, backend=_DISTRIBUTED_BACKEND.value,
            timeout_seconds=_DISTRIBUTED_TIMEOUT.value, schedule=_DISTRIBUTED_SCHEDULE.value,
            rank_weights=_DISTRIBUTED_RANK_WEIGHTS.value, max_payload_mb=_DISTRIBUTED_MAX_PAYLOAD.value,
        ) as context:
            return _main_distributed(context)
    if (_JSON_PATH.value is None) == (_INPUT_DIR.value is None):
        raise ValueError(
            'Exactly one of --json_path or --input_dir must be specified.'
        )

    if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
        raise ValueError(
            'At least one of --run_inference or --run_data_pipeline must be'
            ' set to true.'
        )

    _load_alphafold_dependencies()

    if _INPUT_DIR.value is not None:
        fold_inputs = folding_input.load_fold_inputs_from_dir(
            pathlib.Path(_INPUT_DIR.value)
        )
    elif _JSON_PATH.value is not None:
        fold_inputs = folding_input.load_fold_inputs_from_path(
            pathlib.Path(_JSON_PATH.value)
        )
    else:
        raise AssertionError(
            'Exactly one of --json_path or --input_dir must be specified.'
        )

    # Make sure we can create the output directory before running anything.
    try:
        os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
    except OSError as e:
        print(f'Failed to create output directory {_OUTPUT_DIR.value}: {e}')
        raise

    _print_model_notice()

    if _RUN_DATA_PIPELINE.value:
        def replace_db_dir(x): return string.Template(x).substitute(
            DB_DIR=_DB_DIR.value
        )
        data_pipeline_config = pipeline.DataPipelineConfig(
            jackhmmer_binary_path=_JACKHMMER_BINARY_PATH.value,
            nhmmer_binary_path=_NHMMER_BINARY_PATH.value,
            hmmalign_binary_path=_HMMALIGN_BINARY_PATH.value,
            hmmsearch_binary_path=_HMMSEARCH_BINARY_PATH.value,
            hmmbuild_binary_path=_HMMBUILD_BINARY_PATH.value,
            small_bfd_database_path=replace_db_dir(
                _SMALL_BFD_DATABASE_PATH.value),
            mgnify_database_path=replace_db_dir(_MGNIFY_DATABASE_PATH.value),
            uniprot_cluster_annot_database_path=replace_db_dir(
                _UNIPROT_CLUSTER_ANNOT_DATABASE_PATH.value
            ),
            uniref90_database_path=replace_db_dir(
                _UNIREF90_DATABASE_PATH.value),
            ntrna_database_path=replace_db_dir(_NTRNA_DATABASE_PATH.value),
            rfam_database_path=replace_db_dir(_RFAM_DATABASE_PATH.value),
            rna_central_database_path=replace_db_dir(
                _RNA_CENTRAL_DATABASE_PATH.value
            ),
            pdb_database_path=replace_db_dir(_PDB_DATABASE_PATH.value),
            seqres_database_path=replace_db_dir(_SEQRES_DATABASE_PATH.value),
            jackhmmer_n_cpu=_JACKHMMER_N_CPU.value,
            nhmmer_n_cpu=_NHMMER_N_CPU.value,
        )
    else:
        print('Skipping running the data pipeline.')
        data_pipeline_config = None

    if _RUN_INFERENCE.value:
        device_name = _DEVICE.value
        if device_name == 'auto':
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_name)
        print(f'Found local device: {device}')

        print('Building model from scratch...')
        model_runner = ModelRunner(
            model_dir=pathlib.Path(_MODEL_DIR.value),
            device=device,
            cpu_precision=_CPU_PRECISION.value,
            glu_backend=_GLU_BACKEND.value,
            triangle_backend=_TRIANGLE_BACKEND.value,
            attention_backend=_ATTENTION_BACKEND.value,
            cache_mode=_CACHE_MODE.value,
            input_cache_max_mb=_INPUT_CACHE_MAX_MB.value,
            layer_norm_backend=_LAYER_NORM_BACKEND.value,
            cpu_threads=_CPU_THREADS.value,
            cpu_interop_threads=_CPU_INTEROP_THREADS.value,
            sample_rng=_SAMPLE_RNG.value,
            diffusion_steps=_DIFFUSION_STEPS.value,
        )
    else:
        print('Skipping running model inference.')
        model_runner = None

    print(f'Processing {len(fold_inputs)} fold inputs.')
    for fold_input in fold_inputs:
        process_fold_input(
            fold_input=fold_input,
            data_pipeline_config=data_pipeline_config,
            model_runner=model_runner,
            output_dir=os.path.join(
                _OUTPUT_DIR.value, fold_input.sanitised_name()),
        )

    print(f'Done processing {len(fold_inputs)} fold inputs.')


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'output_dir',
    ])
    app.run(main)
