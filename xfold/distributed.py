"""Optional CPU inference coordination; no distributed imports at import time."""

from contextlib import contextmanager
from datetime import timedelta
from importlib import import_module
import hashlib
import io
import os
import pickle
import platform
import re


def work_indices(total, rank, world_size, schedule='cyclic', weights=()):
    """Exact disjoint coverage, including remainders and ranks with no work."""
    if type(total) is not int or total < 0 or type(world_size) is not int or world_size < 1:
        raise ValueError('Work count must be nonnegative and world size positive.')
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError('Rank must be within world size.')
    if schedule == 'cyclic':
        if weights:
            raise ValueError('Rank weights require weighted scheduling.')
        return range(rank, total, world_size)
    if schedule == 'block':
        if weights:
            raise ValueError('Rank weights require weighted scheduling.')
        quotient, remainder = divmod(total, world_size)
        begin = rank * quotient + min(rank, remainder)
        return range(begin, begin + quotient + int(rank < remainder))
    if schedule == 'weighted':
        if len(weights) != world_size or any(type(w) is not int or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError('Provide one nonnegative integer weight per rank and a positive total.')
        weight_total = sum(weights)
        begin = total * sum(weights[:rank]) // weight_total
        end = total * sum(weights[:rank + 1]) // weight_total
        return range(begin, end)
    raise ValueError('Schedule must be cyclic, block, or weighted.')


def parse_rank_weights(value):
    if not value:
        return ()
    items = value.split(',')
    if any(re.fullmatch(r'[0-9]+', item.strip()) is None for item in items):
        raise ValueError('Rank weights must be comma-separated nonnegative integers.')
    return tuple(int(item) for item in items)


def sample_seed(base_seed, sample_index):
    """Stable CPU stream identity; process/rank/schedule never enter the seed."""
    if type(sample_index) is not int or sample_index < 0:
        raise ValueError('Sample index must be nonnegative.')
    value = f'xfold-sample-v1:{base_seed}:{sample_index}'.encode('ascii')
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), 'big') % (2 ** 63)


def rank_environment():
    """Prefer torchrun's pair; otherwise use a complete MPI/Slurm family."""
    families = (
        ('RANK', 'WORLD_SIZE', ('LOCAL_RANK',)),
        ('PMI_RANK', 'PMI_SIZE', ('MPI_LOCALRANKID', 'PMI_LOCAL_RANK')),
        ('OMPI_COMM_WORLD_RANK', 'OMPI_COMM_WORLD_SIZE', ('OMPI_COMM_WORLD_LOCAL_RANK',)),
        ('SLURM_PROCID', 'SLURM_NTASKS', ('SLURM_LOCALID',)),
    )
    for rank_name, size_name, local_names in families:
        if rank_name not in os.environ and size_name not in os.environ:
            continue
        if rank_name not in os.environ or size_name not in os.environ:
            raise ValueError(f'{rank_name} and {size_name} must be supplied together.')
        rank, world_size = int(os.environ[rank_name]), int(os.environ[size_name])
        local_rank = next((int(os.environ[name]) for name in local_names if name in os.environ), 0)
        if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
            raise ValueError('Invalid rank/world-size/local-rank environment.')
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


class DistributedContext:
    """Collective phases use CPU int64 controls and chunked uint8 data."""

    chunk_bytes = 16 * 1024 * 1024

    def __init__(self, torch, dist, mode, rank, world_size, local_rank, schedule, weights, max_payload_bytes):
        self.torch, self.dist, self.mode = torch, dist, mode
        self.rank, self.world_size, self.local_rank = rank, world_size, local_rank
        self.schedule, self.weights = schedule, weights
        self.max_payload_bytes = max_payload_bytes

    def indices(self, total, rank=None):
        return work_indices(total, self.rank if rank is None else rank,
                            self.world_size, self.schedule, self.weights)

    def collective_call(self, factory, phase):
        """Propagate Python computation errors before the next data collective."""
        if self.world_size == 1:
            return factory()
        result, failure = None, None
        try:
            result = factory()
        except Exception as error:
            failure = error
        first_failed = self.torch.tensor(
            [self.rank if failure is not None else self.world_size], dtype=self.torch.int64, device='cpu')
        self.dist.all_reduce(first_failed, op=self.dist.ReduceOp.MIN)
        failed_rank = int(first_failed.item())
        if failed_rank != self.world_size:
            message = f'{type(failure).__name__}: {failure}'[:8192] if self.rank == failed_rank else None
            message = self.broadcast_object(message, src=failed_rank)
            raise RuntimeError(f'Distributed phase {phase} failed on rank {failed_rank}: {message}') from failure
        return result

    def broadcast_object(self, value, src=0, root_only=False):
        """Variable-sized objects without backend-specific object collectives.

        All ranks take part. root_only retains/decodes data only on rank 0;
        other receivers reuse one bounded discard buffer.
        """
        if self.world_size == 1:
            return value
        payload, failed = None, False
        if self.rank == src:
            try:
                stream = io.BytesIO()
                pickle.dump(value, stream, protocol=5)
                payload = stream.getbuffer()
                if len(payload) > self.max_payload_bytes:
                    raise ValueError('Object exceeds --distributed_max_payload_mb; increase the explicit budget.')
            except Exception as error:
                failed = True
                payload = memoryview(bytearray(f'{type(error).__name__}: {error}'[:8192].encode('utf-8')))
        header = self.torch.tensor([int(failed), len(payload)] if self.rank == src else [0, 0],
                                   dtype=self.torch.int64, device='cpu')
        self.dist.broadcast(header, src=src)
        failed, length = bool(header[0].item()), int(header[1].item())
        retain = failed or not root_only or self.rank == 0
        # The source has already allocated its stream. Other ranks allocate a
        # full receiving object only when needed; allocation failures agree first.
        def allocate():
            if self.rank == src:
                return payload
            return memoryview(bytearray(length if retain else min(length, self.chunk_bytes)))
        # Avoid recursive error transport here: allocation failure is encoded in
        # a fixed scalar, then all participants fail before payload broadcasting.
        allocation_failure = None
        try:
            buffer = allocate()
        except Exception as error:
            allocation_failure, buffer = error, None
        status = self.torch.tensor([int(allocation_failure is not None)], dtype=self.torch.int64, device='cpu')
        self.dist.all_reduce(status, op=self.dist.ReduceOp.MAX)
        if int(status.item()):
            raise RuntimeError('Distributed object receive allocation failed on a rank.') from allocation_failure
        for begin in range(0, length, self.chunk_bytes):
            size = min(self.chunk_bytes, length - begin)
            offset = begin if self.rank == src or retain else 0
            chunk = self.torch.frombuffer(buffer[offset:offset + size], dtype=self.torch.uint8)
            self.dist.broadcast(chunk, src=src)
        if failed:
            raise RuntimeError(bytes(buffer).decode('utf-8'))
        return self.collective_call(
            lambda: (value if self.rank == src else pickle.loads(buffer)) if retain else None,
            'object decoding')

    def gather_objects_to_root(self, value):
        # All ranks call all sources in the same order, including empty workers.
        received = [] if self.rank == 0 else None
        for source in range(self.world_size):
            item = self.broadcast_object(value if self.rank == source else None, source, root_only=True)
            if self.rank == 0:
                received.append(item)
        return received

    def broadcast_tensor(self, tensor, src=0):
        if self.world_size == 1:
            return tensor
        def prepare():
            if tensor.device.type != 'cpu' or tensor.layout != self.torch.strided:
                raise ValueError('Distributed inference tensors must be strided CPU tensors.')
            output = tensor.contiguous()
            # Shape/dtype view errors must agree before any rank broadcasts data.
            raw = output.reshape(-1).view(self.torch.uint8)
            return output, raw
        output, raw = self.collective_call(prepare, 'tensor preparation')
        for begin in range(0, raw.numel(), self.chunk_bytes):
            self.dist.broadcast(raw[begin:begin + self.chunk_bytes], src=src)
        return output

    def broadcast_tensor_dict(self, values, src=0):
        metadata = self.collective_call(
            lambda: [(name, tuple(value.shape), str(value.dtype).removeprefix('torch.'))
                     for name, value in values.items()] if self.rank == src else None,
            'embedding metadata')
        metadata = self.broadcast_object(metadata, src)
        output = {}
        for name, shape, dtype in metadata:
            tensor = self.collective_call(
                lambda: values[name] if self.rank == src else self.torch.empty(
                    shape, dtype=getattr(self.torch, dtype), device='cpu'),
                'embedding allocation')
            output[name] = self.broadcast_tensor(tensor, src)
        return output


@contextmanager
def distributed_scope(mode='off', backend='gloo', timeout_seconds=3600,
                      schedule='cyclic', rank_weights='', max_payload_mb=4096):
    """Explicit fixed process group; serial mode never loads distributed/CCL."""
    if mode == 'off':
        yield None
        return
    if mode not in ('samples', 'tasks') or backend not in ('gloo', 'ccl'):
        raise ValueError('Distributed mode must be samples/tasks; backend gloo/ccl.')
    if platform.system() != 'Linux':
        raise RuntimeError('Stage 9 distributed CPU inference requires Linux.')
    if type(timeout_seconds) is not int or timeout_seconds <= 0 or type(max_payload_mb) is not int or max_payload_mb < 1:
        raise ValueError('Distributed timeout and object budget must be positive integers.')
    rank, world_size, local_rank = rank_environment()
    weights = parse_rank_weights(rank_weights)
    work_indices(0, rank, world_size, schedule, weights)
    torch = import_module('torch')
    dist = None
    if world_size > 1:
        if not os.environ.get('MASTER_ADDR') or not os.environ.get('MASTER_PORT'):
            raise ValueError('Multiple ranks require explicit MASTER_ADDR and MASTER_PORT.')
        if not 1 <= int(os.environ['MASTER_PORT']) <= 65535:
            raise ValueError('MASTER_PORT must be within 1..65535.')
        dist = import_module('torch.distributed')
        if not dist.is_available() or dist.is_initialized():
            raise RuntimeError('Distributed support must be available with no existing process group.')
        if backend == 'ccl':
            if platform.machine().lower() not in ('x86_64', 'amd64'):
                raise RuntimeError('Optional CPU oneCCL requires Linux x86_64.')
            try:
                import_module('oneccl_bindings_for_pytorch')
            except (ImportError, OSError, RuntimeError, SystemExit) as error:
                raise RuntimeError('Install CPU oneCCL bindings matching PyTorch, or select gloo.') from error
        elif not dist.is_gloo_available():
            raise RuntimeError('This PyTorch installation lacks the requested Gloo backend.')
    try:
        if dist is not None:
            dist.init_process_group(backend=backend, init_method='env://',
                                    timeout=timedelta(seconds=timeout_seconds), rank=rank, world_size=world_size)
        context = DistributedContext(torch, dist, mode, rank, world_size, local_rank,
                                     schedule, weights, max_payload_mb * 1024 * 1024)
        print(f'Distributed {mode}: rank={rank}/{world_size}, local_rank={local_rank}, '
              f'backend={backend if world_size > 1 else "unused (one rank)"}, schedule={schedule}; '
              f'PyTorch={torch.__version__}')
        yield context
    finally:
        # No exit barrier: computation/communication failures must not block
        # cleanup on a second collective. Only this scope owns the group.
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()
