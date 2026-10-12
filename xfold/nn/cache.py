"""Bounded, example-scoped caches for immutable inference dependencies."""

from contextlib import contextmanager
from contextvars import ContextVar
import dataclasses

import torch

from xfold.fastnn import config


_ACTIVE_SCOPE = ContextVar('xfold_input_cache_scope', default=None)


class _Uncacheable(Exception):
    pass


def _signature(value, allow_inference=True):
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise _Uncacheable
        try:
            version = value._version
        except RuntimeError:
            if not allow_inference or not value.is_inference():
                raise _Uncacheable
            # Versionless feature tensors are immutable within an explicit example scope.
            version = None
        return (id(value), value.data_ptr(), version, tuple(value.shape),
                tuple(value.stride()), value.dtype, value.device)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value), tuple((field.name, _signature(getattr(value, field.name), allow_inference))
                                   for field in dataclasses.fields(value)))
    if isinstance(value, dict):
        return tuple((key, _signature(item, allow_inference)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(_signature(item, allow_inference) for item in value)
    if value is None or isinstance(value, (str, int, float, bool, torch.dtype, torch.device)):
        return value
    raise _Uncacheable


def _retained_bytes(value):
    """Charge full backing storage, counting aliases once within this entry."""
    storages = {}

    def visit(item):
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            storages[(item.device, storage.data_ptr())] = storage.nbytes()
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            raise _Uncacheable

    visit(value)
    return sum(storages.values())


def clone_tensors(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_tensors(item) for item in value)
    if isinstance(value, list):
        return [clone_tensors(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_tensors(item) for key, item in value.items()}
    return value


class _Scope:
    def __init__(self):
        self.limit = config.input_cache_max_bytes
        self.used = 0
        self.caches = set()


@contextmanager
def input_cache_scope():
    """Start an immutable example; clear all retained inputs even on exceptions."""
    scope = _Scope()
    token = _ACTIVE_SCOPE.set(scope)
    try:
        yield
    finally:
        for cache in tuple(scope.caches):
            cache.clear()
        _ACTIVE_SCOPE.reset(token)


class InputCache:
    """Fixed named slots sharing an example-wide byte budget; never a module/buffer."""

    def __init__(self):
        self._entries = {}

    def _discard(self, name):
        entry = self._entries.pop(name, None)
        if entry is not None:
            entry[4].used -= entry[5]

    def clear(self):
        for name in tuple(self._entries):
            self._discard(name)

    def get(self, name, dependencies, owner, factory, *, modules=None, copy=None):
        scope = _ACTIVE_SCOPE.get()
        if not config.input_cache_enabled or scope is None or owner.training or torch.is_grad_enabled():
            self.clear()
            return factory()
        modules = (owner,) if modules is None else modules
        modules = tuple(dict.fromkeys(child for module in modules for child in module.modules()))
        parameters = tuple(parameter for module in modules for parameter in module.parameters(recurse=False))
        if any(module.training for module in modules):
            self.clear()
            return factory()
        try:
            settings = tuple((id(module), type(module),
                              _signature(tuple(getattr(module, name, None) for name in ('eps', 'normalized_shape'))))
                             for module in modules)
            key = (scope, _signature(dependencies), _signature(parameters, allow_inference=False), settings,
                   torch.is_autocast_enabled('cpu'), torch.get_autocast_dtype('cpu'),
                   torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda'),
                   config.layer_norm_implementation, config.gated_linear_unit_implementation,
                   config.dot_product_attention_implementation, config.triangle_multiplication_implementation,
                   config.packed_weight_cache_enabled)
        except _Uncacheable:
            self._discard(name)
            return factory()
        entry = self._entries.get(name)
        if entry is not None and entry[0] == key:
            return entry[1] if copy is None else copy(entry[1])
        self._discard(name)
        value = factory()
        try:
            size = _retained_bytes(value)
        except _Uncacheable:
            return value
        if size > scope.limit - scope.used:
            return value
        # Holding dependencies/parameters prevents allocator/id reuse while a key is live.
        self._entries[name] = (key, value, dependencies, parameters, scope, size)
        scope.used += size
        scope.caches.add(self)
        return value if copy is None else copy(value)


def reset_input_cache(model):
    """Release feature caches while retaining version-checked weight caches."""
    for module in model.modules():
        cache = getattr(module, '_input_cache', None)
        if isinstance(cache, InputCache):
            cache.clear()
