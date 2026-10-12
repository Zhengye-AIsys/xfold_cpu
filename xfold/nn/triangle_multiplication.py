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


import torch
import torch.nn as nn

from xfold import fastnn
from xfold.fastnn import config as fastnn_config
from xfold.fastnn.gated_linear_unit import GLUWeightCache


class TriangleMultiplication(nn.Module):
    def __init__(self, c_pair: int = 128, _outgoing: bool = True) -> None:
        super(TriangleMultiplication, self).__init__()

        self.c_pair = c_pair
        self.left_norm_input = fastnn.LayerNorm(self.c_pair)
        self.projection = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.gate = nn.Linear(self.c_pair, 2 * self.c_pair, bias=False)
        self.center_norm = fastnn.LayerNorm(self.c_pair)
        self.output_projection = nn.Linear(
            self.c_pair, self.c_pair, bias=False)
        self.gating_linear = nn.Linear(self.c_pair, self.c_pair, bias=False)

        self.equation='ckj,cki->cij'
        if _outgoing is True:
            self.equation='cik,cjk->cij'
        self._native_weights = {name: GLUWeightCache() for name in (
            'projection', 'gate', 'output_projection', 'gating_linear')}

    def forward(self, pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if fastnn_config.triangle_multiplication_implementation == 'cpp':
            return self._forward_cpp(pair, mask)
        if fastnn_config.triangle_multiplication_implementation == 'torch':
            return self._forward_torch(pair, mask)
        raise ValueError('Unknown triangle_multiplication_implementation: '
                         f'{fastnn_config.triangle_multiplication_implementation}')

    def _forward_cpp(self, pair, mask):
        from xfold.fastnn.native import check_cpu_inference, compute_dtype, load_native_operator, prepare_tensor

        parameters = tuple(self.parameters())
        check_cpu_inference(pair, mask, *parameters)
        names = ('projection', 'gate', 'output_projection', 'gating_linear')
        dtype = compute_dtype(pair, *(getattr(self, name).weight for name in names))
        extension = load_native_operator('triangle_multiplication')
        with torch.amp.autocast(device_type='cpu', enabled=False):
            packed = fastnn_config.packed_weight_cache_enabled and dtype == torch.bfloat16
            weights = [
                (self._native_weights[name].prepare_packed if packed else self._native_weights[name].prepare)(
                    getattr(self, name).weight.T.resolve_conj().resolve_neg(), dtype) for name in names]
            arguments = (
                prepare_tensor(pair), prepare_tensor(mask), self.equation == 'cik,cjk->cij',
                prepare_tensor(self.left_norm_input.weight), prepare_tensor(self.left_norm_input.bias),
                *weights[:2], prepare_tensor(self.center_norm.weight), prepare_tensor(self.center_norm.bias),
                *weights[2:], self.left_norm_input.eps, self.center_norm.eps,
            )
            if packed:
                return extension.triangle_multiplication(*arguments, weights_packed=True)
            return extension.triangle_multiplication(*arguments)

    def _forward_torch(self, pair: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair (torch.Tensor): [N_token, N_token, c_pair]
            mask (torch.Tensor): [N_token]
        Returns:
            torch.Tensor: [N_token, N_token, c_pair]
        """

        pair = self.left_norm_input(pair)
        input_pair = pair

        projection = self.projection(pair)
        projection = projection.permute(2, 0, 1)
        if mask is not None:
            projection *= mask[None, ...]

        gate = self.gate(pair)
        gate = gate.permute(2, 0, 1)
        projection *= torch.sigmoid(gate)

        projection = projection.reshape(self.c_pair, 2, *projection.shape[1:])

        a, b = torch.chunk(projection, 2, dim=1)
        a, b = torch.squeeze(a, dim=1), torch.squeeze(b, dim=1)
        pair = torch.einsum(self.equation, a, b)

        pair = pair.permute(1, 2, 0)
        pair = self.center_norm(pair)
        pair = self.output_projection(pair)

        gate_out = self.gating_linear(input_pair)
        pair *= torch.sigmoid(gate_out)

        return pair
