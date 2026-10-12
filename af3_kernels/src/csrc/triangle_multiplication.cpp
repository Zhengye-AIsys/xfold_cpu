// SPDX-License-Identifier: BSD-3-Clause
#include "tools.h"

#include <cmath>

at::Tensor xsmm_triangle_multiplication(
    const at::Tensor& pair, const c10::optional<at::Tensor>& mask, bool outgoing,
    const at::Tensor& left_weight, const c10::optional<at::Tensor>& left_bias,
    const at::Tensor& projection, const at::Tensor& gate,
    const at::Tensor& center_weight, const c10::optional<at::Tensor>& center_bias,
    const at::Tensor& output_weight, const at::Tensor& gating_weight,
    double left_eps, double center_eps, int64_t block_size, bool weights_packed) {
    af3_detail::check_tensor(pair);
    TORCH_CHECK(pair.dim() == 3 && pair.size(0) == pair.size(1) && pair.is_contiguous(),
                "Triangle input must be contiguous [N, N, C].");
    const int64_t n = pair.size(0), c = af3_detail::xsmm_size(pair.size(2));
    af3_detail::xsmm_size(block_size);
    TORCH_CHECK(!weights_packed || projection.scalar_type() == at::kBFloat16, "Packed Triangle weights require BF16.");
    TORCH_CHECK(std::isfinite(left_eps) && left_eps >= 0.0 && std::isfinite(center_eps) && center_eps >= 0.0,
                "LayerNorm eps must be finite and nonnegative.");
    for (const auto& weight : {projection, gate, output_weight, gating_weight}) {
        af3_detail::check_tensor(weight);
        TORCH_CHECK(weight.dim() == 2 && weight.size(0) == c + (weights_packed ? c % 2 : 0) && weight.is_contiguous() &&
                    weight.scalar_type() == projection.scalar_type(), "Invalid Triangle projection weight.");
    }
    TORCH_CHECK(projection.size(1) == 2 * c && gate.size(1) == 2 * c &&
                output_weight.size(1) == c && gating_weight.size(1) == c, "Triangle weight widths do not match C.");
    for (const auto& weight : {left_weight, center_weight}) {
        af3_detail::check_tensor(weight);
        TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c, "LayerNorm weights must be [C].");
    }
    for (const auto& bias : {left_bias, center_bias}) {
        if (bias.has_value()) {
            af3_detail::check_tensor(*bias);
            TORCH_CHECK(bias->dim() == 1 && bias->size(0) == c, "LayerNorm biases must be [C].");
        }
    }
    if (mask.has_value()) {
        af3_detail::check_auxiliary(*mask);
        TORCH_CHECK(mask->dim() <= 2 && !mask->is_complex(), "Triangle mask must broadcast to [N, N].");
        // Validate broadcasting even for an empty pair.
        mask->expand({n, n});
    }
    if (n == 0) return af3_detail::aligned_empty(projection, {n, n, c}, 64);
    auto flat = at::layer_norm(pair, {c}, left_weight, left_bias, left_eps, true)
                    .to(projection.scalar_type()).contiguous().view({n * n, c});
    at::Tensor contracted;
    {
        auto projected = af3_detail::tiled_matmul(flat, projection, c10::nullopt, block_size, block_size, weights_packed)
                         .view({n, n, 2 * c}).permute({2, 0, 1});
        if (mask.has_value()) projected.mul_(mask->unsqueeze(0));
        auto gate_values = af3_detail::tiled_matmul(flat, gate, c10::nullopt, block_size, block_size, weights_packed)
                           .view({n, n, 2 * c}).permute({2, 0, 1});
        projected.mul_(at::sigmoid(gate_values));
        // Original reshape groups adjacent channels: a_c = projected[2*c], b_c = projected[2*c+1].
        auto interleaved = projected.reshape({c, 2, n, n});
        auto a = interleaved.select(1, 0).contiguous();
        auto b = interleaved.select(1, 1).contiguous();
        // Outgoing: A B^T; incoming: B^T A (ckj,cki->cij).
        auto left = outgoing ? a : b.transpose(1, 2).contiguous();
        auto right = outgoing ? b.transpose(1, 2).contiguous() : a;
        contracted = af3_detail::tiled_matmul(left, right, c10::nullopt, block_size, block_size)
                         .permute({1, 2, 0});
    }  // Release projection/gate and contraction input storage before the output projections.
    auto centered = at::layer_norm(contracted, {c}, center_weight, center_bias, center_eps, true)
                        .to(projection.scalar_type()).contiguous().view({n * n, c});
    contracted = at::Tensor();
    auto result = af3_detail::tiled_matmul(centered, output_weight, c10::nullopt, block_size, block_size, weights_packed);
    auto output_gate = af3_detail::tiled_matmul(flat, gating_weight, c10::nullopt, block_size, block_size, weights_packed);
    result.mul_(at::sigmoid(output_gate));
    // Return the update only. Pairformer owns the residual addition.
    return result.view({n, n, c});
}
