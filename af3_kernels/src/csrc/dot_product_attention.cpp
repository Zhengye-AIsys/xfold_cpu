// SPDX-License-Identifier: BSD-3-Clause
#include "tools.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <omp.h>

namespace {

std::array<int64_t, 4> strides(const at::Tensor& tensor) {
    if (!tensor.defined()) return {0, 0, 0, 0};
    return {tensor.stride(0), tensor.stride(1), tensor.stride(2), tensor.stride(3)};
}

template <typename T>
void attend(const at::Tensor& q, const at::Tensor& keys, const at::Tensor& values,
            at::Tensor& output, const at::Tensor& mask, const at::Tensor& bias,
            int64_t batches, int64_t heads, int64_t queries,
            libxsmm_blasint key_count, libxsmm_blasint d, libxsmm_blasint dv,
            libxsmm_blasint padded_keys, libxsmm_blasint qb, libxsmm_datatype dtype,
            libxsmm_bitfield flags) {
    const int64_t tiles = (queries - 1) / qb + 1;
    const libxsmm_blasint heights[2] = {qb, static_cast<libxsmm_blasint>((queries - 1) % qb + 1)};
    libxsmm_gemmfunction qk[2], pv[2];
    for (int edge = 0; edge < 2; ++edge) {
        const auto score_shape = libxsmm_create_gemm_shape(
            key_count, heights[edge], d, key_count, d, padded_keys,
            dtype, dtype, dtype, LIBXSMM_DATATYPE_F32);
        const auto value_shape = libxsmm_create_gemm_shape(
            dv, heights[edge], padded_keys, dv, padded_keys, dv,
            dtype, dtype, dtype, LIBXSMM_DATATYPE_F32);
        qk[edge] = libxsmm_dispatch_gemm(score_shape, flags, LIBXSMM_GEMM_PREFETCH_NONE);
        pv[edge] = libxsmm_dispatch_gemm(value_shape, flags, LIBXSMM_GEMM_PREFETCH_NONE);
        TORCH_CHECK(qk[edge] != nullptr && pv[edge] != nullptr,
                    "LIBXSMM could not generate an attention kernel for this CPU/shape.");
    }
    const int threads = std::max(1, omp_get_max_threads());
    auto scratch = af3_detail::aligned_empty(q, {threads, qb, padded_keys}, 64);
    auto exponentials = af3_detail::aligned_empty(
        at::empty({0}, q.options().dtype(at::kFloat)), {threads, key_count}, 64);
    const auto* q_data = q.data_ptr<T>();
    const auto* k_data = keys.data_ptr<T>();
    const auto* v_data = values.data_ptr<T>();
    const auto* mask_data = mask.defined() ? mask.data_ptr<bool>() : nullptr;
    const auto* bias_data = bias.defined() ? bias.data_ptr<float>() : nullptr;
    const auto ms = strides(mask), bs = strides(bias);
    auto* y = output.data_ptr<T>();
    auto* temporary = scratch.data_ptr<T>();
    auto* exp_data = exponentials.data_ptr<float>();
    const int64_t tile_elements = static_cast<int64_t>(qb) * padded_keys;

    #pragma omp parallel for collapse(3) schedule(static)
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t head = 0; head < heads; ++head) {
            for (int64_t tile = 0; tile < tiles; ++tile) {
                const int edge = tile == tiles - 1 ? 1 : 0;
                const int64_t row = tile * qb, bh = batch * heads + head;
                const int64_t tid = omp_get_thread_num();
                auto* probabilities = temporary + tid * tile_elements;
                auto* exps = exp_data + tid * key_count;
                libxsmm_gemm_param params = {};
                params.a.primary = const_cast<T*>(k_data + bh * d * key_count);
                params.b.primary = const_cast<T*>(q_data + (bh * queries + row) * d);
                params.c.primary = probabilities;
                qk[edge](&params);

                for (int64_t r = 0; r < heights[edge]; ++r) {
                    auto* scores = probabilities + r * padded_keys;
                    float maximum = -std::numeric_limits<float>::infinity();
                    bool has_nan = false;
                    for (int64_t j = 0; j < key_count; ++j) {
                        float value = static_cast<float>(scores[j]);
                        if (bias_data != nullptr) {
                            const int64_t offset = batch * bs[0] + head * bs[1] + (row + r) * bs[2] + j * bs[3];
                            // Baseline logits += bias rounds back into the logits dtype.
                            value = static_cast<float>(T(value + bias_data[offset]));
                        }
                        if (mask_data != nullptr) {
                            const int64_t offset = batch * ms[0] + head * ms[1] + (row + r) * ms[2] + j * ms[3];
                            if (!mask_data[offset]) value = static_cast<float>(T(-1e9f));
                        }
                        scores[j] = T(value);
                        has_nan = has_nan || std::isnan(value);
                        maximum = std::max(maximum, value);
                    }
                    if (has_nan) maximum = std::numeric_limits<float>::quiet_NaN();
                    float sum = 0.0f;
                    for (int64_t j = 0; j < key_count; ++j) {
                        exps[j] = std::exp(static_cast<float>(scores[j]) - maximum);
                        sum += exps[j];
                    }
                    for (int64_t j = 0; j < key_count; ++j) scores[j] = T(exps[j] / sum);
                    // Only internal PV padding gets zero probability. Fully masked real
                    // rows keep the finite -1e9 softmax semantics of the PyTorch baseline.
                    for (int64_t j = key_count; j < padded_keys; ++j) scores[j] = T(0.0f);
                }
                params.a.primary = const_cast<T*>(v_data + bh * padded_keys * dv);
                params.b.primary = probabilities;
                params.c.primary = y + (bh * queries + row) * dv;
                pv[edge](&params);
            }
        }
    }
}

}  // namespace

at::Tensor xsmm_dot_product_attention(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    const c10::optional<at::Tensor>& mask, const c10::optional<at::Tensor>& bias,
    int64_t query_block, bool q_pre_scaled) {
    for (const auto& tensor : {q, k, v}) {
        af3_detail::check_tensor(tensor);
        TORCH_CHECK(tensor.dim() == 4 && tensor.is_contiguous(), "Attention requires contiguous [B, H, Q/K, D].");
        TORCH_CHECK(tensor.scalar_type() == q.scalar_type(), "Attention Q/K/V dtypes must match.");
    }
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0) &&
                q.size(1) == k.size(1) && q.size(1) == v.size(1), "Attention batch/head dimensions must match.");
    TORCH_CHECK(q.size(3) == k.size(3) && k.size(2) == v.size(2), "Attention key/head dimensions must match.");
    const int64_t batches = q.size(0), heads = q.size(1), queries = q.size(2), key_count = k.size(2);
    const auto d = af3_detail::xsmm_size(q.size(3)), dv = af3_detail::xsmm_size(v.size(3));
    const auto requested_block = af3_detail::xsmm_size(query_block);
    at::Tensor mask_view, bias_view;
    if (mask.has_value()) {
        af3_detail::check_auxiliary(*mask);
        TORCH_CHECK(mask->dim() <= 4 && !mask->is_complex(), "Invalid attention mask rank/type.");
        auto bool_mask = *mask;
        if (bool_mask.dim() == 1) bool_mask = bool_mask.to(at::kBool).view({1, 1, 1, bool_mask.size(0)});
        else if (bool_mask.dim() == 2) bool_mask = bool_mask.to(at::kBool).unsqueeze(1).unsqueeze(1);
        TORCH_CHECK(bool_mask.scalar_type() == at::kBool, "Attention masks of rank 0/3/4 must be boolean.");
        mask_view = bool_mask.expand({batches, heads, queries, key_count});
    }
    if (bias.has_value()) {
        af3_detail::check_tensor(*bias);
        TORCH_CHECK(bias->dim() <= 4, "Attention bias must broadcast to [B, H, Q, K].");
        bias_view = bias->to(at::kFloat).expand({batches, heads, queries, key_count});
    }
    auto output = af3_detail::aligned_empty(q, {batches, heads, queries, dv}, 64);
    if (batches == 0 || heads == 0 || queries == 0 || key_count == 0) {
        output.zero_();
        return output;
    }
    const auto nk = af3_detail::xsmm_size(key_count);
    const int64_t bh = batches * heads;
    auto scaled = q_pre_scaled ? q : q * (1.0 / std::sqrt(static_cast<double>(d)));
    auto input = scaled.contiguous().view({bh, queries, d});
    auto keys = k.transpose(2, 3).contiguous().view({bh, d, nk});
    auto values = v.view({bh, nk, dv});
    auto inner = d, padded_keys = nk;
    auto dtype = LIBXSMM_DATATYPE_F32;
    libxsmm_bitfield flags = LIBXSMM_GEMM_FLAG_BETA_0;
    if (q.scalar_type() == at::kBFloat16) {
        inner = af3_detail::xsmm_size(static_cast<int64_t>(d) + d % 2);
        padded_keys = af3_detail::xsmm_size(static_cast<int64_t>(nk) + nk % 2);
        input = pad_and_align_tensor(input, {bh, queries, inner}, true, 0.0, 64);
        keys = pad_and_align_tensor(keys, {bh, inner, nk}, true, 0.0, 64);
        keys = vnni_repack_tensor(keys.view({bh * inner, nk}), 2, 0).view({bh, inner, nk});
        values = pad_and_align_tensor(values, {bh, padded_keys, dv}, true, 0.0, 64);
        values = vnni_repack_tensor(values.view({bh * padded_keys, dv}), 2, 0).view({bh, padded_keys, dv});
        dtype = LIBXSMM_DATATYPE_BF16;
        flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
    }
    const auto qb = static_cast<libxsmm_blasint>(std::min<int64_t>(requested_block, queries));
    if (dtype == LIBXSMM_DATATYPE_BF16) {
        attend<at::BFloat16>(input, keys, values, output, mask_view, bias_view,
                            batches, heads, queries, nk, inner, dv, padded_keys, qb, dtype, flags);
    } else {
        attend<float>(input, keys, values, output, mask_view, bias_view,
                      batches, heads, queries, nk, inner, dv, padded_keys, qb, dtype, flags);
    }
    return output;
}
