// SPDX-License-Identifier: BSD-3-Clause
// Padding and packing interfaces are adapted from the af3_kernels reference.
#include "tools.h"

#include <libxsmm.h>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>

namespace {

void check_tensor(const at::Tensor& tensor) {
    TORCH_CHECK(tensor.device().is_cpu(), "Native tools require CPU tensors.");
    TORCH_CHECK(tensor.layout() == at::kStrided, "Native tools require strided tensors.");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kBFloat16,
                "Native tools support FP32 and BF16 tensors only.");
    TORCH_CHECK(!tensor.requires_grad(), "Native tools support inference only.");
    TORCH_CHECK(!tensor.is_conj() && !tensor.is_neg(),
                "Resolve conjugate/negative tensor views before using native tools.");
}

libxsmm_blasint xsmm_size(int64_t size) {
    TORCH_CHECK(size > 0 && size <= std::numeric_limits<libxsmm_blasint>::max(),
                "Matrix dimensions must fit positive LIBXSMM integers.");
    return static_cast<libxsmm_blasint>(size);
}

at::Tensor aligned_empty(const at::Tensor& like, at::IntArrayRef shape, int64_t alignment) {
    TORCH_CHECK(alignment >= static_cast<int64_t>(sizeof(void*)) &&
                (alignment & (alignment - 1)) == 0,
                "Alignment must be a power of two and at least sizeof(void*).");
    size_t elements = 1;
    for (int64_t size : shape) {
        TORCH_CHECK(size >= 0, "Tensor dimensions must be nonnegative.");
        TORCH_CHECK(size == 0 || elements <= std::numeric_limits<size_t>::max() / static_cast<size_t>(size),
                    "Tensor size overflow.");
        elements *= static_cast<size_t>(size);
    }
    TORCH_CHECK(elements <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                "Tensor element count exceeds the ATen limit.");
    TORCH_CHECK(elements <= std::numeric_limits<size_t>::max() / like.element_size(),
                "Tensor byte size overflow.");
    size_t bytes = elements * like.element_size();
    if (bytes == 0) bytes = static_cast<size_t>(alignment);
    void* data = nullptr;
    TORCH_CHECK(posix_memalign(&data, static_cast<size_t>(alignment), bytes) == 0,
                "Aligned tensor allocation failed.");
    auto owner = std::shared_ptr<void>(data, [](void* pointer) { std::free(pointer); });
    return at::from_blob(data, shape, [owner](void*) {}, like.options());
}

}  // namespace

at::Tensor pad_and_align_tensor(const at::Tensor& tensor, at::IntArrayRef shape,
                               bool need_init, double init, int64_t alignment) {
    check_tensor(tensor);
    TORCH_CHECK(shape.size() == static_cast<size_t>(tensor.dim()), "Padding must preserve tensor rank.");
    TORCH_CHECK(alignment >= static_cast<int64_t>(sizeof(void*)) && (alignment & (alignment - 1)) == 0,
                "Invalid alignment.");
    for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
        TORCH_CHECK(shape[dim] >= tensor.size(dim), "Padding cannot shrink dimensions.");
    }
    if (shape.equals(tensor.sizes()) && tensor.is_contiguous() &&
        reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0) {
        return tensor;
    }
    auto output = aligned_empty(tensor, shape, alignment);
    if (need_init) output.fill_(init);
    auto view = output;
    for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
        view = view.narrow(dim, 0, tensor.size(dim));
    }
    view.copy_(tensor);
    return output;
}

at::Tensor vnni_repack_tensor(const at::Tensor& tensor, int64_t type, int64_t block_size) {
    check_tensor(tensor);
    TORCH_CHECK(tensor.dim() == 2 && tensor.is_contiguous(), "VNNI packing requires a contiguous 2D tensor.");
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, "VNNI packing requires BF16.");
    TORCH_CHECK(type == 2, "Only type=2 (normal to VNNI2) is supported.");
    const auto rows = xsmm_size(tensor.size(0));
    const auto cols = xsmm_size(tensor.size(1));
    TORCH_CHECK(rows % 2 == 0, "VNNI2 requires an even row count.");
    TORCH_CHECK(block_size >= 0, "BLK_SZ must be nonnegative.");
    const auto block = block_size == 0 ? cols : xsmm_size(block_size);
    TORCH_CHECK(cols % block == 0, "Columns must be divisible by BLK_SZ.");
    auto output = aligned_empty(tensor, tensor.sizes(), 64);
    // Row-major [rows, cols] is viewed as column-major [cols, rows].
    const auto shape = libxsmm_create_meltw_unary_shape(
        block, rows, cols, block, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16);
    const auto kernel = libxsmm_dispatch_meltw_unary(
        LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI2, shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    TORCH_CHECK(kernel != nullptr, "LIBXSMM could not generate a VNNI2 packing kernel for this CPU/shape.");
    const auto* input = tensor.data_ptr<at::BFloat16>();
    auto* packed = output.data_ptr<at::BFloat16>();
    const int64_t blocks = cols / block;
    #pragma omp parallel for
    for (int64_t index = 0; index < blocks; ++index) {
        libxsmm_meltw_unary_param params = {};
        params.in.primary = const_cast<at::BFloat16*>(input + index * block);
        params.out.primary = packed + index * static_cast<int64_t>(block) * rows;
        kernel(&params);
    }
    return output;
}

at::Tensor xsmm_gemm(const at::Tensor& a, const at::Tensor& b) {
    check_tensor(a);
    check_tensor(b);
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "GEMM requires 2D tensors.");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "GEMM input dtypes must match.");
    TORCH_CHECK(a.size(1) == b.size(0), "GEMM inner dimensions must match.");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "GEMM inputs must be contiguous.");
    const auto m = xsmm_size(a.size(0));
    const auto k = xsmm_size(a.size(1));
    const auto n = xsmm_size(b.size(1));
    auto output = aligned_empty(at::empty({0}, a.options().dtype(at::kFloat)), {m, n}, 64);
    auto right = b;
    auto dtype = LIBXSMM_DATATYPE_F32;
    libxsmm_bitfield flags = LIBXSMM_GEMM_FLAG_BETA_0;
    if (a.scalar_type() == at::kBFloat16) {
        TORCH_CHECK(k % 2 == 0, "BF16 GEMM requires an even inner dimension.");
        right = vnni_repack_tensor(b, 2, 0);
        dtype = LIBXSMM_DATATYPE_BF16;
        flags |= LIBXSMM_GEMM_FLAG_VNNI_A;
    }
    // Compute C^T = B^T A^T because LIBXSMM descriptors use column-major shapes.
    const auto shape = libxsmm_create_gemm_shape(
        n, m, k, n, k, n, dtype, dtype, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    const auto kernel = libxsmm_dispatch_gemm(shape, flags, LIBXSMM_GEMM_PREFETCH_NONE);
    TORCH_CHECK(kernel != nullptr, "LIBXSMM could not generate a GEMM kernel for this CPU/shape.");
    libxsmm_gemm_param params = {};
    params.a.primary = right.data_ptr();
    params.b.primary = a.data_ptr();
    params.c.primary = output.data_ptr();
    kernel(&params);
    return output;
}
