// SPDX-License-Identifier: BSD-3-Clause
#include "tools.h"

#include <libxsmm.h>
#include <omp.h>
#include <torch/version.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    libxsmm_init();
    module.attr("packed_weight_api") = 1;
    module.def("runtime_info", []() {
        py::dict info;
        info["libxsmm_revision"] = XFOLD_LIBXSMM_REVISION;
        info["libxsmm_target"] = libxsmm_get_target_arch();
        info["torch_version"] = TORCH_VERSION;
        info["openmp_max_threads"] = omp_get_max_threads();
        return info;
    });
    module.def("pad_and_align_tensor", &pad_and_align_tensor,
               py::arg("x"), py::arg("shape"), py::arg("need_init") = true,
               py::arg("init") = 0.0, py::arg("alignment") = 64);
    module.def("vnni_repack_tensor", &vnni_repack_tensor,
               py::arg("x"), py::arg("type") = 2, py::arg("BLK_SZ") = 0);
    module.def("gemm", &xsmm_gemm, py::arg("a"), py::arg("b"));
    module.def("gated_linear_unit", &xsmm_gated_linear_unit,
               py::arg("x"), py::arg("weight"),
               py::arg("M_BLK") = 128, py::arg("N_BLK") = 128,
               py::arg("weight_packed") = false,
               py::call_guard<py::gil_scoped_release>());
    module.def("linear", &xsmm_linear, py::arg("x"), py::arg("weight"),
               py::arg("bias") = py::none(), py::arg("weight_packed") = false,
               py::call_guard<py::gil_scoped_release>());
    module.def("triangle_multiplication", &xsmm_triangle_multiplication,
               py::arg("pair"), py::arg("mask"), py::arg("outgoing"),
               py::arg("left_weight"), py::arg("left_bias"),
               py::arg("projection"), py::arg("gate"),
               py::arg("center_weight"), py::arg("center_bias"),
               py::arg("output_weight"), py::arg("gating_weight"),
               py::arg("left_eps") = 1e-5, py::arg("center_eps") = 1e-5,
               py::arg("block_size") = 128, py::arg("weights_packed") = false,
               py::call_guard<py::gil_scoped_release>());
    module.def("dot_product_attention", &xsmm_dot_product_attention,
               py::arg("q"), py::arg("k"), py::arg("v"),
               py::arg("mask") = py::none(), py::arg("bias") = py::none(),
               py::arg("query_block") = 128, py::arg("q_pre_scaled") = false,
               py::call_guard<py::gil_scoped_release>());
}
