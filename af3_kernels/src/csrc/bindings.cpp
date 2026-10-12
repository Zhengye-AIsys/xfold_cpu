// SPDX-License-Identifier: BSD-3-Clause
#include "tools.h"

#include <libxsmm.h>
#include <omp.h>
#include <torch/version.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    libxsmm_init();
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
               py::call_guard<py::gil_scoped_release>());
}
