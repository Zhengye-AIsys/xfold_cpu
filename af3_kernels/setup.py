"""Build the optional native package against the active PyTorch environment."""

import os
from pathlib import Path
import platform
import subprocess
import sys

from setuptools import setup

# Build backends need not put this setup directory on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_support import LIBXSMM_REVISION, ROOT, stage_libxsmm


if platform.system() != 'Linux' or platform.machine().lower() not in ('x86_64', 'amd64'):
    raise RuntimeError('af3_kernels builds require Linux x86_64; xfold CPU baselines do not require this package.')

try:
    from torch.utils.cpp_extension import BuildExtension, CppExtension
except ImportError as error:
    raise RuntimeError(
        'Install the target CPU PyTorch environment first, then build with '
        '--no-build-isolation --no-deps.'
    ) from error


class BuildLibxsmmExtension(BuildExtension):
    def run(self):
        workspace = stage_libxsmm(
            os.environ.get('LIBXSMM_ROOT', ROOT / 'third_party/libxsmm'),
            self.build_temp,
        )
        jobs = int(os.environ.get('MAX_JOBS', os.cpu_count() or 1))
        if jobs < 1:
            raise RuntimeError('MAX_JOBS must be a positive integer.')
        subprocess.check_call([
            'make', '-C', str(workspace), f'-j{jobs}',
            f'CC={os.environ.get("CC", "gcc")}',
            f'CXX={os.environ.get("CXX", "g++")}',
            'STATIC=1', 'PIC=pic', 'BLAS=0', 'FORTRAN=0', 'SYM=1',
            'lib/libxsmm.a', 'lib/libxsmmnoblas.a',
        ])
        archives = [workspace / 'lib/libxsmm.a', workspace / 'lib/libxsmmnoblas.a']
        for archive in archives:
            if not archive.is_file():
                raise RuntimeError(f'LIBXSMM static archive was not built: {archive}')
        for extension in self.extensions:
            extension.include_dirs.append(str(workspace / 'include'))
            extension.extra_objects = list(extension.extra_objects or []) + [str(path) for path in archives]
        super().run()


setup(
    ext_modules=[CppExtension(
        'af3_kernels._C',
        sources=[
            'src/csrc/bindings.cpp', 'src/csrc/tools.cpp', 'src/csrc/gated_linear_unit.cpp',
            'src/csrc/matrix.cpp', 'src/csrc/triangle_multiplication.cpp',
            'src/csrc/dot_product_attention.cpp',
        ],
        extra_compile_args=['-O3', '-std=c++17', '-fopenmp', '-march=native'],
        extra_link_args=['-fopenmp', '-ldl', '-lm', '-lpthread'],
        define_macros=[('XFOLD_LIBXSMM_REVISION', f'"{LIBXSMM_REVISION}"')],
    )],
    cmdclass={'build_ext': BuildLibxsmmExtension},
)
