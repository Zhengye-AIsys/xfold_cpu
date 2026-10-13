# CPU LayerNorm 与线程、NUMA 配置

本文对应第八阶段实现。LayerNorm 后端、线程数和进程绑定分别配置，可用于比较不同运行组合。优化对象为 xfold，xfold_cpu 作为实现参考。

## LayerNorm 后端

| 参数 | 实现 |
| --- | --- |
| `--layer_norm_backend=auto` | CPU 使用 PyTorch；CUDA 在启用 fastnn 时使用 Triton |
| `--layer_norm_backend=torch` | 原有 `F.layer_norm` |
| `--layer_norm_backend=ipex` | IPEX 的 `llm.functional.fast_layer_norm` |

IPEX 路径保留原 LayerNorm 模块及其参数名、weight/bias、normalized shape 和 epsilon。输入和参数按当前精度传入，不额外转换 dtype；参数仍以 FP32 存储。该路径用于 eval 模式下的 FP32/BF16 CPU 推理，其他设备、布局、数据类型或需要反向传播的调用会报错。

仅选择 `ipex` 时加载 `intel_extension_for_pytorch`。模型构建前检查 Linux x86_64 平台、库导入和 API 是否可用，失败时报告错误。IPEX 单独安装，版本应与目标 PyTorch 环境匹配；更换 PyTorch 后需要重新构建原生扩展。

此处接入的是 functional 接口，没有使用 `ipex.optimize`、全模型替换或 JIT。部分 IPEX 版本的 FastLayerNorm 内部仍调用 PyTorch，具体收益需要测量。原生 Triangle 中的 LayerNorm 继续使用 ATen。

切换 LayerNorm 后端会使输入缓存失效。IPEX 导入可能注册进程级算子，因此 PyTorch/IPEX 的数值、时间和内存对照应分别使用新进程。

## 线程设置

`--cpu_threads` 设置 PyTorch intra-op 线程数，`--cpu_interop_threads` 设置 inter-op 线程数。默认不修改运行时设置；指定值时要求为正的 32 位整数，并在模型构建前配置。

inter-op 数量已匹配时跳过设置。若已使用的线程池不能接受新数量，程序报告错误；inter-op 先于 intra-op 设置，避免失败后只改变其中一项。线程数量不随输入重置。

Python 参数配置 PyTorch 线程池。原生扩展使用的 OpenMP 运行时还需要环境变量，因此启动脚本在 Python 启动前设置 `OMP_NUM_THREADS` 和 `MKL_NUM_THREADS`。启动日志记录线程数、可用 CPU 集合及相关环境变量。

## 启动脚本

```text
bash scripts/run_cpu.sh [--dry-run] <run_alphafold.py 参数>
```

`--dry-run` 必须位于首个参数，只打印环境和命令；正式执行要求 Linux。脚本固定选择 CPU，默认关闭数据流水线，输入应已完成 MSA/template 处理。其他模型参数直接转发。

device、cpu_threads 和 cpu_interop_threads 由脚本统一管理，重复传入单破折号或双破折号形式均会被拒绝。flagfile 也不转发，以免覆盖脚本配置。运行参数中的相对路径按调用时的工作目录解释。

| 环境变量 | 默认值或用途 |
| --- | --- |
| `XFOLD_CPU_PYTHON` | `python`，也可指定解释器路径 |
| `XFOLD_CPU_ENV_FILE` | 可选配置文件，需指定可读路径 |
| `XFOLD_CPU_THREADS` | 不设置时继承；设置后同步 PyTorch、OpenMP 和 MKL 线程数 |
| `XFOLD_CPU_INTEROP_THREADS` | 不设置时继承；设置 PyTorch inter-op 数量 |
| `XFOLD_CPU_CPUS` | 指定 CPU 列表，如 `0-7,16-23` 或相对列表 `+0-7` |
| `XFOLD_CPU_NODES` | 按 NUMA 节点选择 CPU，与 `XFOLD_CPU_CPUS` 互斥 |
| `XFOLD_CPU_MEM_POLICY` | `inherit`、`bind`、`interleave` 或 `local` |
| `XFOLD_CPU_MEM_NODES` | bind/interleave 时指定内存节点 |
| `XFOLD_CPU_AFFINITY` | `inherit`、`compact`、`spread` 或 `none` |
| `XFOLD_CPU_KMP_BLOCKTIME` | 可选的非负 Intel OpenMP blocktime |
| `XFOLD_CPU_OMP_LIBRARY` | 可选 OpenMP 库绝对路径 |
| `XFOLD_CPU_ALLOCATOR_LIBRARY` | 可选分配器库绝对路径 |

配置模板为 `scripts/cpu_env.sh.example`。配置文件以 shell 方式读取，其中赋值覆盖同名环境变量，宜复制为单独的作业配置。

compact/spread 对应 `OMP_PLACES=cores` 与 `OMP_PROC_BIND=close/spread`；none 设置 `OMP_PROC_BIND=false`。选择这些策略前需清除已有的 `KMP_AFFINITY` 和 `GOMP_CPU_AFFINITY` 冲突项。inherit 保留已有设置。

CPU 列表和节点列表分别对应 numactl 的 `--physcpubind`、`--cpunodebind`。内存策略对应 `--membind`、`--interleave` 和 `--localalloc`；未选择绑定时不依赖 numactl。CPU 编号、节点和线程数应按调度器分配及实际拓扑填写，dry-run 不检查节点资源是否可用。

库预加载默认关闭。设置预加载时，路径必须为可读绝对路径且不含空白或冒号；已有 `LD_PRELOAD` 保留。OpenMP 库需与 PyTorch、IPEX 和原生扩展的构建环境匹配。

## 集群对照

以下参数用于检查命令构造，线程数需按实际分配调整：

```bash
XFOLD_CPU_THREADS=8 XFOLD_CPU_INTEROP_THREADS=1 \
  bash scripts/run_cpu.sh --dry-run \
    --cpu_precision=bf16 --layer_norm_backend=torch \
    --glu_backend=torch --triangle_backend=torch --attention_backend=torch \
    --cache_mode=off --json_path=./fold_input_data.json \
    --model_dir="$PATH_TO_AF3_MODEL" --output_dir=./stage8_torch
```

正式运行时去掉 `--dry-run`。对照 IPEX 时，在新进程中只修改 LayerNorm 后端和输出目录，分别测 FP32/BF16。之后再调整线程数、绑核和内存策略，避免同时改变多个因素。

```bash
XFOLD_CPU_THREADS=8 XFOLD_CPU_INTEROP_THREADS=1 \
XFOLD_CPU_CPUS=+0-7 XFOLD_CPU_MEM_POLICY=local XFOLD_CPU_AFFINITY=compact \
  bash scripts/run_cpu.sh --dry-run \
    --cpu_precision=bf16 --layer_norm_backend=ipex \
    --json_path=./fold_input_data.json \
    --model_dir="$PATH_TO_AF3_MODEL" --output_dir=./stage8_bound
```

主要检查项目为 LayerNorm 输出 dtype 和数值误差、缺失或不兼容依赖、线程池重设、输入缓存失效，以及实际 CPU/NUMA 放置。可结合 `torch.__config__.parallel_info()` 和原生 `runtime_info()` 检查运行时信息；冷启动、预热后延迟和峰值内存分别记录。

本机已完成线程设置、PyTorch LayerNorm 和启动脚本的隔离运行测试。IPEX 执行、实际绑核/NUMA 放置及性能对照待集群验证，测试记录见 [可执行性测试](executable_validation.md)。

## 接口依据

- [IPEX CPU API](https://intel.github.io/intel-extension-for-pytorch/cpu/2.6.0%2Bcpu/tutorials/api_doc.html)及 [functional 实现](https://github.com/intel/intel-extension-for-pytorch/blob/main/intel_extension_for_pytorch/llm/functional/fusions.py)。
- [IPEX CPU LayerNorm 实现](https://github.com/intel/intel-extension-for-pytorch/blob/main/intel_extension_for_pytorch/transformers/models/cpu/fusions/mha_fusion.py)。
- PyTorch [intra-op](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html)与 [inter-op](https://docs.pytorch.org/docs/stable/generated/torch.set_num_interop_threads.html)设置说明。
- [numactl 参数实现](https://github.com/numactl/numactl/blob/master/numactl.c)与 [Abseil 参数解析](https://github.com/abseil/abseil-py/blob/main/absl/flags/_flagvalues.py)。
