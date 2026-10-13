# xfold CPU 适配测试记录

日期：2026-10-12；代码版本：`2655ef5`（`finalversion`）。

## 测试环境与结果

测试在 Mac ARM 上的两套独立 Python 环境中完成。当前环境未配置 AlphaFold3 上游依赖、模型权重和完成 MSA/template 的输入，因此测试范围为安装、CPU 模块执行、权重文件读写与本机进程通信。完整结构预测和集群性能尚未测试。

| 项目 | 结果 |
| --- | --- |
| 源码、AST、Shell、LIBXSMM 文件校验及原生接口检查 | 110 项静态测试通过 |
| Python 3.12.8 / PyTorch 2.6.0 | 39 项运行测试通过 |
| Python 3.12.14 / PyTorch 2.14.1 | 39 项运行测试通过 |
| NumPy | 两套环境均为 2.5.4 |
| Python wheel | 构建、安装和 `pip check` 通过 |
| 安装后导入与执行 | 在源码目录之外导入模型、参数读取器和协调器，并执行 CPU GLU |
| 可选库加载 | 普通 CPU 路径未加载 Triton、IPEX 或 oneCCL；协调器导入未加载 Torch |
| 命令行入口 | 无上游 AlphaFold3 时可打印帮助；预测入口在模型分配前报告缺失依赖 |
| C++ 源码 | 六个编译单元通过语法检查，未进行链接和运行 |
| 启动脚本 | dry-run 和参数错误分支通过，未实际启动 MPI |

两套环境分别运行同一组 39 项用例。部分用例包含多个参数组合，统计中仍按一个用例计数。Gloo 通信分别使用 3 个和 6 个本机进程。

## 用例覆盖

### CPU 算子与缓存

LayerNorm 与 PyTorch functional 对照，覆盖 FP32/BF16、非连续输入及可选 affine 参数。GLU 覆盖奇数输入通道；Attention 与独立 einsum 参考对照，覆盖批次、掩码和 bias。Transition、带条件的 DiffusionTransition 及小型 Pairformer 在两种精度下运行，检查输出形状、有限值和 FP32 参数存储。Triangle 的两个方向与显式逐通道矩阵收缩对照。

缓存测试覆盖命中、特征和权重变更后的失效、复制隔离、容量限制、异常清理及训练状态禁用。无版本计数的 inference 特征仅在当前输入作用域内缓存，无版本权重不复用。线程设置在新子进程中测试，检查重复设置相同 inter-op 数量，以及已使用线程池的重设错误。

### 调度与通信

分配矩阵包含 1/2/3/5/6/8/17 个 rank，0/1/2/5/7/13/33 个工作项，覆盖轮转、分块和加权调度，包括零容量权重。检查目标为工作项完整覆盖且无重复分配。

本机 Gloo 测试覆盖不等长和空结果、非主进程源广播、NumPy 序列化、分块传输，以及 FP32/BF16/int64/bool、非连续、标量和空张量。三种调度下，1/5/7 个 indexed 样本与串行 indexed 结果逐位一致，CPU 随机数状态保持一致。任务协调覆盖 input/seed/auto 粒度、空闲进程和原始种子顺序。

采样用例使用小型确定性 denoiser，执行原有随机增强、噪声调度和去噪步公式；任务用例使用轻量预测结果。它们验证采样控制流程和通信，不涉及训练模型的预测精度。依赖加载成功的用例使用模块替身，检查名称绑定和重复调用，未验证真实 AlphaFold3 安装。

故障测试注入计算、序列化、接收分配、解码、字节视图准备和主进程输出异常，检查所有进程收到失败信息。部分异常后继续调用 collective，以检查通信顺序。内存分配错误采用异常注入，未耗尽实际内存。

### 权重读取与入口

使用小型合成记录测试压缩和未压缩文件、标量与多维数组、FP32/FP16/uint8、跨分片读取、空分片、EOF 和截断数据。分片测试包含 12 个文件，检查按数字顺序拼接，避免字典序将第 10 片排在第 2 片之前。

入口测试覆盖输入路径互斥、非法采样配置、不可用 CUDA、嵌套 BF16 上下文中的 FP32 执行，以及缺失依赖时避免分配完整模型。原生后端保护测试检查梯度限制，不依赖集群上是否安装扩展。

## C++ 检查条件

检查使用 Apple clang、PyTorch 2.6.0 的 C++ 头文件、固定版本的 LIBXSMM 头文件及其版本头生成器。bindings、tools、GLU、matrix、Triangle 和 Attention 六个编译单元使用以下参数：

```text
-std=c++17 -fsyntax-only -Xclang -fopenmp
-Wno-error=invalid-specialization
```

OpenMP 头文件由 [LLVM 16.0.6 官方模板](https://github.com/llvm/llvm-project/blob/llvmorg-16.0.6/openmp/runtime/src/include/omp.h.var)在临时目录中配置版本字段生成，未安装 OpenMP 库。Mac SDK 对 PyTorch 2.6 自带 strong_type 头中的标准模板特化产生 `invalid-specialization` 诊断，检查时将该诊断降为警告。项目构建参数和系统头文件未修改。

该结果只覆盖语法和接口检查。Linux 原生构建、链接符号、LIBXSMM JIT、OpenMP 执行、原生 BF16 数值和 AMX 指令仍需在目标平台验证。

## 修复内容

- 声明 Python >=3.11、PyTorch >=2.6，并补齐 numpy、zstandard 和 absl-py 依赖。
- 延后加载 AlphaFold3；帮助命令可独立执行，正常 runner/pipeline 调用仍加载上游依赖。
- 修复串行入口对 `json_path` 和 `input_dir` 的互斥检查。
- 修复多文件读取器的关闭状态、资源释放、SEEK_END、空分片和 EOF 处理，并显式关闭解压读取器。
- 修复 `.bin.N` 文件名匹配和分片数字排序。
- 增加五个运行测试文件，并补测依赖加载与原生梯度保护。

所有安装、缓存和构建输出均位于临时副本，验证后已清理，未更新全局 Python 依赖。`xfold_cpu` 的 1598 个参考文件内容及清单与验证前一致。

## 复验命令

在独立环境中，从仓库根目录执行：

```bash
python run_alphafold.py --helpshort
python -m unittest discover -s tests -p 'test_*static.py' -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_*runtime.py' -v
```

Abseil 的 `--helpshort` 打印帮助后可能返回 1。运行测试需要共享内存和本机通信权限；Mac 上可设置 `GLOO_SOCKET_IFNAME=lo0`。这些测试不下载权重、不安装库，也不启动 MPI。

下一步在 Linux AMX 集群准备真实 AlphaFold3 环境、CCD、模型文件和小型输入，先完成 PyTorch FP32 预测，再依次验证 BF16、原生算子、缓存、并行和 IPEX。跨节点通信、绑核/NUMA 及性能测试安排在数值对照之后。模型目录当前需要 `af3.bin.zst`、`fourier_weight.npy` 和 `fourier_bias.npy`。
