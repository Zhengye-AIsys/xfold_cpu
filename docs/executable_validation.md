# 可执行性验证：2026-10-12

## 结论与边界

当前源码可以构建并安装为 Python wheel；安装后的核心模块可导入，普通
PyTorch CPU 算子、缓存、权重记录读取和本机 Gloo 协调已实际执行通过。
这比仅做源码检查提供了更多证据，但尚不能保证完整蛋白质预测或 Intel AMX
优化路径可以在目标集群运行。

本机没有 AlphaFold3 依赖、模型权重或完成 MSA/template 的输入，硬件是
Mac ARM。因此未执行真实结构预测、Linux x86_64 原生扩展链接/运行、AMX、
IPEX、MPI 启动、oneCCL 或跨节点测试。没有进行性能测试。

采样测试使用小型确定性测试 denoiser，执行 xfold 原有的随机增强、噪声调度、
去噪步公式和 indexed 采样方法；任务测试使用轻量预测结果替身，执行真实的
任务协调、Gloo 传输和排名顺序恢复逻辑。它们验证控制流程与通信，不证明
训练模型的数值精度或预测质量。

## 隔离方式

- 从 stage9 提交导出临时源码副本，再同步本次修复及测试文件。
- 在本次专用临时目录中复制已有的临时 Python 环境；所有包安装、构建和
  测试输出均在这些私有副本中完成，没有安装或更新全局 Python 包。
- 禁止生成项目字节码，缓存指向本次临时目录；没有生成工作区虚拟环境、
  build、dist 或 egg-info。
- Gloo 测试使用本机回环接口与临时 FileStore，各进程在 finally 中销毁
  进程组，临时 rendezvous 文件随测试退出清理。
- 沙箱最初阻断 OpenMP 共享内存；获得权限后在相同临时副本重跑成功。
  没有绕过生产代码的 Linux 平台检查，测试直接调用可移植的协调器类。
- xfold_cpu 保持只读；验证结束核对其 1598 个参考文件内容与清单。
- 本次专用临时环境、安装包、头文件和构建输出在验证结束后清理。

## 验证结果

| 检查 | 环境与结果 |
| --- | --- |
| 源码、AST、Shell、LIBXSMM 清单及原生源码契约 | 110 项静态测试通过 |
| 运行测试 | PyTorch 2.6.0 / Python 3.12.8：39 项通过 |
| 运行测试 | PyTorch 2.14.1 / Python 3.12.14：39 项通过 |
| NumPy | 两套环境均为 2.5.4 |
| wheel 构建、安装、依赖检查 | 构建成功；两套私有环境安装成功；pip check 通过 |
| 安装后的包 | 在源码目录之外成功导入模型、参数读取器和协调器，并执行 CPU GLU |
| 可选依赖隔离 | CPU 导入未加载 Triton、IPEX 或 oneCCL；协调器导入未加载 Torch |
| 命令行入口 | 无 AlphaFold3 时可打印帮助；缺失依赖有明确诊断并在大模型分配前失败 |
| C++ 编译器语法检查 | 6 个 translation units 通过；仅语法检查，未链接/运行 |
| 原生包的平台拒绝 | Mac 上构建入口按设计拒绝，提示需要 Linux x86_64 |
| 启动脚本 | 真实执行 dry-run 与错误分支；未执行 MPI 或推理 |

39 项运行测试中的部分测试使用多个子场景；3 和 6 进程的 Gloo 场景都实际运行。
没有把这些子场景重复计入 unittest 测试数量。

### CPU 算子与缓存

LayerNorm 对照 PyTorch functional，覆盖 FP32/BF16、非连续输入及可选 affine。
GLU 覆盖奇数输入通道。Attention 对照独立 einsum 参考，覆盖批次、掩码和 bias。
Transition、带条件的 DiffusionTransition 及小型 Pairformer 在 FP32/BF16 下
运行，确认输出形状、有限值及 FP32 参数存储。Triangle 两个方向对照显式
逐通道矩阵收缩。

输入缓存验证命中、特征与权重变更失效、复制隔离、字节预算、异常清理和
训练状态禁用；无版本计数的 inference 特征只在明确 scope 内缓存，无版本
权重不缓存。权重转置缓存验证 dtype 与 mutation 失效。线程配置在新子进程
中验证重复匹配设置和禁止重设已用 inter-op 池。

### 调度、通信和失败传播

分配矩阵覆盖 1/2/3/5/6/8/17 ranks，0/1/2/5/7/13/33 工作项，三种调度及
零容量权重，检查全部索引恰好出现一次；错误参数拒绝测试也通过。

真实 3/6-process Gloo 验证不等长/空结果、非主进程源广播、NumPy 序列化、
分块传输、FP32/BF16/int64/bool、非连续/标量/空张量及 embedding 字典。
三种调度下 1/5/7 个 indexed 样本与串行 indexed 测试头结果逐位一致，CPU
RNG 状态保持一致。任务协调覆盖 input/seed/auto 粒度、空闲进程和种子顺序。

实际注入计算、序列化、接收分配、解码、字节视图准备及主进程输出异常，
验证所有进程收到失败信息。部分失败后还执行后续 collective，确认通信
调用顺序仍一致。分配与输出失败采用受控异常注入，没有耗尽真实内存。

### C++ 检查限制

使用 Apple clang、PyTorch 2.6.0 的真实 C++ 头文件、项目固定的 LIBXSMM
头文件及其官方版本头生成器，对 bindings/tools/GLU/matrix/Triangle/Attention
六个源文件执行 `-std=c++17 -fsyntax-only -Xclang -fopenmp`。

OpenMP 头使用 [LLVM 16.0.6 官方模板](https://github.com/llvm/llvm-project/blob/llvmorg-16.0.6/openmp/runtime/src/include/omp.h.var)，
只在临时目录配置版本字段，没有安装 OpenMP 库。Apple 当前 SDK 将 PyTorch
2.6 自带 strong_type 头中的标准模板特化报为 `invalid-specialization`；检查命令
使用 `-Wno-error=invalid-specialization` 将此第三方诊断降为警告。
项目构建参数和系统头文件均未修改。这不是 Linux GCC 原生构建证明，也不
验证链接符号、LIBXSMM JIT 生成、OpenMP 执行、BF16 数值或 AMX 指令。

## 本次修复

1. 声明 Python >=3.11、PyTorch >=2.6（已验证的最低版本），并补齐 numpy、
   zstandard、absl-py 直接依赖。
2. 延后加载上游 AlphaFold3，帮助和核心测试可独立执行；正常预测及公开
   runner/pipeline 调用仍加载真实依赖，缺失依赖在大模型分配前报告。
3. 修复串行入口的输入互斥条件，正确拒绝两个路径同时指定或同时缺失。
4. 修复多文件权重读取器的 closed 状态、关闭资源、SEEK_END 偏移/返回值、
   空分片、EOF、整数 seek 参数及意外截断处理，显式关闭解压读取器。
5. 修复 `.bin.N` 匹配，并按分片编号排序，避免第 10 片排在第 2 片之前。
6. 增加五个可重复运行的 runtime 测试文件。
   提交前复查补测依赖加载成功/复用，并调整原生保护测试，使其在已安装
   原生扩展的集群上也不依赖“扩展不存在”的假设。依赖成功测试使用模块
   替身，只验证延迟导入绑定，不表示真实 AlphaFold3 环境已验证。

## 后续复验

在准备好依赖的独立环境中，从仓库根目录运行：

```bash
python run_alphafold.py --helpshort
python -m unittest discover -s tests -p 'test_*static.py' -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -p 'test_*runtime.py' -v
```

Abseil 的 `--helpshort` 正常打印帮助后可能返回 1；不要据此认定导入失败。
需要允许进程、共享内存和本机通信；Mac 上可设置 `GLOO_SOCKET_IFNAME=lo0`。
这些测试不会安装包、下载权重或启动 MPI。

目标 Linux AMX 集群还必须执行：

1. 安装兼容的真实 AlphaFold3 环境，准备 CCD、完整输入及模型目录中的
   `af3.bin.zst`、`fourier_weight.npy`、`fourier_bias.npy`。
2. 跑一个小型 PyTorch FP32 完整预测，再跑 BF16、缓存开关与 indexed 基线，
   验证结构输出、全局排名及数值差异。
3. 按 af3_kernels README 原生构建，逐个对照 GLU、Triangle、Attention，覆盖
   奇数通道、分块尾部、掩码、packed weights、FP32/BF16 及复用模型。
4. 用真实模型验证 samples/tasks 与串行结果；本机 Gloo 之后再验证跨节点
   Gloo、MPI 启动及兼容的 oneCCL 环境。
5. 独立验证 IPEX、实际绑核/NUMA、进程退出与超时，最后测吞吐、延迟和内存。

本报告不会把尚未执行的集群步骤记为通过。
