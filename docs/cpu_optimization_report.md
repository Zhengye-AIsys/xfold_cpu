# xfold CPU 优化实现与复现说明

日期：2026-10-12。代码版本：`2655ef5`（`finalversion`）。

## 工作范围

本次工作以原始 xfold 为基础，参考 xfold_cpu 的 CPU 优化实现，逐步接入 BF16、C++/LIBXSMM 算子、静态特征及权重缓存、运行时配置和并行调度。修改集中在 xfold，xfold_cpu 保留作为参考。PyTorch 路径继续保留，便于逐项对照。

实现面向 Linux x86_64 CPU，AMX 为目标硬件。本机开发和验证环境为 Mac ARM，已完成安装、源码检查、CPU 模块和本机通信测试；原生算子的目标平台执行及完整结构预测尚待集群验证。

## 实现内容

| 阶段 | 改动 | 提交 |
| --- | --- | --- |
| 1 | CPU FP32 基线；设备选择、同步及 GPU 依赖延迟加载 | `73b2a7a` |
| 2 | CPU BF16 autocast，保留 FP32 参数与浮点输入存储 | `d3e3a5a` |
| 3 | 可选 C++ 扩展、固定 LIBXSMM 源码及构建流程 | `91537ff` |
| 4 | 普通和扩散 Transition 中的 GLU 接入原生实现 | `3511d51` |
| 5–6 | Triangle Multiplication 和 Grid/SelfAttention 接入原生实现 | `9b21e9c` |
| 7 | 单次输入作用域内的静态特征缓存及 BF16 VNNI2 权重缓存 | `9c8dcda` |
| 8 | 可选 IPEX LayerNorm，线程、绑核和 NUMA 配置 | `76f40e1` |
| 9 | 可配置采样/任务并行，MPI 启动及可选 oneCCL 通信 | `e11972e` |
| 修复与验证 | 依赖、入口、权重读取修复及隔离运行测试 | `2655ef5` |

算子替换以矩阵计算为主。GLU 使用两个分块 GEMM，并在 C++ 中完成 SiLU 和乘法；Triangle 的投影与收缩使用 LIBXSMM，LayerNorm/sigmoid 使用 ATen；Attention 按 query 分块计算完整 key 维度上的 softmax，接入 Grid 和扩散 SelfAttention 的相关投影。MSA/CrossAttention、自适应归一化及输出条件化保留原实现。

参数名称、初始化和 checkpoint 布局保留。BF16 基线通过 CPU autocast 实现，与参考项目显式转换为 BF16 的方式有所不同。原生接口按 FP32/BF16 提供，AMX 是否执行取决于目标机器、构建结果和实际生成的内核。

缓存分为两类：权重缓存保存原生 BF16 VNNI2 数据，输入缓存保存当前模型调用中可重复使用的静态条件和特征。动态坐标、噪声相关计算和 recycle 更新继续执行。输入缓存按调用及阶段清理，默认容量为 512 MiB；权重缓存按版本、布局和精度变化失效。该容量不包含模型、权重副本和临时工作区。

## 并行方案

第九阶段按完整工作项分配，支持轮转、连续分块和加权规则，处理余数及空闲进程。进程数、节点数、线程数和工作量分别配置，采样数量与步数默认仍为 5 和 200。

| 模式 | 每个进程的工作 | 结果处理 |
| --- | --- | --- |
| samples | 同一输入、同一种子的若干完整扩散轨迹 | rank 0 执行主干及输出头，汇总坐标并写出 |
| tasks | 完整输入或输入/种子组合的完整预测 | rank 0 按原始种子顺序汇总、排名并写出 |

各进程保存完整模型，算子内部计算仍位于单个进程。这与参考项目的算子级多 rank 划分不同。samples 的主干和输出头集中在 rank 0；tasks 按同步轮次执行，慢任务可能影响下一轮。当前调度没有动态扩缩容和断点恢复。

MPI 用于启动进程，通信经 `torch.distributed` 的 Gloo 或 ccl 后端完成。默认 Gloo，oneCCL 为可选依赖。采样并行使用按原始种子和样本索引确定的 indexed 随机流，对照时应使用串行 indexed；原有 legacy 随机流单独保留。

## 验证结果

110 项静态测试通过。同一组 39 项运行测试分别在 Python 3.12.8/PyTorch 2.6.0 和 Python 3.12.14/PyTorch 2.14.1 中通过，两套环境的 NumPy 均为 2.5.4。wheel 构建、安装和依赖检查通过，安装后可在源码目录之外导入核心模块并执行 CPU GLU。

运行测试覆盖 FP32/BF16 小型算子、Pairformer、缓存命中与失效、权重记录读写，以及真实 3/6-process 本机 Gloo。调度分配矩阵覆盖至 17 个 rank，通信用例覆盖空闲进程、不等长结果、三种分配规则、RNG 状态及计算/序列化/输出等异常传播。

采样用例使用小型确定性 denoiser，任务用例使用轻量结果替身。它们执行真实的采样或协调方法，但不涉及训练模型的预测质量。六个 C++ 编译单元通过本机语法检查，其中 PyTorch 2.6 与 Mac SDK 的一处模板特化诊断降为警告；未进行原生链接或运行。

尚缺的实验为完整结构预测、原生算子数值对照、AMX 执行确认、IPEX、MPI/oneCCL 及跨节点测试。目前没有集群耗时、吞吐或峰值内存数据。

测试在临时环境中完成，未更新全局依赖；参考项目的 1598 个文件内容及清单保持一致。测试条件及覆盖范围见 [测试记录](executable_validation.md)。

## 集群复现安排

先准备 Python >=3.11、PyTorch >=2.6 和兼容的上游 AlphaFold3 环境，配置 CCD、完成 MSA/template 的小型输入，以及模型目录中的 `af3.bin.zst`、`fourier_weight.npy` 和 `fourier_bias.npy`。

1. 用 PyTorch FP32 跑通完整预测，再以相同输入、权重、种子和线程设置对照 BF16。
2. 在目标机器构建 af3_kernels 0.4.0，分别启用 GLU、Triangle 和 Attention，先记录算子误差，再比较完整输出。
3. 分别比较 off、weights、inputs 和 all；增加复用模型 A→B→A、权重重载、容量不足和不同精度的用例。
4. 在固定配置下比较 PyTorch/IPEX，再调整线程、CPU 绑定和 NUMA 放置。
5. 从单机 Gloo 开始，比较串行 indexed 与 samples、串行 legacy 与 tasks；随后测试 MPI 启动、跨节点 Gloo 和匹配的 oneCCL 环境。
6. 数值及输出核对后，分别记录冷启动、预热后延迟、吞吐和峰值内存，并检查生成内核确认 AMX 使用。

实验中一次只改变一组后端或配置，保留各组输出和运行版本，便于区分精度、算子、缓存、通信和资源放置的影响。

## 配置文档

- [C++/LIBXSMM 接口与构建](../af3_kernels/README.md)
- [LayerNorm、线程和 NUMA](stage8_cpu.md)
- [采样与任务调度](stage9_distributed.md)
- [测试条件及复验命令](executable_validation.md)
