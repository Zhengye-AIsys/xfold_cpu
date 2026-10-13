# C++ / LIBXSMM CPU 算子

`af3_kernels` 是 xfold 的可选扩展包，当前版本为 0.4.0，提供 GLU、Linear、Triangle Multiplication 和 Attention 接口。主项目默认使用 PyTorch，安装或导入 xfold 不会构建该扩展。原生后端用于 Linux x86_64 CPU 推理。

## 构建

LIBXSMM 源码固定为 `0d1f61ed425aae8318fffafeea9834971a33aec7`，对应参考 af3_kernels 的提交 `40d3b7dfb8d2328c96e2f05863e3e0dfcd654301`。`libxsmm.lock.json` 保存来源、版本和 SHA-256；仓库包含构建所需文件及许可证，不包含样例数据、Git 元数据或预编译库。

在目标 CPU PyTorch 环境中，从 xfold 根目录执行：

```bash
MAX_JOBS=8 CC=gcc CXX=g++ \
  python -m pip install -v --no-build-isolation --no-deps ./af3_kernels
```

构建需要 GNU Make、setuptools、wheel 和支持 OpenMP 的编译器。参考环境为 Python 3.11、PyTorch 2.6.0+cpu、GCC/G++ 13.3。`--no-build-isolation` 使用当前 PyTorch 的头文件和 ABI，`--no-deps` 避免构建时替换依赖。

构建流程先校验源码，再将清单文件复制到独立目录，生成静态 PIC 库 `libxsmm.a`、`libxsmmnoblas.a` 并链接到 `af3_kernels._C`。扩展编译和链接均启用 OpenMP。使用 `-march=native`，因此更换 CPU 架构或 PyTorch 后需重新构建。

`LIBXSMM_ROOT` 可指定其他源码目录，但文件需与固定清单一致，构建仍在副本中进行。默认源码构建无需网络。Python 包导入不调用编译器。

## 基础接口

```python
import af3_kernels

print(af3_kernels.is_available())
print(af3_kernels.runtime_info())
```

`runtime_info()` 返回 LIBXSMM 版本和目标架构、PyTorch 版本及 OpenMP 最大线程数。是否实际执行 AMX 指令需进一步检查目标机器生成的内核。

原生接口接受无梯度的 strided CPU FP32/BF16 张量。共轭和 negative-bit 视图应先 resolve，直接接口不处理 autocast 或 autograd；xfold 的桥接层负责数据准备。

| 接口 | 输入和行为 |
| --- | --- |
| `pad_and_align_tensor(x, shape, need_init=True, init=0.0, alignment=64)` | 保持 rank/dtype，只扩展不缩小；已连续且对齐时可返回原张量。关闭初始化后，填充区内容未指定 |
| `vnni_repack_tensor(x, type=2, BLK_SZ=0)` | 非空连续二维 BF16，行数为偶数；type=2 按两行打包。正 BLK_SZ 需整除列数 |
| `gemm(a, b)` | 非空连续二维矩阵，dtype 和内部维度匹配；BF16 的 K 为偶数。累积及输出为 FP32，用于接口检查 |

VNNI 输出是与原矩阵形状相同的打包存储，不能作为普通 PyTorch 矩阵使用。对齐分配采用共享所有权管理存储。

## GLU 与 Linear

```text
gated_linear_unit(x, weight, M_BLK=128, N_BLK=128, weight_packed=False)
linear(x, weight, bias=None, weight_packed=False)
```

GLU 输入为 `[..., K]`，weight 为转置后的 `[K, 2N]`，前 N 列是 SiLU gate，后 N 列是 value projection。输出为 `[..., N]`。Linear 输入为 `[..., K]`，weight 为 `[K, N]`，可选 bias 为 `[N]`。普通输入均需连续且 dtype 匹配，输出使用计算 dtype。

GLU 每个 tile 使用两个 LIBXSMM GEMM，累积为 FP32，SiLU 与乘法在 C++ 中写入输出，减少完整 `[M, 2N]` 中间张量。BF16 按 K 两行打包，奇数 K 先补零。实现处理任意前导维度、空前导维度和 M/N 尾块；K、输出宽度及块尺寸应为正且符合 LIBXSMM 整数范围。

描述符、尾块和每线程 scratch 在 OpenMP 区域前准备。GLU 按选定 dtype 的边界存储中间结果；Linear 在 FP32 累积后加入计算 dtype 的 bias，再舍入输出。实际数值误差仍需与目标 PyTorch 路径对照。

`Transition` 和 `DiffusionTransition` 使用 GLU 桥接层，读取 CPU autocast 状态后转换局部输入及权重副本，原参数保持 FP32。每个模块保留转置/连续化权重缓存，按版本、存储、形状、stride、device 和 dtype 变化失效。无版本计数的权重不复用；通过 `.data` 或外部存储修改参数会绕过版本跟踪，不适用于该缓存。

## Triangle Multiplication

```text
triangle_multiplication(pair, mask, outgoing,
    left_weight, left_bias, projection, gate,
    center_weight, center_bias, output_weight, gating_weight,
    left_eps=1e-5, center_eps=1e-5, block_size=128,
    weights_packed=False)
```

pair 和返回的更新量均为 `[N, N, C]`。mask 可省略、为标量、`[N]` 或可广播到 `[N, N]` 的张量，按乘法作用于 projection。四组转置权重分别为 `[C, 2C]`、`[C, 2C]`、`[C, C]` 和 `[C, C]`，使用选定计算 dtype；归一化向量保留兼容 dtype，bias 可选。

LIBXSMM 完成投影及通道收缩，ATen 完成 LayerNorm 和 sigmoid。相邻投影通道构成 A/B；outgoing 计算 `A @ B.T`，incoming 计算 `B.T @ A`。接口返回更新量，残差由 Pairformer 加入。输出投影前释放不再使用的 projection/gate/contraction 缓冲区。

## Attention

```text
dot_product_attention(q, k, v, mask=None, bias=None,
    query_block=128, q_pre_scaled=False)
```

Q/K/V 分别为 `[B, H, Q, D]`、`[B, H, K, D]`、`[B, H, K, Dv]`，需连续且 FP32/BF16 dtype 匹配。Q/K 长度及 D/Dv 可不同。bias 为 FP32/BF16，可广播至 `[B, H, Q, K]`。一维 mask 表示 `[K]`，二维表示 `[B, K]`，转换为 bool；其余最高四维 mask 应为可广播的 bool，True 表示有效 key。

原生实现按 query 分块计算完整 key 维度上的 softmax，LIBXSMM 执行 QK/PV，max/exp/sum 使用 FP32。该实现不是 online softmax。scratch 约随 `线程数 × query_block × K` 增长，打包缓冲区和原模型 bias 存储另计。

桥接层先缩放 Q 再转换 dtype，并传入 `q_pre_scaled=True`，保持原计算顺序。bias 后应用有限值 `-1e9` mask，全掩码行保留原实现对真实 key 的均匀 softmax；NaN/Inf 不替换为零。BF16 奇数 D/K 为 VNNI2 补齐，PV 内部补位概率为零。零长度 key 返回零输出，空 batch/head/query 返回空输出。

`GridSelfAttention` 接入原生 bias/Q/K/V/gating/output 投影和 Attention，归一化及转置继续使用 PyTorch。扩散 `SelfAttention` 接入 Q（含 bias）/K/V/gating 投影和 Attention，自适应归一化及输出条件化继续使用 PyTorch。MSA 和 CrossAttention 保留原实现。

## 后端与缓存

`--glu_backend`、`--triangle_backend` 和 `--attention_backend` 分别选择 `auto|torch|cpp`。auto 保留 CPU PyTorch 和原 CUDA 选择；cpp 在模型构建前检查扩展及所需符号，失败时报告错误。参数名称和 checkpoint 布局保留。

| `--cache_mode` | 缓存内容 |
| --- | --- |
| `off` | 关闭新增缓存，保留既有连续化权重缓存 |
| `weights` | 保留原生 BF16 GLU/Linear/Triangle 的 VNNI2 权重 |
| `inputs` | 单次模型调用内的可复用静态特征 |
| `all` | 同时启用以上两类新增缓存 |

0.4.0 扩展提供 `_C.packed_weight_api == 1`。GLU/Linear 的 `weight_packed` 和 Triangle 的 `weights_packed` 默认 False；True 表示全宽 VNNI2 存储，原始 K 补至偶数，形状为 `[K_padded, N]`，GLU 为 `[K_padded, 2N]`。预打包权重跳过重复转换，Triangle 的动态收缩操作数仍按普通矩阵处理。

输入缓存通过 `InputCache` 和 `input_cache_scope()` 实现。每次 AlphaFold3 调用开启新作用域，退出及异常时清理输入数据；recycle 与 diffusion 阶段切换时释放对应缓存，权重缓存可继续保留。独立模块测试需在 eval、`torch.inference_mode()` 和显式作用域下运行。

缓存内容包括 recycle 初始 embedding/mask、relative/bond 特征、扩散静态条件、transformer pair logits 和 atom encoder 静态条件及布局转换。动态坐标、噪声级 embedding、噪声条件化 single transition、动态 Q/K、MSA 随机采样和 recycle 更新继续计算。可变初始激活在更新前复制。

key 包含输入作用域、依赖对象身份及版本/布局、参数版本、LayerNorm eps/shape、head 布局、精度和后端设置。无版本计数的 inference 特征按作用域内不可变处理，修改前需重置缓存；无版本参数不复用。training、启用梯度或未进入作用域的调用跳过输入缓存。

`--input_cache_max_mb` 默认 512 MiB，0 关闭保留。计费按底层 storage，单条记录中的别名去重，跨条目保守计费；超额条目重新计算。该限额不包含模型、权重副本、scratch 和瞬时分配，不能作为 RSS 上限。

## 验证进展与集群复现

本机完成源码/接口检查、六个编译单元的语法检查，以及 PyTorch CPU 模块和缓存测试。原生扩展未在 Linux 上链接或执行，GLU/GEMM/VNNI 的原生数值及 AMX 性能待验证，条件见 [测试记录](../docs/executable_validation.md)。

集群复现按以下顺序进行：

1. 在目标环境重新构建 0.4.0，核对 `runtime_info()`、算子符号和 packed ABI 标记。
2. 固定权重和输入，分别比较原生 GLU、Linear、Triangle 两方向和 Attention 与 PyTorch 的 FP32/BF16 输出，记录绝对、相对误差。
3. 覆盖奇数通道、空维度、块尺寸以下/等于/以上及不能整除的尺寸；Attention 增加不同 D/Dv、Q/K 长度、广播 mask/bias、全掩码、NaN/Inf 情况。
4. 比较普通和扩散 Transition、Grid 两个方向及带条件的 SelfAttention，确认残差仅加入一次。重载和修改权重、切换精度/后端，核对缓存失效。
5. 对 `off/weights/inputs/all` 比较复用模型 A→B→A、多种子、相同长度不同输入，以及容量不足和异常清理。先分别启用算子及缓存，再组合运行完整预测。
6. 分别记录冷启动、预热后延迟及峰值内存，检查生成内核中的 AMX 指令后再分析性能变化。

## 来源与许可证

padding/VNNI 接口及分块计算思路参考 BSD 许可的 af3_kernels，GLU、Triangle 和 Attention 按 LIBXSMM 公开 API 与 xfold 的计算语义实现。第三方声明保留于 [LICENSE](LICENSE)。LIBXSMM 使用其 [BSD-3-Clause 许可证](third_party/libxsmm/LICENSE.md)，二进制包附带副本 `src/af3_kernels/licenses/LIBXSMM.txt`。
