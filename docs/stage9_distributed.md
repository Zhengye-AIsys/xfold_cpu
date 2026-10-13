# CPU 采样与任务并行调度

第九阶段将并行粒度设为完整扩散样本或完整预测任务。参考 xfold_cpu 的相关采样调度固定为 5 个样本、200 步，最多 5 个 rank；当前实现按工作项统一分配，进程数、样本数和扩散步数可独立配置。rank 指进程，不能直接等同于物理节点。

## 计算划分

| 模式 | 分配单位 | 主进程职责 |
| --- | --- | --- |
| `off` | 原有串行推理 | 执行完整预测 |
| `samples` | 同一输入、同一种子的完整扩散轨迹 | 特征准备、主干计算、置信度/距离分布及输出 |
| `tasks` | 完整输入或输入/种子组合 | 输入读取、结果汇总、排名及输出 |

默认模式为 `off`。普通串行路径保留原有随机数序列和后端选择。

采样模式按原始输入和种子顺序运行。rank 0 准备特征并执行主干及 recycle，向其他进程广播特征和 embeddings。各进程完成分配到的样本，随后按一致顺序广播坐标；rank 0 计算输出头、提取结构并写出。每个样本的全部扩散步骤由同一进程顺序执行，空闲进程仍参与同步。

任务模式中，每个进程独立执行分配任务的特征准备、主干、采样和输出头。结果汇总到 rank 0，每个输入按原始种子顺序生成全局排名。`--distributed_task_granularity` 可选 `input`、`seed` 或 `auto`：auto 在多个输入时按输入分配，单个输入时按种子分配；seed 将所有输入/种子组合展开为任务列表。

各进程保存完整模型。该方案属于样本/任务并行，算子内部仍在单个进程上执行，与参考项目的多 rank Triangle/Attention 实现有所不同。进程数超过独立工作数量时会出现空闲；采样模式的主干和输出头仍集中在 rank 0。

## 分配规则与随机数

`--num_diffusion_samples` 和 `--diffusion_steps` 为正整数，默认值分别为 5 和 200。

| `--distributed_schedule` | 分配规则 |
| --- | --- |
| `cyclic` | rank r 处理 r、r+world_size、r+2×world_size 等索引 |
| `block` | 连续均分，余数分配给前面的 rank |
| `weighted` | 按全局 rank 权重的累积比例划分连续区间 |

加权调度通过 `--distributed_rank_weights` 指定每个 rank 的非负整数权重，数量应等于实际进程数，总和大于 0。对总工作量 T，区间边界为 `floor(T × prefix_weight / total_weight)`。零权重不给该 rank 分配计算任务；rank 0 的汇总和采样模式主干职责仍保留。权重由用户配置，任务按同步轮次执行，慢任务可能推迟下一轮。

`--sample_rng=auto|legacy|indexed` 中，auto 在串行和任务模式使用 legacy，在采样模式使用 indexed；采样模式不接受 legacy。indexed 根据原始种子和样本索引，通过 BLAKE2b 生成 CPU 随机流，每个样本独立保存和恢复 CPU RNG 状态。种子生成不依赖 rank、world size 或分配规则，去噪步公式沿用原实现。

采样数值对照应使用串行 `--sample_rng=indexed`，并固定输入、参数、步数、精度和后端。legacy 与 indexed 的随机流不同，不宜直接做逐位比较。任务模式的 auto/legacy 与原串行模式对照。

## 通信与异常处理

通信由 `torch.distributed` 的 Gloo 或可选 ccl 后端执行，默认 Gloo。MPI 在此作为进程启动方式，不是单独的 PyTorch 通信后端。选择 ccl 时加载 `oneccl_bindings_for_pytorch`，要求 Linux x86_64 及匹配的 CPU bindings。单 rank 模式不创建进程组，也不加载 oneCCL。

rank 环境变量按 torchrun、PMI、Open MPI、Slurm 的顺序识别。每组 rank/world-size 变量需完整，重新使用 MPI 启动前应清除残留的 `RANK`、`WORLD_SIZE` 和 `LOCAL_RANK`。多 rank 运行通过 `env://` 初始化，要求各进程可访问 `MASTER_ADDR` 和 `MASTER_PORT`。

控制信息使用 CPU int64，张量按 uint8 字节视图分块广播，保留 FP32/BF16 位表示。对象采用 pickle protocol 5，按 16 MiB 分块传输；仅汇总到主进程的结果由 rank 0 保留和解码，其他接收进程复用单块缓冲区。

`--distributed_max_payload_mb=4096` 限制单个序列化对象，不限制模型、张量或进程总内存。所有进程需使用兼容的 Python/PyTorch/AlphaFold3 类。真实输入和结构结果的序列化应在目标 AlphaFold3 版本上复验。

计算、转换、分配、解码、提取结构和写出阶段在下一次通信前同步 Python 异常。进程组在 finally 中释放，异常退出不增加额外 barrier。进程被杀、通信故障或系统内存耗尽仍可能依赖启动器终止或超时；目前没有断点恢复和动态增减 rank。

`--distributed_timeout_seconds` 默认 3600，需覆盖主干、完整轨迹/任务和输出等最长阶段。模型、输入、输出和调度设置在各 rank 间应一致，线程数和本地绑定可以不同。模型调用继续使用原有输入缓存作用域，各进程缓存独立。

并行运行要求 CPU、启用推理、关闭数据流水线，并提供完成 MSA/template 的输入。输入名称清洗后应唯一，同一输入的种子不能重复，以免输出目录冲突。

## 启动配置

```text
bash scripts/run_distributed.sh [--dry-run] <推理参数>
```

脚本复用 `scripts/run_cpu.sh`，分布式设置通过 `XFOLD_DIST_*` 环境变量配置。`--dry-run` 位于首个参数，只打印命令；正式启动要求 Linux。配置文件赋值覆盖同名环境变量，模板见 `scripts/distributed_env.sh.example`。重复传入 distributed 参数或 flagfile 会被拒绝。

| 环境变量 | 默认值或用途 |
| --- | --- |
| `XFOLD_DIST_LAUNCHER` | `torchrun`；可选 openmpi、intelmpi、external |
| `XFOLD_DIST_MODE` | `samples`；也可选 tasks |
| `XFOLD_DIST_BACKEND` | `gloo`；可选 ccl |
| `XFOLD_DIST_SCHEDULE` | `cyclic`；可选 block、weighted |
| `XFOLD_DIST_RANK_WEIGHTS` | weighted 时填写各 rank 权重 |
| `XFOLD_DIST_TASK_GRANULARITY` | `auto`；可选 input、seed |
| `XFOLD_DIST_NNODES` | `1`，同构启动节点数 |
| `XFOLD_DIST_PPN` | `1`；MPI 仅在显式设置时强制每节点进程数 |
| `XFOLD_DIST_NODE_RANK` | 单节点为 0；多节点 torchrun 需分别设置 |
| `XFOLD_DIST_WORLD_SIZE` | MPI 总进程数，未设置时为 NNODES×PPN |
| `XFOLD_DIST_HOSTFILE` | MPI 对应格式的主机文件 |
| `XFOLD_DIST_MPIRUN` | `mpirun`，可指定实际启动器 |
| `XFOLD_DIST_MPI_BINDING` | `none` 或 inherit |
| `XFOLD_DIST_TIMEOUT_SECONDS` | `3600` |
| `XFOLD_DIST_MAX_PAYLOAD_MB` | `4096` |
| `XFOLD_DIST_ENV_FILE` | 可选 shell 配置文件 |

单机 rendezvous 默认 `127.0.0.1:29500`。多节点或使用主机文件时需指定可达的 MASTER_ADDR；每个作业使用独立端口。

### 单机与同构 torchrun

以下为 8 个进程计算 7 个样本的命令示例，线程数按实际资源调整：

```bash
XFOLD_DIST_PPN=8 XFOLD_DIST_MODE=samples XFOLD_CPU_THREADS=4 \
  bash scripts/run_distributed.sh --dry-run \
    --num_diffusion_samples=7 --cpu_precision=bf16 \
    --json_path=./fold_input_data.json \
    --model_dir="$PATH_TO_AF3_MODEL" --output_dir=./samples_8ranks
```

多节点时，每个节点启动一次脚本，使用相同 NNODES、PPN、MASTER_ADDR 和端口，以及不同 NODE_RANK。torchrun 要求每个节点的本地进程数相同。脚本关闭自动重启，避免重复执行已部分写出的任务；worker 在模型 Python 进程启动前应用 CPU/NUMA 配置。

### 异构 MPI 分配

Open MPI 主机文件可以为不同节点分配不同 slot 数。采用主机文件布局时，应在环境和配置文件中都取消 PPN 设置，并指定总进程数：

```bash
MASTER_ADDR=node01 MASTER_PORT=29500 XFOLD_DIST_LAUNCHER=openmpi \
XFOLD_DIST_WORLD_SIZE=6 XFOLD_DIST_HOSTFILE=/shared/hosts.openmpi \
XFOLD_DIST_MODE=tasks XFOLD_DIST_TASK_GRANULARITY=seed XFOLD_CPU_THREADS=8 \
  bash scripts/run_distributed.sh --dry-run \
    --input_dir=/shared/inputs --model_dir=/shared/models --output_dir=/shared/results
```

Open MPI 使用 `--hostfile`，Intel MPI 使用 `-f`，主机文件分别按对应启动器的格式准备。MPI 默认关闭自身绑核，将 CPU 放置交给调度器或第八阶段配置；已有 MPI 绑定时可选择 inherit。脚本不启用超额分配。

已有 MPI/Slurm 作业可设置 `XFOLD_DIST_LAUNCHER=external`，通过集群的 mpirun/srun 为每个 rank 调用一次脚本，并导出 MASTER_ADDR/MASTER_PORT；external 不再启动第二层启动器。程序、模型和输入路径需在各节点可访问，输出由 rank 0 写入。

## 每进程 CPU 配置

使用 `XFOLD_CPU_ENV_FILE` 指定 worker 配置，在启动器设置 rank 变量后按本地 rank 和主机选择 CPU/NUMA 资源。例如：

```bash
worker_local_rank=${LOCAL_RANK:-${MPI_LOCALRANKID:-${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}}}
# 按实际分配，将 worker_local_rank 映射为 CPU 和内存节点列表。
```

同一主机上的进程通常需要不同 CPU 列表。oneCCL 的通信线程也需纳入核数分配。权重对应启动器实际给出的全局 rank 顺序，不能只按主机名顺序填写。CPU 配置详见 [线程与 NUMA 说明](stage8_cpu.md)。

## 验证进展

本机已执行调度分配矩阵，以及真实 3/6-process Gloo 通信和轻量采样/任务用例。测试数据、版本和限制见 [测试记录](executable_validation.md)。完整预测、MPI/oneCCL 启动、跨节点运行和性能测量待集群完成。

集群上先比较串行 indexed 与 samples，再比较串行 legacy 与 tasks；保持精度、后端、参数、输入和种子一致。重点覆盖工作项不足、不能均分和零权重，核对全局排名、输出目录及所有任务的完成情况。之后验证跨节点 Gloo、匹配的 ccl 环境、真实结构序列化和进程故障退出，再测耗时、吞吐和内存。

## 接口依据

- [PyTorch distributed](https://docs.pytorch.org/docs/stable/distributed)及 [torchrun](https://docs.pytorch.org/docs/stable/elastic/run)。
- [Open MPI 主机调度](https://docs.open-mpi.org/en/main/launching-apps/scheduling.html)与 [Intel MPI 启动说明](https://www.intel.com/content/www/us/en/docs/mpi-library/developer-guide-linux/2021-15/running-an-mpi-program.html)。
- [oneCCL bindings](https://github.com/intel/torch-ccl)项目已标注弃用，使用时需确认匹配版本；[类型与 reduction 映射](https://github.com/intel/torch-ccl/blob/master/src/utils.cpp)。
- [AlphaFold3 Structure 序列化](https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/structure/structure.py)。
