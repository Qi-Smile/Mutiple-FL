# Multi-Server Federated Learning

一个基于 PyTorch 和 Flower 的多服务器联邦学习框架，支持真正的多GPU并行训练。

## ✨ 特性

- ✅ **多服务器架构**：支持多个参数服务器并行聚合
- ✅ **Flower集成**：集成Flower框架，支持10+种聚合策略  
- ✅ **真正的多GPU并行**：使用多进程实现真正的多GPU并行训练
- ✅ **Non-IID数据**：支持Dirichlet分布的数据异构性模拟
- ✅ **SwanLab集成**：实验跟踪和可视化
- ✅ **按需加载模型**：客户端模型只在训练时迁移到指定GPU（`--device`），降低大规模场景的显存压力

## 🖥️ GPU支持

### 单GPU训练（推荐用于调试）
```bash
python scripts/run_flower_example.py \
  --num-clients 10 \
  --rounds 10 \
  --batch-size 128 \
  --max-workers 1
```

### 多GPU并行训练（生产环境推荐）
```bash
python scripts/run_multigpu_simple.py \
  --num-clients 20 \
  --rounds 10 \
  --batch-size 128
```

**性能对比**：
- 单GPU: 1x速度
- 双GPU: ~2x速度  
- 需要: 2× GPU显存

详见：[docs/MULTI_GPU_GUIDE.md](docs/MULTI_GPU_GUIDE.md)

## 📦 安装

```bash
# 1. 克隆项目
git clone <repository-url>
cd Mutiple-FL

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装项目（可编辑模式）
pip install -e .
```

## 🚀 快速开始

### 1. 原生多服务器FL

```bash
python scripts/run_example.py \
  --dataset cifar10 \
  --num-clients 10 \
  --num-servers 2 \
  --rounds 5
```

### 2. Flower集成版本

```bash
python scripts/run_flower_example.py \
  --dataset mnist \
  --num-clients 10 \
  --num-servers 2 \
  --rounds 5 \
  --strategy fedavg
```

### 3. 多GPU并行训练

```bash
# 自动使用所有GPU
python scripts/run_multigpu_simple.py \
  --num-clients 20 \
  --rounds 10

# 指定特定GPU
python scripts/run_multigpu_simple.py \
  --gpu-ids 0 1 \
  --num-clients 20
```

## 📊 项目结构

```
Mutiple-FL/
├── multi_server_fl/          # 核心代码
│   ├── client.py             # 原生客户端实现
│   ├── server.py             # 原生服务器实现
│   ├── coordinator.py        # 协调器
│   ├── flower_client.py      # Flower客户端包装
│   ├── flower_server.py      # Flower服务器（单GPU）
│   ├── flower_server_multigpu.py  # ⭐ Flower服务器（多GPU）
│   ├── data/                 # 数据处理
│   ├── models/               # 模型定义
│   └── utils.py              # 工具函数
│
├── scripts/                  # 运行脚本
│   ├── run_example.py        # 原生实现示例
│   ├── run_flower_example.py # Flower实现示例
│   └── run_multigpu_simple.py # ⭐ 多GPU训练示例
│
├── docs/                     # 文档
│   ├── FLOWER_GUIDE.md       # Flower使用指南
│   ├── MULTI_GPU_GUIDE.md    # ⭐ 多GPU完整指南
│   ├── MULTI_GPU_SUMMARY.md  # 多GPU快速总结
│   └── WHY_GPU_CANNOT_PARALLEL.md  # GPU并行原理
│
├── data/                     # 数据集（自动下载）
├── setup.py                  # 包配置
└── requirements.txt          # 依赖列表
```

## 📚 文档

- [Flower集成指南](docs/FLOWER_GUIDE.md) - Flower框架使用说明
- [多GPU训练指南](docs/MULTI_GPU_GUIDE.md) - 多GPU并行训练完整教程
- [GPU并行原理](docs/WHY_GPU_CANNOT_PARALLEL.md) - 为什么单GPU无法并行
- [多GPU快速总结](docs/MULTI_GPU_SUMMARY.md) - 多GPU训练快速上手

## 🎯 核心概念

### 多GPU并行策略

**单GPU问题**：
- ❌ ThreadPoolExecutor：假并行，反而更慢
- ❌ ProcessPoolExecutor + 单GPU：仍然串行

**多GPU解决方案**：
- ✅ ProcessPoolExecutor + 多GPU：真正并行
- ✅ 每个进程绑定独立GPU
- ✅ 速度提升接近GPU数量倍数

详见：[docs/WHY_GPU_CANNOT_PARALLEL.md](docs/WHY_GPU_CANNOT_PARALLEL.md)

## ⚙️ 配置选项

### 通用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | cifar10 | 数据集 (mnist/cifar10) |
| `--device` | auto | 计算设备 (`cpu` / `cuda`, 可指定 `cuda:1`) |
| `--num-clients` | 10 | 客户端数量 |
| `--num-servers` | 2 | 服务器数量 |
| `--rounds` | 5 | 全局训练轮次 |
| `--local-epochs` | 2 | 本地训练轮次 |
| `--batch-size` | 32 | 批次大小 |
| `--alpha` | 0.5 | Dirichlet参数 |
| `--lr` | 0.01 | 学习率 |
| `--optimizer` | sgd | 客户端优化器 (sgd/adam/adamw) |
| `--momentum` | 0.9 | SGD动量（仅在 `sgd` 下使用） |
| `--weight-decay` | 0.0 | L2权重衰减 |
| `--defense` | ours | 服务器聚合/防御策略（`ours`/`fedavg`/`krum`/`median`/`fltrust`/`dnc`/`clipped`/`signguard`） |
| `--krum-byzantine-ratio` | 同客户端比例 | 覆盖传递给 Krum 的拜占庭占比（缺省使用 `--malicious-client-ratio`） |
| `--fltrust-root-percent` | 0.01 | FLTrust 根数据集采样比例（占训练集百分比，至少 1 个样本） |
| `--dnc-num-clusters` | 2 | DnC 聚类数量 |
| `--clipped-num-clusters` | 2 | Clipped Clustering 聚类数量 |
| `--clipped-threshold` | auto | Clipped Clustering 裁剪阈值（`auto` 或显式浮点数） |

> ℹ️ 六种传统拜占庭防御（Krum、Median、FLTrust、DnC、Clipped Clustering、SignGuard）的算法细节与调参建议见 [DEFENSE_STRATEGIES_DOCUMENTATION.md](DEFENSE_STRATEGIES_DOCUMENTATION.md)。在运行 `scripts/run_example.py` 时，通过 `--defense` 及相关超参数即可一键切换基线。

### Flower特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--strategy` | fedavg | 聚合策略 (fedavg/fedprox/fedadam...) |
| `--max-workers` | None | 并行worker数（不推荐在单GPU使用） |

### 多GPU特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpu-ids` | None | 指定GPU ID (如: --gpu-ids 0 1) |
| `--auto-gpu` | True | 自动使用所有GPU |

## 📈 性能优化建议

### 单GPU优化
```bash
# 增大batch size（最有效）
--batch-size 128  # 而不是32

# 串行执行
--max-workers 1  # 或不指定
```

### 多GPU优化
```bash
# 充分利用GPU数量
--num-clients 20  # 2个GPU

# 增大batch size
--batch-size 256
```

## 🔍 监控GPU使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 查看GPU属性
nvidia-smi --query-gpu=name,memory.total --format=csv
```

## 📝 实验示例

### 示例1: MNIST基准测试
```bash
python scripts/run_flower_example.py \
  --dataset mnist \
  --num-clients 10 \
  --rounds 10 \
  --batch-size 64
```

### 示例2: 多GPU加速训练
```bash
python scripts/run_multigpu_simple.py \
  --num-clients 40 \
  --rounds 20 \
  --batch-size 128
```

## 📄 许可证

MIT License

## 🙏 致谢

- [Flower](https://flower.ai/) - 联邦学习框架
- [SwanLab](https://swanlab.cn/) - 实验跟踪
- PyTorch - 深度学习框架

---

**提示**：查看 [docs/](docs/) 目录获取详细文档和教程。
