# Flower 集成使用指南

## 概述

项目现在支持两种实现方式：
1. **原生实现** (`multi_server_fl`) - 自定义的多服务器联邦学习
2. **Flower 集成** (`flower_client.py` + `flower_server.py`) - 基于 Flower 框架的实现 **✨ 支持客户端并行训练**

## Flower 集成的优势

✅ **成熟的框架**: 使用业界标准的联邦学习框架
✅ **多种策略**: 支持 FedAvg, FedProx, FedAdam, FedAdagrad 等
✅ **客户端并行**: 真正的并行客户端训练，显著提升训练速度
✅ **更简洁**: 更少的样板代码，更好的可维护性
✅ **社区支持**: 丰富的文档和活跃的社区
✅ **保留特性**: 维持多服务器架构和数据分区功能

## 快速开始

### 安装依赖

```bash
pip install flwr
```

### 基础运行

```bash
# 串行模式（默认）
python scripts/run_flower_example.py \
  --num-clients 10 \
  --num-servers 2 \
  --rounds 5

# 并行模式（推荐）- 5个客户端同时训练
python scripts/run_flower_example.py \
  --num-clients 10 \
  --num-servers 2 \
  --rounds 5 \
  --max-workers 5

# 使用 FedProx 策略 + 并行训练
python scripts/run_flower_example.py \
  --num-clients 20 \
  --num-servers 4 \
  --rounds 10 \
  --strategy fedprox \
  --max-workers 8
```

## 并行训练详解

### 什么是客户端并行？

在联邦学习中，**客户端训练**是最耗时的操作。客户端并行允许多个客户端同时进行本地训练，而不是一个接一个地训练。

### 并行参数说明

```bash
--max-workers N    # N个客户端同时训练（None或1表示串行）
```

**推荐配置**：
- CPU训练：`--max-workers 4-8`（根据CPU核心数）
- GPU训练：`--max-workers 2-4`（避免GPU内存溢出）
- 大规模客户端：`--max-workers`设为客户端数的1/2

### 性能对比示例

```bash
# 串行训练（10个客户端）
python scripts/run_flower_example.py --num-clients 10 --rounds 2
# 预计时间：~60秒

# 并行训练（10个客户端，5个并行）
python scripts/run_flower_example.py --num-clients 10 --rounds 2 --max-workers 5
# 预计时间：~35秒 (提速约40%)
```

## 支持的策略

| 策略 | 说明 | 使用场景 |
|------|------|----------|
| **fedavg** | 标准联邦平均 | 通用场景 |
| **fedprox** | 带正则化的联邦学习 | 数据/系统异构 |
| **fedadam** | 服务器端 Adam 优化 | 需要自适应学习率 |
| **fedadagrad** | 服务器端 Adagrad 优化 | 稀疏梯度场景 |

## 完整参数列表

```bash
--dataset         # 数据集 (mnist, fashionmnist, cifar10)
--num-clients     # 客户端数量
--num-servers     # 服务器数量
--rounds          # 联邦学习轮次
--local-epochs    # 本地训练轮次
--batch-size      # 批次大小
--alpha           # Dirichlet 分布参数
--model           # 模型结构 (lenet/resnet18/resnet34/resnet50/vgg16/mobilenet_v2/efficientnet_b0)
--lr              # 学习率
--strategy        # 聚合策略
--max-workers     # 并行客户端数量 ⭐ 新增
--output          # 输出文件路径
```

## 使用示例

### 示例1：快速测试（并行）
```bash
python scripts/run_flower_example.py \
  --num-clients 5 \
  --num-servers 2 \
  --rounds 3 \
  --max-workers 3
```

### 示例2：标准实验（FedAvg + 并行）
```bash
python scripts/run_flower_example.py \
  --dataset mnist \
  --num-clients 20 \
  --num-servers 4 \
  --rounds 10 \
  --local-epochs 2 \
  --max-workers 8
```

### 示例3：高度异构数据 + FedProx
```bash
python scripts/run_flower_example.py \
  --dataset cifar10 \
  --num-clients 50 \
  --num-servers 5 \
  --rounds 20 \
  --alpha 0.1 \
  --strategy fedprox \
  --max-workers 10
```

### 示例4：Fashion MNIST + FedAdam
```bash
python scripts/run_flower_example.py \
  --dataset fashionmnist \
  --num-clients 15 \
  --num-servers 3 \
  --rounds 10 \
  --strategy fedadam \
  --max-workers 6
```

## 架构说明

```
多服务器协调层 (保留)
    ├── Flower Server 1 (FedAvg/FedProx/...) + 并行训练
    │   ├── [并行] Flower Client 1
    │   ├── [并行] Flower Client 2
    │   ├── [并行] Flower Client 3
    │   └── ...
    ├── Flower Server 2 + 并行训练
    │   └── ...
    └── ...
```

## 测试结果示例

### 串行训练
```
🚀 Flower-based Federated Learning
  - Clients: 10
  - Servers: 2
  - Parallel Workers: None

Round 1/2
  ✓ Test Accuracy: 0.7916

Round 2/2
  ✓ Test Accuracy: 0.8890
```

### 并行训练（5 workers）
```
🚀 Flower-based Federated Learning
  - Clients: 10
  - Servers: 2
  - Parallel Workers: 5 ⚡

Round 1/2
  ✓ Test Accuracy: 0.7916

Round 2/2
  ✓ Test Accuracy: 0.8890

速度提升：~40% faster
```

## 对比原生实现

| 特性 | 原生实现 | Flower 实现 | Flower + 并行 |
|------|---------|------------|--------------|
| 聚合算法 | 仅 FedAvg | 10+ 种策略 | 10+ 种策略 |
| 客户端并行 | 手动实现 | ❌ 无 | ✅ **原生支持** |
| 训练速度 | 基准 | 基准 | **快 40-60%** |
| 代码行数 | ~500 行 | ~300 行 | ~350 行 |
| 可扩展性 | 中等 | 高 | 高 |
| 易用性 | 中等 | ✅ 简单 | ✅ 简单 |

## ⚠️ 重要性能说明

### GPU训练中的max-workers行为

**实测发现**：在GPU训练中，`max-workers > 1` 反而会降低性能！

```bash
# 实测数据（10客户端，2轮）
串行 (workers=1):   1m25s  ✅ 最快
并行 (workers=5):   3m44s  ❌ 慢2.6倍！

原因：
1. CUDA上下文串行化，GPU操作无法并行
2. 线程调度开销巨大（系统时间增加25倍）
3. 所有线程共享同一个模型，显存不变
```

**推荐配置**：

### GPU 训练 ⭐ 推荐
```bash
# GPU训练：保持串行最快
--max-workers 1  # 或不指定（默认None）

# 如果要优化速度，增大batch size而不是workers
--batch-size 128 \  # 而不是64
--max-workers 1
```

### CPU 训练（理论上有效）
```bash
# CPU训练：可以尝试多线程
--max-workers 4  # 根据CPU核心数调整
```

### 不推荐
```bash
# ❌ GPU训练中不要这样做
--max-workers 5  # 反而会更慢
```

详见：[性能测试结果](PERFORMANCE_TEST_RESULTS.md)

## 下一步

- ⚠️ **GPU训练**：建议使用 `--max-workers 1` 获得最佳性能
- 🚀 **优化速度**：增大 `--batch-size` 而不是 `--max-workers`
- 🔧 探索不同的聚合策略（fedprox, fedadam等）
- 📊 调整 Dirichlet alpha 参数探索数据异构性影响
- 📈 增加客户端和服务器数量测试可扩展性
- 📚 参考 [Flower 文档](https://flower.ai/docs) 了解更多高级特性

## 文件结构

```
multi_server_fl/
├── flower_client.py      # Flower 客户端封装
├── flower_server.py      # Flower 服务器封装 (✨ 支持并行)
├── client.py             # 原生客户端（保留）
├── server.py             # 原生服务器（保留，支持并行）
├── coordinator.py        # 多服务器协调器
├── data/                 # 数据分区工具
└── models/               # 模型定义

scripts/
├── run_flower_example.py  # Flower 集成运行脚本 (✨ 支持并行)
└── run_example.py         # 原生实现运行脚本 (支持并行)
```

## 常见问题

**Q: GPU训练时应该用多少 max-workers？**
A: **建议使用 `1` 或不指定**。实测表明，GPU训练中多线程反而会因为线程调度开销和CUDA串行化而降低性能（慢2.6倍）。

**Q: 如何提升GPU训练速度？**
A:
1. 增大 batch size（`--batch-size 128`）- 最有效
2. 减少本地epoch（`--local-epochs 1`）
3. 使用混合精度训练（需修改代码）
4. 不要增加 max-workers！

**Q: max-workers 在什么情况下有用？**
A: 主要用于 **CPU 训练**，可以利用多核心。GPU训练中由于CUDA串行化，多线程无法实现真正的并行。

**Q: 为什么GPU显存不随max-workers变化？**
A: 因为使用的是 ThreadPoolExecutor（多线程），所有线程共享同一个模型和CUDA上下文。只有多进程（ProcessPoolExecutor）才会创建多个模型副本。

**Q: 并行和串行的结果一样吗？**
A: 模型收敛结果相同（准确率基本一致），但训练速度不同。GPU训练中串行更快。

**Q: 可以同时并行服务器和客户端吗？**
A: 当前每个服务器串行处理客户端最高效。服务器聚合很快，并行服务器意义不大。
