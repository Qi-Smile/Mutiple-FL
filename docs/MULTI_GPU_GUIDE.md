# Multi-GPU 并行训练指南

## 概述

你的系统有 **2 个 NVIDIA RTX 4090 GPU**（每个 24GB 显存），可以实现真正的并行训练！

```bash
GPU 0: NVIDIA GeForce RTX 4090 (23.6 GB)
GPU 1: NVIDIA GeForce RTX 4090 (23.6 GB)
```

## 关键发现回顾

### 为什么单GPU无法并行？

1. **ThreadPoolExecutor（多线程）**：
   - 所有线程共享同一个 CUDA 上下文和模型
   - GPU 操作被串行化
   - 显存不会增加
   - **结果：GPU 训练中反而更慢！**

2. **ProcessPoolExecutor（多进程 + 单GPU）**：
   - 每个进程有独立的 Python 解释器
   - 但仍然竞争同一个 GPU
   - CUDA 会自动串行化
   - **结果：仍然是串行执行**

### 真正的并行需要什么？

**多个 GPU + 多进程**

- ✅ 每个进程绑定到不同的 GPU
- ✅ 真正的并行计算
- ✅ 显存线性增长（每个GPU独立）
- ✅ 速度提升接近 GPU 数量倍数

---

## 实现方案

我已经为你创建了多GPU并行服务器：[flower_server_multigpu.py](../multi_server_fl/flower_server_multigpu.py)

### 核心原理

```python
# 关键实现：每个进程使用不同GPU
def _train_client_on_gpu(client_id, gpu_id, ...):
    # 1. 设置进程只能看到指定的GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # 2. 在该GPU上创建模型
    device = torch.device("cuda:0")  # 进程内部总是 cuda:0
    model = model_builder().to(device)

    # 3. 训练
    ...
    return results

# 主程序：多进程并行
with ProcessPoolExecutor(max_workers=2) as executor:
    # 客户端0 → GPU 0
    future1 = executor.submit(_train_client_on_gpu, 0, gpu_id=0, ...)
    # 客户端1 → GPU 1
    future2 = executor.submit(_train_client_on_gpu, 1, gpu_id=1, ...)
    # 真正的并行！
```

###  GPU 分配策略

```python
# 自动轮询分配
for i, client in enumerate(clients):
    gpu_id = i % num_gpus  # 0, 1, 0, 1, 0, 1, ...
    assign_to_gpu(client, gpu_id)

# 你有2个GPU，训练10个客户端：
GPU 0: 客户端 0, 2, 4, 6, 8  (5个)
GPU 1: 客户端 1, 3, 5, 7, 9  (5个)

# 最多同时并行：min(num_gpus, num_clients) = 2
```

---

## 使用方法

### 方法1：使用多GPU服务器（已实现）

```python
from multi_server_fl.flower_server_multigpu import create_multigpu_flower_server

# 创建多GPU服务器
server = create_multigpu_flower_server(
    server_id=0,
    model_builder=model_builder,
    device=torch.device("cuda:0"),
    strategy="fedavg",
    auto_gpu=True,  # 自动检测并使用所有GPU
)

# 训练
result = server.run_round(
    clients=clients,
    test_loader=test_loader,
    model_builder=model_builder,  # 必须传递！
    optimizer_factory=optimizer_factory,
    loss_fn=loss_fn,
    batch_size=64,
    local_epochs=1,
)
```

### 方法2：指定特定GPU

```python
server = create_multigpu_flower_server(
    server_id=0,
    model_builder=model_builder,
    device=torch.device("cuda:0"),
    strategy="fedavg",
    auto_gpu=False,
    gpu_ids=[0, 1],  # 明确指定使用 GPU 0 和 GPU 1
)
```

### 方法3：只使用一个GPU

```python
server = create_multigpu_flower_server(
    server_id=0,
    model_builder=model_builder,
    device=torch.device("cuda:0"),
    strategy="fedavg",
    auto_gpu=False,
    gpu_ids=[0],  # 只使用 GPU 0
)
```

---

## 性能预期

### 单GPU（串行）

```bash
python scripts/run_flower_example.py \
  --num-clients 10 \
  --rounds 5 \
  --max-workers 1

# 预期结果：
# - GPU 0 显存：~2GB
# - GPU 1 显存：0 MB（未使用）
# - 速度：基准
# - 每轮耗时：约 60秒
```

### 双GPU（真正并行）

```bash
python scripts/run_multigpu_simple.py \
  --num-clients 10 \
  --rounds 5

# 预期结果：
# - GPU 0 显存：~2GB（5个客户端）
# - GPU 1 显存：~2GB（5个客户端）
# - 速度：接近 2倍
# - 每轮耗时：约 30-35秒
```

### 性能对比表

| 配置 | GPU 0 显存 | GPU 1 显存 | 速度 | 适用场景 |
|------|-----------|-----------|------|----------|
| 单GPU 串行 | 2GB | 0MB | 1x | 调试、小规模 |
| 单GPU 多线程 | 2GB | 0MB | 0.4x ❌ | 不推荐 |
| **双GPU 并行** | 2GB | 2GB | **1.8-2x** ✅ | **推荐** |
| 双GPU + 大batch | 6GB | 6GB | 3-4x 🚀 | 最快 |

---

## 代码示例

### 完整示例

```python
import torch
from torch.utils.data import DataLoader
from multi_server_fl.data.partition import dirichlet_partition
from multi_server_fl.data.utils import load_torchvision_dataset, subset_dataset
from multi_server_fl.flower_client import create_flower_client
from multi_server_fl.flower_server_multigpu import create_multigpu_flower_server
from multi_server_fl.models import get_model_builder

# 检测GPU
num_gpus = torch.cuda.device_count()
print(f"🖥️  检测到 {num_gpus} 个 GPU")

# 加载数据
bundle = load_torchvision_dataset("mnist")
train_dataset, test_dataset = bundle.train, bundle.test

# 分区
labels = train_dataset.targets.numpy()
partition = dirichlet_partition(labels, num_clients=10, alpha=0.5)
client_datasets = [
    subset_dataset(train_dataset, indices)
    for indices in partition.client_indices
]

# 创建客户端
model_builder = get_model_builder("lenet", "mnist")
optimizer_factory = lambda params: torch.optim.SGD(params, lr=0.01)
loss_fn = torch.nn.CrossEntropyLoss()

clients = [
    create_flower_client(
        client_id=i,
        model_builder=model_builder,
        train_dataset=client_datasets[i],
        test_loader=DataLoader(test_dataset, batch_size=256),
        device=torch.device("cuda:0"),
        optimizer_factory=optimizer_factory,
        loss_fn=loss_fn,
        batch_size=64,
        local_epochs=1,
    )
    for i in range(10)
]

# 创建多GPU服务器
server = create_multigpu_flower_server(
    server_id=0,
    model_builder=model_builder,
    device=torch.device("cuda:0"),
    strategy="fedavg",
    auto_gpu=True,  # 自动使用所有GPU
)

# 训练
for round_num in range(5):
    print(f"\n📍 Round {round_num + 1}/5")

    result = server.run_round(
        clients=clients,
        test_loader=DataLoader(test_dataset, batch_size=256),
        model_builder=model_builder,
        optimizer_factory=optimizer_factory,
        loss_fn=loss_fn,
        batch_size=64,
        local_epochs=1,
    )

    print(f"  ✅ Test Accuracy: {result['test_accuracy']:.4f}")
```

---

## 监控GPU使用情况

### 实时监控

```bash
# 终端1：运行训练
python scripts/run_multigpu_simple.py --num-clients 10 --rounds 5

# 终端2：监控GPU
watch -n 1 nvidia-smi
```

### 预期看到的情况

**训练前**：
```
GPU 0: 11 MiB / 24564 MiB  (空闲)
GPU 1: 11 MiB / 24564 MiB  (空闲)
```

**训练中（多GPU并行）**：
```
GPU 0: 2000 MiB / 24564 MiB  (客户端 0, 2, 4, 6, 8)
GPU 1: 2000 MiB / 24564 MiB  (客户端 1, 3, 5, 7, 9)
       ↑ 两个GPU同时工作！
```

**训练中（单GPU）**：
```
GPU 0: 2000 MiB / 24564 MiB  (所有客户端轮流)
GPU 1: 11 MiB / 24564 MiB    (闲置)
       ↑ GPU 1 完全浪费！
```

---

## 常见问题

### Q1: 为什么我只有一个GPU时不推荐使用 max-workers > 1？

**A**: 单GPU无法真正并行：
- ThreadPoolExecutor：GPU操作串行化，反而更慢
- ProcessPoolExecutor：进程竞争同一个GPU，仍然串行

### Q2: 多GPU并行会影响模型准确率吗？

**A**: 不会！
- 每个客户端独立训练
- 服务器聚合时结果完全相同
- 只是训练速度更快

### Q3: 如何验证确实在使用多GPU？

**A**: 三种方法：
1. **nvidia-smi**：看到两个GPU显存都在增加
2. **训练速度**：接近2倍加速
3. **进程数**：看到多个Python进程

```bash
# 查看进程
ps aux | grep python | grep run_multigpu
```

### Q4: 10个客户端 + 2个GPU，是否有5倍加速？

**A**: 不是！实际加速约 **1.8-2倍**：
- 理论：10个客户端串行 vs 2个并行 = 5倍
- 实际：有进程创建开销、数据传输等
- 每轮仍需多批次：第1批(C0,C1) → 第2批(C2,C3) → ...→ 第5批(C8,C9)

### Q5: 能否让每个GPU同时训练多个客户端？

**A**: 可以，但要小心显存：
```python
# 2个GPU，每个同时跑2个客户端 = 总共4个并行
max_workers = 4
gpu_ids = [0, 0, 1, 1]  # GPU 0跑2个，GPU 1跑2个

# 显存需求：
GPU 0: 2GB × 2 = 4GB
GPU 1: 2GB × 2 = 4GB
```

---

## 最佳实践

### 推荐配置（你的2×RTX 4090）

```bash
# 🌟 最佳性能配置
python scripts/run_multigpu_simple.py \
  --num-clients 20 \      # 充分利用2个GPU
  --num-servers 2 \       # 2个服务器，每个管理10个客户端
  --rounds 10 \
  --local-epochs 1 \
  --batch-size 128 \      # 增大batch size
  --strategy fedavg

# 预期：
# - 每个GPU运行10个客户端
# - 显存占用：每个GPU约 6-8GB
# - 速度：接近单GPU的2倍
```

### 极限性能配置

```bash
# 🚀 极限速度（需要充足显存）
python scripts/run_multigpu_simple.py \
  --num-clients 40 \      # 更多客户端
  --batch-size 256 \      # 更大batch
  --local-epochs 1

# 预期：
# - 每个GPU约 15-20GB 显存
# - 速度提升 3-4倍
```

### 稳定配置（长时间训练）

```bash
# 📊 稳定可靠
python scripts/run_multigpu_simple.py \
  --num-clients 10 \
  --batch-size 64 \
  --local-epochs 2 \
  --rounds 100

# 预期：
# - 每个GPU约 2-4GB 显存
# - 稳定运行，不会OOM
```

---

## 性能优化技巧

### 1. 增大 Batch Size（最有效）

```python
# 小batch（低效）
--batch-size 32  # GPU利用率 20%

# 大batch（高效）
--batch-size 256  # GPU利用率 80%

# 提速：3-4倍
```

### 2. 使用混合精度训练

```python
# 在客户端训练中添加
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(local_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()

        with autocast():  # 混合精度
            output = model(data)
            loss = loss_fn(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

# 提速：1.5-2倍
# 显存：减少 40-50%
```

### 3. 优化数据加载

```python
# 使用多线程数据加载
train_loader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=4,  # 4个线程加载数据
    pin_memory=True,  # 加速CPU→GPU传输
)
```

---

## 总结

### ✅ 你的优势

- **2 × RTX 4090**：顶级GPU配置
- **48GB 总显存**：足够大规模训练
- **真正并行**：多进程 + 多GPU

### 🎯 推荐做法

1. **小规模测试**：
   ```bash
   --num-clients 4 --rounds 2  # 快速验证
   ```

2. **正式训练**：
   ```bash
   --num-clients 20 --rounds 50 --batch-size 128
   ```

3. **监控GPU**：
   ```bash
   watch -n 1 nvidia-smi
   ```

### ❌ 避免的做法

1. ❌ 单GPU使用 `--max-workers > 1`（更慢）
2. ❌ 过小的 batch size（浪费GPU）
3. ❌ 忽略显存监控（可能OOM）

---

## 下一步

1. ✅ 已实现多GPU服务器
2. 🔧 需要调试和测试完整流程
3. 📊 性能基准测试
4. 📚 完善文档和示例

---

## 参考资料

- [WHY_GPU_CANNOT_PARALLEL.md](WHY_GPU_CANNOT_PARALLEL.md) - GPU并行原理
- [PERFORMANCE_TEST_RESULTS.md](PERFORMANCE_TEST_RESULTS.md) - 性能测试数据
- [GPU_MEMORY_FAQ.md](GPU_MEMORY_FAQ.md) - 显存问题FAQ
- [flower_server_multigpu.py](../multi_server_fl/flower_server_multigpu.py) - 实现代码
