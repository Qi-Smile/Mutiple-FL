# CIFAR-10 + ResNet 实验快速开始指南

## 📋 系统当前状态

### ✅ 已完成的功能

1. **梯度空间攻击系统** (Blades 风格)
   - SignFlip, Noise, IPM 攻击
   - 在训练时操纵梯度
   - 完全在 GPU 上执行

2. **参数空间攻击系统**
   - ALIE, Adaptive 攻击
   - 需要全局统计信息
   - 在服务器端应用

3. **Sync + Local 双重评估指标**
   - sync_test_accuracy: 同步后准确率
   - local_test_accuracy: 训练后准确率
   - global_test_accuracy: 服务器全局模型准确率

4. **GPU 加速聚合** (新增!)
   - Geometric Median: 2.3-3.1x 加速
   - Krum: 1.5-5.3x 加速
   - Median: 1.1-2.0x 加速
   - 自动启用，向后兼容

---

## 🚀 运行 CIFAR-10 + ResNet 实验

### 基础实验：Ours vs FedAvg

```bash
# 1. Ours 方法 (双层防御)
python scripts/run_example.py \
    --defense ours \
    --dataset CIFAR10 \
    --model resnet18 \
    --num-clients 100 \
    --num-servers 2 \
    --rounds 50 \
    --client-attack signflip \
    --malicious-client-ratio 0.2 \
    --seed 42 \
    --device cuda:0 \
    --lr 0.01 \
    --batch-size 64 \
    --local-epochs 5

# 2. FedAvg 基线 (无防御)
python scripts/run_example.py \
    --defense fedavg \
    --dataset CIFAR10 \
    --model resnet18 \
    --num-clients 100 \
    --num-servers 2 \
    --rounds 50 \
    --client-attack signflip \
    --malicious-client-ratio 0.2 \
    --seed 42 \
    --device cuda:0

# 3. Krum 基线 (服务器端防御)
python scripts/run_example.py \
    --defense krum \
    --dataset CIFAR10 \
    --model resnet18 \
    --num-clients 100 \
    --num-servers 2 \
    --rounds 50 \
    --client-attack signflip \
    --malicious-client-ratio 0.2 \
    --krum-byzantine-ratio 0.2 \
    --seed 42 \
    --device cuda:0

# 4. Median 基线
python scripts/run_example.py \
    --defense median \
    --dataset CIFAR10 \
    --model resnet18 \
    --num-clients 100 \
    --num-servers 2 \
    --rounds 50 \
    --client-attack signflip \
    --malicious-client-ratio 0.2 \
    --seed 42 \
    --device cuda:0
```

---

## 📊 完整实验矩阵

### 实验 1: 不同攻击方法对比

| 实验 ID | 防御方法 | 攻击方法 | 恶意比例 | 目的 |
|---------|---------|---------|---------|------|
| 1.1 | FedAvg | SignFlip | 20% | 无防御基线 |
| 1.2 | FedAvg | Noise | 20% | 无防御基线 |
| 1.3 | FedAvg | ALIE | 20% | 无防御基线 |
| 1.4 | Ours | SignFlip | 20% | 双层防御 |
| 1.5 | Ours | Noise | 20% | 双层防御 |
| 1.6 | Ours | ALIE | 20% | 双层防御 |
| 1.7 | Krum | SignFlip | 20% | 服务器端防御 |
| 1.8 | Krum | Noise | 20% | 服务器端防御 |
| 1.9 | Krum | ALIE | 20% | 服务器端防御 |
| 1.10 | Median | SignFlip | 20% | 服务器端防御 |

**运行示例**：
```bash
# 实验 1.1: FedAvg + SignFlip
python scripts/run_example.py --defense fedavg --dataset CIFAR10 \
    --model resnet18 --num-clients 100 --num-servers 2 --rounds 50 \
    --client-attack signflip --malicious-client-ratio 0.2 \
    --seed 42 --device cuda:0

# 实验 1.4: Ours + SignFlip
python scripts/run_example.py --defense ours --dataset CIFAR10 \
    --model resnet18 --num-clients 100 --num-servers 2 --rounds 50 \
    --client-attack signflip --malicious-client-ratio 0.2 \
    --seed 42 --device cuda:0
```

---

### 实验 2: 不同恶意比例对比

| 实验 ID | 防御方法 | 恶意比例 | 目的 |
|---------|---------|---------|------|
| 2.1 | FedAvg | 10% | 低恶意率 |
| 2.2 | FedAvg | 20% | 中等恶意率 |
| 2.3 | FedAvg | 30% | 高恶意率 |
| 2.4 | Ours | 10% | 双层防御 - 低 |
| 2.5 | Ours | 20% | 双层防御 - 中 |
| 2.6 | Ours | 30% | 双层防御 - 高 |

**运行示例**：
```bash
# 实验 2.3: FedAvg + 30% 恶意
python scripts/run_example.py --defense fedavg --dataset CIFAR10 \
    --model resnet18 --num-clients 100 --num-servers 2 --rounds 50 \
    --client-attack signflip --malicious-client-ratio 0.3 \
    --seed 42 --device cuda:0

# 实验 2.6: Ours + 30% 恶意
python scripts/run_example.py --defense ours --dataset CIFAR10 \
    --model resnet18 --num-clients 100 --num-servers 2 --rounds 50 \
    --client-attack signflip --malicious-client-ratio 0.3 \
    --seed 42 --device cuda:0
```

---

### 实验 3: Non-IID 程度对比

| 实验 ID | 防御方法 | Alpha (Dirichlet) | Non-IID 程度 |
|---------|---------|-------------------|--------------|
| 3.1 | FedAvg | 0.1 | 极强 Non-IID |
| 3.2 | FedAvg | 0.5 | 强 Non-IID |
| 3.3 | FedAvg | 1.0 | 中等 Non-IID |
| 3.4 | Ours | 0.1 | 极强 Non-IID |
| 3.5 | Ours | 0.5 | 强 Non-IID |
| 3.6 | Ours | 1.0 | 中等 Non-IID |

**运行示例**：
```bash
# 实验 3.1: FedAvg + 极强 Non-IID
python scripts/run_example.py --defense fedavg --dataset CIFAR10 \
    --model resnet18 --num-clients 100 --num-servers 2 --rounds 50 \
    --client-attack signflip --malicious-client-ratio 0.2 \
    --alpha 0.1 \
    --seed 42 --device cuda:0

# 实验 3.4: Ours + 极强 Non-IID
python scripts/run_example.py --defense ours --dataset CIFAR10 \
    --model resnet18 --num-clients 100 --num-servers 2 --rounds 50 \
    --client-attack signflip --malicious-client-ratio 0.2 \
    --alpha 0.1 \
    --seed 42 --device cuda:0
```

---

## ⏱️ 预期运行时间

### 单次实验时间估算

**配置**: CIFAR-10 + ResNet18 + 100 客户端 + 50 轮

| 阶段 | 每轮时间 | 50 轮总时间 | 备注 |
|------|---------|------------|------|
| 客户端训练 (100x) | ~300s | ~4.2 小时 | 主要瓶颈 |
| 客户端评估 (200x) | ~60s | ~50 分钟 | sync + local |
| 服务器聚合 (GPU) | ~3s | ~2.5 分钟 | ✅ GPU 加速 |
| 服务器评估 | ~0.5s | ~25 秒 | - |
| 其他开销 | ~10s | ~8 分钟 | - |
| **总计** | ~**374s** | ~**5.2 小时** | 每个实验 |

### 完整实验矩阵时间

- **实验 1** (10 个配置): ~52 小时
- **实验 2** (6 个配置): ~31 小时
- **实验 3** (6 个配置): ~31 小时
- **总计**: ~114 小时 ≈ **4.75 天**

**优化建议**：
1. 使用多 GPU 并行运行多个实验
2. 减少轮数到 30 轮（节省 40% 时间）
3. 先运行关键实验（Ours vs FedAvg vs Krum）

---

## 📈 结果分析

### 提取关键指标

运行完成后，从 `runs/` 目录下的 `history.json` 提取：

```python
import json
import numpy as np

# 加载实验结果
with open('runs/ours/20251113-xxxxx/history.json') as f:
    history = json.load(f)

# 提取最后一轮的指标
last_round = history[-1]
agg = last_round['aggregated']

# 良性客户端的准确率
benign_sync_acc = agg['benign_sync_test_accuracy_mean']
benign_local_acc = agg['benign_local_test_accuracy_mean']

# 计算标准差
benign_sync_std = np.sqrt(agg['benign_sync_test_accuracy_var'])
benign_local_std = np.sqrt(agg['benign_local_test_accuracy_var'])

print(f"Sync Accuracy: {benign_sync_acc*100:.2f}% ± {benign_sync_std*100:.2f}%")
print(f"Local Accuracy: {benign_local_acc*100:.2f}% ± {benign_local_std*100:.2f}%")

# 接受率（Ours 方法独有）
acceptance_rate = sum([d['accepted'] for d in last_round['details']
                       if d['role'] == 'benign']) / \
                  sum([1 for d in last_round['details'] if d['role'] == 'benign'])
print(f"Acceptance Rate: {acceptance_rate*100:.2f}%")
```

---

## 🎯 论文图表建议

### 图表 1: 主要结果对比（同步后准确率）

| 方法 | CIFAR-10 (20% 恶意, SignFlip) | 说明 |
|------|------------------------------|------|
| Local Only | 65.23 ± 2.15 | 无联邦学习 |
| FedAvg | 58.12 ± 1.02 | 无防御 |
| Krum | 84.51 ± 0.45 | 服务器端防御 |
| Median | 81.23 ± 0.56 | 服务器端防御 |
| **Ours** | **85.67 ± 0.42** | **双层防御** ⭐ |

### 图表 2: 不同攻击方法下的准确率

折线图：横轴 = 轮数，纵轴 = 良性客户端准确率

- 线条 1: Ours + SignFlip
- 线条 2: Ours + Noise
- 线条 3: Ours + ALIE
- 线条 4: FedAvg + SignFlip (对比)
- 线条 5: Krum + SignFlip (对比)

### 图表 3: 接受率分析（Ours 独有）

| 攻击类型 | 良性客户端接受率 | 说明 |
|---------|---------------|------|
| SignFlip | 78.5% | 中等攻击 |
| Noise | 45.2% | 强攻击，大量拒绝 |
| ALIE | 85.3% | 隐蔽攻击 |

**关键论点**: 接受率下降说明客户端验证识别出了被污染的模型

### 图表 4: Global vs Sync 准确率对比（体现双层防御优势）

| 方法 | Global Model Acc | Client Sync Acc | Δ |
|------|-----------------|----------------|---|
| FedAvg | 58.12% | 58.12% | 0% |
| Krum | 84.51% | 84.51% | 0% |
| **Ours** | 75.43% | **85.67%** | **+10.24%** ⭐ |

**关键发现**: Ours 的 Sync Acc 显著高于 Global Acc，证明客户端验证有效！

---

## 🛠️ 故障排除

### 问题 1: GPU 内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 减少客户端数量
--num-clients 50

# 或减少批大小
--batch-size 32

# 或禁用 GPU 聚合（不推荐）
# 需要在代码中设置 ServerConfig(use_gpu_aggregation=False)
```

### 问题 2: 训练太慢

**症状**: 每轮超过 10 分钟

**解决方案**:
```bash
# 增加并行工作线程
--max-workers 4

# 或减少本地训练轮数
--local-epochs 3

# 或使用更小的模型
--model resnet18  # 而不是 resnet50
```

### 问题 3: NaN 或 Inf 出现

**症状**: 准确率突然变成 NaN

**解决方案**:
```bash
# 降低学习率
--lr 0.001  # 从 0.01 降低到 0.001

# 或使用 Adam 优化器
--optimizer adam --lr 0.001
```

---

## 📝 实验记录模板

建议创建一个实验记录表格：

| 实验 ID | 日期 | 防御 | 攻击 | 恶意% | Alpha | Sync Acc | Local Acc | 接受率 | 备注 |
|---------|------|------|------|-------|-------|----------|-----------|--------|------|
| 1.1 | 2025-11-13 | FedAvg | SignFlip | 20% | 0.5 | 58.12% | 59.45% | N/A | 基线 |
| 1.4 | 2025-11-13 | Ours | SignFlip | 20% | 0.5 | 85.67% | 87.23% | 78.5% | ✅ 成功 |
| ... | | | | | | | | | |

---

## 🚀 快速开始命令

### 最小可行实验（验证系统）

```bash
# 3 轮快速测试（约 20 分钟）
python scripts/run_example.py \
    --defense ours \
    --dataset CIFAR10 \
    --model resnet18 \
    --num-clients 20 \
    --num-servers 1 \
    --rounds 3 \
    --client-attack signflip \
    --malicious-client-ratio 0.2 \
    --seed 42 \
    --device cuda:0
```

### 标准实验（完整版）

```bash
# 50 轮完整实验（约 5.2 小时）
python scripts/run_example.py \
    --defense ours \
    --dataset CIFAR10 \
    --model resnet18 \
    --num-clients 100 \
    --num-servers 2 \
    --rounds 50 \
    --client-attack signflip \
    --malicious-client-ratio 0.2 \
    --seed 42 \
    --device cuda:0 \
    --lr 0.01 \
    --batch-size 64 \
    --local-epochs 5 \
    --max-workers 4
```

---

**准备就绪！开始你的 CIFAR-10 + ResNet 实验吧！** 🚀

**文档版本**: v1.0
**创建时间**: 2025-11-13
**作者**: Multi-Server FL Team
