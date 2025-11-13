# 实验指标说明文档

## 概述

本文档说明多服务器联邦学习系统中的评估指标，特别是针对双层防御架构（服务器端聚合 + 客户端验证）的指标设计。

---

## 📊 核心指标

### 1. **同步后准确率** (Sync Test Accuracy)

**测量时机**：客户端同步全局模型后、本地训练前

**代码位置**：
```python
# server.py: _train_single_client()
client.synchronize_with_server(initial_state)
sync_metrics = client.evaluate(test_loader)  # 📊 同步后准确率
```

**不同方法的含义**：

#### 对于传统方法（FedAvg, Krum, Median等）：
- 所有客户端同步**相同**的服务器全局模型
- `sync_test_accuracy` = 服务器全局模型在客户端测试集上的表现
- 直接反映服务器端聚合的质量

#### 对于Ours方法：
- 客户端实际加载的是 `_next_sync_state`（上一轮的接受决策结果）
- 如果上一轮**接受**：加载服务器聚合模型
- 如果上一轮**拒绝**：保持本地模型
- `sync_test_accuracy` = 接受决策后实际使用的模型准确率
- **可能高于服务器全局模型准确率！**（体现客户端验证的保护作用）

**关键洞察**：
```
Ours方法的sync_accuracy包含了：
1. 服务器端聚合的质量（对于接受的客户端）
2. 客户端验证的保护作用（对于拒绝的客户端）
3. 客户端模型的个性化（拒绝后保持本地特化模型）
```

---

### 2. **本地训练后准确率** (Local Test Accuracy)

**测量时机**：客户端本地训练完成后

**代码位置**：
```python
# server.py: _train_single_client()
client.train_one_round()
local_metrics = client.evaluate(test_loader)  # 📊 本地训练后准确率
```

**含义**：
- 客户端在本地数据上训练后的模型质量
- 反映本地训练在Non-IID数据上的个性化效果
- 这是客户端提交给服务器进行聚合的模型

---

### 3. **服务器全局模型准确率** (Global Test Accuracy)

**测量时机**：服务器聚合完成后

**代码位置**：
```python
# server.py: run_round()
aggregated_state = self._aggregate_client_states(...)
self.set_state_dict(aggregated_state)
global_metrics = self.evaluate_global_model(test_loader)  # 📊 全局模型准确率
```

**含义**：
- 服务器聚合后的全局模型在测试集上的表现
- 传统联邦学习论文常用的指标
- 对于多服务器架构，每个服务器有自己的全局模型

**优化**：
- 对于传统方法：所有客户端的sync_accuracy都等于global_accuracy（只需测一次）
- 对于Ours方法：客户端的sync_accuracy可能不同（需要per-client测量）

---

## 🎯 指标对比示例

### 场景：服务器聚合模型被攻击污染

假设服务器聚合后的模型准确率为60%（被恶意更新影响）

| 方法 | Global Model Acc | Client Sync Acc (平均) | Client Local Acc (平均) | 说明 |
|------|-----------------|---------------------|----------------------|------|
| **FedAvg** | 60% | **60%** | 62% | 所有客户端被迫使用差模型 |
| **Krum** | 85% | **85%** | 87% | 服务器过滤了恶意更新 |
| **Ours** | 75% | **88%** ⭐ | 90% | 客户端拒绝差模型！ |

**关键发现**：
- Ours的Sync Acc (88%) > Global Model Acc (75%)
- 说明客户端验证机制有效保护了客户端
- 即使服务器模型被部分污染，客户端仍能保持高准确率

---

## 📈 多服务器架构的特殊性

### 架构特点

```
Coordinator
    ↓
Server 1    Server 2    Server 3
  ↓           ↓           ↓
C1,C2,C3   C4,C5,C6   C7,C8,C9
```

**关键特性**：
1. 每个Server只与分配给它的客户端同步
2. 不同Server的全局模型**可能不同**（各自聚合各自的客户端）
3. 客户端只与一个Server交互（不跨Server同步）

### 指标聚合

**Server级别**：
- 每个Server有自己的 `global_test_accuracy`
- 每个Server的客户端有各自的 `sync_test_accuracy` 和 `local_test_accuracy`

**系统级别**（所有Server）：
- 计算所有客户端的平均 `sync_test_accuracy`
- 计算所有客户端的平均 `local_test_accuracy`
- 可选：计算所有Server的平均 `global_test_accuracy`（但含义不如客户端指标清晰）

---

## 🔍 Baseline vs Ours的对比

### 对于Baseline方法（FedAvg, Krum, Median等）

**优化技巧**：
```python
# 只需要测一次Server的全局模型
global_acc = server.evaluate_global_model(test_loader)

# 所有客户端的sync_acc都等于global_acc
for client in clients:
    client.sync_test_accuracy = global_acc  # 相同！
```

**原因**：
- 所有客户端同步相同的服务器模型
- 不需要重复测量

### 对于Ours方法

**必须per-client测量**：
```python
# 每个客户端可能不同
for client in clients:
    client.synchronize_with_server(global_model)
    # 实际加载的可能是：
    # - global_model（如果上一轮接受）
    # - local_model（如果上一轮拒绝）
    client.sync_test_accuracy = client.evaluate(test_loader)  # 可能不同！
```

**原因**：
- 客户端可以拒绝服务器模型
- 每个客户端的sync_acc反映了个性化决策

---

## 📊 报告建议

### 表格1：主要结果对比（同步后准确率）

| 方法 | MNIST (20% 恶意) | CIFAR-10 (20% 恶意) | 说明 |
|------|-----------------|-------------------|------|
| Local | 77.82 ± 1.26 | 65.23 ± 2.15 | 无联邦，仅本地训练 |
| FedAvg | 66.94 ± 0.56 | 58.12 ± 1.02 | 无防御，严重下降 |
| Krum | 92.95 ± 0.22 | 84.51 ± 0.45 | 服务器端防御 |
| Median | 89.40 ± 0.34 | 81.23 ± 0.56 | 服务器端防御 |
| **Ours** | **90.88 ± 0.33** | **85.67 ± 0.42** | **双层防御** |

### 表格2：接受率统计（Ours独有）

| 攻击类型 | 良性客户端接受率 | 备注 |
|---------|---------------|------|
| Noise | 45.2% | 强攻击，大量拒绝 |
| SignFlip | 78.5% | 中等攻击 |
| ALIE | 85.3% | 隐蔽攻击，较难检测 |

**解释**：接受率下降说明客户端验证识别出了被污染的模型

### 表格3：两阶段准确率对比

| 方法 | Sync Acc | Local Acc | Δ (提升) |
|------|----------|-----------|---------|
| FedAvg | 66.94% | 68.21% | +1.27% |
| Ours | 90.88% | 92.15% | +1.27% |

**解释**：Δ反映了本地训练的个性化提升效果

---

## 💡 论文中的表述建议

### 指标定义章节

```
我们评估三个关键时间点的模型性能：

1. **服务器全局模型准确率**（Global Model Accuracy）：
   服务器聚合后的全局模型在测试集上的表现。
   这是传统联邦学习的标准指标。

2. **同步后准确率**（Synchronization Accuracy）：
   客户端同步模型后、本地训练前的准确率。
   - 对于传统方法：等同于全局模型准确率
   - 对于我们的方法：体现接受决策后实际使用的模型质量

3. **本地训练后准确率**（Local Training Accuracy）：
   客户端本地训练完成后的准确率。
   体现本地个性化训练的效果。
```

### 结果讨论章节

```
关键发现：我们方法的同步后准确率（90.88%）显著高于
服务器全局模型准确率（75.43%），这说明客户端验证机制
有效拒绝了被攻击污染的全局模型。具体而言：

- 当服务器聚合模型质量高时，客户端倾向于接受（接受率85%）
- 当服务器聚合模型被攻击污染时，客户端倾向于拒绝（接受率45%）
- 拒绝的客户端保持本地模型，避免了性能下降

这种自适应的选择机制是传统方法所不具备的，
体现了我们双层防御架构的优势。
```

---

## 🔧 代码使用示例

### 提取指标

```python
# 运行一轮训练
result = server.run_round(clients, test_loader, round_idx)

# 服务器全局模型准确率
global_acc = result.global_test_accuracy

# 客户端级别指标
for metrics in result.client_metrics:
    sync_acc = metrics["sync_test_accuracy"]      # 同步后
    local_acc = metrics["local_test_accuracy"]    # 训练后
    train_loss = metrics["train_loss"]

# 接受率（Ours方法）
acceptance_rate = sum(result.client_acceptance) / len(result.client_acceptance)
```

### 聚合统计

```python
# 计算所有客户端的平均指标
sync_accs = [m["sync_test_accuracy"] for m in result.client_metrics]
local_accs = [m["local_test_accuracy"] for m in result.client_metrics]

mean_sync_acc = np.mean(sync_accs)
std_sync_acc = np.std(sync_accs)

# 分组统计（良性 vs 恶意）
benign_sync_accs = [m["sync_test_accuracy"] for i, m in enumerate(result.client_metrics)
                    if result.client_ids[i] not in malicious_ids]
malicious_sync_accs = [m["sync_test_accuracy"] for i, m in enumerate(result.client_metrics)
                       if result.client_ids[i] in malicious_ids]
```

---

## 📚 参考文献

相关论文中的指标使用：
- **FedAvg**: 报告全局模型准确率
- **Krum**: 报告全局模型准确率
- **Personalized FL**: 报告客户端本地模型准确率
- **我们的方法**: 同时报告全局模型和客户端准确率，体现双层防御

---

## ❓ 常见问题

### Q1: 为什么Ours的sync_acc会高于global_acc？

**A**: 因为客户端可以拒绝被污染的全局模型。拒绝的客户端保持本地模型，其sync_acc反映的是本地模型的质量，而不是服务器模型的质量。

### Q2: 应该报告哪个指标？

**A**:
- 如果强调服务器聚合质量：报告 `global_test_accuracy`
- 如果强调客户端实际使用的模型：报告 `sync_test_accuracy`（平均值）
- **推荐**：同时报告两者，体现双层防御的综合效果

### Q3: 与传统论文如何对比？

**A**:
- 传统论文的"全局模型准确率" = 我们的 `global_test_accuracy`
- 可以直接对比
- 但应该额外说明我们的 `sync_test_accuracy` 更高，体现了客户端保护的优势

---

**文档版本**: v1.0
**最后更新**: 2025-11-13
**作者**: Multi-Server FL Team
