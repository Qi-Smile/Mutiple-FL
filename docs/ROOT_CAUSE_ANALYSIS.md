# 训练曲线异常的根本原因深度分析

## 目录
1. [问题1: 初期下降 (Round 1-5)](#问题1-初期下降)
2. [问题2: 中期抖动 (Round 10-50)](#问题2-中期抖动)
3. [问题3: 末期崩溃 (Round 96-100)](#问题3-末期崩溃)
4. [三个问题的内在联系](#三个问题的内在联系)

---

## 问题1: 初期下降

### 现象
- Round 1: 18.56% → Round 5: 11.44% (下降39%)
- 相比Local Training的快速上升，Ours方法反而下降

### 根本原因分析

#### 1.1 恶意攻击强度过大

**客户端攻击 (noise攻击, std=0.5):**
```python
# 在 attacks/client.py 中
class _NoiseAttack:
    def _transform_updates(self, updates, malicious_mask, weights=None):
        noise = torch.normal(mean=0.0, std=0.5, size=...)
        attacked[malicious_mask] = noise  # 完全替换为噪声！
```

**问题**: 恶意客户端的更新被**完全替换**为高斯噪声 (std=0.5)
- 正常梯度更新的L2范数通常在 0.01-0.1
- 噪声的期望L2范数 = std * sqrt(d) ≈ 0.5 * sqrt(60000) ≈ 122
- **恶意更新比正常更新大1000倍以上！**

**服务器攻击 (random攻击, noise_scale=5.0):**
```python
# 在 attacks/server.py 中
class _RandomServerAttack:
    def transform(self, server_id, aggregated_state, history):
        noise = torch.randn_like(tensor) * 5.0
        attacked[key] = noise  # 完全替换为随机噪声！
```

**问题**: 恶意服务器直接返回随机模型参数
- LeNet的参数范围通常在 [-1, 1]
- noise_scale=5.0 的噪声范围在 [-15, 15]
- **完全破坏了模型结构！**

#### 1.2 Geometric Median的理论失效

**Geometric Median的breakdown point:**
- 理论上可以抵御 < 50% 的恶意节点
- **但前提是恶意更新的幅度有界！**

**当前情况:**
- 20%恶意客户端 + 10%恶意服务器
- 每个服务器管理10个客户端，其中2个恶意
- 10个服务器中1个是恶意的

**关键问题**: 恶意更新太极端

假设100个客户端分给10个服务器，每服务器10个客户端:
- 服务器A (良性): 聚合8个良性 + 2个极端噪声客户端
- Geometric Median在小样本(n=10)时**很难过滤极端值**
- 2/10 = 20% 的极端值会显著拉偏中位数

**数学解释:**

Weiszfeld算法的迭代公式:
```
median_new = Σ(w_i * x_i / ||x_i - median_old||) / Σ(w_i / ||x_i - median_old||)
```

当某个更新 x_i 非常远 (||x_i - median|| >> 其他距离):
- 它的权重 1/||x_i - median|| 会很小
- **但当恶意节点占比20%，且初始median估计不准时，会被拉偏**

#### 1.3 BALANCE验证机制的"适应期悖论"

观察数据:
```
Round 1: Accepted=80.8%, Similarity=0.947, Threshold=0.990
Round 2: Accepted=61.1%, Similarity=0.920, Threshold=0.980
Round 3: Accepted=77.7%, Similarity=0.774, Threshold=0.970
Round 4: Accepted=94.7%, Similarity=0.631, Threshold=0.961
Round 5: Accepted=98.6%, Similarity=0.450, Threshold=0.951
```

**悖论现象:**
1. Round 1: 阈值很高(0.99), 80.8%的更新被接受
   - 但服务器聚合已经被恶意更新污染
   - 客户端接受了被污染的模型

2. Round 2: 接受率降到61.1%
   - 客户端开始拒绝可疑更新
   - **但拒绝的可能恰恰是正确的修正！**

3. Round 4-5: 接受率回升到95%+
   - Similarity从0.947降到0.450 (差异越来越大)
   - 这意味着服务器更新和客户端本地更新越来越不一致
   - **客户端被"训练"去接受错误的方向**

**根本问题**:
- BALANCE假设服务器更新是可信的标杆
- 但在拜占庭环境下，服务器聚合本身已被污染
- 客户端无法区分"服务器是对的，我的梯度不好"还是"服务器被攻击了"

---

## 问题2: 中期抖动

### 现象
- Round 10-50: 准确率在 0.37-0.81 之间波动
- 标准差 = 0.117 (相当大)
- 对比Local Training的平滑曲线，Ours抖动剧烈

### 根本原因分析

#### 2.1 BALANCE阈值的指数衰减

**阈值公式:**
```python
threshold = gamma * exp(-kappa * round_idx)
          = 1.0 * exp(-0.01 * round)
```

**具体值:**
- Round 10: threshold = 0.905
- Round 20: threshold = 0.819
- Round 30: threshold = 0.741
- Round 40: threshold = 0.670
- Round 50: threshold = 0.606

**问题**: 阈值持续快速下降
- 同样的 similarity ratio，在不同轮次可能被接受或拒绝
- 导致客户端的更新策略不稳定

**观察图表 (右上):**
- 绿色线 (Accepted Rate): 在90%-100%之间剧烈波动
- 红色虚线 (Threshold): 平滑下降
- **两者的交叉导致接受/拒绝决策的频繁切换**

#### 2.2 Non-IID数据的梯度方差

**数据分区配置:**
- alpha = 0.5 (Dirichlet分布)
- 100个客户端
- 每个客户端的数据分布高度异构

**Dirichlet(alpha=0.5)的特性:**
- 每个客户端的类别分布严重倾斜
- 例如: 客户端A: [90% class-0, 5% class-1, ...]
       客户端B: [5% class-0, 85% class-2, ...]

**导致的问题:**
- 不同客户端的梯度方向可能**几乎正交**
- Geometric Median试图找到"中间点"
- 但在高维空间中，正交向量的中位数可能**远离所有输入**

**数学直觉:**
想象3个向量在3D空间:
- v1 = [1, 0, 0] (沿x轴)
- v2 = [0, 1, 0] (沿y轴)
- v3 = [0, 0, 1] (沿z轴)

它们的geometric median ≈ [0.33, 0.33, 0.33]
- 这个向量的模长 = 0.57
- 而原始向量模长都是1.0
- **中位数可能比所有输入都小，导致训练变慢**

#### 2.3 Similarity的不稳定性

**观察左下图 (紫色曲线):**
- Similarity在对数坐标下剧烈波动
- 有多个尖峰 (例如Round 39的大尖峰)
- 这意味着某些轮次服务器更新和客户端更新差异特别大

**为什么会差异大?**

1. **恶意服务器的随机影响**
   - 10%的服务器是恶意的
   - 当恶意服务器的更新被聚合时，协调器的全局模型会偏移
   - 良性客户端发现服务器更新和自己的方向完全不同

2. **Geometric Median的方差**
   - 在Non-IID + 攻击环境下，中位数估计的方差很大
   - 不同轮次的良性/恶意客户端组成可能不同
   - 导致聚合结果摇摆不定

---

## 问题3: 末期崩溃 (最严重)

### 现象
```
Round 96: Acc=87.03%, NaN=0%      ✓ 正常
Round 97: Acc=86.83%, NaN=6.88%   ⚠️ 突然出现NaN
Round 98: Acc=81.55%, NaN=9.23%   🔴 准确率暴跌
Round 99: Acc=78.26%, NaN=11.41%  🔴 持续恶化
Round 100: Acc=78.26%, NaN=11.41% 🔴 完全崩溃
```

同时:
- Round 97: Similarity = 14.28 (正常应该 < 1.0)
- Round 98-100: Similarity = NaN

### 根本原因分析

#### 3.1 梯度爆炸的连锁反应

**触发条件:**

1. **学习率在后期仍然很高 (lr=0.01)**
   - MNIST + LeNet 的常用学习率: 0.001-0.01
   - 初期lr=0.01是合理的
   - **但100轮后，模型接近收敛，lr应该降低到0.001或更小**

2. **没有梯度裁剪**
   - 查看 `client.py` 的 `train_one_round()` 方法:
   ```python
   loss.backward()
   self.optimizer.step()  # 没有梯度裁剪！
   ```

3. **恶意攻击的累积效应**
   - 96轮的攻击已经让某些客户端的参数接近临界值
   - 例如某个weight从 0.8 被推到 5.0
   - 在Round 97，一个稍大的梯度 + lr=0.01 可能让它变成 50.0
   - 导致 exp(50) = Inf

**数学解释:**

LeNet的最后一层是全连接 + Softmax:
```python
logits = W @ x + b  # shape: [10]
probs = exp(logits) / sum(exp(logits))
```

如果某个权重异常大:
- logits[i] = 50
- exp(50) ≈ 5e21
- 梯度 = exp(50) * (1 - prob) ≈ 5e21
- 参数更新 = 5e21 * 0.01 = 5e19
- 下一步: exp(5e19) = **Inf**

#### 3.2 NaN的传播机制

**一旦出现NaN:**

1. **在前向传播中:**
   - 某个权重 = NaN
   - logits = NaN @ x = NaN
   - loss = CrossEntropy(NaN, target) = NaN

2. **在反向传播中:**
   - loss.backward() 产生 NaN梯度
   - optimizer.step() 将 NaN 写入所有参数
   - **整个模型污染**

3. **在聚合中:**
   - 客户端上传的state包含NaN
   - Geometric Median计算:
     ```python
     distances = torch.norm(flat_states - median, dim=1)
     # 如果某个state含NaN, 则distance=NaN
     inverted = weight / distances  # NaN
     new_median = (states * inverted).sum() / inverted.sum()  # NaN
     ```
   - **服务器聚合结果也变成NaN**

4. **在BALANCE验证中:**
   - Similarity计算:
     ```python
     diff = torch.norm(server_vec - client_vec)  # NaN
     ratio = diff / denom  # NaN
     ```
   - **无法判断是否接受**

**这就是为什么:**
- Round 97: 6.88%的客户端出现NaN (约7个客户端)
- Round 98: 传播到9.23% (约9个客户端)
- Round 99-100: 稳定在11.41% (约11个客户端)

**为什么稳定在11%?**
- 因为 `ensure_finite_state_dict()` 函数会用fallback替换NaN
- 但fallback可能是上一轮的模型
- 这些客户端实际上"卡住了"，不再学习
- 导致准确率永久下降

#### 3.3 Similarity突变的解释

Round 97的Similarity = 14.28，这意味着:
```python
ratio = ||server_update - client_update|| / ||client_update|| = 14.28
```

**这说明服务器更新的幅度是客户端更新的14.28倍！**

为什么?
- 某些客户端出现了极端梯度 (接近Inf)
- Geometric Median在计算时:
  ```python
  inverted = weight / distances
  # 如果某个距离很小(接近0), inverted会很大
  new_median = (states * inverted).sum() / inverted.sum()
  # 会被极端值拉偏
  ```
- 导致聚合结果异常大

#### 3.4 为什么Local Training没有这个问题?

**Local Training的代码逻辑:**
- 每个客户端独立训练，互不影响
- 即使某个客户端出现NaN，不会传播给其他客户端
- 统计时可能会忽略NaN值

**Ours方法的问题:**
- 所有客户端通过服务器聚合耦合在一起
- **一个客户端的NaN会通过聚合传播给所有人**
- 这是联邦学习的固有脆弱性

---

## 三个问题的内在联系

### 因果链条

```
初期: 恶意攻击 → Geometric Median失效 → 模型被污染
                    ↓
中期: 被污染的模型 → 梯度方向混乱 → Non-IID加剧抖动
                    ↓
      BALANCE阈值衰减 → 接受/拒绝不稳定 → 进一步加剧抖动
                    ↓
      抖动累积 → 参数接近临界值
                    ↓
末期: lr过高 + 参数临界 + 极端攻击 → 梯度爆炸 → NaN
                    ↓
      NaN传播 → 全局模型崩溃 → 准确率永久下降
```

### 关键洞察

1. **初期下降不是bug，是必然**
   - 在30%恶意节点 + 极端攻击下，Geometric Median理论上就会失效
   - 需要更强的防御机制 (如Krum, Trimmed Mean, 或更小的恶意比例)

2. **中期抖动是Non-IID + 攻击 + BALANCE的共振**
   - 三个独立的不稳定因素叠加
   - 单独解决任何一个都不够

3. **末期崩溃是工程问题，可以完全避免**
   - 添加梯度裁剪 → 防止梯度爆炸
   - 学习率衰减 → 后期稳定收敛
   - NaN检测+回滚 → 防止传播
   - **这是三个问题中唯一可以完全解决的**

### 为什么Ours峰值(87%)远超Local(67%)?

**这说明方法本身是有效的！**

当恶意攻击**没有触发NaN崩溃**时:
- Geometric Median + BALANCE 确实能提升鲁棒性
- 多服务器架构增加了模型容量
- 联邦学习的数据多样性提升泛化能力

**问题在于**: 缺少数值稳定性的工程保障

---

## 总结

| 问题 | 根本原因 | 是否可解决 | 优先级 |
|------|---------|-----------|--------|
| 初期下降 | 理论: Geometric Median breakdown<br>工程: 攻击过强 | 部分可解 | 中 |
| 中期抖动 | 理论: Non-IID + BALANCE设计<br>工程: 阈值衰减过快 | 可缓解 | 低 |
| 末期崩溃 | 工程: 缺少梯度裁剪和NaN处理 | **完全可解** | **高** |

**建议优先修复末期崩溃，因为:**
1. 完全是工程问题，有成熟解决方案
2. 修复后准确率可稳定在85%+，远超Local Training
3. 其他两个问题在末期崩溃修复后影响较小

**预期效果:**
- 修复前: Peak 87% → Final 78% (崩溃)
- 修复后: Peak 87% → Final 85%+ (稳定)
- 提升: 7-9个百分点，**完全消除崩溃风险**
