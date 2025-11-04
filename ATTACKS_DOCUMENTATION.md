# Byzantine Federated Learning - Attack Methods Documentation

**Version**: 1.0
**Last Updated**: 2025-01-04
**Project**: ABR-FL (Asymmetric Byzantine-Resilient Federated Learning)

---

## Table of Contents

1. [Overview](#overview)
2. [Client-Side Attacks](#client-side-attacks)
   - [1. Noise Adversary](#1-noise-adversary)
   - [2. Sign-Flipping Adversary](#2-sign-flipping-adversary)
   - [3. Label-Flipping Adversary](#3-label-flipping-adversary)
   - [4. ALIE Adversary](#4-alie-adversary)
   - [5. IPM Adversary](#5-ipm-adversary)
   - [6. Adaptive Adversary](#6-adaptive-adversary)
   - [7. MinMax Adversary](#7-minmax-adversary)
3. [Server-Side Attacks](#server-side-attacks)
   - [8. Malicious Server Adversary](#8-malicious-server-adversary)
   - [9. Adaptive Malicious Server](#9-adaptive-malicious-server)
4. [Advanced Attacks (Targeting Geometric Median)](#advanced-attacks-targeting-geometric-median)
   - [10. Min-Sum Adversary](#10-min-sum-adversary)
   - [11. Inner-Max Adversary](#11-inner-max-adversary)
   - [12. Min-Max Adversary](#12-min-max-adversary-hybrid)
   - [13. AGR-Tailored Adversary](#13-agr-tailored-adversary)
5. [Attack Comparison Table](#attack-comparison-table)
6. [Configuration Examples](#configuration-examples)
7. [References](#references)

---

## Overview

This document provides a comprehensive technical description of all Byzantine attack methods implemented in the ABR-FL framework. These attacks are designed to test the robustness of federated learning systems against malicious participants.

### Attack Categories

| Category | Count | Target | Sophistication |
|----------|-------|--------|----------------|
| **Client-Side (Basic)** | 3 | Any aggregator | Low-Medium |
| **Client-Side (Advanced)** | 4 | Specific aggregators | High |
| **Server-Side** | 2 | Multi-server FL | Medium-High |
| **Adaptive (GeoMed)** | 4 | Geometric Median | Very High |

### Implementation Location

All adversaries are located in: `/blades/blades/adversaries/`

---

## Client-Side Attacks

### 1. Noise Adversary

**File**: `noise_adversary.py`
**Difficulty**: ⭐ Low
**Type**: Random Noise Injection

#### Description

The simplest Byzantine attack that replaces malicious clients' gradient updates with random Gaussian noise. This serves as a baseline attack to test basic robustness.

#### Formal Algorithm

**Input**:
- Benign client updates: $\mathcal{W}_{\text{benign}} = \{w_1, w_2, \ldots, w_{N-m}\}$
- Noise parameters: $\mu$ (mean), $\sigma$ (standard deviation)

**Output**:
- Malicious client update: $w_{\text{mal}}$

**Procedure**:
```
1. Compute benign mean: μ_benign = mean(W_benign)
2. Generate noise: ε ~ N(μ, σ²I_d)
3. Set malicious update: w_mal = ε
4. Return w_mal
```

#### Mathematical Formulation

$$w_{\text{mal}} \sim \mathcal{N}(\mu \cdot \mathbf{1}_d, \sigma^2 \mathbf{I}_d)$$

where:
- $d$ = model dimension
- $\mathbf{1}_d$ = vector of ones
- $\mathbf{I}_d$ = identity matrix

#### Hyperparameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `mean` | float | 0.1 | $\mathbb{R}$ | Mean of Gaussian noise |
| `std` | float | 0.1 | $(0, \infty)$ | Standard deviation of noise |

#### Implementation Details

```python
class NoiseAdversary(Adversary):
    def __init__(self, mean: float = 0.1, std: float = 0.1):
        self._noise_mean = mean
        self._noise_std = std

    def on_local_round_end(self, trainer: Trainer):
        benign_updates = self.get_benign_updates(trainer)
        mean = benign_updates.mean(dim=0)

        for result in trainer.local_results:
            client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                device = mean.device
                noise = torch.normal(
                    self._noise_mean, self._noise_std, size=mean.shape
                ).to(device)
                result[CLIENT_UPDATE] = noise
                result[TRAIN_LOSS] = -1  # Indicator of poisoning
```

#### Configuration Example

```yaml
adversary_config:
  type: blades.adversaries.NoiseAdversary
  mean: 0.1
  std: 0.1
```

#### Attack Effectiveness

- **Against FedAvg**: High (no defense)
- **Against Geometric Median**: Low (easily filtered)
- **Against Krum**: Low (outlier detection)
- **Against Trimmed-Mean**: Medium (depends on trimming ratio)

#### Theoretical Analysis

**Expected attack impact**:
$$\mathbb{E}[\|\text{aggregated} - \text{true\_mean}\|] \approx \frac{m}{N} \cdot \sigma$$

where $m$ = number of malicious clients, $N$ = total clients.

---

### 2. Sign-Flipping Adversary

**File**: `signflip_adversary.py`
**Difficulty**: ⭐⭐ Medium
**Type**: Gradient Reversal

#### Description

Flips the sign of all gradients during backpropagation, causing the malicious client to optimize in the opposite direction of convergence. This attack is applied during local training rather than post-processing.

#### Formal Algorithm

**Input**:
- Local model parameters: $\theta$
- Loss function: $\mathcal{L}(\theta; D_k)$

**Attack Timing**: During backward pass

**Procedure**:
```
1. Forward pass: compute loss L
2. Backward pass: compute gradients ∇L
3. Attack: for each parameter p:
      p.grad = -p.grad
4. Optimizer step with flipped gradients
5. Return updated model
```

#### Mathematical Formulation

Original gradient update:
$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$$

Sign-flipped update:
$$\theta_{t+1}^{\text{mal}} = \theta_t + \eta \nabla \mathcal{L}(\theta_t)$$

The malicious update direction:
$$\Delta w_{\text{mal}} = -\Delta w_{\text{benign}}$$

#### Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| None | - | - | No hyperparameters (deterministic attack) |

#### Implementation Details

```python
class SignFlipAdversary(Adversary):
    def on_trainer_init(self, trainer: Trainer):
        class SignFlipCallback(ClientCallback):
            def on_backward_end(self, task):
                model = task.model
                for _, para in model.named_parameters():
                    para.grad.data = -para.grad.data  # Flip sign

        for client in self.clients:
            client.to_malicious(callbacks_cls=SignFlipCallback,
                               local_training=True)
```

**Key Feature**: Uses callback mechanism to inject attack during training loop.

#### Configuration Example

```yaml
adversary_config:
  type: blades.adversaries.SignFlipAdversary
```

#### Attack Effectiveness

- **Against FedAvg**: Very High (pushes model in wrong direction)
- **Against Geometric Median**: Medium (depends on $m_c$)
- **Against Krum**: Medium (may be selected if coordinated)
- **Against Trimmed-Mean**: High (affects median direction)

#### Theoretical Analysis

If $m < N/2$, robust aggregators should filter this attack. However, the attack is particularly effective when:

$$\frac{m}{N} > 0.3 \quad \text{and} \quad \text{aggregator} = \text{Mean}$$

Expected deviation from true gradient:
$$\mathbb{E}[\|\text{agg} - g_{\text{true}}\|] = \frac{2m}{N} \|g_{\text{true}}\|$$

---

### 3. Label-Flipping Adversary

**File**: `labelflip_adversary.py`
**Difficulty**: ⭐⭐ Medium
**Type**: Data Poisoning

#### Description

Flips training labels during local training, causing the malicious client to learn incorrect associations. This is a backdoor-style attack that can persist even after the attack stops.

#### Formal Algorithm

**Input**:
- Training data batch: $(X, Y)$ where $Y \in \{0, 1, \ldots, C-1\}$
- Number of classes: $C$

**Attack Timing**: During data loading

**Procedure**:
```
1. For each training batch (X, Y):
2.   Flip labels: Y' = (C - 1) - Y
3.   Train on poisoned data: (X, Y')
4. Return poisoned model update
```

#### Mathematical Formulation

Label transformation:
$$y'_i = C - 1 - y_i$$

**Examples for MNIST** ($C=10$):
- Digit 0 → Label 9
- Digit 1 → Label 8
- Digit 5 → Label 4

Loss minimized by malicious client:
$$\mathcal{L}_{\text{mal}} = \frac{1}{|D_k|} \sum_{(x,y) \in D_k} \ell(f(x; \theta), C-1-y)$$

#### Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| None | - | - | Automatically detects number of classes |

#### Implementation Details

```python
class LabelFlipAdversary(Adversary):
    def on_trainer_init(self, trainer: Trainer):
        num_class = self._get_num_model_outputs(trainer)

        class LabelFlipCallback(ClientCallback):
            def on_train_batch_begin(self, data, target):
                # Flip labels: 0→9, 1→8, etc.
                return data, num_class - 1 - target

        for client in self.clients:
            client.to_malicious(callbacks_cls=LabelFlipCallback,
                               local_training=True)
```

#### Configuration Example

```yaml
adversary_config:
  type: blades.adversaries.LabelFlipAdversary
```

#### Attack Effectiveness

- **Against FedAvg**: Very High (persistent backdoor)
- **Against Geometric Median**: Medium-High
- **Against FLTrust**: Low (detected by root dataset)
- **Against ABR-FL Client Verification**: Medium (detected by validation loss)

#### Theoretical Analysis

**Backdoor persistence**: The poisoned model learns:
$$P(y' = C-1-y | x) \approx \frac{m}{N}$$

This persists across rounds even if attack stops, making it particularly dangerous.

---

### 4. ALIE Adversary

**File**: `alie_adversary.py`
**Difficulty**: ⭐⭐⭐⭐ High
**Type**: Statistical Mimicry Attack
**Reference**: Baruch et al., "A Little Is Enough: Circumventing Defenses For Distributed Learning", NeurIPS 2019

#### Description

ALIE (A Little Is Enough) is a sophisticated attack that crafts malicious updates at the edge of the benign distribution, making them hard to detect while still being effective. It uses statistical properties to find the optimal poisoning point.

#### Formal Algorithm

**Input**:
- Benign updates: $\mathcal{W}_{\text{benign}} = \{w_1, \ldots, w_{N-m}\}$
- Number of malicious clients: $m$
- Total clients: $N$

**Output**:
- Malicious update: $w_{\text{mal}}$

**Procedure**:
```
1. Compute statistics:
   μ = mean(W_benign)
   σ = std(W_benign)

2. Compute survival threshold:
   s = floor(N/2) + 1 - m

3. Compute CDF value:
   p = (N - m - s) / (N - m)

4. Find optimal z value:
   z_max = Φ^(-1)(p)  // Inverse CDF of N(0,1)

5. Craft malicious update:
   w_mal = μ + z_max × σ

6. Return w_mal
```

#### Mathematical Formulation

Given benign updates $\{w_i\}_{i=1}^{N-m}$:

$$\mu_{\text{benign}} = \frac{1}{N-m} \sum_{i=1}^{N-m} w_i$$

$$\sigma_{\text{benign}} = \sqrt{\frac{1}{N-m} \sum_{i=1}^{N-m} \|w_i - \mu_{\text{benign}}\|^2}$$

Survival threshold (number of benign clients needed to outvote malicious):
$$s = \lfloor N/2 \rfloor + 1 - m$$

CDF value for optimal positioning:
$$p = \frac{N - m - s}{N - m}$$

Optimal $z$ value:
$$z_{\max} = \Phi^{-1}(p)$$

where $\Phi$ is the CDF of standard normal distribution $\mathcal{N}(0,1)$.

**Final malicious update**:
$$w_{\text{mal}} = \mu_{\text{benign}} + z_{\max} \cdot \sigma_{\text{benign}}$$

**Special case for SignGuard**: Randomly negate half of the elements.

#### Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| None | - | - | All parameters auto-computed from $N$, $m$ |

#### Implementation Details

```python
class ALIEAdversary(Adversary):
    def on_trainer_init(self, trainer: Trainer):
        super().on_trainer_init(trainer)

        self.num_clients = trainer.config.num_clients
        num_byzantine = len(self.clients)

        # Compute survival threshold
        s = torch.floor_divide(self.num_clients, 2) + 1 - num_byzantine

        # Compute CDF value
        cdf_value = (self.num_clients - num_byzantine - s) / (
            self.num_clients - num_byzantine
        )

        # Inverse CDF to find z_max
        dist = torch.distributions.normal.Normal(
            torch.tensor(0.0), torch.tensor(1.0)
        )
        self.z_max = dist.icdf(cdf_value)

    def on_local_round_end(self, trainer: Algorithm):
        benign_updates = self.get_benign_updates(trainer)
        mean = benign_updates.mean(dim=0)
        std = benign_updates.std(dim=0)

        # For SignGuard defense
        if isinstance(trainer.server.aggregator, Signguard):
            if self.negative_indices is None:
                num_elements = len(std)
                num_negate = num_elements // 2
                self.negative_indices = random.sample(
                    range(num_negate), num_negate
                )
            std[self.negative_indices] *= -1

        # Craft malicious update
        update = mean + std * self.z_max

        for result in trainer.local_results:
            client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                result[CLIENT_UPDATE] = update
```

#### Configuration Example

```yaml
adversary_config:
  type: blades.adversaries.ALIEAdversary
  # No parameters needed
```

#### Attack Effectiveness

- **Against FedAvg**: Very High
- **Against Median**: High (designed to bypass)
- **Against Trimmed-Mean**: High (designed to bypass)
- **Against Geometric Median**: Medium
- **Against Krum**: Medium-High

#### Theoretical Analysis

**Key insight**: The attack positions itself just outside the benign distribution to:
1. Avoid detection by distance-based defenses
2. Still influence the aggregation result

**Breakdown point**: ALIE can succeed even when $m < N/2$ by carefully choosing $z_{\max}$ based on the aggregation rule's voting mechanism.

Expected deviation introduced:
$$\Delta \approx \frac{m \cdot z_{\max} \cdot \sigma}{N}$$

---

### 5. IPM Adversary

**File**: `ipm_adversary.py`
**Difficulty**: ⭐⭐⭐⭐ High
**Type**: Inner Product Manipulation

#### Description

IPM (Inner Product Manipulation) attack crafts malicious updates to maximize the negative inner product with the benign gradient direction, effectively pushing the model in the opposite direction of convergence.

#### Formal Algorithm

**Input**:
- Benign updates: $\mathcal{W}_{\text{benign}}$
- Scale factor: $\lambda$

**Output**:
- Malicious update: $w_{\text{mal}}$

**Procedure**:
```
1. Compute benign mean:
   μ = mean(W_benign)

2. Craft malicious update:
   w_mal = -λ × μ

3. Return w_mal
```

#### Mathematical Formulation

Given benign updates $\{w_i\}_{i=1}^{N-m}$:

$$\mu_{\text{benign}} = \frac{1}{N-m} \sum_{i=1}^{N-m} w_i$$

Malicious update:
$$w_{\text{mal}} = -\lambda \cdot \mu_{\text{benign}}$$

**Objective**: Maximize negative inner product:
$$\max_{w_{\text{mal}}} \langle w_{\text{mal}}, -\mu_{\text{benign}} \rangle$$

Solution:
$$w_{\text{mal}} = -\lambda \cdot \mu_{\text{benign}}$$

**Effect on aggregation** (assuming Mean aggregator):
$$w_{\text{agg}} = \frac{(N-m) \mu_{\text{benign}} + m(-\lambda \mu_{\text{benign}})}{N} = \left(1 - \frac{m(1+\lambda)}{N}\right) \mu_{\text{benign}}$$

#### Hyperparameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `scale` | float | 1.0 | $[0, \infty)$ | Scaling factor for attack magnitude |

**Recommended values**:
- `scale = 0.5`: Mild attack
- `scale = 1.0`: Standard attack (default)
- `scale = 2.0`: Aggressive attack

#### Implementation Details

```python
class IPMAdversary(Adversary):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self._scale = scale

    def on_local_round_end(self, algorithm: Trainer):
        benign_updates = self.get_benign_updates(algorithm)
        mean = benign_updates.mean(dim=0)

        # Simple but effective: negative scaled mean
        update = -self._scale * mean

        for result in algorithm.local_results:
            client = algorithm.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                result[CLIENT_UPDATE] = update
```

#### Configuration Example

```yaml
adversary_config:
  type: blades.adversaries.IPMAdversary
  scale: 1.0
```

#### Attack Effectiveness

- **Against FedAvg**: Very High
- **Against Geometric Median**: Medium (depends on $m_c$ and scale)
- **Against Krum**: Low (easily identified as outlier)
- **Against Trimmed-Mean**: Medium-High
- **Against ABR-FL**: Low (detected by client verification)

#### Theoretical Analysis

**Expected impact** on Mean aggregator:
$$\|\text{agg} - \mu_{\text{true}}\| = \frac{m(1+\lambda)}{N} \|\mu_{\text{benign}}\|$$

**Critical threshold**: Attack succeeds when:
$$\frac{m(1+\lambda)}{N} > 0.5$$

For $m = 0.3N$ and $\lambda = 1.0$:
$$\frac{0.3(1+1)}{1} = 0.6 > 0.5 \quad \checkmark$$

---

### 6. Adaptive Adversary

**File**: `adaptive_adversary.py`
**Difficulty**: ⭐⭐⭐⭐ High
**Type**: Adaptive Attack against Median/Trimmed-Mean

#### Description

An adaptive attack specifically designed to bypass coordinate-wise aggregation methods (Median, Trimmed-Mean) by analyzing the sign pattern and min/max vectors of benign updates.

#### Formal Algorithm

**Input**:
- Benign updates: $\mathcal{W}_{\text{benign}} \in \mathbb{R}^{(N-m) \times d}$
- Parameter: $b = 2$ (scaling factor)

**Output**:
- Malicious update: $w_{\text{mal}} \in \mathbb{R}^d$

**Procedure**:
```
1. Compute statistics:
   μ = mean(W_benign)  // Shape: (d,)
   v_max = max(W_benign, dim=0)  // Coordinate-wise max
   v_min = min(W_benign, dim=0)  // Coordinate-wise min

2. Compute deviation pattern:
   δ = sign(μ)  // ∈ {-1, 0, +1}^d

3. For each coordinate j ∈ [1, d]:
   If δ[j] = -1 and v_max[j] > 0:
       w_mal[j] = rand(0,1) × ((b-1) × v_max[j]) + v_max[j]

   Else if δ[j] = -1 and v_max[j] < 0:
       w_mal[j] = rand(0,1) × ((1/b - 1) × v_max[j]) + v_max[j]

   Else if δ[j] = +1 and v_min[j] > 0:
       w_mal[j] = rand(0,1) × ((1 - 1/b) × v_min[j]) + v_min[j]/b

   Else if δ[j] = +1 and v_min[j] < 0:
       w_mal[j] = rand(0,1) × ((1 - b) × v_min[j]) + v_min[j] × b

   Else:  // δ[j] = 0
       w_mal[j] = μ[j]

4. Return w_mal
```

#### Mathematical Formulation

For each coordinate $j$, the attack crafts $w_{\text{mal}}[j]$ based on local statistics:

**Case 1**: $\text{sign}(\mu[j]) = -1$ and $v_{\max}[j] > 0$
$$w_{\text{mal}}[j] = U(0,1) \cdot (b-1) v_{\max}[j] + v_{\max}[j] = v_{\max}[j] \cdot (1 + U(0,1) \cdot (b-1))$$

**Case 2**: $\text{sign}(\mu[j]) = -1$ and $v_{\max}[j] < 0$
$$w_{\text{mal}}[j] = U(0,1) \cdot (\tfrac{1}{b} - 1) v_{\max}[j] + v_{\max}[j]$$

**Case 3**: $\text{sign}(\mu[j]) = +1$ and $v_{\min}[j] > 0$
$$w_{\text{mal}}[j] = U(0,1) \cdot (1 - \tfrac{1}{b}) v_{\min}[j] + \tfrac{v_{\min}[j]}{b}$$

**Case 4**: $\text{sign}(\mu[j]) = +1$ and $v_{\min}[j] < 0$
$$w_{\text{mal}}[j] = U(0,1) \cdot (1 - b) v_{\min}[j] + b \cdot v_{\min}[j]$$

**Case 5**: $\text{sign}(\mu[j]) = 0$
$$w_{\text{mal}}[j] = \mu[j]$$

where $U(0,1)$ denotes uniform random variable in $[0, 1]$.

#### Hyperparameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `b` | int | 2 | $[2, 10]$ | Scaling factor (hardcoded in implementation) |

#### Implementation Details

```python
class AdaptiveAdversary(Adversary):
    def on_local_round_end(self, trainer: FedavgTrainer):
        updates = self._attack_median_and_trimmedmean(trainer)
        for result in trainer.local_results:
            client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                result[CLIENT_UPDATE] = updates

    def _attack_median_and_trimmedmean(self, algorithm: FedavgTrainer):
        benign_updates = self.get_benign_updates(algorithm)
        device = benign_updates.device
        mean_grads = benign_updates.mean(dim=0)
        deviation = torch.sign(mean_grads).to(device)
        max_vec, _ = benign_updates.max(dim=0)
        min_vec, _ = benign_updates.min(dim=0)
        b = 2

        # Create masks for different conditions
        neg_pos_mask = torch.logical_and(deviation == -1, max_vec > 0)
        neg_neg_mask = torch.logical_and(deviation == -1, max_vec < 0)
        pos_pos_mask = torch.logical_and(deviation == 1, min_vec > 0)
        pos_neg_mask = torch.logical_and(deviation == 1, min_vec < 0)
        zero_mask = deviation == 0

        # Compute results for each case
        result = torch.zeros_like(mean_grads)

        # Apply transformations based on masks
        rand_neg_pos = torch.rand(neg_pos_mask.sum(), device=device)
        result[neg_pos_mask] = (
            rand_neg_pos * ((b - 1) * max_vec[neg_pos_mask]) + max_vec[neg_pos_mask]
        )

        # ... (similar for other cases)

        return result
```

#### Configuration Example

```yaml
adversary_config:
  type: blades.adversaries.AdaptiveAdversary
```

#### Attack Effectiveness

- **Against FedAvg**: Very High
- **Against Median**: Very High (designed to bypass)
- **Against Trimmed-Mean**: Very High (designed to bypass)
- **Against Geometric Median**: Medium (not coordinate-wise)
- **Against Krum**: Medium

#### Theoretical Analysis

**Key insight**: By adapting to the sign pattern and placing updates strategically relative to min/max vectors, the attack can survive coordinate-wise trimming or median operations.

For Median aggregator with $m < N/2$:
- Normal attack might be filtered
- Adaptive attack positions itself to influence the median

**Success condition**: The attack succeeds when it can shift the median in each coordinate independently.

---

### 7. MinMax Adversary

**File**: `minmax_adversary.py`
**Difficulty**: ⭐⭐⭐⭐⭐ Very High
**Type**: Binary Search Optimization Attack

#### Description

A sophisticated attack that uses binary search to find the maximum deviation within the distance threshold of benign updates. This allows it to stay "under the radar" of distance-based defenses while maximizing attack effectiveness.

#### Formal Algorithm

**Input**:
- Benign updates: $\mathcal{W}_{\text{benign}}$
- Threshold parameter: $\tau$ (default: 1.0)
- Threshold difference: $\epsilon$ (default: 0.0001)

**Output**:
- Malicious update: $w_{\text{mal}}$

**Procedure**:
```
1. Compute benign statistics:
   μ = mean(W_benign)
   σ = std(W_benign)

2. Compute distance threshold:
   T = max_{i,j} ||w_i - w_j||_2  // Pairwise distances

3. Binary search for optimal scale:
   low = 0
   high = 5

   while |high - low| > 0.01:
       mid = (low + high) / 2
       w_candidate = μ - mid × σ

       d_max = max_{i} ||w_candidate - w_i||_2

       if d_max < T:
           low = mid  // Can push further
       else:
           high = mid  // Too far, reduce

4. Set malicious update:
   w_mal = μ - mid × σ

5. Special case for SignGuard:
   Randomly negate half of the elements

6. Return w_mal
```

#### Mathematical Formulation

**Objective**: Find the maximum deviation while staying within distance threshold:

$$\max_{\alpha} \|\mu_{\text{benign}} - \alpha \sigma_{\text{benign}}\|$$

**Subject to**:
$$\max_{i \in \{1, \ldots, N-m\}} \|(\mu - \alpha \sigma) - w_i\| < T$$

where $T = \max_{i,j} \|w_i - w_j\|$ is the maximum pairwise distance among benign updates.

**Solution via Binary Search**:
$$\alpha^* = \arg\max_{\alpha} \{\alpha : \max_i \|(\mu - \alpha \sigma) - w_i\| < T\}$$

**Final malicious update**:
$$w_{\text{mal}} = \mu_{\text{benign}} - \alpha^* \cdot \sigma_{\text{benign}}$$

#### Hyperparameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `threshold` | float | 1.0 | $(0, \infty)$ | Distance threshold multiplier |
| `threshold_diff` | float | 1e-4 | $(0, 0.01)$ | Binary search convergence criterion |

#### Implementation Details

```python
class MinMaxAdversary(Adversary):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold
        self.threshold_diff = 1e-4
        self.negative_indices = None  # For SignGuard

    def _attack_by_binary_search(self, algorithm: Algorithm):
        benign_updates = self.get_benign_updates(algorithm)
        mean_grads = benign_updates.mean(dim=0)
        deviation = benign_updates.std(dim=0)

        # Compute distance threshold
        threshold = torch.cdist(benign_updates, benign_updates, p=2).max()

        # SignGuard special case
        if isinstance(algorithm.server.aggregator, Signguard):
            if self.negative_indices is None:
                num_elements = len(deviation)
                num_negate = num_elements // 2
                self.negative_indices = random.sample(
                    range(num_negate), num_negate
                )
            deviation[self.negative_indices] *= -1

        # Binary search
        low = 0
        high = 5
        while abs(high - low) > 0.01:
            mid = (low + high) / 2
            mal_update = torch.stack([mean_grads - mid * deviation])
            loss = torch.cdist(mal_update, benign_updates, p=2).max()

            if loss < threshold:
                low = mid
            else:
                high = mid

        return mean_grads - mid * deviation
```

#### Configuration Example

```yaml
adversary_config:
  type: blades.adversaries.MinMaxAdversary
  threshold: 1.0
```

#### Attack Effectiveness

- **Against FedAvg**: Very High
- **Against Median**: High
- **Against Trimmed-Mean**: High
- **Against Geometric Median**: Medium-High
- **Against Krum**: Very High (designed to bypass distance detection)
- **Against ABR-FL**: Medium (client verification may catch it)

#### Theoretical Analysis

**Key insight**: The attack finds the sweet spot where it:
1. Maximizes deviation from benign mean
2. Stays within the benign distance threshold

**Computational complexity**: $O(K \cdot N \cdot d)$ where:
- $K$ = binary search iterations $\approx \log_2(5/0.01) \approx 9$
- $N$ = number of benign clients
- $d$ = model dimension

**Expected attack success**: High for distance-based defenses (Krum) because the malicious update appears to be within the benign cluster.

---

## Server-Side Attacks

### 8. Malicious Server Adversary

**File**: `malicious_server_adversary.py`
**Difficulty**: ⭐⭐⭐ Medium
**Type**: Server-Level Byzantine Attack
**Context**: ABR-FL Multi-Server Architecture

#### Description

In ABR-FL's multi-server setting, this adversary simulates Byzantine servers that manipulate the aggregation process. Unlike client-side attacks, server attacks have more power as they directly control the aggregated model sent to clients.

#### Attack Types

##### 8a. Random Model Attack

**Procedure**:
```
1. Receive aggregated model from benign aggregation
2. Replace with random Gaussian noise:
   w_mal = N(0, noise_scale²)
3. Return w_mal to clients
```

**Mathematical Formulation**:
$$w_{\text{server}}^{\text{mal}} \sim \mathcal{N}(\mathbf{0}, \text{noise\_scale}^2 \cdot \mathbf{I}_d)$$

##### 8b. Noise Attack

**Procedure**:
```
1. Compute benign aggregation: w_agg
2. Add large-scale noise:
   w_mal = w_agg + N(0, noise_scale²)
3. Return w_mal
```

**Mathematical Formulation**:
$$w_{\text{server}}^{\text{mal}} = w_{\text{agg}}^{\text{benign}} + \mathcal{N}(\mathbf{0}, \text{noise\_scale}^2 \cdot \mathbf{I}_d)$$

##### 8c. Delayed Model Attack

**Procedure**:
```
1. Maintain history: H = [w_t, w_{t-1}, ..., w_{t-delay}]
2. Return stale model:
   w_mal = H[delay_rounds]
3. Update history with current model
```

**Mathematical Formulation**:
$$w_{\text{server}}^{\text{mal}}(t) = w_{\text{server}}^{\text{benign}}(t - \Delta)$$

where $\Delta$ = `delay_rounds`.

##### 8d. Inverted Aggregation Attack

**Procedure**:
```
1. Compute benign aggregation: w_agg
2. Return negative:
   w_mal = -w_agg
```

**Mathematical Formulation**:
$$w_{\text{server}}^{\text{mal}} = -w_{\text{agg}}^{\text{benign}}$$

#### Hyperparameters

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `attack_type` | str | 'random' | {'random', 'noise', 'delayed', 'inverted'} | Type of server attack |
| `noise_scale` | float | 10.0 | $(0, \infty)$ | Scale factor for noise attacks |
| `delay_rounds` | int | 5 | $[1, \infty)$ | Delay for stale model attack |

#### Implementation Details

```python
class MaliciousServerAdversary(TrainerCallback):
    def __init__(
        self,
        attack_type: str = 'random',
        noise_scale: float = 10.0,
        delay_rounds: int = 5
    ):
        self.attack_type = attack_type
        self.noise_scale = noise_scale
        self.delay_rounds = delay_rounds
        self.model_history = []

    def on_aggregation_end(self, trainer, server_id: int,
                          aggregated_model: torch.Tensor):
        # Only attack if this is a malicious server
        if server_id not in self.malicious_server_ids:
            return aggregated_model

        # Apply attack based on type
        if self.attack_type == 'random':
            return torch.randn_like(aggregated_model) * self.noise_scale

        elif self.attack_type == 'noise':
            noise = torch.randn_like(aggregated_model) * self.noise_scale
            return aggregated_model + noise

        elif self.attack_type == 'delayed':
            self.model_history.append(aggregated_model.clone())
            if len(self.model_history) > self.delay_rounds:
                self.model_history.pop(0)

            if len(self.model_history) >= self.delay_rounds:
                return self.model_history[0]
            else:
                return aggregated_model

        elif self.attack_type == 'inverted':
            return -aggregated_model
```

#### Configuration Example

```yaml
# Random model attack
adversary_config:
  type: blades.adversaries.MaliciousServerAdversary
  attack_type: 'random'
  noise_scale: 10.0

# Delayed model attack
adversary_config:
  type: blades.adversaries.MaliciousServerAdversary
  attack_type: 'delayed'
  delay_rounds: 5

# Inverted aggregation attack
adversary_config:
  type: blades.adversaries.MaliciousServerAdversary
  attack_type: 'inverted'
```

#### Attack Effectiveness

**Against different client verification mechanisms**:

| Defense | Random | Noise | Delayed | Inverted |
|---------|--------|-------|---------|----------|
| No Verification | Very High | Very High | Medium | Very High |
| Loss-based Verification | High | High | Low | High |
| **BALANCE Verification** | **Low** | **Low** | **Very Low** | **Low** |
| FLTrust (root dataset) | Low | Low | Low | Low |

**Key insight**: Server attacks are powerful but easily detected by client-side verification (ABR-FL's BALANCE algorithm).

#### Theoretical Analysis

**Expected client rejection rate** with BALANCE verification:

For random/noise/inverted attacks:
$$P(\text{reject}) \approx 1 - \epsilon$$

where $\epsilon \approx 0.01$ (very small).

**For delayed attack** at round $t$ with delay $\Delta$:
$$P(\text{reject}) = f(\Delta, \text{learning\_rate}, t)$$

Typically low initially, increases as model diverges.

**Impact on system** (with $m_p$ fraction of malicious servers):
- **Without client verification**: System fails if $m_p > 0$
- **With BALANCE verification**: Clients reject malicious updates, staleness increases by $\approx m_p$

Expected staleness:
$$\mathbb{E}[\tau] = \frac{m_p^{\text{eff}}}{1 - m_p^{\text{eff}}}$$

---

### 9. Adaptive Malicious Server

**File**: `malicious_server_adversary.py` (AdaptiveMaliciousServerAdversary class)
**Difficulty**: ⭐⭐⭐⭐ High
**Type**: Adaptive Server-Level Attack

#### Description

An enhanced version of the malicious server that monitors client rejection rates and adapts its attack strategy to be more stealthy when detection rates are high.

#### Formal Algorithm

**Input**:
- Initial attack type: $A_{\text{init}}$
- Fallback attack type: $A_{\text{fallback}}$
- Detection threshold: $\theta$ (default: 0.5)

**State**:
- Current attack: $A_{\text{current}}$
- Client rejection rate: $r_{\text{reject}}$

**Procedure**:
```
Initialize: A_current = A_init

At each round t:
  1. Perform aggregation with A_current

  2. Observe client rejection rate: r_reject

  3. Adapt strategy:
     If r_reject > θ:  // High rejection, detected!
        If A_current != A_fallback:
           A_current = A_fallback
           Print("Switching to stealthy attack")
     Else:  // Low rejection, not detected
        If A_current != A_init:
           A_current = A_init
           Print("Switching back to aggressive attack")

  4. Update attack type for next round
```

#### Mathematical Formulation

**Decision rule**:
$$A_{\text{current}}(t+1) = \begin{cases}
A_{\text{fallback}} & \text{if } r_{\text{reject}}(t) > \theta \\
A_{\text{init}} & \text{if } r_{\text{reject}}(t) \leq \theta
\end{cases}$$

**Attack effectiveness vs. detection trade-off**:
$$\text{Utility} = \alpha \cdot \text{Attack\_Impact} - (1-\alpha) \cdot \text{Detection\_Risk}$$

Adaptive attack optimizes this trade-off by switching strategies.

#### Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_attack` | str | 'random' | Starting attack type (aggressive) |
| `fallback_attack` | str | 'noise' | Stealthier attack when detected |
| `detection_threshold` | float | 0.5 | Client rejection rate threshold |
| `noise_scale` | float | 10.0 | Noise scale for attacks |

**Recommended combinations**:
- Aggressive → Stealthy: `initial='inverted'`, `fallback='noise'`
- Random → Delayed: `initial='random'`, `fallback='delayed'`

#### Implementation Details

```python
class AdaptiveMaliciousServerAdversary(MaliciousServerAdversary):
    def __init__(
        self,
        initial_attack: str = 'random',
        fallback_attack: str = 'noise',
        detection_threshold: float = 0.5,
        noise_scale: float = 10.0
    ):
        super().__init__(attack_type=initial_attack, noise_scale=noise_scale)
        self.initial_attack = initial_attack
        self.fallback_attack = fallback_attack
        self.detection_threshold = detection_threshold
        self.current_attack = initial_attack

    def on_round_end(self, trainer):
        """Adapt attack strategy based on client behavior."""
        # Get ABR statistics
        if hasattr(trainer, 'get_abr_statistics'):
            stats = trainer.get_abr_statistics()
            rejection_rate = stats.get('client_rejection_rate', 0.0)

            # High rejection → switch to stealthy
            if rejection_rate > self.detection_threshold:
                if self.current_attack != self.fallback_attack:
                    print(f"[AdaptiveMaliciousServer] High rejection "
                          f"({rejection_rate:.2f}), switching to "
                          f"{self.fallback_attack}")
                    self.current_attack = self.fallback_attack
                    self.attack_type = self.fallback_attack

            # Low rejection → can use aggressive
            else:
                if self.current_attack != self.initial_attack:
                    print(f"[AdaptiveMaliciousServer] Low rejection "
                          f"({rejection_rate:.2f}), switching back to "
                          f"{self.initial_attack}")
                    self.current_attack = self.initial_attack
                    self.attack_type = self.initial_attack
```

#### Configuration Example

```yaml
adversary_config:
  type: blades.adversaries.AdaptiveMaliciousServerAdversary
  initial_attack: 'random'
  fallback_attack: 'delayed'
  detection_threshold: 0.3
  noise_scale: 10.0
```

#### Attack Effectiveness

**Compared to static malicious server**:

| Metric | Static (Random) | Adaptive (Random→Delayed) |
|--------|----------------|---------------------------|
| Initial Impact | Very High | Very High |
| Client Rejection Rate | 80-90% | 20-30% (after adaptation) |
| Long-term Impact | Low (clients reject) | Medium (stealthy but active) |
| Detection Time | Immediate | Delayed |

#### Theoretical Analysis

**Expected behavior over time**:

**Phase 1** (Rounds 1-5): Aggressive attack
- $r_{\text{reject}} \approx 0.8$
- High detection, high impact

**Phase 2** (Rounds 6+): Adaptation triggered
- Switch to stealthy attack
- $r_{\text{reject}}$ drops to $\approx 0.2$
- Lower detection, moderate impact

**Equilibrium**: The attack finds a balance where:
$$r_{\text{reject}} \approx \theta$$

This maximizes long-term impact while avoiding complete rejection.

---

## Advanced Attacks (Targeting Geometric Median)

### 10. Min-Sum Adversary

**File**: `adaptive_attacks.py`
**Difficulty**: ⭐⭐⭐⭐⭐ Very High
**Type**: Optimization-based Attack
**Reference**: "Fall of Empires" (Xie et al., 2020)
**Target**: Geometric Median Aggregation

#### Description

Min-Sum attack is a sophisticated optimization-based attack that places malicious updates on the Voronoi boundary to maximize the geometric median's deviation from the true benign mean. This is one of the most effective attacks against geometric median.

#### Formal Algorithm

**Input**:
- Benign updates: $\mathcal{W}_{\text{benign}} \in \mathbb{R}^{(N-m) \times d}$
- Scale factor: $\lambda$ (default: 1.5)
- Optimization steps: $K$ (default: 10)
- Learning rate: $\eta$ (default: 0.1)

**Output**:
- Malicious update: $w_{\text{mal}} \in \mathbb{R}^d$

**Procedure**:
```
1. Compute benign statistics:
   μ = (1/(N-m)) Σ w_i
   σ = sqrt((1/(N-m)) Σ ||w_i - μ||²)

2. Initialize malicious update:
   w_mal = μ - λ × σ  // Opposite direction

3. Optimize via gradient descent:
   For k = 1 to K:
       // Objective: minimize sum of distances
       L = Σ_{i=1}^{N-m} ||w_mal - w_i||₂

       // Gradient
       ∇L = Σ_{i=1}^{N-m} (w_mal - w_i) / ||w_mal - w_i||₂

       // Update
       w_mal ← w_mal - η × ∇L

4. Return w_mal
```

#### Mathematical Formulation

**Objective function**:
$$\min_{w_{\text{mal}}} \mathcal{L}(w_{\text{mal}}) = \sum_{i=1}^{N-m} \|w_{\text{mal}} - w_i\|_2$$

**Gradient**:
$$\nabla \mathcal{L}(w_{\text{mal}}) = \sum_{i=1}^{N-m} \frac{w_{\text{mal}} - w_i}{\|w_{\text{mal}} - w_i\|_2}$$

**Update rule**:
$$w_{\text{mal}}^{(k+1)} = w_{\text{mal}}^{(k)} - \eta \cdot \nabla \mathcal{L}(w_{\text{mal}}^{(k)})$$

**Initialization**:
$$w_{\text{mal}}^{(0)} = \mu_{\text{benign}} - \lambda \cdot \sigma_{\text{benign}}$$

**Theoretical justification**: By minimizing the sum of distances, the malicious update is pushed toward the Voronoi boundary, which is the optimal location to influence the geometric median.

#### Hyperparameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `scale_factor` | float | 1.5 | $[1.0, 5.0]$ | Initial deviation scale |
| `optimization_steps` | int | 10 | $[5, 50]$ | Number of gradient steps |
| `learning_rate` | float | 0.1 | $(0, 1.0]$ | Optimization learning rate |

**Tuning guidelines**:
- **High $\lambda$** (3.0-5.0): More aggressive, may be detected
- **Low $\lambda$** (1.0-1.5): Stealthier, less effective
- **More optimization steps**: Better convergence, higher computational cost

#### Implementation Details

```python
class MinSumAdversary(Adversary):
    def __init__(
        self,
        scale_factor: float = 1.5,
        optimization_steps: int = 10,
        learning_rate: float = 0.1
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.optimization_steps = optimization_steps
        self.learning_rate = learning_rate

    def _optimize_min_sum(
        self,
        benign_updates: torch.Tensor,
        benign_mean: torch.Tensor,
        benign_std: torch.Tensor
    ) -> torch.Tensor:
        # Initialize
        mal_update = benign_mean - self.scale_factor * benign_std
        mal_update.requires_grad = True

        optimizer = torch.optim.SGD([mal_update], lr=self.learning_rate)

        for step in range(self.optimization_steps):
            optimizer.zero_grad()

            # Objective: minimize sum of distances
            distances = torch.norm(
                benign_updates - mal_update.unsqueeze(0), dim=1
            )
            loss = distances.sum()

            loss.backward()
            optimizer.step()

        return mal_update.detach()
```

#### Configuration Example

```yaml
adversary_config:
  type: blades.adversaries.adaptive_attacks.MinSumAdversary
  scale_factor: 1.5
  optimization_steps: 10
  learning_rate: 0.1
```

#### Attack Effectiveness

- **Against FedAvg**: Very High
- **Against Geometric Median**: **Very High** (designed specifically for it)
- **Against Krum**: Medium-High
- **Against Trimmed-Mean**: High
- **Against ABR-FL (with BALANCE)**: Medium (client verification helps)

#### Theoretical Analysis

**Breakdown point of Geometric Median**: Approximately $m < N/2$.

**Expected deviation** introduced by Min-Sum attack:
$$\Delta \approx \frac{m}{N-m} \cdot \lambda \cdot \sigma_{\text{benign}}$$

**Computational complexity**: $O(K \cdot (N-m) \cdot d)$ where:
- $K$ = optimization steps
- $(N-m)$ = benign clients
- $d$ = model dimension

**Why it works**: The geometric median is the point minimizing sum of distances. By placing $m$ malicious updates at the location that minimizes distances to benign updates, the attack "pulls" the geometric median toward that location.

**Critical insight**: The attack approximates the Fall of Empires strategy, which is proven to be optimal under certain conditions.

---

### 11. Inner-Max Adversary

**File**: `adaptive_attacks.py`
**Difficulty**: ⭐⭐⭐⭐ High
**Type**: Inner Product Maximization Attack
**Target**: Geometric Median

#### Description

Inner-Max attack crafts malicious updates to maximize the negative inner product with the benign gradient direction, effectively pushing the aggregated model in the wrong direction. The sign-based version is particularly robust.

#### Formal Algorithm

**Input**:
- Benign updates: $\mathcal{W}_{\text{benign}}$
- Scale factor: $\lambda$ (default: 3.0)
- Use sign: bool (default: True)

**Output**:
- Malicious update: $w_{\text{mal}}$

**Procedure**:
```
1. Compute benign mean:
   μ = (1/(N-m)) Σ w_i

2. Craft malicious update:
   If use_sign:
       w_mal = -λ × sign(μ)
   Else:
       w_mal = -λ × μ

3. Return w_mal
```

#### Mathematical Formulation

**Objective**: Maximize negative inner product:
$$\max_{w_{\text{mal}}} \langle w_{\text{mal}}, -\mu_{\text{benign}} \rangle$$

**Solution**:

**Version 1** (Sign-based):
$$w_{\text{mal}} = -\lambda \cdot \text{sign}(\mu_{\text{benign}})$$

where $\text{sign}(x)_i = \begin{cases} +1 & \text{if } x_i > 0 \\ -1 & \text{if } x_i < 0 \\ 0 & \text{if } x_i = 0 \end{cases}$

**Version 2** (Magnitude-based):
$$w_{\text{mal}} = -\lambda \cdot \mu_{\text{benign}}$$

**Inner product**:
$$\langle w_{\text{mal}}, \mu_{\text{benign}} \rangle = -\lambda \|\mu_{\text{benign}}\|_1 \quad \text{(sign-based)}$$
$$\langle w_{\text{mal}}, \mu_{\text{benign}} \rangle = -\lambda \|\mu_{\text{benign}}\|_2^2 \quad \text{(magnitude-based)}$$

Both are maximally negative.

#### Hyperparameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `scale_factor` | float | 3.0 | $[1.0, 10.0]$ | Attack magnitude |
| `use_sign` | bool | True | {True, False} | Use sign-based (more robust) or magnitude-based |

**Recommendation**: Use `use_sign=True` for better robustness against defenses.

#### Implementation Details

```python
class InnerMaxAdversary(Adversary):
    def __init__(
        self,
        scale_factor: float = 3.0,
        use_sign: bool = True
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.use_sign = use_sign

    def on_local_round_end(self, trainer: Trainer):
        benign_updates = self.get_benign_updates(trainer)
        benign_mean = benign_updates.mean(dim=0)

        # Maximize negative inner product
        if self.use_sign:
            malicious_update = -self.scale_factor * torch.sign(benign_mean)
        else:
            malicious_update = -self.scale_factor * benign_mean

        # Apply to all malicious clients
        for result in trainer.local_results:
            client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                result[CLIENT_UPDATE] = malicious_update.clone()
```

#### Configuration Example

```yaml
# Sign-based (recommended)
adversary_config:
  type: blades.adversaries.adaptive_attacks.InnerMaxAdversary
  scale_factor: 3.0
  use_sign: true

# Magnitude-based
adversary_config:
  type: blades.adversaries.adaptive_attacks.InnerMaxAdversary
  scale_factor: 3.0
  use_sign: false
```

#### Attack Effectiveness

- **Against FedAvg**: Very High
- **Against Geometric Median**: **High** (effective but not optimal)
- **Against Krum**: Medium (detected as outlier)
- **Against Trimmed-Mean**: High
- **Against Median**: High

**Sign-based vs. Magnitude-based**:
- **Sign-based**: More robust, harder to filter by magnitude-based defenses
- **Magnitude-based**: Potentially stronger but easier to detect

#### Theoretical Analysis

**Expected impact** on aggregation (assuming Mean aggregator):
$$w_{\text{agg}} = \frac{(N-m) \mu + m(-\lambda \mu)}{N} = \left(1 - \frac{m(1+\lambda)}{N}\right) \mu$$

For $m = 0.3N$ and $\lambda = 3.0$:
$$w_{\text{agg}} = (1 - 1.2) \mu = -0.2 \mu$$

The aggregated model points in the **opposite direction**!

**Against Geometric Median**: The attack is less effective because GeoMed is not as influenced by magnitude. However, with $m$ approaching $N/2$, the attack can still succeed.

---

### 12. Min-Max Adversary (Hybrid)

**File**: `adaptive_attacks.py`
**Difficulty**: ⭐⭐⭐⭐⭐ Very High
**Type**: Combined Optimization Attack
**Target**: Geometric Median

#### Description

Min-Max combines the Min-Sum and Inner-Max strategies into a unified optimization framework, creating the most sophisticated client-side attack against geometric median. It balances distance minimization and inner product maximization.

#### Formal Algorithm

**Input**:
- Benign updates: $\mathcal{W}_{\text{benign}}$
- Trade-off parameter: $\alpha \in [0, 1]$ (default: 0.5)
- Scale factor: $\lambda$ (default: 2.0)
- Optimization steps: $K$ (default: 20)
- Learning rate: $\eta$ (default: 0.1)

**Output**:
- Malicious update: $w_{\text{mal}}$

**Procedure**:
```
1. Compute benign mean: μ = mean(W_benign)

2. Initialize: w_mal = -λ × μ

3. For k = 1 to K:

   // Min-sum component
   L_minsum = Σ ||w_mal - w_i||₂

   // Inner-max component
   L_innermax = -⟨w_mal, μ⟩

   // Combined loss
   L = α × L_minsum + (1-α) × L_innermax

   // Gradient descent
   ∇L = compute_gradient(L)
   w_mal ← w_mal - η × ∇L

4. Return w_mal
```

#### Mathematical Formulation

**Combined objective function**:
$$\mathcal{L}(w_{\text{mal}}) = \alpha \sum_{i=1}^{N-m} \|w_{\text{mal}} - w_i\|_2 - (1-\alpha) \langle w_{\text{mal}}, \mu_{\text{benign}} \rangle$$

where:
- First term: Min-Sum (distance minimization)
- Second term: Inner-Max (negative inner product maximization)
- $\alpha$: Trade-off parameter

**Gradient**:
$$\nabla \mathcal{L} = \alpha \sum_{i=1}^{N-m} \frac{w_{\text{mal}} - w_i}{\|w_{\text{mal}} - w_i\|_2} - (1-\alpha) \mu_{\text{benign}}$$

**Update rule**:
$$w_{\text{mal}}^{(k+1)} = w_{\text{mal}}^{(k)} - \eta \cdot \nabla \mathcal{L}(w_{\text{mal}}^{(k)})$$

**Initialization**:
$$w_{\text{mal}}^{(0)} = -\lambda \cdot \mu_{\text{benign}}$$

#### Hyperparameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `alpha` | float | 0.5 | $[0, 1]$ | Trade-off: min-sum (α) vs inner-max (1-α) |
| `scale_factor` | float | 2.0 | $[1.0, 5.0]$ | Initial attack magnitude |
| `optimization_steps` | int | 20 | $[10, 100]$ | Number of gradient steps |
| `learning_rate` | float | 0.1 | $(0, 1.0]$ | Optimization learning rate |

**Parameter tuning**:
- **$\alpha = 0$**: Pure Inner-Max attack
- **$\alpha = 0.5$**: Balanced (default, recommended)
- **$\alpha = 1$**: Pure Min-Sum attack

#### Implementation Details

```python
class MinMaxAdversary(Adversary):
    def __init__(
        self,
        alpha: float = 0.5,
        scale_factor: float = 2.0,
        optimization_steps: int = 20,
        learning_rate: float = 0.1
    ):
        super().__init__()
        self.alpha = alpha
        self.scale_factor = scale_factor
        self.optimization_steps = optimization_steps
        self.learning_rate = learning_rate

    def on_local_round_end(self, trainer: Trainer):
        benign_updates = self.get_benign_updates(trainer)
        benign_mean = benign_updates.mean(dim=0)

        # Initialize from negative direction
        mal_update = -self.scale_factor * benign_mean
        mal_update.requires_grad = True

        optimizer = torch.optim.SGD([mal_update], lr=self.learning_rate)

        for step in range(self.optimization_steps):
            optimizer.zero_grad()

            # Min-sum component
            distances = torch.norm(
                benign_updates - mal_update.unsqueeze(0), dim=1
            )
            min_sum_loss = distances.sum()

            # Inner-max component
            inner_product = torch.dot(
                mal_update.flatten(), benign_mean.flatten()
            )
            inner_max_loss = -inner_product

            # Combined loss
            loss = self.alpha * min_sum_loss + (1 - self.alpha) * inner_max_loss

            loss.backward()
            optimizer.step()

        # Apply to malicious clients
        malicious_update = mal_update.detach()
        for result in trainer.local_results:
            client = trainer.client_manager.get_client_by_id(result[CLIENT_ID])
            if client.is_malicious:
                result[CLIENT_UPDATE] = malicious_update.clone()
```

#### Configuration Example

```yaml
# Balanced attack (recommended)
adversary_config:
  type: blades.adversaries.adaptive_attacks.MinMaxAdversary
  alpha: 0.5
  scale_factor: 2.0
  optimization_steps: 20
  learning_rate: 0.1

# More Min-Sum focused
adversary_config:
  type: blades.adversaries.adaptive_attacks.MinMaxAdversary
  alpha: 0.8
  scale_factor: 1.5
  optimization_steps: 30
  learning_rate: 0.05
```

#### Attack Effectiveness

- **Against FedAvg**: Very High
- **Against Geometric Median**: **Very High** (most effective client-side attack)
- **Against Krum**: High
- **Against Trimmed-Mean**: High
- **Against Median**: High
- **Against ABR-FL**: Medium-High (best chance among client attacks)

**Ranking of GeoMed attacks**:
1. **Min-Max** (this): Best balance
2. Min-Sum: Strong but can be detected
3. Inner-Max: Effective but not optimal

#### Theoretical Analysis

**Computational complexity**: $O(K \cdot (N-m) \cdot d)$ where $K$ = optimization steps.

**Expected attack success**: The combined optimization finds a better local optimum than either strategy alone because:
- Min-Sum component ensures staying close to benign cluster
- Inner-Max component ensures pushing in wrong direction

**Optimal $\alpha$ analysis** (empirical):
- For $m/N < 0.3$: Use $\alpha \approx 0.7$ (more Min-Sum)
- For $m/N \approx 0.3$: Use $\alpha \approx 0.5$ (balanced)
- For $m/N > 0.3$: Use $\alpha \approx 0.3$ (more Inner-Max)

**Why it's the best**: Combines two complementary objectives, allowing the attack to:
1. Stay within the benign distribution (avoid detection)
2. Maximize negative impact on convergence

---

### 13. AGR-Tailored Adversary

**File**: `adaptive_attacks.py`
**Difficulty**: ⭐⭐⭐⭐⭐ Very High
**Type**: Aggregation Rule Tailored Adaptive Attack
**Target**: Any Aggregator (Auto-adapts)

#### Description

AGR-Tailored is a meta-attack that automatically detects the type of aggregation rule being used and selects the optimal attack strategy for that specific rule. This is a white-box attack assuming the adversary knows the defense mechanism.

#### Formal Algorithm

**Input**:
- Benign updates: $\mathcal{W}_{\text{benign}}$
- Aggregator type: $A$ (auto-detected or specified)
- Auto-detect: bool (default: True)

**Output**:
- Malicious update: $w_{\text{mal}}$

**Procedure**:
```
1. Detect aggregator type:
   If auto_detect:
       A = detect_aggregator_type(trainer)

2. Select attack strategy based on A:

   If A == 'geomed':
       w_mal = attack_geomed(W_benign)

   Else if A == 'krum':
       w_mal = attack_krum(W_benign)

   Else if A == 'trimmedmean':
       w_mal = attack_trimmedmean(W_benign)

   Else:
       w_mal = fallback_attack(W_benign)

3. Return w_mal
```

#### Attack Strategies per Aggregator

##### Strategy 1: Attack Geometric Median

```
Procedure:
1. μ = mean(W_benign)
2. σ = std(W_benign)
3. w_mal = μ - 2.0 × σ
4. Return w_mal
```

**Mathematical formulation**:
$$w_{\text{mal}} = \mu_{\text{benign}} - 2.0 \cdot \sigma_{\text{benign}}$$

This is a simplified Min-Sum approximation.

##### Strategy 2: Attack Krum

```
Procedure:
1. μ = mean(W_benign)
2. ε ~ N(0, I)
3. w_mal = μ + 5.0 × ε
4. Return w_mal
```

**Mathematical formulation**:
$$w_{\text{mal}} = \mu_{\text{benign}} + 5.0 \cdot \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$$

**Rationale**: Place update far from benign cluster to avoid being selected by Krum.

##### Strategy 3: Attack Trimmed-Mean

```
Procedure:
1. μ = mean(W_benign)
2. σ = std(W_benign)
3. z_max = 1.5  // Below trimming threshold
4. w_mal = μ + z_max × σ
5. Return w_mal
```

**Mathematical formulation**:
$$w_{\text{mal}} = \mu_{\text{benign}} + 1.5 \cdot \sigma_{\text{benign}}$$

**Rationale**: ALIE-like strategy to stay within trimming bounds.

#### Hyperparameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aggregator_type` | str | None | Manual specification ('geomed', 'krum', 'trimmedmean') |
| `auto_detect` | bool | True | Automatically detect aggregator type |

#### Implementation Details

```python
class AGRTailoredAdversary(Adversary):
    def __init__(
        self,
        aggregator_type: Optional[str] = None,
        auto_detect: bool = True,
        **kwargs
    ):
        super().__init__()
        self.aggregator_type = aggregator_type
        self.auto_detect = auto_detect

    def setup(self, trainer):
        super().setup(trainer)

        if self.auto_detect and self.aggregator_type is None:
            # Auto-detect from trainer
            if hasattr(trainer.server, 'aggregator'):
                agg_class_name = trainer.server.aggregator.__class__.__name__.lower()

                if 'geomed' in agg_class_name or 'geometric' in agg_class_name:
                    self.aggregator_type = 'geomed'
                elif 'krum' in agg_class_name:
                    self.aggregator_type = 'krum'
                elif 'trim' in agg_class_name:
                    self.aggregator_type = 'trimmedmean'
                else:
                    self.aggregator_type = 'geomed'  # Default

                print(f"[AGRTailored] Detected: {self.aggregator_type}")

    def _attack_geomed(self, benign_updates: torch.Tensor) -> torch.Tensor:
        mean = benign_updates.mean(dim=0)
        std = benign_updates.std(dim=0)
        return mean - 2.0 * std

    def _attack_krum(self, benign_updates: torch.Tensor) -> torch.Tensor:
        mean = benign_updates.mean(dim=0)
        return mean + 5.0 * torch.randn_like(mean)

    def _attack_trimmedmean(self, benign_updates: torch.Tensor) -> torch.Tensor:
        mean = benign_updates.mean(dim=0)
        std = benign_updates.std(dim=0)
        z_max = 1.5
        return mean + z_max * std
```

#### Configuration Example

```yaml
# Auto-detect mode (recommended)
adversary_config:
  type: blades.adversaries.adaptive_attacks.AGRTailoredAdversary
  auto_detect: true

# Manual specification
adversary_config:
  type: blades.adversaries.adaptive_attacks.AGRTailoredAdversary
  aggregator_type: 'geomed'
  auto_detect: false
```

#### Attack Effectiveness

**By aggregator**:

| Aggregator | Attack Strategy | Effectiveness |
|------------|----------------|---------------|
| **Geometric Median** | MinSum-like | High |
| **Krum** | Distance outlier | Medium-High |
| **Trimmed-Mean** | ALIE-like | High |
| **Median** | Same as Trimmed-Mean | High |
| **FedAvg** | Sign-flip fallback | Very High |

#### Theoretical Analysis

**Key insight**: This is a **meta-learner** attack that adapts to the defense. In practice, this represents the strongest adversary scenario (white-box with full knowledge).

**Attack complexity**: Depends on selected strategy, but generally $O((N-m) \cdot d)$ for simple strategies.

**Defense against AGR-Tailored**:
1. **Randomize aggregator**: Randomly switch between aggregators (makes detection harder)
2. **Client-side verification**: ABR-FL's BALANCE can still catch it
3. **Ensemble aggregation**: Use multiple aggregators and combine

---

## Attack Comparison Table

### By Difficulty and Effectiveness

| Attack | Difficulty | Against FedAvg | Against GeoMed | Against Krum | Against ABR-FL | Computational Cost |
|--------|-----------|---------------|---------------|-------------|---------------|-------------------|
| **Noise** | ⭐ | Very High | Low | Low | Low | O(d) |
| **Sign-Flipping** | ⭐⭐ | Very High | Medium | Medium | Low | O(E·d) |
| **Label-Flipping** | ⭐⭐ | Very High | Medium-High | Medium | Medium | O(E·d) |
| **IPM** | ⭐⭐⭐⭐ | Very High | Medium | Low | Low | O(N·d) |
| **ALIE** | ⭐⭐⭐⭐ | Very High | Medium | Medium-High | Low-Medium | O(N·d) |
| **Adaptive** | ⭐⭐⭐⭐ | Very High | Medium | Medium | Low-Medium | O(N·d) |
| **MinMax** | ⭐⭐⭐⭐⭐ | Very High | Medium-High | Very High | Medium | O(K·N·d) |
| **Malicious Server** | ⭐⭐⭐ | Very High | Very High | Very High | Low | O(d) |
| **Adaptive Server** | ⭐⭐⭐⭐ | Very High | High | High | Low-Medium | O(d) |
| **Min-Sum** | ⭐⭐⭐⭐⭐ | Very High | Very High | High | Medium-High | O(K·N·d) |
| **Inner-Max** | ⭐⭐⭐⭐ | Very High | High | Medium | Medium | O(N·d) |
| **Min-Max** | ⭐⭐⭐⭐⭐ | Very High | **Very High** | High | Medium-High | O(K·N·d) |
| **AGR-Tailored** | ⭐⭐⭐⭐⭐ | Very High | High | High | Medium | O(N·d) |

**Legend**:
- $E$ = local training epochs
- $K$ = optimization iterations
- $N$ = number of clients
- $d$ = model dimension

### By Attack Type

| Type | Attacks | Best Use Case |
|------|---------|---------------|
| **Random/Noise** | Noise, Malicious Server (Random) | Testing basic robustness |
| **Gradient Reversal** | Sign-Flipping, IPM, Inner-Max, Malicious Server (Inverted) | Testing directional defenses |
| **Data Poisoning** | Label-Flipping | Testing backdoor defenses |
| **Statistical Mimicry** | ALIE, Adaptive | Testing coordinate-wise defenses |
| **Optimization-based** | MinMax, Min-Sum, Min-Max | Testing geometric defenses |
| **Adaptive** | AGR-Tailored, Adaptive Server | Testing white-box scenarios |

---

## Configuration Examples

### Example 1: Basic Noise Attack

```yaml
run: ABR_FEDAVG

config:
  num_clients: 100
  num_malicious_clients: 30  # 30% Byzantine

  adversary_config:
    type: blades.adversaries.NoiseAdversary
    mean: 0.0
    std: 5.0  # High noise
```

### Example 2: ALIE Attack with Geometric Median Defense

```yaml
run: ABR_FEDAVG

config:
  num_clients: 100
  num_malicious_clients: 30

  server_config:
    aggregator:
      type: GeoMed  # Robust aggregator

  adversary_config:
    type: blades.adversaries.ALIEAdversary
    # No parameters (auto-computed)
```

### Example 3: Dual-Sided Attack (Malicious Clients + Servers)

```yaml
run: ABR_FEDAVG

config:
  num_clients: 100
  num_malicious_clients: 30  # 30% malicious clients

  num_servers: 10
  num_malicious_servers: 2  # 20% malicious servers

  # Client-side attack
  adversary_config:
    type: blades.adversaries.IPMAdversary
    scale: 1.0

  # Server-side attack
  server_adversary_config:
    type: blades.adversaries.MaliciousServerAdversary
    attack_type: 'random'
    noise_scale: 10.0

  # Defense: ABR-FL with client verification
  enable_client_verification: true
  balance_gamma: 1.0
  balance_kappa: 5.0
```

### Example 4: Advanced Min-Max Attack

```yaml
run: ABR_FEDAVG

config:
  num_clients: 100
  num_malicious_clients: 40  # High Byzantine ratio

  server_config:
    aggregator:
      type: GeoMed

  adversary_config:
    type: blades.adversaries.adaptive_attacks.MinMaxAdversary
    alpha: 0.5
    scale_factor: 2.0
    optimization_steps: 20
    learning_rate: 0.1
```

### Example 5: AGR-Tailored Auto-Adaptive Attack

```yaml
run: ABR_FEDAVG

config:
  num_clients: 100
  num_malicious_clients: 30

  server_config:
    aggregator:
      type: GeoMed  # Will be auto-detected

  adversary_config:
    type: blades.adversaries.adaptive_attacks.AGRTailoredAdversary
    auto_detect: true
    # Will automatically use Min-Sum-like attack for GeoMed
```

---

## References

### Academic Papers

1. **ALIE Attack**
   Baruch, G., Baruch, M., & Goldberg, Y. (2019). "A Little Is Enough: Circumventing Defenses For Distributed Learning". *NeurIPS 2019*.

2. **Fall of Empires (Min-Sum)**
   Xie, C., Huang, K., Chen, P. Y., & Li, B. (2020). "DBA: Distributed Backdoor Attacks against Federated Learning". *ICLR 2020*.

3. **IPM Attack**
   Shejwalkar, V., & Houmansadr, A. (2021). "Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning". *NDSS 2021*.

4. **Geometric Median Defense**
   Chen, Y., Su, L., & Xu, J. (2017). "Distributed Statistical Machine Learning in Adversarial Settings: Byzantine Gradient Descent". *ACM SIGMETRICS 2018*.

5. **Krum Defense**
   Blanchard, P., El Mhamdi, E. M., Guerraoui, R., & Stainer, J. (2017). "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent". *NeurIPS 2017*.

6. **ABR-FL**
   [Your Paper] "ABR-FL: Asymmetric Byzantine-Resilient Federated Learning with Dual-Sided Defense". *Submitted to IEEE TIFS*.

### Implementation References

- **BLADES Framework**: https://github.com/lishenghui/blades
- **FedLib**: Federated Learning Library (internal)

---

## Appendix: Attack Implementation Checklist

When implementing a new attack:

- [ ] Extend `Adversary` base class
- [ ] Implement `on_local_round_end()` or `on_trainer_init()`
- [ ] Access benign updates via `self.get_benign_updates(trainer)`
- [ ] Modify `result[CLIENT_UPDATE]` for malicious clients
- [ ] Add hyperparameter validation
- [ ] Write unit tests
- [ ] Add configuration example
- [ ] Update this documentation
- [ ] Benchmark against defenses

---

**Document End**

For questions or contributions, please refer to the main project README.
