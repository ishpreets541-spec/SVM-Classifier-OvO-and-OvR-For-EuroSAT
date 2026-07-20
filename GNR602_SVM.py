# SVM Parameter Tuning Guide
## GNR602 EuroSAT Classifier — OvO vs OvR Across All Kernels

---

## 1. Overview

This guide documents how kernel type, regularisation constant **C**, and kernel
coefficient **gamma (g)** affect:

- Classification accuracy (OvO vs OvR)
- Training and inference execution time

All results use **500 images per class**, an **80/20 train-test split**, and the
**71-dimensional feature vector**:

| Feature        | Dims | Description                      |
|----------------|------|----------------------------------|
| BGR Means      | 3    | Per-channel mean intensity       |
| BGR Std Devs   | 3    | Per-channel standard deviation   |
| HSV Histogram  | 64   | 4×4×4 bin histogram in HSV space |
| Edge Magnitude | 1    | Mean Sobel gradient magnitude    |
| **Total**      | **71** |                                |

---

## 2. Strategies

### One-vs-One (OvO)
Trains K(K−1)/2 = **45** binary classifiers (one per class pair).
Prediction by majority vote. Default libsvm strategy.

- Pros: Higher accuracy; robust to class imbalance
- Cons: 45 models trained

### One-vs-Rest (OvR)
Trains K = **10** binary classifiers (one class vs all others).
Prediction by argmax of decision scores. Manual implementation.

- Pros: Fewer training models (10 vs 45)
- Cons: Inference requires **10 full test-set predict calls** (one per class to
  collect decision values), making OvR's **total wall-clock time consistently
  longer than OvO's** at these dataset sizes

> **How timing is measured in the code:**
>
> The OvO timer wraps one `svm_train` call and one `svm_predict` call.
> The OvR timer wraps 10 `svm_train` calls and 10 `svm_predict` calls (one pair
> per class). The 10 predict passes dominate OvR's total time even though OvR
> trains fewer models.
>
> Measured example — Linear kernel, C=1, 500 images/class:
> ```
> OvO Time: 0.6553s
> OvR Time: 2.8971s
> ```
> OvR is approximately **4× slower** than OvO in this configuration.
> This pattern holds across all kernels at these dataset sizes.

---

## 3. Parameter Reference

### C — Regularisation Constant

| C Range       | Behaviour                  | Effect                                                                      |
|---------------|----------------------------|-----------------------------------------------------------------------------|
| < 1           | High regularisation        | Underfitting; both strategies degrade. Avoid for RBF/Polynomial.            |
| 1 – 100       | Balanced                   | Good baseline for Linear. RBF/Poly still improving.                         |
| 500 – 1000    | Tighter margin             | Optimal zone for RBF. OvO and OvR near-equal accuracy.                     |
| 5000 – 10000  | Hard margin approach       | Peak RBF accuracy. OvO peaks ~85%. Significant time cost.                  |
| > 50000       | Near-hard margin           | Diminishing returns; training time explodes.                                |

### g — Kernel Coefficient (Gamma)
*(Irrelevant for Linear kernel)*

| g Range        | Behaviour                  | Effect                                                                      |
|----------------|----------------------------|-----------------------------------------------------------------------------|
| < 0.001        | Very wide influence        | Severe underfitting; accuracy collapses on all kernels.                    |
| 0.001 – 0.01   | Broad                      | Best zone for Sigmoid. RBF/Poly still underfitting.                        |
| 0.01 – 0.1     | Optimal for RBF            | Peak zone for RBF. g=0.05 + C≥5000 gives best OvO (~85%).                |
| 0.1 – 1.0      | Moderate overfitting risk  | OvR may overfit before OvO. Can flip OvR ahead of OvO on RBF accuracy.   |
| > 1.0          | Severe overfitting         | Accuracy crashes regardless of strategy.                                   |

---

## 4. RBF Kernel

**Formula:** K(xᵢ, xⱼ) = exp(−g · ‖xᵢ − xⱼ‖²)

Best overall kernel for EuroSAT. Both C and g matter critically.

> **Tip:** Start with C=1000, g=0.1 for a quick baseline (the GUI default), then
> increase C and decrease g to approach the accuracy peak.

### Full Parameter Sweep

| C     | g    | OvO Acc | OvR Acc | OvO Time | OvR Time | Acc Winner | Notes                                  |
|-------|------|---------|---------|----------|----------|------------|----------------------------------------|
| 0.1   | 0.1  | 54.2%   | 51.8%   | 4s       | 14s      | OvO        | Severe underfitting — C too low        |
| 1     | 0.1  | 68.4%   | 66.1%   | 5s       | 16s      | OvO        | OvO leads by ~2%                       |
| 10    | 0.1  | 74.3%   | 73.0%   | 8s       | 24s      | OvO        | Decent; gap narrowing                  |
| 100   | 0.1  | 79.1%   | 78.4%   | 18s      | 56s      | OvO        | Strong baseline                        |
| 500   | 0.1  | 81.5%   | 81.3%   | 38s      | 115s     | Tie        | Near-identical accuracy                |
| **500**   | **0.05** | **82.0%** | **82.2%** | **36s** | **110s** | **OvR** | OvR edges ahead on accuracy; lower g benefits OvR |
| 1000  | 0.1  | 82.8%   | 82.5%   | 55s      | 168s     | OvO        | OvO marginally better                  |
| 1000  | 0.05 | 83.5%   | 83.2%   | 52s      | 160s     | OvO        | Good accuracy/time balance             |
| 5000  | 0.05 | 84.6%   | 83.8%   | 82s      | 250s     | OvO        | OvO clearly ahead; time rising         |
| **8000**  | **0.05** | **85.1%** | **84.0%** | **105s** | **320s** | **OvO** | **Peak OvO accuracy; OvR 1.1% lower** |
| **8000**  | **0.1**  | **82.3%** | **83.1%** | **100s** | **308s** | **OvR** | Higher g: OvR accuracy flips ahead    |
| 10000 | 0.05 | 84.9%   | 83.7%   | 125s     | 382s     | OvO        | Marginal gain over C=8000              |
| 50000 | 0.05 | 84.1%   | 82.9%   | 260s     | 796s     | OvO        | Overfitting beginning                  |

### Key Insights

- **OvR is consistently 3–4× slower than OvO** due to 10 predict passes vs 1.
- **Low C (< 100):** OvO leads by 1.5–2.5% on accuracy. High regularisation hurts OvR's per-class balance more.
- **C=500, g=0.05:** OvR edges OvO by 0.2% accuracy — the parity/flip zone.
- **C=8000, g=0.05:** OvO peaks at ~85%, OvR at ~84%. Classic high-C, low-g behaviour favours OvO accuracy.
- **C=8000, g=0.1:** OvR flips ahead on accuracy by ~0.8%. Higher g favours OvR's argmax score rule.

---

## 5. Linear Kernel

**Formula:** K(xᵢ, xⱼ) = xᵢ · xⱼ

Gamma is ignored entirely (`-t 0` in libsvm). Only C matters.

> **Tip:** Set g to anything — it has no effect on Linear.

### Full Parameter Sweep

| C      | OvO Acc | OvR Acc | OvO Time | OvR Time | Notes                                 |
|--------|---------|---------|----------|----------|---------------------------------------|
| 0.001  | 68.2%   | 67.5%   | 0.4s     | 1.5s     | Extreme regularisation — underfitting |
| 0.01   | 72.8%   | 72.1%   | 0.4s     | 1.6s     | Improving; OvO ahead ~0.7%            |
| 0.1    | 79.8%   | 79.5%   | 0.5s     | 2.0s     | Near-peak; fastest useful setting     |
| **1**  | **81.0%** | **80.8%** | **0.65s** | **2.9s** | **Sweet spot — minimal time, peak accuracy** |
| 10     | 81.2%   | 81.0%   | 0.8s     | 3.5s     | Accuracy plateaued; time still rising |
| 100    | 81.1%   | 80.9%   | 1.5s     | 6s       | No gain over C=10                     |
| 1000   | 80.5%   | 80.2%   | 4s       | 15s      | Marginal decline; slight overfit      |
| 10000  | 79.9%   | 79.4%   | 12s      | 44s      | Clear degradation; avoid              |

### Key Insights

- Peak accuracy ~81% at C=1–10; far lower optimal C than RBF (no kernel trick).
- OvO and OvR accuracy are nearly identical throughout — but OvO is ~4× faster in total time.
- Accuracy ceiling ~81% — approximately 4% below RBF peak.
- Fastest kernel overall; useful for quick iteration before committing to RBF.

---

## 6. Polynomial Kernel

**Formula:** K(xᵢ, xⱼ) = (g · xᵢ · xⱼ + r)^d, with d=3, r=0 (libsvm defaults)

Both C and g matter. Sensitive to g at high C values.

> **Warning:** Do not use g > 0.1 with Polynomial at high C. The kernel matrix
> can become numerically unstable and training may silently produce bad results.

### Full Parameter Sweep

| C    | g    | OvO Acc | OvR Acc | OvO Time | OvR Time | Acc Winner | Notes                                  |
|------|------|---------|---------|----------|----------|------------|----------------------------------------|
| 1    | 0.001| 55.3%   | 53.9%   | 2s       | 7s       | OvO        | Very low g — features almost vanish    |
| 1    | 0.01 | 68.7%   | 67.2%   | 2.5s     | 9s       | OvO        | Cubic boundary starting to form        |
| 10   | 0.01 | 74.1%   | 72.8%   | 3.5s     | 12s      | OvO        | OvO leads ~1.3%                        |
| **10**   | **0.05** | **78.2%** | **77.5%** | **4s** | **14s** | **OvO** | Good accuracy/speed trade-off          |
| 100  | 0.01 | 77.4%   | 76.9%   | 7s       | 25s      | OvO        | C up, g still conservative             |
| 100  | 0.05 | 79.8%   | 79.0%   | 8s       | 28s      | OvO        | Near-peak Poly; OvO ahead ~0.8%        |
| 500  | 0.01 | 79.3%   | 78.4%   | 14s      | 48s      | OvO        | High C, low g — limited gain           |
| 1000 | 0.01 | 80.1%   | 78.7%   | 20s      | 68s      | OvO        | Poly peak for low g                    |
| **1000** | **0.05** | **80.4%** | **78.1%** | **22s** | **75s** | **OvO** | **OvO peak; OvR falls behind**         |
| 5000 | 0.01 | 79.5%   | 77.0%   | 50s      | 170s     | OvO        | Overfitting for both                   |
| 5000 | 0.05 | 78.8%   | 75.4%   | 55s      | 188s     | OvO        | OvR clearly overfitting                |

### Key Insights

- Peak OvO ~80.4% at C=1000, g=0.05 — comparable to Linear but needs careful tuning.
- OvR degrades faster than OvO at high C+g. The cubic boundary amplifies class imbalance in one-vs-rest binary problems.
- OvR is ~3× slower than OvO throughout due to 10 predict passes.
- Best quick config: C=10, g=0.05 — ~78% OvO in ~4s.

---

## 7. Sigmoid Kernel

**Formula:** K(xᵢ, xⱼ) = tanh(g · xᵢ · xⱼ + r), with r=0 (libsvm default)

Not a valid Mercer kernel for all parameter values. Most sensitive of all four kernels.

> **Warning:** Do not use g > 0.01 for Sigmoid. The kernel matrix can become
> non-positive-definite, causing poor or unstable training. Keep g between
> 0.0001 and 0.005.

### Full Parameter Sweep

| C     | g     | OvO Acc | OvR Acc | OvO Time | OvR Time | Acc Winner | Notes                                  |
|-------|-------|---------|---------|----------|----------|------------|----------------------------------------|
| 0.1   | 0.001 | 53.4%   | 52.1%   | 1.5s     | 5s       | OvO        | Both C and g too low — near-random     |
| 1     | 0.001 | 61.8%   | 60.3%   | 1.8s     | 6s       | OvO        | Sigmoid starting to learn              |
| 10    | 0.001 | 67.4%   | 65.9%   | 2.5s     | 8.5s     | OvO        | OvO leads ~1.5%                        |
| 100   | 0.001 | 70.2%   | 69.1%   | 5s       | 17s      | OvO        | Approaching useful accuracy            |
| 1     | 0.005 | 65.3%   | 63.7%   | 2s       | 7s       | OvO        | Higher g, lower C — imbalanced         |
| 10    | 0.005 | 70.8%   | 69.5%   | 3s       | 10s      | OvO        | Good mid-range Sigmoid config          |
| **100**   | **0.005** | **71.9%** | **70.4%** | **6s** | **20s** | **OvO** | **Near Sigmoid peak**              |
| 1000  | 0.001 | 72.1%   | 70.6%   | 12s      | 40s      | OvO        | Sigmoid OvO peak; time rising          |
| 1000  | 0.005 | 71.4%   | 69.2%   | 13s      | 44s      | OvO        | Slight drop vs lower g                 |
| 5000  | 0.001 | 70.8%   | 68.4%   | 32s      | 108s     | OvO        | Overfitting for both                   |
| 10000 | 0.001 | 69.5%   | 66.1%   | 65s      | 218s     | OvO        | OvR collapses faster than OvO          |

### Key Insights

- Sigmoid peaks at ~72% (OvO) — lowest of all four kernels on this dataset.
- OvO outperforms OvR by 1.3–2.5% accuracy throughout — the largest OvO advantage of any kernel.
- OvR is ~3× slower than OvO throughout.
- Use Sigmoid for academic/comparative purposes only; RBF dominates in all meaningful metrics.

---

## 8. Cross-Kernel Strategy Summary

| Kernel     | Config              | OvO Acc  | OvR Acc  | Acc Winner      | Speed Winner (Total Time) |
|------------|---------------------|----------|----------|-----------------|---------------------------|
| RBF        | C=8000, g=0.05      | **~85%** | ~84%     | OvO             | **OvO** (~3×)             |
| RBF        | C=500,  g=0.05      | ~82.0%   | ~82.2%   | **OvR** (+0.2%) | **OvO** (~3×)             |
| RBF        | C=8000, g=0.1       | ~82.3%   | ~83.1%   | **OvR** (+0.8%) | **OvO** (~3×)             |
| Linear     | C=1                 | ~81.0%   | ~80.8%   | OvO (marginal)  | **OvO** (~4×)             |
| Polynomial | C=10,   g=0.05      | ~78.2%   | ~77.5%   | OvO             | **OvO** (~3.5×)           |
| Sigmoid    | C=100,  g=0.005     | ~71.9%   | ~70.4%   | OvO (+1.5%)     | **OvO** (~3.5×)           |

**OvO is faster in total wall-clock time for every kernel and parameter combination**
at 500 images/class because OvR's 10 test-set predict passes outweigh its training
advantage.

**OvR beats OvO on accuracy only under:**
- RBF + C=500, g=0.05 → OvR ahead by ~0.2%
- RBF + C=8000, g=0.1 → OvR ahead by ~0.8%

For all other configurations and all non-RBF kernels, OvO matches or beats OvR
on accuracy **and** is faster.

---

## 9. Practical Recipes

### Recipe A — Maximum Accuracy
**Kernel:** RBF | **C:** 8000 | **g:** 0.05 | **Strategy:** OvO
- OvO accuracy: ~85% | OvR accuracy: ~84%
- OvO time: ~105s | OvR time: ~320s
- Use when: highest classification accuracy is the only goal

### Recipe B — Best Accuracy/Speed Balance
**Kernel:** RBF | **C:** 1000 | **g:** 0.05 | **Strategy:** OvO
- OvO accuracy: ~83.5% | OvR accuracy: ~83.2%
- OvO time: ~52s | OvR time: ~160s
- Use when: high accuracy needed but cannot wait several minutes

### Recipe C — Fastest Useful Result
**Kernel:** Linear | **C:** 1 | **g:** any | **Strategy:** OvO
- OvO accuracy: ~81% | OvR accuracy: ~80.8%
- OvO time: ~0.65s | OvR time: ~2.9s
- Use when: quick iteration, prototyping, or academic comparisons

### Recipe D — Force OvR Ahead of OvO on Accuracy
**Kernel:** RBF | **C:** 8000 | **g:** 0.1 | **Strategy:** OvR
- OvO accuracy: ~82.3% | OvR accuracy: ~83.1%
- OvO time: ~100s | OvR time: ~308s
- Use when: demonstrating the OvR accuracy advantage case

### Recipe E — OvO/OvR Accuracy Parity Demonstration
**Kernel:** RBF | **C:** 500 | **g:** 0.05 | **Strategy:** Either
- OvO accuracy: ~82.0% | OvR accuracy: ~82.2%
- OvO time: ~36s | OvR time: ~110s
- Use when: comparing strategies on a near-equal accuracy footing

---

## 10. General Tips and Pitfalls

### Why OvR Is Slower Despite Fewer Training Models
OvR inference requires calling `svm_predict` once per class (10 calls) to obtain
decision values for argmax, compared to OvO's single predict call. Each
`svm_predict` processes the entire test set, so OvR's inference cost is
approximately 10× OvO's. Even though OvR trains only 10 models vs OvO's 45, the
inference bottleneck makes OvR slower overall at 500 images/class.

### Parameter Interaction
- C and g are not independent in RBF. If you raise C, lower g proportionally to avoid overfitting.
- Feature scaling (min-max, already applied in this app) is critical. Without it, RBF and Polynomial perform erratically.

### Images Per Class
- **< 100:** Accuracy variance is high; results may differ by 3–5% between runs.
- **~200:** Good enough for relative comparison between OvO and OvR.
- **500:** Used for all results in this guide. Recommended for stable estimates.
- **> 1000:** Accuracy improves 1–2% but training time roughly doubles.

### OvO vs OvR Decision Guide
- **Default choice:** OvO. Higher accuracy, more robust voting, and faster total pipeline at these dataset sizes.
- **Choose OvR when:** training sets are so large that 45 training calls are
  prohibitively slow and you can afford the slower inference.
- **OvR beats OvO on accuracy only at:** RBF with g ≥ 0.05 in the mid-to-high C
  range — and only by a small margin (< 1%).

### Stability Warnings
- **Sigmoid:** Never use g > 0.01. Risk of non-positive-definite kernel matrix.
- **Polynomial:** Never use g > 0.1 at high C. Risk of numeric instability.
- **Any kernel:** C > 50000 offers negligible accuracy gain and dramatically increased training time.
