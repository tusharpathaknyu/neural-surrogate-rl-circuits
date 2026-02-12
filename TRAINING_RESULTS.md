# Training Results — SPICE-in-the-Loop RL (Feb 12, 2026)

## Run Configuration

| Parameter | Value |
|-----------|-------|
| **Script** | `train_intensive_spice.py` |
| **Device** | CPU (MacBook Air M3) |
| **Total env steps** | 11,673,600 (11.7M) |
| **Wall time** | 11.47 hours |
| **Topologies** | 7 (buck, boost, buck_boost, sepic, cuk, flyback, qr_flyback) |
| **Episode length** | 50 steps |
| **SPICE rate** | 20% of episodes |
| **Trust dampening** | EMA α=0.3, reward × (0.5 + 0.5×trust) |
| **Per-topology bounds** | Yes (e.g., buck V_in 8–48V, flyback 12–400V) |

### Per-Topology Training Config

| Topology | Iterations | Steps/iter | Total steps | Time |
|----------|-----------|------------|-------------|------|
| buck | 300 | 2,048 | 614,400 | 40.3 min |
| boost | 500 | 4,096 | 2,048,000 | 114.4 min |
| buck_boost | 400 | 2,048 | 819,200 | 21.2 min |
| sepic | 500 | 4,096 | 2,048,000 | 164.4 min |
| cuk | 400 | 2,048 | 819,200 | 130.8 min |
| flyback | 600 | 4,096 | 2,457,600 | 95.3 min |
| qr_flyback | 700 | 4,096 | 2,867,200 | 121.5 min |

---

## Results Summary

### Final Test Metrics

| Topology | Surrogate MSE | ± Std | SPICE MSE | Best Training MSE | Best Reward | Verdict |
|----------|:------------:|:-----:|:---------:|:-----------------:|:-----------:|---------|
| **buck** | **3.1** | 1.4 | **4.2** | 1.16 | 99.4 | ✅ Excellent |
| **boost** | **3.8** | 2.8 | — | 1.22 | 4.0 | ✅ Excellent |
| **sepic** | **3.3** | 1.9 | — | 1.31 | 43.1 | ✅ Excellent |
| **cuk** | **3.5** | 1.1 | **4.3** | 1.48 | 176.8 | ✅ Excellent |
| qr_flyback | 47.1 | 83.3 | **2.2** | 8.56 | -78.0 | ⚠️ Fair |
| flyback | 170.6 | 470.6 | — | 5.61 | -89.9 | ❌ Needs work |
| buck_boost | 594.9 | 1538.8 | — | 144.3 | -207.1 | ❌ Needs work |

### Key Observations

**4/7 topologies achieved excellent results (MSE < 5):**
- Buck, boost, sepic, and cuk all converged to surrogate MSE ~3–4 with low variance
- SPICE validation MSE closely matches surrogate MSE (4.2–4.3), confirming no exploitation

**3 topologies need further investigation:**

1. **buck_boost (MSE=594.9):** Worst performer. Only trained 21.2 min (3,479 SPICE calls vs 40k for cuk). Negative best reward (-207) indicates the agent never found a viable policy. High variance (±1538) = erratic behavior.

2. **flyback (MSE=170.6):** Moderate failure. Best training MSE was actually 5.61 (good) but test MSE is 170.6 — suggests the best checkpoint doesn't generalize, or test targets differ from training targets. High variance (±470) = inconsistent.

3. **qr_flyback (MSE=47.1):** The SPICE MSE of **2.2** is excellent, while surrogate MSE is 47.1. This likely indicates the **surrogate is inaccurate** for qr_flyback rather than the policy being bad — the agent found good real-world parameters that the surrogate can't evaluate correctly.

---

## SPICE Exploitation Fix Validation

This run was the first with the surrogate exploitation fix (commit `0b5a971`). Comparing buck's SPICE metrics:

| Metric | Before fix (old run) | After fix (this run) |
|--------|---------------------|---------------------|
| wf_mse range | 1,672 – **52,782** | 478 – 3,182 |
| Best dc_err | 52% | **38.8%** |
| Best wf_mse | 1,672 | **478** |
| Max blowup | 52,782 | 3,182 |

The per-topology bounds and trust dampening successfully eliminated catastrophic SPICE divergence.

---

## Checkpoint Files

All saved to `checkpoints/`:
```
rl_agent_buck.pt          (2.8 MB)  ← Production ready
rl_agent_boost.pt         (10 MB)   ← Production ready
rl_agent_buck_boost.pt    (2.8 MB)  ← Needs retraining
rl_agent_sepic.pt         (10 MB)   ← Production ready
rl_agent_cuk.pt           (2.8 MB)  ← Production ready
rl_agent_flyback.pt       (10 MB)   ← Needs retraining
rl_agent_qr_flyback.pt    (10 MB)   ← Fair, surrogate issue
intensive_training_summary.json     ← Full JSON results
training_progress.json              ← Per-topology progress
```

---

## Next Steps

1. **Diagnose buck_boost failure** — Investigate why only 3,479 SPICE calls (vs 40k for similar cuk)
2. **Diagnose flyback generalization gap** — Training MSE 5.6 vs test MSE 170.6
3. **Improve qr_flyback surrogate** — SPICE MSE 2.2 but surrogate MSE 47.1
4. **Targeted retraining** for failing topologies with tuned hyperparameters
5. **Production run** — Longer training (2-3x iterations) for the 4 excellent topologies
