# Bug Fixes Documentation

A comprehensive record of all bugs found and fixed in the RL training pipeline for power electronics circuit optimization.

---

## Session 1: Initial Diagnosis (8 bugs + 1 extra fix)

### Context
After generating demo visualizations, it became clear that the RL agents performed **at or below random search level** across all topologies. A deep investigation of `rl/environment.py`, `rl/ppo_agent.py`, `rl/topology_rewards.py`, and `train_intensive_spice.py` revealed 8 interconnected bugs that collectively made learning impossible.

**Aggregate Impact:**  
- MSE: 130 → 0.298 (**439× improvement**)  
- Rewards: -6,959 → +7.47  
- RL vs Random win rate: 0% → 52%

---

### BUG 1: V_in Out of Parameter Bounds
**File:** `train_intensive_spice.py` — `TopologySpecificEnv.reset()`  
**Severity:** Medium

**Problem:** V_in was sampled from `Uniform(8, 36)` but `PARAM_BOUNDS['V_in'] = (10, 24)`. The agent saw target waveforms it could never reproduce because the surrogate would clip V_in to [10, 24], creating an impossible optimization target.

**Fix:** Constrained sampling to `PARAM_BOUNDS` range `(10, 24)`.

---

### BUG 2: Synthetic Rise Time in Target Waveforms
**File:** `train_intensive_spice.py` — `create_target_waveform()`  
**Severity:** Critical

**Problem:** Target waveforms included a synthetic exponential rise from 0V to steady state (`target *= (1 - np.exp(-t / tau))`). But the surrogate model outputs **steady-state** waveforms — it has no concept of startup transients. The agent was trying to match a waveform shape that the surrogate could never produce.

**Before:** Target starts at 0V, rises to 12V over time  
**After:** Target is flat at 12V (steady-state), with realistic ripple superimposed

**Fix:** Removed the `(1 - exp(-t/tau))` envelope. Targets are now steady-state waveforms with topology-appropriate ripple.

---

### BUG 3: Raw MSE Drowning Engineering Metrics
**File:** `rl/topology_rewards.py` — reward computation  
**Severity:** Critical

**Problem:** Raw MSE values ranged from 50–200, while engineering metrics (THD, ripple, efficiency) contributed 0.1–3.0 to the reward. The MSE term completely dominated, making the agent ignore all other optimization objectives.

**Before:** `mse_reward = -mse * weight` (e.g., -150 × 3.0 = -450)  
**After:** `mse_reward = -log1p(mse) * weight` (e.g., -log1p(150) × 3.0 = -15)

**Fix:** Applied `log1p(MSE)` scaling to compress the MSE range into the same magnitude as other reward terms.

---

### BUG 4: Step Size Mismatch
**File:** `train_intensive_spice.py` — `TopologySpecificEnv.step()`  
**Severity:** Medium

**Problem:** `TopologySpecificEnv` used `max_change_pct = 0.1` (10%) while the base `CircuitDesignEnv` used 0.2 (20%). The reduced exploration range made it much harder for the agent to reach the target parameters within the episode horizon.

**Fix:** Changed to `max_change_pct = 0.2` to match base environment.

---

### BUG 5: Unreachable Success Threshold
**File:** `rl/topology_rewards.py` and `train_intensive_spice.py`  
**Severity:** High

**Problem:** Success threshold was set to `0.001` MSE, but the best achievable MSE (even with perfect parameters) was ~3.3. The agent never experienced a "success" signal, so it had no positive reinforcement to learn from.

**Fix:** Raised threshold to `~5.0` (topology-specific, based on `success_mse / efficiency_target`).

---

### BUG 6: Missing Current Prediction in State
**File:** `rl/environment.py` — `_get_state()`  
**Severity:** Critical

**Problem:** State vector was 41 dimensions: `[target_features(32), params(6), error(3)]`. It did **not** include the current predicted waveform. The agent could see the target and its parameters, but not what those parameters actually produce. It was flying blind.

**Before:** 41 dims = target(32) + params(6) + error(3)  
**After:** 73 dims = target(32) + prediction(32) + params(6) + error(3)

**Fix:** Added 32 current-prediction features to the state vector. This lets the agent see the effect of its actions and learn to reduce the gap between prediction and target.

**Side effect:** All existing checkpoints became incompatible (wrong input dimension).

---

### BUG 7: Missing Topology in topology_map
**File:** `rl/environment.py` — `topology_map`  
**Severity:** High

**Problem:** `topology_map` dict didn't include `qr_flyback`, so it defaulted to index 0 (buck). The surrogate's topology embedding for qr_flyback was never used — it always ran with buck's embedding.

**Fix:** Added `'qr_flyback': 6` to `topology_map`.

---

### BUG 8: GAE Bootstrap Always Zero
**File:** `rl/ppo_agent.py` — `compute_gae()` and `collect_rollouts()`  
**Severity:** High

**Problem:** The GAE (Generalized Advantage Estimation) bootstrap value was always 0 at rollout boundaries. When a rollout ends mid-episode (not at a terminal state), the value function V(s_next) should be used to bootstrap the advantage calculation. Instead, the truncated returns were being treated as if the episode ended with 0 future reward.

**Before:** `advantages = compute_gae(rewards, values, dones, gamma, lam)` — bootstrap implicitly 0  
**After:** `advantages = compute_gae(rewards, values, dones, gamma, lam, last_value=critic(s_next))`

**Fix:** Added `last_value` parameter to `compute_gae()`, populated from the critic's estimate of the next state at rollout boundary.

---

### EXTRA FIX: Uncapped Reward Terms
**File:** `rl/topology_rewards.py`  
**Severity:** Medium

**Problem:** Individual reward terms (ripple penalty, THD penalty, rise time penalty, overshoot penalty) could grow unboundedly, causing reward spikes of -1000+ that destabilized PPO training.

**Fix:** Capped all individual reward terms at `±5.0` using `max(term, -5.0)` / `min(term, 5.0)`. Total reward now stays bounded roughly in [-10, +10].

---

## Session 2: Pre-Training Audit (2 bugs)

### Context
Before launching full training across all 7 topologies, a comprehensive 7-section audit was run covering: surrogate outputs, target generation, target-surrogate alignment, reward function behavior, environment mechanics, PPO agent, and training config. The audit passed all checks but revealed 2 critical alignment bugs.

---

### BUG 9: buck_boost/cuk Inverted Sign Mismatch
**File:** `train_intensive_spice.py` — `create_target_waveform()` vs surrogate  
**Severity:** Critical

**Problem:** `create_target_waveform()` computed buck_boost and cuk targets with **negative** output (as per the real physics: V_out = -V_in × D/(1-D)). But the surrogate model's `compute_theoretical_vout()` returns **positive** values for these topologies (it was trained this way — the denormalization uses positive scaling internally). Result: target mean = -23.97, surrogate output mean = +25.27, yielding **MSE = 4,339** even with identical parameters.

**Impact:** Training on buck_boost or cuk was completely impossible — the agent would see enormous MSE that could never be reduced because the target and surrogate lived in opposite sign spaces.

---

### BUG 10: qr_flyback Target-Surrogate Scaling Mismatch  
**File:** `train_intensive_spice.py` — `create_target_waveform()` vs surrogate  
**Severity:** High

**Problem:** `create_target_waveform()` used a QR-specific formula (`V_in × eff_duty × n / (1 - eff_duty)`) with a 0.95 efficiency factor for qr_flyback. But the surrogate's `compute_theoretical_vout()` has no qr_flyback case — it defaults to the buck formula (`V_in × duty`). Result: target mean = 19.31, surrogate mean = 8.44, **MSE = 216** even with identical parameters.

**Impact:** qr_flyback training would converge to a suboptimal solution, never able to achieve low MSE because of the systematic scaling offset.

---

### FIX for BUG 9 + BUG 10: Surrogate-Generated Targets
**File:** `train_intensive_spice.py` — `TopologySpecificEnv.reset()`

**Root Cause:** Both bugs stem from the same design flaw — using a **manual transfer function** (`create_target_waveform()`) to generate targets, while the surrogate uses its own **learned** transfer function. Any mismatch between manual and learned functions creates a systematic, unfixable error.

**Solution:** Instead of computing targets from physics equations, generate targets by **running the surrogate itself** on random parameters. This guarantees perfect alignment because both target and prediction use the exact same model.

```python
# OLD (buggy): manual transfer function
self.target_waveform = create_target_waveform(self.topology, v_in, duty, ...)

# NEW (fixed): surrogate-generated target
target_params = self._random_params()
self.target_waveform = self._simulate(target_params)
```

**Verification:** After the fix, same-params MSE = 0.000000 for all 7 topologies (perfect alignment confirmed).

---

## Summary Table

| Bug | File | Severity | Problem | Fix |
|-----|------|----------|---------|-----|
| 1 | train_intensive_spice.py | Medium | V_in sampled outside PARAM_BOUNDS | Constrained to (10, 24) |
| 2 | train_intensive_spice.py | Critical | Synthetic rise time ≠ surrogate steady-state | Removed rise envelope |
| 3 | topology_rewards.py | Critical | Raw MSE (50-200) drowns metrics (0.1-3) | log1p(MSE) scaling |
| 4 | train_intensive_spice.py | Medium | Step size 10% vs base 20% | Aligned to 20% |
| 5 | topology_rewards.py | High | Success threshold 0.001, best achievable ~3.3 | Raised to ~5.0 |
| 6 | environment.py | Critical | State missing current prediction (41 dim) | Added pred features (73 dim) |
| 7 | environment.py | High | topology_map missing qr_flyback | Added qr_flyback: 6 |
| 8 | ppo_agent.py | High | GAE bootstrap always 0 | Bootstrap from V(s_next) |
| Extra | topology_rewards.py | Medium | Reward terms unbounded (spikes to -1000) | Capped at ±5.0 |
| 9 | train_intensive_spice.py | Critical | buck_boost/cuk sign inverted vs surrogate | Surrogate-generated targets |
| 10 | train_intensive_spice.py | High | qr_flyback formula ≠ surrogate's scaling | Surrogate-generated targets |

---

## Cleanup

After fixing all bugs, the following were deleted:
- **10 diagnostic scripts:** `diagnose_bugs.py`, `diagnose_bugs2.py`, `diagnose_bugs3.py`, `verify_fixes.py`, `validate_fixes.py`, `validate_fixes_long.py`, `pre_training_audit.py`, `test_rl_vs_random.py`, `_verify_fix.py`, `_verify_fix2.py`
- **19 old log files:** All `*.log` and `training_log.txt` files from previous training runs
- **26 old RL checkpoints:** All `rl_agent_*.pt` and `multi_topo_rl_agent_*.pt` files (incompatible state_dim=41 vs new 73)
- **7 unused training scripts:** `train_production.py`, `train_robust.py`, `train_rl_only.py`, `train_spice_loop.py`, `train_spice_topology_aware.py`, `resume_training.py`, `train_remaining.sh`
- **4 old result files:** `training_results.json`, `evaluation_results.json`, `validation_results.png`, `fair_evaluation_comparison.png`

**Kept:** Surrogate model checkpoints (`multi_topology_surrogate.pt`, etc.), the canonical training script (`train_intensive_spice.py`), evaluation scripts, demo assets, and web demos.
