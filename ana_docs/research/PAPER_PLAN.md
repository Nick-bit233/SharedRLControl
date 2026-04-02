# Paper Plan: Safe Residual Shared Autonomy for UAV Navigation

**Target venues**: NeurIPS 2026 (Datasets & Benchmarks / Main) · CoRL 2026  
**Date**: 2026-04-01  
**Codebase**: SharedRLControl (Isaac Sim + ROS)

---

## 1. Positioning & Core Claim

### Problem Statement

Consumer drone usage is exploding, but most novice pilots lack the skill to navigate safely in complex environments. Unlike automobiles—which evolved through graduated ADAS levels (ABS → ESC → lane-keeping → adaptive cruise)—drones offer only manual RC or full autonomy, with **no principled "copilot" in between**.

### One-Sentence Thesis

> **A constrained residual RL policy can learn *when* and *how much* to correct a novice drone pilot's commands — intervening near danger and staying transparent in safe regions — achieving a Pareto-superior trade-off between safety and pilot autonomy preservation.**

### Key Innovation (vs. Prior Art)

| Prior Work | What They Did | What We Add |
|-----------|---------------|-------------|
| Schaff & Walter (RSS 2020) | Residual shared autonomy, 6-DOF quadrotor reaching | **3D obstacle navigation**, **danger-aware reward**, **constrained RL**, **multi-pilot benchmark** |
| Backman et al. (2022/2025) | RL shared autonomy, drone landing | Fixed assistance level; landing-only; no obstacle-rich 3D navigation |
| ASMA (Sanyal & Roy, RA-L 2024) | CBF + MPC for VLN drones | No human pilot; fixed safety margins; reactive only |
| DiSCo (Wang et al., HRI 2026) | Diffusion shared autonomy | Robot arms + driving; latency concern for drones; no safety constraint |
| Yoneda et al. (RSS 2023) | Diffusion for shared autonomy | Goal inference, not safety correction; no drone application |

### Three Contributions

1. **Danger-Aware Residual Policy (DARP)** — A novel reward formulation that naturally teaches the policy to be transparent (residual → 0) in safe regions and intervene (residual > 0) near obstacles, without requiring explicit pilot skill estimation.

2. **Constrained Residual PPO (CR-PPO)** — True Lagrangian-constrained RL that enforces a hard collision-rate ceiling while simultaneously minimizing deviation from human input. Unlike prior constrained RL work, the constraint targets a *shared control* metric (collision rate under human-robot blending), not just autonomous performance.

3. **SharedNav Benchmark** — A standardized evaluation protocol for drone shared autonomy across 4 pilot behavior profiles × 4 environment difficulty levels, with metrics that decompose safety, autonomy preservation, and task completion.

---

## 2. Title Options

**Option A (NeurIPS — emphasizes method)**:
> *Learning When to Intervene: Constrained Residual Reinforcement Learning for Safe Human-Drone Shared Autonomy*

**Option B (CoRL — emphasizes system)**:
> *DARP: Danger-Aware Residual Policy for Safe Novice Drone Piloting*

**Option C (concise)**:
> *Safe Shared Autonomy for UAVs via Constrained Residual Policy Learning*

**Recommended**: Option A for NeurIPS, Option B for CoRL.

---

## 3. Abstract Sketch (~250 words)

> Novice drone pilots face a steep learning curve: small control errors in complex environments quickly escalate into crashes. Existing drone systems offer either manual RC control or full autonomy, with no graduated assistance in between — unlike the automobile industry's progressive ADAS levels. We propose **DARP (Danger-Aware Residual Policy)**, a deep RL-based copilot that learns additive corrections to the human pilot's raw control signals. Our key insight is a **danger-aware tracking reward** that naturally teaches the policy *when* to intervene: in safe regions the residual approaches zero (preserving pilot autonomy), while near obstacles the residual grows to prevent collisions. We formulate shared control as a **constrained RL problem** (CR-PPO) that explicitly enforces a collision-rate ceiling via Lagrangian dual optimization, avoiding the known failure mode of fixed regularization weights that either over-constrain or under-protect. We train in NVIDIA Isaac Sim with a population of **4 simulated pilot profiles** spanning novice to expert behavior, and evaluate on a systematic **SharedNav benchmark** with 4 difficulty levels. Experiments show that DARP reduces novice pilot collision rates by **>60%** while preserving **>80%** of pilot autonomy (measured by intervention rate), achieving a Pareto-superior trade-off compared to fixed-assistance, pure-CBF, and unconstrained residual baselines. We further demonstrate sim-to-real transfer via ROS deployment on a quadrotor platform. Code, models, and benchmark are open-sourced.

---

## 4. Paper Structure

### Section 1: Introduction (1.5 pages)

**Story arc**: Automobile ADAS analogy → drone industry gap → why "residual copilot" → contributions

- Figure 1: **ADAS level analogy** — show automobile levels 0-5 mapped to drone equivalents, with the gap at levels 3-4 highlighted
- Emphasize practical motivation: 1M+ new drone pilots/year, >70% of crashes are pilot error
- State the three contributions concisely

### Section 2: Related Work (1 page)

**5 categories**:
1. **Drone shared autonomy**: Backman et al. (2020, 2022, 2025) — RL for landing; fixed assistance
2. **Residual policy learning**: Schaff & Walter (RSS 2020) — shared autonomy via residual; Trumpp et al. (2026) — autonomous racing RPL; Silver et al. (2019) — residual RL overview
3. **Safety filters for drones**: ASMA (2024) — CBF+MPC; Zhang & Tron (2024) — haptic CBF; Harms et al. (IROS 2024) — neural CBF multirotor
4. **Constrained RL**: Lagrangian methods (Tessler et al. 2018; Stooke et al. 2020); CPO (Achiam et al. 2017)
5. **Diffusion/VLM for shared autonomy**: DiSCo (2026), Yoneda et al. (RSS 2023) — different paradigm, latency limitations for drones

**Differentiation table**: Same as Section 1 positioning table above.

### Section 3: Problem Formulation (1 page)

**3.1 Shared Control MDP**

State: `x = (p, v, ψ, ω, L, d_goal)` where L = LiDAR scan, d_goal = direction to waypoint

Human action: `u_h ∈ ℝ³` (velocity command from pilot)

Residual action: `δ = π_θ(x, u_h)` (learned correction)

Executed action: `u = u_h + δ`

Transition: `x' = f(x, u)` (Isaac Sim physics)

**3.2 Danger-Aware Tracking Objective**

```
danger(x) = exp(-d_min(x) / d_safe)  ∈ [0, 1]

r_track(x, δ) = -‖δ‖² · (1 - β · danger(x))      (intent preservation)
r_safety(x)   = -c_s · 𝟙[d_min(x) < d_safe]        (proximity penalty)
r_progress(x) = c_p · Δ_goal(x)                     (task progress)
r_total       = r_track + r_safety + r_progress
```

**Key insight**: When `danger → 0` (safe), `r_track ≈ -‖δ‖²` (strong penalty for any residual). When `danger → 1` (near obstacle), `r_track ≈ -0.2·‖δ‖²` (weak penalty — policy is *freed* to intervene). This elegantly encodes "intervene only when needed" without explicit mode switching.

**3.3 Constrained Formulation (CR-PPO)**

```
max_θ  E[Σ r_total]
s.t.   E[collision_rate] ≤ ε_target
```

Solved via Lagrangian dual:
```
L(θ, λ) = E[Σ r_total] - λ · (E[collision_rate] - ε_target)
```

λ updated by dual gradient ascent: `λ ← max(0, λ + η_λ · (collision_rate - ε_target))`

**Difference from standard constrained RL**: The constraint is on a *shared control* metric (collision rate when human + policy fly together), not just the policy's autonomous performance. This creates a coupled optimization where the policy must model and react to human behavior.

### Section 4: Method — DARP Architecture (1.5 pages)

**4.1 Network Architecture**

```
LiDAR (36×4) → CNN (4 layers, 128-dim) ─┐
Drone state (8D) ──────────────────────── ├→ MLP (256, 128) → δ_mean (3D)
Human action u_h (3D) ─────────────────── │                  → concentration (3D)
Direction to goal (3D) ────────────────── ┘
```

- Output distribution: **Beta(α, β)** for bounded residual ∈ [-δ_max, δ_max]
  - α = mean·conc + 1, β = (1-mean)·conc + 1
  - Ensures unimodal distribution (no multi-modal instability)
  - Bounded actions prevent extreme corrections

- **Identity initialization**: Last layer bias → 0, weights → ε (δ ≈ 0 at start → safe warm-start)

**4.2 CR-PPO Training Algorithm**

```
Algorithm 1: Constrained Residual PPO (CR-PPO)
─────────────────────────────────────────────
Input: Pilot population P, environment E, target collision rate ε
Initialize: π_θ, V_φ, λ=0
for iteration = 1 to N:
  Sample pilot p ~ P
  Collect trajectories τ = {(x, u_h, δ, r, x')} using π_θ + p
  Compute advantages A_t (GAE)
  Compute constraint: C = mean(collision_indicator(τ))
  
  // Primal update (PPO)
  for epoch = 1 to K:
    L_actor = PPO_clip(θ, τ, A)
    L_track = mean(‖δ_mean‖² · (1 - β·danger))
    L_total = L_actor + reg_coeff · L_track + L_critic + L_entropy
    θ ← θ - η_θ · ∇_θ L_total
  
  // Dual update (Lagrangian)
  λ ← max(0, λ + η_λ · (C - ε))
  reg_coeff ← λ · base_reg  // Adaptive regularization!
```

**Key difference from prior constrained RL**: `reg_coeff` is *dynamically* controlled by the Lagrange multiplier, not fixed. When collision rate exceeds target, λ increases → reg_coeff decreases → policy is freed to intervene more aggressively. When collision rate is below target, λ decreases → reg_coeff increases → policy tightens residuals.

**4.3 Curriculum Training**

| Stage | Obstacles | Map Size | Pilot Speed | Duration |
|-------|-----------|----------|-------------|----------|
| S1 | 30 | 12×24m | 0.5 m/s | 10k iter |
| S2 | 80 | 20×40m | 1.0 m/s | 15k iter |
| S3 | 150 | 20×40m | 1.5 m/s | 20k iter |
| S4 | 250 | 20×40m | 2.0 m/s | 30k iter |

Promotion criterion: eval_success > 85% for 3 consecutive evaluations.

**4.4 Simulated Pilot Population**

| Profile | Behavior | Speed | Noise | Reaction | Purpose |
|---------|----------|-------|-------|----------|---------|
| **Novice** | Perlin 3D, high freq | 0.5-1.0 m/s | σ=0.5 | slow (laziness=0.4) | Primary assistance target |
| **Intermediate** | Perlin 3D, medium freq | 1.0-1.5 m/s | σ=0.3 | medium (laziness=0.2) | Moderate assistance |
| **Expert** | Smooth arc/waypoint | 1.5-2.0 m/s | σ=0.1 | fast (laziness=0.05) | Minimal intervention test |
| **Distracted** | Expert + sudden pauses/jerks | Variable | Burst σ=0.8 | intermittent | Robustness test |

### Section 5: SharedNav Benchmark (0.5 page)

**4 Difficulty Levels**:
| Level | Obstacles | Density | Dynamic | Passages |
|-------|-----------|---------|---------|----------|
| Easy | 30 | Sparse | 0 | Wide (>3m) |
| Medium | 80 | Moderate | 0 | Mixed (1.5-3m) |
| Hard | 150 | Dense | 5 | Narrow (<1.5m) |
| Extreme | 250 | Very dense | 10 | Tunnel-like |

**4 × 4 = 16 evaluation conditions** (pilot × difficulty), each with 5 random seeds × 100 episodes = 500 episodes per cell → 8,000 total episodes.

### Section 6: Experiments (2.5 pages)

**6.1 Metrics** (decomposed into 3 axes)

| Axis | Metric | Formula | Desired |
|------|--------|---------|---------|
| **Safety** | Collision Rate (CR) | #crashes / #episodes | ↓ |
| **Safety** | Min Distance (d_min) | mean(min LiDAR per step) | ↑ |
| **Safety** | Near-Miss Rate (NMR) | fraction of steps with d < d_warn | ↓ |
| **Autonomy** | Intervention Rate (IR) | fraction of steps with ‖δ‖ > ε_int | ↓ |
| **Autonomy** | Residual Magnitude (RM) | mean(‖δ‖) | ↓ |
| **Autonomy** | Intent Fidelity (IF) | cos(u_h, u) averaged | ↑ |
| **Task** | Success Rate (SR) | #goal_reached / #episodes | ↑ |
| **Task** | Time-to-Goal (TTG) | mean steps to reach goal | ↓ |
| **Task** | Path Efficiency (PE) | optimal_dist / actual_dist | ↑ |
| **Composite** | **SARP Score** | SR × (1 - CR) × (1 - IR) | ↑ |

**SARP (Safety-Autonomy-Reachability Product)**: A single composite metric that captures all three axes. A perfect copilot achieves SARP = 1.0 (never crashes, never intervenes, always reaches goal).

**6.2 Baselines** (8 methods)

| ID | Method | Description |
|----|--------|-------------|
| B1 | **No Assist** | Raw human input, no correction |
| B2 | **Fixed Residual** (Schaff & Walter) | Unconstrained residual RL, α_reg fixed |
| B3 | **Fixed α-Blend** | u = 0.5·u_h + 0.5·π_auto, no adaptation |
| B4 | **CBF Safety Filter** | Hard CBF with fixed margins (ASMA-style) |
| B5 | **Adaptive CBF** | CBF with hand-tuned adaptive margins |
| B6 | **Full Autonomy** | RL policy alone, no human input |
| B7 | **Danger-Only** (ablation) | DARP without constrained RL (fixed reg_coeff) |
| B8 | **Constraint-Only** (ablation) | CR-PPO without danger-aware reward (standard ‖δ‖² penalty) |

**B7 + B8 are ablations** that isolate the two contributions.

**6.3 Key Experiment Design**

**Experiment 1: Main Comparison (Table 1)**
- All 8 methods × 4 pilot profiles × 4 difficulty levels
- Report: CR, IR, SR, SARP
- Expected result: DARP achieves lowest CR among methods with IR < 30%

**Experiment 2: Pareto Frontier (Figure 2)**
- X-axis: Intervention Rate (IR), Y-axis: Collision Rate (CR)
- Sweep ε_target ∈ {0.01, 0.05, 0.1, 0.2, 0.5} for DARP
- Show DARP's Pareto curve dominates all baselines

**Experiment 3: Danger-Aware Behavior Visualization (Figure 3)**
- Show residual magnitude ‖δ‖ over time, overlaid with distance-to-nearest-obstacle
- Expected: residual spikes correlate with proximity to obstacles
- Include ego-centric LiDAR heatmap showing "when the copilot activates"

**Experiment 4: Pilot Profile Generalization (Table 2)**
- Train on {Novice, Intermediate, Expert}, test on {Distracted, OOD-style}
- Show zero-shot generalization to unseen pilot behaviors

**Experiment 5: Curriculum Ablation (Figure 4)**
- Compare: no curriculum vs. 2-stage vs. 4-stage
- Show curriculum is necessary for high-density environments

**Experiment 6: Constraint Sensitivity (Figure 5)**
- Vary ε_target from 0.01 to 0.5
- Show λ dynamics and how the system adapts reg_coeff

**Experiment 7 (optional, CoRL bonus): Sim-to-Real Transfer**
- ROS deployment on real quadrotor with joystick pilot
- Qualitative comparison: pilot confidence, crash avoidance
- Even 3-5 real flights would significantly strengthen the paper

**6.4 Ablation Table**

| Ablation | What's Removed | Expected Effect |
|----------|---------------|-----------------|
| No danger-aware reward | Remove `(1-β·danger)` weighting | IR increases (intervenes everywhere) |
| No constraint (fixed reg) | Remove Lagrangian, use fixed reg_coeff=0.01 | Either too many crashes or too many interventions |
| No curriculum | Train directly on Hard | Training collapse (seen in S3 experiments) |
| No Beta distribution | Use TanhNormal | Unbounded residuals, potential instability |
| Smaller LiDAR | 18 beams instead of 36 | Higher collision rate due to blind spots |
| Single pilot training | Train on novice only | Poor generalization to other profiles |

### Section 7: Results & Analysis (1.5 pages)

Expected results based on current experiments + planned improvements:

| Method | CR↓ (Novice/Hard) | IR↓ | SR↑ | SARP↑ |
|--------|-------------------|-----|-----|-------|
| No Assist (B1) | 0.85 | 0.00 | 0.10 | 0.015 |
| Fixed Residual (B2) | 0.45 | 0.55 | 0.50 | 0.124 |
| Fixed Blend (B3) | 0.40 | 1.00 | 0.55 | 0.000 |
| CBF Filter (B4) | 0.15 | 0.65 | 0.45 | 0.056 |
| Adaptive CBF (B5) | 0.12 | 0.45 | 0.55 | 0.145 |
| Full Autonomy (B6) | 0.08 | 1.00 | 0.75 | 0.000 |
| **DARP (Ours)** | **0.10** | **0.22** | **0.70** | **0.491** |

> DARP achieves comparable collision rates to CBF-based methods while intervening **3× less often**, resulting in a dramatically higher SARP score.

### Section 8: Discussion & Limitations (0.5 page)

**Limitations to acknowledge honestly**:
1. Simulated pilots only — real human study needed for definitive validation
2. Skill estimation is implicit (in residual magnitude), not explicit — less interpretable than modular approaches (cite SkillCopilot direction from IDEA_REPORT)
3. Beta distribution bounds may limit extreme evasive maneuvers
4. LiDAR-only perception — no RGB camera, limited to geometric obstacles
5. Single drone — no multi-agent scenarios

**Future work**:
- Add explicit skill estimator head for interpretability (SkillCopilot extension)
- Real user study with 10+ participants
- Multi-modal perception (RGB + LiDAR)
- Transfer to different drone platforms

### Section 9: Conclusion (0.25 page)

---

## 5. Figure List

| # | Figure | Type | Content |
|---|--------|------|---------|
| 1 | ADAS Analogy | Diagram | Automobile ADAS levels → drone equivalents, gap highlighted |
| 2 | System Architecture | Diagram | DARP architecture: human input → residual policy → blended output → drone |
| 3 | Danger-Aware Behavior | Time-series plot | ‖δ‖ vs. d_min over an episode, showing correlation |
| 4 | Pareto Frontier | Scatter plot | CR vs. IR for all methods, DARP curve dominates |
| 5 | Curriculum Progression | Multi-panel line plot | Success rate across curriculum stages |
| 6 | Lambda Dynamics | Line plot | Lagrange multiplier λ and reg_coeff over training |
| 7 | Qualitative Trajectories | Bird's-eye trajectory plot | Compare trajectories: no assist (crash) vs. DARP (safe) vs. full auto (ignores human) |
| 8 | Benchmark Results Heatmap | Heatmap | 4 pilots × 4 difficulties, color = SARP score |

---

## 6. What Exists vs. What's Needed

### ✅ Already Implemented
- [x] Isaac Sim environment with Hummingbird quadrotor
- [x] LiDAR-based observation (36×4 beams, 4m range)
- [x] Residual policy architecture (PPO + residual action module)
- [x] Beta distribution action head (ConstrainedResidualBetaPPO)
- [x] Simulated pilot model (Perlin 3D, offline dataset, multiple modes)
- [x] Curriculum training infrastructure (stage-based promotion)
- [x] WandB logging for all metrics
- [x] ROS1/ROS2 deployment pipeline
- [x] Quick demo scripts with pretrained checkpoint
- [x] Obstacle generation (cylinders, variable density)
- [x] Detailed training diagnostics and analysis

### 🔧 Needs Modification (from existing code)
- [ ] **True Lagrangian dual update** — replace fixed `reg_coeff` with dynamic λ-controlled coefficient (currently `ppo_constrained_beta.py` line ~598 uses fixed weight)
- [ ] **Danger-aware tracking reward** — implement `r_track = -‖δ‖² · (1 - β·danger(x))` in environment reward (currently no tracking reward: `enable_task_reward: False`)
- [ ] **Progress reward** — add waypoint/distance-to-goal reward (currently only survival + safety)
- [ ] **Fix user model** — expand beyond forward-only bias; add backward, sideways, diagonal, hover commands
- [ ] **Expand pilot profiles** — add explicit Novice/Intermediate/Expert/Distracted configs with distinct parameters
- [ ] **Increase LiDAR resolution** — 36→72 beams to reduce blind spots (diagnostic report P2)
- [ ] **Lower crash penalty** — -50 → -10, raise survival reward (diagnostic report P1)
- [ ] **Fix lambda saturation** — remove max_clamp=10, or use adaptive learning rate for λ

### 🆕 Needs New Implementation
- [ ] **SharedNav Benchmark** — standardized 4-level difficulty config set
- [ ] **SARP metric** — composite metric computation in evaluation
- [ ] **Baseline B2** (Schaff & Walter) — implement unconstrained residual (easy: just remove constraint)
- [ ] **Baseline B3** (Fixed Blend) — `u = α·u_h + (1-α)·π_auto` with fixed α=0.5
- [ ] **Baseline B4** (CBF Filter) — implement simple CBF safety filter baseline
- [ ] **Baseline B5** (Adaptive CBF) — CBF with distance-dependent margins
- [ ] **Baseline B6** (Full Autonomy) — train policy without human input (SimplePPO already exists)
- [ ] **Pareto sweep** — script to sweep ε_target and plot CR vs. IR frontier
- [ ] **Visualization tools** — residual magnitude heatmaps, trajectory plots, LiDAR overlays
- [ ] **Statistical analysis** — 5 random seeds × all conditions, significance tests
- [ ] **Paper figures** — publication-quality matplotlib/tikz figures

### ⏱ Estimated Effort

| Category | Items | Est. Time |
|----------|-------|-----------|
| Core algorithm fixes | Lagrangian, danger-aware reward, user model | 1-2 weeks |
| Baseline implementations | B2-B6 | 1 week |
| Training (all methods × configs) | 8 methods × 16 conditions × 5 seeds | 2-3 weeks (parallel on 8×4090) |
| Evaluation & analysis | Metric computation, figures, tables | 1 week |
| Paper writing | LaTeX, figures, revisions | 2-3 weeks |
| **Total** | | **7-10 weeks** |

---

## 7. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Training instability persists | Medium | High | Curriculum + danger-aware reward addresses root causes; fallback: reduce to 2 difficulty levels |
| DARP doesn't beat CBF baseline | Low-Medium | High | CBF is reactive-only; DARP should handle predictive correction better; if not, emphasize autonomy preservation angle |
| Simulated pilots unrealistic | High | Medium | Acknowledge as limitation; add 3-5 real pilot validation flights; calibrate against Backman et al. telemetry data |
| Lambda still saturates | Medium | Medium | Use PID-style dual update (Stooke et al. 2020) instead of gradient ascent |
| LiDAR blind spots cause random crashes | Medium | Low | Increase to 72 beams; add temporal history buffer |
| Reviewer says "just Schaff & Walter + curriculum" | Medium | High | Emphasize: (1) danger-aware reward is new formulation, (2) constrained RL for shared control is new, (3) comprehensive benchmark contribution |

---

## 8. Venue-Specific Strategy

### NeurIPS 2026 Main Track
- **Strength**: Novel constrained RL formulation for shared control
- **Risk**: May be seen as "application paper" — need stronger theoretical contribution
- **Suggestion**: Add a Theorem section proving: "Under Assumptions 1-3, CR-PPO with danger-aware reward converges to a policy that satisfies the collision rate constraint with probability ≥ 1-δ"
- **Deadline**: ~May 2026 (TBD)

### NeurIPS 2026 Datasets & Benchmarks
- **Strength**: SharedNav benchmark is a first-of-kind evaluation protocol for drone shared autonomy
- **Risk**: Need to release benchmark as a pip-installable package
- **Suggestion**: Strongest path if benchmark is polished; include leaderboard

### CoRL 2026
- **Strength**: Perfect venue for robot learning + human-in-the-loop
- **Risk**: Need real robot experiments (at least qualitative)
- **Suggestion**: Add 3-5 real drone flights via ROS pipeline; CoRL values practical systems
- **Deadline**: ~June 2026

### ICRA / IROS 2027
- **Fallback venue**: More systems-oriented, accepts less novelty
- **Strength**: ROS deployment + real drone would be very strong
- **Deadline**: ~September 2026

**Recommended primary target**: **CoRL 2026** (best fit for method + system + robot learning)  
**Stretch target**: **NeurIPS 2026 Main** (requires theoretical contribution)

---

## 9. 中文概要 / Chinese Summary

### 论文定位

**标题**: 学习何时介入：基于约束残差强化学习的无人机安全人机共享自主

**核心观点**: 基于深度RL的残差策略可以学习*何时*以及*多大程度地*修正新手飞行员的控制指令——在安全区域保持透明，在危险区域主动干预——在安全性和飞行员自主权之间实现帕累托最优权衡。

### 三大贡献

1. **危险感知残差策略（DARP）** — 新颖的奖励公式，自然地教会策略在安全区域保持零残差、在障碍物附近主动干预
2. **约束残差PPO（CR-PPO）** — 真正的拉格朗日约束RL，将碰撞率作为硬约束优化
3. **SharedNav基准** — 4种飞行员×4种难度的标准化评估协议

### 创新点 vs 先前工作

| 对比项 | Schaff & Walter (RSS 2020) | 我们的DARP |
|--------|--------------------------|-----------|
| 任务 | 简单6-DOF到达 | 3D障碍物导航 |
| 安全约束 | 无 | 拉格朗日约束碰撞率上限 |
| 奖励设计 | 标准‖δ‖²正则 | 危险感知追踪奖励 |
| 飞行员模型 | 单一 | 4种行为画像 |
| 评估 | 单任务 | 16条件基准 (4×4) |
| 部署 | 仿真仅 | 仿真 + ROS实机 |

### 验证指标

| 指标 | 含义 | 目标 |
|------|------|------|
| 碰撞率 (CR) | 以碰撞结束的飞行比例 | ↓ <10% |
| 干预率 (IR) | 残差 > 阈值的时步比例 | ↓ <25% |
| 成功率 (SR) | 到达目标的飞行比例 | ↑ >70% |
| SARP分数 | SR × (1-CR) × (1-IR) | ↑ >0.45 |
| 意图保真度 (IF) | cos(u_h, u) 平均值 | ↑ >0.85 |

### Baseline对比 (8种方法)

1. **无辅助** — 原始人类输入（下界）
2. **固定残差** (Schaff & Walter 2020) — 无约束残差RL
3. **固定混合** — α=0.5 线性混合
4. **CBF安全滤波器** — 固定裕度CBF
5. **自适应CBF** — 手动调参的可变裕度CBF
6. **完全自主** — 纯RL策略（上界-安全/下界-自主）
7. **仅危险感知** (消融) — 无约束RL
8. **仅约束** (消融) — 无危险感知奖励

### 实验设计

- **训练**: Isaac Sim, 8×RTX 4090, 4阶段课程学习
- **评估**: 4飞行员 × 4难度 × 5种子 × 100回合 = 8000回合
- **关键图表**: 帕累托前沿（CR vs IR）、残差-距离相关性、课程进展、λ动态

### 当前代码状态

✅ 已有: Isaac Sim环境、残差PPO、Beta分布、飞行员模型、课程训练、ROS部署
🔧 需修改: 拉格朗日对偶更新、危险感知奖励、飞行员模型扩展
🆕 需新建: SharedNav基准、Baseline实现、帕累托扫描、论文图表

**预估周期**: 7-10周（算法修复→训练→评估→写作）
**推荐目标**: CoRL 2026（首选）/ NeurIPS 2026（冲刺）
