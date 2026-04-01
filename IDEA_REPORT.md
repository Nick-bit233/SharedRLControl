# Robotics Idea Discovery Report

**Direction**: Autopilot model/system for drones (UAVs) that helps human pilots without much skill or experience fly easily and safely — analogous to automobile autopilot/ADAS  
**Date**: 2026-04-01  
**Pipeline**: research-lit → idea-creator (robotics framing) → novelty-check → research-review  
**Target Venues**: CoRL, RSS, ICRA, IROS, RA-L

---

## Robotics Problem Frame

| Field | Value |
|-------|-------|
| **Embodiment** | Quadrotor UAV (multirotor) |
| **Task family** | Assisted flight / shared autonomy / novice pilot support / safe navigation |
| **Environment type** | Aerial — indoor & outdoor, varying clutter density |
| **Observation modalities** | RGB-D (ego-centric), IMU, barometer, GPS (outdoor), optical flow |
| **Action interface** | Blended: human stick inputs (velocity/attitude) + autonomous corrections |
| **Learning regime** | RL + safety filters (CBF), online pilot skill estimation, shared control |
| **Available assets** | AirSim / Flightmare / Isaac Sim; 8×RTX 4090; no real drone assumed |
| **Compute budget** | Moderate (simulation training on multi-GPU) |
| **Safety constraints** | Critical — collision avoidance, geofencing, attitude limits, battery |
| **Desired contribution** | Method (adaptive shared autonomy) + benchmark (multi-skill-level evaluation) |

---

## Landscape Matrix

### Papers Surveyed (grouped by approach)

| Paper | Year | Embodiment | Task | Approach | Eval | Benchmark | Main Limitation |
|-------|------|------------|------|----------|------|-----------|----------------|
| Backman et al., "Learning to Assist Drone Landings" | 2020 | Quadrotor | Landing | Shared autonomy (heuristic) | Sim+real | Custom | Landing only; no skill modeling |
| Backman et al., "RL for Shared Autonomy Drone Landings" | 2022 | Quadrotor | Landing | RL shared autonomy | Sim+real | Custom | Landing only; fixed assistance level |
| ASMA (Sanyal & Roy) — RA-L | 2024 | Quadrotor | VLN navigation | CBF + MPC + VLN | Sim (Gazebo) | Gazebo/Bebop2 | Reactive CBF; fixed safety margins; no human pilot |
| Zhang & Tron, "Safe and Stable Teleop of Quadrotors" | 2024 | Quadrotor | Teleop | CBF + haptic | Sim | Custom | Requires haptic hardware; fixed margins |
| Zhang et al., haptic CBF feedback | 2021 | Quadrotor | Teleop | CBF + haptic | Sim+real | Custom | Haptic HW dependency |
| VLN-Pilot (Dominguez-Dager et al.) | 2026 | Quadrotor | Indoor nav | VLM autonomous | Sim | Custom | Fully autonomous — no human in loop |
| VLA-AN (Wu et al.) | 2025 | Quadrotor | Aerial nav | VLA onboard | Sim+real | Custom | Fully autonomous; no shared control |
| Zhang et al., "Safety-Shielded RL Flight" | 2026 | Quadrotor | Agile flight | RL + safety shield | Sim+real | Custom | Autonomous racing; no human |
| TACOS (Nazzari et al.) | 2025 | Multi-UAV | Fleet coordination | Hierarchical modes | Sim | Custom | Fleet management, not individual assist |
| Casper (Liu et al.) | 2025 | Robot arm | Manipulation | VLM intent inference | Sim+real | Various | Not drones; manipulator-focused |
| LAMS (Tao et al.) | 2025 | Robot arm | Mode switching | LLM-driven | Sim+real | Custom | Not drones; DoF mismatch problem |
| PATO (Dass et al.) | 2022 | Robot arm | Data collection | Policy-assisted teleop | Real | Custom | Not drones; data collection focus |
| Sha et al., "Real-to-Sim-to-Real Shared Autonomy" | 2026 | Robot arm | Teleop | Copilot BC | Sim+real | Custom | Arm-only; not applicable to aerial |
| Abraham et al., "Adaptive Autonomy" | 2021 | General | Vision reliability | Rule-based autonomy levels | Sim | Custom | Not learning-based; rule triggers |

### Key Gaps

1. **No graduated "drone ADAS" framework** — automobiles evolved through SAE Levels 0–5 with layered assistance (ABS → ESC → lane-keeping → adaptive cruise → autopilot). Drones have either manual RC or full autonomy, with no principled middle ground that adapts to pilot needs.

2. **No online pilot skill modeling** — no existing work estimates a drone pilot's skill level in real time and adapts the shared autonomy blending ratio. Automotive research models driver drowsiness/distraction, but drone systems assume a fixed operator capability.

3. **CBF safety filters are reactive, not adaptive** — ASMA and related work use fixed safety margins. A novice needs wider margins; an expert finds fixed margins overly restrictive. No system adjusts CBF conservatism based on who is flying.

4. **No skill progression / learning support** — existing systems either assist or don't. No system is designed to progressively hand off control as the pilot's skills improve over time (the "training wheels" concept).

5. **No benchmark for multi-skill-level drone shared control** — evaluation in existing work uses a single user profile. There is no standard evaluation protocol that tests shared autonomy across a range of pilot skill levels.

6. **VLM/VLA approaches replace humans, don't assist them** — the latest VLN-Pilot and VLA-AN work makes drones fully autonomous. This is SAE Level 5 without Levels 1–4.

---

## Ranked Ideas

### Idea 1: SkillCopilot — RECOMMENDED ⭐

**Full title**: *SkillCopilot: Pilot Skill-Aware Adaptive Shared Autonomy for UAVs via Online Skill Estimation and Dynamic Authority Blending*

**One-sentence summary**: An RL-trained copilot that continuously estimates a human pilot's skill level from stick-input patterns and dynamically adjusts both the human-autonomy authority blending ratio and the safety filter margins — giving novices strong guardrails and experts minimal interference.

- **Embodiment**: Quadrotor UAV
- **Benchmark / simulator**: AirSim or Flightmare with custom multi-difficulty obstacle courses
- **Bottleneck addressed**: Fixed-level shared autonomy either over-assists experts or under-protects novices; no system adapts to who is actually flying
- **Pilot type**: Simulation (sim-first)
- **Positive signal**: Collision rate drops >50% for novice pilots while expert override rate stays <10%
- **Hardware risk**: LOW — pure simulation

#### Core Technical Contributions

**1. Online Pilot Skill Estimator (PSE)**
A lightweight recurrent neural network (GRU or 1D-CNN) that processes a sliding window (last 2–5 seconds) of raw stick inputs (4-axis: throttle, yaw, pitch, roll) and outputs a continuous skill score s(t) ∈ [0, 1].

Training data: Simulated pilots spanning a spectrum from novice (large, jerky, delayed inputs with drift) to expert (smooth, anticipatory, precise inputs). Skill ground truth labels derived from task performance metrics (tracking error, smoothness, reaction time).

Key design: The estimator must work online with <50ms latency and be robust to transient skill drops (e.g., distraction, unfamiliar maneuver).

**2. Skill-Conditioned Authority Blending**
The executed command is:

```
u(t) = α(s(t)) · u_human(t) + (1 − α(s(t))) · u_copilot(t)
```

Where:
- `u_human` = raw pilot stick input (mapped to velocity commands)
- `u_copilot` = autonomous copilot policy output (RL-trained)
- `α(s)` = authority function: α→1 for s→1 (expert gets full control), α→0.3 for s→0 (novice retains some agency but copilot dominates)

The copilot policy is **not** a replacement autopilot — it is a corrective policy trained to output the minimum modification needed to keep the flight safe and on-task.

**3. Adaptive CBF Safety Margins**
The Control Barrier Function constraint `h(x) ≥ γ(s)` uses a skill-dependent margin:
- `γ(s=0) = γ_max` (large margin, conservative — novice)
- `γ(s=1) = γ_min` (tight margin, permissive — expert)
- Smooth interpolation via sigmoid or learned mapping

This means a novice pilot's commands are clipped further from obstacles, while an expert can fly closer to obstacles without triggering the safety filter.

**4. RL Training Objective**
The copilot policy and authority function are jointly trained via RL (PPO or SAC) in simulation:

```
R = w_task · R_task + w_safe · R_safe + w_authority · R_authority + w_smooth · R_smooth
```

- `R_task`: Task completion reward (reach waypoints, complete course)
- `R_safe`: Zero-collision constraint (large negative for any collision)
- `R_authority`: Penalty for overriding the human (minimize interference)
- `R_smooth`: Smoothness of blended commands (no jarring transitions)

Training uses a population of simulated pilots with diverse skill levels so the policy generalizes.

**5. Simulated Pilot Population**
Critically, this work requires a realistic *simulated pilot model*. We propose:
- **Expert model**: PID controller with tight gains, small noise, fast reaction
- **Intermediate model**: PID with moderate gains, medium noise, occasional overshoot
- **Novice model**: PID with slow gains, large noise, delayed reaction, systematic drift
- **Dynamic skill model**: Pilot that improves within a session (starts novice, becomes intermediate)
- Calibration: parameters derived from published drone telemetry data (e.g., Backman et al. 2022)

#### Why This Is Novel

| Aspect | Existing Work | SkillCopilot (Ours) |
|--------|--------------|---------------------|
| Shared autonomy level | Fixed α | Adaptive α(s(t)) based on real-time skill |
| Safety margins | Fixed CBF margins | Skill-conditioned γ(s) |
| Pilot modeling | Not modeled | Online skill estimation from stick inputs |
| Assistance adaptation | Manual mode switching | Automatic, continuous, learned |
| Evaluation | Single user type | Multi-skill-level benchmark |
| Domain | Mostly robot arms | Drone-specific (3D flight, attitude dynamics) |

**Closest work comparison:**
- Backman et al. (2022) — fixed shared autonomy for landing only, no skill estimation
- ASMA (2024) — fixed CBF margins, no human pilot in loop at all
- Abraham et al. (2021) — adapts autonomy but based on vision model reliability, not pilot skill; rule-based, not learned
- Automotive ADAS — adapts to driver state (drowsiness), but uses physiological sensors (eye tracking, steering wheel); our approach uses only stick inputs

**Novelty assessment: NOVEL** — The combination of (a) online pilot skill estimation from control inputs, (b) skill-conditioned authority blending, and (c) adaptive CBF margins for drones has not been demonstrated. Individual components (shared autonomy, CBFs, pilot modeling) exist, but their integration into a skill-adaptive drone copilot is new.

#### Review Assessment

##### Internal Assessment (Self-Review)
**Overall: 6.5/10** (weak accept — solid contribution if execution is clean)

##### External Assessment (GPT-5.4, CoRL/RSS/ICRA Reviewer)
**Overall: 6/10** | **Novelty: 3/5** | **Feasibility: 4/5** | **Significance: 4/5**

> *"This paper proposes a shared-autonomy UAV copilot that estimates pilot skill online from joystick behavior and adapts both autonomy authority and safety conservatism accordingly. The idea is sensible and practically relevant. However, the contribution is mostly a synthesis of known components — shared autonomy, operator modeling, and CBF safety filtering — rather than a new control paradigm. Acceptance would depend heavily on clean problem formulation and unusually strong evaluation."*

> **Novelty verdict: PARTIALLY NOVEL** — The combination of (a) online pilot skill estimation from control inputs, (b) skill-conditioned authority blending, and (c) adaptive CBF margins for drones is new. But it is **composition novelty** — each piece exists; the integration is the contribution.

**Strengths (combined):**
1. Clear, practical motivation — the automobile ADAS analogy is compelling and the gap is real
2. Technically well-defined — each component (PSE, blending, adaptive CBF) is concrete and trainable
3. Pure sim-first — entire system trainable and evaluable in simulation with clear metrics
4. The simulated pilot population is a contribution in itself — enables reproducible evaluation of shared autonomy
5. Applicable to real consumer drones (DJI SDK, Betaflight) with minimal modification
6. Strong practical relevance: novice/expert-adaptive assistance is easy to motivate
7. Potentially stronger than landing-only prior work if shown across general navigation tasks

**Weaknesses (combined):**
1. **Incremental novelty** — many pieces already exist separately; composition must be justified
2. **Skill estimation confound** — stick-input patterns may reflect task difficulty, interface latency, or piloting style — not skill (reviewer's key concern)
3. **Simulated pilots are a realism bottleneck** — reviewer confidence drops if "skill" is just scripted noise
4. **Confound risk** — adaptive assistance may improve outcomes simply by becoming more conservative, not because it truly inferred skill
5. **CBF-margin adaptation may look heuristic** unless grounded theoretically or empirically
6. No user study (sim-only) — stronger paper with even a small real pilot study
7. The authority blending is linear — may not handle cases where pilot and copilot disagree on direction

**Positioning advice (from external reviewer):**
> *"Do NOT oversell as 'first adaptive drone shared autonomy.' Say instead: 'first UAV shared-autonomy framework that jointly adapts authority allocation and safety conservatism from online pilot-skill inference.'"*

**Required Baselines:**
1. No assistance (raw pilot input)
2. Fixed shared autonomy (α = 0.5, constant)
3. Adaptive blending only (no adaptive CBF margin)
4. Adaptive CBF margin only (no adaptive blending)
5. Fixed CBF safety filter (ASMA-style, constant margins)
6. Rule-based skill estimator / confidence heuristic
7. Oracle skill-conditioned assistance (true skill label — upper bound)
8. Full autonomy (no human in loop)
9. Backman-style shared autonomy baseline (if reproducible)

**Required Ablations:**
1. Remove the skill estimator entirely
2. Replace learned PSE with simple handcrafted metrics (input smoothness, reaction time)
3. Freeze authority blending (adaptive CBF only)
4. Freeze CBF margin (adaptive blending only)
5. Remove RL copilot, use simpler PID-based autopilot
6. Skill estimation window size: 1s vs. 2s vs. 5s
7. Mid-flight skill change scenario
8. Out-of-distribution pilot styles (not seen in training)
9. Task difficulty control (ensure estimator reads pilot, not course hardness)

**Minimum Evidence Package for Acceptance:**
- [ ] 3+ difficulty levels of obstacle courses
- [ ] 3+ distinct simulated pilot profiles (novice/intermediate/expert)
- [ ] Cross-profile generalization (test on pilot styles not in training)
- [ ] A convincing, non-circular definition of "skill"
- [ ] Calibration of skill estimator against meaningful pilot strata
- [ ] Collision rate, task success, override frequency reported per pilot type
- [ ] Statistical significance (5+ random seeds)
- [ ] Ablation on each component (PSE, adaptive α, adaptive γ)
- [ ] Comparison with all baselines above
- [ ] Oracle-skill upper bound showing how much estimation quality matters
- [ ] Domain-shift test: new layouts, unseen pilots, changed controller sensitivity
- [ ] Failure case analysis: when does the system make things worse?
- [ ] Pilot autonomy preservation metrics: intervention count, magnitude, time under human control

**Suggestions to Strengthen (combined):**
1. Add a small real-user pilot study (even 5 participants) to validate simulated pilot assumptions
2. Collect small real pilot telemetry from novice/expert pilots to calibrate the skill proxy
3. Show that the system helps a "learning pilot" — one that improves over time — better than fixed assistance
4. Include a "mismatched skill" scenario (expert in unfamiliar environment) — show fast adaptation
5. Add task-difficulty control so the estimator is not just reading course hardness
6. Report pilot autonomy preservation explicitly: intervention count, magnitude, time under human control
7. Consider visual feedback channel (HUD showing skill estimate and assistance level) to increase user trust
8. One domain-shift test: new obstacle layouts, unseen pilot styles, changed controller sensitivity

---

### Idea 2: DroneTutor — Curriculum-Based Skill Transfer

**One-sentence summary**: A copilot inspired by driving instruction that decomposes piloting into atomic skills, assesses mastery per-skill, and progressively reduces assistance as the human demonstrates competence.

- **Embodiment**: Quadrotor
- **Benchmark / simulator**: AirSim with structured skill courses
- **Bottleneck addressed**: Existing systems assist but never teach; pilots remain dependent
- **Pilot type**: Simulation (requires real human study for strong claims)
- **Risk**: MEDIUM — mastery definitions are hard to validate without real users

**Novelty assessment: PARTIALLY NOVEL** (confirmed by external reviewer) — The concept of curriculum-based human skill transfer exists in HRI (human-robot interaction) literature for robot-assisted rehabilitation and industrial training. The mechanism "help more early, fade support after mastery" is not new. Applying it to drone piloting with per-skill autonomy fading and the explicit goal of making the human independent is novel in framing.

**External reviewer note**: *"This framing is more distinctive than Idea 1, but the paper becomes much more of a human learning / training systems paper. Without human evidence, this may be dismissed as 'assist-as-needed tutoring ported to drones.'"*

**Why not recommended as top idea**: Strong claims about human learning require real user studies. A purely simulated evaluation would be unconvincing to reviewers. Better as a follow-up study building on SkillCopilot. **However, if a real human training study (10+ participants) is feasible, this idea could be more compelling than Idea 1.**

---

### Idea 3: PredictiveGuardrails — Learned World-Model Safety Layer

**One-sentence summary**: A lightweight drone world model predicts state trajectories N seconds ahead given current pilot inputs and intervenes with minimal corrections when predicted collisions are detected.

- **Embodiment**: Quadrotor with RGB-D
- **Benchmark / simulator**: AirSim / Flightmare with cluttered environments
- **Bottleneck addressed**: Reactive CBFs activate too late for novice errors; predictive intervention is smoother
- **Pilot type**: Simulation
- **Risk**: MEDIUM-HIGH — world model accuracy in novel environments is uncertain

**Novelty assessment: NOT NOVEL** (external reviewer's harsh verdict) — MPC-based approaches already do look-ahead planning. Predictive intervention before constraint violation is an obvious next step closely related to MPC / look-ahead safety filtering / predictive shared control. The visual explanation overlay is useful but not enough to carry novelty.

**External reviewer note**: *"This is predictive shielding/MPC-style shared control with a learned predictor; the main contribution is systems integration. I would avoid Idea 3 for a top robotics venue unless you have a truly exceptional technical breakthrough."*

**Why not recommended**: Technically complex with high risk of world model prediction failures. The reviewer's inevitable question "why not just use MPC?" would be hard to answer convincingly. Killed for top venues.

---

## Eliminated Ideas (considered but rejected)

- **"Apply VLM to drone copilot"** — killed because this is foundation-model theater. Slapping a VLM on drone control without a specific bottleneck analysis is not novel.
- **"Haptic feedback for novice drone pilots"** — killed because it requires specialized haptic hardware not available on consumer drones. Also already explored by Zhang et al. (2019-2024).
- **"Multi-drone shared autonomy for novice operators"** — killed because multi-robot adds too much complexity for a first paper. Better to nail single-drone first.
- **"Natural language drone control for novices"** — killed because VLN-Pilot (2026) already does this in fully autonomous mode. Adding shared control to VLN is incremental.

---

## Evidence Package for the Top Idea (SkillCopilot)

### Required Baselines
1. **No assistance** — raw pilot input, no copilot
2. **Fixed shared autonomy** — constant blending ratio (α = 0.5)
3. **Fixed CBF safety filter** — constant safety margins (ASMA-style)
4. **Oracle skill** — adaptive system but with ground-truth skill labels (upper bound)
5. **Full autonomy** — no human, copilot-only (shows human still adds value)

### Required Metrics
| Metric | What It Measures |
|--------|-----------------|
| Collision rate | Safety (primary) |
| Task success rate | Effectiveness |
| Override frequency | How often copilot overrides human |
| Override magnitude | How much the human input is modified |
| Skill estimation accuracy | PSE quality (vs. ground truth) |
| Time-to-adapt | Latency when pilot skill changes |
| Course completion time | Efficiency |
| Trajectory smoothness | Flight quality |
| Authority ratio α | Average human vs. copilot control |

### Required Failure Cases
- When does the skill estimator misclassify? (expert pilot in novel environment looks like novice)
- When does adaptive assistance make things worse than fixed assistance?
- Edge case: pilot and copilot give opposing commands — what happens?
- What if the pilot deliberately fights the copilot?

### Whether Real Robot Evidence Is Mandatory
**No** — a strong simulation study with well-calibrated simulated pilots is sufficient for a first paper. Real human experiments would strengthen the paper significantly but are not required for acceptance at ICRA/IROS. A real user study would be required for CoRL/RSS.

---

## Sim-First Pilot Plan (Phase 3)

```
Embodiment:          Quadrotor (AirSim or Flightmare)
Benchmark/Simulator: AirSim Blocks environment + custom obstacle courses (3 difficulty levels)
Simulated pilots:    5 profiles (pure novice, novice-improving, intermediate, expert, expert-distracted)
Baselines:           No assist, fixed α=0.5, fixed CBF, oracle, full autonomy
Pilot type:          SIM (no real robot)
Compute estimate:    ~2-4 GPU-days for RL training (PPO, 8×4090)
Human/operator time: 0 (fully simulated pilots)
Success metrics:     Collision rate <5% for all pilot types; >50% collision reduction vs. no-assist for novices
Failure metrics:     Override rate >30% for experts = system too aggressive
Safety concerns:     N/A (simulation only)
Positive signal:     Adaptive system matches or beats best fixed-α for every pilot type
Negative result still publishable: If skill estimation is accurate but adaptive blending doesn't outperform fixed — this reveals that skill-awareness alone is insufficient, pointing to need for intent-awareness
```

---

## Next Steps

- [ ] Implement SkillCopilot sim-first pilot in AirSim/Flightmare
  - [ ] Build simulated pilot population (5 profiles)
  - [ ] Implement Online Pilot Skill Estimator (GRU on stick input window)
  - [ ] Implement authority blending + adaptive CBF
  - [ ] Train copilot policy via PPO/SAC
  - [ ] Run baselines and ablations
- [ ] Run `/novelty-check` on final idea wording
- [ ] Consider small real-user study (5-10 participants) with DJI Tello or sim interface
- [ ] Only after sim results: consider hardware validation on real quadrotor
- [ ] Target venue: ICRA 2027 or RA-L (rolling deadline)

---

## Appendix: Automotive ADAS Analogy

| Automobile ADAS Level | Drone Equivalent (Proposed) | Status |
|----------------------|----------------------------|--------|
| **ABS / Traction Control** | Attitude stabilization, altitude hold | ✅ Exists (flight controllers) |
| **Electronic Stability Control** | Geofencing, return-to-home | ✅ Exists (DJI, PX4) |
| **Lane-Keeping Assist** | CBF collision avoidance (fixed margins) | ✅ Exists (ASMA, etc.) |
| **Adaptive Cruise Control** | **SkillCopilot: adaptive authority blending** | ❌ **THIS PAPER** |
| **Traffic Jam Pilot** | **SkillCopilot: high autonomy for novices** | ❌ **THIS PAPER** |
| **Full Autopilot** | VLN-Pilot, VLA-AN (full autonomous) | ✅ Exists (2025-2026) |

The key insight: the drone industry skipped Levels 3-4. SkillCopilot fills this critical gap.

---

## Supplement: End-to-End Control Signal Remapping Direction

> *This supplement explores an alternative E2E architecture — a single learned model that takes the human pilot's raw control signal and directly outputs the actual safe control signal — and compares it against the modular SkillCopilot approach above.*

### Motivation

Instead of SkillCopilot's modular pipeline (Skill Estimator → Authority Blender → Adaptive CBF), can we build a **single neural network** that maps `u_human → u_safe` conditioned on observations? This would be simpler to deploy, avoids multi-stage training, and could be more data-efficient.

### Additional Literature (E2E-specific)

| Paper | Year | Venue | Approach | Relevance |
|-------|------|-------|----------|-----------|
| Schaff & Walter, "Residual Policy Learning for Shared Autonomy" | 2020 | RSS | Residual RL correction of human input; tested on **6-DOF quadrotor reaching** | ⚠️ **Direct prior art for E2E residual drone shared control** |
| Yoneda et al., "To the Noise and Back: Diffusion for Shared Autonomy" | 2023 | RSS | Diffusion model infers human goal, blends with autonomous policy | ⚠️ **Diffusion shared autonomy already published** |
| DiSCo (Wang et al.) | 2026 | HRI | Diffusion sequence copilots; seeds diffusion with user actions | Driving + arm, not drones; latency concern |
| Sha et al., "Real-to-Sim-to-Real Shared Autonomy" | 2026 | — | Sim-based copilot for teleoperation | Arm-only; concept transferable |
| Xiao et al., "Differentiable CBF for E2E Autonomous Driving" | 2022 | — | Joint learning of controller + CBF | E2E + safety, but autonomous driving |
| Kalaria et al., "Safety Assured E2E Vision-Based Racing" | 2023 | — | Neural CBF + E2E for autonomous racing | Close to SafeStickNet; no human-in-loop |
| So et al., "How to Train Your Neural CBF" | 2023 | — | Neural CBF training for complex systems | Safety filter methodology |
| Harms et al., "Neural CBF for Safe Navigation" | 2024 | IROS | Neural CBF on multirotor | Drone-specific neural safety |
| Trumpp et al., "Attenuated Residual Policy Optimization" | 2026 | — | RPL for real-world drone racing | Autonomous racing; no human pilot |
| Backman et al., "From Novice to Skilled..." | 2025 | THRI | RL shared autonomy across skill levels | Drone landing; skill progression study |

### E2E Candidate Ideas

#### E2E-1: ResidualPilot — Residual RL over Human Input ❌ KILLED

**Concept**: `u_safe = u_human + π_residual(obs, u_human)` — a learned residual correction on the human's raw stick commands.

**Verdict: NOT NOVEL.** Schaff & Walter (RSS 2020) already demonstrated residual policy learning for shared autonomy with a 6-DOF quadrotor reaching task. Our formulation is essentially a replication with different reward shaping.

- GPT-5.4 Review: **3/10** | Novelty 1/5 | Feasibility 4/5 | Significance 3/5
- **Eliminated in Phase 3 (novelty check)**

#### E2E-2: DiffuSafe — Diffusion-Based Stick Command Refinement ❌ KILLED

**Concept**: Conditional diffusion model denoises/refines human stick commands into safe commands. CBF-guided sampling rejects unsafe diffusion outputs. Consistency distillation for 50Hz+ inference.

**Verdict: WEAKLY NOVEL (domain port only).** Yoneda et al. (RSS 2023) already did diffusion for shared autonomy. DiSCo (HRI 2026) extended it. Porting to drones + adding CBF guidance is incremental. Additionally, real-time 50-100Hz inference for drone control is unproven and high-risk.

- GPT-5.4 Review: **4/10** | Novelty 2/5 | Feasibility 2/5 | Significance 3/5
- **Eliminated in Phase 4 (feasibility + insufficient novelty)**

#### E2E-3: SafeStickNet — Neural Safety-Certified Control Remapper ⚠️ BACKUP

**Concept**: Dual-head neural network with shared backbone — Head 1 outputs safe action `u_safe = f(u_human, obs)`, Head 2 outputs learned CBF value `h(x)`. Jointly trained with task reward + fidelity loss + CBF validity loss + forward-invariance certification loss.

**What makes it potentially novel**:
1. Joint learning of human-input control remapper + safety certificate for *drone shared control* (existing work does E2E+CBF for autonomous driving/racing, not human-in-loop drones)
2. The `h(x)` readout provides interpretable safety margin despite E2E architecture
3. Bridges E2E simplicity with formal safety story

**Critical weaknesses**:
1. "Formal certification" claim is likely overclaiming without real verification (SMT solvers, interval bound propagation, etc.)
2. Joint optimization of action head + certificate head is fragile (competing objectives)
3. If CBF guarantees fail in practice, degenerates into a black-box policy

- GPT-5.4 Review: **5/10** | Novelty 2/5 | Feasibility 2/5 | Significance 4/5
- **Status: BACKUP — salvageable only with rigorous safety verification and toned-down claims**

### Ranking: E2E vs Modular

| Rank | Idea | Type | Score | Novelty | Feasibility | Status |
|------|------|------|-------|---------|-------------|--------|
| **1** | **SkillCopilot** | Modular | **6/10** | 3/5 | 4/5 | ⭐ **RECOMMENDED** |
| **2** | SafeStickNet | E2E | 5/10 | 2/5 | 2/5 | ⚠️ BACKUP |
| **3** | DiffuSafe | E2E | 4/10 | 2/5 | 2/5 | ❌ ELIMINATED |
| **4** | ResidualPilot | E2E | 3/10 | 1/5 | 4/5 | ❌ ELIMINATED |

### Why E2E Loses to Modular (for this problem)

1. **Safety accountability**: SkillCopilot's modular CBF provides a separable safety guarantee. E2E approaches bundle safety into a monolithic network, making verification harder and reviewers skeptical.

2. **Interpretability**: SkillCopilot's skill score `s(t)` and blending ratio `α(s)` are directly inspectable. E2E models are black boxes — a critical concern for safety-critical drone systems.

3. **Prior art trap**: The E2E shared autonomy space is more crowded than it appears. Residual shared autonomy (RSS 2020) and diffusion shared autonomy (RSS 2023) already exist. The modular *skill-adaptive* angle is less explored.

4. **Ablation story**: SkillCopilot's modular design enables clean ablations (skill estimator accuracy, blending function choice, CBF margin adaptation). E2E architectures are harder to ablate meaningfully.

### Potential Hybrid: SkillCopilot + E2E Safety Head

A future direction could combine both approaches:
- Use SkillCopilot's online skill estimator + authority blending (interpretable, ablatable)
- Replace the hand-designed CBF with a **learned neural safety filter** (from SafeStickNet's Head 2)
- This gets the best of both worlds: modular interpretability + learned safety adaptation
- Not recommended as the initial paper — adds complexity without clear benefit over standard CBF

### Conclusion

**The E2E direction does not currently outrank the modular SkillCopilot approach** for this problem domain. The key reasons are:
- Critical prior art exists for residual and diffusion shared autonomy
- Safety certification of E2E models is an unsolved problem
- Modular design is more reviewer-friendly for safety-critical robotics

**Recommendation**: Proceed with SkillCopilot as the primary research direction. SafeStickNet could serve as a follow-up paper once SkillCopilot establishes the problem framing and evaluation benchmark.
