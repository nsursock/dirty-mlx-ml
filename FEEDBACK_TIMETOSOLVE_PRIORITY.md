# Time-to-Solve Feedback Priority List

Issues from AI feedback on making bench solve time **fast**, organized by priority.

| Priority | Issue | Main Finding | Fix | Impact | Reviewer(s) |
|----------|-------|--------------|-----|--------|-------------|
| **P0** | SAC host↔device sync barriers | `SAC.train()` calls `to_float()` ~7× per gradient step (`mx.eval` + `.item()`); plus 3–4 extra `mx.eval`s in update path → **~10+ syncs per step** | Accumulate losses as `mx.array` across `gradient_steps`; single `to_float`/`mx.eval` at end of `train()` | Explains SAC train FPS 3–4× worse than PPO at same env counts; dominant wall-time killer at low env counts | Claude |
| **P0** | SAC redundant forwards | Critic re-forwarded purely for `l1`/`l2` logging; `actor.sample()` called **3×** per step (entropy, actor loss, log_pi stat) | Return `(l1,l2,q1,q2)` from critic loss closure; reuse `log_prob` from actor loss for stats | Cuts ~2 actor forwards + 1 dual-Q forward per gradient step | Claude |
| **P0** | Compile the train step | Python-orchestrated many small MLX calls; no fused update at low env counts where dispatch dominates | Wrap gradient step (and ideally full update) in `mx.compile` | Highest-leverage systems win for wall TTS; DeepSeek/Claude both flag this as core | Claude, DeepSeek |
| **P1** | Eval/logging dominates wall time | Effective solve FPS (~3.4k) ≪ train FPS (~28k) → **>85% wall time outside pure train** | Cheap/online competence check; eval every N updates + one confirmation; stop on threshold; log every N steps | Gemini: most of 9.55s is not training; must fix or TTS stays slow even after train FPS improves | Gemini, ChatGPT, Claude |
| **P1** | Sample inefficiency (over-budget) | CartPole ~32k steps (need ~5–10k); Pendulum ~65k with 16 envs — HPs not tuned for fast competence | Tune LR/batch/GAE/warmup; **early-stop on threshold** (no fixed 32k/65k budget) | Cut steps 50–70% → proportional TTS cut if systems path is fixed | Gemini, ChatGPT, DeepSeek |
| **P1** | TTS metric & protocol | Single seed, fixed timestep budget, “solve” framing weak; cold JIT can dominate short runs | Primary metric = **time-to-competence** (median + success over 3–5 seeds); stop at threshold; report STS + TFPS; warm vs cold start | Makes “fast” honest and comparable; ChatGPT hierarchy: TTC > STS > TFPS > env FPS | ChatGPT, Claude, Gemini |
| **P2** | Hyperparams for low-latency convergence | Defaults not optimized for classic-control speed | CartPole PPO: higher LR (1e-3–2.5e-3), larger minibatches, λ=0.98; SAC: higher LR, short warmup (256–512) | Targets: CartPole **<1s**, Pendulum **<4–5s** | Gemini, DeepSeek |
| **P2** | Online return tracking (zero-sync) | Separate eval rollouts force host sync and stall unified-memory pipeline | Track episode returns in vectorized step graph; gate competence without full eval until near threshold | Removes main-thread blocking from solve path | Gemini |
| **P2** | Benchmark hierarchy / README framing | 8k-env sim FPS distracts from real training speed; TTS should be headline | Make TTS/TTC Tier-1; label huge-env tables as sim/microbench; return-vs-wall-clock plot | Aligns README with what practitioners care about | ChatGPT, DeepSeek |
| **P3** | Baseline TTS comparisons | No CleanRL/SB3/JAX TTS on same Mac | Add same-hardware TTS baselines once internal path is fast | Credibility; DeepSeek claims 3–20× slower than CPU baselines today | DeepSeek |
| **P3** | Log less aggressively | 7 scalars stringified/logged every gradient step even after sync collapse | Log every N gradient steps | Minor after P0 #1 | Claude |

## Target goals (from reviewers)

| Env | Algo | Current (approx) | Target wall | Target steps | Notes |
|-----|------|------------------|-------------|--------------|-------|
| CartPole-v1 | PPO | ~9.5–11s / ~32k | **< 0.8–1.5s** | ~8–20k | Gemini/DeepSeek |
| Pendulum-v1 | SAC | ~45–50s / ~65k | **< 3–5s** | ~12–15k | Gemini/DeepSeek |

## Summary

- **P0 (Critical path)**: 3 issues — SAC syncs, redundant forwards, `mx.compile` on update
- **P1 (High)**: 3 issues — eval overhead, sample budget/early-stop, TTS protocol
- **P2 (Medium)**: 3 issues — HP tune, online tracking, README hierarchy
- **P3 (Low)**: 2 issues — external baselines, log cadence

## Recommended attack order

1. **P0 SAC systems** (sync collapse → kill redundant forwards → compile) — unlocks Pendulum TTS
2. **P0/P1 compile + log cadence** on PPO path if not already fused — unlocks CartPole wall
3. **P1 early-stop + cheap eval** — stop paying fixed 32k/65k and eval tax
4. **P1 multi-seed median TTS protocol** + warm/cold split
5. **P2 HP tune** toward target steps, then **P2 README** reframe
6. **P3** baselines once numbers are honest and fast

## Key consensus

| Theme | Reviewers |
|-------|-----------|
| Train step must be compiled / low-sync | Claude, DeepSeek, Gemini |
| Wall time ≠ train FPS (eval/logging tax) | Gemini, ChatGPT |
| Optimize samples **and** samples/sec (TTS = STS/FPS) | ChatGPT, Gemini |
| Multi-seed median after systems fix | Claude, ChatGPT |
| Huge-env FPS is secondary / misleading as headline | DeepSeek, ChatGPT |

## Overall assessment

Reviewers agree solve time is the **primary** product metric and is currently bottlenecked by **host sync + unfused updates + eval/budget overhead**, not by env simulation. Fix systems (P0) first, then stop-on-competence and protocol (P1), then HP/docs (P2).

No code changes until you approve this list / pick what to implement.
