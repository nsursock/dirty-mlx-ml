# Feedback Priority List

Issues identified from AI feedback reviews, organized by priority level.

| Priority | Issue | Main Finding | Fix | Impact | Reviewer(s) | Commit Hash |
|----------|-------|--------------|-----|--------|-------------|-------------|
| **P0** | SAC Terminated/Truncated Bootstrapping | SAC has incorrect Bellman target handling for terminated/truncated states | Fix SAC terminated/truncated bootstrapping; make `terminated` and `truncated` first-class internally | Fast wrong RL is still wrong RL - fundamental correctness issue | ChatGPT | `4929631` |
| **P0** | Algorithmic Validation Layer | Gap between systems quality and scientific quality; need algorithmic validation to match systems engineering | Build Gymnasium transition-by-transition differential tests; build SB3 numerical differential tests for PPO/SAC | Essential for credibility; current performance claims aren't scientifically validated | ChatGPT, Claude | `7853d4e` |
| **P1** | Multi-Seed Benchmarks | Single training run, one seed, one evaluation lacks statistical significance | Change benchmarks to use 5 seeds with median±std reporting; define solve as evaluation return ≥ threshold for N consecutive evaluations | Critical for scientific validity; current results could be noise vs regression | Claude, ChatGPT | `b416fc8` |
| **P1** | SB3 Baseline Comparisons | No comparison against established baselines on same hardware | Add SB3 on CPU, JAX implementations on same Mac, PyTorch MPS comparisons; report normalized metrics (samples/sec, time-to-threshold) | Without baselines, performance claims are not scientifically meaningful | ChatGPT, DeepSeek, Gemini | - |
| **P1** | PPO Train FPS Bottleneck | PPO train FPS plateaus at ~700k FPS between 1k-8k envs while env FPS scales linearly | Check if GAE/advantage calculations are running on CPU or forcing uncompiled loop; ensure `mx.compile()` wraps minibatch update loop | Major performance bottleneck preventing scaling benefits from being realized | Gemini | `325cf12` |
| **P1** | Memory Non-Monotonicity | PPO memory drops from 96.6MB → 67.6MB at 8,192 envs; unclear if measurement artifact | Sample memory multiple times and take median, or note caveat in README | Undermines credibility of memory measurements | Claude | `92d2b03` |
| **P2** | Hardware Context Missing | No Apple Silicon chip model specified (e.g., M3 Max 16-core CPU/40-core GPU, 64GB Unified Memory) | Add hardware specifications to README benchmark section | Numbers mean nothing without hardware context | Gemini | `48bfc74` |
| **P2** | Wall Time vs Throughput Discrepancy | PPO 16 envs shows 28,420 Train FPS in scaling table but only 3,433 FPS effective solve rate | Add footnote explaining solve time overhead (policy evaluation rollouts, environment reset delays, logging, checkpointing) | Readers will think scaling table numbers are artificial micro-benchmarks | Gemini | `140925f` |
| **P2** | PPO Threshold Clarity | PPO threshold 440.0 is custom, not Gym's actual 475 solved criterion | Label threshold as custom bar rather than implying it's canonical benchmark | Avoids misleading benchmark claims | Claude | `6ae429a` |
| **P2** | SAC Solve Performance Investigation | SAC 16 envs worse than 8 envs (−162.7 vs −148.2) despite more timesteps | Investigate if this is noise or regression; need multi-seed data | Currently unclear if more parallelism helps or hurts | Claude | `6c9e7f1` |
| **P3** | SAC Sweep Truncation Explanation | SAC train FPS still climbing at table end (178K→1.18M) but stops at 8,192 envs | Explain why sweep stopped at 8,192 (truncated for brevity vs real ceiling) | Currently misleading about where scaling actually ends | Claude | - |
| **P3** | Solve Performance Methodology | Current "time to solve" based on one training run, one seed, one evaluation | Improve methodology as noted in P1 multi-seed benchmarks | Better benchmark practices | ChatGPT | - |

## Summary

- **P0 (Critical)**: 2 issues - Fundamental correctness problems (**2 completed** ✅)
- **P1 (High)**: 4 issues - Performance and reliability concerns (**3 completed** ✅)
- **P2 (Medium)**: 4 issues - Documentation and clarity needs (**4 completed** ✅)
- **P3 (Low)**: 2 issues - Minor improvements

## Key Consensus Issues

Multiple reviewers flagged these as high priority:
- **Multi-seed benchmarks** (Claude, ChatGPT)
- **SB3 baseline comparisons** (ChatGPT, DeepSeek, Gemini)

## Overall Assessment

The consensus across reviewers is that the systems engineering is solid (especially after the FPS fix), but the scientific validation layer needs significant work to match the engineering quality.
