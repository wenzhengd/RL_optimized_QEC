# Human-Codex Q&A

## Purpose
Track key Q/A decisions between human and Codex for this project.

## Entries

### 2026-03-09
Q: 我们现在做的所有 simulation all stage，是不是 train 完之后只有 final_logical_success_rate 这一个指标？还有一个问题：我们需要对比如果不用 RL 学习，原始 success_rate 更低，这个是不是没有计算？还有什么其他指标可以拿来说明 RL-agent improve performance？
A:
1. 不是只有一个指标。当前每个 run 都会保存 `eval_metrics`，其中有三组策略：`learned`、`fixed_zero`（不用 RL 的固定零控制）、`random_uniform`。
2. 所以“不用 RL 的原始 success_rate”其实已经在算，字段是 `eval_metrics.fixed_zero.success_rate`（以及对应的 `detector_rate`、`ler_proxy`）。
3. 当前 staged 汇总里默认聚合的是：
   - `improvement_vs_fixed_zero`（`detector_rate`/`ler_proxy` 的 mean/std）
   - `learned_policy`（`success_rate`/`ler_proxy` 的 mean/std）
4. 另外每个 run 还存了可用指标：`improvement_vs_random_uniform`、`phase1_mean_rollout_reward`、`trace_finetune_mean_rollout_reward`，以及开启 trace-eval 时的 `trace_eval_metrics`。
5. 如果要更强地证明 RL 有效，建议额外报告：
   - `delta_success = learned.success_rate - fixed_zero.success_rate` 的跨 seed 均值与置信区间
   - seed-level 胜率（多少 seed 上 learned 优于 fixed_zero）
   - 每单位时间收益（improvement / wall-clock）
   - 不同噪声通道下的一致性（composite/correlated 等）

Q: Google 的 paper 里采用了哪些指标来 demonstrate using RL 的 advantage？这些里面哪些可以从我们现在 simulation 直接得到，哪些需要改代码甚至改 architecture？
A:
1. 这里按 arXiv:2511.08493（Google Quantum AI, *Reinforcement Learning Control of Quantum Error Correction*）口径总结。论文核心指标是：
   - `LER`（logical error rate，论文明确说是 QEC 质量主指标）
   - `DR`（detection rate，作为可扩展 surrogate objective/learning signal）
   - `LER stability under drift`（在漂移下的稳定性改善，含 fold 提升）
   - 相对传统 calibration / human expert 的对比结果
   - scalability 指标：优化收敛速度是否随系统规模恶化（论文报告对 system size 近似独立）
2. 我们现在“直接可得”的：
   - `learned/fixed_zero/random` 三组 `success_rate`
   - `ler_proxy = 1 - success_rate`
   - `improvement_vs_fixed_zero`、`improvement_vs_random_uniform`
   - 多 seed 的 mean/std 聚合
3. “部分可得，但证据强度有限”的：
   - `DR`：当前 fast path 下常退化为 `1-success` proxy；只有 trace/高保真路径才更接近 detector-driven 指标
   - drift 下稳定性：环境支持 drift，可跑出趋势，但当前缺少固定 protocol 的自动化统计报表
4. “需要改代码”的：
   - 系统化输出 `LER stability`（如 std/percentile/Allan-like 指标）并做跨 seed 置信区间
   - 固定 drift benchmark 套件（step/sine/stroboscopic）和统一对照表
   - 将 trace-based detector metrics 作为常规 eval 工件（而非可选一次性附加）
5. “需要改 architecture / 训练范式”的：
   - 若要接近论文的“实时 steering during computation”，需要从当前以 `candidate_eval` 为主的一次候选策略，扩展到更强的在线闭环策略（`online_rounds` + 时序策略，例如 RNN/GRU/Transformer policy）
   - 若要复现“优于传统 calibration + human expert”的结论，需要引入对应 baseline pipeline（目前我们的 baseline 主要是 `fixed_zero/random`，口径不同）

Q: 我想具体测试 composite channel 下 RL 对 QEC 的性能提升。请制定详细计划：考虑不同数据规模、不同 correlated 参数（强度 g、频率 f），分析 RL 在低频/高频、强噪声/弱噪声下的学习能力。参数组合很多，需要先做可执行计划并估算时间。
A:
1. 目标与核心问题：
   - 目标是建立 `improve(LER~)` 和 `delta_success = learned - fixed_zero` 在 `(f,g)` 平面上的响应图，并验证结论在更高预算下是否稳健。
   - 重点回答：低频 vs 高频、弱噪声 vs 强噪声时，RL 提升是否显著、是否稳定。
2. 参数与预算设计（避免组合爆炸）：
   - 频率分层：`f in {1e2, 1e3, 1e4}`（低/中/高频）。
   - 强度分层：`g in {0.4, 1.0, 1.6}`（弱/中/强噪声）。
   - 先做 `3x3=9` 点粗筛，再对少量候选做中高预算确认；不要一开始跑全量大网格。
3. Phase 0（30-60 分钟，校准）：
   - 先跑 2 个点的单条件小样本（例如 `(1e2,1.6)` 和 `(1e4,0.4)`）测当前机器真实耗时，用于修正后续预算。
4. Phase 1（粗筛，Quick 预算）：
   - 配置参考现有 quick spec：`total_timesteps=512, rounds=4, shots=4, trace_ft=0, seeds=3`。
   - 跑 9 个 `(f,g)` 组合，输出每点 `improve(LER~)`、`delta_success`、seed-level 正收益比例。
   - 时间估算（5 核并行）：
     - 保守（按旧日志）：每条件约 `11-12 min`，9 条件约 `1.7-2.0 h`。
     - 期望（按近期 backend 提速经验）：约 `0.9-1.2 h`。
5. Phase 2（聚焦，Pilot 预算）：
   - 从 Phase 1 选 4 个代表点（通常：低频最佳/高频最佳/低频困难点/高频困难点），再加 2 个锚点（`f=1e3,g=1.0` 与当前默认附近），共 6 条件。
   - 配置参考 pilot/stage7：`total_timesteps=1024, rounds=6, shots=6, trace_ft=192, seeds=5`。
   - 时间估算：
     - 保守：每条件 `60-75 min`，总计 `6-7.5 h`。
     - 期望：总计 `3-4.5 h`。
6. Phase 3（确认，10 seeds）：
   - 选 3 个最终条件做确认（建议：低频最优、高频最优、中心参考点）。
   - 先用 stage7 预算做 `10 seeds` 确认；如结论仍有争议，再对冠军点做 stage8 高预算压力测试。
   - 时间估算：
     - stage7-10seed：每条件约 `1.2-2.5 h`，3 条件约 `3.5-7.5 h`。
     - stage8-10seed（可选）：每条件约 `3-6 h`。
7. 统计与决策门槛（每阶段都执行）：
   - 主指标：`improve(LER~)`、`delta_success`（learned-fixed_zero）。
   - 稳健性：seed-level 正收益比例、mean±std、95% CI（或 bootstrap CI）。
   - 晋级规则建议：进入下一阶段的条件需满足 `delta_success` 均值为正且 seed-level 正收益比例 >= `70%`。
8. 可直接复用的现有资产：
   - 已有 quick/pilot/confirm 模板：`code/rl_train/benchmarks/examples/stage_specs_composed_corr_g_ladder_*.json`。
   - 已有 quick 实测产物：`code/data_generated/steane_composed_f1e3_g_ladder_quick/summary.json`。
   - 下一步只需新增 `f` 维度的 stage-spec（目前模板主要固定 `f=1e3`）。

Q: 可以，跑的数据要进入到 benchmarks。
A:
1. 已把 `(f,g)` 计划接入 benchmark 目录，新增了可直接运行的 stage-spec：
   - `code/rl_train/benchmarks/examples/stage_specs_composed_corr_fg_phase0_calibration.json`
   - `code/rl_train/benchmarks/examples/stage_specs_composed_corr_fg_phase1_quick_grid.json`
   - `code/rl_train/benchmarks/examples/stage_specs_composed_corr_fg_phase2_pilot_focus.json`
   - `code/rl_train/benchmarks/examples/stage_specs_composed_corr_fg_phase3_confirm_template.json`
2. 已新增 runbook：
   - `code/rl_train/benchmarks/examples/composed_corr_fg_benchmark_plan.md`
   - 统一输出目录约定为 `code/data_generated/benchmarks/composed_corr_fg/phaseX_YYYYMMDD/`
3. 已新增自动汇总脚本：
   - `code/rl_train/benchmarks/summarize_composed_fg_grid.py`
   - 可把 `summary.json` 转为 CSV/Markdown 排名表（含 improve(LER~)、delta success、正收益 seed 比例）。
4. 已完成首轮落地运行（Phase0 calibration）：
   - 输出目录：`code/data_generated/benchmarks/composed_corr_fg/phase0_20260309/`
   - 产物：`summary.json`、`wall_seconds.txt`、`composed_fg_grid_summary.csv`、`composed_fg_grid_summary.md`

Q: 通过 plots_20260309，我感觉 RL advantage 没那么强。能不能更强？要参考之前 staged correlated 结果，不要盲目加数据量；另外这些 phase 跑得很快，会不会是数据量太小？
A:
1. 你的观察是对的，但“跑得快”不等于“数据量小”。当前 phase2/phase3 用的是 stage7 级别预算（`total_timesteps=1024`, `trace_finetune_timesteps=192`, `n_rounds=6`），和此前 composed-staged 有可比性；只是后端优化后 wall-clock 变短了。
2. 与历史结果对齐：
   - 历史 composed-staged（`f=1e3,g=1.6`）在 `result_analysis.md` 里是 `+36.91%`（stage7）和 `+40.64%`（stage8）。
   - 当前 phase3 top3 分别是 `+37.52%`、`+29.77%`、`+28.50%`，不是“无优势”，但确实存在条件敏感和方差问题。
3. 你现在图里主指标来自 fast `eval_metrics`；在当前配置下它偏 proxy（`collect_traces=false`），建议把 `trace_eval_metrics` 也作为主报告通道。对 phase3 的快速核算显示：
   - `f=1e4,g=0.4`: fast `+37.52%`，trace `+42.86%`
   - `f=1e3,g=1.6`: fast `+29.77%`，trace `+28.19%`
   - `f=1e3,g=1.0`: fast `+28.50%`，trace `+20.99%`（方差更大）
4. 不盲目加数据量、但最可能提强的 3 个方向：
   - 在 top 条件（`f=1e4,g=0.4`）做小规模超参复用：优先试此前有效的 tune-a 组合（更低 LR + trace LR + entropy 组合），先 5 seeds 再 10 seeds。
   - 提升“信号质量”而非全面加算力：固定 phase1 budget，只加 `trace_finetune_timesteps`（如 `192 -> 512/1024`）做对照。
   - 统一用 trace-eval 作为最终结论口径（例如 `trace_eval_episodes` 与 `trace_eval_shots` 提高），避免被 fast-proxy 低估或扭曲。
5. 建议下一轮最小实验包（成本可控）：
   - 条件固定：`f=1e4,g=0.4`
   - 对照 A：当前超参
   - 对照 B：tune-a 超参
   - 对照 C：当前超参 + 更长 trace-finetune
   - 每个先跑 5 seeds；仅将表现最好的 1-2 个扩展到 10 seeds。

Q: OK, 请继续按照你的计划，跑下一步的数据，给 RL advantage 补充更有力的证据。
A:
1. 已执行 phase4 top-condition ablation（固定 `f=1e4,g=0.4`，每组 5 seeds）：
   - A: baseline
   - B: tune-a（wider+LayerNorm+lower LR）
   - C: long-trace（`trace_finetune_timesteps: 192 -> 1024`）
2. 输出目录：
   - `code/data_generated/benchmarks/composed_corr_fg/phase4_20260309_topcond_ablation/`
   - 含 `summary.json`、`wall_seconds.txt`、`composed_fg_grid_summary.csv/.md`
   - 额外图：`topcond_ablation_fast_vs_trace.png`
3. 结果（fast 口径）：
   - baseline: `+28.26% +- 7.05%`
   - tune-a: `+34.86% +- 19.86%`（均值最高但方差最大）
   - long-trace: `+33.66% +- 11.78%`
4. 结果（trace 口径，更建议作为主证据）：
   - baseline: `+33.27% +- 11.16%`
   - tune-a: `+26.37% +- 21.73%`（`trace +seed=80%`）
   - long-trace: `+36.61% +- 15.14%`（`trace +seed=100%`）
5. 结论：
   - 若强调“稳健证据”，当前最优是 `long-trace`（trace 口径最高且正收益比例最好）。
   - 若只看 fast 均值，tune-a 最高，但不稳，暂不建议直接定为最终方案。
6. 继续强化（phase5, 10 seeds, baseline vs long-trace）：
   - 目录：`code/data_generated/benchmarks/composed_corr_fg/phase5_20260309_confirm10/`
   - baseline（10 seeds）：
     - fast: `+31.90% +- 8.10%`
     - trace: `+36.81% +- 6.26%`
   - long-trace（10 seeds）：
     - fast: `+45.01% +- 11.67%`
     - trace: `+48.23% +- 11.30%`
   - 两臂都 `+seed ratio = 100%`，且 long-trace 在 fast/trace 双口径都明显优于 baseline，构成更强证据链。

Q: 请基于 composed_corr_fg 所有 simulation 给一个小节分析，说明 RL advantage 程度；并给出下一步简单计划。
A:
1. 总体结论：
   - RL advantage 在 composed-corr 设置下是“明确为正、但对 `(f,g)` 条件和训练信号质量敏感”。
   - 经过 phase2/3/5 逐步确认，结论已经从“可能有效”提升到“在关键条件上稳健有效”。
2. 证据分层：
   - 粗筛（phase1, 3 seeds）波动很大，出现正负混杂，不能单独下结论。
   - 聚焦（phase2, 5 seeds）全部候选转为显著正收益，顶点条件 `f=1e4,g=0.4` 达 `+45.20%`。
   - 确认（phase3, 10 seeds）top3 仍全正，其中 `f=1e4,g=0.4` 为 `+37.52% +- 6.16%`（100% 正 seed）。
   - 强化（phase5, 10 seeds）显示 long-trace 相比 baseline 有稳定增益：
     - fast: `31.90% -> 45.01%`（+13.11 pct-points）
     - trace: `36.81% -> 48.23%`（+11.42 pct-points）
3. 与历史 staged-composed 对齐：
   - `f=1e3,g=1.6` 的历史 stage7/8 为 `+36.91%` / `+40.64%`；
   - 本轮在 `f=1e4,g=0.4` + long-trace 下已达到同量级甚至更高（trace `+48.23%`）。
4. 现阶段最稳方案：
   - `f=1e4,g=0.4` 条件下采用 long-trace finetune（`trace_finetune_timesteps=1024`）。
5. 下一步简计划（低风险、可执行）：
   - 先固定“top condition + long-trace”为主线，做一次 20-seed power run（其余设置不变）。
   - 报告统一以 trace 口径为主，fast 口径为辅（双口径并列）。
   - 若 20-seed 仍稳定为正，再只做一个小规模横向检查（如 `f=1e3,g=1.6`）验证可迁移性。

### 2026-03-10
Q: 这个地方我没看懂，是不是比较 `topcond_ablation_fast_vs_trace` 在 `phase6_20260310_confirm40` 和 `phase5_20260309_confirm10` 两个文件中的 error bar？95%CI 半宽具体在哪个图里看出来？
A:
1. 你看得很对，比较对象就是这两次 confirm（10 seeds vs 40 seeds）。
2. 但原图 `topcond_ablation_fast_vs_trace.png` 的 error bar 不是 95% CI，它画的是 `std`。
3. 所以“95% CI 半宽”不能直接从原图读出来，需要按公式计算：`CI95_half = 1.96 * std / sqrt(n_seeds)`。
4. 现在已补一张专门的 95% CI 图（对比 10 vs 40，Baseline vs Long-Trace，fast/trace 两个面板）：
   - `code/data_generated/benchmarks/composed_corr_fg/plots_20260310/topcond_confirm10_vs_40_ci95.png`
5. 对应数值（`improve(LER~)`）：
   - confirm10 baseline：fast `31.90% +- 5.02%`，trace `36.81% +- 3.88%`
   - confirm10 long-trace：fast `45.01% +- 7.23%`，trace `48.23% +- 7.01%`
   - confirm40 baseline：fast `32.49% +- 3.91%`，trace `32.74% +- 4.00%`
   - confirm40 long-trace：fast `43.43% +- 3.01%`，trace `44.22% +- 3.94%`
6. 结论：40 seeds 的 CI 半宽显著缩小（尤其 long-trace fast 从 `+-7.23%` 降到 `+-3.01%`），但均值层面并没有“因为加 seed 而变更强”，而是变得更稳、更可置信。

Q: not for beamer, but how google model the measurement error actually?
A:
1. 根据 Google RL-QEC 论文与补充材料（Dec 2025），measurement error 被纳入其“器件表征驱动的 Pauli 模型”中，而不是作为一个单独展开推导的 RL 专用新模型。
2. 文中明确给出 measurement error 的量级约为 `5×10^-3`（对 `|0>` 与 `|1>` 平均）。
3. 同时还计入了 readout/reset 期间数据比特 DD 对应的误差分量（约 `6.9×10^-3`）。
4. 论文没有给出完整测量通道的显式方程（如完整 POVM / confusion-matrix 细节），但说明了该 Pauli 模型可复现实验 detector rate（Fig. S1(c)）。

Q: do they model the measurement error in form of channel?
A:
1. 是，但属于“有效电路层”的误差通道表征。
2. 也就是说，measurement error 作为 Pauli/detector-error workflow 的一部分被建模；论文里没有把它写成独立的一套一阶原理测量通道推导。

Q: any suggestions of modeling the stabilizer measurent error in a form of channel ? any reference did such a job and can be incorporated into our work?
A:
1. 推荐分层建模路径（从易到难）：
   - `Phenomenological readout flip`: `m~ = m xor e`, `e ~ Bernoulli(p_m)`。
   - `Asymmetric readout channel`: 用 2x2 confusion matrix (`p01`, `p10`) 描述 `|0>` 与 `|1>` 非对称读出。
   - `Spatially correlated readout`: 以邻近 ancilla 小簇建立联合 assignment model，刻画读出串扰。
   - `Temporally correlated readout`: 用 HMM/Markov 隐状态驱动 `p_m(t)`，与现有 correlated-noise 框架兼容。
2. 对我们当前代码的最实用落地顺序：
   - 先实现 `readout_flip` + `asymmetric_readout`；
   - 再增量加最近邻簇相关（spatial）和时间相关（temporal）版本。
3. 参考文献与资料：
   - Google RL-QEC (arXiv:2511.08493): https://arxiv.org/abs/2511.08493
   - Stim gate reference (`M(p)`, `MR(p)`): https://github.com/quantumlib/Stim/wiki/Stim-v1.10-Gate-Reference
   - Stim generated-circuit params (`before_measure_flip_probability`, `after_reset_flip_probability`): https://github.com/quantumlib/Stim/wiki/Stim-v1.13-Python-API-Reference
   - Fowler et al. 2012 (surface-code circuit-level baseline): https://journals.aps.org/pra/abstract/10.1103/PhysRevA.86.032324
   - Bravyi et al. (phenomenological readout-focused surface-code analysis): https://quantum-journal.org/papers/q-2023-09-21-1116/
   - Nation et al. (assignment-matrix readout modeling): https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.042605
   - Bravyi et al. (correlated readout mitigation/modeling direction): https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040326
4. 结论：
   - Google 的实现语境是“characterization-driven effective channel”；
   - 对我们来说，先上 assignment-matrix / phenomenological channel 最快，也最容易并入现有 QEC+RL 基线并开展 ablation。

### 2026-03-11
Q: the whole RL part must be well-defined by some parameters that are customizible, are they ONLY configured by the config.py ? [This is crucial question! 🔥🔥🔥]
A:
1. 不是。`code/rl_train/config.py` 只定义了 `PPOConfig` 这个 PPO 侧默认参数容器，例如 `total_timesteps`、`rollout_steps`、`learning_rate`、`hidden_dim`、`seed` 等。
2. 实际运行时，主要配置入口不是只靠 `config.py`，而是 `code/rl_train/train.py` 里的 CLI 参数。这里暴露了：
   - PPO 超参数：如 `--total-timesteps`、`--rollout-steps`、`--ppo-learning-rate`、`--ppo-hidden-dim`
   - 环境/任务参数：如 `--max-steps`
   - Steane 适配器参数：如 `--steane-n-rounds`、`--steane-shots-per-step`、`--steane-control-mode`、`--steane-noise-channel`
   - reward 相关参数：如 `--steane-reward-mode`、`--steane-action-penalty-coef`
3. `train.py` 会把这些 CLI 参数再组装成 `PPOConfig`，所以运行时通常是 **CLI 覆盖** `config.py` 默认值，而不是直接只读 `config.py`。
4. 除了 `PPOConfig`，Steane 环境本身还有单独的配置数据结构：`code/rl_train/steane_adapter.py` 里的 `SteaneAdapterConfig`。也就是说，R**L 的“可配置参数”分成至少两层**：
   - PPO/优化器层
   - simulator / code-family / noise / reward 层
5. benchmark 入口也有自己的一套参数面：
   - `code/rl_train/benchmarks/eval_steane_ppo.py`
   - `code/rl_train/benchmarks/staged_steane_experiments.py`
   这些脚本除了复用训练参数，还增加了 `trace_finetune`、`post_eval`、分 stage 运行、seed 并行等控制项。
6. staged benchmark 还支持 JSON 驱动配置，而不需要改 Python：
   - `--stage-specs-json`
   - `--base-overrides-json`
   所以批量实验时，很多 RL 参数实际上是通过 stage-spec / override JSON 注入的。
7. 准确说法应该是：
   - `config.py` 只是 PPO 默认参数定义处之一；
   - 真正的主要配置面是 `train.py` / benchmark scripts 的 CLI；
   - 更大规模实验还会通过 stage-spec JSON 和 override JSON 配置，而不是只改 `config.py`。

Q: Can you classify the RL parameters into clear groups, so I know where each kind of thing should be configured? [This is crucial question! 🔥🔥🔥]
A:
1. 可以。当前这套 RL 配置大致可以分成 5 层：
   - `PPO / optimizer`
   - `environment / task protocol`
   - `noise / simulator physics`
   - `reward shaping`
   - `benchmark orchestration / experiment management`
2. `PPO / optimizer` 层：
   - 作用：决定 policy/value 网络怎么训练。
   - 典型参数：`total_timesteps`、`rollout_steps`、`ppo_learning_rate`、`ppo_update_epochs`、`ppo_minibatch_size`、`ppo_ent_coef`、`ppo_hidden_dim`、`ppo_use_layer_norm`、`seed`、`device`
   - 主要入口：`code/rl_train/config.py` + `code/rl_train/train.py` / `code/rl_train/benchmarks/eval_steane_ppo.py`
3. `environment / task protocol` 层：
   - 作用：定义一个 RL episode/step 到底对应什么物理或仿真过程。
   - 典型参数：`max_steps`、`steane_stepping_mode`、`steane_n_rounds`、`steane_shots_per_step`、`steane_control_mode`、`steane_control_dim`、`steane_n_1q_control_slots`、`steane_n_2q_control_slots`、`steane_syndrome_mode`
   - 主要入口：`code/rl_train/train.py`、`code/rl_train/steane_adapter.py`
4. `noise / simulator physics` 层：
   - 作用：定义漂移、门噪声、idle noise、correlated channel 等物理扰动模型。
   - 典型参数：`steane_noise_channel`、`steane_drift_period_steps`、`steane_drift_amplitude`、`steane_p1q_base`、`steane_p2q_base`、`steane_sensitivity_1q`、`steane_sensitivity_2q`、`steane_p_clip_max`、`steane_idle_p_total_per_idle`、`steane_channel_corr_f`、`steane_channel_corr_g`、`steane_channel_corr_g_mode`
   - 主要入口：`code/rl_train/steane_adapter.py`，底层实现进一步落到 `code/quantum_simulation/`
5. `reward shaping` 层：
   - 作用：定义 agent 学什么，而不是 simulator 本身怎样演化。
   - 典型参数：`steane_reward_mode`、`steane_action_penalty_coef`、`steane_miscal_penalty_coef`、`steane_success_bonus_coef`
   - 主要入口：`code/rl_train/train.py` 里的 `make_steane_reward_fn(...)`
6. `benchmark orchestration / experiment management` 层：
   - 作用：不是单次训练本身，而是如何批量跑实验、分阶段、做后评估、trace finetune、保存 JSON。
   - 典型参数：`post_eval_episodes`、`eval_steane_shots_per_step`、`trace_finetune_timesteps`、`trace_finetune_rollout_steps`、`trace_finetune_shots_per_step`、`trace_eval_episodes`、`save_json`、`stages`、`stage_specs_json`、`base_overrides_json`、`seed_workers`
   - 主要入口：`code/rl_train/benchmarks/eval_steane_ppo.py`、`code/rl_train/benchmarks/staged_steane_experiments.py`
7. 最实用的理解方式：
   - 如果你想改“怎么训练 policy”，先看 PPO 层。
   - 如果你想改“agent 每一步在控制什么任务”，先看 environment 层。
   - 如果你想改“噪声/漂移/相关信道长什么样”，先看 noise 层。
   - 如果你想改“奖励到底鼓励什么”，看 reward 层。
   - 如果你想批量扫参数、做多 seed / 多 stage 实验，主要看 benchmark orchestration 层。
8. 所以配置责任应该这样理解：
   - `config.py` 负责的是 PPO 默认值的一部分；
   - `train.py` 负责单次训练的大部分可调入口；
   - `steane_adapter.py` 负责 simulator/task/noise 这层的结构化配置；
   - `benchmarks/*.py` 负责实验管理层，而不是替代前面三层。



Q: 我们现在针对 noisy circuit 模拟里各种 error channel，怎么理解每个 channel 的参数物理意义？尤其 correlated noise 里的 `f,g`，如果电路只有 layered 次序、没有真实时间长短，那它们怎么才算 well-defined？[🔥🔥🔥]
    292 +A:
    293 +1. 先说总原则：在当前代码里，`f,g` 不是“悬空”的自由符号；它们只有在你先接受一套**时间约定**之后才有物理意义。
    294 +2. 这个 repo 已经明确写死了一套时间约定，在 `code/quantum_simulation/noise_engine.py`：
    295 +   - 串行调度（不是并行 layer）
    296 +   - `1q gate = 10 ns`
    297 +   - `2q gate = 20 ns`
    298 +   - `measurement = 100 ns`
    299 +   - `reset = 100 ns`
    300 +   - 相邻指令之间固定插入 `idle_ns = 200 ns`
    301 +3. 因此，当前 noisy-circuit 其实不是“只有顺序、完全没时间”，而是：
    302 +   - **文本顺序 -> serial timeline**
    303 +   - **timeline -> 带 ns 单位的 synthetic clock**
    304 +   - 各种 time-dependent / correlated channel 都是相对于这个 synthetic clock 定义的
    305 +4. 各 channel 及参数语义：
    306 +   - `auto`
    307 +     - 不是独立物理模型，只是兼容层。
    308 +     - `control_mode=global` 时映射到 `google_global`；`gate_specific` 时映射到 `google_gate_specific`。
    309 +   - `google_global`
    310 +     - 物理含义：所有门共享一个全局控制失配标量，失配越大，门 depolarizing rate 越高。
    311 +     - 关键参数：
    312 +       - `p_1q_base`, `p_2q_base`: 1q/2q 的 action 无关底噪
    313 +       - `sensitivity_1q`, `sensitivity_2q`: 控制失配转成门错误率的斜率
    314 +       - `p_clip_max`: 概率截断上限
    315 +       - `drift_period_steps`, `drift_amplitude`: 最优控制 `u_opt` 的漂移周期和幅度
    316 +     - 设定建议：
    317 +       - `p_base` 先来自你想模拟的器件 baseline gate infidelity
    318 +       - `sensitivity` 用来控制“错控一点会带来多大额外 gate noise”
    319 +       - `drift_period_steps` 决定非平稳性快慢；大表示慢漂移
    320 +   - `google_gate_specific`
    321 +     - 物理含义：不是所有门共用一个失配，而是每类/每位置门映射到某个 control slot，各自有局部失配。
    322 +     - 额外参数：
    323 +       - `n_1q_control_slots`, `n_2q_control_slots`: 1q/2q gate 对应多少个独立控制槽
    324 +     - 设定建议：
    325 +       - 想表达“不同门族/不同位置误差可分开校准”时用它
    326 +       - slot 数越大，自由度越强，但训练更难
    327 +   - `idle_depolarizing`
    328 +     - 物理含义：只在 idle window 注入 action 无关的 Pauli idle noise。
    329 +     - 关键参数：
    330 +       - `idle_p_total_per_idle`: 每个 idle window 的总非平凡 Pauli 概率
    331 +       - `idle_px_weight`, `idle_py_weight`, `idle_pz_weight`: X/Y/Z 分配权重
    332 +     - 设定建议：
    333 +       - 如果你只想加静态 idle 背景噪声，这是最直接模型
    334 +       - 若没有方向偏置证据，先设成 `(1,1,1)`
    335 +   - `parametric_google`
    336 +     - 物理含义：结构上和 `google_gate_specific` 一样，但把 gate-noise 整体强度交给显式 regime knob 来扫参数。
    337 +     - 关键参数：
    338 +       - `channel_regime_a`, `channel_regime_b`
    339 +     - 代码语义：
    340 +       - `a` 同时缩放 1q 的 `p_base` 和 `sensitivity`
    341 +       - `b` 同时缩放 2q 的 `p_base` 和 `sensitivity`
    342 +     - 设定建议：
    343 +       - 想做“同一类模型，不同器件 regime / 不同门族强弱”扫描时用它
    344 +   - `correlated_pauli_noise_channel`
    345 +     - 物理含义：只在 idle window 注入 temporally correlated Pauli noise；X/Y/Z 各有独立两态 HMM 链，但共享同一个 `(f,g)`。
    346 +     - 关键参数：
    347 +       - `channel_corr_f`: 相关频率，控制 hidden state 在相邻 idle window 之间切换多快
    348 +       - `channel_corr_g`: 强度尺度，控制平均 idle Pauli 强度
    349 +       - `channel_corr_g_mode`: `per_window` 或 `per_circuit`
    350 +       - `p_1q_base`, `sensitivity_1q`, `p_clip_max`: 用来先生成目标总强度，再乘上 `g`
    351 +     - 设定建议：
    352 +       - `f` 决定 memory time；小 `f` = 慢变化、长相关
    353 +       - `g` 决定总体噪声强度，不直接决定相关时间
    354 +       - 若比较不同 circuit 长度，优先用 `per_circuit`
    355 +   - `composed_google_global_correlated` / `composed_google_gate_specific_correlated`
    356 +     - 物理含义：门上有 Google-like depolarizing，idle 上有 correlated Pauli，两者一次性组合。
    357 +     - 参数就是前两类的并集。
    358 +     - 设定建议：
    359 +       - 如果你认为器件同时有“门错控 + correlated idle background”，这是最合理的综合模型。
    360 +5. `f` 在当前实现里的严格含义：
    361 +   - 它不是直接对“整条 circuit 总时间”起作用，而是先转成**相邻 idle window 的状态翻转概率**。
    362 +   - 代码里：
    363 +     - `idle_s = idle_ns * 1e-9`
    364 +     - `rho = exp(-f * idle_s)`
    365 +     - `gamma = (1 - rho) / 2`
    366 +   - 然后 Hidden-Markov telegraph 链每经过一个 idle window，就按 `gamma` 更新一次状态。
    367 +6. 所以在当前模型下，`f` 的真实解释是：
    368 +   - 给定固定 `idle_ns` 后，`f` 决定“每个 idle window 之间相关性衰减多快”
    369 +   - 或者等价地，决定以 ns 计的相关时间常数大约是多少
    370 +   - 但因为代码只在 idle window 更新状态，所以它最终体现为“每窗口一步”的 Markov 记忆强度
    371 +7. `g` 在当前实现里的严格含义：
    372 +   - 先算一个由 control mismatch 给出的 baseline strength：
    373 +     - `base_strength = p_1q_base + sensitivity_1q * mismatch_global`
    374 +   - 再乘 `g` 得到目标总 Pauli 强度：
    375 +     - `p_total_target = clip(base_strength * g, 0, p_clip_max)`
    376 +   - 所以 `g` 是**强度缩放因子**，不是时间参数。
    377 +8. 你提到的关键问题是对的：
    378 +   - 同一个物理过程，如果实验总时长 `T` 不同，那么“低频/高频”“强/弱”这些词本来都依赖时间尺度。
    379 +   - 如果模拟里完全没有时间标尺，只剩 layer index，那么 `f` 用 Hz 表示其实是不严谨的。
    380 +9. 但在我们当前代码里，这个问题被下面这一步“封闭”了：
    381 +   - repo 明确规定了 `GateDurations` 和 `idle_ns`
    382 +   - 所以 `f` 的 Hz 单位是相对于这套 synthetic time 定义的
    383 +   - 也就是说，当前问题是 well-defined 的，但**只在这套时间约定下 well-defined**
    384 +10. 更直白地说：
    385 +   - 现在的 `f=1e4 Hz` 并不是“真实实验平台上绝对的 10 kHz”
    386 +   - 它是“在本 repo 规定的 serial+idle=200ns 时间刻度下，对应的 10 kHz telegraph correlation frequency”
    387 +   - 如果你改了 `idle_ns` 或整体 scheduling 假设，同一个 `f` 的统计意义会变
    388 +11. `g` 的 well-defined 也依赖你选择的归一化口径：
    389 +   - `per_window`: `g` 表示每个 idle window 的局部强度缩放
    390 +   - `per_circuit`: 先把整条 circuit / 一次 simulator step 的总预算固定，再反推每窗口强度
    391 +   - 所以若 circuit 长度会变，`per_window` 下同一个 `g` 不再对应同样的“整条 circuit 总噪声预算”；这时 `per_circuit` 更合理
    392 +12. 因此，当前最严谨的结论是：
    393 +   - `f` 的定义依赖 `GateDurations.idle_ns`
    394 +   - `g` 的定义依赖你选 `per_window` 还是 `per_circuit`
    395 +   - 对跨不同电路长度的比较，应优先使用 `per_circuit`
    396 +   - 对跨不同时间假设的比较，必须固定 `GateDurations`，否则 `(f,g)` 不可直接横比
    397 +13. 如果你问“怎样才算真正物理上更稳妥”：
    398 +   - 最低要求：把 `GateDurations` 当成模型的一部分写进实验 protocol，不能默认隐含
    399 +   - 更进一步：把 `f` 不只写 Hz，还换算成“每 idle window 的 `gamma`”或“相关窗口数 `tau_corr / idle_ns`”
    400 +   - 最推荐的报告方式是同时记：
    401 +     - `f_hz`
    402 +     - `idle_ns`
    403 +     - `gamma_per_window`
    404 +     - `corr_windows_per_step`
    405 +     - `g_mode`
    406 +14. 实务建议：
    407 +   - 如果你当前目标是做 repo 内部自洽比较，现有定义已经够用，但必须固定 `GateDurations`
    408 +   - 如果你要把 `(f,g)` 解释成更接近真实器件的物理参数，就需要先声明 serial timing model 是否合理，以及 `idle_ns` 是否对应你要映射的真实硬件时间
    409 +15. 一句话总结：
    410 +   - 在“纯 layer、无时间”的抽象里，`f,g` 本来不够 well-defined；
    411 +   - 在我们当前 repo 里，它们之所以变得 well-defined，是因为代码已经人为补上了一套 synthetic physical clock；
    412 +   - 所以它们现在是 **model-defined**, 不是 **hardware-universal**。

Q: 我们现在针对 noisy circuit 模拟里各种 error channel，怎么理解每个 channel 的参数物理意义？尤其 correlated noise 里的 `f,g`，如果电路只有 layered 次序、没有真实时间长短，那它们怎么才算 well-defined？[This is crucial question! 🔥🔥🔥🔥🔥🔥]

A:
1. 先说总原则：在当前代码里，`f,g` 不是“悬空”的自由符号；它们只有在你先接受一套**时间约定**之后才有物理意义。
2. 这个 repo 已经明确写死了一套时间约定，在 `code/quantum_simulation/noise_engine.py`：
   - 串行调度（不是并行 layer）
   - `1q gate = 10 ns`
   - `2q gate = 20 ns`
   - `measurement = 100 ns`
   - `reset = 100 ns`
   - 相邻指令之间固定插入 `idle_ns = 200 ns`
3. 因此，当前 noisy-circuit 其实不是“只有顺序、完全没时间”，而是：
   - **文本顺序 -> serial timeline**
   - **timeline -> 带 ns 单位的 synthetic clock**
   - 各种 time-dependent / correlated channel 都是相对于这个 synthetic clock 定义的
4. 各 channel 及参数语义：
   - `auto`
     - 不是独立物理模型，只是兼容层。
     - `control_mode=global` 时映射到 `google_global`；`gate_specific` 时映射到 `google_gate_specific`。
   - `google_global`
     - 物理含义：所有门共享一个全局控制失配标量，失配越大，门 depolarizing rate 越高。
     - 关键参数：
       - `p_1q_base`, `p_2q_base`: 1q/2q 的 action 无关底噪
       - `sensitivity_1q`, `sensitivity_2q`: 控制失配转成门错误率的斜率
       - `p_clip_max`: 概率截断上限
       - `drift_period_steps`, `drift_amplitude`: 最优控制 `u_opt` 的漂移周期和幅度
     - 设定建议：
       - `p_base` 先来自你想模拟的器件 baseline gate infidelity
       - `sensitivity` 用来控制“错控一点会带来多大额外 gate noise”
       - `drift_period_steps` 决定非平稳性快慢；大表示慢漂移
   - `google_gate_specific` (🙏🙏🙏)
     - 物理含义：不是所有门共用一个失配，而是每类/每位置门映射到某个 control slot，各自有局部失配。
     - 额外参数：
       - `n_1q_control_slots`, `n_2q_control_slots`: 1q/2q gate 对应多少个独立控制槽
     - 设定建议：
       - 想表达“不同门族/不同位置误差可分开校准”时用它
       - slot 数越大，自由度越强，但训练更难
   - `idle_depolarizing`
     - 物理含义：只在 idle window 注入 action 无关的 Pauli idle noise。
     - 关键参数：
       - `idle_p_total_per_idle`: 每个 idle window 的总非平凡 Pauli 概率
       - `idle_px_weight`, `idle_py_weight`, `idle_pz_weight`: X/Y/Z 分配权重
     - 设定建议：
       - 如果你只想加静态 idle 背景噪声，这是最直接模型
       - 若没有方向偏置证据，先设成 `(1,1,1)`
   - `parametric_google`
     - 物理含义：结构上和 `google_gate_specific` 一样，但把 gate-noise 整体强度交给显式 regime knob 来扫参数。
     - 关键参数：
       - `channel_regime_a`, `channel_regime_b`
     - 代码语义：
       - `a` 同时缩放 1q 的 `p_base` 和 `sensitivity`
       - `b` 同时缩放 2q 的 `p_base` 和 `sensitivity`
     - 设定建议：
       - 想做“同一类模型，不同器件 regime / 不同门族强弱”扫描时用它
   - `correlated_pauli_noise_channel` (🙏🙏🙏)
     - 物理含义：只在 idle window 注入 temporally correlated Pauli noise；X/Y/Z 各有独立两态 HMM 链，但共享同一个 `(f,g)`。
     - 关键参数：
       - `channel_corr_f`: 相关频率，控制 hidden state 在相邻 idle window 之间切换多快
       - `channel_corr_g`: 强度尺度，控制平均 idle Pauli 强度
       - `channel_corr_g_mode`: `per_window` 或 `per_circuit`
       - `p_1q_base`, `sensitivity_1q`, `p_clip_max`: 用来先生成目标总强度，再乘上 `g`
     - 设定建议：
       - `f` 决定 memory time；小 `f` = 慢变化、长相关
       - `g` 决定总体噪声强度，不直接决定相关时间
       - 若比较不同 circuit 长度，优先用 `per_circuit`
   - `composed_google_global_correlated` / `composed_google_gate_specific_correlated` (🙏🙏🙏)
     - 物理含义：门上有 Google-like depolarizing，idle 上有 correlated Pauli，两者一次性组合。
     - 参数就是前两类的并集。
     - 设定建议：
       - 如果你认为器件同时有“门错控 + correlated idle background”，这是最合理的综合模型。
5. `f` 在当前实现里的严格含义：
   - 它不是直接对“整条 circuit 总时间”起作用，而是先转成**相邻 idle window 的状态翻转概率**。(🙏🙏🙏)
   - 代码里：
     - `idle_s = idle_ns * 1e-9`
     - `rho = exp(-f * idle_s)`
     - `gamma = (1 - rho) / 2`
   - 然后 Hidden-Markov telegraph 链每经过一个 idle window，就按 `gamma` 更新一次状态。
6. 所以在当前模型下，`f` 的真实解释是：
   - 给定固定 `idle_ns` 后，`f` 决定“每个 idle window 之间相关性衰减多快”
   - 或者等价地，决定以 ns 计的相关时间常数大约是多少
   - 但因为代码只在 idle window 更新状态，所以它最终体现为“每窗口一步”的 Markov 记忆强度
7. `g` 在当前实现里的严格含义：
   - 先算一个由 control mismatch 给出的 baseline strength：
     - `base_strength = p_1q_base + sensitivity_1q * mismatch_global`
   - 再乘 `g` 得到目标总 Pauli 强度：
     - `p_total_target = clip(base_strength * g, 0, p_clip_max)`
   - 所以 `g` 是**强度缩放因子**，不是时间参数。(🙏🙏🙏) 
8. 你提到的关键问题是对的：
   - 同一个物理过程，如果实验总时长 `T` 不同，那么“低频/高频”“强/弱”这些词本来都依赖时间尺度。
   - 如果模拟里完全没有时间标尺，只剩 layer index，那么 `f` 用 Hz 表示其实是不严谨的。
9. 但在我们当前代码里，这个问题被下面这一步“封闭”了：
   - repo 明确规定了 `GateDurations` 和 `idle_ns`
   - 所以 `f` 的 Hz 单位是相对于这套 synthetic time 定义的
   - 也就是说，当前问题是 well-defined 的，但**只在这套时间约定下 well-defined**
10. 更直白地说：
   - 现在的 `f=1e4 Hz` 并不是“真实实验平台上绝对的 10 kHz”
   - 它是“在本 repo 规定的 serial+idle=200ns 时间刻度下，对应的 10 kHz telegraph correlation frequency”
   - 如果你改了 `idle_ns` 或整体 scheduling 假设，同一个 `f` 的统计意义会变
11. `g` 的 well-defined 也依赖你选择的归一化口径：
   - `per_window`: `g` 表示每个 idle window 的局部强度缩放
   - `per_circuit`: 先把整条 circuit / 一次 simulator step 的总预算固定，再反推每窗口强度
   - 所以若 circuit 长度会变，`per_window` 下同一个 `g` 不再对应同样的“整条 circuit 总噪声预算”；这时 `per_circuit` 更合理
12. 因此，当前最严谨的结论是：
   - `f` 的定义依赖 `GateDurations.idle_ns`
   - `g` 的定义依赖你选 `per_window` 还是 `per_circuit`
   - 对跨不同电路长度的比较，应优先使用 `per_circuit`
   - 对跨不同时间假设的比较，必须固定 `GateDurations`，否则 `(f,g)` 不可直接横比
13. 如果你问“怎样才算真正物理上更稳妥”：
   - 最低要求：把 `GateDurations` 当成模型的一部分写进实验 protocol，不能默认隐含
   - 更进一步：把 `f` 不只写 Hz，还换算成“每 idle window 的 `gamma`”或“相关窗口数 `tau_corr / idle_ns`”
   - 最推荐的报告方式是同时记：
     - `f_hz`
     - `idle_ns`
     - `gamma_per_window`
     - `corr_windows_per_step`
     - `g_mode`
14. 实务建议：
   - 如果你当前目标是做 repo 内部自洽比较，现有定义已经够用，但必须固定 `GateDurations`
   - 如果你要把 `(f,g)` 解释成更接近真实器件的物理参数，就需要先声明 serial timing model 是否合理，以及 `idle_ns` 是否对应你要映射的真实硬件时间
15. 一句话总结：
   - 在“纯 layer、无时间”的抽象里，`f,g` 本来不够 well-defined；
   - 在我们当前 repo 里，它们之所以变得 well-defined，是因为代码已经人为补上了一套 synthetic physical clock；
   - 所以它们现在是 **model-defined**, 不是 **hardware-universal**。

Q: 我们的 simulation circuit 结束后，有没有什么机制统计 circuit 的总时间（把所有 gate、measurement、idle 都算进去）？
A:
1. 有底层机制，但目前没有看到一个单独对外暴露的“总时长统计 API”。
2. 当前代码里，circuit 的时间定义来自 `code/quantum_simulation/noise_engine.py` 的 `TimelineBuilder`：
   - 它会把 circuit 展开成一串 `OperationEvent`
   - 每个 event 都带 `start_ns` / `end_ns`
   - 各类操作时长由 `GateDurations` 给定：
     - `1q = 10 ns`
     - `2q = 20 ns`
     - `measure = 100 ns`
     - `reset = 100 ns`
     - `idle = 200 ns`
3. 在这个模型下，总时长其实是可直接定义的：
   - `total_time_ns = events[-1].end_ns`（若 `events` 非空）
   - `active_time_ns = sum(duration_for(inst))`
   - `idle_time_ns = (len(events)-1) * idle_ns`
   - 并且 `total_time_ns = active_time_ns + idle_time_ns`
4. 要特别注意一个实现细节：
   - `build_events()` 内部在处理每条指令后都会把时间推进一个 `idle_ns`
   - 但根据文件注释和噪声注入逻辑，**最后一条指令后不算 idle**
   - 所以真正代表电路结束时间的应当取最后一个 event 的 `end_ns`，而不是内部累计变量的最终值
5. 这套 timeline 机制已经被实际使用，而不是停留在注释里：
   - `code/quantum_simulation/steane_code_simulator.py`
   - `code/rl_train/steane_adapter.py`
   都会调用 `build_events(...)` 来推导噪声窗口或相关 channel 的归一化窗口数
6. 所以准确说法是：
   - 现在“能算总时间”，而且定义已经清楚；
   - 只是还没有封装成类似 `compute_circuit_total_time_ns(circuit)` 这样的统一 helper / report 字段。

Q: 加入 measurement error（pre-measure bit-flip）后，会不会让 composite RL simulation 明显变慢？
A:
1. 做过一次小而有代表性的 3-run timing 对比，配置选的是一个实际跑过的 composite benchmark family，但规模控制在 10 分钟以内：
   - `total_timesteps=512`
   - `rollout_steps=32`
   - `steane_n_rounds=4`
   - `steane_shots_per_step=4`
   - `post_eval_episodes=8`
   - `eval_steane_shots_per_step=24`
   - `steane_noise_channel=composed_google_gate_specific_correlated`
   - `steane_channel_corr_f=1e4`
   - `steane_channel_corr_g=0.4`
   - `steane_channel_corr_g_mode=per_circuit`
2. 比较对象：
   - baseline composite: `google + correlated`
   - full composite: `google + correlated + measurement bit-flip`
   - measurement 参数：`steane_measurement_bitflip_prob=0.01`
3. 3 次 wall-clock 结果：
   - baseline composite:
     - `19.61s`
     - `20.78s`
     - `21.22s`
   - full composite:
     - `23.07s`
     - `22.68s`
     - `23.18s`
4. 汇总：
   - baseline mean/std: `20.54s +- 0.84s`
   - full composite mean/std: `22.98s +- 0.26s`
   - 平均额外开销：`+2.44s`
   - 相对增幅：约 `+11.9%`
   - 运行时间比：约 `1.12x`
5. 结论：
   - measurement error 会带来**可见但不大的**额外 runtime；
   - 当前实现下，它不是 correlated channel 那种大 bottleneck；
   - 对这个代表性 composite 配置来说，可以把它理解成“大约 10% 出头”的额外成本，而不是数量级变慢。

Q: 目前 Steane simulation 里，6 个 stabilizers 和 `n_rounds / n_steps / RL step` 的关系是什么？`n_rounds` 现在一般怎么设定？`data_generated` 里其他实验用了多少 `n_rounds`？
A:
1. 先把三个量分开：
   - `n_stab = 6`
   - `n_steps = 6 * n_rounds`
   - 但 `RL step` 不一定等于一个 stabilizer step
2. 对一次 memory simulation 而言，`n_steps = 6 * n_rounds` 是对的。
   - 代码和 README 都明确把 Steane 映射写成：
     - `n_stab = 6`
     - `n_steps = 6 * n_rounds`
   - simulator 里也是按 `step % 6` 轮转这 6 个 stabilizer
3. 真正容易混淆的是 RL 这一层，因为 adapter 支持两种 stepping mode：（🙏🙏🙏🙏）
   - `candidate_eval`：
     - 每个 RL step 跑一个完整的 `n_rounds` memory experiment
     - 所以此时 `1 RL step = 6 * n_rounds` 个 stabilizer steps
   - `online_rounds`：
     - 每个 RL step 只跑 `1 round = 6` 个 stabilizer steps
     - 整个 episode 一共跑 `n_rounds` 个 RL steps
4. 代码实现上就是：
   - `candidate_eval -> n_rounds_eval = cfg.n_rounds`
   - `online_rounds -> n_rounds_eval = 1`
   - 然后统一用 `n_steps = 6 * n_rounds_eval`
5. 所以最准确的说法是：
   - “因为有 6 个 stabilizers，所以 `n_steps = 6 * n_rounds`”是对的；
   - 但“`n_rounds * 6 = RL n_steps`”只有在你先说明是在一次 simulator memory experiment 内部，或者当前用的是 `candidate_eval` 时才严格准确
6. 当前 RL / benchmark 实际主流口径基本是 `candidate_eval`，因此现实里通常就是：
   - `1 RL step = 1 次完整 memory experiment = 6 * n_rounds` 个 stabilizer steps
7. `n_rounds` 的默认/预设来源：
   - `SteaneAdapterConfig` 默认：`n_rounds = 10`
   - `train.py` 的 paper-style preset 会改成：`steane_n_rounds = 25`
   - 但最近大多数实际 benchmark / staged run 并没有用 25，而是被 stage specs 覆盖
8. `data_generated` 里现有实验的主流取值很集中：
   - `n_rounds = 4`
     - 主要用于 quick / calibration / power / trace-finetune 这类较轻量阶段
   - `n_rounds = 6`
     - 是目前大多数 composite / pilot / confirm / scale / architecture-tuning runs 的主流设定
   - `n_rounds = 1`
     - 只在少数 parametric example spec 里出现，属于最小示例，不是主实验口径
9. 更具体地说：
   - quick / calibration 常见 `n_rounds = 4`
   - 更正式的 composite / confirm / scale 常见 `n_rounds = 6`
   - 所以如果现在问“近期 `data_generated` 的主流 `n_rounds` 是多少”，答案应是：
     - 轻量扫描常用 `4`
     - 主力实验常用 `6`
10. 这一点背后的更重要问题不是乘 6 本身，而是：
   - `n_rounds` 到底被当成 memory depth / noise accumulation depth，
   - 还是仅仅被当成一个计算预算旋钮。
   后面做新实验时，这个解释口径必须先固定。
