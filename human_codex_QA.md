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
