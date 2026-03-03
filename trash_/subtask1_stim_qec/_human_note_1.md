# Codex 执行规范：`subtask1_stim_qec`

## 1. 任务目标
- 构建一个通用的 Stim-QEC 仿真流程。
- 输入：`code + rounds + time-dependent Pauli schedule`。
- 输出：按时间轮次组织的 syndrome/detector history。
- 设计上不局限于 7-1-3 码，需支持后续扩展到多种小规模 QECC。

当前明确纳入的构建目标：
- `[[4,2,2]]`（资源受限下的 proof-of-principle 检错演示）
- `[[9,1,3]]`（rotated surface code, d=3）

---

## 2. 本任务边界
- 只处理 **Pauli 随时间变化噪声**（每轮 `p_x, p_y, p_z`）。
- 只处理 **Stim 可表达的 stabilizer/Clifford 范围**。
- 不在本任务中处理非-Clifford精确演化。

---

## 3. 实现范围（仅改以下文件）
- `subtask1_stim_qec/simulate_stim.py`
- `subtask1_stim_qec/summarize_measurements.py`

不得修改其他 subtask。

---

## 4. 输入接口（`simulate_stim.py`）

### 必选参数
- `--code`
- `--rounds`（`>=1`）
- `--shots`（`>=1`）
- `--seed`
- `--schedule_file`（JSON 文件）

### 可选参数
- `--distance`（默认 `3`，用于 surface code 等）
- `--out`
- `--save_measurements`
- `--strict`

---

## 5. schedule 文件规范（JSON）

格式示例：
```json
{
  "version": 1,
  "schedule_type": "per_round",
  "rounds": 5,
  "default": {"p_x": 0.001, "p_y": 0.0, "p_z": 0.001},
  "overrides": {
    "2": {"p_x": 0.005, "p_y": 0.0, "p_z": 0.002}
  }
}
```

约束：
- `p_x, p_y, p_z >= 0`
- `p_x + p_y + p_z <= 1`
- `overrides` 的 key 为轮次索引（字符串形式的非负整数）

---

## 6. 输出接口（`.npz`）

### 必须字段
- `detector_history`：shape `(shots, rounds, n_detectors_per_round)`，二值（0/1）
- `time_round`：shape `(rounds,)`，取值 `0..rounds-1`
- `meta_json`：JSON 字符串

### 可选字段
- `measurement_history`（仅当 `--save_measurements`）

### `meta_json` 至少包含
- `code`, `rounds`, `shots`, `seed`
- `schedule_file`, `schedule_digest`
- `n_detectors_total`, `n_detectors_per_round`
- `backend_mode`（例如 `stim` 或 `scaffold`）
- `stim_version`（可用时）

---

## 7. history 粒度约定
- 默认使用 **detector-event history** 作为主输出。
- 原因：结构紧凑、便于解码统计、便于后续 RL 使用。

---

## 8. 验证标准
- shape 正确：
  - `detector_history.shape[0] == shots`
  - `detector_history.shape[1] == rounds`
- 二值正确：仅含 `0/1`
- 可复现：同参数同 seed 输出一致
- schedule 非法时报错清晰
- 小规模 smoke case 可快速运行

---

## 9. 推荐推进顺序（避免复杂度失控）
1. Phase A：先做 `[[4,2,2]]`（最小 PoP）
2. Phase B：做 `[[9,1,3]]`（small_surface, d=3）
3. Phase C：再考虑 `steane7`、`shor9`、`gauge_color_15`

注意：
- 一次只推进一个 code backend。
- 输出字段名和 shape 不要改，保持下游兼容。

---

## 10. `summarize_measurements.py` 目标
- 读取 `.npz`
- 输出基础统计：
  - `per_round_trigger_rate`
  - `global_trigger_rate`
  - shape 和二值检查
- 支持 `--out_json` 和 `--save_plot`

---

## 11. 代码风格要求
- 注释清晰，面向学习者。
- 函数边界清晰，避免过度抽象。
- 保持 CLI 稳定，不随意改字段名。
