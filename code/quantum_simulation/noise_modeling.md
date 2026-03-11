# Noise Modeling Notes

This note defines the noise-model terminology used by the current Steane RL stack.

It exists to avoid three common confusions:

- confusing a base `noise_channel` with an overlay parameter,
- confusing a model-defined time scale with hardware-universal physical time,
- confusing "composed" with "all possible noise sources are already included".

## 1. Layers

Current noise modeling has three layers:

1. Base noise channel family
   - selected by `--steane-noise-channel`
   - implemented in [noise_channels.py](./noise_channels.py)
2. Measurement overlay
   - controlled by `--steane-measurement-bitflip-prob`
   - implemented as a wrapper model, not a separate `noise_channel` key
3. Timing model
   - defined by [noise_engine.py](./noise_engine.py)
   - gives meaning to `idle_ns`, correlated `f`, and circuit timing summaries

## 2. Base Noise Channel Families

Supported `noise_channel` keys:

- `auto`
- `google_global`
- `google_gate_specific`
- `idle_depolarizing`
- `parametric_google`
- `correlated_pauli_noise_channel`
- `composed_google_global_correlated`
- `composed_google_gate_specific_correlated`

### `auto`

Compatibility alias only:

- `control_mode=global` -> `google_global`
- `control_mode=gate_specific` -> `google_gate_specific`

### `google_global`

Google-like gate depolarizing model with one shared control mismatch scalar.

Interpretation:

- RL action changes one global control vector
- mismatch to drifting optimum increases 1q and 2q depolarizing probabilities

Main parameters:

- `p_1q_base`, `p_2q_base`
- `sensitivity_1q`, `sensitivity_2q`
- `drift_period_steps`, `drift_amplitude`
- `p_clip_max`

### `google_gate_specific`

Google-like gate depolarizing model with gate-slot-specific control mismatch.

Interpretation:

- 1q and 2q gates map to separate control slots
- each slot has its own mismatch to the drifting optimum

Main parameters:

- all `google_global` parameters
- `n_1q_control_slots`, `n_2q_control_slots`

### `idle_depolarizing`

Action-independent idle Pauli noise.

Interpretation:

- noise acts only on idle windows
- no explicit temporal correlation

Main parameters:

- `idle_p_total_per_idle`
- `idle_px_weight`, `idle_py_weight`, `idle_pz_weight`

### `parametric_google`

Same structural family as `google_gate_specific`, but with explicit regime scalers.

Interpretation:

- use this for parameter sweeps over gate-noise regimes
- not a different physics family from Google-like gate noise

Main parameters:

- all `google_gate_specific` parameters
- `channel_regime_a`, `channel_regime_b`

### `correlated_pauli_noise_channel`

Temporally correlated idle Pauli noise.

Interpretation:

- noise acts on idle windows only
- each Pauli axis uses a two-state hidden-Markov telegraph chain
- qubits are independent
- axes are independent but share the same `(f, g)` controls

Main parameters:

- `channel_corr_f`
- `channel_corr_g`
- `channel_corr_g_mode`
- `p_1q_base`, `sensitivity_1q`, `p_clip_max`

### `composed_google_*_correlated`

One-pass composition of:

- Google-like gate depolarizing noise
- correlated idle Pauli noise

Interpretation:

- gate events and idle windows are handled together in one model
- this is the current "composite channel" terminology used in benchmark discussions

## 3. Measurement Overlay

Measurement error is currently not selected by `--steane-noise-channel`.

Instead it is an overlay:

- parameter: `--steane-measurement-bitflip-prob`
- implementation: pre-measurement `X` flip before each supported `M`-family gate

This means:

- base channel handles gate/idle noise
- measurement overlay adds readout-like error on top

Current scope:

- only the current Steane simulator's actual `M`-family Z-basis measurements
- symmetric bit-flip model
- no asymmetric `p01/p10`
- no spatially correlated readout model
- no temporally correlated readout model

## 4. Composite vs Full Composite

Repository terminology should be read as:

- `composite`
  - `google + correlated`
  - concretely:
    - `composed_google_global_correlated`
    - `composed_google_gate_specific_correlated`
- `full composite`
  - `google + correlated + measurement`
  - concretely:
    - one of the above composed channels
    - plus `steane_measurement_bitflip_prob > 0`

So "full composite" is currently an experiment-level composition, not a separate
`noise_channel` enum entry.

## 5. Timing Model

Correlated noise and circuit timing summaries depend on the synthetic serial
timeline defined in [noise_engine.py](./noise_engine.py).

Current defaults:

- `1q gate = 10 ns`
- `2q gate = 20 ns`
- `measurement = 100 ns`
- `reset = 100 ns`
- `idle = 200 ns`

Important implication:

- correlated `f` is well-defined only relative to this timing model
- if `GateDurations` changes, the meaning of the same numeric `f` changes

## 6. `f` and `g`

### `f`

- input unit: Hz
- role: sets temporal correlation rate of the hidden Markov chain
- effective update happens once per idle window

So in practice:

- smaller `f` -> longer memory
- larger `f` -> faster decorrelation across idle windows

### `g`

- role: scales correlated idle-noise strength
- not a time parameter

Interpretation depends on `channel_corr_g_mode`:

- `per_window`
  - `g` applies at idle-window level
- `per_circuit`
  - `g` is normalized to one whole simulator-step circuit budget
  - preferred when comparing different circuit lengths

## 7. Nominal Circuit Timing

The RL adapter now records nominal circuit timing summaries for one RL step.

These fields are nominal because they assume:

- one successful state-preparation attempt
- no extra stochastic retry cost

They are useful for:

- checking timing-scale assumptions behind correlated noise
- logging model context in benchmark JSON

They are not primary RL-advantage metrics.

## 8. Practical Rules

When adding or analyzing a noise model, keep these rules explicit:

1. State whether it is a base channel or an overlay.
2. State whether it acts on gates, idle windows, measurements, or combinations.
3. State whether its parameters are defined per-window or per-circuit.
4. State whether its time scale depends on the synthetic `GateDurations`.
5. When reporting results, state whether measurement overlay was enabled.

## 9. Current Recommended Language

Recommended phrasing in notes, QA, and analysis:

- "Google-like gate noise"
- "correlated idle Pauli noise"
- "measurement bit-flip overlay"
- "composite channel" = Google + correlated
- "full composite" = Google + correlated + measurement

Avoid saying:

- "measurement is not a noise model"
- "f is hardware-universal"
- "composed already includes all readout effects"
