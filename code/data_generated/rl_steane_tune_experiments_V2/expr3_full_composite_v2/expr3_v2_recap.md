# Expr3 V2 Recap

`Expr3 V2` tests whether the RL advantage found in `Expr2 V2` survives after
adding measurement bit-flip noise on top of the confirmed composite anchors.

## Design

- inherited anchors from confirmed `Expr2 V2`:
  - `scale=0.025, f=1e4, g=1.0`
  - `scale=0.025, f=1e3, g=1.6`
  - `scale=0.02, f=1e2, g=0.4`
- introduced `p_meas in {1e-3, 3e-3, 1e-2}`
- fully inherited the stronger training recipe from `Expr1/Expr2 V2`

## Main Outcome

`Expr3 V2` confirms that the RL-positive margin can survive after adding
measurement noise, but the margin shrinks as `p_meas` grows.

Confirmed ranking from `Phase C`:

1. `scale=0.025, f=1e4, g=1.0, p_meas=3e-3`
   - `improve(LER~) = +0.2703 +- 0.1003`
2. `scale=0.025, f=1e3, g=1.6, p_meas=1e-2`
   - `improve(LER~) = +0.1687 +- 0.0992`
3. `scale=0.025, f=1e4, g=1.0, p_meas=1e-2`
   - `improve(LER~) = +0.1123 +- 0.0967`

## Interpretation

- the best current `Expr3 V2` headline point is:
  - `scale=0.025, f=1e4, g=1.0, p_meas=3e-3`
- the best `Expr2` anchor remains the best `Expr3` anchor after adding
  measurement noise
- increasing `p_meas` from `3e-3` to `1e-2` weakens the RL-positive margin but
  does not eliminate it
- this supports the intended `Expr3` claim:
  - gate-specific RL control remains useful in a fuller composite setting
  - but the robustness window narrows as measurement noise gets stronger

## Readout

The current evidence supports this `Expr3 V2` story:

- `Expr2` gains are not wiped out immediately by realistic added measurement
  noise
- there is a measurable robustness window in `p_meas`
- `p_meas=3e-3` is currently the cleanest headline point
- `p_meas=1e-2` should be treated as a higher-noise boundary where positive
  margin still exists but is clearly reduced
