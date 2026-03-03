import numpy as np
import stim
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any


@dataclass(frozen=True)
class PauliChannel1:
    """A single-qubit Pauli channel specified by (pX, pY, pZ).

    Meaning:
      With probability pX apply X, with probability pY apply Y, with probability pZ apply Z,
      and with remaining probability 1 - (pX+pY+pZ) do nothing.

    This matches Stim's PAULI_CHANNEL_1(px,py,pz) gate.
    """
    pX: float
    pY: float
    pZ: float

    def validate(self) -> None:
        s = self.pX + self.pY + self.pZ
        if self.pX < 0 or self.pY < 0 or self.pZ < 0:
            raise ValueError(f"Negative Pauli probability: {(self.pX, self.pY, self.pZ)}")
        if s > 1 + 1e-12:
            raise ValueError(f"Invalid Pauli channel: pX+pY+pZ={s} > 1")


class SteaneMemoryStim:
    """
    Teaching-oriented simulator for Steane [[7,1,3]] logical memory in Stim.

    What it does:
      1) Unitary encoding to prepare |+>_L on 7 data qubits.
      2) Repeated syndrome extraction rounds using ONE ancilla qubit (serial reuse).
      3) Time-dependent, qubit-dependent Pauli noise applied after each 'layer' (TICK).

    Key conventions (important to avoid confusion later):
      - Data qubits are q0..q6.
      - Ancilla is q7.
      - Each round produces 6 measurement bits: 3 X-checks then 3 Z-checks.
      - We return:
            m[shot, r, s] = raw stabilizer measurement bit (0/1).
            d[shot, r, s] = detection event bit = m[shot, r, s] XOR m[shot, r-1, s]
                             (with m[shot, -1, s] = 0 by convention).
        If shots=1 we also provide convenient squeezed shapes.

    Extensibility:
      - You can later add logical gates between rounds by appending to the circuit.
      - You can later add decoding by exporting detection events / building a DEM.
      - You can swap encoder / stabilizers by editing the tables below.
    """

    def __init__(
        self,
        noise_per_layer: Optional[np.ndarray],
        # noise_per_layer shape:
        #   [num_layers_total, num_qubits_total(=8), 3] for (pX,pY,pZ)
        #
        # If you don't yet know num_layers_total beforehand, you can pass None
        # and use constant_noise instead.
        constant_noise: Optional[PauliChannel1] = None,
        noise_on_ancilla: bool = True,
    ):
        """
        Args:
          noise_per_layer:
            A numpy array of per-layer per-qubit Pauli channels.
            Each entry is (pX,pY,pZ). Applied after each layer/TICK.
            Shape must be [L, 8, 3] where 8 = 7 data + 1 ancilla.
          constant_noise:
            If provided, uses the same Pauli channel for every qubit at every layer.
            (Useful for a first learner-friendly baseline.)
          noise_on_ancilla:
            Whether to apply noise to the ancilla as well as data qubits.
        """
        self.n_data = 7
        self.anc = 7
        self.n_total = 8
        self.data_qubits = list(range(self.n_data))
        self.all_qubits = list(range(self.n_total))
        self.noise_on_ancilla = noise_on_ancilla

        if constant_noise is None and noise_per_layer is None:
            # Default: a small depolarizing-ish channel (you can tune this).
            constant_noise = PauliChannel1(1e-4 / 3, 1e-4 / 3, 1e-4 / 3)

        if constant_noise is not None:
            constant_noise.validate()

        self.constant_noise = constant_noise
        self.noise_per_layer = noise_per_layer
        if noise_per_layer is not None:
            if noise_per_layer.ndim != 3 or noise_per_layer.shape[1:] != (self.n_total, 3):
                raise ValueError(
                    f"noise_per_layer must have shape [L, {self.n_total}, 3], "
                    f"got {noise_per_layer.shape}"
                )
            # Validate every entry lightly (can be expensive if huge).
            # As a compromise, validate a few random samples + endpoints.
            idxs = {0, noise_per_layer.shape[0] - 1}
            if noise_per_layer.shape[0] > 5:
                idxs |= {noise_per_layer.shape[0] // 2}
            for i in sorted(idxs):
                for q in range(self.n_total):
                    PauliChannel1(*map(float, noise_per_layer[i, q])).validate()

        # --- Steane stabilizers (canonical CSS/Hamming-style supports) ---
        #
        # We store stabilizers as supports (which qubits participate).
        # The Steane code has 3 X-type and 3 Z-type stabilizers, each weight-4.
        #
        # There are multiple equivalent generator sets; this one is a common choice.
        # If you later want a different generator basis, you can change these lists.
        self.x_checks = [
            [0, 1, 2, 3],  # X X X X on these qubits
            [0, 1, 4, 5],
            [0, 2, 4, 6],
        ]
        self.z_checks = [
            [0, 1, 2, 3],  # Z Z Z Z on these qubits
            [0, 1, 4, 5],
            [0, 2, 4, 6],
        ]

        # Measurement ordering per round: X checks first then Z checks.
        self.checks_per_round = len(self.x_checks) + len(self.z_checks)  # = 6

    # -------------------------------------------------------------------------
    # ENCODING CIRCUIT
    # -------------------------------------------------------------------------
    def build_unitary_encoder_plus_L(self) -> stim.Circuit:
        """
        Build a unitary Clifford encoder that prepares |+>_L.

        Teaching note:
          A common trick:
            1) Build an encoder for |0>_L (a CSS-style encoding circuit).
            2) Apply transversal H on all 7 data qubits.
               For the Steane code (self-dual CSS), transversal H maps |0>_L -> |+>_L.

        This function returns ONLY unitary gates (no measurements).
        """
        c = stim.Circuit()

        # Start state: Stim assumes |0> for all qubits.
        # We will build a known Clifford encoder for |0>_L.
        #
        # This encoder pattern is derived from a CSS/Hamming construction.
        # It is not the only encoder, but it is standard and deterministic.

        # Step 1: Put some qubits into superposition.
        # (These act like "information + parity" seeds in the CSS construction.)
        c.append("H", [0, 1, 2])

        # Step 2: CNOT network (a common Steane encoder layout).
        # Intuition:
        #   These CNOTs spread the phase relationships so that the final state
        #   becomes stabilized by the Steane Z-checks (for |0>_L).
        #
        # IMPORTANT:
        #   This is one *fixed* encoder choice. If you later want to verify it,
        #   we can add a test that checks stabilizers using stim.TableauSimulator.

        # From qubit 0
        c.append("CX", [0, 3])
        c.append("CX", [0, 5])
        c.append("CX", [0, 6])

        # From qubit 1
        c.append("CX", [1, 3])
        c.append("CX", [1, 4])
        c.append("CX", [1, 6])

        # From qubit 2
        c.append("CX", [2, 3])
        c.append("CX", [2, 4])
        c.append("CX", [2, 5])

        # Now we have an encoded |0>_L (under this encoder convention).
        # To get |+>_L, apply transversal Hadamard to all data qubits.
        c.append("H", self.data_qubits)

        return c

    # -------------------------------------------------------------------------
    # SYNDROME MEASUREMENT PRIMITIVES
    # -------------------------------------------------------------------------
    def _append_noise_after_layer(self, c: stim.Circuit, layer_index: int) -> None:
        """
        Apply the Pauli noise after a layer.

        Why layer-based noise?
          You told me your noise is a time-dependent discrete signal and you want it
          inserted "between layers". The clean Stim way is:
            - Use TICK markers to define layers
            - After each layer, apply PAULI_CHANNEL_1 to every qubit.

        This function is called after we finish a layer (and before adding TICK).
        """
        qubits = self.data_qubits + ([self.anc] if self.noise_on_ancilla else [])

        if self.noise_per_layer is not None:
            if layer_index >= self.noise_per_layer.shape[0]:
                raise ValueError(
                    f"Not enough noise samples: need at least {layer_index+1} layers "
                    f"but noise_per_layer has {self.noise_per_layer.shape[0]}"
                )
            for q in qubits:
                px, py, pz = map(float, self.noise_per_layer[layer_index, q])
                # Skip if exactly zero to keep circuits smaller.
                if px != 0.0 or py != 0.0 or pz != 0.0:
                    PauliChannel1(px, py, pz).validate()
                    c.append("PAULI_CHANNEL_1", [q], [px, py, pz])
        else:
            # Constant noise baseline.
            ch = self.constant_noise
            assert ch is not None
            for q in qubits:
                if ch.pX != 0.0 or ch.pY != 0.0 or ch.pZ != 0.0:
                    c.append("PAULI_CHANNEL_1", [q], [ch.pX, ch.pY, ch.pZ])

    def _tick(self, c: stim.Circuit, layer_index: int) -> int:
        """
        Finish the current layer: apply noise, then add a TICK.

        Returns:
          next layer_index
        """
        self._append_noise_after_layer(c, layer_index)
        c.append("TICK")
        return layer_index + 1

    def _measure_z_stabilizer_with_ancilla(
        self, c: stim.Circuit, support: List[int], layer_index: int
    ) -> int:
        """
        Measure a Z-type stabilizer Z⊗Z⊗Z⊗Z on the given support using ONE ancilla.

        Standard circuit idea:
          - Ancilla starts in |0>
          - For each data qubit i in support, apply CX(i -> anc)
            This accumulates the parity of data qubits onto the ancilla.
          - Measure ancilla in Z basis (M / MR).
          - Reset ancilla for reuse.

        Stim has MR = measure in Z and reset to |0> in one instruction.

        Returns:
          updated layer_index
        """
        # Ensure ancilla is |0>. Since we use MR every time, it will already be reset,
        # but being explicit helps learners.
        # (If you want, you can omit this R.)
        c.append("R", [self.anc])
        layer_index = self._tick(c, layer_index)

        # Apply CNOTs; we treat each CNOT as its own layer for a clean time model.
        for q in support:
            c.append("CX", [q, self.anc])
            layer_index = self._tick(c, layer_index)

        # Measure & reset ancilla.
        c.append("MR", [self.anc])
        layer_index = self._tick(c, layer_index)

        return layer_index

    def _measure_x_stabilizer_with_ancilla(
        self, c: stim.Circuit, support: List[int], layer_index: int
    ) -> int:
        """
        Measure an X-type stabilizer X⊗X⊗X⊗X on the given support using ONE ancilla.

        Standard circuit idea:
          Measuring X-parity is like measuring Z-parity in a rotated basis.
          A common approach:
            - Prepare ancilla |0>
            - H on ancilla (so ancilla is |+>)
            - For each data qubit i in support, apply CX(anc -> i)
              This couples the ancilla phase to the X-parity of the data.
            - H on ancilla
            - Measure ancilla in Z basis (MR)

        All gates are 1q or 2q, matching your constraint.

        Returns:
          updated layer_index
        """
        c.append("R", [self.anc])
        layer_index = self._tick(c, layer_index)

        c.append("H", [self.anc])
        layer_index = self._tick(c, layer_index)

        for q in support:
            c.append("CX", [self.anc, q])
            layer_index = self._tick(c, layer_index)

        c.append("H", [self.anc])
        layer_index = self._tick(c, layer_index)

        c.append("MR", [self.anc])
        layer_index = self._tick(c, layer_index)

        return layer_index

    # -------------------------------------------------------------------------
    # FULL CIRCUIT BUILDER
    # -------------------------------------------------------------------------
    def build_memory_circuit(
        self,
        num_rounds: int,
        start_with_x_checks: bool = True,
    ) -> Tuple[stim.Circuit, Dict[str, Any]]:
        """
        Build the full circuit: unitary encoding then num_rounds of syndrome extraction.

        Args:
          num_rounds: number of repeated measurement rounds.
          start_with_x_checks: if True each round measures X-checks then Z-checks.
                               if False, Z then X.

        Returns:
          (circuit, metadata)

        metadata contains:
          - 'layers_total': total number of layers/TICKs (for checking noise length)
          - 'checks_per_round': 6
          - 'measurement_order': a list describing (round, check_type, check_index)
        """
        c = stim.Circuit()
        layer = 0

        # 1) Unitary encoding: prepares |+>_L on data qubits.
        enc = self.build_unitary_encoder_plus_L()

        # Teaching detail:
        # Stim circuits don't automatically insert TICKs.
        # Because *you* want noise after each layer, we define each gate as a layer.
        #
        # This is the simplest, most explicit layer model for a learner.
        # Later, we can parallelize commuting gates into the same layer.
        for inst in enc:
            c.append(inst.name, inst.targets_copy(), inst.gate_args_copy())
            layer = self._tick(c, layer)

        measurement_order = []

        # 2) Repeated syndrome extraction rounds.
        for r in range(num_rounds):
            if start_with_x_checks:
                # X checks first
                for j, supp in enumerate(self.x_checks):
                    layer = self._measure_x_stabilizer_with_ancilla(c, supp, layer)
                    measurement_order.append((r, "X", j))
                # Z checks next
                for j, supp in enumerate(self.z_checks):
                    layer = self._measure_z_stabilizer_with_ancilla(c, supp, layer)
                    measurement_order.append((r, "Z", j))
            else:
                # Z checks first
                for j, supp in enumerate(self.z_checks):
                    layer = self._measure_z_stabilizer_with_ancilla(c, supp, layer)
                    measurement_order.append((r, "Z", j))
                for j, supp in enumerate(self.x_checks):
                    layer = self._measure_x_stabilizer_with_ancilla(c, supp, layer)
                    measurement_order.append((r, "X", j))

        meta = {
            "layers_total": layer,
            "checks_per_round": self.checks_per_round,
            "measurement_order": measurement_order,
        }
        return c, meta

    # -------------------------------------------------------------------------
    # SAMPLING AND POST-PROCESSING
    # -------------------------------------------------------------------------
    def sample_syndromes(
        self,
        num_rounds: int,
        shots: int = 1,
        start_with_x_checks: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compile the circuit and sample measurement outcomes.

        Returns dict containing:
          - 'm': raw syndrome bits
          - 'd': detection event bits
          - 'circuit': the stim.Circuit used
          - 'meta': metadata about layer count and measurement ordering

        Shapes:
          m: (shots, num_rounds, 6)
          d: (shots, num_rounds, 6)

        For convenience, if shots == 1, we also provide:
          m_single: (num_rounds, 6)
          d_single: (num_rounds, 6)
        """
        c, meta = self.build_memory_circuit(num_rounds=num_rounds, start_with_x_checks=start_with_x_checks)

        # Compile a sampler. This returns measurement bits for each M/MR in order.
        sampler = c.compile_sampler(seed=seed)
        raw = sampler.sample(shots)  # shape: (shots, total_measurements)

        total_meas_expected = num_rounds * self.checks_per_round
        if raw.shape[1] != total_meas_expected:
            raise RuntimeError(
                f"Unexpected number of measurements: got {raw.shape[1]} "
                f"but expected {total_meas_expected} = {num_rounds}*{self.checks_per_round}"
            )

        # Reshape into rounds x stabilizers.
        m = raw.reshape(shots, num_rounds, self.checks_per_round).astype(np.uint8)

        # Detection events: d[r,s] = m[r,s] XOR m[r-1,s], with m[-1,s]=0.
        d = np.zeros_like(m, dtype=np.uint8)
        d[:, 0, :] = m[:, 0, :]  # since previous is 0 by convention
        if num_rounds > 1:
            d[:, 1:, :] = m[:, 1:, :] ^ m[:, :-1, :]

        out = {
            "m": m,
            "d": d,
            "circuit": c,
            "meta": meta,
        }
        if shots == 1:
            out["m_single"] = m[0]
            out["d_single"] = d[0]
        return out


# -------------------------
# Teaching usage example
# -------------------------
if __name__ == "__main__":
    # Example 1: Constant (small) depolarizing channel on every layer/qubit.
    sim = SteaneMemoryStim(
        noise_per_layer=None,
        constant_noise=PauliChannel1(1e-4/3, 1e-4/3, 1e-4/3),
        noise_on_ancilla=True,
    )
    result = sim.sample_syndromes(num_rounds=5, shots=3, seed=123)
    print("m shape:", result["m"].shape)  # (shots, rounds, 6)
    print("d shape:", result["d"].shape)
    print("One shot syndromes (m_single):\n", result["m_single"])
    print("One shot detections (d_single):\n", result["d_single"])

    # Example 2: Time-dependent per-layer noise.
    # Suppose we have L layers total. We don't know L until we build the circuit.
    # A simple workflow is:
    #   (i) build circuit once with constant_noise to get layers_total,
    #   (ii) create noise_per_layer array of that length,
    #   (iii) rebuild sim with noise_per_layer and resample.