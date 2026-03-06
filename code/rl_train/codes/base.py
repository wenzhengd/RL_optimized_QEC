"""Common abstractions for code-family simulator adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..env import ExternalSimulatorEnv
from ..interfaces import RewardFn, SimulatorProtocol


@dataclass(frozen=True)
class CodeComponents:
    """Built simulator/env bundle for one code family."""

    code_family: str
    code_cfg: Any
    simulator: SimulatorProtocol
    env: ExternalSimulatorEnv
    action_mapper: Callable[[np.ndarray], np.ndarray]
    reward_fn: RewardFn
