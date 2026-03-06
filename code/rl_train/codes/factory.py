"""Factory for code-family simulator bundles."""

from __future__ import annotations

import argparse
from typing import Tuple

from ..interfaces import RewardFn
from .base import CodeComponents
from .steane import build_steane_components


def available_code_families() -> Tuple[str, ...]:
    """Return currently supported code families."""
    return ("steane",)


def resolve_code_family(args: argparse.Namespace) -> str:
    """Resolve code family from args with backward-compatible default."""
    return str(getattr(args, "code_family", "steane"))


def build_code_components(args: argparse.Namespace, reward_fn: RewardFn) -> CodeComponents:
    """Build simulator/env bundle for the selected code family."""
    code_family = resolve_code_family(args)
    if code_family == "steane":
        return build_steane_components(args, reward_fn=reward_fn)
    raise ValueError(
        f"Unsupported code_family '{code_family}'. "
        f"Supported: {', '.join(available_code_families())}"
    )
