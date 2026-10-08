"""Consolidated public API for temporal-enhanced spiking experiments."""

from Code.neurons import TeSConfig, TeSNeurons
from Code.tes_events import (
    BackwardResult,
    ContinuousArrival,
    ContinuousForwardResult,
    ContinuousOmegaCheckpoint,
    ContinuousOmegaForwardResult,
    ContinuousThreshold,
    DelGradLoss,
    DiscreteArrival,
    DiscreteForwardResult,
    DiscreteThreshold,
    EffectiveReadout,
    InputSpike,
    OmegaBackwardResult,
    StopInfo,
)

__all__ = (
    "BackwardResult",
    "ContinuousArrival",
    "ContinuousForwardResult",
    "ContinuousOmegaCheckpoint",
    "ContinuousOmegaForwardResult",
    "ContinuousThreshold",
    "DelGradLoss",
    "DiscreteArrival",
    "DiscreteForwardResult",
    "DiscreteThreshold",
    "EffectiveReadout",
    "InputSpike",
    "OmegaBackwardResult",
    "StopInfo",
    "TeSConfig",
    "TeSNeurons",
)
