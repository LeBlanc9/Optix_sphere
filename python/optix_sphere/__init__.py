"""
OptiX Sphere - Monte Carlo simulation for integrating spheres

This package provides Python bindings for OptiX-based Monte Carlo
simulation of integrating spheres with high accuracy and performance.
"""

from ._core import (
    Sphere,
    LightSource,
    Detector,
    SimConfig,
    SimulationResult,
    TheoryResult,
    Simulator,
    configure_detector_chord,
    calculate_theory,
    __version__,
)

__all__ = [
    "Sphere",
    "LightSource",
    "Detector",
    "SimConfig",
    "SimulationResult",
    "TheoryResult",
    "Simulator",
    "configure_detector_chord",
    "calculate_theory",
    "__version__",
]
