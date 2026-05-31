"""MechFerret: autonomous mechanistic-interpretability research system."""

__version__ = "0.1.0"

from .controller import MechFerret
from .discovery import DiscoveryController
from .models import Discovery, ExperimentResult, Hypothesis, ResearchRun

__all__ = [
    "MechFerret",
    "DiscoveryController",
    "ResearchRun",
    "Hypothesis",
    "ExperimentResult",
    "Discovery",
]

