"""Virtual Ward OMOP Generator - Synthetic data generator for virtual ward scenarios."""

__version__ = "0.1.0"
__author__ = "Virtual Ward Team"
__email__ = "team@virtualward.com"

from .config import ConfigurationManager
from .generators import (
    PopulationGenerator,
    EpisodeManager,
    SignalGenerator,
    InterventionEngine,
)
from .validation import DataValidator
from .output import DuckDBWriter

__all__ = [
    "ConfigurationManager",
    "PopulationGenerator", 
    "EpisodeManager",
    "SignalGenerator",
    "InterventionEngine",
    "DataValidator",
    "DuckDBWriter",
]