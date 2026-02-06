"""Source package - main training logic and models."""

import sys
import os

# Import Config from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import Config

from .entity import EntityBelief, Creature, Encoder, SimpleGATLayer, init_single_creature
from .simulate import System

__all__ = [
    "Config",
    "EntityBelief",
    "Creature",
    "Encoder",
    "SimpleGATLayer",
    "init_single_creature",
    "System",
    "HelperMath",
]
