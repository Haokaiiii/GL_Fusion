"""
Core training components.
"""

from .config import ConfigManager, parse_args
from .trainer import Trainer

__all__ = ['ConfigManager', 'parse_args', 'Trainer'] 