"""
Training script for the GL-Fusion model.
This is a wrapper that uses the modular components from training_v2.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import and run the modular main function
from src.training_v2.main import main

if __name__ == "__main__":
    main()