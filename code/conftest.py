"""Adds the code/ directory to sys.path so pytest can import the modules from tests/."""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
