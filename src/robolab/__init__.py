"""robolab: a pedagogical MuJoCo control playground.

The core idea: keep the robot/environment and the controller decoupled, so any
controller you design can be dropped into any robot behind one small interface.
"""

from pathlib import Path

__version__ = "0.1.0"

# Absolute path to the bundled MJCF assets, so code never hard-codes paths.
# __file__ is THIS file; .parent is the robolab/ folder; then /"assets".
ASSETS_DIR = Path(__file__).parent / "assets"

__all__ = ["__version__", "ASSETS_DIR"]