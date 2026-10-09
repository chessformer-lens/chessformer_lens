import sys
from pathlib import Path

# battery.py and the toy adapter live next to the tests, not in the package
sys.path.insert(0, str(Path(__file__).resolve().parent))
