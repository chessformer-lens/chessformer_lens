import os
import sys
from pathlib import Path

os.environ.setdefault("CHESSFORMER_NO_DOWNLOAD", "1")   # the suite never downloads a net

# battery.py and the toy adapter live next to the tests, not in the package
sys.path.insert(0, str(Path(__file__).resolve().parent))

import toy_adapter  # noqa: E402,F401  registers "toy" / "toy-causal"
