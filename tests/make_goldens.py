"""Write the golden files the regression tests compare against.

    python tests/make_goldens.py                # every engine whose weights are present
    python tests/make_goldens.py maia3-5m       # just one

Run this only on a commit whose outputs are known good: the goldens are the
definition of "unchanged" for every later refactor. They are local working
data (tests/golden/ is gitignored), so to record them from an older commit
while working on a newer one, check the older one out next to this one and
point both directories at the same place:

    git worktree add ../baseline <known-good commit>
    cp tests/battery.py tests/make_goldens.py ../baseline/tests/
    CHESSFORMER_GOLDEN_DIR=$PWD/tests/golden CHESSFORMER_WEIGHTS=$PWD/weights \
        python ../baseline/tests/make_goldens.py
    git worktree remove ../baseline
"""
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import battery  # noqa: E402


def main(names):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=battery.REPO).stdout.strip()
    except OSError:
        commit = None
    for name in names:
        if name == "bt4" and battery.bt4_weights() is None:
            print("bt4: no weights found, skipped")
            continue
        t = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            eng = battery.make_engine(name, tmp)
            result = battery.run(eng)
        battery.save(name, result, {"torch": torch.__version__, "commit": commit,
                                    "threads": torch.get_num_threads()})
        print(f"{name}: written in {time.time() - t:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or ["maia3-5m", "bt4"])
