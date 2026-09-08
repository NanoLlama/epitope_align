"""Write the runnable synthetic example into ``examples/synthetic/``.

    python examples/make_example.py
    epitope-map --config examples/run.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

from epitope_map import demo as synthetic  # noqa: E402


def main() -> int:
    target = Path(__file__).resolve().parent / "synthetic"
    paths = synthetic.write_inputs(target)
    for name, path in paths.items():
        print(f"{name}: {path.relative_to(ROOT)}")
    print("\nnow run:  epitope-map --config examples/run.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
