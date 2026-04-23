#!/usr/bin/env python3
"""Reproduce the paper's Section 7 benchmark.

Standalone helper that uses the same defaults as ``python -m debcompare``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installation: add SOFTWARE/src and SOFTWARE/ to sys.path.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from debcompare.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
