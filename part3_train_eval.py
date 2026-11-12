"""Part 3 in one command: build the data, score base nomic, fine tune, score again, report.

    python part3_train_eval.py --query_field abstract

See README.md (Part 3) and the Part 3 section of CLAUDE.md.
"""

import os
import sys
from pathlib import Path

# MPS falls back to the CPU for operations it lacks. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from patent_search.training import main  # noqa: E402

if __name__ == "__main__":
    main()
