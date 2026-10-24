"""Held-out v2 is frozen: editing it to make a failure go away would turn it into a training set.

If a change is genuinely required (e.g. a reference answer is provably wrong), update the hash below in a
separate, reviewed commit that explains why, and note that results before and after are not comparable.
"""

import hashlib
from pathlib import Path

FROZEN_SHA256 = "7f18bffe6abf51a24f7d366718cc45e44dcab463d212c6b7b05cf085d6b96eef"


def test_holdout_v2_is_unchanged():
    path = Path(__file__).resolve().parent.parent / "eval" / "holdout_v2.yaml"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_SHA256, "eval/holdout_v2.yaml is frozen"
