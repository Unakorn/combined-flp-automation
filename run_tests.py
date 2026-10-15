"""Run synthetic regression tests; no FL Studio, songs or plugins required."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
import unittest
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "work"), str(ROOT / "tests")]


def inherited_acl_mkdtemp(suffix=None, prefix=None, dir=None):
    """Use an inherited Windows ACL when a sandbox rejects Python's 0700 ACL.

    This opt-in workaround only changes fixture-directory creation during this
    test process. Production code and the normal test command stay unchanged.
    """
    folder = Path(dir or tempfile.gettempdir())
    for _ in range(100):
        candidate = folder / ((prefix or "tmp") + uuid.uuid4().hex + (suffix or ""))
        try:
            candidate.mkdir()
            return str(candidate)
        except FileExistsError:
            continue
    raise FileExistsError("Could not allocate a unique test fixture directory.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="test_*.py", help="A unittest discovery filename pattern.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--inherit-temp-acl", action="store_true",
                        help="Windows sandbox only: inherit directory ACLs for temporary test fixtures.")
    args = parser.parse_args()
    if args.inherit_temp_acl:
        if os.name != "nt":
            parser.error("--inherit-temp-acl is only needed on Windows")
        tempfile.mkdtemp = inherited_acl_mkdtemp
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=args.pattern)
    result = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
