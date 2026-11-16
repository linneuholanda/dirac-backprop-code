"""Robustness check: rerun every experiment with other random seeds (new data, initializations and test
points) and count the failed checks. Figures and result files are not written."""
from __future__ import annotations

import sys

from . import common, dirac_backprop

MODULES = {"dirac_backprop": dirac_backprop}
OFFSETS = (100, 200, 300, 400)


def main() -> None:
    common.WRITE = False
    only = sys.argv[1:] or list(MODULES)
    summary = {}
    for name in only:
        mod = MODULES[name]
        default = mod.SEED
        for off in OFFSETS:
            mod.SEED = default + off
            common.FAILURES.clear()
            print(f"\n===== {name}, seed {mod.SEED} =====")
            mod.main()
            summary[(name, mod.SEED)] = list(common.FAILURES)
        mod.SEED = default
    print("\n===== summary =====")
    for (name, seed), fails in summary.items():
        print(f"{name:15s} seed {seed:4d}: " + ("all checks passed" if not fails else f"{len(fails)} failed"))
        for f in fails:
            print(f"    {f}")


if __name__ == "__main__":
    main()
