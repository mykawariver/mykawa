"""knob_sweep -- how much each knob moves the terrain, seen side by side.

Every knob is set to each level in turn (the others stay at 0) and the
finished terrain is drawn as contours, one row per knob and one column per
level.  With the default 6 knobs x 5 levels that is 30 maps of 128 x 128
(1.5 km square).

    python -m tools.knob_sweep                       # all 6 knobs, -2..+2, seed 1
    python -m tools.knob_sweep --stage1 flood        # stage 1 by gen_flood
    python -m tools.knob_sweep --seed 3
    python -m tools.knob_sweep --knobs relief smooth --levels -1 0 1
    python -m tools.knob_sweep --method two_bones    # stage 2 by two bones

The contours are drawn at a FIXED interval (the same 18-level spacing the
zero uses), not renormalised per map.  `relief` only exists as an absolute
height, so renormalised contours would make it invisible: here it shows as
the number of lines.

Besides the figure, one line per map is printed:
    span      the height range of the map, relative to the zero (1.00 = zero)
    outlets   how many rivers leave the map
    channel   channel length per area [km/km2]

`density`, `floor` and `roots` redraw the river network, so their maps are
not the same landscape edited but a different landscape -- that is expected.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.baseline import generate, RELIEF

KNOBS = ["concavity", "relief", "floor", "density", "smooth", "roots"]
LEVELS = [-2, -1, 0, 1, 2]
PX_M = 1500.0 / 128.0               # 1 px = 11.7 m
DZ = (0.98 - 0.02) / 17.0           # the zero's contour interval
CONTOURS = np.arange(0.02, 1.60, DZ)


def run(knobs, levels, seed, n, method, stage1, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nr, nc = len(knobs), len(levels)
    f, ax = plt.subplots(nr, nc, figsize=(2.5 * nc, 2.62 * nr), squeeze=False)
    area_km2 = (n * PX_M / 1000.0) ** 2
    print(f"{'knob':10s} {'level':>5s} {'span':>6s} {'outlets':>7s} "
          f"{'channel':>8s}")
    for i, k in enumerate(knobs):
        for j, v in enumerate(levels):
            u, net, meta = generate(seed, n=n, method=method, stage1=stage1,
                                    **{k: v})
            a = ax[i][j]
            a.contour(u, levels=CONTOURS, colors="#4a3a20", linewidths=0.6)
            a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
            a.set_ylim(u.shape[0] - 1, 0)        # row 0 at the top
            a.set_title(f"{k} {v:+d}" if v else f"{k} 0 (zero)", fontsize=9,
                        color="#a03000" if v == 0 else "#333333")
            if j == 0:
                a.set_ylabel(k, fontsize=11)
            chan = net.sum() * PX_M / 1000.0 / area_km2
            print(f"{k:10s} {v:+5d} {meta['span'] / RELIEF:6.2f} "
                  f"{meta['roots']:7d} {chan:8.2f}", flush=True)
    f.suptitle(f"knob sweep, seed {seed}, {n}x{n} ({n * PX_M / 1000:.1f} km), "
               f"stage 1 = {stage1}, stage 2 = {method}  (fixed contour "
               f"interval: more lines = more relief)", fontsize=12)
    f.tight_layout(rect=[0, 0, 1, 0.975])
    f.savefig(out, dpi=100)
    print("saved", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--knobs", nargs="+", default=KNOBS, choices=KNOBS)
    ap.add_argument("--levels", nargs="+", type=int, default=LEVELS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--method", default="poisson",
                    choices=["poisson", "two_bones"])
    ap.add_argument("--stage1", default="headward",
                    choices=["headward", "flood"])
    ap.add_argument("--out", default=None,
                    help="output PNG (default: knob_sweep_<stage1>_s<seed>.png)")
    a = ap.parse_args(argv)
    out = a.out or f"knob_sweep_{a.stage1}_s{a.seed}.png"
    run(a.knobs, a.levels, a.seed, a.n, a.method, a.stage1, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
