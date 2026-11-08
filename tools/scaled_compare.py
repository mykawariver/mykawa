"""scaled_compare -- the old generator against experiments/scaled.py.

    python -m tools.scaled_compare                    # seeds 1-3, table only
    python -m tools.scaled_compare --fig docs/images/scaled_vs_old_3x12km_s1.png

For each generator, at 128 x 128 and 3 x 12 km (one outlet on the east
edge), the summary numbers of tools/terrain_metrics, averaged over seeds:

    isolated   knobs standing on their own, per km2
    hier       log-log slope of divide height against the river size it
               separates (bigger = more hierarchy); noisy
    orth       small valleys (0.01-0.3 km2): cosine of their flow against
               the downslope of the ground smoothed at 6 px (1 = straight
               down off the ridges)
    gap99      99th percentile of the land-to-channel distance [px]
    hand       log-log slope of the mean height of the ground above the
               river >= A it drains to, A = 0.03..3 km2 (all cells; stable)
    W1, W10    valley floor width [px] at 1 and >= 10 km2
    sin_s      sinuosity over 32 steps, rivers < 0.1 km2
    sin_L      sinuosity over 32 steps, rivers >= 3 km2
    rough      roughness next to rivers >= 3 km2 / next to rivers < 0.1 km2
    dens       channel density, same ratio
    t          seconds to generate

The old generator is baseline.generate (stage1="flood" on the long map, the
zero at 128); its heights are fractions of the relief, so its `isolated`
uses 5 % of the relief where the new one uses 10 m.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.terrain_metrics import measure

LONG = (256, 1024)
ROOTS_LONG = [(128, 1023, np.pi)]


def run(which, size, seed):
    t = time.time()
    if which == "old":
        from experiments.baseline import generate, edge_roots
        if size == "128":
            u, net, _m = generate(seed)
            rng = np.random.default_rng(seed)
            k = int(round(int(rng.integers(3, 6)) * 1.0))
            roots = edge_roots(rng, 128, 128, max(1, k))
        else:
            u, net, _m = generate(seed, shape=LONG, outlets=ROOTS_LONG,
                                  stage1="flood")
            roots = ROOTS_LONG
        dt = time.time() - t
        return u, net, measure(u, net, roots), dt
    from experiments.scaled import generate_scaled
    if size == "128":
        u, net, m = generate_scaled(seed)
    else:
        u, net, m = generate_scaled(seed, shape=LONG, outlets=ROOTS_LONG)
    dt = time.time() - t
    return u, net, measure(u, net, m["roots"], meters=True), dt


def summary(m):
    def mean_over(d, keys):
        v = [d[k] for k in keys if k in d and np.isfinite(d[k])]
        return float(np.mean(v)) if v else np.nan
    small, large = (0.01, 0.03), (3.0, 10.0, 30.0)
    return dict(
        isolated=m["isolated"],
        hier=m["divide_slope"],
        hand=m["hand_slope"],
        orth=m["orth"],
        gap99=m["gap99"],
        W1=m["floor_w"].get(1.0, np.nan),
        W10=mean_over(m["floor_w"], (10.0, 30.0)),
        sin_s=mean_over(m["sinuosity"][32], small),
        sin_L=mean_over(m["sinuosity"][32], large),
        rough=mean_over(m["rough"], large) / mean_over(m["rough"], small),
        dens=mean_over(m["chan_dens"], large) / mean_over(m["chan_dens"], small))


def figure(out, seed=1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    ls = LightSource(315, 45)
    fig, axs = plt.subplots(2, 1, figsize=(16, 9.4))
    for a, which in zip(axs, ("old", "new")):
        u, net, _m, _dt = run(which, "long", seed)
        z = u - u.min()
        if which == "old":
            z = z / z.max() * 300.0          # show it at a plausible relief
        a.imshow(ls.shade(z, cmap=plt.cm.gist_earth, vert_exag=0.25,
                          blend_mode="soft", vmin=-0.15 * z.max(),
                          vmax=z.max()), interpolation="nearest")
        a.contour(z, levels=np.arange(10, z.max(), 10), colors="k",
                  linewidths=0.3, alpha=0.55)
        a.set_xticks([]); a.set_yticks([])
        a.set_title(("baseline.generate (stage1='flood'), shown at 300 m "
                     "relief" if which == "old" else
                     "experiments/scaled.generate_scaled, in metres "
                     f"(relief {z.max():.0f} m)") +
                    f", 3 km x 12 km, seed {seed}, contours every 10 m",
                    fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    print("saved", out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--fig", default=None)
    a = ap.parse_args(argv)
    keys = ["isolated", "hier", "hand", "orth", "gap99", "W1", "W10", "sin_s", "sin_L", "rough", "dens"]
    print(f"{'':16}" + "".join(f"{k:>9}" for k in keys) + f"{'t':>7}")
    for size in ("128", "long"):
        for which in ("old", "new"):
            rows, ts = [], []
            for sd in a.seeds:
                _u, _n, m, dt = run(which, size, sd)
                rows.append(summary(m)); ts.append(dt)
            name = f"{which} {'128' if size == '128' else '3x12km'}"
            print(f"{name:16}" + "".join(
                f"{np.nanmean([r[k] for r in rows]):9.2f}" for k in keys)
                + f"{np.mean(ts):7.1f}", flush=True)
    if a.fig:
        figure(a.fig)


if __name__ == "__main__":
    main()
