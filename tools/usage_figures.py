"""usage_figures -- redraw every figure of docs/USAGE.md.

The four examples of USAGE.md (one outlet, two outlets, 128 x 128, a sea to
the east), each through stage 1, 2 and 3, with the same seeds, outlets and
calls as the text: per example the stage 1 network, the stage 2 and stage 3
elevation maps, and stage 3 minus stage 2.  16 PNGs.

    python -m tools.usage_figures docs/images              # gen_headward
    python -m tools.usage_figures docs/images --stage1 flood

With --stage1 flood the files get a `_flood` suffix (stage1_single_flood.png,
...), so the figures USAGE.md shows are never overwritten.  The grid is one
line per px on the 36 x 36 maps and one per 8 px on 128 x 128.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.baseline import edge_roots
from experiments.flood import gen_flood
from experiments.mini import gen_headward, solve
from experiments.mini_finish import finish_micro

BG = "#fcfcfb"
SEA = "#d9dde3"
BASIN = {-1: SEA, 0: BG, 1: "#2a78d6", 2: "#eb6834", 3: "#2e9e6b",
         4: "#8a5cc2", 5: "#c9a227"}
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]


def examples():
    """(key, seed, size, roots, outlets, label, domain) per USAGE example;
    roots None for example 3 means edge_roots draws them from the rng."""
    n = 36
    coast = np.round(26 + 2 * np.sin(np.arange(n) * 2 * np.pi * 1.3 / n)
                     ).astype(int)
    sea = np.arange(n)[None, :] < coast[:, None]
    return [
        ("single", 3, 36, None, [(35, 18)], "1 outlet", None),
        ("two", 5, 36, [(35, 12, -np.pi / 2), (18, 35, np.pi)],
         [(35, 12), (18, 35)], "2 outlets", None),
        ("128", 1, 128, None, None, "128 x 128", None),
        ("sea", 7, 36, [(r, coast[r] - 1, np.pi) for r in (9, 26)],
         [(r, coast[r] - 1) for r in (9, 26)], "east sea", sea),
    ]


def draw(out, stage1="headward"):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans",
                                "savefig.dpi": 150})
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    seq.set_bad(SEA)
    gen = gen_flood if stage1 == "flood" else gen_headward
    suf = "_flood" if stage1 == "flood" else ""

    def grid(ax, n, step, color, alpha):
        ax.set_xticks(np.arange(-0.5, n, step), minor=True)
        ax.set_yticks(np.arange(-0.5, n, step), minor=True)
        ax.grid(which="minor", color=color, linewidth=0.4, alpha=alpha)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    def rivers(ax, net):
        ry, rx = np.where(net)
        ax.scatter(rx, ry, s=1.5, c="#0b0b0b", alpha=0.55, linewidths=0,
                   zorder=3)

    def plot_net(net, lab, step, title, path):
        rgb = np.zeros(net.shape + (3,))
        for k, h in BASIN.items():
            rgb[lab == k] = np.array([int(h[i:i + 2], 16)
                                      for i in (1, 3, 5)]) / 255.0
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.imshow(rgb, interpolation="nearest")
        grid(ax, net.shape[0], step, "#0b0b0b", 0.15)
        ax.set_title(title)
        fig.savefig(path, facecolor=BG); plt.close(fig)

    def plot_elev(u, net, step, title, path):
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        im = ax.imshow(u, cmap=seq, interpolation="nearest")
        cs = ax.contour(u, levels=np.linspace(0.05, 0.95, 10), colors=BG,
                        linewidths=0.7, alpha=0.9)
        cs.set_path_effects([pe.Stroke(linewidth=1.6, foreground="#0b0b0b",
                                       alpha=0.5), pe.Normal()])
        rivers(ax, net); grid(ax, u.shape[0], step, BG, 0.35)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="elevation (0-1, normalized)")
        fig.savefig(path, facecolor=BG); plt.close(fig)

    def plot_diff(d, net, step, title, path):
        m = np.nanmax(np.abs(d))
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        cm = matplotlib.colormaps["RdBu_r"].copy(); cm.set_bad(SEA)
        im = ax.imshow(d, cmap=cm, vmin=-m, vmax=m, interpolation="nearest")
        rivers(ax, net); grid(ax, d.shape[0], step, "#0b0b0b", 0.15)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="stage 3 minus stage 2 (elevation units)")
        fig.savefig(path, facecolor=BG); plt.close(fig)

    os.makedirs(out, exist_ok=True)
    for key, seed, size, roots, outlets, label, domain in examples():
        step = 8 if size == 128 else 1
        rng = np.random.default_rng(seed)
        if key == "128":
            roots = edge_roots(rng, 128, 128, int(rng.integers(3, 6)))
            outlets = [(r, c) for (r, c, _a) in roots]
        net, lab = gen(rng, n=size, roots=roots, domain=domain, labels=True)
        u = solve(net, outlets=outlets, labels=lab)
        f = finish_micro(u, net, rng, domain=domain)
        plot_net(net, lab, step, f"Stage 1: {gen.__name__} ({label})",
                 f"{out}/stage1_{key}{suf}.png")
        plot_elev(u, net, step, f"Stage 2: solve ({label})",
                  f"{out}/stage2_{key}{suf}.png")
        plot_elev(f, net, step, f"Stage 3: finish_micro ({label})",
                  f"{out}/stage3_{key}{suf}.png")
        plot_diff(f - u, net, step, f"Stage 3 - Stage 2 ({label})",
                  f"{out}/stage3_{key}_diff{suf}.png")
        print("drawn", key, flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="output directory")
    ap.add_argument("--stage1", default="headward",
                    choices=["headward", "flood"])
    a = ap.parse_args(argv)
    draw(a.out, a.stage1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
