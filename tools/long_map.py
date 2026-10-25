"""long_map -- a 3 km x 12 km map (256 x 1024 px) with one trunk.

One outlet in the middle of the east edge, upstream heading west, so the
trunk runs the length of the map; the figure is the stage 1 network (line
width ~ sqrt(drainage)) over the stage 3 terrain after stage 2 = poisson and
after stage 2 = two_bones.

    python -m tools.long_map --stage1 flood      # ~2 min (stage 3 dominates)
    python -m tools.long_map --stage1 headward   # ~8 min: stage 1 alone ~170 s

October 2026, seed 1:
                 channel   median dist   largest gap   heads
    headward     6.22 %    4.12 px       15 px         694
    flood        4.78 %    5.10 px       30 px         308
gen_flood is thinner than gen_headward on this map (at n=128 they match):
its valley-floor passes act harder along a 12 km trunk.  Open question.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.baseline import generate
from tools.network_metrics import channel_acc

H, W = 256, 1024
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]


def run(stage1, seed, out):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "DejaVu Sans"})
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    roots = [(H // 2, W - 1, np.pi)]
    maps = {}
    for method in ("poisson", "two_bones"):
        u, net, meta = generate(seed, shape=(H, W), outlets=roots,
                                stage1=stage1, method=method)
        maps[method] = u
        print(method, "channel %.2f%%" % (100 * net.mean()), flush=True)
    rr, cc, _idx, _G, acc = channel_acc(net, roots)
    fig, axs = plt.subplots(3, 1, figsize=(16, 13.2))
    a = axs[0]
    a.set_facecolor("#fcfcfb")
    w = 0.9 + 2.0 * np.sqrt(acc / acc.max())
    a.scatter(cc, rr, s=w ** 2, c="#2a78d6", linewidths=0, marker="s")
    a.set_xlim(-0.5, W - 0.5); a.set_ylim(H - 0.5, -0.5)
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    a.plot(roots[0][1], roots[0][0], "o", mfc="none", mec="k", ms=12, mew=2,
           clip_on=False)
    a.set_title(f"stage 1: gen_{stage1}, 3 km x 12 km ({H} x {W} px), seed "
                f"{seed}, one outlet on the east edge (circle); line width ~ "
                "sqrt(drainage)")
    for a, m in zip(axs[1:], ("poisson", "two_bones")):
        u = maps[m]
        z = (u - u.min()) / (u.max() - u.min())
        a.imshow(z, cmap=seq, interpolation="nearest")
        cs = a.contour(z, levels=np.linspace(0.025, 0.975, 20),
                       colors="#fcfcfb", linewidths=0.6)
        cs.set_path_effects([pe.Stroke(linewidth=1.3, foreground="k",
                                       alpha=0.4), pe.Normal()])
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"stage 3 after stage 2 = {m}")
    fig.tight_layout()
    fig.savefig(out, dpi=80, facecolor="white")
    print("saved", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage1", default="flood", choices=["headward", "flood"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None,
                    help="output PNG (default: <stage1>_3x12km_s<seed>.png)")
    a = ap.parse_args(argv)
    run(a.stage1, a.seed, a.out or f"{a.stage1}_3x12km_s{a.seed}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
