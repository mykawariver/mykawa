"""real_basin -- real rivers of about 12 km that run to the sea, as the
reference for the long maps of experiments/scaled.py.

    python -m tools.real_basin                 # the numbers of each basin
    python -m tools.real_basin --fig out.png   # ... and a figure beside a
                                               # 3 x 12 km generated map

Three basins in southern Boso (Chiba), found by routing the water of GSI's
10 m mesh over the whole peninsula, keeping the rivers of 8-17 km that reach
the sea, and reading the maps and air photos for golf courses and dams:

    nagao   Nagao-gawa, Shirahama (Minamiboso).  About 13 km, 17 km2, to
            214 m.  Forest nearly everywhere, no dam, no golf course; the
            mouth cuts through marine terraces straight into the sea.  The
            closest to what scaled.py makes.
    soro    Soro-gawa, Futomi (Kamogawa).  About 15 km, 12 km2, to 323 m.
            Long and narrow with the mouth in the east, like the 3 x 12 km
            map.  The north side is the Mineoka landslide belt (smooth), the
            south side finely dissected: two rocks in one basin.
    futama  Futama-gawa, Amatsu (Kamogawa).  About 9 km, 17 km2, to 373 m.
            The steep south face of Kiyosumi-yama; long parallel spurs.
Left out: Matsusaki-gawa (a golf course and Hodai dam), Sakuma-gawa (a dam
and a motorway), Mihara-gawa (Komukai dam).

The 10 m mesh (elevation tile layer "dem", zoom 14) covers the whole country,
where the 5 m layer of tools/fetch_dem.py is corridors only at this size.  It
is made from the contours of the 1:25 000 map, so it is a little SMOOTHER
than the ground: the real slopes are at least what is measured here.

Source: Geospatial Information Authority of Japan (GSI) elevation tiles,
processed.  出典: 国土地理院 標高タイル（基盤地図情報数値標高モデル）を加工して作成。
Tiles are cached under valley_samples/dem_cache/ (not in the repository).
"""
import argparse
import heapq
import math
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.fetch_dem import _fetch_tile, _tile_of
from tools.terrain_metrics import surface_stats, PX_M

Z = 14
# name -> (window centre lat, lon; window km east-west, north-south; outlet)
BASINS = {
    "nagao": (34.93070, 139.90316, 12.0, 7.2, (34.90635, 139.86814)),
    "soro": (35.09126, 140.06195, 12.0, 4.8, (35.08410, 140.09817)),
    "futama": (35.14490, 140.14212, 12.0, 6.6, (35.12089, 140.15310)),
}


def window(lat, lon, w_km, h_km):
    """The 10 m mesh around (lat, lon), resampled to PX_M (11.72 m).
    Returns (z [m], sea mask, (row, col) -> (lat, lon) helpers' origin)."""
    g = 2 * math.pi * 6378137.0 / (2 ** Z * 256) * math.cos(math.radians(lat))
    xc, yc = _tile_of(lat, lon, Z)
    gx0, gx1 = int(xc * 256 - w_km * 500 / g), int(xc * 256 + w_km * 500 / g)
    gy0, gy1 = int(yc * 256 - h_km * 500 / g), int(yc * 256 + h_km * 500 / g)
    tx0, tx1, ty0, ty1 = gx0 // 256, gx1 // 256, gy0 // 256, gy1 // 256
    mos = np.full(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), np.nan)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            mos[(ty - ty0) * 256:(ty - ty0 + 1) * 256,
                (tx - tx0) * 256:(tx - tx0 + 1) * 256] = _fetch_tile(
                    tx, ty, z=Z, layer="dem")
    dem = mos[gy0 - ty0 * 256:gy1 - ty0 * 256, gx0 - tx0 * 256:gx1 - tx0 * 256]
    sea = np.isnan(dem)
    f = g / PX_M
    z = ndi.zoom(ndi.gaussian_filter(np.where(sea, 0.0, dem), 0.5), f, order=1)
    sea = ndi.zoom(sea.astype(float), f, order=1) > 0.5
    return z, sea, (gx0, gy0, f)


def basin_of(z, sea, r0, c0, search=40):
    """The cells that drain to the sea through the outlet nearest (r0, c0):
    a priority flood from the coast gives every cell the coastal cell its
    water reaches; of the coastal cells within `search` px of (r0, c0), the
    one with the largest catchment is the river's mouth."""
    H, W = z.shape
    par = np.full(H * W, -2, np.int64)
    zf = np.where(sea, np.inf, z).ravel()
    seaf = sea.ravel()
    edge = np.zeros((H, W), bool)
    edge[0] = edge[-1] = True
    edge[:, 0] = edge[:, -1] = True
    coast = (ndi.binary_dilation(sea, np.ones((3, 3), bool)) | edge) & ~sea
    heap = [(zf[i], int(i)) for i in np.flatnonzero(coast.ravel())]
    for _zv, i in heap:
        par[i] = -1
    heapq.heapify(heap)
    order = []
    offs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while heap:
        zc, i = heapq.heappop(heap)
        order.append(i)
        r, c = divmod(i, W)
        for dr, dc in offs:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                j = rr * W + cc
                if par[j] == -2 and not seaf[j]:
                    par[j] = i
                    heapq.heappush(heap, (max(zf[j], zc), j))
    out = np.full(H * W, -1, np.int64)
    for i in order:
        out[i] = i if par[i] < 0 else out[par[i]]
    size = np.bincount(out[out >= 0], minlength=H * W)
    mouths = np.flatnonzero(size)
    mr, mc = np.divmod(mouths, W)
    near = np.hypot(mr - r0, mc - c0) <= search
    if not near.any():
        raise ValueError("no river mouth near the given outlet")
    best = mouths[near][np.argmax(size[mouths[near]])]
    return (out == best).reshape(H, W)


def load(name):
    """name -> dict(z, sea, basin, stats) for one of BASINS."""
    lat, lon, w_km, h_km, (olat, olon) = BASINS[name]
    z, sea, (gx0, gy0, f) = window(lat, lon, w_km, h_km)
    ox, oy = _tile_of(olat, olon, Z)
    r0, c0 = (oy * 256 - gy0) * f, (ox * 256 - gx0) * f
    basin = basin_of(z, sea, r0, c0)
    return dict(z=z, sea=sea, basin=basin, stats=surface_stats(z, basin))


def line(name, s, extra=""):
    return (f"{name:14} slope {s['slope50']:.2f}/{s['slope90']:.2f}  relief in "
            f"250/500/1000 m {s['lr250']:.0f}/{s['lr500']:.0f}/{s['lr1000']:.0f}"
            f"  flat {s['flat']:.2f}  fine {s['fine']:.2f}{extra}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default=None)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    real = {}
    for name in BASINS:
        real[name] = d = load(name)
        km2 = d["basin"].sum() * (PX_M / 1000.0) ** 2
        print(line(name, d["stats"], f"  ({km2:.0f} km2, to "
                                     f"{d['z'][d['basin']].max():.0f} m)"))
    from experiments.scaled import generate_scaled
    u, _net, _m = generate_scaled(a.seed, shape=(256, 1024),
                                  outlets=[(128, 1023, np.pi)])
    s = surface_stats(u)
    print(line(f"scaled s{a.seed}", s, f"  knobs {s['knobs']}"))
    if not a.fig:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    ls = LightSource(315, 45)
    items = [(f"experiments/scaled.generate_scaled, seed {a.seed}, 3 km x 12 km",
              u - u.min(), None, None)]
    for name, d in real.items():
        items.append((f"REAL: {name} (GSI 10 m mesh, processed)", d["z"],
                      d["sea"], d["basin"]))
    hs = [z.shape[0] for _t, z, _s, _b in items]
    fig, axs = plt.subplots(len(items), 1, gridspec_kw=dict(height_ratios=hs),
                            figsize=(16, 16 * sum(hs) / 1024 + 0.4 * len(items)))
    for ax, (title, z, sea, bm) in zip(axs, items):
        rgb = ls.shade(z, cmap=plt.cm.gist_earth, vert_exag=0.25,
                       blend_mode="soft", vmin=-37.5, vmax=380)
        if sea is not None:
            rgb[sea] = (0.82, 0.88, 0.95, 1)
            out = ~bm & ~sea
            rgb[out, :3] = 0.55 * rgb[out, :3] + 0.45
        ax.imshow(rgb, interpolation="nearest")
        zz = z if sea is None else np.where(sea, np.nan, z)
        ax.contour(zz, levels=np.arange(10, np.nanmax(zz), 10), colors="k",
                   linewidths=0.3, alpha=0.55)
        if bm is not None:
            ax.contour(bm, levels=[0.5], colors="r", linewidths=0.9)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(0, 1024); ax.set_ylim(z.shape[0], 0)
        ax.set_title(title + ", 12 km wide, contours every 10 m", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.fig, dpi=90)
    print("saved", a.fig)


if __name__ == "__main__":
    main()
