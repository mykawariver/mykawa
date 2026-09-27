"""
fetch_dem -- download a square patch of real terrain from the GSI elevation
tiles (DEM5A) to serve as the real-terrain reference.

Source: 国土地理院 標高タイル (https://maps.gsi.go.jp/development/ichiran.html)
  https://cyberjapandata.gsi.go.jp/xyz/dem5a/{z}/{x}/{y}.txt
  256x256 comma-separated elevations in metres, missing cells are the
  letter 'e'.  Tiles are in Web-Mercator (EPSG:3857) at z=15, i.e. the
  pixel is square in Mercator metres but shrinks east-west by cos(lat)
  on the ground -> we resample onto a metric square grid before use.
出典: 国土地理院 標高タイル（DEM5A）を加工して利用。

Checkpointing: every downloaded tile is cached under valley_samples/dem_cache/
and the final patch is saved as an .npz, so an interrupted run resumes.

Usage:
  python3 fetch_dem.py                 # fetch the registered reference sites
  from fetch_dem import fetch_patch    # fetch_patch(lat, lon, size_m) -> dict
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import math
import os
import time
import urllib.request

import numpy as np
from scipy import ndimage as ndi

TILE_URL = "https://cyberjapandata.gsi.go.jp/xyz/{layer}/{z}/{x}/{y}.txt"
CACHE = "valley_samples/dem_cache"
Z = 15
# above this nodata fraction, nearest-neighbour fill produces flat-facet
# artefacts and the patch must not be used for metrics (see fetch_patch).
VOID_WARN = 0.05

# reference sites: name -> (lat, lon) or (lat, lon, layer).  kameyama/sendai/
# kawauchi are archetypes of the target landform (satoyama: forested hills,
# fluvially dissected); nobeyama (volcanic plateau) and saku (basin) are
# deliberate *contrasts* so that widening the reference separates true
# morphology-independent invariants from metrics that merely describe WHICH
# landscape it is.  Tally after five clean (0% void) sites:
#   * fluvial-organisation STYLE axis (collapses on the non-fluvial nobeyama):
#     theta+r2, coherence, valley_elong, slope_cv, sinuosity.
#   * lowland-shape STYLE axis: low_area, low_width_px (45-70 px hills/upland
#     -> 115 px basin at saku).  kawauchi is the high-HI, narrow-valley end
#     (HI 0.45, low_area 0.10, low_width 45) of the satoyama family.
#   * candidate invariants: beta (roughness) and pit_frac (no interior pits).
#     A flat depositional PLAIN is expected to break both, but no clean
#     (void ~0) natural plain is in the set to test it.
SITES = {
    # 房総丘陵・亀山ダム上流（千葉県君津市）
    "kameyama": (35.220275, 140.133963),
    # 仙台南部の丘陵（宮城）: 谷はそれほど深くない
    "sendai": (38.113861, 140.811296),
    # 野辺山高原（八ヶ岳東麓, 長野）: 標高~1350m・比高
    # わずか80mの構成的高原＋縁の刻み。里山の樹枝状dissectionと別morphology
    # ＝流水指標の帯を割るための対照参照。
    "nobeyama": (35.940256, 138.478675),
    # 佐久盆地（千曲川, 長野）: 標高~700-900m・比高202m。
    # 構造性の盆地＝広い沖積底（low_width_px 115＝里山の倍）＋周囲の急な刻み。
    # 流水組織は残る（theta r²0.74, coh 0.40）が低地形状が別＝low_width を割る。
    "saku": (36.176778, 138.503931),
    # 川内村（阿武隈高地, 福島）: void 0%・標高564-756m・
    # 比高192m。標高700-750mの平頂がそろう隆起準平原を谷が開析＝里山ファミリーの
    # 高地・深開析の端点（HI 0.45 最高, low_area 0.10, 細谷 45px）。尾根が「長い
    # 連続稜線＋少数の頂」の実物。山頂同高性（平頂がそろうこと）は
    # drainage-first では作れない「受け継がれた基準面」の表れ。
    "kawauchi": (37.338875, 140.755935),
}


def site_coords(v):
    """SITES value -> (lat, lon, layer). layer defaults to dem5a."""
    return (v[0], v[1], v[2] if len(v) > 2 else "dem5a")


def _tile_of(lat, lon, z=Z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _fetch_tile(x, y, z=Z, layer="dem5a", retries=3):
    """One 256x256 tile as float array (NaN where missing/absent)."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{layer}_{z}_{x}_{y}.txt")
    if not os.path.exists(path):
        url = TILE_URL.format(layer=layer, z=z, x=x, y=y)
        for k in range(retries):
            try:
                data = urllib.request.urlopen(url, timeout=30).read()
                break
            except urllib.error.HTTPError as err:
                if err.code == 404:          # outside DEM5A coverage
                    data = b""
                    break
                if k == retries - 1:
                    raise
                time.sleep(2 ** k)
            except Exception:
                if k == retries - 1:
                    raise
                time.sleep(2 ** k)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
        time.sleep(0.2)                      # be polite to the tile server
    txt = open(path).read()
    if not txt.strip():
        return np.full((256, 256), np.nan)
    rows = [r.split(",") for r in txt.strip().split("\n")]
    arr = np.array([[np.nan if v == "e" else float(v) for v in r] for r in rows])
    return arr


def fetch_patch(lat, lon, size_m=1500.0, layer="dem5a", zoom=None):
    """Square DEM patch centred on (lat, lon), resampled to a metric grid.

    Returns dict(dem=2D float array [m], px_m=pixel size in metres, ...).
    Cached as an .npz keyed by the arguments.

    zoom: dem5a is the 5 m LiDAR survey and is served at z=15,
    but its coverage is CORRIDOR-ONLY -- a 12 km square comes back 23-38%
    nodata, which the fill then fabricates into flat facets.  The nationwide
    complete layer is the 10 m mesh, served as layer="dem" at z=14.  Pass
    zoom=14 with it.  It is contour-derived and therefore SMOOTHER than
    dem5a, so measure the layer's own bias on a window where both exist
    before comparing a "dem" number against a "dem5a" one."""
    os.makedirs(CACHE, exist_ok=True)
    z = Z if zoom is None else int(zoom)
    key = f"patch_{layer}{'' if z == Z else f'_z{z}'}_{lat:.6f}_{lon:.6f}_{int(size_m)}.npz"
    path = os.path.join(CACHE, key)
    if os.path.exists(path):
        d = np.load(path)
        vf = float(d["void_frac"]) if "void_frac" in d else 0.0
        return dict(dem=d["dem"], px_m=float(d["px_m"]), void_frac=vf)

    # ground size of one z pixel (Mercator pixel * cos(lat) on the ground)
    n = 2 ** z
    merc_px = 2 * math.pi * 6378137.0 / (n * 256)          # ~4.78 m at z=15
    px_ground = merc_px * math.cos(math.radians(lat))      # ground metres/px

    half_px = size_m / 2.0 / px_ground
    xc, yc = _tile_of(lat, lon, z)
    xc_px, yc_px = xc * 256, yc * 256
    x0, x1 = int(xc_px - half_px), int(math.ceil(xc_px + half_px))
    y0, y1 = int(yc_px - half_px), int(math.ceil(yc_px + half_px))

    tx0, tx1 = x0 // 256, x1 // 256
    ty0, ty1 = y0 // 256, y1 // 256
    mosaic = np.full(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256), np.nan)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            t = _fetch_tile(tx, ty, z=z, layer=layer)
            mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256,
                   (tx - tx0) * 256:(tx - tx0 + 1) * 256] = t
    dem = mosaic[y0 - ty0 * 256:y1 - ty0 * 256, x0 - tx0 * 256:x1 - tx0 * 256]

    # fill missing cells (water/no-data) with the nearest valid elevation.
    # IMPORTANT: nearest-neighbour fill over LARGE voids fabricates flat facets
    # and radial creases (Voronoi of the data edge) -- and every metric run on
    # such a patch is meaningless (e.g. Hokkaido plain DEM5A is corridor-only,
    # ~46% void).  So warn loudly above VOID_WARN;
    # the caller must treat a high-void patch as unusable, not fill-and-measure.
    void_frac = float(np.isnan(dem).mean())
    if void_frac > VOID_WARN:
        import sys as _sys
        print(f"WARNING fetch_patch({lat:.4f},{lon:.4f},{layer}): "
              f"{void_frac*100:.0f}% nodata -> nearest-neighbour fill will "
              f"fabricate flat facets; metrics on this patch are unreliable. "
              f"Pick a fully-surveyed point or a coarser complete layer.",
              file=_sys.stderr)
    if 0 < void_frac < 0.5:
        m = np.isnan(dem)
        _, (ir, ic) = ndi.distance_transform_edt(m, return_indices=True)
        dem = dem[ir, ic]

    # north-south pixel equals the Mercator pixel * cos(lat) too (same factor),
    # so the grid is already metrically square; just record the size.
    out = dict(dem=dem, px_m=px_ground, void_frac=void_frac)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, dem=dem, px_m=px_ground, void_frac=void_frac)
    os.replace(tmp, path)
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2 * len(SITES), figsize=(11 * len(SITES), 5))
    axes = np.atleast_1d(axes)

    for i, (name, v) in enumerate(SITES.items()):
        lat, lon, layer = site_coords(v)
        p = fetch_patch(lat, lon, size_m=1500.0, layer=layer)
        dem, px = p["dem"], p["px_m"]
        print(f"{name}: {dem.shape[1]}x{dem.shape[0]} px, {px:.2f} m/px, "
              f"z {np.nanmin(dem):.1f}..{np.nanmax(dem):.1f} m, "
              f"relief {np.nanmax(dem)-np.nanmin(dem):.1f} m, "
              f"void {p.get('void_frac', 0.0)*100:.0f}%")

        ax = axes[2 * i]
        im = ax.imshow(dem, cmap="terrain")
        ax.contour(dem, levels=15, colors="k", linewidths=0.3, alpha=0.6)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{name} DEM5A ({px:.1f} m/px)")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[2 * i + 1]
        f, rad = m["_spectrum"]
        sel = (f >= 2) & (rad > 0)
        ax.loglog(f[sel], rad[sel], "b-", lw=1)
        f0 = np.array([4, int(0.35 * (dem.shape[0] // 2))])
        p0 = rad[f0[0]]
        ax.loglog(f0, p0 * (f0 / f0[0]) ** (-m["beta"]), "r--",
                  label=f"beta={m['beta']:.2f}")
        ax.legend(); ax.set_title(f"{name}: radial PSD")

    fig.suptitle("REAL reference terrain (GSI DEM5A) + terrain metrics\n"
                 "Source: GSI (Geospatial Information Authority of Japan) "
                 "elevation tiles DEM5A, processed", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("valley_samples/real_reference.png", dpi=120)
    print("saved valley_samples/real_reference.png")
