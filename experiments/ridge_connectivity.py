"""
ridge_connectivity -- DEM hydrology helpers, plus a comparison script.

Used by the generator: `priority_flood` (depression filling) and
`d8_accumulation` (D8 flow accumulation).

Run as a script, it holds the solver and the river profile fixed and swaps
only the valley network, to see where ridge structure comes from:

  A  real Kawauchi DEM                                     (ground truth)
  B  REAL channel network @ REAL channel z   -> solve_two_bones, no ridge pin
  B' REAL channel network @ SYNTHETIC z      -> the generator's own geodesic
                                                river profile, no ridge pin
  C  the generator (its own network, own z, own solver)

A perennial-channel network from a real DEM pins only ~3% of the pixels,
which makes any solve trivially smooth; so the real network is taken down to
the zero-order hollows until its pinned band matches the generator's.
Everything is cropped to the interior 60%.

Usage:  python3 -m experiments.ridge_connectivity
        -> valley_samples/ridge_connectivity.png  + a printed table
"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "valley_samples")
SITE = "kawauchi"
CROP = 0.60          # interior window: keep the centre 60% of each side


# ---------------------------------------------------------------------------
# hydrology on the real DEM (priority-flood fill + D8, so the channel network
# is connected by construction: accumulation is monotone downstream)
# ---------------------------------------------------------------------------

_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def priority_flood(z, domain=None):
    """Fill every depression to its spill level.

    domain: a bool mask of the land.  Water leaves the map at the
    DOMAIN's edge, not the array's, so the flood is seeded from the land cells
    that touch the sea (or the array border) and off-map cells never enter the
    queue at all.  They come back as NaN.  domain=None is the
    whole array."""
    import heapq
    H, W = z.shape
    out = np.full((H, W), np.inf)
    done = np.zeros((H, W), bool)
    pq = []
    if domain is None:
        seed = np.zeros((H, W), bool)
        seed[0], seed[-1], seed[:, 0], seed[:, -1] = 1, 1, 1, 1
    else:
        dom = np.asarray(domain, bool)
        done |= ~dom                       # the sea is not part of the problem
        pad = np.pad(dom, 1, constant_values=False)
        touch = ~(pad[:-2, 1:-1] & pad[2:, 1:-1] &
                  pad[1:-1, :-2] & pad[1:-1, 2:])
        seed = dom & touch                 # land on the coast
    for r, c in zip(*np.where(seed)):
        r, c = int(r), int(c)
        heapq.heappush(pq, (float(z[r, c]), r, c))
        done[r, c] = True
        out[r, c] = z[r, c]
    eps = 1e-4
    while pq:
        zv, r, c = heapq.heappop(pq)
        for dr, dc in _NB:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= H or cc < 0 or cc >= W or done[rr, cc]:
                continue
            nz = max(float(z[rr, cc]), zv + eps)
            out[rr, cc] = nz
            done[rr, cc] = True
            heapq.heappush(pq, (nz, rr, cc))
    return out if domain is None else np.where(domain, out, np.nan)


def d8_accumulation(zf):
    H, W = zf.shape
    big = zf.max() + 1e6
    pad = np.full((H + 2, W + 2), big)
    pad[1:-1, 1:-1] = zf
    best = np.full((H, W), -np.inf)
    rec = np.zeros((H, W), np.int64)
    for dr, dc in _NB:
        nb = pad[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
        s = (zf - nb) / np.hypot(dr, dc)
        m = s > best
        best = np.where(m, s, best)
        rr = np.clip(np.arange(H)[:, None] + dr, 0, H - 1)
        cc = np.clip(np.arange(W)[None, :] + dc, 0, W - 1)
        rec = np.where(m, rr * W + cc, rec)
    outlet = np.zeros((H, W), bool)
    outlet[0], outlet[-1], outlet[:, 0], outlet[:, -1] = 1, 1, 1, 1
    acc = np.ones(H * W)
    recf, outf = rec.ravel(), outlet.ravel()
    for i in np.argsort(-zf.ravel(), kind="stable"):
        if not outf[i]:
            acc[recf[i]] += acc[i]
    return acc.reshape(H, W)


def real_network(dem, target_band):
    """Channels of the real DEM taken down to the zero-order hollows until
    the pinned band matches `target_band`."""
    acc = d8_accumulation(priority_flood(dem))
    srt = np.sort(acc.ravel())
    best = None
    for frac in np.arange(0.01, 0.13, 0.005):
        n = int(frac * dem.size)
        ch = acc >= srt[-n]
        band = ndi.binary_dilation(ch, iterations=1)
        err = abs(band.mean() - target_band)
        if best is None or err < best[0]:
            best = (err, ch, band)
    return best[1], best[2]


def geodesic_profile(band, dem):
    """The generator's own river profile: geodesic
    arc length along the network from each component's lowest pixel."""
    from skimage import graph as skgraph
    lab, n = ndi.label(band, structure=np.ones((3, 3)))
    src = []
    for i in range(1, n + 1):
        rr, cc = np.where(lab == i)
        k = int(np.argmin(dem[rr, cc]))
        src.append((int(rr[k]), int(cc[k])))
    cum, _ = skgraph.MCP_Geometric(np.where(band, 1.0, np.inf)).find_costs(src)
    zr = np.where(np.isfinite(cum) & band, cum, np.nan)
    return np.where(band, np.nan_to_num(zr / np.nanmax(zr), nan=0.0), 0.0)


# ---------------------------------------------------------------------------

def crop(a, f=CROP):
    H, W = a.shape[:2]
    h, w = int(H * f), int(W * f)
    return a[(H - h) // 2:(H - h) // 2 + h, (W - w) // 2:(W - w) // 2 + w]


def norm(u):
    u = np.asarray(u, float)
    return (u - u.min()) / (u.max() - u.min() + 1e-12)


def hillshade(u, az=315, alt=45):
    dy, dx = np.gradient(norm(u) * u.shape[0])
    slope = np.pi / 2 - np.arctan(np.hypot(dx, dy))
    asp = np.arctan2(-dx, dy)
    a, z = np.radians(az), np.radians(alt)
    return (np.sin(z) * np.sin(slope)
            + np.cos(z) * np.cos(slope) * np.cos(a - np.pi / 2 - asp))


def undrained_rgb(mask):
    m = np.asarray(mask, bool)
    ed = ndi.distance_transform_edt(~m)
    p50 = np.median(ed[~m])
    rgb = np.ones(m.shape + (3,))
    rgb[ed > 2 * p50] = (1.00, 0.85, 0.45)
    rgb[ed > 3 * p50] = (0.95, 0.35, 0.15)
    rgb[m] = (0.05, 0.15, 0.45)
    return rgb, p50, float((ed > 3 * p50).mean())
