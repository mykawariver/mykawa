"""
mini -- stage 1 (valley network) and stage 2 (elevation), raster-native.

    gen_headward(rng, N) -> 1-px centreline raster     (stage 1)
    solve(net)           -> surface: floor, geodesic z, Laplace/Poisson,
                            NO ridge pins                 (stage 2)
    score(net, u)        -> a fixed scoreboard, printed next to the real row

It is small and fast (~0.1 s per candidate) so that a rule for placing
valleys can be tested in seconds.

FIXED SCOREBOARD (keep it fixed across experiments so results stay
comparable).  Reference = Kawauchi at the same N:
    undrained   area farther than 3x the median distance-to-channel
    R           Clark-Evans index of the channel heads (>1 = evenly spread)
    tips        number of channel heads
    len50/95    link length distribution (is the net built of the right pieces?)
    t/j         tips per junction (is it branching, or a comb?)
    pieces      connected pieces of the top 25% of the relief
    summits     summits with prominence >= 20% of relief

Usage:
    python3 -m experiments.mini            # score every registered generator
    python3 -m experiments.mini --fig      # + valley_samples/mini.png
"""
import bisect
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi, spatial
from skimage import graph as skgraph, morphology as skmo, measure as skm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import harmonic as EL
from tools import network as NW

N = 128
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "valley_samples")
CACHE = os.path.join(OUT, "cache", f"mini_ref_{N}.npz")
# Trunk accumulation of the calibration map in 1-PX CHANNEL CELLS (median over
# seeds 1-5 at n=128), the reference for the growth-time half-width.  It is the
# same idea as baseline.ACC_REF -- fix the reference so valley width is an
# ABSOLUTE function of drainage area and does not rescale itself when the
# canvas grows -- but in the unit tree_accum counts, which is ~4.9x smaller
# than the dilated band ACC_REF is measured on.
ACC_REF_NET = 341.0


# ---------------------------------------------------------------- reference

def reference():
    """The real Kawauchi network + DEM, at N.  Cached."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["net"].astype(bool), d["dem"]
    from tools.fetch_dem import SITES, site_coords, fetch_patch
    from experiments.ridge_connectivity import priority_flood, d8_accumulation
    lat, lon, layer = site_coords(SITES["kawauchi"])
    dem = fetch_patch(lat, lon, 1500.0, layer)["dem"].astype(float)
    z = ndi.zoom(dem, N / dem.shape[0], order=1)
    acc = d8_accumulation(priority_flood(ndi.gaussian_filter(z, 1.0)))
    # drainage density chosen so the median hillslope is ~4 px, as in the
    # full-resolution reference
    net = skmo.skeletonize(acc >= np.sort(acc.ravel())[-int(0.06 * z.size)])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, net=net, dem=z)
    return net, z


# ---------------------------------------------------------------- machinery

def channel_accum(band, s):
    """How much of the network drains through each channel cell.

    Order the cells by decreasing distance-from-outlet `s` and push each one's
    load into its lowest-`s` neighbour: a tree accumulation.  The trunk is then
    simply the cells with a large load -- which is what tells a main stem from
    a tributary."""
    H, W = band.shape
    cells = np.column_stack(np.where(band & np.isfinite(s)))
    order = cells[np.argsort(-s[cells[:, 0], cells[:, 1]], kind="stable")]
    acc = np.zeros((H, W))
    acc[band] = 1.0
    for (r, c) in order:
        best, bs = None, s[r, c]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (0 <= rr < H and 0 <= cc < W and band[rr, cc]
                        and np.isfinite(s[rr, cc]) and s[rr, cc] < bs):
                    best, bs = (rr, cc), s[rr, cc]
        if best is not None:
            acc[best] += acc[r, c]
    return acc


def hollow_heads(net, px, d=None):
    """Continue every channel head upslope as an UNCHANNELLED HOLLOW.

    Measured against Kawauchi at n=128, the generated
    hillslope has as many concave cells as the real one (65-72%) but its
    hollows are only half as deep -- plan curvature p10 -0.29 against -0.53 --
    and its noses only a third as sharp (p90 0.11 against 0.29).  Water on a
    surface that gently curved never gathers: the share of cells carrying a
    moderate flow (A >= 30) is 6.6% against the real 8.7%, so a quarter of the
    channels the generator DREW never reach the critical area that makes them
    channels at all.  The network is right; the ground between its lines is
    too smooth to feed it.

    What real terrain has and this did not is the ZERO-ORDER BASIN: above every
    channel head the hollow CONTINUES, unchannelled, for a hillslope length or
    so, and that is where flow first converges.  Here the head is simply
    extended: `px` more cells in the direction the head was already pointing,
    curving up the distance-to-network gradient so the extension climbs into
    the middle of its divide instead of running at a neighbour.  The extension
    joins the network BEFORE the profile is integrated, so it gets its own
    slope-area rise (steep, because its accumulation is tiny) and its own 1-px
    floor -- it is a hollow, not a channel, and nothing has to special-case it.

    Adding no relief anywhere, it only turns a smooth slope into an alternating
    hollow-and-nose one.  px = 0 is a no-op."""
    net = np.asarray(net, bool)
    if px <= 0:
        return net
    H, W = net.shape
    deg = (ndi.convolve(net.astype(np.uint8), np.ones((3, 3), np.uint8),
                        mode="constant") - net)
    tips = np.column_stack(np.where(net & (deg == 1)))
    if not len(tips):
        return net
    if d is None:
        d = ndi.distance_transform_edt(~net)
    gr, gc = np.gradient(ndi.gaussian_filter(d, 2.0))
    out = net.copy()
    for (r0, c0) in tips:
        # the head's own heading: away from its single neighbour
        nb = np.column_stack(np.where(
            net[max(0, r0 - 1):r0 + 2, max(0, c0 - 1):c0 + 2]))
        if not len(nb):
            continue
        nb = nb + [max(0, r0 - 1), max(0, c0 - 1)]
        nb = nb[(nb[:, 0] != r0) | (nb[:, 1] != c0)]
        if not len(nb):
            continue
        ang = np.arctan2(r0 - nb[0][0], c0 - nb[0][1])
        r, c = float(r0), float(c0)
        for _k in range(int(round(px))):
