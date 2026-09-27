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
            gy = gr[int(np.clip(r, 0, H - 1)), int(np.clip(c, 0, W - 1))]
            gx = gc[int(np.clip(r, 0, H - 1)), int(np.clip(c, 0, W - 1))]
            ga = np.arctan2(gy, gx)              # uphill, away from channels
            ang = np.arctan2(0.7 * np.sin(ang) + 0.3 * np.sin(ga),
                             0.7 * np.cos(ang) + 0.3 * np.cos(ga))
            r, c = r + np.sin(ang), c + np.cos(ang)
            ri, ci = int(round(r)), int(round(c))
            if not (1 <= ri < H - 1 and 1 <= ci < W - 1):
                break
            if net[ri, ci]:                      # ran into another channel
                break
            out[ri, ci] = True
    return out


def cone_floor(band, hw, step=0.5, field=False):
    """The valley floor as a CONE DILATION of the channels.

        floor(x) = max over channels y of [ hw(y) - |x - y| ] >= 0

    Asking only the NEAREST channel (floor = do <= hw[nearest]) fails: in a
    network whose channels are 5-6 px apart, a cell 5 px from the trunk is
    usually nearest to some 1-px tributary, whose hw = 1 vetoes it, so the
    trunk's wide floor is eaten away by whatever small channel comes close
    and floor width stops following drainage area (W ~ A^0.02 on a 12 km
    strip, against A^0.49 in real transects).  A cone dilation asks
    instead whether ANY channel reaches the cell, which is what "this ground is
    the floor of that valley" means.

    Implemented as a union of level sets: channels are bucketed by hw, each
    bucket is distance-transformed once, and the best (hw - distance) wins.
    The owner indices come back with it, so the floor cell can be pinned at the
    z of the channel whose valley it belongs to -- again not the nearest one.

    Returns (floor, (iy, ix)) with the same meaning as the indices from
    distance_transform_edt: for a floor cell, the channel cell that owns it.
    With field=True a third item comes back: max(hw - distance) itself, whose
    negative part is the DISTANCE TO THE NEAREST VALLEY (not to the nearest
    centreline) -- which is what "undrained land" has to mean once valleys have
    width, and what gen_headward grows against."""
    H, W = band.shape
    hwv = np.where(band, hw, 0.0)
    hmax = float(hwv.max())
    iy, ix = np.mgrid[0:H, 0:W]
    if hmax <= 0:
        z = np.zeros((H, W), bool)
        return (z, (iy, ix), np.full((H, W), -np.inf)) if field else (z, (iy, ix))
    step = max(float(step), hmax / 24.0)     # <=24 transforms, whatever hmax is
    best = np.full((H, W), -np.inf)
    lo = 0.0
    for hi in np.arange(step, hmax + step, step):
        sub = band & (hwv > lo) & (hwv <= hi + 1e-12)
        lo = hi
        if not sub.any():
            continue
        d, (jy, jx) = ndi.distance_transform_edt(~sub, return_indices=True)
        v = hwv[jy, jx] - d
        m = v > best
        best = np.where(m, v, best)
        iy = np.where(m, jy, iy)
        ix = np.where(m, jx, ix)
    return ((best >= 0.0, (iy, ix), best) if field
            else (best >= 0.0, (iy, ix)))


def tree_accum(net, S):
    """Cells draining through each 1-px network cell, from the growth record.

    channel_accum does the same job on the DILATED band after the fact; this
    one runs DURING growth, where the only thing known about the tree is S, the
    distance-from-outlet stamped on each cell as it was painted.  A cell's
    receiver is its neighbour with the smallest S, which is exactly the tree
    the growth built, so one reverse sweep gives the accumulation.  The unit is
    1-px channel cells, not band cells -- hence its own reference constant
    (ACC_REF_NET), not baseline.ACC_REF."""
    H, W = net.shape
    Sp = np.where(net & np.isfinite(S), S, np.inf)
    pad = np.pad(Sp, 1, constant_values=np.inf)
    best = np.full((H, W), np.inf)
    par = np.full((H, W), -1, np.int64)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nb = pad[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
            m = nb < best
            best = np.where(m, nb, best)
            rr, cc = np.where(m)
            par[rr, cc] = (rr + dr) * W + (cc + dc)
    par = np.where(best < Sp, par, -1)      # the outlet has no receiver
    ok = net & np.isfinite(Sp)
    acc = np.ones(H * W)
    pf = par.ravel()
    cells = np.flatnonzero(ok.ravel())
    cells = cells[np.argsort(-Sp.ravel()[cells], kind="stable")]
    for k in cells:
        if pf[k] >= 0:
            acc[pf[k]] += acc[k]
    return np.where(ok, acc.reshape(H, W), 0.0)


def enforce_downstream(z, band, s):
    """Make z non-increasing downstream, by LOWERING, never raising.

    harmonic.lipschitz_cap works on planar distance, so it can pull a
    headwater below a trunk cell that is downstream of it -- water would run
    uphill.  Sweeping the cells from the top of the network down and taking
    z[downstream] = min(z[downstream], z[upstream]) restores monotonicity while
    staying under the cap (it only ever lowers)."""
    H, W = band.shape
    z = z.copy()
    cells = np.column_stack(np.where(band & np.isfinite(s)))
    for (r, c) in cells[np.argsort(-s[cells[:, 0], cells[:, 1]], kind="stable")]:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (0 <= rr < H and 0 <= cc < W and band[rr, cc]
                        and np.isfinite(s[rr, cc]) and s[rr, cc] < s[r, c]):
                    if z[rr, cc] > z[r, c]:
                        z[rr, cc] = z[r, c]
    return z


def profile_slope_area(band, s, acc, theta=0.5, bow=0.0):
    """River long profile by integrating slope = acc**-theta upstream.

    A UNIFORM rise per unit length along the network makes cliffs where the
    network folds back on itself (two reaches far apart along the network but
    close on the map).  Real rivers obey Flint's law, slope ~ area**-theta
    with theta 0.4-0.7, so the trunk is nearly flat and only the
    headwaters climb steeply.  Two reaches that meet after a long detour are
    then both far down the profile and differ little.

    Walk the cells in order of increasing distance-from-outlet and add
    acc**-theta per step to the downstream neighbour's z.

    `bow` (default 0 = the frozen behaviour).  theta does TWO jobs at once: it bows the long profile (wanted) and it sets the
    elevation CONTRAST BETWEEN NEIGHBOURING CHANNELS of different accumulation
    (texture's job).  Two headwater branches with acc 5 and 50 differ by
    10**theta, so raising theta sharpens every headwater notch on the map.

    `bow` bows the profile by a route that does not touch that contrast: it
    multiplies the slope by exp(bow * (s/s_max - 0.5)), a function of DISTANCE
    FROM THE OUTLET alone.  Two channels at the same distance are scaled
    identically, so their contrast -- the texture -- is unchanged, while the
    lower reach is flattened (bow > 0) or steepened (bow < 0) and the profile
    bows either way.  bow = 0 makes the weight identically 1.

    WHY THIS IS NOT AN AD-HOC HACK.  Along a trunk, Hack's law makes drainage
    area grow as A ~ s**1.8, so a weight s**q is a slope ~ A**-(q/1.8) applied
    ALONG THE PROFILE ONLY.  In other words theta and bow reach the same
    concavity through the same physics; the difference is that theta also
    prices the area difference BETWEEN NEIGHBOURS, and bow does not.  bow is
    the basin-scale half of concavity with the local half removed.  (The
    exponential rather than the power keeps the weight finite at the outlet,
    which a negative exponent would not.)"""
    H, W = band.shape
    z = np.zeros((H, W))
    cells = np.column_stack(np.where(band & np.isfinite(s)))
    cells = cells[np.argsort(s[cells[:, 0], cells[:, 1]], kind="stable")]
    a = np.maximum(acc, 1.0) ** (-theta)
    if bow:
        smax = float(np.nanmax(s[band])) or 1.0
        x = np.clip(np.where(band, s, 0.0) / smax, 0.0, 1.0)
        a = a * np.exp(float(bow) * (x - 0.5))
    for (r, c) in cells:
        best, bs = None, s[r, c]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if (0 <= rr < H and 0 <= cc < W and band[rr, cc]
                        and np.isfinite(s[rr, cc]) and s[rr, cc] < bs):
                    best, bs = (rr, cc), s[rr, cc]
        if best is not None:
            z[r, c] = z[best] + a[r, c] * (s[r, c] - bs)
    return z


def basin_labels(net, roots, domain=None):
    """Which OUTLET each channel cell belongs to, as an int raster (0 = none).

    THE CONTRACT between stage 1 and stage 2 is the network plus this one
    raster, and nothing else (keep what is passed minimal): stage 2 does not
    have to guess which river is which.

    Why it is needed at all: stage 2 builds its Dirichlet band as
    dilate(hollow_heads(net)), and the zero-order hollow reaches 1.6*ell past
    every channel head with NO separation test, so two headwaters facing each
    other across a divide would weld their basins into one.  `s` would then be
    measured to the NEAREST outlet, the land beyond the weld would change
    hands, and a small river could be handed the widest valley floor.

    HOW IT IS COMPUTED, and what a different stage 1 would have to do.
    `_walk` refuses to come within `sep` of any channel but its own, so the
    trees are disjoint and connected components recover the assignment
    exactly.  That is a property of THIS growth rule, not of the interface: a
    stage 1 that lets trees touch must paint the labels as it grows and return
    them, and stage 2 will not care which way they were made."""
    lab, nl = ndi.label(np.asarray(net, bool), structure=np.ones((3, 3)))
    out = np.zeros(lab.shape, np.int16)
    if domain is not None:
        # THE NA OF THIS PIPELINE.  bool cannot carry a third state, so the missing value lives
        # in the label raster: -1 = off the map, 0 = land with no channel,
        # k > 0 = a channel draining to outlet k-1.  `net_na` below turns the
        # pair back into one float raster with np.nan for anyone plotting it.
        out[~np.asarray(domain, bool)] = -1
    _d, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True)
    for i, rc in enumerate(roots):
        r, c = int(rc[0]), int(rc[1])
        k = int(lab[iy[r, c], ix[r, c]])
        if k:
            out[lab == k] = i + 1
    return out


def net_na(net, labels):
    """The stage 1 output as ONE float raster: 1 = channel, 0 = land,
    np.nan = off the map.

    Two rasters is the right thing to pass between stages -- bool stays bool,
    so nothing downstream can mistake a NaN for True -- and one raster with a
    hole in it is the right thing to look at.  This is the second."""
    out = np.where(np.asarray(net, bool), 1.0, 0.0)
    return np.where(np.asarray(labels) < 0, np.nan, out)


def clip_to_own_basin(mask, labels, gap=1.5):
    """Drop the cells of `mask` that lie nearer another basin's channels.

    The zero-order hollow may climb its own hillslope; it may not cross the
    divide.  A cell is kept when its own basin's channels are at least `gap`
    px closer than any other basin's, so the two hollows stop short of each
    other.  Channel cells themselves are always kept.  With one basin (or no
    labels) nothing is dropped.

    WHY gap = 1.5 AND NOT 1.  Two kept cells of different basins must not end
    up 8-adjacent, or the band welds anyway.  Between neighbours the two
    distance fields can each move by up to sqrt(2), so a kept pair needs
    gap < sqrt(2) to exist; anything at or above 1.415 makes it impossible.
    (With gap = 1.0 the weld survives at isolated cell pairs.)

    ORPHAN FRAGMENTS.  Clipping can shear the far end off a hollow, leaving a
    handful of cells with no channel in them.  They would be Dirichlet islands
    with no outlet to measure a profile from, so they are dropped too."""
    lab = np.asarray(labels, np.int16)
    ids = [int(t) for t in np.unique(lab) if t > 0]
    m = np.asarray(mask, bool)
    if len(ids) < 2:
        return m
    D = np.stack([ndi.distance_transform_edt(lab != t) for t in ids])
    D.sort(axis=0)
    m = m & ((lab > 0) | (D[0] + float(gap) < D[1]))
    comp, nc = ndi.label(m, structure=np.ones((3, 3)))
    keep = np.zeros(nc + 1, bool)
    keep[np.unique(comp[m & (lab > 0)])] = True
    keep[0] = False
    return keep[comp]


def solve(net, tilt=None, poisson=0.006, theta=0.3, cap_px=0.0, floor_px=4.0,
          outlets=None, acc_ref=None, bow=0.0, floor_mul=1.0, cone=True,
          floor_tilt=0.0, hollow_px=0.0, labels=None):
    """1-px net -> surface.  NO ridge pins.

    The river profile.  Coloured by z, a real network reads as one distinctly
    low trunk with every branch warming upstream.  Plain geodesic arc length
    from the outlet does not give that -- in a space-filling network it is
    close to Euclidean distance, a smooth radial ramp with no hierarchy.  So z
    along the channels comes from profile_slope_area (Flint's law,
    slope ~ acc**-theta, integrated ALONG the network): the trunk is nearly
    flat, headwaters climb steeply, and the profile is monotone upstream by
    construction.

    poisson: with Dirichlet only on the channels the maximum principle puts the
    surface maximum on the highest channel, so a divide can never rise above
    the channels and the channels end up spread through the whole relief
    (measured: 49th percentile of elevation, against 31% for Kawauchi).
    Solving nabla^2 u = -poisson gives the hillslope a convex rise instead.

    cap_px defaults to OFF.  With the cap on, the trunk profile becomes a
    staircase (46-60% of trunk steps dead flat, the largest single-px drop
    0.22-0.27 of the whole relief; real Kawauchi: 0% flat, max 0.024), seen on
    a contour map as chains of contour pinches along the trunk.  It is the
    cap+enforce interaction -- lipschitz_cap pulls single cells down by PLANAR
    distance, then enforce_downstream propagates each dip as a flat shelf
    ending in a cliff.  A headwater towering over a nearby trunk is already
    prevented at GENERATION time by the fold_cap test in _walk, so the profile
    needs no after-the-fact clamping.

    cone: paint the floor by CONE DILATION (see cone_floor) rather than by
    asking each cell's nearest channel.  cone=False is the nearest-channel
    rule.

    floor_px: without a floor the cross-valley section is 2x too steep near
    the channel (mean rise within 2 px: 0.0215 vs real 0.0121) -- a 1-px
    Dirichlet slot with Poisson walls.  Real satoyama trunk valleys have a
    FLOOR.  So the Dirichlet band is widened by channel accumulation (hydraulic
    geometry, width ~ acc^0.5), each floor
    cell pinned at its nearest channel cell's z: tributaries stay 1 px, the
    trunk gets a flat bottom up to floor_px half-width.

    THE LONG PROFILE IS MEASURED ON THE 1-PX NETWORK, NOT THE DILATED BAND.
    Measuring s (distance-from-outlet) and acc over the dilated band let a
    bend's own side cells shortcut past the channel's true, longer path; their
    own tiny accumulation then integrates a different rise, so about half of
    them came out LOWER than the channel cell right next to them -- the
    river was often not at the bottom of its own valley.  net_h has one
    profile per stream; every wider cell -- the band's own 1-px margin as well
    as the floor -- is instead PAINTED at its nearest net_h cell's z, the same
    rule the floor already used.  Measured against the nearest channel cell
    (seeds 1-3, lower by more than 0.5% of relief), the share of cells
    1-1.5 px from a channel that sit BELOW it drops from ~43% to ~1% (real
    Kawauchi: 7%), and at 2-4 px from ~23% to ~6% (real 1%).

    Valley WIDTH is still read from the accumulation of the dilated band
    (`acc_w` below), because ACC_REF and floor_px were calibrated in those
    units; the 1-px network carries far less accumulation per cell, and using
    it there would shrink every floor (trunk half-width 4.7 -> 1.8 px)."""
    net_h = hollow_heads(net, hollow_px)
    # labels: the one thing stage 1 hands over besides the raster.
    # Without it the hollow welds neighbouring basins and the solver performs a
    # river capture the network never drew -- see basin_labels above.
    if labels is not None:
        net_h = clip_to_own_basin(net_h, labels)
    band = ndi.binary_dilation(net_h)
    if labels is not None:
        band = clip_to_own_basin(band, labels)
    # THE DOMAIN RIDES IN ON THE SAME RASTER.  labels < 0 is the NA stage 1
    # writes for "off the map", so stage 2 needs no third argument to know the
    # shape of the land.  An all-land map takes the whole-array path.
    dom = None
    if labels is not None and (np.asarray(labels) < 0).any():
        dom = np.asarray(labels) >= 0
    H, W = band.shape
    iy0, ix0 = np.mgrid[0:H, 0:W]
    n = H                       # alias, square maps only
    # default tilt: the BOTTOM row is low.  gen_headward's single-root default
    # grows its tree from the bottom centre, and argmin of this ramp picks the
    # outlet; the other way round, distance-from-outlet, accumulation and the
    # whole Flint profile would be computed from the wrong end (the scoreboard
    # would not notice: undrained/R/tips are z-free).
    ref = (tilt if tilt is not None
           else np.linspace(1, 0, H)[:, None] * np.ones((1, W)))
    if outlets is not None:
        # explicit outlets (e.g. the roots of edge-rooted trees), snapped to
        # the nearest 1-px channel cell -- skeletonize may have shifted them a px
        _, (iyb, ixb) = ndi.distance_transform_edt(~net_h, return_indices=True)
        src = [(int(iyb[int(r), int(c)]), int(ixb[int(r), int(c)]))
               for (r, c) in outlets]
    else:
        lab, nl = ndi.label(net_h, structure=np.ones((3, 3)))
        outs = []
        for i in range(1, nl + 1):
            rr, cc = np.where(lab == i)
            k = int(np.argmin(ref[rr, cc]))
            outs.append((int(cc[k]) / (W - 1), 1.0 - int(rr[k]) / (H - 1)))
        src = [(int(round((1 - y) * (H - 1))), int(round(x * (W - 1))))
               for (x, y) in outs]
    cum, _ = skgraph.MCP_Geometric(np.where(net_h, 1.0, np.inf)).find_costs(src)
    s = np.where(net_h, cum, np.inf)
    acc = channel_accum(net_h, s)
    zn_net = profile_slope_area(net_h, np.where(net_h, s, np.nan), acc,
                                theta=theta, bow=bow)
    zn_net = np.where(net_h, zn_net / (zn_net.max() or 1.0), 0.0)
    # the accumulation valley WIDTH is calibrated in (see the docstring)
    if floor_px:
        cum_b, _ = skgraph.MCP_Geometric(
            np.where(band, 1.0, np.inf)).find_costs(src)
        acc_w = channel_accum(band, np.where(band, cum_b, np.inf))
    if cap_px:
        from experiments.harmonic import lipschitz_cap
        zn_net = np.nan_to_num(lipschitz_cap(np.where(net_h, zn_net, np.nan),
                                             net_h, cap_px), nan=0.0)
        zn_net = enforce_downstream(zn_net, net_h, s)
        zn_net = np.where(net_h, zn_net - zn_net[net_h].min(), 0.0)
        zn_net = zn_net / (zn_net.max() or 1.0)
    # paint the band's own 1-px margin (band minus net_h) at its owner's z --
    # exactly the rule the floor uses below, applied here to the dilation
    # margin that used to get its own (wrong) profile.
    _, (iyn, ixn) = ndi.distance_transform_edt(~net_h, return_indices=True)
    zn = np.where(net_h, zn_net, zn_net[iyn, ixn])
    zn = np.where(band, zn, 0.0)
    if floor_px:
        # acc_ref.  Normalising by
        # acc.max() makes the widest channel of ANY map exactly floor_px wide,
        # so a trunk draining a hundred times more land comes out the same
        # width and the measured law W ~ A^0.49 (four transects, ANCOVA
        # r2=0.56) silently loses its prefactor as the canvas grows.  A FIXED
        # reference accumulation restores the absolute law: pass the trunk
        # accumulation of the calibration map (n=128) and every larger map's
        # trunk then widens as sqrt(A) as it should.  None keeps the
        # map-relative behaviour for the mini bench.
        # floor_mul: a multiplier on the whole half-width.  It widens every
        # valley at its own drainage area, i.e. it moves the PREFACTOR of the
        # measured law W ~ A**0.49 while leaving the exponent alone (that
        # prefactor, W(A=1e4) = 24..110 m across four real transects, is a
        # style axis).  Scaling floor_px alone would only widen the part that
        # carries sqrt(A), and at n=128 the median channel cell drains ~2
        # cells, so sqrt(A/ACC_REF) ~ 0.04 there.  floor_mul = 1 is the frozen
        # behaviour; baseline's `floor` knob goes through floor_px.
        ref = acc_w.max() if acc_ref is None else float(acc_ref)
        accn = np.where(band, acc_w / (ref or 1.0), 0.0)
        hw_b = np.where(band, (1.0 + (floor_px - 1.0) * np.sqrt(accn))
                        * float(floor_mul), 0.0)
        # carry the band's half-widths onto the 1-px network: each channel
        # cell reaches as far as the widest band cell next to it, plus 0.7 px
        # for the band cell's own stand-off from the centreline.  0.7 is set
        # so the pinned area matches the band-owned floor it replaces (seeds
        # 1-3: 38.7 / 42.4 / 40.8% of the map, against 38.0 / 41.1 / 39.7%);
        # the floor keeps its calibrated width but is owned by channel cells.
        hw = np.where(net_h, ndi.maximum_filter(hw_b, size=3) + 0.7, 0.0)
        if cone:
            floor, (iyo, ixo) = cone_floor(net_h, hw)
        else:                       # nearest-channel rule
            do, (iyo, ixo) = ndi.distance_transform_edt(
                ~net_h, return_indices=True)
            floor = do <= hw[iyo, ixo]
        # THE FLOOR MUST SLOPE TOWARDS ITS RIVER.  Pinning
        # every floor cell at exactly its channel's z makes the floor DEAD
        # FLAT, and D8 on dead flat ground is decided by arithmetic noise: the
        # router lays channels wherever the fill happens to tip.  Measured at
        # n=128, 25-36% of the channel cells the finished surface routes are
        # nowhere near a channel the generator drew, and 69-87% of those sit
        # inside a painted floor -- they are routing artefacts, not valleys.
        #
        # A real alluvial floor falls gently towards its river, so the floor cell is pinned at
        # its owner's z PLUS a small rise per px of distance from it.  The
        # tilt is symmetric about the channel, so it cannot reverse the
        # downstream direction; it only makes the cross-valley direction
        # decided instead of arbitrary.  floor_tilt = 0 (the default) is the
        # flat floor.
        if labels is not None:
            # the painted floor must respect the divide too, or the two basins
            # touch again through their floors and the Dirichlet band -- which
            # is what the surface actually sees -- is welded after all.
            floor = clip_to_own_basin(floor, labels)
        # The cone decides WHERE the floor is; its z comes from the NEAREST
        # channel cell, not the cone's owner.  The owner is usually a wider,
        # DOWNSTREAM trunk cell whose cone reaches up beside the channel, so
        # the floor next to a channel cell would be painted lower than that
        # cell and the river would sit on a step above its own floor
        # (measured: ~10% of channel cells higher than their own cross-
        # section; with the nearest cell's z, ~6%; real Kawauchi 1%).
        dow = np.hypot(iy0 - iyn, ix0 - ixn) if floor_tilt else 0.0
        zn = np.where(floor & ~band, zn_net[iyn, ixn] + float(floor_tilt) * dow, zn)
        band = band | floor
    z0 = np.zeros_like(band, bool)
    u = EL.solve_two_bones(band, zn, z0, np.zeros(band.shape),
                           poisson_f=poisson, domain=dom)
    # nanmin/nanmax so the off-map NaNs do not poison the normalisation; with
    # no domain these are the plain min/max and the result is unchanged.
    lo, hi = np.nanmin(u), np.nanmax(u)
    return (u - lo) / (hi - lo + 1e-12)


def score(net, u):
    net = np.asarray(net, bool)
    ed = ndi.distance_transform_edt(~net)
    p50 = np.median(ed[~net])
    deg = ndi.convolve(net.astype(int), np.ones((3, 3)), mode="constant") - net
    tips, junc = net & (deg == 1), net & (deg >= 3)
    njun = ndi.label(junc, structure=np.ones((3, 3)))[1]
    links = net & ~ndi.binary_dilation(junc, iterations=1)
    lab, nl = ndi.label(links, structure=np.ones((3, 3)))
    L = np.bincount(lab.ravel(), minlength=nl + 1)[1:]
    L = L[L >= 3]
    pts = np.column_stack(np.where(tips))
    if len(pts) >= 8:
        dd = spatial.cKDTree(pts).query(pts, k=2)[0][:, 1]
        R = dd.mean() / (0.5 / np.sqrt(len(pts) / net.size))
    else:
        R = 0.0
    hi = u >= np.quantile(u, 0.75)
    lb = skm.label(hi, connectivity=2)
    pieces = sum(1 for p in skm.regionprops(lb) if p.area >= 0.0005 * u.size)
    # loops: a drainage network is a TREE, so any enclosed hole is a defect
    # Holes =
    # components - Euler number.  The real network scores a handful from
    # braided reaches and raster noise; a generator that scores tens is
    # running new branches alongside old ones instead of joining them.
    ncomp = ndi.label(net, structure=np.ones((3, 3)))[1]
    loops = int(ncomp - skm.euler_number(net, connectivity=2))
    return dict(dens=net.mean() * 100, p50=p50,
                undr=float((ed > 3 * p50).mean()) * 100, R=R, tips=len(pts),
                len50=float(np.median(L)) if L.size else 0,
                len95=float(np.quantile(L, .95)) if L.size else 0,
                tj=len(pts) / max(njun, 1), loops=loops, pieces=pieces,
                summits=NW.summit_count(ndi.gaussian_filter(u, 1.0), 0.20))


HDR = (f"{'generator':<16}{'s':>5}{'dens%':>7}{'p50':>5}{'undr%':>7}{'R':>6}"
       f"{'tips':>6}{'len50':>7}{'len95':>7}{'t/j':>6}{'loops':>7}{'pieces':>8}"
       f"{'summits':>8}")


def row(name, s, secs):
    return (f"{name:<16}{secs:5.2f}{s['dens']:7.2f}{s['p50']:5.1f}{s['undr']:7.2f}"
            f"{s['R']:6.2f}{s['tips']:6d}{s['len50']:7.1f}{s['len95']:7.1f}"
            f"{s['tj']:6.2f}{s['loops']:7.0f}{s['pieces']:8d}{s['summits']:8d}")


_FOLD = [0, None, None]   # diagnostic: rejects, S field, raw net

# ---------------------------------------------------------------- generators
# A generator is (rng, N) -> 1-px bool centreline.  Keep each one SHORT: if a
# rule needs more than ~40 lines to state, it is not a rule, it is a pipeline.

def _walk(net, r, c, ang, steps, rng, wander=0.5, stop_on_hit=False, sep=0,
          fold=None, s_here=0.0, forbid=None, hw0=0.0, domain=None):
    def s_here_k(k):
        return s_here + k + 1.0
    """Draw a short wandering stroke; returns the end cell, or None if it left
    the grid / had to stop.

    sep: the stroke must stay at least `sep` px away from every channel EXCEPT
    the one it grew out of.  A stroke free to run alongside an existing
    channel and rejoin it crowds the fine detail AND closes a loop (a drainage network is a tree, so a loop is
    always an error).  Stopping instead of joining is the physically right
    choice: a tributary belongs to the catchment it started in, so it must not
    reach across into a neighbour's channel.  Every stroke then touches the
    network only at its mouth and the result is a tree by construction.

    The mouth itself sits ON the network, so a disc of radius sep+1 around the
    start is exempted -- otherwise every stroke would be rejected at birth.

    forbid is `sep` grown up: a bool field of the ground that already lies
    inside some channel's own valley floor.  sep is a CONSTANT 2 px, which is
    the right exclusion for a 1-px headwater rill and the wrong one for a trunk
    whose floor is 12 px wide -- with a constant sep alone, channels get laid
    5-6 px from the trunk, so the trunk has no valley to speak of, no ridge can
    run beside it, and the widened floor has nowhere to go.  See gen_headward for how the field is built.

    fold = (S, cap) with s_here_k(k) forbids the stroke from CURLING BACK.
    Two reaches that are far apart ALONG the
    network but close in the PLANE must differ a lot in elevation across a thin
    divide, so no slope cap can make the surface realisable -- the geometry
    itself is contradictory.  The test is therefore the Lipschitz condition
    imposed at GENERATION time: a step is refused when

        |s(here) - s(nearest existing channel)|  >  cap * planar distance

    i.e. the network may only fold back on itself as tightly as a real
    hillslope gradient allows."""
    H, W = net.shape
    s = int(sep) if sep else 2
    others = net.copy()
    r0, c0 = int(round(r)), int(round(c))
    others[max(0, r0 - s - 1):r0 + s + 2,
           max(0, c0 - s - 1):c0 + s + 2] = False
    if forbid is not None:
        # the same exemption as `others`, at the scale of the valley the stroke
        # is being born into: a tributary leaving a trunk whose floor is 12 px
        # wide has to cross those 12 px, so a disc of the mouth's own half-
        # width is cleared.  Beyond it the stroke must be OUT of the valley --
        # which is what stops channels running along inside the trunk's floor.
        e = int(np.ceil(max(s, hw0))) + 1
        forbid = forbid.copy()
        forbid[max(0, r0 - e):r0 + e + 1, max(0, c0 - e):c0 + e + 1] = False
    if fold is not None:
        # measure against the network EXCLUDING this stroke and its mouth.  The
        # nearest channel to a growing tip is otherwise its own fresh trail,
        # whose S equals the tip's own, so the test would never fire.
        do, (iyo, ixo) = ndi.distance_transform_edt(~others, return_indices=True)
    for k in range(steps):
        ang += rng.normal(0, wander)
        r2, c2 = r + np.sin(ang), c + np.cos(ang)
        ri, ci = int(round(r2)), int(round(c2))
        if not (1 <= ri < H - 1 and 1 <= ci < W - 1):
            return None
        if domain is not None and not domain[ri, ci]:
            # OFF THE LAND.  Unlike `forbid`, this test has no
            # exemption disc around the mouth: an outlet sits ON the coast, so
            # clearing a disc there would let every stroke start by walking out
            # to sea.  The domain is the one boundary a stroke may never cross.
            return None
        if sep and others[max(0, ri - s):ri + s + 1,
                          max(0, ci - s):ci + s + 1].any():
            return None                         # too close to another channel
        if forbid is not None and forbid[ri, ci]:
            return None                         # inside another valley's floor
        if fold is not None:
            S, cap = fold
            sn = S[iyo[ri, ci], ixo[ri, ci]]
            dd = do[ri, ci]
            if np.isfinite(sn) and dd > 0 and abs(s_here_k(k) - sn) > cap * dd:
                _FOLD[0] += 1
                return None                     # would curl back on itself
        if stop_on_hit and k >= 2 and net[ri, ci]:
            net[ri, ci] = True                  # a confluence, then stop
            return None
        net[ri, ci] = True
        r, c = r2, c2
    return r, c, ang


def gen_comb(rng, n=N, ell=4.0):
    """A comparison case: a few long smooth trunks, then ONE generation of
    short barbs combed onto them."""
    net = np.zeros((n, n), bool)
    # trunks: smooth long strokes from the bottom edge
    for k in range(3):
        r, c, ang = n - 2, n * (k + 1) / 4.0, -np.pi / 2
        st = _walk(net, r, c, ang, int(n * 1.2), rng, wander=0.08)
    # barbs: attach perpendicular-ish, one generation only, no re-branching
    for _ in range(600):
        cand = np.column_stack(np.where(net))
        r, c = cand[rng.integers(len(cand))]
        ang = rng.choice([-1, 1]) * np.pi / 2 + rng.normal(0, 0.5)
        _walk(net, r, c, ang, int(rng.integers(3, 14)), rng, wander=0.25)
    return skmo.skeletonize(net)


def gen_bisect(rng, n=N, ell=4.0):
    """A comparison case: lay each new valley ALONG the divide.  It scores well
    on the scoreboard and looks wrong (nature never cuts a valley along a
    crest)."""
    net = np.zeros((n, n), bool)
    net[n - 2, n // 2] = True
    _walk(net, n - 2, n // 2, -np.pi / 2, int(n * 1.1), rng, wander=0.1)
    for _ in range(14):
        d = ndi.distance_transform_edt(~net)
        core = skmo.skeletonize(skmo.medial_axis(~net) & (d >= ell))
        if not core.any():
            break
        net |= core
    return skmo.skeletonize(net)


def _restart_S(net, roots):
    """Distance from the nearest outlet, along an already-grown network.

    During growth S is stamped cell by cell; on a restart it has to be
    recovered, and the geodesic distance through the channels is the same
    quantity (`solve` computes it the same way to build the long profile)."""
    src = []
    _, (iy, ix) = ndi.distance_transform_edt(~net, return_indices=True)
    for (r0, c0, _a) in roots:
        r0, c0 = int(r0), int(c0)
        src.append((r0, c0) if net[r0, c0] else (int(iy[r0, c0]),
                                                int(ix[r0, c0])))
    cum, _ = skgraph.MCP_Geometric(np.where(net, 1.0, np.inf)).find_costs(src)
    return np.where(net & np.isfinite(cum), cum, np.nan)


def _restart_tips(net, S, rng, reach, seed_flanks):
    """Where the next, finer round of growth starts.

    Two kinds of seed, and they are the two kinds of hollow real terrain has:
    the HEAD of an existing channel (the zero-order basin above it) and a point
    on the FLANK of a reach (a new first-order tributary).  Both are handed to
    the ordinary growth loop as tips, so both are grown by the ordinary rule.

    A restarted head is a tributary of what it grew from, so it never gets a
    trunk-sized budget: `reach` here is already the round's own (smaller) one.
    The heading of a flank seed is the local uphill, which is the direction the
    divide lies in -- the same field `look` steers by."""
    deg = (ndi.convolve(net.astype(np.uint8), np.ones((3, 3), np.uint8),
                        mode="constant") - net)
    d = ndi.distance_transform_edt(~net)
    gr, gc = np.gradient(ndi.gaussian_filter(d, 2.0))
    tips = []
    for (r, c) in np.column_stack(np.where(net & (deg == 1))):
        nb = np.column_stack(np.where(net[max(0, r - 1):r + 2,
                                          max(0, c - 1):c + 2]))
        nb = nb + [max(0, r - 1), max(0, c - 1)]
        nb = nb[(nb[:, 0] != r) | (nb[:, 1] != c)]
        if not len(nb):
            continue
        a = np.arctan2(r - nb[0][0], c - nb[0][1])   # away from its neighbour
        s = S[r, c]
        tips.append((float(r), float(c), float(a), 0.0, 0.0, float(reach),
                     float(s) if np.isfinite(s) else 0.0))
    if seed_flanks:
        reach_px = np.column_stack(np.where(net & (deg == 2)))
        k = int(round(seed_flanks * len(reach_px) / 100.0))
        if k and len(reach_px):
            for i in rng.choice(len(reach_px), size=min(k, len(reach_px)),
                                replace=False):
                r, c = reach_px[i]
                a = np.arctan2(gr[r, c], gc[r, c])
                if not np.isfinite(a):
                    continue
                s = S[r, c]
                tips.append((float(r), float(c), float(a), 0.0, 0.0,
                             float(reach), float(s) if np.isfinite(s) else 0.0))
    rng.shuffle(tips)
    return tips


def gen_headward(rng, n=N, ell=5.0, split=0.45, split_p=0.0, sep=2,
                 budget0=1.0, side_keep=0.45, side_ang=(0.5, 1.05),
                 main_jit=0.25, run_max=26, step=(3, 7), fold_cap=6.0,
                 roots=None, shape=None, reach=None, corridor=1.0,
                 floor_px=4.0, acc_ref=ACC_REF_NET, net0=None, seed_flanks=0,
                 pin_frac_max=None, reserve=None, trend=None, trend_max=1.15,
                 reseed=True, labels=False, domain=None):
    """Stage 1: valleys grow HEADWARD into the undrained land and split.

    Nature never cuts a valley along a crest.
    A wide interfluve is consumed by its neighbouring valleys lengthening at
    their heads and sprouting side branches.  So: keep a list of tips; extend
    the tip that faces the most undrained land, up the distance-to-network
    gradient; occasionally split it in two.  Stop when nothing is farther than
    ell from a channel.

    STREAM BUDGET.  A tributary is shorter than its trunk, and a branch of a
    tributary never grows longer than what it branched from.  A single global
    cap on every link (`run_max` alone) gives no trunk/tributary hierarchy --
    every channel is allowed the same length.  The rule is a MONOTONE one:
    length may only shrink going upstream, and a child can never outrun its
    parent.

    So a tip carries (run, budget) for the STREAM it belongs to, not the link:
      * at a junction the MAIN child continues the same stream -- same budget,
        run NOT reset, heading barely changed;
      * the SIDE child starts a new tributary with budget * side_keep (< 1) and
        run = 0, leaving at a real angle.
      * a stream retires when run >= budget.
    Budgets therefore form a decreasing cascade trunk > tributary > sub-
    tributary, which is the visible hierarchy, while individual LINK lengths
    stay roughly constant across orders -- which is what Kawauchi actually
    shows (exterior links mean 10.7 vs interior 9.6, i.e. Horton's law lives in
    how many links a stream has, not in how long each link is).

    The budget and run_max are DIFFERENT constraints and both are needed.  The
    budget alone gets the median unbranched length right (9.0, as real) but
    lets the tail out (p90 23-27, max 46-58 vs real 18.5 / 31), because
    nothing stops a single channel running a long way without branching.  So
    a tip also carries `linkrun`, the length since the LAST junction, which
    resets for both children and forces a split at run_max.

    trend / trend_max.  On a long rectangle the map has a
    GRAIN: the trunk runs the long way and every tributary must join it going
    the same way.  Without a constraint, `main_jit` is a random walk on the
    heading and `side_ang` throws branches up to 60 deg off,
    so over a 12 km map a stream could turn right round and flow back up the
    grain.  Two channels flowing opposite ways cannot share an interfluve, so
    the ridge between them breaks.

    `trend` is the heading GROWTH is allowed to take (i.e. the UPSTREAM
    direction; water flows the other way), and `trend_max` is the largest
    angular deviation from it, applied every step and at every split.  For a
    map draining west -> east, the mouth is on the east edge and trend = pi.
    trend = None (default) applies no constraint.

    budget0 is in units of the map side.

    roots: the map must be consistent within its own frame -- never a window
    cut out of a larger imagined world (otherwise larger maps cannot be
    generated consistently).  So every boundary point is either a divide or an
    outlet.  So the square's own edges
    become the boundary: several roots sit ON the edges, each grows its own
    tree inward, the sep rule keeps the trees apart so each catchment is
    complete and drains through its own edge crossing.  Everything is
    consistent within the frame -- no outside is ever referenced.
    roots = list of (r, c, heading); default is a single bottom-centre root.

    ell defaults to baseline.ELL0 (5.00) -- the two must stay in step, they
    are one calibration of one drainage density.  The corridor does NOT change
    it: it steers growth away from valley floors without changing how much of
    the map gets dissected (see the stop/steer split in the growth loop).

    corridor.  Normally a ridge runs alongside each side of a river.  With the
    same 2 px of elbow room for a river of any size, the trunk gets another
    channel ~66 m away (p90 141 m) and no room for a ridge; the floor cannot
    widen downstream, and the tributary count per km does not thin.

    The fix is to make the exclusion the CHANNEL'S OWN VALLEY rather than a
    constant: a cell is closed to new channels when it lies inside
    hw(y) = 1 + (floor_px - 1) * sqrt(A(y) / acc_ref) of some channel y -- the
    very half-width `solve` then paints as the floor, so generation and
    painting agree on where the valley is instead of contradicting each other.
    It is one continuous function of drainage area, not a second regime bolted
    on downstream (no seams).  corridor scales it; corridor = 0 is the
    constant-sep behaviour.

    Two facts make this the same rule the four transects measured.  Valley
    floor width goes as A^0.49 (ANCOVA, r2=0.56) and tributary spacing along a
    trunk goes as roughly A^0.48 -- their product is constant, i.e. THE SPACING
    BETWEEN TRIBUTARIES IS A FIXED MULTIPLE OF THE VALLEY WIDTH.  So the width
    law and the corridor law are one law, used twice.

    net0 / seed_flanks (the cascade; not used by the baseline).  With `net0` the growth does
    not start from bare roots: it RESTARTS from a network already grown, at a
    finer `ell`.  That makes the unchannelled hollow above a head, and the side
    hollow off a reach, the SAME rule as the channel itself -- one growth rule
    run at successive scales -- instead of a separate extender.
    (`hollow_heads` has no clearance test at all, which is why the baseline
    needs basin labels to keep trees apart.)  Here the whole
    `_walk` contract -- sep, forbid, fold_cap, the budget cascade, the
    branching angles -- applies to a hollow exactly as it applies to a river.

      * every tip of net0 becomes a growing head (the head extension);
      * `seed_flanks` heads per 100 px of existing channel are additionally
        started ON the flanks of links, aimed up the distance gradient (the
        side hollow).  A real tributary joins ALONG a reach, so that is where a
        new link has to be born; seeding it here rather than writing a second
        rule means it obeys `sep` and the budget like everything else.
      * S (distance from the outlet) is recomputed for net0 geodesically from
        the outlets, so fold_cap keeps working across rounds.

    pin_frac_max: stop once the 1-px network covers this fraction of the map.
    THE SIMPLICITY GUARD.  Too dense a network over-constrains the PDE and
    the result is steep terrain.  It is measurable: the Dirichlet band is dilate(net), so this number fixes how
    much of the surface is IMPOSED rather than solved.  The real Kawauchi
    network at n=128 covers 5.8% of the map (17.0% dilated); an uncapped
    cascade reached 13% and its worst gradient was 4x the real one.  None = no
    ceiling (what parameter zero uses)."""
    # shape=(H, W) for a rectangular terrain; n stays the square shorthand.
    # reach = the STREAM BUDGET in px.  For a square it is the side
    # (budget0 * n); for a rectangle it must be the river's own run -- see
    # baseline.hydraulic_reach -- because a budget tied to a side would stop
    # the trunk half way up a long map, or let it wander forever in a short
    # fat one.
    H, W = (int(n), int(n)) if shape is None else (int(shape[0]), int(shape[1]))
    # ARBITRARY DOMAINS (not only rectangles).  `domain` is a bool mask of
    # the LAND; everything else is off the map and never drains, never counts
    # as undissected, and is never walked into.  The first two of those are
    # exactly what `reserve` already does -- it holds ground back from the
    # growth by setting d = dstop = 0 and adding it to `forbid` -- so the mask
    # rides that machinery instead of a second one.  The third needs the hard
    # test in `_walk`, because `forbid` is cleared around every stroke's mouth.
    dom = None if domain is None else np.asarray(domain, bool)
    if dom is not None:
        if dom.shape != (H, W):
            raise ValueError(f"domain {dom.shape} does not match {(H, W)}")
        reserve = ~dom if reserve is None else (np.asarray(reserve, bool) | ~dom)
        if reach is None:
            # the stream budget is the river's own run, so on a shaped map it
            # must be measured INSIDE the land, not across the bounding box.
            m = np.ones((H, W), bool)
            for rc in (roots or []):
                m[int(rc[0]), int(rc[1])] = False
            reach = float(np.where(dom, ndi.distance_transform_edt(m),
                                   0.0).max())
    if reach is None:
        reach = budget0 * max(H, W)
    net = np.zeros((H, W), bool)
    if roots is None:
        roots = [(H - 2, W // 2, -np.pi / 2)]
    # S = distance along the network from the outlet, recorded as each cell is
    # painted.  It is what the no-folding test compares across the plane.
    S = np.full((H, W), np.nan)
    tips = []
    if net0 is None:
        for (r0, c0, a0) in roots:
            net[int(r0), int(c0)] = True
            S[int(r0), int(c0)] = 0.0
            tips.append((float(r0), float(c0), float(a0), 0.0, 0.0,
                         float(reach), 0.0))
    else:
        net = np.asarray(net0, bool).copy()
        S = _restart_S(net, roots)
        tips = _restart_tips(net, S, rng, reach, seed_flanks)
    d = ndi.distance_transform_edt(~net)
    iy, ix = np.mgrid[0:H, 0:W]
    gr, gc = np.gradient(ndi.gaussian_filter(d, 2.0))

    def at(f, r, c):
        return f[int(np.clip(r, 0, H - 1)), int(np.clip(c, 0, W - 1))]

    def keep_trend(a):
        """流向で成長方向を拘束する。trend から trend_max 以上は外さない。"""
        if trend is None:
            return a
        dv = np.arctan2(np.sin(a - trend), np.cos(a - trend))
        return trend + float(np.clip(dv, -trend_max, trend_max))

    def push(tip):
        """Insert a new tip where a stable descending sort would put it."""
        if trend is not None:
            tip = (tip[0], tip[1], keep_trend(tip[2])) + tuple(tip[3:])
        v = look(tip[0], tip[1], tip[2], 2 * ell)
        i = bisect.bisect_right(neg_v, -v)   # appended last => after equals
        tips.insert(i, tip)
        look_v.insert(i, v)
        neg_v.insert(i, -v)

    def look(r, c, a, k):
        """Undrained land k px AHEAD of a tip.  Testing the tip's own cell is
        wrong -- the tip sits on the channel it just cut, so d there is ~0 and
        every head would die at birth."""
        return at(d, r + k * np.sin(a), c + k * np.cos(a))

    forbid, hwc = None, np.zeros((H, W))
    dstop = np.zeros((H, W))
    look_v = []                      # cached `look` per tip (see the refresh)
    neg_v = []                       # -look_v, kept ascending for bisect
    # THE ITERATION BUDGET SCALES WITH AREA (the same scale contract baseline
    # applies to outlets and the carve budget; 3000 at n=128).  An iteration
    # is spent whether the tip advances or merely retires, and steering on the
    # valley field retires many more tips -- so on a 128x1024 strip a fixed
    # 3000 runs out with the map unfilled, which looks like a growth-rule
    # failure but is a budget failure.
    # ...and the corridor field is refreshed a fixed NUMBER OF TIMES over the
    # growth, not once per 12 iterations: the budget above grew with area, and
    # the field costs one distance transform per half-px band of channel, so a
    # fixed cadence would have made a big map pay for it quadratically.  The
    # valleys move slowly compared with a single stroke.  At n=128 the ratio
    # is 1 and the cadence is 12.
    every = 12 * max(1, int(round(H * W / (N * N))))
    for it in range(int(3000 * H * W / (N * N))):
        if it % 12 == 0 or (not tips and net0 is None):   # refresh the field
            # `not tips` does not force a refresh in a RESTART round (net0
            # given).  There the heads deplete early, so the loop spends most
            # of its budget re-seeding ONE head at a time, and rebuilding the
            # whole field for each would dominate the run time.
            # A field up to 12 iterations stale is fine to re-seed against; the
            # network moves by one short stroke per iteration.
            if corridor and (it % every == 0 or forbid is None
                             or (not tips and net0 is None)):
                # UNDRAINED LAND, ONCE VALLEYS HAVE WIDTH.  Distance to the
                # nearest CENTRELINE would say a cell 6 px out on the floor of
                # a 20 px valley is dry land crying out for a channel -- growth
                # would keep planting one there and the trunk would never get
                # a valley of its own.  So d is the distance to the
                # nearest VALLEY: a floor is drained by the river that made it,
                # and growth goes elsewhere.  The same field is the forbidden
                # zone `_walk` refuses to enter.
                accn = tree_accum(net, S) / float(acc_ref or 1.0)
                hwc = float(corridor) * np.where(
                    net, 1.0 + (floor_px - 1.0) * np.sqrt(accn), 0.0)
                forbid, _idx, cf = cone_floor(net, hwc, field=True)
                d = np.maximum(-cf, 0.0)
                # ...but STOP on the centreline distance.  The valley field is
                # the right thing to steer by -- a floor is drained, so do not
                # dig it -- and the wrong thing to STOP by.  Stopping on the
                # valley field leaves the median gap unchanged while the
                # LARGEST gap grows (~38% on a strip), because a fat valley
                # makes d small over a wide area and the whole map passes
                # `d.max() <= 3*ell` while patches of real hillslope are still
                # undissected.  The Poisson term lifts a divide as the SQUARE
                # of its gap, so a 38% wider hole becomes a 1.9x higher smooth
                # dome: too many contours away from the riverbed.
                dstop = ndi.distance_transform_edt(~net)
            elif corridor:
                # between corridor refreshes: the cheap fields still track the
                # network every 12 steps, only the valley cone is held.
                dstop = ndi.distance_transform_edt(~net)
                d = np.maximum(np.minimum(d, dstop), 0.0)
            else:
                d = dstop = ndi.distance_transform_edt(~net)
            if reserve is not None:
                # RESERVED RIDGE GROUND.  Land held for a divide of a coarser
                # level is treated as ALREADY DRAINED: d = 0 there, so no head
                # aims at it and the stop test does not demand it be dissected,
                # and it is forbidden ground to `_walk` like a valley floor.
                # Without both halves the growth would keep trying to serve it
                # and spin out its iteration budget on rejected strokes.
                d = np.where(reserve, 0.0, d)
                dstop = np.where(reserve, 0.0, dstop)
                forbid = reserve.copy() if forbid is None else (forbid |
                                                               reserve)
            if net0 is not None:
                # RESTART ROUNDS: cache the nearest-channel indices with the
                # field.  The re-seed branch below needs them, and it fires on
                # most iterations once the heads have retired, and recomputing
                # a distance transform there would dominate the run time.
                _dn, _nearest = ndi.distance_transform_edt(
                    ~net, return_indices=True)
            gr, gc = np.gradient(ndi.gaussian_filter(d, 2.0))
            # `look` reads `d`, which only changes here, so its value per tip
            # is cached and refreshed with the field instead of being recomputed
            # for every tip on every iteration.  Identical ordering (Python's
            # sort is stable both ways), and it is what makes a RESTART round
            # tractable: there the tip list starts as the whole network's heads,
            # so a full re-evaluation would be O(tips) python calls per step.
            look_v = [look(t[0], t[1], t[2], 2 * ell) for t in tips]
            _order = sorted(range(len(tips)), key=lambda i: -look_v[i])
            tips = [tips[i] for i in _order]
            look_v = [look_v[i] for i in _order]
            neg_v = [-v for v in look_v]
            # stop where the REAL network stops.  Driving max(d) down to ell
            # over-fills: Kawauchi has p50 4.1 with max/p50 ~5.4, i.e. it
            # tolerates ~22 px of undissected land, so the ceiling is 3*ell.
            if dstop.max() <= 3 * ell:
                break
            # the simplicity guard: the pinned band is dilate(net), so this
            # ceiling is what keeps the solve from being mostly Dirichlet
            if pin_frac_max is not None and net.mean() >= pin_frac_max:
                break
        if not tips:
            if not reseed:
                # Undrained land is allowed to remain.  With a flow-direction constraint the
                # re-seed branch is a trap: it aims a new head at the farthest dry
                # point, the clamp turns that heading back into drained land, the
                # head retires at once, and the loop burns its whole budget on
                # distance transforms.  On a deliberately sparse map, land the
                # river cannot reach is a legitimate interfluve, not a failure.
                break
            # every head has retired but land is still undrained: start a new
            # head on the channel nearest the farthest dry point, aimed at it.
            if net0 is not None:
                ny, nx = _nearest
            else:
                _, (ny, nx) = ndi.distance_transform_edt(~net,
                                                         return_indices=True)
            if trend is None:
                fr, fc = np.unravel_index(int(np.argmax(d)), d.shape)
            else:
                # Most of the network (~94% on a strip) is built by this
                # re-seed branch, not by headward growth from the outlet, and
                # each re-seeded head is aimed at the farthest dry point in
                # whatever direction it lies -- which lets streams run back up
                # the grain.  Clamping the heading alone does not work: the
                # head is aimed out of the allowed cone, retires at once, and
                # the loop re-seeds forever.  So the TARGET must respect the
                # trend too: aim only at dry
                # points that lie within trend_max of the trend from the
                # channel that would feed them.
                _rr, _cc = np.indices(d.shape)
                _ang = np.arctan2(_rr - ny, _cc - nx)
                _dev = np.abs(np.arctan2(np.sin(_ang - trend),
                                         np.cos(_ang - trend)))
                _cand = np.where(_dev <= trend_max, d, -1.0)
                if _cand.max() <= 0:
                    break                      # 錐の中に乾いた土地はもう無い
                fr, fc = np.unravel_index(int(np.argmax(_cand)), d.shape)
            sr, sc = float(ny[fr, fc]), float(nx[fr, fc])
            # a re-seeded head is a tributary of whatever it grew from, so it
            # gets a tributary-sized budget, never a trunk-sized one
            s_at = S[int(sr), int(sc)]
            tips = [(sr, sc, float(np.arctan2(fr - sr, fc - sc)), 0.0, 0.0,
                     float(reach) * side_keep,
                     float(s_at) if np.isfinite(s_at) else 0.0)]
            look_v = [look(t[0], t[1], t[2], 2 * ell) for t in tips]
            neg_v = [-v for v in look_v]
        # advance the head facing the most undrained land (ties broken randomly)
        # The list is kept sorted by `look` (descending) instead of being
        # re-sorted from scratch each step.  It is the same order: `look` only
        # changes at a field refresh, where the list IS fully re-sorted, and
        # between refreshes new tips are inserted where a stable sort would put
        # them (after every equal value, since they were appended last).
        # A top-4 selection is NOT a valid shortcut -- the sort also reorders
        # the survivors, and with tied values that order decides the next
        # step's ranking (it would change parameter zero).
        k_pop = int(rng.integers(0, max(1, min(4, len(tips)))))
        r, c, a, run, lrun, bud, sd = tips.pop(k_pop)
        neg_v.pop(k_pop)
        if look_v.pop(k_pop) <= ell:
            continue                          # nothing left ahead: head retires
        if run >= bud:
            continue                          # this stream has run its course
        ga = np.arctan2(at(gr, r, c), at(gc, r, c))     # up the gradient
        a = np.arctan2(0.65 * np.sin(a) + 0.35 * np.sin(ga),
                       0.65 * np.cos(a) + 0.35 * np.cos(ga))
        a = keep_trend(a)
        nstep = int(rng.integers(step[0], step[1]))
        before = net.copy() if fold_cap else None
        out = _walk(net, r, c, a, nstep, rng, wander=0.30,
                    stop_on_hit=True, sep=sep,
                    fold=((S, fold_cap) if fold_cap else None), s_here=sd,
                    forbid=forbid, hw0=at(hwc, r, c) if corridor else 0.0,
                    domain=dom)
        if fold_cap:                     # record S on the cells just painted
            new = net & ~before
            if new.any():
                rr2, cc2 = np.where(new)
                order = np.argsort((rr2 - r) ** 2 + (cc2 - c) ** 2)
                S[rr2[order], cc2[order]] = sd + 1.0 + np.arange(len(order))
        if out is None:
            continue                          # left the grid or joined another
        r, c, a = out
        run += nstep
        lrun += nstep
        sd += nstep                      # distance from the outlet: never resets
        # BRANCHING IS POSITION-DEPENDENT.  Measured, the real network's branching rate rises
        # 17x from headwater to trunk (0.53 -> 8.98 junctions per km of
        # channel, Kawauchi at n=128) while a constant
        # `split` gives 2.6x.  The predictor available at growth time is the
        # stream's REMAINING BUDGET: a point where the stream has bud-run left
        # to run has that much length upstream of it, and Hack's law turns
        # length into area, so x = (bud - run)/reach is a normalised stand-in
        # for "how big will the river be here".  x = 1 at the trunk's mouth,
        # small at every headwater.  split_p = 0 (default) is the constant rate.
        x = max(0.0, bud - run) / float(reach or 1.0)
        if rng.random() < split * (x ** split_p if split_p else 1.0) \
                or lrun >= run_max:
            # MAIN: the same stream carries on -- same budget, stream run kept,
            # but the LINK starts over at the junction
            push((r, c, a + rng.normal(0, main_jit), run, 0.0, bud, sd))
            # SIDE: a new tributary, strictly shorter than the stream it joins.
            # With a trend the map has a grain, and a branch that leaves ACROSS the grain has less room to run
            # before it hits the next valley, so its budget is cut by how far off
            # the trend it goes.  cos(dev) = 1 along the grain, 0.5 at 60 deg.
            sg = rng.choice([-1, 1])
            sa = a + sg * rng.uniform(*side_ang)
            sb = bud * side_keep
            if trend is not None:
                sa = keep_trend(sa)
                dev = np.arctan2(np.sin(sa - trend), np.cos(sa - trend))
                sb *= max(0.25, abs(float(np.cos(dev))))
            push((r, c, sa, 0.0, 0.0, sb, sd))
        else:
            push((r, c, a, run, lrun, bud, sd))
    _FOLD[1] = S.copy(); _FOLD[2] = net.copy()
    skel = skmo.skeletonize(net)
    # labels=True adds the basin raster to the return.  It is the WHOLE of
    # the stage1 -> stage2 contract; everything else stage 2 needs
    # it still derives for itself.  Default False keeps every other caller on
    # the single-array return.
    return (skel, basin_labels(skel, roots, dom)) if labels else skel


def divide_skeleton(net):
    """The divide network of a channel network, as a 1-px skeleton.

    For a non-crossing channel tree the complement is simply connected, so its
    MEDIAL AXIS is the divide: every medial pixel is equidistant from the two
    channels flanking it.
    """
    from skimage import morphology as skmo
    free = ~ndi.binary_dilation(np.asarray(net, bool))
    return skmo.skeletonize(skmo.medial_axis(free) & free)


def reserve_ridges(net, width_px, prev=None, min_len=8):
    """Ground held open for a MAIN RIDGE ("a ridge will come here, so keep
    this ground clear").

    THE PROBLEM IT SOLVES.  A cascade fills the map to one drainage density, so
    every interfluve ends up the same width and no divide is more important
    than any other -- there are sub-ridges everywhere and no main ridge.  In
    real terrain the main divide between two trunk catchments is a continuous
    high spine that the tributaries of both sides stop short of; it is wide
    BECAUSE nothing was allowed to cut it.

    The rule is the ridge counterpart of the valley corridor already in
    `gen_headward`.  A trunk keeps other channels out of its own valley floor;
    here the COARSE network's divide keeps the finer rounds off its crest.  The
    band is `width_px` wide and is taken from the divide of the network as it
    stood at the coarse round, so it is decided by the water that is already
    there, not by a random line -- and it is hierarchical for free: round 0's
    divide gets the widest band, each later round's a narrower one, and the
    union is the main-ridge/sub-ridge structure.

    Only spines longer than `min_len` are kept, so the reservation follows real
    interfluves and not the medial-axis noise at a confluence.

    prev: the reservation from earlier rounds, unioned in."""
    sk = divide_skeleton(net)
    lab, k = ndi.label(sk, structure=np.ones((3, 3)))
    if k:
        keep = np.zeros(k + 1, bool)
        keep[1:] = np.bincount(lab.ravel(), minlength=k + 1)[1:] >= min_len
        sk = keep[lab]
    band = (ndi.distance_transform_edt(~sk) <= float(width_px)) & sk.any()
    return band if prev is None else (band | prev)


def grow_cascade(rng, roots, n=N, ell=5.0, rounds=1, decay=0.65,
                 seed_flanks=25, pin_frac_max=0.058, budget_decay=0.45,
                 ridge_keep=0.0, ridge_levels=1, ridge_min_len=1.5, **kw):
    """The whole network as ONE rule run at successive scales (not used by
    the baseline).

    A hollow is the stage-1 growth rule itself, run again at a finer scale.
    Round 0 is the ordinary headward growth at `ell`;
    each later round restarts from the network already there, at ell*decay**k,
    seeding both the heads and the flanks (see `_restart_tips`).  The stream
    budget shrinks by `budget_decay` per round, so a hollow is short and a
    trunk is long by the same law that already makes a tributary shorter than
    its parent -- `side_keep`, one scale up.

    WHY ONE RULE AND NOT AN EXTENDER.  An extender like `hollow_heads` can
    only LENGTHEN existing heads, so it cannot raise the channel-head count,
    and it has no clearance test, so it merges separate trees.  Restarting the growth
    rule gets the new links, the clearance, the branching angles and the
    no-folding test from the one place they are already written.

    SIMPLICITY.  `pin_frac_max` caps the 1-px network at a
    fraction of the map -- the default 0.058 is what the real Kawauchi network
    covers at n=128 -- so the cascade cannot buy its statistics with density.
    Every round is subject to it, so adding rounds refines the ARRANGEMENT of a
    fixed amount of channel rather than adding more.

    Returns the 1-px network."""
    net = gen_headward(rng, n=n, ell=ell, roots=roots,
                       pin_frac_max=pin_frac_max, **kw)
    reserve = None
    for k in range(1, int(rounds)):
        if pin_frac_max is not None and net.mean() >= pin_frac_max:
            break
        if ridge_keep and k <= ridge_levels:
            # the ridge of the level just finished is reserved before the next,
            # finer level is grown -- widest for round 0 (the main ridge),
            # narrower for each later one (see reserve_ridges).
            # ridge_levels bounds HOW MANY levels get a reservation: with 1
            # only the trunk network's divide is held, which is the main ridge.
            # Reserving every level's divide chokes the growth instead of
            # ranking it -- the channel fraction falls (6.0% -> 3.4%) and the
            # longest crest gets SHORTER, because by the third round most of
            # the map is inside somebody's reservation.
            ell_k = ell * decay ** (k - 1)
            reserve = reserve_ridges(net, ridge_keep * ell_k, prev=reserve,
                                     min_len=ridge_min_len * ell_k)
        net = gen_headward(rng, n=n, ell=ell * decay ** k, roots=roots,
                           net0=net, seed_flanks=seed_flanks, reserve=reserve,
                           reach=(kw.get("reach") or budget0_reach(n, kw))
                           * budget_decay ** k,
                           pin_frac_max=pin_frac_max,
                           **{a: b for a, b in kw.items() if a != "reach"})
    return net


def budget0_reach(n, kw):
    """The stream budget round 0 would have used (see gen_headward)."""
    shape = kw.get("shape")
    H, W = (int(n), int(n)) if shape is None else (int(shape[0]), int(shape[1]))
    return kw.get("budget0", 1.0) * max(H, W)


GENERATORS = {"comb": gen_comb, "bisect": gen_bisect, "headward": gen_headward}


# ---------------------------------------------------------------------- main

def main(fig=False, seeds=(1, 2, 3)):
    ref_net, ref_dem = reference()
    print(HDR)
    t = time.time()
    print(row("REAL kawauchi", score(ref_net, solve(ref_net, tilt=ref_dem)),
              time.time() - t))
    results = {"REAL kawauchi": (ref_net, solve(ref_net, tilt=ref_dem))}
    for name, g in GENERATORS.items():
        acc, secs = [], 0.0
        for sd in seeds:
            t = time.time()
            net = g(np.random.default_rng(sd))
            u = solve(net)
            secs += time.time() - t
            acc.append(score(net, u))
            if sd == seeds[0]:
                results[name] = (net, u)
        avg = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
        for k in ("tips", "pieces", "summits"):
            avg[k] = int(round(avg[k]))
        print(row(name, avg, secs / len(seeds)))

    if fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = list(results)
        f, ax = plt.subplots(2, len(ks), figsize=(4.2 * len(ks), 8.8))
        for j, k in enumerate(ks):
            net, u = results[k]
            ed = ndi.distance_transform_edt(~net)
            p50 = np.median(ed[~net])
            rgb = np.ones(net.shape + (3,))
            rgb[ed > 2 * p50] = (1.0, .85, .45)
            rgb[ed > 3 * p50] = (.95, .35, .15)
            rgb[net] = (.05, .15, .45)
            ax[0, j].imshow(rgb, interpolation="nearest")
            ax[0, j].set_title(k, fontsize=11)
            lv = np.linspace(0, 1, 15)
            ax[1, j].contour(u, levels=lv, colors="#5a4326", linewidths=.6)
            for a in (ax[0, j], ax[1, j]):
                a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
        f.suptitle(f"mini bench, N={N}: valley-placement rules vs the real network",
                   fontsize=13)
        f.tight_layout(rect=[0, 0, 1, 0.95])
        f.savefig(os.path.join(OUT, "mini.png"), dpi=110)
        print("saved valley_samples/mini.png")
    return 0


if __name__ == "__main__":
    sys.exit(main(fig="--fig" in sys.argv))


# ---------------------------------------------------------------- the canvas
# (Not used by the baseline.)  A real map is cut out of a larger landscape,
# so several rivers flow out across its frame (9 for the Kawauchi patch).
# Generating a big canvas and cropping a small one breaks the terrain at the
# big canvas's rim; the approach here is to have no outside at all.
#
# A map with NO OUTSIDE needs every boundary point to be one of exactly two
# things:
#   * a DIVIDE -- water flows inward, so nothing beyond it is ever referenced;
#   * an OUTLET -- one channel crosses and leaves, pinned at its own z.
# A closed curve made of divide arcs joined by a few channel crossings encloses
# a COMPLETE drainage system.  The elevation is then solved on that region
# alone: the working grid's own edge never enters the problem (unlike
# "generate big, crop small", where the solve still runs on the big square
# and its broken rim leaks inward).

def canvas_from_network(net, centre=None, r_lo=0.22, r_hi=0.42, n_ang=720,
                        n_rad=64, lam=3.0):
    """Cut a natural map out of a grown network: follow the divides, cross a
    channel only where the boundary must.

    The boundary is found as a closed curve in polar coordinates about
    `centre`: a dynamic program over (angle, radius) whose per-pixel cost is
    cheap where the distance-to-channel is large (i.e. on a divide) and
    expensive on a channel.  Crossing is allowed, not forbidden -- each
    crossing is an outlet, and a map with no outlets would be an endorheic
    basin, which is not what a real window looks like.

    r_lo/r_hi bound the radius (as a fraction of the grid) so the map comes out
    at roughly the intended size.  Returns (mask, outlets, boundary_rc)."""
    net = np.asarray(net, bool)
    n = net.shape[0]
    if centre is None:
        centre = (n / 2.0, n / 2.0)
    d = ndi.distance_transform_edt(~net)
    cost = np.exp(-d / lam)                 # ~1 on a channel, ->0 on a divide

    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    rad = np.linspace(r_lo * n, r_hi * n, n_rad)
    rr = centre[0] + rad[None, :] * np.sin(ang[:, None])
    cc = centre[1] + rad[None, :] * np.cos(ang[:, None])
    W = ndi.map_coordinates(cost, [rr.ravel(), cc.ravel()], order=1,
                            mode="nearest").reshape(n_ang, n_rad)
    # off-grid samples must never be chosen
    W = np.where((rr < 1) | (rr > n - 2) | (cc < 1) | (cc > n - 2), 1e3, W)

    JUMP = 2                                 # radius may move +-2 per angle step
    best_path, best_cost = None, np.inf
    for start in range(n_rad):
        dp = np.full((n_ang, n_rad), np.inf)
        bk = np.zeros((n_ang, n_rad), int)
        dp[0, start] = W[0, start]
        for a in range(1, n_ang):
            prev = dp[a - 1]
            cand = np.full((2 * JUMP + 1, n_rad), np.inf)
            for k, dj in enumerate(range(-JUMP, JUMP + 1)):
                cand[k] = np.roll(prev, dj)
                if dj > 0:
                    cand[k, :dj] = np.inf
                elif dj < 0:
                    cand[k, dj:] = np.inf
            bi = np.argmin(cand, axis=0)
            dp[a] = cand[bi, np.arange(n_rad)] + W[a]
            bk[a] = np.arange(n_rad) - (bi - JUMP)
        if dp[-1, start] < best_cost:        # must close back on `start`
            best_cost = dp[-1, start]
            path = np.zeros(n_ang, int)
            path[-1] = start
            for a in range(n_ang - 1, 0, -1):
                path[a - 1] = bk[a, path[a]]
            best_path = path
    if best_path is None:
        raise RuntimeError("no closed boundary found")

    br = centre[0] + rad[best_path] * np.sin(ang)
    bc = centre[1] + rad[best_path] * np.cos(ang)
    from matplotlib.path import Path
    yy, xx = np.mgrid[0:n, 0:n]
    mask = Path(np.column_stack([bc, br])).contains_points(
        np.column_stack([xx.ravel(), yy.ravel()])).reshape(n, n)

    ring = np.zeros((n, n), bool)
    ring[np.clip(br.round().astype(int), 0, n - 1),
         np.clip(bc.round().astype(int), 0, n - 1)] = True
    ring = ndi.binary_dilation(ring)
    # the outlets BELONG to the map -- that is where the water leaves -- so the
    # mask keeps the ring, and the outlet cells are the channel cells on it.
    mask = mask | ring
    outlets = ring & net
    return mask, outlets, np.column_stack([br, bc])


def solve_masked(net, mask, outlets, theta=0.3, cap_px=0.006, poisson=0.006):
    """Laplace inside `mask` only.  No cell outside it enters the system.

    INLETS vs OUTLETS.  Sourcing the river profile at EVERY boundary crossing
    makes every crossing an outlet at z = 0, and the map comes out as a basin
    turned inside out: high in the middle, valleys running radially outward.
    A real window is not like
    that -- Kawauchi has a trunk PASSING THROUGH it, so water enters at some
    crossings and leaves at others.

    The network already knows which is which: it was grown from one root, so
    its own distance-from-root and accumulation give every reach a profile.
    Computing that profile on the WHOLE grown network and then cropping to the
    map makes the crossings sort themselves out -- the reach nearest the root
    is the outlet and sits low, the upstream crossings are inlets and sit high.
    The elevation solve still touches only the masked cells, so there is still
    no outside.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    from experiments.harmonic import lipschitz_cap
    net = np.asarray(net, bool)
    mask = np.asarray(mask, bool)
    n = mask.shape[0]
    full = ndi.binary_dilation(net)                 # the WHOLE grown network
    # the network's own root: the cell it was seeded from (bottom centre)
    seed = (n - 2, n // 2)
    if not full[seed]:
        rr, cc = np.where(full)
        k = int(np.argmax(rr))
        seed = (int(rr[k]), int(cc[k]))
    cum, _ = skgraph.MCP_Geometric(np.where(full, 1.0, np.inf)).find_costs([seed])
    s_full = np.where(full & np.isfinite(cum), cum, np.nan)
    ok = full & np.isfinite(s_full)
    if (full & ~ok).any():
        _, (iy, ix) = ndi.distance_transform_edt(~ok, return_indices=True)
        s_full = np.where(full & ~np.isfinite(s_full), s_full[iy, ix], s_full)
    acc = channel_accum(full, np.where(full, s_full, np.inf))
    z_full = profile_slope_area(full, np.where(full, s_full, np.nan), acc,
                                theta=theta)
    if cap_px:
        z_full = np.nan_to_num(lipschitz_cap(np.where(full, z_full, np.nan),
                                             full, cap_px), nan=0.0)
        z_full = enforce_downstream(z_full, full,
                                    np.where(full, s_full, np.nan))
    band = full & mask
    zn = np.where(band, z_full, 0.0)
    zn = np.where(band, zn - zn[band].min(), 0.0)
    zn = zn / (zn.max() or 1.0)

    idx = -np.ones((n, n), int)
    cells = np.column_stack(np.where(mask))
    idx[mask] = np.arange(len(cells))
    rows, cols, vals, rhs = [], [], [], np.zeros(len(cells))
    for k, (r, c) in enumerate(cells):
        if band[r, c]:
            rows.append(k); cols.append(k); vals.append(1.0)
            rhs[k] = zn[r, c]
            continue
        deg = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            j = idx[r + dr, c + dc] if 0 <= r + dr < n and 0 <= c + dc < n else -1
            if j >= 0:
                rows.append(k); cols.append(j); vals.append(-1.0)
                deg += 1
        rows.append(k); cols.append(k); vals.append(float(max(deg, 1)))
        rhs[k] = poisson            # see solve(): lifts the divides
    A = sp.coo_matrix((vals, (rows, cols)), shape=(len(cells),) * 2).tocsr()
    u = np.full((n, n), np.nan)
    sol = spsolve(A, rhs)
    u[mask] = sol
    lo, hi = np.nanmin(u), np.nanmax(u)
    return (u - lo) / (hi - lo + 1e-12)
