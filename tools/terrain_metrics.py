"""terrain_metrics -- numbers for what looks wrong on big maps (no reference DEM).

    from tools.terrain_metrics import measure, report
    m = measure(u, net, roots)      # u: elevations, net: 1-px channels
    print(report(m))

    python -m tools.terrain_metrics            # 128x128 and 3 x 12 km, seed 1

Five features, each turned into numbers.  Areas are in km2 (11.72 m per px),
heights in fractions of the map's relief (the generator normalises every map
to the same span, so metres would mean nothing yet).

1. bumps     closed-contour knobs: summits whose prominence exceeds h, per
             km2, for h = 2.5 % and 5 % of the relief (5 % = one contour of
             the 20 in tools/long_map).  Real dissected uplands: crests are
             continuous, summits sit on them, so the count stays low.
2. hierarchy (a) relief in an L x L window, R(L), and its log-log slope over
             the large windows (>= 32 px).  A landscape that is one 128 px
             tile repeated saturates there (slope ~ 0).
             (b) divide height against the size of the rivers the divide
             separates: at levels A_k, the channel network is cut into links
             of rivers >= A_k, every cell joins the link its slope drains to,
             and the cells where two links meet are the divides.  Their median
             height above their own river, per A_k, and the log-log slope of
             that.  Big rivers should be split by high ridges.
3. floor     valley floor width against drainage area: floor = flat ground
             (slope < 20 % of the median slope) at most 1 % of the relief
             above the nearest channel, connected to it.  Width per channel
             cell = floor cells nearest to it.  Fitted exponent and the width
             at fixed areas (comparable across map sizes).
4. direction sinuosity of the channels against drainage area: path length /
             straight distance over n = 8, 32, 128 steps downstream.  Large
             rivers should not wander at large n.
5. simplicity roughness of the ground against the size of the river it
             drains to: std of u - gaussian(u, 2 px) in 5 x 5 px, and the
             channel density within 9 x 9 px.  Should fall downstream.

Everything is vectorised (pointer jumping on the drainage tree), a few
seconds on 3 x 12 km.
"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.flood import depth_of, catchment_sum
from tools.network_metrics import net_graph

PX_M = 1500.0 / 128.0
KM2 = (PX_M / 1000.0) ** 2            # km2 per cell
from scipy import sparse
from scipy.sparse import csgraph


# --------------------------------------------------------------------------
# drainage tree on the drawn network
# --------------------------------------------------------------------------
def network_tree(net, roots):
    """Parent (in network-node indices, -1 at the outlets) of the BFS tree of
    the 1-px network from the outlets, plus rr, cc, idx and the edge length
    to the parent (1 or sqrt 2)."""
    rr, cc, idx, G = net_graph(net)
    N = len(rr)
    R = [idx[int(r), int(c)] for (r, c, *_x) in roots if idx[int(r), int(c)] >= 0]
    Gu = G + G.T
    one = np.ones(len(R))
    Gu = sparse.bmat([[Gu, sparse.coo_matrix((one, (R, [0] * len(R))), shape=(N, 1))],
                      [sparse.coo_matrix((one, ([0] * len(R), R)), shape=(1, N)), None]]).tocsr()
    _o, pred = csgraph.breadth_first_order(Gu, N, directed=False,
                                           return_predecessors=True)
    parent = pred[:N].astype(np.int64)
    parent[(parent == N) | (parent < 0)] = -1
    p = np.where(parent >= 0, parent, np.arange(N))
    step = np.hypot(rr - rr[p], cc - cc[p])
    return rr, cc, idx, parent, step


def _jump_until(parent, stop):
    """For every node, the first node at or below it (towards the outlet)
    where stop is True (or the outlet)."""
    n = len(parent)
    f = np.where(stop | (parent < 0), np.arange(n), parent)
    while True:
        g = f[f]
        if np.array_equal(g, f):
            return f
        f = g


def _ancestor(parent, k):
    a = np.arange(len(parent))
    p = np.where(parent >= 0, parent, np.arange(len(parent)))
    for _ in range(k):
        a = p[a]
    return a


def _path_dist(parent, step):
    """Distance along the tree from each node to its outlet."""
    anc = parent.copy()
    d = np.where(parent >= 0, step, 0.0)
    while True:
        live = anc >= 0
        if not live.any():
            return d
        a = anc[live]
        d[live] = d[live] + d[a]
        anc[live] = anc[a]


def _bins(x, y, edges, stat=np.median, minn=20):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (x >= lo) & (x < hi)
        out.append(float(stat(y[k])) if k.sum() >= minn else np.nan)
    return np.array(out)


def _loglog_slope(x, y):
    k = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if k.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(x[k]), np.log(y[k]), 1)[0])


AREA_EDGES = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100])   # km2
LEVELS = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0])                # km2


def measure(u, net, roots, meters=False):
    u = np.asarray(u, float)
    net = np.asarray(net, bool)
    H, W = u.shape
    relief = float(u.max() - u.min())
    # meters=True: u is in metres and every height below stays in metres;
    # otherwise heights are fractions of the map's relief
    z = (u - u.min()) if meters else (u - u.min()) / relief
    area_km2 = H * W * KM2
    out = dict(shape=(H, W), area_km2=area_km2, meters=meters, relief=relief)
    BUMP_H = (5.0, 10.0) if meters else (0.025, 0.05)
    FLOOR_H = 2.0 if meters else 0.01

    rr, cc, idx, parent, step = network_tree(net, roots)
    # every cell joins the channel cell its water reaches first (routed on
    # the surface by a priority flood from the outlets; nearest channel cell
    # where the route never meets the network); drainage area accumulates
    # those allocations down the tree
    from experiments.flood import flood_tree
    _d, (ir, ic) = ndi.distance_transform_edt(~net, return_indices=True)
    nearest = idx[ir, ic]
    par = flood_tree(z, roots)
    stop = net.ravel()
    hit = _jump_until(par, stop)
    near = np.where(stop[hit], idx.ravel()[hit], nearest.ravel()).reshape(H, W)
    local = np.bincount(near.ravel(), minlength=len(rr)).astype(float)
    depth = depth_of(parent)
    area = catchment_sum(parent, depth, local) * KM2   # km2, per node
    zc = z[rr, cc]
    A_cell = area[near]                                 # area of the river a cell drains to

    # ---- 1. bumps ------------------------------------------------------
    from skimage.morphology import h_maxima
    out["bumps"] = {}
    for h in BUMP_H:
        lab, n = ndi.label(h_maxima(z, h), structure=np.ones((3, 3)))
        out["bumps"][h] = n / area_km2
    # ISOLATED knobs: summits whose prominence is at least half their height
    # above the river they drain to -- a knob standing on its own rather
    # than a summit on a crest.  Prominence is bracketed by a ladder of h.
    ladder = BUMP_H[0] * 2.0 ** np.arange(8)
    lab0, n0 = ndi.label(h_maxima(z, ladder[0]), structure=np.ones((3, 3)))
    prom = np.full(n0 + 1, ladder[0])
    for h in ladder[1:]:
        ids = np.unique(lab0[h_maxima(z, h).astype(bool)])
        prom[ids[ids > 0]] = h
    pk = ndi.maximum_position(z, lab0, np.arange(1, n0 + 1))
    hand_pk = np.array([z[p] - zc[near[p]] for p in pk]) if n0 else np.zeros(0)
    pr = prom[1:]
    iso = (pr >= BUMP_H[1]) & (pr >= 0.5 * np.maximum(hand_pk, 1e-9))
    out["isolated"] = float(iso.sum() / area_km2)

    # ---- 2a. relief vs window -------------------------------------------
    Ls, R = [], []
    L = 4
    while L <= min(H, W):
        mx = ndi.maximum_filter(z, size=L, mode="nearest")
        mn = ndi.minimum_filter(z, size=L, mode="nearest")
        h2 = L // 2
        sub = (mx - mn)[h2:H - h2 or None, h2:W - h2 or None] if min(H, W) > L else (mx - mn)
        Ls.append(L); R.append(float(sub.mean()))
        L *= 2
    Ls, R = np.array(Ls, float), np.array(R)
    out["R_L"] = dict(zip(Ls.astype(int).tolist(), R.round(4).tolist()))
    big = Ls >= 32
    out["R_slope_small"] = _loglog_slope(Ls[Ls <= 16], R[Ls <= 16])
    out["R_slope_large"] = _loglog_slope(Ls[big], R[big]) if big.sum() >= 2 else np.nan

    # ---- 2b. divides vs river size ---------------------------------------
    div_h = []
    for Ak in LEVELS:
        on = area >= Ak
        if on.sum() < 10 or (~on).all():
            div_h.append(np.nan); continue
        entry = _jump_until(parent, on)                # first node >= Ak below
        # junctions at this level: >= 2 children that are on
        kid = (parent >= 0) & on
        nkids = np.bincount(parent[kid], minlength=len(rr))
        stop_link = np.zeros(len(rr), bool)
        pj = parent >= 0
        stop_link[pj] = (nkids[parent[pj]] >= 2)
        stop_link |= ~pj
        link = _jump_until(parent, stop_link | ~on)
        lab = link[entry[near]]
        hand = z - zc[entry[near]]
        d = np.zeros((H, W), bool)
        d[:, :-1] |= lab[:, :-1] != lab[:, 1:]
        d[:-1, :] |= lab[:-1, :] != lab[1:, :]
        d &= ~net
        div_h.append(float(np.median(hand[d])) if d.sum() > 20 else np.nan)
    div_h = np.array(div_h)
    # HAND_SLOPE: the same levels, but over EVERY land cell -- the mean height
    # of the ground above the river of size >= A_k it drains to.  Stable
    # where the divide medians rest on a few hundred cells.
    hand_m = []
    for Ak in LEVELS:
        on = area >= Ak
        if on.sum() < 10 or (~on).all():
            hand_m.append(np.nan); continue
        entry = _jump_until(parent, on)
        hand_m.append(float(np.mean(z - zc[entry[near]])))
    hand_m = np.array(hand_m)
    out["hand_mean"] = dict(zip(LEVELS.tolist(), np.round(hand_m, 4).tolist()))
    k3 = LEVELS <= 3.0
    out["hand_slope"] = _loglog_slope(LEVELS[k3], hand_m[k3])
    out["divide_h"] = dict(zip(LEVELS.tolist(), np.round(div_h, 4).tolist()))
    out["divide_slope"] = _loglog_slope(LEVELS, div_h)

    # ---- 2c. ridges with room: small valleys square to the big divides, and
    # ground left without channels ----------------------------------------
    # ORTH: for every channel node of a small river (0.01-0.3 km2), the
    # cosine between its flow over 4 steps downstream and the downslope
    # direction of the ground smoothed at 6 px (70 m) -- the slope of the
    # ridges and big valley sides, with the small valleys themselves blurred
    # out.  1 = every small valley runs straight down off its ridge;
    # ~0 = they wander across it.  (A version measured against the
    # direction away from the nearest big divide did not respond: the link
    # boundaries of 2b also cut across valleys at the junctions.)
    a4 = _ancestor(parent, 4)
    fy, fx = (rr[a4] - rr).astype(float), (cc[a4] - cc).astype(float)
    sy, sx = np.gradient(ndi.gaussian_filter(z, 6.0))
    gy_, gx_ = -sy[rr, cc], -sx[rr, cc]
    nf, ng = np.hypot(fy, fx), np.hypot(gy_, gx_)
    k = (area >= 0.01) & (area < 0.3) & (nf > 0) & (ng > 0) & (depth >= 4)
    out["orth"] = float(np.mean((fy * gy_ + fx * gx_)[k] / (nf * ng)[k])) \
        if k.sum() > 20 else np.nan
    # GAP: how far land gets from any channel -- the 99th percentile of the
    # land-to-channel distance, in px, and its ratio to the median.  A
    # uniform network keeps it near 3-4 x the median; ridges with room
    # around their crests raise it.
    dch = ndi.distance_transform_edt(~net)[~net]
    out["gap99"] = float(np.percentile(dch, 99))
    out["gap_ratio"] = float(np.percentile(dch, 99) / max(np.median(dch), 1e-9))

    # ---- 3. valley floor width vs area --------------------------------------
    gy, gx = np.gradient(z)
    slope = np.hypot(gy, gx)
    flat = (slope < 0.2 * np.median(slope)) & (z - zc[near] < FLOOR_H)
    lab, _n = ndi.label(flat | net, structure=np.ones((3, 3)))
    keep = np.unique(lab[net])
    floor = np.isin(lab, keep[keep > 0])
    width = np.bincount(near[floor], minlength=len(rr)).astype(float)
    wb = _bins(area, width, AREA_EDGES, np.mean)
    out["floor_w"] = dict(zip(AREA_EDGES[:-1].tolist(), np.round(wb, 2).tolist()))
    k = area >= 0.1
    out["floor_exp"] = _loglog_slope(area[k], np.maximum(width[k], 1e-9)) \
        if k.sum() > 20 else np.nan

    # ---- 4. sinuosity vs area ----------------------------------------------
    D = _path_dist(parent, step)
    out["sinuosity"] = {}
    for n in (8, 32, 128):
        a = _ancestor(parent, n)
        ok = depth >= n
        straight = np.hypot(rr - rr[a], cc - cc[a])
        s = np.where(ok & (straight > 0), (D - D[a]) / np.maximum(straight, 1e-9), np.nan)
        sb = _bins(area[ok], s[ok], AREA_EDGES, np.median)
        out["sinuosity"][n] = dict(zip(AREA_EDGES[:-1].tolist(), np.round(sb, 3).tolist()))

    # ---- 5. simplicity vs area of the river drained to -------------------
    res = z - ndi.gaussian_filter(z, 2.0)
    m1 = ndi.uniform_filter(res, 5); m2 = ndi.uniform_filter(res * res, 5)
    rough = np.sqrt(np.maximum(m2 - m1 * m1, 0))
    dens = ndi.uniform_filter(net.astype(float), 9)
    off = ~net
    out["rough"] = dict(zip(AREA_EDGES[:-1].tolist(),
                            np.round(_bins(A_cell[off], rough[off], AREA_EDGES, np.mean) * (1.0 if meters else 1e3), 3).tolist()))
    out["chan_dens"] = dict(zip(AREA_EDGES[:-1].tolist(),
                                np.round(_bins(A_cell[off], dens[off], AREA_EDGES, np.mean), 3).tolist()))
    out["trunk_km2"] = float(area.max())
    return out


def report(m):
    H, W = m["shape"]
    unit = "m" if m["meters"] else "of relief"
    lines = [f"{H} x {W} px = {m['area_km2']:.1f} km2, largest river {m['trunk_km2']:.1f} km2, "
             f"relief {m['relief']:.3g}" + (" m" if m["meters"] else "")]
    lines.append("1 bumps / km2         " + "   ".join(
        f"prom>{h:g}{'m' if m['meters'] else ''}: {v:.2f}" for h, v in m["bumps"].items())
        + f"   isolated: {m['isolated']:.2f}")
    lines.append("2a relief R(L)        " + "  ".join(f"{k}:{v:.3f}" for k, v in m["R_L"].items())
                 + f"   slope small {m['R_slope_small']:.2f} large {m['R_slope_large']:.2f}")
    lines.append("2b divide height      " + "  ".join(f"{k:g}:{v:.3f}" for k, v in m["divide_h"].items())
                 + f"   slope {m['divide_slope']:.2f}")
    lines.append(f"2c ridges             orth {m['orth']:.2f}   gap99 {m['gap99']:.1f} px"
                 f"   gap99/median {m['gap_ratio']:.2f}")
    hdr = "   ".join(f"{a:g}" for a in AREA_EDGES[:-1])
    lines.append(f"   area bins (km2 >=) {hdr}")
    def row(name, d, fmt="{:.2f}"):
        return f"{name:<22}" + "  ".join("  -  " if not np.isfinite(v) else fmt.format(v)
                                          for v in d.values())
    lines.append(row("3 floor width px", m["floor_w"]) + f"   exp {m['floor_exp']:.2f}")
    for n, d in m["sinuosity"].items():
        lines.append(row(f"4 sinuosity n={n}", d))
    lines.append(row("5 roughness " + ("m" if m["meters"] else "x1e3"), m["rough"]))
    lines.append(row("5 channel density", m["chan_dens"], "{:.3f}"))
    return "\n".join(lines)


def main():
    import time
    from experiments.baseline import generate
    cases = [("128 x 128, seed 1", dict(), None)]
    H, W = 256, 1024
    roots = [(H // 2, W - 1, np.pi)]
    for s1 in ("flood", "headward"):
        cases.append((f"3 x 12 km {s1}, seed 1",
                      dict(shape=(H, W), outlets=roots, stage1=s1), roots))
    for name, kw, roots_ in cases:
        t = time.time()
        u, net, meta = generate(1, **kw)
        tg = time.time() - t
        if roots_ is None:
            from experiments.baseline import edge_roots
            rng = np.random.default_rng(1)
            n = u.shape[0]
            k = int(round(int(rng.integers(3, 6)) * (2 * n) / (2 * 128)))
            roots_ = edge_roots(rng, n, n, max(1, k))
        t = time.time()
        m = measure(u, net, roots_)
        print(f"== {name}  (generate {tg:.0f} s, measure {time.time() - t:.1f} s)")
        print(report(m), flush=True)


if __name__ == "__main__":
    main()
