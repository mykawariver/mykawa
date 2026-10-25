"""valley_view -- stand on a valley floor and look upstream: one still image.

    python -m tools.valley_view                         # generate(1), the trunk
    python -m tools.valley_view --camera tributary --relief 200 --out v.png

The terrain is generate(seed) at knob zero (128 x 128, 1.5 km square); its
heights are scaled to `--relief` metres.  The camera stands on a channel
cell -- half way up the largest river by default -- with its eye 1.6 m above
the river bed, and looks upstream.

HOW IT IS DRAWN, and why each step is needed (found while drawing it):
  * HEIGHT BETWEEN THE GRID NODES.  Stage 3 ends with a light gaussian
    (sigma 0.9 px), which leaves ~2% of the amplitude at the grid's own
    Nyquist scale: the surface is smooth below 11.7 m, so interpolating it
    adds no detail that is not there.  Linear and cubic-spline interpolation
    give the same view; only the shading (which needs the slope) shows
    linear's facets.  Cubic is the default.
  * VOXEL-SPACE RAY MARCH.  For each screen column, samples go out from the
    eye; the spacing grows with distance like a screen pixel's footprint
    (distance x FOV / width), but never below 4 mm: near the eye a coarser
    step jumps several screen rows per sample and leaves saw-teeth along
    every edge.  A pixel takes the first sample that rises to its row; rows
    between two visible samples get a blend of the two.
  * THE RIVER (drawing only -- the terrain itself is not changed).  The 1-px
    network is cut into reaches (confluence to confluence) on the BFS tree
    from the outlets and each reach is rounded by Chaikin corner cutting,
    ends fixed so reaches still meet.  The distance to that curve is
    computed EXACTLY (nearest of the k nearest segments) -- a distance
    transform on a finer raster still shows its own steps at eye level.  The
    river is cut in as a bowl, 1-3 m wide and 0.4-1.2 m deep by drainage,
    and its slope for the shading is taken analytically from the direction
    to the curve.
  * COLOUR.  Lambert shading, a green-to-ochre tint by height, the river in
    blue (darker where deeper), outlines where a nearer ridge hides farther
    ground, and aerial perspective (far ground towards a pale blue-grey).
  * The image is drawn at 2 x and averaged down.

Cost: ~40 s for 960 x 540 (2 x supersampled, ~5900 samples per column).
Making it fast is still open: most of the samples serve the first few metres.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.baseline import RELIEF, edge_roots, generate
from tools.network_metrics import channel_acc, net_graph

PX_M = 1500.0 / 128.0
EYE = 1.6
SS = 2                              # supersampling: render SS x larger, then average
OUT_W, OUT_H = 960, 540
W_SCR, H_SCR = OUT_W * SS, OUT_H * SS
FOV = np.radians(70.0)
SKY_TOP, SKY_HOR = np.array([0.42, 0.62, 0.86]), np.array([0.83, 0.89, 0.94])
SLOPE = np.array([0.50, 0.46, 0.34])       # steep ground: bare-ish
LOW, HIGH = np.array([0.24, 0.42, 0.18]), np.array([0.55, 0.52, 0.30])
SHALLOW, DEEP_C = np.array([0.36, 0.52, 0.66]), np.array([0.16, 0.30, 0.46])
FAR = np.array([0.62, 0.70, 0.80])
CAMERA = {"trunk": (0.35, 0.55), "tributary": (0.06, 0.12)}


def chaikin(P, n=4):
    """Chaikin corner cutting, n rounds, end points fixed."""
    for _ in range(n):
        if len(P) < 3:
            return P
        Q = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            Q.append(0.75 * a + 0.25 * b)
            Q.append(0.25 * a + 0.75 * b)
        Q.append(P[-1])
        P = np.array(Q)
    return P


def thalweg_segments(net, roots):
    """The network as smooth curves: (start, end) points of short segments
    [px], one Chaikin-rounded polyline per reach of the BFS tree."""
    rr, cc, idx, G = net_graph(net)
    N = len(rr)
    R = [idx[r, c] for (r, c, *_rest) in roots if idx[r, c] >= 0]
    Gu = G + G.T
    one = np.ones(len(R))
    Gu = sparse.bmat([[Gu, sparse.coo_matrix((one, (R, [0] * len(R))),
                                             shape=(N, 1))],
                      [sparse.coo_matrix((one, ([0] * len(R), R)),
                                         shape=(1, N)), None]]).tocsr()
    _, pred = csgraph.breadth_first_order(Gu, N, directed=False,
                                          return_predecessors=True)
    kids = [[] for _ in range(N)]
    for v in range(N):
        if 0 <= pred[v] < N:
            kids[pred[v]].append(v)
    SA, SB = [], []
    for v in range(N):
        p = pred[v]
        if not (p == N or (0 <= p < N and len(kids[p]) != 1)):
            continue                                   # v does not start a reach
        chain = [] if p == N else [p]
        k = v
        while True:
            chain.append(k)
            if len(kids[k]) != 1:
                break
            k = kids[k][0]
        P = chaikin(np.column_stack([rr[chain], cc[chain]]).astype(float), 4)
        SA.append(P[:-1]); SB.append(P[1:])
    return np.concatenate(SA), np.concatenate(SB)


def render(seed=1, relief_m=300.0, camera="trunk", order=3):
    u, net, meta = generate(seed)
    z = u / RELIEF * relief_m                  # metres
    H, W = z.shape
    rng = np.random.default_rng(seed)          # generate()'s outlets, redrawn
    roots = edge_roots(rng, H, W, int(rng.integers(3, 6)))
    rr, cc, idx, G, acc = channel_acc(net, roots)

    # camera: a channel cell in the chosen drainage band, looking upstream
    lo, hi = CAMERA[camera]
    cand = [k for k in np.argsort(acc) if lo * acc.max() < acc[k] < hi * acc.max()]
    k0 = cand[len(cand) // 2]
    r0, c0 = rr[k0], cc[k0]
    path = [(r0, c0)]
    k = k0
    for _ in range(12):                        # follow the largest upstream neighbour
        nb = [idx[r, c] for r in range(rr[k] - 1, rr[k] + 2)
              for c in range(cc[k] - 1, cc[k] + 2)
              if 0 <= r < H and 0 <= c < W and idx[r, c] >= 0 and idx[r, c] != k]
        nb = [j for j in nb if acc[j] < acc[k]]
        if not nb:
            break
        k = max(nb, key=lambda j: acc[j])
        path.append((rr[k], cc[k]))
    dy, dx = np.array(path[-1], float) - np.array(path[0], float)
    heading = np.arctan2(dy, dx)

    coef = ndi.spline_filter(z, order=3) if order == 3 else z

    def height(py, px):
        return ndi.map_coordinates(coef, [py, px], order=order, mode="nearest",
                                   prefilter=False)

    accmap = np.zeros(z.shape); accmap[rr, cc] = acc
    _, (iy, ix) = ndi.distance_transform_edt(~net, return_indices=True)
    SA, SB = thalweg_segments(net, roots)
    seglen = np.hypot(*(SB - SA).T)
    tree = cKDTree(0.5 * (SA + SB))
    a_rel = np.sqrt(accmap[iy, ix] / acc.max())
    half = ndi.gaussian_filter(0.5 + 1.0 * a_rel, 1.0)    # river half-width [m]
    deep = ndi.gaussian_filter(0.4 + 0.8 * a_rel, 1.0)    # river depth [m]
    reach = half.max() / PX_M + 0.5 * seglen.max() + 0.5  # px: beyond this, no river

    def thalweg(py, px, k=6):
        """Exact distance [m] from map points [px] to the thalweg, and the
        unit vector (y, x) from the nearest curve point; +inf far away."""
        pts = np.column_stack([py, px])
        d1, _ = tree.query(pts, k=1)
        near = d1 < reach
        dist = np.full(len(pts), np.inf)
        uy = np.zeros(len(pts)); ux = np.zeros(len(pts))
        if near.any():
            q = pts[near]
            _, cn = tree.query(q, k=k)
            a = SA[cn]; b = SB[cn]
            ab = b - a
            t = np.clip(((q[:, None, :] - a) * ab).sum(-1)
                        / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
            dv = q[:, None, :] - (a + t[..., None] * ab)
            dd = np.hypot(dv[..., 0], dv[..., 1])
            j = dd.argmin(1)
            m = np.arange(len(q))
            dmin = dd[m, j]
            dist[near] = dmin * PX_M
            vec = dv[m, j] / np.maximum(dmin, 1e-9)[:, None]
            uy[near] = vec[:, 0]; ux[near] = vec[:, 1]
        return dist, uy, ux

    def river(py, px):
        dl, uy, ux = thalweg(py, px)
        hw = ndi.map_coordinates(half, [py, px], order=1, mode="nearest")
        dp = ndi.map_coordinates(deep, [py, px], order=1, mode="nearest")
        return dl, hw, dp, uy, ux

    cy, cx = float(r0), float(c0)
    dl0, hw0, dp0, _, _ = river(np.array([cy]), np.array([cx]))
    cam_z = (height(np.array([cy]), np.array([cx]))
             - dp0 * np.clip(1 - (dl0 / hw0) ** 2, 0, None))[0] + EYE
    f = (W_SCR / 2) / np.tan(FOV / 2)
    horizon = H_SCR / 2

    d = [0.5]
    while d[-1] < 2500:
        d.append(d[-1] + max(0.004, d[-1] * FOV / OUT_W))
    d = np.array(d)
    ang = heading + np.arctan((np.arange(W_SCR) - W_SCR / 2 + 0.5) / f)
    t0 = time.time()
    corr = np.cos(ang - heading)[:, None]            # fish-eye correction
    dist_px = d[None, :] / PX_M / corr
    sy = cy + dist_px * np.sin(ang)[:, None]
    sx = cx + dist_px * np.cos(ang)[:, None]
    inside = (sy >= 0) & (sy <= H - 1) & (sx >= 0) & (sx <= W - 1)
    e = 0.05                                         # px
    fy, fx = sy.ravel(), sx.ravel()
    dl, hw_, dp_, uy_, ux_ = river(fy, fx)
    cut = np.clip(1 - (dl / hw_) ** 2, 0, None)
    inb = dl < hw_
    hz = (height(fy, fx) - dp_ * cut).reshape(sy.shape)
    with np.errstate(invalid="ignore"):          # dl is +inf away from the river
        gy = ((height(fy + e, fx) - height(fy - e, fx)) / (2 * e * PX_M)
              + np.where(inb, 2 * dp_ * dl * uy_ / hw_ ** 2, 0)).reshape(sy.shape)
        gx = ((height(fy, fx + e) - height(fy, fx - e)) / (2 * e * PX_M)
              + np.where(inb, 2 * dp_ * dl * ux_ / hw_ ** 2, 0)).reshape(sy.shape)
    dl = dl.reshape(sy.shape); hw_ = hw_.reshape(sy.shape)
    print("terrain sampled in %.1f s (%d samples per column)"
          % (time.time() - t0, len(d)), flush=True)

    ys = horizon - (hz - cam_z) / d[None, :] * f
    ys = np.where(inside, ys, np.inf)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.array([-0.5, 0.6, 0.62]); L /= np.linalg.norm(L)
    lam = np.clip((n * L).sum(-1), 0, 1)
    steep = np.clip((np.hypot(gx, gy) - 0.25) / 0.5, 0, 1)[..., None]
    hf = np.clip((hz - z.min()) / (z.max() - z.min()), 0, 1)[..., None]
    base = (LOW * (1 - hf) + HIGH * hf) * (1 - steep) + SLOPE * steep
    wet = (dl < hw_)[..., None]
    wdep = np.clip(1 - (dl / hw_) ** 2, 0, 1)[..., None]
    base = np.where(wet, SHALLOW * (1 - wdep) + DEEP_C * wdep, base)
    col = base * (0.40 + 0.75 * lam[..., None])

    img = np.zeros((H_SCR, W_SCR, 3))
    dpix = np.full((H_SCR, W_SCR), np.inf)
    rows = np.arange(H_SCR)[:, None]
    img[:] = (SKY_TOP + (SKY_HOR - SKY_TOP) * np.clip(rows / horizon, 0, 1))[:, None, :]
    yy = np.arange(H_SCR)
    for c in range(W_SCR):
        cm = np.minimum.accumulate(ys[c])                 # occlusion horizon so far
        kfirst = np.searchsorted(-cm, -yy, side="left")   # first sample reaching row y
        vis = kfirst < len(d)
        img[yy[vis], c] = col[c, kfirst[vis]]
        k = kfirst[vis]
        yv = yy[vis]
        ok = k > 0
        kk = np.where(ok, k - 1, 0)
        span = cm[kk] - cm[k]
        ok &= (ys[c, kk] <= cm[kk] + 1e-9) & (span > 1)
        w = np.clip((yv - cm[k]) / np.where(span > 0, span, 1), 0, 1)[:, None]
        img[yv[ok], c] = (col[c, k] * (1 - w) + col[c, kk] * w)[ok]
        dpix[yy[vis], c] = d[kfirst[vis]]
    ground = np.isfinite(dpix)
    sil = np.zeros((H_SCR, W_SCR), bool)
    with np.errstate(invalid="ignore"):
        for s0, s1 in (((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                       ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
            g = ground[s0] & ground[s1]
            ratio = np.maximum(dpix[s0], dpix[s1]) / np.minimum(dpix[s0], dpix[s1])
            sil[s0] |= g & (ratio > 1.25)
    img = np.where(sil[..., None], img * 0.40, img)
    ga = np.clip(np.where(ground, dpix, 0) / 1400.0, 0, 1)[..., None] ** 0.7
    img = np.where(ground[..., None], img * (1 - ga) + FAR * ga, img)
    img = np.clip(img, 0, 1).reshape(OUT_H, SS, OUT_W, SS, 3).mean(axis=(1, 3))
    return img, dict(z=z, net=net, cam=(cy, cx), heading=heading)


def save(img, info, out, seed, relief_m, order):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z, net = info["z"], info["net"]
    (cy, cx), heading = info["cam"], info["heading"]
    H, W = z.shape
    fig = plt.figure(figsize=(OUT_W / 100, OUT_H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.axis("off")
    ia = fig.add_axes([0.80, 0.62, 0.19, 0.36])      # the map, camera and view
    ia.imshow(z, cmap="terrain", interpolation="nearest")
    ry, rx = np.nonzero(net); ia.scatter(rx, ry, s=0.3, c="#1f4fa0", linewidths=0)
    for a in (heading - FOV / 2, heading + FOV / 2):
        ia.plot([cx, cx + 60 * np.cos(a)], [cy, cy + 60 * np.sin(a)], "r-", lw=0.8)
    ia.plot(cx, cy, "ro", ms=3); ia.set_xticks([]); ia.set_yticks([])
    ia.set_xlim(0, W - 1); ia.set_ylim(H - 1, 0)
    fig.text(0.01, 0.01, f"generate({seed}), 1.5 km square, relief {relief_m:.0f} m "
             f"(assumed), eye {EYE} m on the river bed, FOV {np.degrees(FOV):.0f} deg, "
             f"{'cubic spline' if order == 3 else 'linear'} interpolation, river drawn "
             "1-3 m wide, 0.4-1.2 m deep (drawing only)", fontsize=8, color="white")
    fig.savefig(out, dpi=100)
    print("saved", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--relief", type=float, default=300.0, help="relief [m]")
    ap.add_argument("--camera", default="trunk", choices=sorted(CAMERA))
    ap.add_argument("--order", type=int, default=3, choices=[1, 3],
                    help="interpolation: 1 linear, 3 cubic spline")
    ap.add_argument("--out", default=None,
                    help="output PNG (default: valley_view_s<seed>_<camera>.png)")
    a = ap.parse_args(argv)
    img, info = render(a.seed, a.relief, a.camera, a.order)
    save(img, info, a.out or f"valley_view_s{a.seed}_{a.camera}.png", a.seed,
         a.relief, a.order)
    return 0


if __name__ == "__main__":
    sys.exit(main())
