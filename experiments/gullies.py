"""gullies -- a PROTOTYPE: add gullies to the network before stage 2, down to
a target spacing, so that the fineness of the dissection is one adjustable
number.

    from experiments.gullies import generate
    u, net, meta = generate(1, reach=8.0)              # 128 x 128
    u, net, meta = generate(1, reach=8.0, shape=(256, 1024),
                            outlets=[(128, 1023, np.pi)])

reach [px]: no cell is left farther than this from a channel.  None = off
(plain generate_scaled).  Smaller = finer; below about 6 px the valleys are
too close to tell apart at 11.7 m per px and nothing more is gained.

WHY.  Measured without regard to the height of the relief (contour-curvature
valley lines, tools/terrain_metrics.valley_lines), real hills in southern
Boso carry 6.4-6.9 km of valley line per km2; scaled.py carries 2.7-4.0.
Stage 3 does not change that number: its cuts are carved into a slope that
is already there, as deep as that slope is tall at most, and then smoothed.
A gully added BEFORE stage 2 is a channel like any other: it is pinned on
its own long profile and the ridges rise between it and its neighbours.

NOT FINISHED (see HANDOFF, 2026-11-15, "刻みの細かさ"):
  * the gullies run straight through the main ridges that ridge_room keeps
    clear, so the broad main ridges are lost;
  * small knobs double (ridge scraps between gullies);
  * the farthest cell is found by a whole-map distance transform per gully
    (3 x 12 km: +4 s at reach 10, +12 s at reach 6); stage 3's _TopBlocks
    would make it local;
  * with gullies this dense stage 3 has little left to do.
"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiments.scaled as S
from experiments.flood import noise

PX = S.PX_M
def add_gullies(net, roots, rng, reach=4.0, sc=0.3, amp=6.0, sigma=1.5, head_gap=1, max_steps=200, min_len=3):
    """Until no cell is farther than `reach` px from a channel: take the
    farthest cell, run down the provisional surface (the network's own
    profile + sc per metre of distance + a little noise) to the network, and
    add the path (less its first head_gap cells) as a channel."""
    from skimage.draw import line
    from skimage import morphology as skmo
    from experiments.mini import rejoin_after_thinning
    net = np.asarray(net, bool).copy(); H, W = net.shape
    rr, cc, idx, parent, step, area = S.channel_area(net, roots)
    z = S.integrate_profile(parent, step, S.slope_of(area))
    zmap = np.zeros((H, W)); zmap[rr, cc] = z
    d, (iy, ix) = ndi.distance_transform_edt(~net, return_indices=True)
    nz = noise(rng, H, W, smin=4.0, beta=0.5, fast=True); nz = (nz - nz.mean()) / (nz.std() or 1.0)
    zp = ndi.gaussian_filter(zmap[iy, ix] + sc * d * PX + amp * nz, sigma)
    gy, gx = np.gradient(zp); gn = np.hypot(gy, gx) + 1e-12
    out = net.copy(); n_add = 0
    for _ in range(H * W):
        k = int(np.argmax(d))
        if d.flat[k] <= reach: break
        r0, c0 = divmod(k, W); y, x = float(r0), float(c0); pts = [(r0, c0)]; ok = False
        for _s in range(max_steps):
            ri, ci = int(round(y)), int(round(x))
            if out[ri, ci] and len(pts) > 1: ok = True; break
            g = gn[ri, ci]; ny, nx = y - gy[ri, ci] / g, x - gx[ri, ci] / g
            nr, nc = int(round(ny)), int(round(nx))
            if not (0 <= nr < H and 0 <= nc < W) or zp[nr, nc] >= zp[ri, ci]:
                a0, a1, b0, b1 = max(0, ri - 1), min(H, ri + 2), max(0, ci - 1), min(W, ci + 2)
                win = zp[a0:a1, b0:b1]; kk = np.unravel_index(np.argmin(win), win.shape); nr, nc = a0 + kk[0], b0 + kk[1]
                if zp[nr, nc] >= zp[ri, ci]: break
                ny, nx = float(nr), float(nc)
            y, x = ny, nx; pts.append((nr, nc))
        paint = np.zeros((H, W), bool)
        use = pts[head_gap:] if (ok and len(pts) - head_gap >= min_len) else []
        for (a0, b0), (a1, b1) in zip(use[:-1], use[1:]):
            lr, lc = line(a0, b0, a1, b1); paint[lr, lc] = True
        if use: out |= paint; n_add += 1
        # the cell is dealt with either way: mark it and its path as near
        mark = paint.copy(); mark[r0, c0] = True
        for (a, b) in pts[:head_gap + 1]: mark[a, b] = True
        d = np.minimum(d, ndi.distance_transform_edt(~mark))
    net2 = rejoin_after_thinning(out, skmo.skeletonize(out), roots)
    for (r, c, *_x) in roots: net2[int(r), int(c)] = True
    return net2, n_add
def generate(seed, reach=None, shape=(128, 128), outlets=None, gully_kw=None, **kw):
    """generate_scaled with the gullies added after stage 1 (reach=None: off)."""
    if reach is None: return S.generate_scaled(seed, shape=shape, outlets=outlets, **kw)
    orig = S.smooth_network
    def patched(net, roots, **k):
        net = orig(net, roots, **k)
        net2, n = add_gullies(net, roots, np.random.default_rng(10_000 + seed), reach=reach, **(gully_kw or {}))
        return net2
    S.smooth_network = patched
    try: return S.generate_scaled(seed, shape=shape, outlets=outlets, cache=False, **kw)
    finally: S.smooth_network = orig
