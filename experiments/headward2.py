"""
headward2 -- gen_headward's growth rule without its n**4 run time.

    gen_headward2(rng, n=128, ..., labels=True)  -> same call, same return

WHY gen_headward IS SLOW ON A BIG MAP.  The iteration budget is proportional
to the map's area (one short stroke per iteration), and every 12 iterations
the growth rebuilt its fields over the WHOLE map: the distance to the nearest
channel (one distance transform), the steering field d, and the gradient of
its gaussian.  _walk added two more whole-map costs per stroke: a copy of the
network and of the forbidden field, and a distance transform for the fold
test.  Area x area = n**4: 1.1 s at n=128, 8.3 s at 256, 40 s at 384.

WHAT IS DIFFERENT HERE.  Only how the fields are kept, not the rule:
  * the distance to the nearest channel (and the channel it belongs to) is
    updated INCREMENTALLY: a new stroke can only lower the distance where it
    is nearer than the old network, and per-block maxima of the field bound
    where that can happen, so only those blocks are touched;
  * d and the gradient of its gaussian are recomputed only around the cells
    whose distance changed (the gaussian reaches 8 px, the gradient 1 more);
  * _walk tests the live network in a small window around each step instead
    of copying the whole map, and finds the nearest channel for the fold test
    with an expanding local search;
  * the stop test and the re-seed target read block maxima instead of
    scanning the map.
The valley-floor field (cone_floor) is still rebuilt over the whole map, but
on a schedule whose count does not grow with the map (see gen_headward).

Every field holds the same values gen_headward computes, except where two
channel cells are exactly equally near: the nearest-channel INDEX can then
differ, which feeds the fold test and the re-seed target, so a network is
not bit-identical to gen_headward's -- it is a different draw of the same
process.  `net0` (restart rounds) and `trend` re-seeding are delegated to
gen_headward unchanged.
"""
import bisect
import os
import sys

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology as skmo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import mini as M

BLOCK = 16          # side of the blocks whose maxima bound the field updates
GAUSS_R = 8         # reach of gaussian_filter(sigma=2, truncate=4)


class _Fields:
    """dstop (distance to the nearest channel), its nearest-channel indices,
    and per-block maxima of dstop and d."""

    def __init__(self, H, W, reserve):
        self.H, self.W = H, W
        self.reserve = reserve
        self.nby = -(-H // BLOCK)
        self.nbx = -(-W // BLOCK)
        by = np.arange(self.nby)
        bx = np.arange(self.nbx)
        self.br0 = (by * BLOCK)[:, None] * np.ones((1, self.nbx), int)
        self.br1 = (np.minimum((by + 1) * BLOCK, H) - 1)[:, None] \
            * np.ones((1, self.nbx), int)
        self.bc0 = np.ones((self.nby, 1), int) * (bx * BLOCK)[None, :]
        self.bc1 = np.ones((self.nby, 1), int) \
            * (np.minimum((bx + 1) * BLOCK, W) - 1)[None, :]

    def _bmax(self, f):
        pad = np.full((self.nby * BLOCK, self.nbx * BLOCK), -np.inf)
        pad[:self.H, :self.W] = f
        return pad.reshape(self.nby, BLOCK, self.nbx, BLOCK).max(axis=(1, 3))

    def rebuild(self, net):
        self.dstop, (self.ny, self.nx) = ndi.distance_transform_edt(
            ~net, return_indices=True)
        if self.reserve is not None:
            self.dstop = np.where(self.reserve, 0.0, self.dstop)
        self.bmax = self._bmax(self.dstop)

    def blk(self, i, j):
        return (slice(self.br0[i, j], self.br1[i, j] + 1),
                slice(self.bc0[i, j], self.bc1[i, j] + 1))

    def add_cells(self, rr, cc):
        """Lower dstop where the new channel cells (rr, cc) are nearer than
        the old network.  Returns the set of blocks that changed."""
        if not len(rr):
            return set()
        r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
        dy = np.maximum(0, np.maximum(self.br0 - r1, r0 - self.br1))
        dx = np.maximum(0, np.maximum(self.bc0 - c1, c0 - self.bc1))
        cand = np.sqrt(dy * dy + dx * dx) < self.bmax
        changed = set()
        pr = rr[:, None, None]
        pc = cc[:, None, None]
        for i, j in zip(*np.where(cand)):
            sr, sc = self.blk(i, j)
            gr = np.arange(sr.start, sr.stop)[None, :, None]
            gc = np.arange(sc.start, sc.stop)[None, None, :]
            sq = (gr - pr) ** 2 + (gc - pc) ** 2
            k = sq.argmin(axis=0)
            dnew = np.sqrt(np.take_along_axis(sq, k[None], 0)[0].astype(float))
            old = self.dstop[sr, sc]
            m = dnew < old
            if self.reserve is not None:
                m &= ~self.reserve[sr, sc]
            if m.any():
                old[m] = dnew[m]
                self.ny[sr, sc][m] = rr[k][m]
                self.nx[sr, sc][m] = cc[k][m]
                self.bmax[i, j] = old.max()
                changed.add((i, j))
        return changed

    def argmax(self, f, bmax):
        """np.unravel_index(np.argmax(f)), via block maxima (same tie-break:
        the first maximal cell in row-major order)."""
        top = bmax.max()
        best = None
        for i, j in zip(*np.where(bmax == top)):
            sr, sc = self.blk(i, j)
            sub = f[sr, sc]
            r, c = np.unravel_index(int(np.argmax(sub)), sub.shape)
            cell = (sr.start + r, sc.start + c)
            if best is None or cell < best:
                best = cell
        return best


def _grad_window(d, r0, r1, c0, c1):
    """gradient(gaussian_filter(d, 2)) on rows r0:r1, cols c0:c1, computed on
    a window wide enough that the values equal the whole-map computation."""
    H, W = d.shape
    pad = GAUSS_R + 2
    R0, R1 = max(0, r0 - pad), min(H, r1 + pad)
    C0, C1 = max(0, c0 - pad), min(W, c1 + pad)
    g = ndi.gaussian_filter(d[R0:R1, C0:C1], 2.0)
    gy, gx = np.gradient(g)
    return (gy[r0 - R0:r1 - R0, c0 - C0:c1 - C0],
            gx[r0 - R0:r1 - R0, c0 - C0:c1 - C0])


def _nearest_other(net, new, box, ri, ci, H, W):
    """Nearest cell of `others` (the network as it was when the stroke began,
    minus the mouth box) to (ri, ci): (distance, row, col), or None.
    An expanding square search: a hit at distance m is final once the square
    has half-width >= m."""
    b0, b1, b2, b3 = box
    rad = 4
    while True:
        R0, R1 = max(0, ri - rad), min(H, ri + rad + 1)
        C0, C1 = max(0, ci - rad), min(W, ci + rad + 1)
        w = net[R0:R1, C0:C1].copy()
        for (pr, pc) in new:
            if R0 <= pr < R1 and C0 <= pc < C1:
                w[pr - R0, pc - C0] = False
        w[max(0, b0 - R0):max(0, b1 - R0), max(0, b2 - C0):max(0, b3 - C0)] = False
        yy, xx = np.nonzero(w)
        if len(yy):
            sq = (yy + R0 - ri) ** 2 + (xx + C0 - ci) ** 2
            k = int(sq.argmin())
            dist = float(np.sqrt(sq[k]))
            if dist <= rad or (R0 == 0 and C0 == 0 and R1 == H and C1 == W):
                return dist, int(yy[k] + R0), int(xx[k] + C0)
        elif R0 == 0 and C0 == 0 and R1 == H and C1 == W:
            return None
        rad *= 2


def _walk2(net, r, c, ang, steps, rng, wander=0.5, stop_on_hit=False, sep=0,
           fold=None, s_here=0.0, forbid=None, hw0=0.0, domain=None):
    """mini._walk with local tests.  Returns (end or None, cells newly painted
    in painting order)."""
    H, W = net.shape
    s = int(sep) if sep else 2
    r0, c0 = int(round(r)), int(round(c))
    box = (max(0, r0 - s - 1), r0 + s + 2, max(0, c0 - s - 1), c0 + s + 2)
    if forbid is not None:
        e = int(np.ceil(max(s, hw0))) + 1
        fbox = (max(0, r0 - e), r0 + e + 1, max(0, c0 - e), c0 + e + 1)
    new = []            # cells this stroke turned into channel; `others`
                        # (the network when the stroke began) excludes them
    for k in range(steps):
        ang += rng.normal(0, wander)
        r2, c2 = r + np.sin(ang), c + np.cos(ang)
        ri, ci = int(round(r2)), int(round(c2))
        if not (1 <= ri < H - 1 and 1 <= ci < W - 1):
            return None, new
        if domain is not None and not domain[ri, ci]:
            return None, new
        if sep:
            R0, R1 = max(0, ri - s), ri + s + 1
            C0, C1 = max(0, ci - s), ci + s + 1
            w = net[R0:R1, C0:C1]
            if w.any():
                w = w.copy()
                for (pr, pc) in new:
                    if R0 <= pr < R1 and C0 <= pc < C1:
                        w[pr - R0, pc - C0] = False
                w[max(0, box[0] - R0):max(0, box[1] - R0),
                  max(0, box[2] - C0):max(0, box[3] - C0)] = False
                if w.any():
                    return None, new
        if forbid is not None and forbid[ri, ci] and not (
                fbox[0] <= ri < fbox[1] and fbox[2] <= ci < fbox[3]):
            return None, new
        if fold is not None:
            S, cap = fold
            hit = _nearest_other(net, new, box, ri, ci, H, W)
            if hit is not None:
                dd, yr, xc = hit
                sn = S[yr, xc]
                if np.isfinite(sn) and dd > 0 and \
                        abs(s_here + k + 1.0 - sn) > cap * dd:
                    M._FOLD[0] += 1
                    return None, new
        if stop_on_hit and k >= 2 and net[ri, ci]:
            return None, new
        if not net[ri, ci]:
            new.append((ri, ci))
        net[ri, ci] = True
        r, c = r2, c2
    return (r, c, ang), new


def gen_headward2(rng, n=M.N, ell=5.0, split=0.45, split_p=0.0, sep=2,
                  budget0=1.0, side_keep=0.45, side_ang=(0.5, 1.05),
                  main_jit=0.25, run_max=26, step=(3, 7), fold_cap=6.0,
                  roots=None, shape=None, reach=None, corridor=1.0,
                  floor_px=4.0, acc_ref=M.ACC_REF_NET, net0=None,
                  seed_flanks=0, pin_frac_max=None, reserve=None, trend=None,
                  trend_max=1.15, reseed=True, labels=False, domain=None):
    """Drop-in replacement for mini.gen_headward (same arguments, same
    return).  See the module docstring for what differs."""
    if net0 is not None or trend is not None:
        return M.gen_headward(
            rng, n=n, ell=ell, split=split, split_p=split_p, sep=sep,
            budget0=budget0, side_keep=side_keep, side_ang=side_ang,
            main_jit=main_jit, run_max=run_max, step=step, fold_cap=fold_cap,
            roots=roots, shape=shape, reach=reach, corridor=corridor,
            floor_px=floor_px, acc_ref=acc_ref, net0=net0,
            seed_flanks=seed_flanks, pin_frac_max=pin_frac_max,
            reserve=reserve, trend=trend, trend_max=trend_max, reseed=reseed,
            labels=labels, domain=domain)
    H, W = (int(n), int(n)) if shape is None else (int(shape[0]), int(shape[1]))
    dom = None if domain is None else np.asarray(domain, bool)
    if dom is not None:
        if dom.shape != (H, W):
            raise ValueError(f"domain {dom.shape} does not match {(H, W)}")
        reserve = ~dom if reserve is None else (np.asarray(reserve, bool) | ~dom)
        if reach is None:
            m = np.ones((H, W), bool)
            for rc in (roots or []):
                m[int(rc[0]), int(rc[1])] = False
            reach = float(np.where(dom, ndi.distance_transform_edt(m),
                                   0.0).max())
    if reach is None:
        reach = budget0 * max(H, W)
    if reserve is not None:
        reserve = np.asarray(reserve, bool)
    net = np.zeros((H, W), bool)
    if roots is None:
        roots = [(H - 1, W // 2, -np.pi / 2)]
    S = np.full((H, W), np.nan)
    tips = []
    for (r0, c0, a0) in roots:
        net[int(r0), int(c0)] = True
        S[int(r0), int(c0)] = 0.0
        tips.append((float(r0), float(c0), float(a0), 0.0, 0.0,
                     float(reach), 0.0))
    F = _Fields(H, W, reserve)
    pending = []                         # cells painted since the last refresh

    def at(f, r, c):
        return f[int(np.clip(r, 0, H - 1)), int(np.clip(c, 0, W - 1))]

    def look(r, c, a, k):
        return at(d, r + k * np.sin(a), c + k * np.cos(a))

    def push(tip):
        v = look(tip[0], tip[1], tip[2], 2 * ell)
        i = bisect.bisect_right(neg_v, -v)
        tips.insert(i, tip)
        look_v.insert(i, v)
        neg_v.insert(i, -v)

    forbid, hwc = None, np.zeros((H, W))
    d = None
    dmax = None                          # block maxima of d
    look_v, neg_v = [], []
    every = 12 * max(1, int(round(H * W / (M.N * M.N))))
    for it in range(int(3000 * H * W / (M.N * M.N))):
        if it % 12 == 0 or not tips:
            full = corridor and (it % every == 0 or forbid is None or not tips)
            # a re-seed (no tips) rebuilds everything, as gen_headward does:
            # its target and the channel it grows from must be exact
            if full or d is None or not tips:
                if corridor:
                    accn = M.tree_accum(net, S) / float(acc_ref or 1.0)
                    hwc = float(corridor) * np.where(
                        net, 1.0 + (floor_px - 1.0) * np.sqrt(accn), 0.0)
                    forbid, _idx, cf = M.cone_floor(net, hwc, field=True)
                    d = np.maximum(-cf, 0.0)
                    F.rebuild(net)
                else:
                    F.rebuild(net)
                    d = F.dstop.copy()
                if reserve is not None:
                    d = np.where(reserve, 0.0, d)
                    forbid = reserve.copy() if forbid is None else (forbid |
                                                                   reserve)
                gr, gc = np.gradient(ndi.gaussian_filter(d, 2.0))
                dmax = F._bmax(d)
                pending = []
            elif pending:
                pr = np.array([p[0] for p in pending])
                pc = np.array([p[1] for p in pending])
                pending = []
                changed = F.add_cells(pr, pc)
                for (i, j) in changed:
                    sr, sc = F.blk(i, j)
                    if corridor:
                        d[sr, sc] = np.maximum(np.minimum(d[sr, sc],
                                                          F.dstop[sr, sc]), 0.0)
                    else:
                        d[sr, sc] = F.dstop[sr, sc]
                    if reserve is not None:
                        d[sr, sc] = np.where(reserve[sr, sc], 0.0, d[sr, sc])
                    dmax[i, j] = d[sr, sc].max()
                # the gaussian reaches GAUSS_R px and the gradient one more:
                # refresh that margin around every changed block
                for (i, j) in changed:
                    sr, sc = F.blk(i, j)
                    m = GAUSS_R + 1
                    r0, r1 = max(0, sr.start - m), min(H, sr.stop + m)
                    c0, c1 = max(0, sc.start - m), min(W, sc.stop + m)
                    gy, gx = _grad_window(d, r0, r1, c0, c1)
                    gr[r0:r1, c0:c1] = gy
                    gc[r0:r1, c0:c1] = gx
            look_v = [look(t[0], t[1], t[2], 2 * ell) for t in tips]
            _order = sorted(range(len(tips)), key=lambda i: -look_v[i])
            tips = [tips[i] for i in _order]
            look_v = [look_v[i] for i in _order]
            neg_v = [-v for v in look_v]
            if F.bmax.max() <= 3 * ell:
                break
            if pin_frac_max is not None and net.mean() >= pin_frac_max:
                break
        if not tips:
            if not reseed:
                break
            fr, fc = F.argmax(d, dmax)
            sr_, sc_ = float(F.ny[fr, fc]), float(F.nx[fr, fc])
            s_at = S[int(sr_), int(sc_)]
            tips = [(sr_, sc_, float(np.arctan2(fr - sr_, fc - sc_)), 0.0, 0.0,
                     float(reach) * side_keep,
                     float(s_at) if np.isfinite(s_at) else 0.0)]
            look_v = [look(t[0], t[1], t[2], 2 * ell) for t in tips]
            neg_v = [-v for v in look_v]
        k_pop = int(rng.integers(0, max(1, min(4, len(tips)))))
        r, c, a, run, lrun, bud, sd = tips.pop(k_pop)
        neg_v.pop(k_pop)
        if look_v.pop(k_pop) <= ell:
            continue
        if run >= bud:
            continue
        ga = np.arctan2(at(gr, r, c), at(gc, r, c))
        a = np.arctan2(0.65 * np.sin(a) + 0.35 * np.sin(ga),
                       0.65 * np.cos(a) + 0.35 * np.cos(ga))
        nstep = int(rng.integers(step[0], step[1]))
        out, new = _walk2(net, r, c, a, nstep, rng, wander=0.30,
                          stop_on_hit=True, sep=sep,
                          fold=((S, fold_cap) if fold_cap else None),
                          s_here=sd, forbid=forbid,
                          hw0=at(hwc, r, c) if corridor else 0.0, domain=dom)
        if new:
            pending.extend(new)
            if fold_cap:
                # gen_headward stamps S on the new cells in the order of
                # np.argsort over their row-major listing
                rr2, cc2 = np.array(sorted(new)).T
                order = np.argsort((rr2 - r) ** 2 + (cc2 - c) ** 2)
                S[rr2[order], cc2[order]] = sd + 1.0 + np.arange(len(order))
        if out is None:
            continue
        r, c, a = out
        run += nstep
        lrun += nstep
        sd += nstep
        x = max(0.0, bud - run) / float(reach or 1.0)
        if rng.random() < split * (x ** split_p if split_p else 1.0) \
                or lrun >= run_max:
            push((r, c, a + rng.normal(0, main_jit), run, 0.0, bud, sd))
            sg = rng.choice([-1, 1])
            sa = a + sg * rng.uniform(*side_ang)
            sb = bud * side_keep
            push((r, c, sa, 0.0, 0.0, sb, sd))
        else:
            push((r, c, a, run, lrun, bud, sd))
    M._FOLD[1] = S.copy()
    M._FOLD[2] = net.copy()
    skel = M.rejoin_after_thinning(net, skmo.skeletonize(net), roots)
    return (skel, M.basin_labels(skel, roots, dom)) if labels else skel
