"""ridges -- the ridge network and its heights, for the two-bone stage 2.

solve(method="two_bones") pins the valleys AND the ridges and solves the
harmonic surface between them (harmonic.solve_two_bones).  This module
supplies the second bone:

  (1) Cut the channel network into links at the confluences, and label every
      hillslope cell with the link it drains into (D8 on a provisional
      surface).  The boundaries between labels are the ridges.
  (2) The RANK of a ridge segment is the drainage area at the confluence where
      the two links it separates meet [km2].  A segment separating two
      different river systems is a main divide and gets the whole map's area.
  (3) Ridge height = z of the nearest valley cell + C * rank**q  [m].
      Heights are averaged along the ridge (the rank is constant per segment,
      so it steps at segment ends) and capped so no pin rises more than
      cap_px m per px above any other pin (harmonic.lipschitz_cap).

Rank is an ABSOLUTE area, so its range grows with the map: a 3 x 12 km map
spans ~3 decades (0.04-36 km2), a 1.5 km square only ~1.5.  On a small map
the ridges cannot form a hierarchy and the solution collapses to flat ground
with a collar along each ridge; there the default Poisson solve is the better
choice.  On a long map the two-bone solve reads as continuous crests and
spurs.  The two methods are exclusive: once the ridges are Dirichlet pins,
the maximum principle stops a Poisson source from lifting the hillslope
above them.
"""
import heapq

import numpy as np
from scipy import ndimage as ndi

PX_M = 1500.0 / 128.0          # the scale contract: n=128 is 1.5 km
MIN_LINK_HA = 4.0              # channel links smaller than this merge into their parent
NB8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def fill_sinks(z):
    """Priority flood (+eps): fill every depression so all water leaves the map."""
    ny, nx = z.shape
    out = np.full_like(z, np.inf)
    done = np.zeros(z.shape, bool)
    h = []
    for i in range(ny):
        for j in (0, nx - 1):
            heapq.heappush(h, (z[i, j], i, j)); done[i, j] = True; out[i, j] = z[i, j]
    for j in range(nx):
        for i in (0, ny - 1):
            if not done[i, j]:
                heapq.heappush(h, (z[i, j], i, j)); done[i, j] = True; out[i, j] = z[i, j]
    eps = 1e-4
    while h:
        zc, i, j = heapq.heappop(h)
        for di, dj in NB8:
            a, b = i + di, j + dj
            if 0 <= a < ny and 0 <= b < nx and not done[a, b]:
                done[a, b] = True
                out[a, b] = max(z[a, b], zc + eps)
                heapq.heappush(h, (out[a, b], a, b))
    return out


def d8(zf, px):
    """Steepest-descent receiver of every cell, drainage area [cells], and the
    cells ordered from high to low."""
    ny, nx = zf.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    rec = idx.copy()
    best = np.zeros(zf.shape)
    for di, dj in NB8:
        zs = np.full_like(zf, -np.inf)
        sl = (slice(max(0, -di), ny - max(0, di)), slice(max(0, -dj), nx - max(0, dj)))
        sh = (slice(max(0, di), ny - max(0, -di)), slice(max(0, dj), nx - max(0, -dj)))
        zs[sl] = zf[sh]
        drop = (zf - zs) / (px * np.hypot(di, dj))
        ns = np.full(idx.shape, -1)
        ns[sl] = idx[sh]
        take = (drop > best) & (ns >= 0)
        best[take] = drop[take]
        rec.flat[np.flatnonzero(take.ravel())] = ns[take]
    order = np.argsort(zf.ravel())[::-1]
    acc = np.ones(ny * nx)
    r = rec.ravel()
    for k in order:
        if r[k] != k:
            acc[r[k]] += acc[k]
    return rec.ravel(), acc.reshape(zf.shape), order


def channel_links(chan, rec, order, shape):
    """Cut the channel network into links at the confluences."""
    n = shape[0] * shape[1]
    ch = chan.ravel()
    donors = np.zeros(n, int)
    for k in np.flatnonzero(ch):
        if rec[k] != k and ch[rec[k]]:
            donors[rec[k]] += 1
    conf = donors >= 2
    link = -np.ones(n, int)
    nxt = 0
    for k in order[::-1]:                    # from low (downstream) to high
        if not ch[k]:
            continue
        r = rec[k]
        if r == k or not ch[r] or conf[r]:   # a mouth, or just above a confluence
            link[k] = nxt; nxt += 1
        else:
            link[k] = link[r]
    return link, conf, nxt


def link_parent(link, rec, chan, nlink, shape):
    """link -> the link one step downstream (a mouth link is its own parent)."""
    ch = chan.ravel()
    par = np.arange(nlink)
    for k in np.flatnonzero(ch):
        r = rec[k]
        if r == k or not ch[r]:
            continue
        if link[r] != link[k]:
            par[link[k]] = link[r]
    return par


def lca_depth(par):
    """Depth of every link below its mouth (for the lowest common ancestor)."""
    nl = len(par)
    dep = -np.ones(nl, int)
    for i0 in range(nl):
        i, stack = i0, []
        while dep[i] < 0:
            stack.append(i)
            if par[i] == i:
                dep[i] = 0
                break
            i = par[i]
        base = dep[i]
        for j in reversed(stack):
            if dep[j] < 0:
                base += 1
                dep[j] = base
    return dep


def lca(a, b, par, dep):
    while dep[a] > dep[b]:
        a = par[a]
    while dep[b] > dep[a]:
        b = par[b]
    while a != b:
        if par[a] == a:
            return a
        a, b = par[a], par[b]
    return a


def ridge_network(z, chan, px, zroute=None):
    """The ridge network as the dual of the channel network, with ranks.

    zroute: the surface D8 routes on.  On a flat valley floor D8 is decided by
    rounding error and draws parallel stripes, so pass a surface with the
    channels cut into it and the routing follows them.

    Returns (ridge, rank [km2], kind, junction, acc)."""
    zf = fill_sinks(z if zroute is None else zroute)
    rec, acc, order = d8(zf, px)
    link, conf, nlink = channel_links(chan, rec, order, z.shape)
    lab = link.copy()
    for k in order[::-1]:
        if lab[k] < 0:
            lab[k] = lab[rec[k]]
    lab2 = lab.reshape(z.shape)

    # merge links that are too small into their parent: the 1-px network has
    # many one-cell branches, which would chop the ridges into thousands of
    # pieces
    par0 = link_parent(link, rec, chan, nlink, z.shape)
    ok = lab2 >= 0
    area = np.bincount(lab2[ok].ravel(), minlength=nlink)
    amin = MIN_LINK_HA * 1e4 / (px * px)
    remap = np.arange(nlink)
    for i in np.argsort(area):
        j = i
        while area[remap[j]] < amin and par0[remap[j]] != remap[j]:
            remap[j] = remap[par0[remap[j]]]
        remap[i] = remap[j]
    lab2 = np.where(ok, remap[np.maximum(lab2, 0)], -1)
    link = np.where(link >= 0, remap[np.maximum(link, 0)], -1)

    ridge = np.zeros(z.shape, bool)
    ny, nx = z.shape
    for di, dj in ((0, 1), (1, 0)):
        a = lab2[:ny - di, :nx - dj]; b = lab2[di:, dj:]
        df = a != b
        ridge[:ny - di, :nx - dj] |= df
        ridge[di:, dj:] |= df

    nlab = ndi.generic_filter(lab2.astype(np.int32),
                              lambda w: len(set(w)), size=3, mode="nearest")
    junction = nlab >= 3

    par = link_parent(link, rec, chan, nlink, z.shape)
    dep = lca_depth(par)
    accl = acc.ravel()
    link_out = np.zeros(nlink, int)
    for k in np.flatnonzero(chan.ravel()):
        l = link[k]
        if accl[k] > accl[link_out[l]]:
            link_out[l] = k

    def root(i):
        while par[i] != i:
            i = par[i]
        return i

    def path_child(x, m):
        while par[x] != m and par[x] != x:
            x = par[x]
        return x

    seg = ridge & ~ndi.binary_dilation(junction, np.ones((3, 3)))
    sl, nseg = ndi.label(seg, np.ones((3, 3)))
    rank = np.zeros(z.shape)
    kind = np.zeros(z.shape, np.int8)
    total = z.size * px * px / 1e6
    for s in range(1, nseg + 1):
        yy, xx = np.nonzero(sl == s)
        if len(yy) < 3:
            continue
        cnt = {}
        for i, j in zip(yy, xx):
            for di, dj in NB8:
                a, b = i + di, j + dj
                if 0 <= a < ny and 0 <= b < nx:
                    cnt[lab2[a, b]] = cnt.get(lab2[a, b], 0) + 1
        top = sorted(cnt, key=cnt.get, reverse=True)[:2]
        if len(top) < 2:
            continue
        a, b = top
        if root(a) != root(b):            # separates two river systems: a main divide
            rank[yy, xx] = total; kind[yy, xx] = 1
            continue
        m = lca(a, b, par, dep)
        child = path_child(b if m == a else a, m)
        kind[yy, xx] = 2 if (m == a or m == b) else 1
        rank[yy, xx] = accl[rec[link_out[child]]] * px * px / 1e6
    return ridge, rank, kind, junction, acc


def smooth_on_ridge(h, R, n=12):
    """Average along the ridge cells only, to remove the rank's steps."""
    v = np.where(R, np.nan_to_num(h), 0.0)
    w = R.astype(float)
    k = np.ones((3, 3))
    for _ in range(n):
        v = ndi.convolve(v, k, mode="nearest")
        ww = ndi.convolve(w, k, mode="nearest")
        v = np.where(R, v / np.maximum(ww, 1e-9), 0.0)
    return np.where(R, v, np.nan)


def ridge_bone(chan, band, zband, px=PX_M, C=28.0, q=0.5, cap_px=3.0,
               hill_slope=0.15, smooth_n=12):
    """The second bone for solve(method="two_bones").

    chan: 1-px channel network; band: every cell pinned as valley (channel +
    floor); zband: their z in METRES.  Returns (R, h): the ridge cells to pin
    and their z in metres (0 elsewhere)."""
    from experiments.harmonic import lipschitz_cap
    d_riv, (iy, ix) = ndi.distance_transform_edt(~band, sampling=px,
                                                 return_indices=True)
    z_riv = zband[iy, ix]
    # a provisional surface, only to route D8 on.  Where the ridges fall is
    # decided by which link each cell drains to, not by this cone's slope.
    z0 = z_riv + hill_slope * d_riv
    ridge, rank, kind, junction, acc = ridge_network(z0, chan, px,
                                                     zroute=z0 - 8.0 * chan)
    R = ridge & (rank > 0) & (d_riv > 2 * px) & ~band
    h = np.where(R, z_riv + C * np.maximum(rank, 1e-3) ** q, np.nan)
    h = np.nan_to_num(smooth_on_ridge(h, R, n=smooth_n))
    if cap_px:
        # rank is a FAR-FIELD quantity (the area at a confluence that may be
        # km away), so a ridge a few tens of metres beside the trunk can get a
        # large rank and would stand as a wall over the river.  Cap every pin
        # to cap_px m per px above every other pin.  iters must be explicit:
        # the default derives from the slope and is 0 for slopes in m/px.
        fixed = band | R
        fval = np.where(band, z_riv, np.where(R, h, 0.0))
        env = lipschitz_cap(np.where(fixed, fval, np.nan), fixed, cap_px,
                            iters=max(band.shape))
        h = np.where(R, np.minimum(h, env), 0.0)
    return R, h
