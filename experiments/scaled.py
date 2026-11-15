"""scaled -- a generator whose terrain is in metres and does not depend on the
size of the map.

    from experiments.scaled import generate_scaled
    u, net, meta = generate_scaled(1, shape=(256, 1024),
                                   outlets=[(128, 1023, np.pi)])   # u in metres

WHY A SECOND GENERATOR.  baseline.generate is right at 128 x 128 and goes
wrong on long maps, and the reasons are all of one kind: quantities that
should be set by the size of a river are set by the size of the MAP.
    * the long profile is normalised to 0..1 per map, so a 12 km trunk falls
      as far as a 1.5 km one, while the Poisson hillslopes keep their height
      in px -- the balance of rivers and hills changes with the canvas;
    * the floor width is scaled by the map's own largest river
      (solve(acc_ref=None)), so a 1 km2 river is half as wide on a long map;
    * the whole map is normalised to the same relief at the end.
Here every one of them is an absolute law, in metres and km2:
    channel slope   S = ks * A**-theta          (Flint), integrated along the
                    network from the outlet at 0 m
    floor width     W = w1 * sqrt(A)            (hydraulic geometry)
    hillslopes      nabla^2 u = -f, f in m/px2  (Poisson, as before)
so a 1 km2 river looks the same on any map, and the big map gets the relief
its big rivers earn.  Stage 1 is gen_flood (fast), with a regional slope
towards the outlet, and stage 3 is finish_micro on a FIXED height scale.

Not the frozen zero: baseline.generate and its --check are untouched.
"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PX_M = 1500.0 / 128.0
KM2 = (PX_M / 1000.0) ** 2

# defaults, chosen by tools/terrain_metrics (see HANDOFF)
KS = 0.02           # channel slope at A = 1 km2
THETA = 0.45        # Flint exponent
S_MAX = 0.35        # headwater slope cap
W1 = 30.0           # floor width [m] at A = 1 km2 (opens from A0 = 0.1 km2)
F_POISSON = 0.6     # hillslope source [m / px2]
Z0 = 60.0           # stage 3 height scale [m]: finish_micro's u = 1


def _tree(net, roots):
    from tools.terrain_metrics import network_tree
    return network_tree(net, roots)


def channel_area(net, roots):
    """The BFS tree of the network from the outlets and the drainage area
    [km2] of each channel cell: every cell joins its nearest channel cell, and
    the allocations accumulate down the tree."""
    from experiments.flood import depth_of, catchment_sum
    rr, cc, idx, parent, step = _tree(net, roots)
    _d, (ir, ic) = ndi.distance_transform_edt(~net, return_indices=True)
    near = idx[ir, ic]
    local = np.bincount(near.ravel(), minlength=len(rr)).astype(float)
    depth = depth_of(parent)
    area = catchment_sum(parent, depth, local) * KM2
    return rr, cc, idx, parent, step, area


def stream_ids(parent, area):
    """For every node, the mouth of its stream: streams run from a head down
    the main stem (the child with the largest area) to where they join a
    bigger river."""
    from tools.terrain_metrics import _jump_until
    n = len(parent)
    main = np.full(n, -1)
    best = np.full(n, -1.0)
    for i in np.argsort(area, kind="stable"):
        p = parent[i]
        if p >= 0 and area[i] >= best[p]:
            best[p] = area[i]; main[p] = i
    pp = np.where(parent >= 0, parent, 0)
    stop = (parent < 0) | (main[pp] != np.arange(n))
    return _jump_until(parent, stop)


def slope_of(area, ks=KS, theta=THETA, s_max=S_MAX):
    return np.minimum(ks * np.maximum(area, 1e-4) ** (-theta), s_max)


def integrate_profile(parent, step, S):
    """z [m] of every node: S * step summed from the outlet (0 m)."""
    w = np.where(parent >= 0, S * step * PX_M, 0.0)
    anc = parent.copy()
    z = w.copy()
    while True:
        live = anc >= 0
        if not live.any():
            return z
        a = anc[live]
        z[live] = z[live] + z[a]
        anc[live] = anc[a]


def channel_profile(net, roots, labels=None, ks=KS, theta=THETA, s_max=S_MAX):
    """z [m] on the channel cells (Flint's law from the outlet) and the
    drainage area [km2] of each."""
    rr, cc, idx, parent, step, area = channel_area(net, roots)
    z = integrate_profile(parent, step, slope_of(area, ks, theta, s_max))
    return rr, cc, idx, z, area


def envelope(pin, zpin, sc, mask=None):
    """min over pinned cells y of [zpin(y) + path length from y * sc], with
    sc [m per m] a field (the hillslope's threshold gradient).  Dijkstra on
    the 8-neighbour grid from a virtual node joined to every pinned cell at
    cost zpin -- a threshold hillslope rising from every valley at once.
    Opposite slopes meet in a sharp crest, and a crest rises upstream with
    the valleys either side of it, so summits sit on continuous ridges
    instead of in the middle of every hollow between channels."""
    from scipy import sparse
    from scipy.sparse import csgraph
    from experiments.flood import grid_edges
    H, W = pin.shape
    N = H * W
    A, B = grid_edges(H, W, mask)
    ra, ca = np.divmod(A, W); rb, cb = np.divmod(B, W)
    ln = np.hypot(ra - rb, ca - cb) * PX_M
    scf = np.broadcast_to(np.asarray(sc, float), (H, W)).ravel()
    w = 0.5 * (scf[A] + scf[B]) * ln
    P = np.flatnonzero(pin.ravel())
    off = 1.0 - float(zpin.ravel()[P].min())        # all weights > 0
    rows = np.concatenate([A, np.full(len(P), N)])
    cols = np.concatenate([B, P])
    ww = np.concatenate([w, zpin.ravel()[P] + off])
    G = sparse.coo_matrix((ww, (rows, cols)), shape=(N + 1, N + 1)).tocsr()
    d = csgraph.dijkstra(G, directed=False, indices=N)
    return (d[:N] - off).reshape(H, W)


def cap_channels(rr, cc, parent, zc, area, H, W, s_cap, ratio=3.0):
    """NO RIVER STANDS HIGH ABOVE ITS NEIGHBOURS OF ITS OWN SIZE.  Flint's
    law integrated from the outlet makes a river as high as its path is
    long: a tributary that winds far round before it joins climbs well
    above a river of the same size just across the divide, and its basin
    stands up as a plateau with a scarp along the divide.  A real divide
    would move until the two sides balance (chi across a divide).

    Here the channel heights z are the largest that keep, everywhere,
      z <= Flint's z;
      z(i) <= z(j) + s_cap * S(A) * distance over the map, for the channel
        cells j of a similar size (area within a factor ratio either way),
        S(A) the Flint slope of that size -- two headwater gullies side by
        side may differ by what their steep beds climb between them, two
        big rivers by little.  (Among ALL channels the bound would flatten
        every steep headwater against its own trunk; with one slope for
        every size it sank whole basins by 200 m.)
      z(i) <= z(j) for every j upstream of i -- a capped head pulls its
        stream down with it; a reach below it goes flat, never uphill.
      z(i) <= z(the cell i drains to) + i's own Flint step -- a lowered
        river takes its tributaries down with it.
    Every bound is "z(i) <= z(j) + a cost >= 0", so the answer is one
    shortest-path problem: a source joined to each channel cell at its
    Flint z, an edge of cost 0 from a cell to the cell it drains to, and
    for each size class a layer of the grid (edge cost s * step length)
    that the cells of the classes either side feed and the class's own
    cells read.  One Dijkstra -- no seed can raise one basin above the
    rest."""
    from scipy import sparse
    from scipy.sparse import csgraph
    from experiments.flood import grid_edges
    n = len(rr)
    la = np.floor(np.log(np.maximum(area, 1e-6)) / np.log(ratio)).astype(int)
    ks = np.unique(la)
    A, B = grid_edges(H, W)
    ra, ca = np.divmod(A, W); rb, cb = np.divmod(B, W)
    ln = np.hypot(ra - rb, ca - cb) * PX_M
    G = H * W
    rows, cols, wts = [], [], []
    # drains-to edges (cost 0 -> a tiny epsilon keeps them in the sparse
    # matrix; it adds < 1e-6 m over a 1e4-cell path)
    k = parent >= 0
    rows.append(np.flatnonzero(k)); cols.append(parent[k])
    wts.append(np.full(int(k.sum()), 1e-10))
    # ... and no cell rises above the cell it drains to by more than its
    # own Flint step.  Without this a lowered river left its smallest twigs
    # behind -- a one-cell gully whose size class has no neighbour near
    # enough to bind it stood 100 m above the stream it joins.
    rows.append(parent[k]); cols.append(np.flatnonzero(k))
    wts.append(np.maximum(zc[k] - zc[parent[k]], 0.0) + 1e-10)
    cell = rr * W + cc
    src = n + len(ks) * G
    N = src + 1
    # The grid layers are the same graph with one slope each, and they are
    # nearly all of the matrix (27 million edges on a 3 x 12 km map), so
    # they are written straight into CSR arrays -- one layer's rows copied
    # with an offset -- instead of going through a sort of the whole edge
    # list.  The few edges that touch the channel cells are added to it.
    g1 = sparse.coo_matrix((ln, (A, B)), shape=(G, G))
    g1 = (g1 + g1.T).tocsr()
    g1.sort_indices()
    nz = g1.nnz
    L = len(ks)
    indices = np.empty(L * nz, np.int64)
    data = np.empty(L * nz)
    indptr = np.zeros(N + 1, np.int64)
    for li, kk in enumerate(ks):
        base = n + li * G
        sk = float(s_cap) * float(slope_of(ratio ** (kk + 0.5)))
        indices[li * nz:(li + 1) * nz] = g1.indices
        indices[li * nz:(li + 1) * nz] += base
        np.multiply(g1.data, sk, out=data[li * nz:(li + 1) * nz])
        indptr[base + 1:base + G + 1] = g1.indptr[1:] + li * nz
        feed = np.flatnonzero(np.abs(la - kk) <= 1)
        rows.append(feed); cols.append(base + cell[feed])
        wts.append(np.full(len(feed), 1e-10))
        own = np.flatnonzero(la == kk)
        rows.append(base + cell[own]); cols.append(own)
        wts.append(np.full(len(own), 1e-10))
    indptr[src + 1:] = L * nz
    layers = sparse.csr_matrix((data, indices, indptr), shape=(N, N))
    off = 1.0 - float(zc.min())
    rows.append(np.full(n, src)); cols.append(np.arange(n))
    wts.append(zc + off)
    M = layers + sparse.coo_matrix(
        (np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N)).tocsr()
    d = csgraph.dijkstra(M, directed=True, indices=src)
    return np.minimum(d[:n] - off, zc)


def solve_m(net, roots, labels=None, ks=KS, theta=THETA, s_max=S_MAX, w1=W1,
            f=F_POISSON, floor_tilt=0.02, hill="envelope", sc=0.3,
            sc_var=0.3, crest_sigma=1.5, rng=None, floor_close=2, a0=0.1,
            stream_var=0.0, theta_var=0.0, lith_var=0.0, ridge_C=28.0,
            ridge_q=0.5, ridge_cap=0.45, ridge_w0=1.5, ridge_h0=0.0,
            zmode="flint", plate_d1=50.0, plate_b=0.4, plate_tilt=0.005,
            plate_ridge="cap", plate_base="zero", plate_rise=3.0,
            ridge_lower=30.0, ridge_border=False, ridge_slope=0.05,
            chan_cap=0.0, floor_keep=True, ridge_level=0, ridge_dome=0.0):
    """Stage 2 in metres.  Returns u [m].

    hill: "poisson" lifts the ground between the valleys by nabla^2 u = -f;
    "envelope" raises threshold hillslopes of gradient sc from every valley
    (see envelope), sc varied by smooth noise (sc_var, log-std) as rock
    strength would, and the crests rounded by a gaussian of crest_sigma px
    applied off the valleys only."""
    from experiments.mini import cone_floor, clip_to_own_basin
    import experiments.harmonic as EL
    net = np.asarray(net, bool)
    H, W = net.shape
    rr, cc, idx, parent, step, area = channel_area(net, roots)
    amap = np.zeros((H, W)); amap[rr, cc] = area
    _dn, (iyn, ixn) = ndi.distance_transform_edt(~net, return_indices=True)
    band = ndi.binary_dilation(net)
    if labels is not None:
        band = clip_to_own_basin(band, labels)
    # floor half-width in px from the absolute law; the channel cell itself
    # is half a px wide
    # headwater valleys are V-shaped: the floor opens from A0 km2 on
    hw = np.where(net, 0.5 * w1 * (np.sqrt(amap) - np.sqrt(a0)) / PX_M + 0.5,
                  0.0)
    hw = np.where(hw > 1.0, hw, 0.0)
    floor, (oy, ox) = cone_floor(net, hw)
    # a strip of ground left between two floors (a tributary running beside
    # the trunk before it joins) would rise as a knife-edge at the threshold
    # slope; it is a low terrace in the real thing, so close the floor over
    # gaps of a few px and fill the small holes it leaves
    floor0 = floor
    if floor_close:
        from skimage.morphology import disk
        floor = ndi.binary_closing(floor | net, structure=disk(int(floor_close)))
        # islands of ground inside a floor: keep only the ones broad enough
        # to be real hills (a meander core), fill the slivers
        holes = ndi.binary_fill_holes(floor) & ~floor
        lab, nh = ndi.label(holes)
        if nh:
            idt = ndi.distance_transform_edt(holes)
            thick = ndi.maximum(idt, lab, np.arange(1, nh + 1))
            size = ndi.sum(holes, lab, np.arange(1, nh + 1))
            fill = np.concatenate([[False], (thick < 3.5) | (size < 600)])
            floor = floor | fill[lab]
    # the ground the closing added belongs to the river of the floor next
    # to it, so a tributary crossing it is on that river's floor too
    onfl = floor0
    if floor_close and floor0.any():
        _df, (fy, fx) = ndi.distance_transform_edt(~floor0, return_indices=True)
        oy, ox = oy[fy, fx], ox[fy, fx]
        # the closing is for floors: two headwater gullies a few px apart
        # have no floor between them
        floor = floor0 | net | (floor & (_df <= 2 * int(floor_close) + 2))
        onfl = floor0 | ndi.binary_dilation(floor & ~net, np.ones((3, 3), bool))
    elif floor_close:
        floor = floor0 | net
    # A TRIBUTARY ON A BIGGER RIVER'S FLOOR FLOWS AT THAT RIVER'S GRADIENT.
    # With its own (steeper) Flint slope it would climb above the floor it
    # crosses, and the floor -- painted at the nearest channel's height --
    # would break into steps and slivers between the two.
    # NOT EVERY TRIBUTARY CLIMBS ALIKE.  One law for every channel gives
    # every side valley the same rise from the trunk to the divide.  Each
    # stream (a head down its main stem to where it joins a bigger river)
    # draws its own steepness ks and concavity theta, and ks also follows a
    # smooth "rock strength" field across the map.
    ksn = np.full(len(rr), float(ks))
    thn = np.full(len(rr), float(theta))
    if rng is not None and (stream_var or theta_var or lith_var):
        sid = stream_ids(parent, area)
        u_ids, inv = np.unique(sid, return_inverse=True)
        ksn = ksn * np.exp(stream_var * rng.standard_normal(len(u_ids)))[inv]
        thn = np.clip(thn + theta_var * rng.standard_normal(len(u_ids))[inv],
                      0.2, 0.7)
        if lith_var:
            from experiments.flood import noise
            nz = noise(rng, H, W, smin=32.0, beta=0.5, fast=True)
            nz = (nz - nz.mean()) / (nz.std() or 1.0)
            ksn = ksn * np.exp(lith_var * nz[rr, cc])
        # the outlet's own stream keeps the plain law, so the trunk and the
        # map's relief stay where the defaults put them
        trunk = sid == sid[np.argmax(area)]
        ksn[trunk], thn[trunk] = ks, theta
    S = slope_of(area, ksn, thn, s_max)
    own = np.where(onfl[rr, cc], idx[oy[rr, cc], ox[rr, cc]], -1)
    onf = (own >= 0) & (own != np.arange(len(rr)))
    onf[onf] &= area[own[onf]] > 2.0 * area[onf]
    S[onf] = slope_of(area[own[onf]], ksn[own[onf]], thn[own[onf]], s_max)
    zc = integrate_profile(parent, step, S)
    plate = None
    if zmode == "plate":
        # CUT FROM ABOVE.  The land was one plate, and the rivers cut down
        # into it: a river lies D(A) = plate_d1 * A**plate_b below the plate,
        # bigger rivers deeper, and no ground stands above the plate.  The
        # plate is level but for a gentle rise (plate_tilt) away from the
        # outlets, and sits as high above the outlet as the map's biggest
        # river cuts.  The Flint profile from below still binds: a channel
        # can rise no faster than S(A) above the river it flows into, so a
        # tributary steepens into its mouth instead of falling over a step.
        seed_o = np.ones((H, W), bool)
        for (r, c, *_x) in roots:
            seed_o[int(r), int(c)] = False
        plate = (plate_d1 * float(area.max()) ** plate_b
                 + plate_tilt * ndi.distance_transform_edt(seed_o) * PX_M)
        Aeff = area.copy()
        Aeff[onf] = area[own[onf]]
        top = plate[rr, cc] - plate_d1 * np.maximum(Aeff, 1e-4) ** plate_b
        from experiments.flood import depth_of, _levels
        dep = depth_of(parent)
        # plate_base="zero": the outlets sit at 0 m and the Flint rise binds
        # (deep cuts far up every river).  "top": the outlets too sit at
        # their own depth below the plate, and the rise from below only
        # smooths the step at a confluence (plate_rise x the Flint slope),
        # so a river lies about D(A) below the plate wherever it is.
        zc = np.zeros(len(rr))
        Sr = S * (plate_rise if plate_base == "top" else 1.0)
        for grp in _levels(dep)[::-1]:                 # outlets first
            p_ = parent[grp]
            k = p_ >= 0
            z_up = np.where(k, zc[np.where(k, p_, 0)] + Sr[grp] * step[grp] * PX_M,
                            0.0)
            z0 = top[grp] if plate_base == "top" else 0.0
            zc[grp] = np.where(k, np.minimum(top[grp], z_up), z0)
    if chan_cap:
        zc = cap_channels(rr, cc, parent, zc, area, H, W, chan_cap)
    zmap = np.zeros((H, W)); zmap[rr, cc] = zc
    if labels is not None:
        floor = clip_to_own_basin(floor, labels)
    pin = band | floor
    # THE FLOOR TAKES ITS HEIGHT FROM ITS OWN RIVERS: the lowest of
    # z(channel) + floor_tilt * distance, along paths that stay on the pinned
    # ground.  The plain nearest channel is often a high headwater across a
    # narrow divide (a wall along the floor's edge), and the nearest channel
    # along the floor still leaves a step where a perched tributary's floor
    # meets the trunk's; the envelope turns every such step into a ramp.
    # ALONG A RIVER THE FLOOR RISES AS THE RIVER DOES.  Every channel cell
    # is pinned ground, so with one tilt for every path the envelope also ran
    # up the channels themselves and no river could climb faster than
    # floor_tilt: every valley of the map lay at 2 % whatever Flint's law
    # gave it, and the hills between them had nothing to stand on.  The
    # tilt next to a channel cell is therefore at least that cell's own
    # slope (and its tributaries' where they join it), so the envelope
    # leaves the channel profile as it is and still ramps the floor across
    # the valley.  floor_keep=False is the old single tilt.
    tilt = floor_tilt
    if floor_keep:
        kk = (parent >= 0) & (step > 0)
        sl = np.zeros(len(rr))
        sl[kk] = (zc[kk] - zc[parent[kk]]) / (step[kk] * PX_M)
        scn = np.maximum(sl, floor_tilt)
        np.maximum.at(scn, parent[kk], sl[kk])
        # ... but only beside the channel itself (1 px), and not where that
        # ground is the floor of a bigger river: there a steep gully coming
        # off the hills would lift a wedge of the floor around its mouth
        nn = idx[iyn, ixn]
        big = floor0 & (area[idx[oy, ox]] > 2.0 * area[nn])
        tilt = np.where((_dn < 1.5) & ~big, scn[nn], floor_tilt)
    zn = envelope(net, zmap, tilt, mask=pin)
    bad = ~np.isfinite(zn)
    zn = np.where(bad, zmap[iyn, ixn] + floor_tilt * _dn * PX_M, zn)
    dom = None
    if labels is not None and (np.asarray(labels) < 0).any():
        dom = np.asarray(labels) >= 0
    ridge_info = {}
    if hill == "two_bones":
        from experiments.ridges import ridge_bone
        R, h = ridge_bone(net, pin, np.where(pin, zn, 0.0))
        u = EL.solve_two_bones(pin, np.where(pin, zn, 0.0), R, h,
                               poisson_f=0.0, fast=True)
        ridge_info = dict(R=R, h=h)
    elif hill == "ridges":
        R, h = ridge_lines(net, roots, pin, zn, C=ridge_C, q=ridge_q,
                           cap=ridge_cap, w0=ridge_w0, h0=ridge_h0,
                           plate=plate, plate_ridge=plate_ridge,
                           lower=ridge_lower, border=ridge_border,
                           r_slope=ridge_slope, level_n=ridge_level,
                           dome=ridge_dome)
        u = EL.solve_two_bones(pin, np.where(pin, zn, 0.0), R, h,
                               poisson_f=0.0, fast=True)
        ridge_info = dict(R=R, h=h)
    elif hill == "poisson":
        z0 = np.zeros_like(pin)
        u = EL.solve_two_bones(pin, np.where(pin, zn, 0.0), z0, np.zeros((H, W)),
                               poisson_f=f, domain=dom, fast=True)
    else:
        scf = float(sc)
        if sc_var and rng is not None:
            from experiments.flood import noise
            nz = noise(rng, H, W, smin=8.0, beta=0.5, fast=True)
            nz = (nz - nz.mean()) / (nz.std() or 1.0)   # noise() keeps a mean
            scf = sc * np.exp(sc_var * nz)
        u = envelope(pin, np.where(pin, zn, 0.0), scf, dom)
        if crest_sigma:
            # round the crests, never touch the valleys
            ub = ndi.gaussian_filter(u, crest_sigma)
            wgt = np.clip(_dn / (2.0 * crest_sigma), 0.0, 1.0)
            wgt = np.where(pin, 0.0, wgt)
            u = u * (1 - wgt) + ub * wgt
            u = np.maximum(u, np.where(pin, zn, -np.inf))
    return u, dict(area=amap, floor=floor, zc=zmap, pin=pin, zn=zn, **ridge_info)


_STAGE1 = {}        # generate_scaled's last few networks, by their arguments


def _one(args):
    seed, kw = args
    return generate_scaled(seed, **kw)


def generate_many(seeds, workers=None, **kw):
    """[generate_scaled(seed, **kw) for seed in seeds], one process per seed
    (as many at a time as there are CPUs, or `workers`).  The same results;
    a sweep over four seeds takes about as long as one map."""
    import concurrent.futures as cf
    import multiprocessing as mp
    seeds = list(seeds)
    n = min(len(seeds), workers or os.cpu_count() or 1)
    # the workers import the main script again, so it must be a file that
    # guards its work with `if __name__ == "__main__":` -- from an
    # interactive prompt or a script on stdin the seeds run one by one
    main = getattr(sys.modules.get("__main__"), "__file__", None)
    if n <= 1 or not (main and os.path.exists(main)):
        return [generate_scaled(sd, **kw) for sd in seeds]
    # "spawn": a forked child inherits the locks of this process's thread
    # pools (cone_floor, the BLAS) and can hang on them
    with cf.ProcessPoolExecutor(n, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(_one, [(sd, kw) for sd in seeds]))


def generate_scaled(seed, shape=(128, 128), outlets=None, n_roots=None,
                    regional=2.0, ell=5.0, floor_px=4.0, finish=True,
                    ks=KS, theta=THETA, s_max=S_MAX, w1=W1, f=F_POISSON,
                    z0=Z0, finish_kw=None, hill="ridges", sc=0.3, sc_var=0.3,
                    crest_sigma=1.5, straighten=2.0, s0=1.0, floor_close=2,
                    stage1="flood", trend=None, stream_var=0.4, theta_var=0.1,
                    lith_var=0.3, ridge_C=40.0, ridge_q=0.3, ridge_cap=0.25,
                    ridge_w0=1.5, flood_kw=None, regrow=True, regrow_kw=None,
                    ridge_h0=0.0, zmode="flint", plate_d1=50.0, plate_b=0.4,
                    plate_tilt=0.005, plate_ridge="cap", plate_base="zero",
                    plate_rise=3.0, ridge_lower=30.0, ridge_border=False,
                    ridge_slope=0.05, chan_cap=1.0, floor_keep=True,
                    ridge_level=30, ridge_dome=10.0, ridge_room=4.0,
                    ridge_w=15.0, cache=True):
    """Stages 1-3, u in metres (the outlets at 0 m)."""
    from experiments.baseline import edge_roots, N
    from experiments.flood import gen_flood
    from experiments.mini_finish import finish_micro
    rng = np.random.default_rng(seed)
    H, W = shape
    if outlets is None:
        k = int(round(int(rng.integers(3, 6)) * (H + W) / (2 * N)))
        k = int(n_roots) if n_roots is not None else max(1, k)
        roots = edge_roots(rng, H, W, k)
    else:
        roots = [(int(r), int(c), float(a)) for (r, c, a) in outlets]
    # STAGE 1 IS KEPT.  The network depends only on the arguments below, so a
    # call that changes stage 2 or 3 alone (which is what tuning them is)
    # takes it from the last few made, with the random generator put back
    # where stage 1 left it -- the result is bit for bit the same.
    key = repr((seed, (H, W), roots, regional, ell, floor_px, stage1, trend,
                sorted((flood_kw or {}).items()), regrow,
                sorted((regrow_kw or {}).items()), straighten, s0,
                ridge_room, ridge_w))
    hit = _STAGE1.get(key) if cache else None
    if hit is not None:
        net, labels = hit[0].copy(), hit[1].copy()
        rng.bit_generator.state = hit[2]
    else:
        if stage1 == "flood":
            # no channel heads along the divides between the rivers of 1 km2 and
            # more (gen_flood's two passes): the valleys do not eat into a main
            # ridge, so it stays broad and runs straight
            fk = dict(ridge_room=ridge_room, ridge_w=ridge_w, fast_noise=True)
            fk.update(flood_kw or {})
            net, labels = gen_flood(rng, roots=roots, shape=(H, W), ell=ell,
                                    floor_px=floor_px, labels=True,
                                    regional=regional, **fk)
        else:
            from experiments.mini import gen_headward
            from experiments.baseline import hydraulic_reach
            reach = None if (H, W) == (N, N) else hydraulic_reach((H, W), roots)
            net, labels = gen_headward(rng, n=H, ell=ell, roots=roots,
                                       shape=(H, W), reach=reach,
                                       floor_px=floor_px, labels=True,
                                       trend=trend)
        if regrow:
            from experiments.mini import basin_labels
            net = regrow_small(net, roots, rng, **(regrow_kw or {}))
            labels = basin_labels(net, roots)
        if straighten:
            from experiments.mini import basin_labels
            net = smooth_network(net, roots, c=straighten, s0=s0)
            labels = basin_labels(net, roots)
        if cache:
            while len(_STAGE1) >= 8:
                _STAGE1.pop(next(iter(_STAGE1)))
            _STAGE1[key] = (net.copy(), labels.copy(), rng.bit_generator.state)
    u, info = solve_m(net, roots, labels, ks, theta, s_max, w1, f, hill=hill,
                      sc=sc, sc_var=sc_var, crest_sigma=crest_sigma, rng=rng,
                      floor_close=floor_close, stream_var=stream_var,
                      theta_var=theta_var, lith_var=lith_var, ridge_C=ridge_C,
                      ridge_q=ridge_q, ridge_cap=ridge_cap, ridge_w0=ridge_w0,
                      ridge_h0=ridge_h0, zmode=zmode, plate_d1=plate_d1,
                      plate_b=plate_b, plate_tilt=plate_tilt,
                      plate_ridge=plate_ridge, plate_base=plate_base,
                      plate_rise=plate_rise, ridge_lower=ridge_lower,
                      ridge_border=ridge_border, ridge_slope=ridge_slope,
                      chan_cap=chan_cap, floor_keep=floor_keep,
                      ridge_level=ridge_level, ridge_dome=ridge_dome)
    if finish:
        kw = dict(depth0=0.13, width=2.4, smooth=1.5,
                  max_cuts=int(500 * H * W / (N * N)), final_sigma=0.9,
                  pick="stable")
        kw.update(finish_kw or {})
        u = finish_micro(u / z0, net, rng, **kw) * z0
    return u, net, dict(roots=roots, labels=labels, **info)


def smooth_network(net, roots, c=2.0, s0=1.0, domain=None):
    """Straighten each river at the scale of its own size.

    The flood tree follows the lowest pass of random ground at every scale,
    so a 30 km2 trunk jitters px by px exactly like a headwater.  A real
    river's valley has no planform detail much finer than its own width.

    The network is cut into STREAMS (from a head down the largest child to
    where it joins a bigger river), each stream's polyline is smoothed by a
    gaussian of sigma = c * sqrt(A [km2]) px, A the drainage area at each
    point (so headwaters keep their wiggle), and the streams are redrawn
    biggest first, every tributary's mouth re-attached to where its parent
    now runs.  Returns the new 1-px network."""
    from skimage.draw import line
    from skimage import morphology as skmo
    from experiments.mini import rejoin_after_thinning
    net = np.asarray(net, bool)
    H, W = net.shape
    rr, cc, idx, _z, area = channel_profile(net, roots)
    _rr, _cc, _idx, parent, _step = _tree(net, roots)
    n = len(rr)
    # main child of every node = its child with the largest area
    main = np.full(n, -1)
    best = np.full(n, -1.0)
    for i in np.argsort(area, kind="stable"):
        p = parent[i]
        if p >= 0 and area[i] >= best[p]:
            best[p] = area[i]; main[p] = i
    has_kid = np.zeros(n, bool); has_kid[parent[parent >= 0]] = True
    heads = np.flatnonzero(~has_kid)
    streams = []
    for h in heads:
        path = [h]
        i = h
        while parent[i] >= 0 and main[parent[i]] == i:
            i = parent[i]; path.append(i)
        # path runs head -> mouth; the node below the mouth (if any) is on a
        # bigger stream
        streams.append(path)
    streams.sort(key=lambda p: -area[p[-1]])
    pos = np.column_stack([rr, cc]).astype(float)
    new = pos.copy()
    done = np.zeros(n, bool)
    out = np.zeros((H, W), bool)
    for path in streams:
        path = np.array(path[::-1])                   # mouth -> head
        P = pos[path]
        sig = s0 + c * np.sqrt(np.maximum(area[path], 0.0))
        L = len(path)
        Q = P.copy()
        if sig.max() >= 0.5 and L >= 3:
            s = np.arange(L)
            for j in range(L):
                sj = sig[j]
                if sj < 0.5:
                    continue
                r = int(min(3 * sj, L))
                lo, hi = max(0, j - r), min(L, j + r + 1)
                w = np.exp(-0.5 * ((s[lo:hi] - j) / sj) ** 2)
                # reflect at the ends so the ends stay put
                Q[j] = (w[:, None] * P[lo:hi]).sum(0) / w.sum()
            # pin the ends: the mouth to where its receiver now is (or the
            # outlet), the head where it was; a linear blend in between
            below = parent[path[0]]
            m_to = new[below] if below >= 0 else P[0]
            t = np.linspace(0.0, 1.0, L)[:, None]
            Q = Q + (1 - t) * (m_to - Q[0]) + t * (P[-1] - Q[-1])
        new[path] = Q
        # draw: receiver -> mouth -> ... -> head
        below = parent[path[0]]
        pts = ([new[below]] if below >= 0 else []) + list(Q)
        pts = np.clip(np.rint(np.array(pts)), [0, 0], [H - 1, W - 1]).astype(int)
        for (r0, c0), (r1, c1) in zip(pts[:-1], pts[1:]):
            lr, lc = line(r0, c0, r1, c1)
            out[lr, lc] = True
    if domain is not None:
        out &= np.asarray(domain, bool)
    net2 = rejoin_after_thinning(out, skmo.skeletonize(out), roots)
    for (r, c, *_x) in roots:
        net2[int(r), int(c)] = True
    return net2


def ridge_lines(net, roots, pin, zn, levels=None, C=28.0, q=0.5, cap=0.45,
                smooth_n=8, w0=1.5, p=0.35, edge=0.25, h0=0.0, plate=None,
                plate_ridge="cap", lower=30.0, border=False, r_slope=0.05,
                main_A=3.0, level_n=0, dome=0.0, min_piece=0, level_A=0.3):
    """The ridge network as the DUAL of the river network, with heights that
    grow with the size of the rivers a ridge separates.

    At each level A_k (km2) the network is cut into links of rivers >= A_k
    (cut at the confluences), every cell joins the link its nearest channel
    cell drains into at that level, and the boundaries between links are the
    divides of that level.  A divide between two big rivers is still a
    boundary at the high levels; its ORDER is the highest level at which it
    is.  Boundaries of regions are closed lines, so the ridges are continuous
    by construction -- no gaps for the surface to sag through into a saddle.

    Height: the higher of the rivers either side + C * A_order**q [m], then
    averaged along the ridge (smooth_n passes) and capped so that no ridge
    stands more than `cap` (gradient) above the valley ground around it.
    Returns (R, h)."""
    from tools.terrain_metrics import _jump_until
    net = np.asarray(net, bool)
    H, W = net.shape
    rr, cc, idx, parent, step, area = channel_area(net, roots)
    _d, (ir, ic) = ndi.distance_transform_edt(~net, return_indices=True)
    near = idx[ir, ic]
    if levels is None:
        levels = 0.01 * 3.0 ** np.arange(12)
        levels = levels[levels < area.max()]
    order = np.full((H, W), -1.0)
    n = len(rr)
    for Ak in levels:
        on = area >= Ak
        entry = _jump_until(parent, on)
        kid = (parent >= 0) & on
        nk = np.bincount(parent[kid], minlength=n)
        stop = np.zeros(n, bool)
        pj = parent >= 0
        stop[pj] = nk[parent[pj]] >= 2
        stop |= ~pj
        link = _jump_until(parent, stop | ~on)
        lab = link[entry[near]]
        # the link below each link (its id is the node just above its
        # junction, so the junction is its parent).  A link and the one it
        # flows into meet ACROSS a valley at the confluence: that boundary is
        # not a divide.
        plink = np.where(pj, link[np.where(pj, parent, 0)], -1)

        def divide(a, b):
            return (a != b) & (plink[a] != b) & (plink[b] != a)
        d = np.zeros((H, W), bool)
        e = divide(lab[:, :-1], lab[:, 1:])
        d[:, :-1] |= e; d[:, 1:] |= e
        e = divide(lab[:-1, :], lab[1:, :])
        d[:-1, :] |= e; d[1:, :] |= e
        order[d] = Ak
    R = (order > 0) & ~pin
    # thin the 2-px boundary to a line
    from skimage.morphology import skeletonize
    R = skeletonize(R) & ~pin
    # the rivers either side: the highest nearest-channel z around the cell
    zr = zn[ir, ic] if zn is not None else np.zeros((H, W))
    zr = ndi.maximum_filter(np.where(net, zn, 0.0)[ir, ic], size=3)
    A_o = ndi.maximum_filter(np.where(order > 0, order, 0.0), size=3)
    if plate is not None and border:
        # THE MAP'S EDGE IS A DIVIDE (except at the outlets): on a map of one
        # basin it is the main ridge itself.  Pin it at the plate, as a
        # ridge of the highest order.
        edge = np.zeros((H, W), bool)
        edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
        seed_o = np.ones((H, W), bool)
        for (r, c, *_x) in roots:
            seed_o[int(r), int(c)] = False
        edge &= ndi.distance_transform_edt(seed_o) > 12
        edge &= ~pin
        R = R | edge
        A_o = np.where(edge, 1e3, A_o)
    h = np.where(R, zr + h0 + C * np.maximum(A_o, 1e-3) ** q, np.nan)
    if plate is not None and plate_ridge == "lowered":
        # RIDGES ARE WORN DOWN TOO, the narrow ones most: a ridge of order A
        # stands lower than the plate by `lower` m at A = 0.01 km2, falling
        # to 0 at A >= 10 km2 (linear in log A).  The main divides keep the
        # plate's height and the spurs fall away from them.
        f = np.clip(np.log10(10.0 / np.maximum(A_o, 1e-6)) / 3.0, 0.0, 1.0)
        h = np.where(R, plate - lower * f, np.nan)
    elif plate is not None and plate_ridge == "graded":
        # A RIDGE FALLS AWAY FROM THE MAIN DIVIDE.  The main ridges (the
        # highest order present, and the map's edge with `border`) sit at the
        # plate; every other ridge cell lies lower by r_slope per metre of
        # path ALONG THE RIDGE NETWORK from them, so a spur descends from its
        # root on the main ridge to its tip at the confluence -- the river
        # profile turned upside down.  Never below the valley beside it.
        from scipy import sparse
        from scipy.sparse import csgraph
        from experiments.flood import grid_edges
        main = R & (A_o >= min(main_A, float(A_o[R].max())))
        A_, B_ = grid_edges(H, W, R)
        (ra_, ca_), (rb_, cb_) = np.divmod(A_, W), np.divmod(B_, W)
        ln = np.hypot(ra_ - rb_, ca_ - cb_) * PX_M
        G = sparse.coo_matrix((ln, (A_, B_)), shape=(H * W, H * W)).tocsr()
        src = np.flatnonzero(main.ravel())
        dist = csgraph.dijkstra(G, directed=False, indices=src,
                                min_only=True).reshape(H, W)
        dist = np.where(np.isfinite(dist), dist, np.nanmax(
            np.where(np.isfinite(dist), dist, 0.0)))
        h = np.where(R, np.maximum(plate - r_slope * dist, zr + 2.0), np.nan)
    elif plate is not None:
        # what is left of the plate: no ridge stands above it.  "level"
        # pins every ridge AT the plate (the cap below lowers the ones the
        # valleys have eaten); "cap" keeps the ridge rule and only clips it
        h = np.where(R, plate, np.nan) if plate_ridge == "level" \
            else np.minimum(h, plate)
    from experiments.ridges import smooth_on_ridge
    h = np.nan_to_num(smooth_on_ridge(h, R, n=smooth_n))
    if cap:
        env = envelope(pin, np.where(pin, zn, 0.0), cap)
        h = np.minimum(h, env)
        if level_n:
            # A CREST RUNS LEVEL.  The cap alone leaves a summit between
            # every two valley heads and a notch above each of them, so the
            # crest line goes up and down at the spacing of the gullies.  A
            # real crest is worn down to a line that falls slowly along the
            # ridge: here it is the capped height averaged along the ridge,
            # and never above the cap.
            # Only the main ridges (order >= level_A km2): the small spurs
            # keep their ups and downs, which is the fine grain of the hills.
            h_raw = h
            for _ in range(2):
                h = np.minimum(h, np.nan_to_num(
                    smooth_on_ridge(np.where(R, h, np.nan), R, n=level_n)))
            main_r = A_o >= level_A
            h = np.where(main_r, h, h_raw)
    if dome:
        # NO SMALL SUMMIT ON A CREST.  Every summit of the crest line is cut
        # down by up to `dome` m, as far as the ground it stands on (the
        # h-maxima transform along the ridge cells): a rise of less than
        # that is gone, a real peak keeps its place with a level top.  Main
        # ridges only, like the levelling.  min_piece > 0 also drops the
        # scraps of ridge of fewer cells (a bump on its own between two
        # gullies); off by default, they are part of the fine grain.
        from skimage.morphology import reconstruction
        rl, rn = ndi.label(R, np.ones((3, 3)))
        if rn and min_piece:
            sz = ndi.sum(R, rl, np.arange(rn + 1))
            R = R & (sz[rl] >= min_piece)
        lo = float(h[R].min()) - 2.0 * dome if R.any() else 0.0
        hm = np.where(R, h, lo)
        cut = np.where(R & (A_o >= level_A), dome, 0.0)   # main ridges only
        h = np.where(R, reconstruction(np.where(R, h - cut, lo), hm,
                                       footprint=np.ones((3, 3))), 0.0)
    if w0:
        # BIG RIDGES ARE BROAD.  A ridge of order A gets a rounded crest of
        # half-width w0 * A**p px: pinned as a parabola falling from the
        # crest line at `edge` gradient at the band's edge.  Small spurs
        # stay knife-thin.
        dR, (jy, jx) = ndi.distance_transform_edt(~R, return_indices=True)
        w = w0 * np.clip(A_o, 1e-6, 10.0) ** p
        wn = w[jy, jx]
        hn = h[jy, jx]
        band = (dR <= wn) & (wn >= 1.0) & ~pin
        zb = hn - edge * PX_M * dR ** 2 / (2.0 * np.maximum(wn, 1e-9))
        h = np.where(R, h, np.where(band, zb, 0.0))
        R = R | band
    return R, np.where(R, h, 0.0)


def regrow_small(net, roots, rng, a_keep=0.2, sc=0.3, amp=12.0, sigma=2.0,
                 max_steps=400):
    """Redraw the small rivers as water would run on the ground the big ones
    make.

    The flood tree follows the lowest pass of random ground, so a headwater
    meanders with the noise -- the "worm" texture of the uplands.  A real
    headwater runs straight down its hillslope.  Here the rivers draining at
    least a_keep km2 are kept as a skeleton; a smooth provisional surface
    rises from it (the skeleton's own Flint profile + sc per metre of
    distance + gentle noise of amplitude amp m, blurred by sigma px); and
    from every head of the original network a streamline is traced down
    that surface's gradient until it meets the skeleton or a streamline
    already drawn.  The heads keep the original drainage density.  Returns
    the new 1-px network."""
    from skimage.draw import line
    from skimage import morphology as skmo
    from experiments.mini import rejoin_after_thinning
    from experiments.flood import noise
    net = np.asarray(net, bool)
    H, W = net.shape
    rr, cc, idx, parent, step, area = channel_area(net, roots)
    keep = area >= a_keep
    skel = np.zeros((H, W), bool)
    skel[rr[keep], cc[keep]] = True
    for (r, c, *_x) in roots:
        skel[int(r), int(c)] = True
    has_kid = np.zeros(len(rr), bool)
    has_kid[parent[parent >= 0]] = True
    heads = np.flatnonzero(~has_kid & ~keep)
    # provisional surface
    z = integrate_profile(parent, step, slope_of(area))
    zmap = np.zeros((H, W)); zmap[rr, cc] = z
    d, (iy, ix) = ndi.distance_transform_edt(~skel, return_indices=True)
    nz = noise(rng, H, W, smin=6.0, beta=0.5, fast=True)
    nz = (nz - nz.mean()) / (nz.std() or 1.0)
    zp = zmap[iy, ix] + sc * d * PX_M + amp * nz
    zp = ndi.gaussian_filter(zp, sigma)
    gy, gx = np.gradient(zp)
    gn = np.hypot(gy, gx) + 1e-12
    out = skel.copy()
    # highest heads first, so later ones can join the streams drawn before
    heads = heads[np.argsort(-zp[rr[heads], cc[heads]], kind="stable")]
    for h in heads:
        y, x = float(rr[h]), float(cc[h])
        pts = [(rr[h], cc[h])]
        for _ in range(max_steps):
            ri, ci = int(round(y)), int(round(x))
            if out[ri, ci] and len(pts) > 1:
                break
            # step one px down the gradient; on a flat or a pit fall back
            # to the lowest neighbour
            g = gn[ri, ci]
            ny, nx = y - gy[ri, ci] / g, x - gx[ri, ci] / g
            nr, nc = int(round(ny)), int(round(nx))
            if not (0 <= nr < H and 0 <= nc < W) or zp[nr, nc] >= zp[ri, ci]:
                r0, r1 = max(0, ri - 1), min(H, ri + 2)
                c0, c1 = max(0, ci - 1), min(W, ci + 2)
                win = zp[r0:r1, c0:c1]
                k = np.unravel_index(np.argmin(win), win.shape)
                nr, nc = r0 + k[0], c0 + k[1]
                if zp[nr, nc] >= zp[ri, ci]:
                    break                       # a true pit: give up
                ny, nx = float(nr), float(nc)
            y, x = ny, nx
            pts.append((nr, nc))
        else:
            continue
        if not out[pts[-1]]:
            continue                            # never reached the network
        for (a0, b0), (a1, b1) in zip(pts[:-1], pts[1:]):
            lr, lc = line(a0, b0, a1, b1)
            out[lr, lc] = True
    net2 = rejoin_after_thinning(out, skmo.skeletonize(out), roots)
    for (r, c, *_x) in roots:
        net2[int(r), int(c)] = True
    return net2
