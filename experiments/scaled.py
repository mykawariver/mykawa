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
KS = 0.035          # channel slope at A = 1 km2
THETA = 0.45        # Flint exponent
S_MAX = 0.35        # headwater slope cap
W1 = 60.0           # floor width [m] at A = 1 km2 (opens from A0 = 0.1 km2)
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


def solve_m(net, roots, labels=None, ks=KS, theta=THETA, s_max=S_MAX, w1=W1,
            f=F_POISSON, floor_tilt=0.02, hill="envelope", sc=0.3,
            sc_var=0.3, crest_sigma=1.5, rng=None, floor_close=2, a0=0.1,
            stream_var=0.0, theta_var=0.0, lith_var=0.0):
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
            nz = noise(rng, H, W, smin=32.0, beta=0.5)
            nz = (nz - nz.mean()) / (nz.std() or 1.0)
            ksn = ksn * np.exp(lith_var * nz[rr, cc])
        # the outlet's own stream keeps the plain law, so the trunk and the
        # map's relief stay where the defaults put them
        trunk = sid == sid[np.argmax(area)]
        ksn[trunk], thn[trunk] = ks, theta
    S = slope_of(area, ksn, thn, s_max)
    own = idx[oy[rr, cc], ox[rr, cc]]
    onf = (own >= 0) & (own != np.arange(len(rr)))
    onf[onf] &= area[own[onf]] > 2.0 * area[onf]
    S[onf] = slope_of(area[own[onf]], ksn[own[onf]], thn[own[onf]], s_max)
    zc = integrate_profile(parent, step, S)
    zmap = np.zeros((H, W)); zmap[rr, cc] = zc
    # a strip of ground left between two floors (a tributary running beside
    # the trunk before it joins) would rise as a knife-edge at the threshold
    # slope; it is a low terrace in the real thing, so close the floor over
    # gaps of a few px and fill the small holes it leaves
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
    if labels is not None:
        floor = clip_to_own_basin(floor, labels)
    pin = band | floor
    # THE FLOOR TAKES ITS HEIGHT FROM ITS OWN RIVERS: the lowest of
    # z(channel) + floor_tilt * distance, along paths that stay on the pinned
    # ground.  The plain nearest channel is often a high headwater across a
    # narrow divide (a wall along the floor's edge), and the nearest channel
    # along the floor still leaves a step where a perched tributary's floor
    # meets the trunk's; the envelope turns every such step into a ramp.
    zn = envelope(net, zmap, floor_tilt, mask=pin)
    bad = ~np.isfinite(zn)
    zn = np.where(bad, zmap[iyn, ixn] + floor_tilt * _dn * PX_M, zn)
    dom = None
    if labels is not None and (np.asarray(labels) < 0).any():
        dom = np.asarray(labels) >= 0
    if hill == "two_bones":
        from experiments.ridges import ridge_bone
        R, h = ridge_bone(net, pin, np.where(pin, zn, 0.0))
        u = EL.solve_two_bones(pin, np.where(pin, zn, 0.0), R, h,
                               poisson_f=0.0)
    elif hill == "poisson":
        z0 = np.zeros_like(pin)
        u = EL.solve_two_bones(pin, np.where(pin, zn, 0.0), z0, np.zeros((H, W)),
                               poisson_f=f, domain=dom)
    else:
        scf = float(sc)
        if sc_var and rng is not None:
            from experiments.flood import noise
            nz = noise(rng, H, W, smin=8.0, beta=0.5)
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
    return u, dict(area=amap, floor=floor, zc=zmap)


def generate_scaled(seed, shape=(128, 128), outlets=None, n_roots=None,
                    regional=2.0, ell=5.0, floor_px=4.0, finish=True,
                    ks=KS, theta=THETA, s_max=S_MAX, w1=W1, f=F_POISSON,
                    z0=Z0, finish_kw=None, hill="envelope", sc=0.3, sc_var=0.3,
                    crest_sigma=1.5, straighten=2.0, s0=1.0, floor_close=2,
                    stage1="flood", trend=None, stream_var=0.4, theta_var=0.1,
                    lith_var=0.3):
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
    if stage1 == "flood":
        net, labels = gen_flood(rng, roots=roots, shape=(H, W), ell=ell,
                                floor_px=floor_px, labels=True,
                                regional=regional)
    else:
        from experiments.mini import gen_headward
        from experiments.baseline import hydraulic_reach
        reach = None if (H, W) == (N, N) else hydraulic_reach((H, W), roots)
        net, labels = gen_headward(rng, n=H, ell=ell, roots=roots,
                                   shape=(H, W), reach=reach,
                                   floor_px=floor_px, labels=True,
                                   trend=trend)
    if straighten:
        from experiments.mini import basin_labels
        net = smooth_network(net, roots, c=straighten, s0=s0)
        labels = basin_labels(net, roots)
    u, info = solve_m(net, roots, labels, ks, theta, s_max, w1, f, hill=hill,
                      sc=sc, sc_var=sc_var, crest_sigma=crest_sigma, rng=rng,
                      floor_close=floor_close, stream_var=stream_var,
                      theta_var=theta_var, lith_var=lith_var)
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
