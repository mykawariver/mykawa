"""
flood -- a fast stage 1: rivers from a priority flood over random ground.

    net, lab = gen_flood(rng, n=128, roots=roots, labels=True)

A different way to draw stage 1 than gen_headward, with its own character
(wigglier channels, fewer and longer links), and fast on big maps: about
0.07 s at n=128, 2 s at n=512 and 11 s at n=1024, where gen_headward takes
about 1 s at n=128 and 40 s at n=384 (its cost grows as n**4).

HOW IT WORKS
  1. random ground: multi-octave smooth noise plus a fine, slightly smoothed
     noise that breaks up parallel flow on smooth slopes;
  2. a priority flood from the outlets -- the minimum spanning forest of the
     8-neighbour grid with edge weight max(h_u, h_v), rooted at the outlets --
     gives every cell the neighbour it drains to;
  3. channels are the cells whose catchment exceeds a threshold, chosen by
     bisection so that the median land-to-channel distance is `p50` px (4.2,
     what gen_headward gives at n=128); every cell downstream of a channel
     cell is channel too, so the network always reaches its outlet;
  4. thinned to 1 px with mini.rejoin_after_thinning, like gen_headward.

FIXES FOR WHAT A PLAIN FLOOD GETS WRONG (each measured at n=128, seeds 1-10)
  * boundary: the map's outer ring and the coast are divides except at the
    outlets; their cells are left out of the flood, so no river runs on
    them, and a slight rim keeps trunks from running just inside them;
  * edge_a / edge_l: a head sits about a hillslope length below a divide, so
    near the edge the threshold drops; otherwise the edge band is left
    undissected and the stage 2 surface gets a high frame;
  * funnel / stem: every outlet sits at the foot of its own valley mouth and
    starts with a `stem`-px reach along its heading (gen_headward likewise
    starts every root with a stroke); otherwise the outlet with the lowest
    pass takes most of the map and others drain a single cell.

With `domain`, the coast is part of that boundary: rivers reach the sea only
at the outlets.
"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.sparse import csgraph
from skimage import morphology as skmo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.mini import ACC_REF_NET, cone_floor, rejoin_after_thinning

OFFS = ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)))
# gen_headward's median land-to-channel distance is 0.85-0.88 x ell over
# density -2..+2 (n=128, seeds 1-5); 0.852 at ell = 5
P50_PER_ELL = 0.852
# The floor acts here after the network is drawn, while gen_headward never
# grows a head into ground within reach of a floor in the first place; twice
# the corridor half-width reproduces its response to `floor` (n=128, seeds
# 1-5, floor +2: median channel distance +1.0 vs +1.2, density -1.0 vs -0.8).
FLOOR_REACH = 2.0
# ...and the floor pass and the thinning raise the median channel distance
# above the target the threshold is bisected on; this factor takes it back
# to P50_PER_ELL * ell at floor 0 (calibrated over density -2..+2).
P50_FIT = 0.91


def noise(rng, H, W, smin=4.0, beta=0.5):
    """Octaves of gaussian-smoothed white noise from `smin` px up to half the
    map, each scaled to std s**beta; the sum has unit std."""
    h = np.zeros((H, W))
    s = smin
    while s <= max(H, W) / 2:
        g = ndi.gaussian_filter(rng.standard_normal((H, W)), s)
        h += (s ** beta) * g / g.std()
        s *= 2
    return h / h.std()


def oriented_noise(rng, H, W, s, ratio, theta, nbins=8):
    """Gaussian-smoothed white noise whose blobs are `ratio` times longer
    along the local direction theta (radians, an (H, W) field; theta and
    theta + pi are the same) than across it (std s px).  One white field,
    filtered in Fourier space at nbins orientations and blended linearly
    between the two nearest; zero mean, unit std."""
    wf = np.fft.rfft2(rng.standard_normal((H, W)))
    ky = np.fft.fftfreq(H)[:, None]
    kx = np.fft.rfftfreq(W)[None, :]
    phis = np.arange(nbins) * np.pi / nbins
    # position of theta between the bins, mod pi
    t = np.mod(theta, np.pi) / (np.pi / nbins)
    k0 = np.floor(t).astype(int) % nbins
    fr = t - np.floor(t)
    out = np.zeros((H, W))
    sa, sc = float(s) * float(ratio), float(s)
    for k, ph in enumerate(phis):
        ka = ky * np.sin(ph) + kx * np.cos(ph)
        kc = -ky * np.cos(ph) + kx * np.sin(ph)
        g = np.fft.irfft2(wf * np.exp(-2 * np.pi ** 2 * (sa ** 2 * ka ** 2
                                                         + sc ** 2 * kc ** 2)),
                          s=(H, W))
        g = (g - g.mean()) / (g.std() or 1.0)
        wgt = np.where(k0 == k, 1.0 - fr, 0.0) + np.where((k0 + 1) % nbins == k, fr, 0.0)
        out += wgt * g
    return (out - out.mean()) / (out.std() or 1.0)


def grid_edges(H, W, mask=None):
    """The 8-neighbour grid as edge lists (a, b), each pair once."""
    idx = np.arange(H * W).reshape(H, W)
    A, B = [], []
    for dr, dc, _ln in OFFS:
        c0, c1 = max(0, -dc), W - max(0, dc)
        A.append(idx[0:H - dr, c0:c1].ravel())
        B.append(idx[dr:H, c0 + dc:c1 + dc].ravel())
    A = np.concatenate(A)
    B = np.concatenate(B)
    if mask is not None:
        m = mask.ravel()
        k = m[A] & m[B]
        A, B = A[k], B[k]
    return A, B


def flood_tree(h, roots, mask=None):
    """Priority flood from the outlets, as the minimum spanning forest of the
    grid with edge weight max(h_u, h_v); a virtual node joins the outlets.
    Returns parent: the flat index each cell drains to, -1 at the outlets
    (and on cells the flood never reaches)."""
    H, W = h.shape
    N = H * W
    A, B = grid_edges(H, W, mask)
    w = np.maximum(h.ravel()[A], h.ravel()[B])
    w = w - w.min() + 1.0                        # the MST needs weights > 0
    R = np.array([int(r) * W + int(c) for (r, c, *_rest) in roots])
    A = np.concatenate([A, np.full(len(R), N)])
    B = np.concatenate([B, R])
    w = np.concatenate([w, np.full(len(R), 1e-9)])
    G = sparse.coo_matrix((w, (A, B)), shape=(N + 1, N + 1)).tocsr()
    T = csgraph.minimum_spanning_tree(G)
    _order, pred = csgraph.breadth_first_order(T, N, directed=False,
                                               return_predecessors=True)
    parent = pred[:N].copy()
    parent[(parent == N) | (parent < 0)] = -1
    return parent


def depth_of(parent):
    """Steps from each cell to its outlet, by pointer jumping."""
    anc = parent.copy()
    dist = (parent >= 0).astype(np.int64)
    while True:
        live = anc >= 0
        if not live.any():
            return dist
        a = anc[live]
        dist[live] = dist[live] + dist[a]
        anc[live] = anc[a]


def _levels(depth):
    order = np.argsort(-depth, kind="stable")
    cuts = np.flatnonzero(np.diff(depth[order])) + 1
    return np.split(order, cuts)


def catchment_sum(parent, depth, w=None):
    """Sum of w over each cell's catchment, itself included (w=None: cells)."""
    acc = np.ones(len(parent)) if w is None else np.asarray(w, float).copy()
    for grp in _levels(depth):
        p = parent[grp]
        k = p >= 0
        np.add.at(acc, p[k], acc[grp[k]])
    return acc


def _jump_to(parent, stop):
    """For every cell, the first cell at or below it (towards the outlet)
    where stop is True, or its outlet."""
    n = len(parent)
    f = np.where(stop | (parent < 0), np.arange(n), parent)
    while True:
        g = f[f]
        if np.array_equal(g, f):
            return f
        f = g


def divides(parent, acc, H, W, a_min):
    """The divides between the rivers of at least a_min cells of catchment,
    on the drainage tree `parent`: every cell joins the link (the reach
    between two junctions) of the first such river below it, and a divide
    is where two cells side by side join different links -- except two
    links of which one drains into the other: that boundary crosses the
    valley at the junction, it is no ridge.  Returns a bool (H, W) map."""
    n = H * W
    on = acc >= a_min
    entry = _jump_to(parent, on)
    pj = parent >= 0
    nkids = np.bincount(parent[pj & on], minlength=n)
    junction = np.zeros(n, bool)
    junction[pj] = nkids[parent[pj]] >= 2
    link = _jump_to(parent, junction | ~on)
    lab = np.where(on[entry], link[entry], -1)
    lpar = np.full(n, -1)
    roots_ = lab >= 0
    pl = parent[link[roots_]]
    lpar_cells = np.where(pl >= 0, lab[np.maximum(pl, 0)], -1)
    lpar[link[roots_]] = lpar_cells
    lab = lab.reshape(H, W)
    off = (parent < 0).reshape(H, W) & (acc < a_min).reshape(H, W)
    d = np.zeros((H, W), bool)
    for sl_a, sl_b in (((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                       ((slice(None, -1), slice(None)), (slice(1, None), slice(None)))):
        a, b = lab[sl_a], lab[sl_b]
        k = (a != b) & (a >= 0) & (b >= 0) & ~off[sl_a] & ~off[sl_b]
        k &= (lpar[np.maximum(a, 0)] != b) & (lpar[np.maximum(b, 0)] != a)
        d[sl_a] |= k
        d[sl_b] |= k
    return d


def outlet_of(parent):
    """The outlet (flat index) each cell drains to.

    By pointer jumping: each cell's pointer goes to its parent (an outlet to
    itself), then is replaced by its pointer's pointer until nothing moves --
    log(depth) rounds instead of one per step down a long trunk."""
    anc = np.where(parent >= 0, parent, np.arange(len(parent)))
    while True:
        nxt = anc[anc]
        if np.array_equal(nxt, anc):
            return anc
        anc = nxt


def _floor_of(raw, parent, depth, W, floor_px, corridor, acc_ref):
    """The valley floor of the network, as cone_floor paints it from the
    corridor half-width hw = corridor * (1 + (floor_px - 1) sqrt(A / acc_ref)),
    leaving out the 1-px channels' own cells (hw <= 1 is no floor)."""
    H = raw.shape[0]
    ch = raw.ravel()
    A = catchment_sum(parent, depth, ch) * ch
    hw = np.where(ch, float(corridor) * (1.0 + (floor_px - 1.0)
                                         * np.sqrt(A / float(acc_ref))), 0.0)
    hw = np.where(hw > 1.0, hw, 0.0)
    floor, _own = cone_floor(raw, hw.reshape(H, W))
    return floor & ~raw


def _corridor(raw, parent, depth, W, roots, floor_px, corridor, acc_ref,
              cut_max=np.inf):
    """Drop channel reaches that run inside a bigger river's valley floor.

    gen_headward never lets a stroke enter another channel's floor except at
    its own mouth, so a tributary may cross the floor to join, but not run
    alongside inside it.  Here the network already exists: a channel cell is
    INTRUDING when it lies in the floor of a channel cell y of a bigger river
    (cone dilation, as gen_headward builds it) but joins that river more than
    hw(y) + 2 steps downstream.  Intruding cells are removed with everything
    upstream of them, as a stroke that hits the floor stops -- but only where
    that is a small headwater (at most cut_max channel cells upstream).  A
    big tributary that runs down a trunk's floor before it joins is real
    terrain, and on a long map, where a trunk's floor is tens of px wide,
    cutting it would take whole tributary trees with it."""
    H = raw.shape[0]
    ch = raw.ravel()
    A = catchment_sum(parent, depth, ch) * ch          # channel cells upstream
    hw = np.where(ch, float(corridor) * (1.0 + (floor_px - 1.0)
                                         * np.sqrt(A / float(acc_ref))), 0.0)
    floor, (oy, ox) = cone_floor(raw, hw.reshape(H, W))
    own = (oy * W + ox).ravel()
    cut = np.zeros(len(ch), bool)
    for x in np.flatnonzero(ch & floor.ravel()):
        y = own[x]
        if y == x or A[y] <= A[x]:
            continue                         # its own floor, or a smaller river
        # walk downstream to where x meets a river at least half y's size
        i, k, lim = x, 0, hw[y] + 2.0
        while i >= 0 and A[i] < 0.5 * A[y] and k <= lim:
            i = parent[i]
            k += 1
        if k > lim and A[x] <= cut_max:
            cut[x] = True
    if not cut.any():
        return raw
    return _drop_upstream(raw, cut, parent, depth, roots)


def _drop_upstream(raw, cut, parent, depth, roots):
    """raw without the `cut` cells and everything upstream of them."""
    ch = raw.ravel()
    gone = cut.copy()
    for grp in _levels(depth)[::-1]:                 # outlets first
        p = parent[grp]
        k = p >= 0
        gone[grp[k]] |= gone[p[k]]
    out = (ch & ~gone).reshape(raw.shape)
    for (r, c, *_rest) in roots:
        out[int(r), int(c)] = True
    return out


def gen_flood(rng, n=128, roots=None, shape=None, domain=None, labels=False,
              ell=5.0, floor_px=4.0, corridor=1.0, acc_ref=ACC_REF_NET,
              p50=None, smin=4.0, beta=0.5, fine=1.5, fine_s=1.5, rim=4.0,
              rim_l=1.5, edge_a=0.9, edge_l=8.0, funnel=10.0, funnel_l=12.0,
              stem=10, stem_cut=3, floor_rounds=2, rim_open=4.0,
              funnel_w=4.0, cut_ell=8.0, regional=0.0, big_s=32.0,
              calm=0.0, align=0.0, align_beta=None, align_reg=0.0,
              calm_q=(50.0, 95.0), crest_thr=0.0, crest_bg=0.0,
              ridge_room=0.0, ridge_a=7300.0, ridge_w=10.0):
    """Stage 1 by priority flood.  Same roots / shape / domain / labels
    contract as mini.gen_headward: returns the 1-px network, and with
    labels=True also the int16 basin raster (-1 off the map, 0 land, k > 0
    channel draining to outlet k).

    ell and floor_px are gen_headward's two stage-1 knobs (baseline's
    `density` and `floor`): ell sets the drainage density through the target
    median land-to-channel distance, P50_PER_ELL * ell (`p50` overrides it);
    floor_px sets the valley-floor half-width a river keeps clear of other
    channels, hw = 1 + (floor_px - 1) * sqrt(A / acc_ref), A in channel
    cells upstream -- the corridor rule of gen_headward, applied here after
    the fact (corridor=0 turns it off).

    SHAPING THE RANDOM GROUND (all off by default; the ground then is the
    plain noise above, draw for draw).  The noise has no structure at any
    scale, so the flood draws an even net everywhere and a divide falls
    wherever the noise happens to be high: ridges are left over, not made.
    Split the ground at big_s px into the large relief (the ridges and the
    basins between them) and the small-scale part (which picks where the
    small valleys go):
      calm   0..1: on the high ground of the large relief the small-scale
             part is damped by up to this factor -- smooth crests shed water
             in diverging sheets, so the heads stay a hillslope below them
             and the crests are left without channels;
      align  >= 1: the small-scale part is redrawn as blobs this many times
             longer down the large relief's slope than across it, so the
             small valleys run straight down off the ridges instead of
             following the noise sideways (0 = off).  align_beta: the
             spectral weight of its octaves (default: beta); align_reg:
             how much of the regional tilt steers the blobs (0: with it
             they all point at the outlet and the valleys run parallel);
      crest_thr: on the high ground of the large relief (calm_q: from
             which percentile to which it counts as crest) a head needs
             exp(crest_thr) times the catchment -- ground without channels
             along the ridges, which stage 2 then raises higher;
      crest_bg: px; when > 0 the crests are judged relative to the large
             relief smoothed at this scale, not by its absolute height.
      ridge_room: the same, but on the REAL divides (TWO PASSES): the
             divides between the rivers of at least ridge_a cells (7300 =
             1 km2 at 11.7 m/px) are taken from the flood tree, and within
             about ridge_w px of them a head needs exp(ridge_room) times
             the catchment.  The tree is the same, so only the heads near
             the ridges go; the noise high points, which the regional slope
             lets the trunks run through, play no part."""
    if p50 is None:
        p50 = P50_FIT * P50_PER_ELL * float(ell)
    H, W = (int(n), int(n)) if shape is None else (int(shape[0]), int(shape[1]))
    if roots is None:
        roots = [(H - 1, W // 2, -np.pi / 2)]
    mask = None if domain is None else np.asarray(domain, bool)
    iy = np.arange(H)[:, None]
    ix = np.arange(W)[None, :]

    h = noise(rng, H, W, smin, beta)
    shaped = bool(calm or align or crest_thr)
    if regional and shaped:
        seed = np.ones((H, W), bool)
        for (r, c, *_rest) in roots:
            seed[int(r), int(c)] = False
        reg = float(regional) * ndi.distance_transform_edt(seed) / 100.0
    if shaped:
        big = ndi.gaussian_filter(h, big_s / 2.0, mode="reflect")
        # 0 below the calm_q[0] percentile of the large relief, 1 above
        # calm_q[1]
        # crest_bg > 0: high RELATIVE to the large relief around it (the
        # relief smoothed again at crest_bg px taken off), so every ridge
        # counts, not just the half of the map the longest waves lift
        rel = big - ndi.gaussian_filter(big, crest_bg / 2.0, mode="reflect") \
            if crest_bg else big
        q50, q95 = np.percentile(rel, calm_q)
        crest = np.clip((rel - q50) / max(q95 - q50, 1e-9), 0.0, 1.0)
        small = h - big
        sd = small.std() or 1.0
        if align:
            # the slope of the large relief, by default without the regional
            # tilt: with it every blob points at the outlet and the small
            # valleys run parallel down the whole map
            bigr = big + (align_reg * reg if regional else 0.0)
            gy, gx = np.gradient(ndi.gaussian_filter(bigr, big_s / 4.0))
            th = np.arctan2(gy, gx)
            ab = beta if align_beta is None else align_beta
            small = np.zeros((H, W))
            s = smin
            while s < big_s:
                small += (s ** ab) * oriented_noise(rng, H, W, s, align, th)
                s *= 2
            small = small / (small.std() or 1.0)
        else:
            small = small / sd
        if calm:
            small = small * (1.0 - float(calm) * crest)
        h = big + sd * small
    if regional:
        # A REGIONAL SLOPE TOWARDS THE OUTLETS.  The noise has no direction at
        # any scale, so a big river wanders as much as a small one.  Real
        # land falls towards its outlet at the scale of the whole basin; a
        # rise of `regional` (in units of the noise's std) per 100 px of
        # distance from the nearest outlet is felt by the long rivers and
        # drowned by the noise at the scale of a hillslope.  0 = off.
        seed = np.ones((H, W), bool)
        for (r, c, *_rest) in roots:
            seed[int(r), int(c)] = False
        h = h + float(regional) * ndi.distance_transform_edt(seed) / 100.0
    if fine:
        if align:
            g = oriented_noise(rng, H, W, fine_s, align, th)
        else:
            g = ndi.gaussian_filter(rng.standard_normal((H, W)), fine_s)
        if calm:
            h += fine * (g / g.std()) * (1.0 - float(calm) * crest)
        else:
            h += fine * g / g.std()
    if funnel:
        # each outlet sits at the foot of a valley that runs INLAND along its
        # heading (deep at the mouth, fading over funnel_l, funnel_w wide).
        # A round bowl would do for ownership, but inside the rim its lowest
        # path is a band along the boundary, and tributaries slide down it.
        for (r, c, a0, *_rest) in roots:
            t = (iy - r) * np.sin(a0) + (ix - c) * np.cos(a0)
            lat = np.abs(-(iy - r) * np.cos(a0) + (ix - c) * np.sin(a0))
            h -= (funnel * np.exp(-np.maximum(t, 0.0) / funnel_l)
                  * np.exp(-(lat / funnel_w) ** 2))
    if stem_cut:
        # a few px cut straight in along the heading make the outlet own its
        # valley mouth; the rest of the stem follows the flood tree (below),
        # so it bends like any other reach
        for (r, c, a0, *_rest) in roots:
            for k in range(int(stem_cut)):
                rr = int(round(r + k * np.sin(a0)))
                cc = int(round(c + k * np.cos(a0)))
                if 0 <= rr < H and 0 <= cc < W:
                    h[rr, cc] = -1e3 + k
    # THE BOUNDARY: the land cells on the map's outer ring or next to the
    # sea.  Except at the outlets it is a divide -- a map edge and a coast
    # alike -- so no river may run on it: those cells are left out of the
    # flood altogether, and the rim and the head threshold below follow it.
    land = np.ones((H, W), bool) if mask is None else mask.copy()
    bnd = land & ~ndi.binary_erosion(np.pad(land, 1), np.ones((3, 3)))[1:-1, 1:-1]
    out_cells = np.zeros((H, W), bool)
    for (r, c, *_rest) in roots:
        out_cells[int(r), int(c)] = True
    if rim:
        # a steep rim just inside the boundary turns flow inland at once, so
        # no river runs alongside an edge or a coast.  It opens smoothly at
        # each outlet (within ~rim_open px) like a valley mouth: a gap that is
        # too wide lets the funnel draw a channel along the boundary into the
        # outlet, and none at all leaves a bank across the mouth that sends
        # the land behind it to another outlet.
        de = ndi.distance_transform_edt(~(bnd & ~out_cells))
        # the opening is a narrow gap along each outlet's heading, not a
        # disc: a disc lets tributaries slide in along the boundary from
        # either side, while a gap only lets the river leave head-on
        dmin = np.full((H, W), np.inf)
        for (r, c, a0, *_rest) in roots:
            t = np.clip((iy - r) * np.sin(a0) + (ix - c) * np.cos(a0),
                        0.0, 2.0 * rim_open)
            dmin = np.minimum(dmin, np.hypot(iy - (r + t * np.sin(a0)),
                                             ix - (c + t * np.cos(a0))))
        opening = 1.0 - np.exp(-(dmin / 1.5) ** 2)
        h += rim * np.exp(-(de - 1.0) / rim_l) * opening * land

    parent = flood_tree(h, roots, land & ~(bnd & ~out_cells))
    depth = depth_of(parent)
    acc = catchment_sum(parent, depth)
    thr = (1.0 - edge_a * np.exp(-ndi.distance_transform_edt(~bnd)
                                 / edge_l)).ravel()
    if crest_thr:
        # ROOM ON THE RIDGES: a head needs exp(crest_thr) times the
        # catchment on the high ground of the large relief, so the crests
        # are left without channels (the bisection below then puts the
        # heads it takes from them on the lower ground)
        thr = thr * np.exp(float(crest_thr) * crest).ravel()
    if ridge_room:
        dv = divides(parent, acc, H, W, float(ridge_a))
        near = np.exp(-(ndi.distance_transform_edt(~dv) / float(ridge_w)) ** 2)
        thr = thr * np.exp(float(ridge_room) * near).ravel()

    # the stem cells: from each outlet along the largest tributary
    stem_cells = np.zeros(H * W, bool)

    def kids(i):
        r0, c0 = divmod(i, W)
        out = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r0 + dr, c0 + dc
                if (dr or dc) and 0 <= rr < H and 0 <= cc < W \
                        and parent[rr * W + cc] == i:
                    out.append(rr * W + cc)
        return out

    for (r, c, *_rest) in roots:
        i = int(r) * W + int(c)
        for _k in range(int(stem)):
            stem_cells[i] = True
            ch = kids(i)
            if not ch:
                break
            i = max(ch, key=lambda j: acc[j])

    def raw_net(Ac, a=None):
        a = acc if a is None else a
        seed = (a >= Ac * thr) | stem_cells
        raw = (catchment_sum(parent, depth, seed) > 0).reshape(H, W)
        if mask is not None:
            raw &= mask
        for (r, c, *_rest) in roots:
            raw[int(r), int(c)] = True
        return raw

    lo, hi = np.log(2.0), np.log(acc.max())
    for _k in range(14):
        mid = 0.5 * (lo + hi)
        nt = raw_net(np.exp(mid))
        if np.median(ndi.distance_transform_edt(~nt)[~nt]) > p50:
            hi = mid
        else:
            lo = mid
    Ac = np.exp(lo)
    raw = raw_net(Ac)
    for _k in range(int(floor_rounds) if corridor else 0):
        # A VALLEY FLOOR DRAINS ITSELF.  Land inside a river's floor goes
        # straight to that river, so it is not catchment for a tributary on
        # the valley side; the wider the floor, the shorter the side slope and
        # the fewer side tributaries reach the threshold.  This is how
        # gen_headward's floor acts on the network too: it counts a floor as
        # drained ground and grows its heads elsewhere.
        fl = _floor_of(raw, parent, depth, W, floor_px,
                       corridor * FLOOR_REACH, acc_ref)
        raw = raw_net(Ac, catchment_sum(parent, depth, ~fl.ravel()))
    if corridor:
        raw = _corridor(raw, parent, depth, W, roots, floor_px,
                        corridor * FLOOR_REACH, acc_ref,
                        cut_max=cut_ell * float(ell))
    net = rejoin_after_thinning(raw, skmo.skeletonize(raw), roots)
    if not labels:
        return net
    ro = outlet_of(parent).reshape(H, W)
    lab = np.zeros((H, W), np.int16)
    for i, (r, c, *_rest) in enumerate(roots):
        lab[(ro == int(r) * W + int(c)) & net] = i + 1
    if mask is not None:
        lab[~mask] = -1
    return net, lab
