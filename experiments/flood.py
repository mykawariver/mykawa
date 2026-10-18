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
  * rim: the map edge is a divide except at the outlets; without a slight
    rim, trunks run along the edge;
  * edge_a / edge_l: a head sits about a hillslope length below a divide, so
    near the edge the threshold drops; otherwise the edge band is left
    undissected and the stage 2 surface gets a high frame;
  * funnel / stem: every outlet sits at the foot of its own valley mouth and
    starts with a `stem`-px reach along its heading (gen_headward likewise
    starts every root with a stroke); otherwise the outlet with the lowest
    pass takes most of the map and others drain a single cell.

`domain` is accepted (the flood stays on the land) but the rim, the edge
threshold and the funnel know only the rectangle's edges, not a coast;
untested on shaped maps.
"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.sparse import csgraph
from skimage import morphology as skmo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.mini import rejoin_after_thinning

OFFS = ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)))


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


def outlet_of(parent):
    """The outlet (flat index) each cell drains to."""
    anc = np.arange(len(parent))
    p = parent.copy()
    while True:
        live = p >= 0
        if not live.any():
            return anc
        anc[live] = p[live]
        p[live] = parent[p[live]]


def gen_flood(rng, n=128, roots=None, shape=None, domain=None, labels=False,
              p50=4.2, smin=4.0, beta=0.5, fine=1.5, fine_s=1.5, rim=0.8,
              rim_l=3.0, edge_a=0.9, edge_l=8.0, funnel=3.0, funnel_l=8.0,
              stem=10):
    """Stage 1 by priority flood.  Same roots / shape / domain / labels
    contract as mini.gen_headward: returns the 1-px network, and with
    labels=True also the int16 basin raster (-1 off the map, 0 land, k > 0
    channel draining to outlet k)."""
    H, W = (int(n), int(n)) if shape is None else (int(shape[0]), int(shape[1]))
    if roots is None:
        roots = [(H - 1, W // 2, -np.pi / 2)]
    mask = None if domain is None else np.asarray(domain, bool)
    iy = np.arange(H)[:, None]
    ix = np.arange(W)[None, :]

    h = noise(rng, H, W, smin, beta)
    if fine:
        g = ndi.gaussian_filter(rng.standard_normal((H, W)), fine_s)
        h += fine * g / g.std()
    if funnel:
        for (r, c, *_rest) in roots:
            h -= funnel * np.exp(-np.hypot(iy - int(r), ix - int(c)) / funnel_l)
    if stem:
        for (r, c, a0, *_rest) in roots:
            for k in range(int(stem)):
                rr = int(round(r + k * np.sin(a0)))
                cc = int(round(c + k * np.cos(a0)))
                if 0 <= rr < H and 0 <= cc < W:
                    h[rr, cc] = -1e3 + k
    edge = np.ones((H, W), bool)
    edge[1:-1, 1:-1] = False
    if rim:
        e = edge.copy()
        for (r, c, *_rest) in roots:
            e[max(0, int(r) - 2):int(r) + 3, max(0, int(c) - 2):int(c) + 3] = False
        h += rim * np.exp(-ndi.distance_transform_edt(~e) / rim_l)

    parent = flood_tree(h, roots, mask)
    depth = depth_of(parent)
    acc = catchment_sum(parent, depth)
    thr = (1.0 - edge_a * np.exp(-ndi.distance_transform_edt(~edge)
                                 / edge_l)).ravel()

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

    def raw_net(Ac):
        seed = (acc >= Ac * thr) | stem_cells
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
    raw = raw_net(np.exp(lo))
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
