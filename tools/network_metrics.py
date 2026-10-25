"""network_metrics -- the numbers used to compare stage 1 generators.

    from tools.network_metrics import metrics, sweep
    metrics(net, roots, floor_px)   # one network
    sweep("flood", density=+2)      # mean over seeds 1-5 at n=128

metrics() returns, for one 1-px network:
    p50, dens, tips, len50, tj, undr   from mini.score (see its docstring)
    intr   % of non-trunk channel cells inside a trunk's valley floor
           (hw = 1 + (floor_px - 1) sqrt(A / 341), A in channel cells
           upstream) yet more than hw + 2 steps from it along the network.
           gen_headward itself scores 0-8% here (it counts some tributary
           mouths), so read it relatively.

sweep() runs generate()'s stage 1 knobs the way generate() does
(ell = 5 / 1.2**density, floor_px = 1 + 3 * 1.75**floor, outlets from
baseline.edge_roots) and averages metrics() over the seeds.  Reference,
n=128, seeds 1-5 (October 2026):

                 gen_headward               gen_flood
                 p50   dens   tips          p50   dens   tips
    density -2   6.31  4.21   34.6          5.80  4.28   25.8
    density  0   4.26  5.86   45.6          4.07  5.99   38.2
    density +2   3.06  8.08   71.4          3.00  8.26   64.6
    floor   -2   4.46  5.67   49.6          4.00  6.17   40.6
    floor   +2   5.42  5.04   41.2          4.89  4.95   29.4
"""
import os
import sys

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments import mini

ACC_REF_NET = mini.ACC_REF_NET


def net_graph(net):
    """The 8-connected graph of a 1-px network: rows, cols, index raster, csr."""
    rr, cc = np.nonzero(net)
    idx = -np.ones(net.shape, int)
    idx[rr, cc] = np.arange(len(rr))
    H, W = net.shape
    A, B = [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        j = np.where(ok, idx[np.clip(r2, 0, H - 1), np.clip(c2, 0, W - 1)], -1)
        k = ok & (j >= 0)
        A.append(np.flatnonzero(k)); B.append(j[k])
    A = np.concatenate(A); B = np.concatenate(B)
    G = sparse.coo_matrix((np.ones(len(A)), (A, B)),
                          shape=(len(rr), len(rr))).tocsr()
    return rr, cc, idx, G


def channel_acc(net, roots):
    """Channel cells upstream of each channel cell (itself included), on the
    BFS tree of the network rooted at the outlets."""
    rr, cc, idx, G = net_graph(net)
    N = len(rr)
    R = [idx[int(r), int(c)] for (r, c, *_rest) in roots
         if idx[int(r), int(c)] >= 0]
    Gu = G + G.T
    one = np.ones(len(R))
    Gu = sparse.bmat([[Gu, sparse.coo_matrix((one, (R, [0] * len(R))),
                                             shape=(N, 1))],
                      [sparse.coo_matrix((one, ([0] * len(R), R)),
                                         shape=(1, N)), None]]).tocsr()
    order, pred = csgraph.breadth_first_order(Gu, N, directed=False,
                                              return_predecessors=True)
    acc = np.ones(N + 1)
    for i in order[::-1]:
        p = pred[i]
        if p >= 0 and p != N and i != N:
            acc[p] += acc[i]
    return rr, cc, idx, G, acc[:N]


def intrusion(net, roots, floor_px):
    """% of non-trunk channel cells inside a trunk's floor (see module doc)."""
    rr, cc, idx, G, acc = channel_acc(net, roots)
    if len(rr) < 10:
        return np.nan
    hw = 1.0 + (floor_px - 1.0) * np.sqrt(acc / ACC_REF_NET)
    trunk = acc >= 50
    D = csgraph.shortest_path(G, method="D", directed=False, unweighted=True)
    pts = np.column_stack([rr, cc]).astype(float)
    E = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    intr = np.zeros(len(rr), bool)
    for t in np.flatnonzero(trunk):
        intr |= (~trunk) & (E[t] < hw[t]) & (D[t] > hw[t] + 2)
    return float(100 * intr.sum() / max((~trunk).sum(), 1))


def metrics(net, roots, floor_px=4.0):
    u = mini.solve(net, outlets=[(r, c) for r, c, *_rest in roots])
    s = mini.score(net, u)
    return dict(p50=s["p50"], dens=s["dens"], tips=s["tips"],
                len50=s["len50"], tj=s["tj"], undr=s["undr"],
                intr=intrusion(net, roots, floor_px))


def sweep(stage1, density=0.0, floor=0.0, n=128, seeds=range(1, 6)):
    from experiments.baseline import edge_roots
    from experiments.flood import gen_flood
    ell = 5.0 / 1.20 ** density
    fpx = 1.0 + 3.0 * 1.75 ** floor
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        roots = edge_roots(rng, n, n, int(round(int(rng.integers(3, 6))
                                                * n / 128)))
        if stage1 == "flood":
            net = gen_flood(rng, n=n, roots=roots, ell=ell, floor_px=fpx)
        else:
            net = mini.gen_headward(rng, n=n, roots=roots, ell=ell,
                                    floor_px=fpx)
        rows.append(metrics(net, roots, fpx))
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
