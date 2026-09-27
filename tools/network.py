"""尾根の位相計器：卓越度（prominence）と頂の数。

mini ベンチのスコアボード（`summits`）が使う。
"""
import numpy as np
from scipy import ndimage as ndi


def prominences(u):
    """Topographic prominence of every summit, by descending union-find
    (persistent homology of the superlevel sets).

    prominence(peak) = peak z - the z of the col at which its component first
    merges into one holding a higher peak.  Counting summits above a
    prominence threshold is the honest version of '頂が多すぎる': it ignores
    the pixel-scale bumps that a plain local-maximum count is dominated by.
    Centre-60% crop, threshold 5% of relief: real kawauchi 5, kameyama 13,
    sendai 27; generator 14-31.  Returns [(row, col, z, prominence)] sorted
    by decreasing prominence."""
    u = np.asarray(u, float)
    H, W = u.shape
    flat = u.ravel()
    order = np.argsort(-flat, kind="stable")
    parent = np.full(H * W, -1, np.int64)
    top, prom = {}, {}
    seen = np.zeros(H * W, bool)

    def find(a):
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:
            parent[a], a = r, parent[a]
        return r

    for idx in order:
        idx = int(idx)
        r, c = divmod(idx, W)
        parent[idx] = idx
        seen[idx] = True
        top[idx] = idx
        roots = set()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and seen[rr * W + cc]:
                    roots.add(find(rr * W + cc))
        if not roots:
            continue                                   # a brand-new summit
        best = max(roots, key=lambda rt: flat[top[rt]])
        for rt in roots:
            if rt == best:
                continue
            pk = top[rt]
            prom[pk] = float(flat[pk] - flat[idx])     # col = current level
            parent[rt] = best
        parent[find(idx)] = best
        top[find(best)] = top[best]
    gmax = top[find(int(order[0]))]
    prom[gmax] = float(flat.max() - flat.min())
    out = [(p // W, p % W, float(flat[p]), v) for p, v in prom.items()]
    out.sort(key=lambda t: -t[3])
    return out


def summit_count(u, frac=0.05):
    """How many summits stand at least `frac` of the relief above their col."""
    u = np.asarray(u, float)
    rel = u.max() - u.min()
    if rel <= 0:
        return 0
    return int(sum(1 for p in prominences(u) if p[3] >= frac * rel))
