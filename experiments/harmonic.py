"""調和補間ソルバ（2骨 Dirichlet）と、ピン場のリプシッツ上限。

`lipschitz_cap` は mini.solve の `cap_px` 経路（ゼロでは無効）と
`solve_two_bones(pin_cap_px=...)` が使う。
"""
import math

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def lipschitz_cap(z, mask, slope_px, iters=None):
    """谷ピン場に閾値斜面の上限を掛ける（円錐の下側包絡への射影）。

    z_cap(p) = min_q [ z(q) + slope_px * d(p, q) ]  （q は mask 画素）
    どの谷ピンも、他の谷ピンから閾値斜面 slope_px を超えて高くなれない。
    「z の高い涸れ沢が低い本流の数 px 隣にピンされて壁（等高線の密集）を
    作る」を防ぐ。チャンファ緩和を反復して解く。"""
    H, W = z.shape
    A = np.where(mask & np.isfinite(z), z, np.inf)
    if iters is None:
        iters = min(max(H, W), int(1.2 / max(slope_px, 1e-6)))
    d2 = slope_px * math.sqrt(2.0)
    INF = np.inf
    for _ in range(iters):
        B = A.copy()
        # 8近傍シフト（枠は inf パディング相当）
        B[1:, :] = np.minimum(B[1:, :], A[:-1, :] + slope_px)
        B[:-1, :] = np.minimum(B[:-1, :], A[1:, :] + slope_px)
        B[:, 1:] = np.minimum(B[:, 1:], A[:, :-1] + slope_px)
        B[:, :-1] = np.minimum(B[:, :-1], A[:, 1:] + slope_px)
        B[1:, 1:] = np.minimum(B[1:, 1:], A[:-1, :-1] + d2)
        B[1:, :-1] = np.minimum(B[1:, :-1], A[:-1, 1:] + d2)
        B[:-1, 1:] = np.minimum(B[:-1, 1:], A[1:, :-1] + d2)
        B[:-1, :-1] = np.minimum(B[:-1, :-1], A[1:, 1:] + d2)
        with np.errstate(invalid="ignore"):
            changed = bool(np.any((A - B) > 1e-9))  # inf→有限の前進も True
        A = B
        if not changed:
            break
    return np.where(mask, A, np.nan)


def _grid_laplacian(shape, domain=None):
    """4-neighbour graph Laplacian (Neumann at the border).

    shape: int N (square) or an (H, W) tuple.

    domain: a bool mask of the cells that are IN the problem.
    Only edges with both ends inside are kept, so the degree of a cell on the
    domain's edge counts its in-domain neighbours only -- that is Neumann on
    an arbitrary boundary, the same free edge the array border already had.
    Cells outside get an identity row from `solve_two_bones`, so they leave
    the system without pulling on anything inside it.  domain=None is the
    whole array."""
    if isinstance(shape, (int, np.integer)):
        shape = (shape, shape)
    H, W = shape
    n = H * W
    idx = np.arange(n).reshape(H, W)
    if domain is None:
        keep_h = np.ones((H, W - 1), bool)
        keep_v = np.ones((H - 1, W), bool)
    else:
        d = np.asarray(domain, bool)
        keep_h = d[:, :-1] & d[:, 1:]
        keep_v = d[:-1, :] & d[1:, :]
    r, c = [], []
    a = idx[:, :-1][keep_h]; b = idx[:, 1:][keep_h]; r += [a, b]; c += [b, a]
    a = idx[:-1, :][keep_v]; b = idx[1:, :][keep_v]; r += [a, b]; c += [b, a]
    r = np.concatenate(r); c = np.concatenate(c)
    Wm = sp.coo_matrix((np.ones(r.size), (r, c)), shape=(n, n)).tocsr()
    deg = np.asarray(Wm.sum(1)).ravel()
    return (sp.diags(deg) - Wm).tolil()



def solve_two_bones(valley, zn, ridge_mask, ridge_z, poisson_f=0.0,
                    pin_cap_px=None, channel_cap_frac=1.0, cap_env=None,
                    domain=None):
    """2骨（谷 zn・尾根 z）を Dirichlet に ∇²u = -f を解く。

    poisson_f=0 で
    調和補間、スカラー >0 で一様ソースのポアソン、
    (N,N) 配列で場所依存ソース（curvature_source）。cap_env を与えると
    骨外で u ≤ cap_env に切り詰める（尾根包絡の安全弁）。
    pin_cap_px: 全ピン（谷＋尾根）の合同リプシッツ上限（z/px）。尾根アーク
    沿いの z スパイク（針）や尾根と反対側の谷の間の壁も縛る。None で無効。"""
    fixed = valley | ridge_mask
    fval = np.zeros(valley.shape)
    fval[valley] = zn[valley]
    fval[ridge_mask] = ridge_z[ridge_mask]
    if pin_cap_px:
        # channel_cap_frac < 1 で「チャネルは丘腹より緩い」二重勾配キャップに
        # なるが、全域の谷が深くなるだけで効果は悪かった。既定 1.0＝単一勾配。
        zpin = np.where(fixed, fval, np.nan)
        env_r = lipschitz_cap(zpin, fixed, pin_cap_px)
        env_v = lipschitz_cap(zpin, fixed, pin_cap_px * channel_cap_frac)
        fval = np.where(valley, np.minimum(fval, env_v),
                        np.where(fixed, np.minimum(fval, env_r), fval))

    # domain: off-map cells are pinned to 0 and disconnected, so
    # they contribute nothing; the caller replaces them with NaN.  Pinning
    # them is not a boundary condition on the land -- the Laplacian above has
    # already dropped every edge that crosses the coast.
    if domain is not None:
        dom = np.asarray(domain, bool)
        fixed = fixed | ~dom
        fval = np.where(dom, fval, 0.0)
    L = _grid_laplacian(valley.shape, domain)
    if np.any(poisson_f):
        b = np.broadcast_to(np.asarray(poisson_f, float),
                            valley.shape).ravel().copy()
        b[fixed.ravel()] = fval.ravel()[fixed.ravel()]
    else:
        b = fval.ravel().copy()
    for k in np.where(fixed.ravel())[0]:
        L.rows[k] = [k]; L.data[k] = [1.0]
    u = spsolve(L.tocsr(), b).reshape(valley.shape)
    if cap_env is not None:
        u = np.where(fixed, u, np.minimum(u, cap_env))
    return np.where(dom, u, np.nan) if domain is not None else u

