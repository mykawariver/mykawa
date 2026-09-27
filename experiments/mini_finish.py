"""mini_finish -- the fine-scale pass (stage 3).

One constraint governs it:

    THE PASS MAY ONLY REMOVE MATERIAL, ALONG DRAINAGE PATHS.

Carving micro-valleys cannot create a closed summit (a subtraction has no local
maxima of its own) and cannot create a pit either, as long as the cut deepens
monotonically downstream: the carved bed z = u - depth then still falls along
the path.  Structure is added as VECTORS (traced streamlines), never as a
field iteration.

The cuts are placed by SPACING, the same way the macro valley tree is
densified: repeatedly take the cell farthest from every channel AND every cut
already made, trace its streamline down, carve, repeat until no cell is
farther than `spacing` from some drainage line.  The hillslope is consumed
wall to wall; what survives between opposing cuts is a THIN CREST -- a ridge
you can trace as a line -- plus spurs that break up long uniform valley walls.

    python3 -m experiments.mini_finish        # score before/after + figure
"""
import os
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _blur(a, sigma, dom):
    """Gaussian blur that does not reach across the coast.

    A plain gaussian on a field with NaN outside poisons the whole map, and
    filling the outside with a number is the same mistake in a quieter form --
    the sea would then average into the land.  So blur the field and the mask
    together and divide: every weight comes from land only (normalised
    convolution).  dom=None is the plain filter, bit for bit."""
    if dom is None:
        return ndi.gaussian_filter(a, sigma)
    m = dom.astype(float)
    num = ndi.gaussian_filter(np.where(dom, a, 0.0), sigma)
    den = ndi.gaussian_filter(m, sigma)
    return np.where(dom, num / np.maximum(den, 1e-12), np.nan)


def finish_micro(u, net, rng, spacing=3.0, depth0=0.13, width=2.4,
                 rel_frac=0.6, smooth=1.5, max_steps=70, max_cuts=None,
                 refresh=12, final_sigma=0.9, domain=None):
    """u (0..1), net (1-px channels) -> u with the hillslopes dissected.

    spacing: stop when no cell is farther than this from channel or cut.
    depth of each cut = min(depth0, rel_frac * relief spanned by its path).
    max_cuts: the carve budget.  None (default) scales it with the map's
    AREA, 500 cuts per 128x128 (36x36 -> 39), the same rule generate() uses;
    pass an int to set it by hand."""
    H, W = u.shape
    if max_cuts is None:
        max_cuts = int(500 * (H * W) / (128 * 128))
    # ARBITRARY DOMAINS.  Nothing here is computed on the sea and
    # then thrown away: the blurs are normalised over land, the sea is drained
    # ground that no cut aims at, a streamline that reaches the coast has
    # drained, and the flood spills at the coast.
    dom = None if domain is None else np.asarray(domain, bool)
    band = ndi.binary_dilation(np.asarray(net, bool))
    us = _blur(u, smooth, dom)
    gy, gx = np.gradient(np.where(dom, us, 0.0) if dom is not None else us)
    gnorm = np.hypot(gy, gx) + 1e-12
    carve = np.zeros_like(u)
    yy, xx = np.mgrid[0:H, 0:W]
    aux = band.copy()                  # channels + centrelines already cut
    if dom is not None:
        aux |= ~dom                    # the sea is already drained
    d = ndi.distance_transform_edt(~aux)
    for it in range(max_cuts):
        if it % refresh == 0:
            d = ndi.distance_transform_edt(~aux)
            if d.max() <= spacing:
                break
        # the most undissected cell, with a little jitter among the top few
        flat = np.argsort(d.ravel())[-8:]
        r, c = np.unravel_index(int(rng.choice(flat)), d.shape)
        if d[r, c] <= spacing:
            continue
        path = []
        reached = False
        r_, c_ = float(r), float(c)
        for _ in range(max_steps):
            ri, ci = int(round(r_)), int(round(c_))
            if not (1 <= ri < H - 1 and 1 <= ci < W - 1):
                break
            if dom is not None and not dom[ri, ci]:
                reached = True           # ran out to the coast: it drains
                break                    # ... and is not painted off the map
            path.append((ri, ci))
            if aux[ri, ci]:
                reached = True           # drains into a channel or an old cut
                break
            g = gnorm[ri, ci]
            r_ -= gy[ri, ci] / g
            c_ -= gx[ri, ci] / g
        drains = len(path) >= 8 and reached
        pr, pc = np.array(path).T if path else (np.array([r]), np.array([c]))
        # always PAINT the centreline so the search moves on, but only CARVE a
        # path that drains -- a cut that stops mid-slope would break the
        # contour nesting and can pond.
        aux[pr, pc] = True
        if not drains:
            continue
        # smooth the course before stamping: the raw streamline inherits the
        # gradient noise and its kinks print straight into the contours as
        # spikes and hooks.
        parr = np.array(path, float)
        if len(parr) >= 5:
            k = np.ones(3) / 3.0
            parr[1:-1, 0] = np.convolve(parr[:, 0], k, mode="same")[1:-1]
            parr[1:-1, 1] = np.convolve(parr[:, 1], k, mode="same")[1:-1]
        path = [(int(round(r2)), int(round(c2))) for r2, c2 in parr]
        L = len(path)
        # depth in proportion to the relief the path actually spans (a real
        # gully is as deep as its hillslope is tall), ramp 0 at head -> full
        # at mouth so the bed still falls downstream
        dfull = min(depth0, rel_frac * float(u[path[0]] - u[path[-1]]))
        if dfull <= 0.004:
            continue
        # head depth: a gully starting AT A CREST notches it (headward erosion
        # attacks the summit -- the fix for the round domes), but a cut whose
        # head sits mid-slope must fade in from zero, or the V appears out of
        # nowhere and reads as a computational error.  Real micro-relief is only legitimate where a process starts -- a crest.
        h0 = 0.3 if u[path[0]] > np.quantile(u, 0.70) else 0.0
        # GRADE TO THE MOUTH: a gully bed cannot cut below the bed it drains
        # into, so no cell of the course is cut below the z of the cell the
        # path reaches (a channel, or an earlier cut).
        zm = float(u[path[-1]])
        for j in range(L):
            dj = dfull * (h0 + (1.0 - h0) * (j + 1) / L)
            dj = min(dj, max(float(u[path[j]]) - zm, 0.0))
            # a deeper gully is also a WIDER one: long parallel contour
            # bundles on the valley walls survive narrow slit cuts; only a cut wide
            # enough to reshape the wall breaks their similarity)
            wj = width * (0.6 + 0.8 * dj / max(depth0, 1e-9))
            R = max(3, int(2.5 * wj))
            sl = (slice(max(0, pr[j] - R), pr[j] + R + 1),
                  slice(max(0, pc[j] - R), pc[j] + R + 1))
            rr2 = (yy[sl] - pr[j]) ** 2 + (xx[sl] - pc[j]) ** 2
            stamp = dj * np.exp(-rr2 / (2 * wj ** 2))
            if dom is not None:
                stamp = np.where(dom[sl], stamp, 0.0)
            carve[sl] = np.maximum(carve[sl], stamp)
    # MOUTH FADE.  In real terrain roughness is highest at the ridges and
    # moderate at the channels; without this fade it would peak AT the
    # channels, because every cut is deepest
    # and widest at its mouth.  A real river planes its own surroundings
    # smooth, so the cuts are faded out over ~6 px of the channels (a stamp
    # is up to ~3 px wide, so a shorter fade leaves a moat of full-depth
    # stamps just beside the river): the mouth melts tangentially into the
    # floor (priority_flood below keeps the bed drainable).
    dnet = ndi.distance_transform_edt(~band)
    carve = carve * np.clip(dnet / 6.0, 0.0, 1.0)
    # a light smoothing of the CUT (not the surface): the raw max-combined
    # stamps leave a scalloped bed whose contour crossings appear as a speckle
    # of tiny closed rings.
    out = u - _blur(carve, 1.0, dom)
    # BASE LEVEL beside the river: within 4 px of a channel no cut may take
    # the ground below the nearest channel cell (ground that was already
    # lower keeps its own z).  Farther out the hillslope is left alone, so
    # the micro-relief is untouched.  Without the three rules above and this
    # one, the carve leaves ~58% of channel cells ABOVE the ground around
    # them (real Kawauchi: 0.4%); with them, ~15% (the solve alone: ~14%).
    _dn, (iyc, ixc) = ndi.distance_transform_edt(~np.asarray(net, bool),
                                                 return_indices=True)
    out = np.where(_dn <= 4, np.maximum(out, np.minimum(u, u[iyc, ixc])), out)
    # DEPOSITION.  The smoothing shallows each cut at its MOUTH (the
    # deepest point, averaged against the higher surroundings), which dams the
    # bed just upstream and leaves closed contours in the valleys (the
    # near-channel depression fraction rises ~5x; real terrain ~0.03).
    # Filling every depression to its spill level is acceptable as a field
    # operation because it only DELETES structure: each cut ends up exactly as deep as it can drain, ponds become
    # small flat valley floors.
    from experiments.ridge_connectivity import priority_flood
    from skimage import morphology as skmo2
    out = priority_flood(out, dom)
    # KNOLL removal.  After the fill, the remaining closed rings in the
    # valleys are not pits but small KNOLLS left standing between cuts.  Greyscale
    # reconstruction-by-dilation from (u - h) removes every bump of prominence
    # < h and nothing else -- still removal-only.
    # ... but NOT on the crests: the one micro-feature real terrain allows is
    # a small peak on a ridge, and a global h would shave exactly those.  Blend the reconstruction away above q75.
    h = 0.02
    # reconstruction needs a finite array.  The sea is set BELOW the land's
    # minimum, which is the one filling that cannot propagate into the land
    # (reconstruction by dilation only ever pushes high values outwards), so
    # this is an exclusion, not a value the result depends on.
    lo = float(np.nanmin(out))
    filled = out if dom is None else np.where(dom, out, lo - 1.0)
    q75 = float(np.nanquantile(out, 0.75))
    rec = skmo2.reconstruction(filled - h, filled, method="dilation")
    wc = 1.0 / (1.0 + np.exp((filled - q75) / 0.02))
    out = filled - wc * (filled - rec)
    # ... and a STRONGER h in the low terrain only: a closed ring on a ridge
    # can be a real hillock, a closed ring on a valley floor cannot (rivers
    # would have planed it).  Blend smoothly so a bump straddling the mask
    # edge is not half-shaved.
    q40 = float(np.nanquantile(np.where(dom, out, np.nan) if dom is not None
                               else out, 0.40))
    recon_lo = skmo2.reconstruction(out - 0.06, out, method="dilation")
    w = 1.0 / (1.0 + np.exp((out - q40) / 0.02))
    out = out - w * (out - recon_lo)
    # FINAL PIXEL-SCALE SMOOTHING.  Without it, contour maps show detached
    # dash fragments: pixel-scale warts of amplitude 0.001-0.007 grazing a
    # contour level.  Real DEM panels have none because the 386->128 zoom
    # interpolates them away -- the real surface is smooth below the feature
    # scale.  So the generated surface gets the same treatment: one light
    # gaussian at sub-feature sigma.  (Adding a de-degeneracy noise field
    # instead manufactures exactly these warts.)
    out = _blur(out, final_sigma, dom)
    return priority_flood(out, dom)


def main():
    from experiments.mini import (gen_headward, solve, score, reference,
                                  HDR, row, OUT)
    ref_net, ref_dem = reference()
    print(HDR)
    print(row("REAL kawauchi", score(ref_net, solve(ref_net, tilt=ref_dem)), 0))
    figs = []
    for sd in (1, 2, 3):
        rng = np.random.default_rng(sd)
        net = gen_headward(rng)
        u0 = solve(net)
        u1 = finish_micro(u0, net, rng)
        print(row(f"bare s{sd}", score(net, u0), 0))
        print(row(f"finished s{sd}", score(net, u1), 0))
        figs.append((net, u0, u1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(2, 3, figsize=(12.5, 8.6))
    for j, (net, u0, u1) in enumerate(figs):
        for i, u in enumerate((u0, u1)):
            ax[i, j].contour(u, levels=np.linspace(0.02, 0.98, 18),
                             colors="#4a3a20", linewidths=0.7, origin="upper")
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[i, j].set_aspect("equal")
        ax[0, j].set_title(f"seed {j + 1}  bare", fontsize=10)
        ax[1, j].set_title("finished (spacing-filled cuts)", fontsize=10)
    f.suptitle("mini_finish: fill the hillslopes by spacing", fontsize=12)
    f.tight_layout(rect=[0, 0, 1, 0.95])
    f.savefig(os.path.join(OUT, "mini_finish.png"), dpi=110)
    print("saved valley_samples/mini_finish.png")


if __name__ == "__main__":
    main()
