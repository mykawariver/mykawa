"""baseline -- the canonical generator, with the FROZEN ZERO.

The whole chain (edge-rooted headward growth -> Flint profile with
accumulation-widened floors -> spacing-filled carve pass with mouth fade,
crest notch, knoll removal and the final pixel-scale smoothing) is fixed, and
every stylistic degree of freedom is exposed as a SIGNED DEVIATION AROUND ZERO:

    generate(seed)                      # the baseline, parameter zero
    generate(seed, relief=-1, floor=+2) # flatter map, wider valley floors

Each knob is dimensionless; one unit is one "noticeable but plausible" step,
implemented as a multiplier on the underlying quantity (listed below).  Zero
must always reproduce the frozen look -- that is what the golden regression
below protects:

    python3 -m experiments.baseline --freeze   # write the golden sample (once)
    python3 -m experiments.baseline --check    # verify zero still equals it
    python3 -m experiments.baseline --fig      # 3x3 contact sheet of the zero

KNOBS (zero = frozen state):
    concavity How the river bed rises along the river.  concavity = bow / 2,
              the along-profile distance weight in mini.profile_slope_area;
              theta stays at 0.30.  bow reaches concavity through Hack's law
              (A ~ s**1.8 along a trunk).  Routing it through theta instead
              would also change the elevation contrast between neighbouring
              channels of different accumulation (a fine-scale effect); bow
              leaks 15-40x less into the fine scale.  bow = 0 makes the weight
              identically 1.
              MEASURED (concavity index CI, seeds 1-3):
                  concavity -1.5 -> CI -0.146   convex
                  concavity -0.75-> CI -0.026   dead straight
                  concavity  0   -> CI  0.119   the zero
                  concavity +0.75-> CI  0.212
                  concavity +1.5 -> CI  0.283
                  concavity +3   -> CI  0.369
              MEASURED (real): kawauchi, a 1.5 km UPSTREAM window, CI 0.096;
              whole rivers reaching the sea 0.144 / 0.238 / 0.384 / 0.420
              (hisanohama / jusanhama / ogatsu / yasukigawa).  The right
              concavity depends on how much of a river the map contains --
              which is why it is a knob.  Zero is calibrated to a 1.5 km
              upstream window.  `theta=` remains available as a raw override.
    relief    overall z span drawn against the fixed 18 contour levels.
              x 1.25**relief on RELIEF=0.66.
    texture   NOT a knob; leave at 0.  It scales carve depth (x 1.30**t) and
              width (x 1.15**t) but has no measurable effect: the carve pass
              adds nothing net to the fine band (the depression fill and knoll
              reconstruction take out what the carve puts in) and the final
              gaussian (`smooth`) erases the rest.  `smooth` is the one working
              fine-scale control.
    floor     valley-bottom width: floor_px = 1.0 + 3.0 * 1.75**floor, i.e.
              the knob multiplies the sqrt(A) TERM of the half-width law
              hw = 1 + (floor_px - 1) * sqrt(A / ACC_REF), leaving the 1 px
              minimum -- the channel itself -- alone.  The term it scales is
              the prefactor of the measured law W ~ A**0.49 (real transects:
              W(A=1e4) = 24..110 m, 4.6x).  Scaling the WHOLE width would also
              widen 1 px headwater channels, a fine-scale change.
              MEASURED (n=128, seeds 1-3): floor -2 -> W_floor -0.85 sigma,
              +2 -> +1.26 sigma.  Stronger on a bigger map (n=256: +1.9 sigma)
              because a 1.5 km upstream window has little wide floor to widen.
              Away from zero it also redraws the network (growth reserves a
              corridor of the same width).
    density   drainage density: headward stop scale ell / 1.20**density on 5.0
              (positive = finer network, more tips).
    smooth    the final pixel-scale gaussian: sigma x 1.25**smooth on 0.9.
    roots     integer OFFSET on the number of edge outlets (base draw 3-5).

SCALE CONTRACT:
    THE PIXEL IS THE PHYSICAL ANCHOR: 1 px = 11.7 m, fixed.  It comes from the
    calibration against the real references (1.5 km patches judged at N=128),
    so n=128 is a 1.5 km map and a bigger canvas means MORE LAND at the same
    resolution: side = 1.5 km * n/128 (n=256 -> 3 km; hillslope p50 stays
    4.1 px = 48 m).  Every internal length (ell, floor_px, spacing, carve
    width, the final sigma) is pixel-denominated ON PURPOSE -- physical
    feature sizes must not grow with the map.  The style knobs are
    dimensionless multipliers on those quantities, so they are independent of
    canvas size.  Three quantities DO scale with the map (handled below): the
    number of edge outlets (per perimeter), the carve budget (per area) and
    the VALLEY FLOOR WIDTH, which follows W ~ A^0.49 from the absolute
    drainage area (ACC_REF): the trunk is ~216 m wide at 1.5 km, 281 m at
    3 km, 377 m at 5 km.
    Tributary density is constant with map size (ell is in pixels), as within
    one physiographic region in real terrain.  Hillslope relief is also
    unchanged with map size (0.133 at n=128, 0.143 at n=256) because outlets
    scale with the perimeter and catchments stay about the same size.
    Raising the RESOLUTION instead (same km, smaller m/px) is NOT supported by
    this contract and would need a px_m conversion first.

Changing what zero MEANS (any default below, or the chain itself) is a
breaking change: re-run --freeze afterwards.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N = 128
RELIEF = 0.66
# trunk accumulation of the calibration map (median over seeds 1-5 at n=128).
# Fixing it makes valley width an ABSOLUTE function of drainage area.
ACC_REF = 1664.0
# The two constants the corridor and the floor share: the trunk
# half-width at ACC_REF, and the hillslope length growth aims for.  They are
# ONE calibration -- the generator keeps other channels out of a valley of
# half-width FLOOR_PX * sqrt(A/ACC_REF) and `solve` then paints exactly that
# valley -- so they are set here, once, rather than in either call site.
FLOOR_PX = 4.0
ELL0 = 5.00
# The zero-order basin: how far the hollow continues above each channel head,
# in units of the hillslope length `ell`.  1.6 x ell puts the head
# of the hollow at the divide, which is where a real one ends; beyond that the
# extensions start running into their neighbours and the tip count falls again
# (measured: the tip count peaks at 1.6).  Expressed in ell, not
# px, so it follows the drainage density instead of fighting it.
HOLLOW = 1.6
GOLDEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "valley_samples", "cache", "baseline_zero_s1.npz")


def hydraulic_reach(shape, root_list):
    """The river's own run: the farthest a stream can get from an outlet
    inside the domain, in px.

    The stream budget is the one length that must
    NOT be read off a side.  A budget tied to the side would stop the trunk half
    way up a long map (river along the long axis) or let it wander far past the
    coast in a short fat one.  What it is really per is the distance the water
    has to travel to leave the map -- which is exactly the distance transform
    from the outlets, maximised over the domain.  For the square with one edge
    root this is ~1.1 x the side; for a 1:3 rectangle with
    the outlet on a short edge it is the long side, as it should be."""
    from scipy import ndimage as ndi
    H, W = shape
    m = np.ones((H, W), bool)
    for (r, c, _a) in root_list:
        m[int(r), int(c)] = False
    return float(ndi.distance_transform_edt(m).max())


def generate(seed, n=N, relief=0.0, texture=0.0, floor=0.0, density=0.0,
             smooth=0.0, roots=0, concavity=0.0, n_roots=None, theta=None,
             shape=None, outlets=None, bow=0.0, stage="finish",
             trend=None, trend_max=1.15, split_p=0.0, reseed=True):
    """The canonical generator.  All knobs zero = the frozen baseline.
    Returns (u, net, meta): u in [0, relief_span], net the 1-px channels.

    stage: "solve" returns the surface as the elevation solve leaves
    it, before `finish_micro` carves micro-relief and fills; "finish" is the
    whole chain and is the default.

    WHY IT MATTERS.  Measured at n=128, the share of routed channel cells that
    sit nowhere near a channel the generator drew goes 11-20% after `solve` to
    25-36% after `finish_micro`, and 69-87% of those extra ones lie inside a
    painted valley floor -- D8 on dead flat ground is decided by arithmetic
    noise, and any instrument that reads the ground counts them.  Reading at "solve" is
    the closest available estimate of the network the generator MEANT.

    The comparison against a real DEM is then not quite symmetric -- nature
    ran its own stage 3 on the real ground.  It is still the better of the two
    comparisons, because what stage 3 adds here has no counterpart in the real
    terrain: it is inside the valley floors, not on the hillslopes."""
    from experiments.mini import gen_headward, solve
    from experiments.mini_finish import finish_micro
    # concavity drives BOW, not theta (see the knob docs above and
    # mini.profile_slope_area).  theta stays frozen at 0.30 -- it is
    # what sets the elevation contrast between neighbouring channels, i.e.
    # texture -- and remains available as a raw override to dial in a
    # measured Flint exponent.
    if theta is None:
        theta = 0.30
    if bow == 0.0:
        bow = 2.0 * float(concavity)
    rng = np.random.default_rng(seed)
    # shape=(H, W) makes the terrain rectangular; n stays the square shorthand.
    H, W = (int(n), int(n)) if shape is None else (int(shape[0]), int(shape[1]))
    # outlets scale with the PERIMETER, the carve budget with the AREA (see the
    # scale contract); both reduce to the frozen values at n=128.  outlets=
    # [(r, c, heading), ...] overrides the geometric draw entirely -- which is
    # the right interface for a rectangle, because where the water leaves a map
    # is a boundary condition, not something a formula can know.
    k = int(round(int(rng.integers(3, 6)) * (H + W) / (2 * N))) + int(roots)
    # n_roots forces an exact outlet count (downstream work needs ONE trunk
    # spanning the map: at n=427 the perimeter rule gives ~13 roots, i.e. a
    # dozen small catchments and no 5 km river).  theta is likewise a pass-
    # through for the transect-measured Flint exponent.  Both default to the
    # frozen behaviour.
    k = int(n_roots) if n_roots is not None else max(1, k)
    if outlets is not None:
        root_list = [(int(r), int(c), float(a)) for (r, c, a) in outlets]
    else:
        root_list = []
        for _ in range(k):
            t = float(rng.uniform(0.12, 0.88))
            e = int(rng.integers(4))
            if e == 0:   rc, a = (H - 2, int(t * W)), -np.pi / 2      # bottom
            elif e == 1: rc, a = (1, int(t * W)), np.pi / 2           # top
            elif e == 2: rc, a = (int(t * H), 1), 0.0                 # left
            else:        rc, a = (int(t * H), W - 2), np.pi           # right
            root_list.append((rc[0], rc[1], a + float(rng.normal(0, 0.25))))
    reach = None if shape is None else hydraulic_reach((H, W), root_list)
    # THE FLOOR KNOB HAS TO REACH BOTH STAGES.  gen_headward reserves a
    # corridor of this width and solve paints a floor of this width; if they
    # differed, the paint could spill past the room made for it.  They are
    # one quantity and must be one expression.  Away from zero this makes
    # `floor` a knob that REDRAWS the network, like density and roots.
    fpx = 1.0 + (FLOOR_PX - 1.0) * 1.75 ** floor
    # STAGE 1 -> STAGE 2 CONTRACT: the network raster PLUS one int raster
    # saying which outlet each channel cell drains to.  Nothing else.  Without
    # the labels, `solve` would weld neighbouring basins with the zero-order
    # hollow and measure distance to the NEAREST outlet, so a small tree could
    # end up carrying the map's largest drainage area.
    net, net_labels = gen_headward(rng, n=n, ell=ELL0 / 1.20 ** density,
                                   roots=root_list, shape=(H, W), reach=reach,
                                   floor_px=fpx, trend=trend,
                                   trend_max=trend_max, split_p=split_p,
                                   reseed=reseed, labels=True)
    # bow: the route to concavity that leaves the fine scale alone
    # (see mini.profile_slope_area).  Default 0 = frozen.
    u = solve(net, outlets=[(r, c) for (r, c, _) in root_list],
              floor_px=fpx, theta=theta, acc_ref=ACC_REF, bow=bow,
              hollow_px=HOLLOW * ELL0 / 1.20 ** density, labels=net_labels)
    # smooth acts on finish_micro's OWN final gaussian, so both directions
    # are real (an extra blur applied afterwards could not go below zero).
    if stage == "solve":
        span = RELIEF * 1.25 ** relief
        u = (u - u.min()) / (u.max() - u.min() + 1e-12) * span
        return u, net, dict(seed=seed, roots=len(root_list), span=span,
                            shape=(H, W), reach=reach, stage="solve",
                            labels=net_labels)
    u = finish_micro(u, net, rng,
                     depth0=0.13 * 1.30 ** texture,
                     width=2.4 * 1.15 ** texture,
                     smooth=1.5, max_cuts=int(500 * (H * W) / (N * N)),
                     final_sigma=0.9 * 1.25 ** smooth)
    span = RELIEF * 1.25 ** relief
    u = (u - u.min()) / (u.max() - u.min() + 1e-12) * span
    meta = dict(seed=seed, roots=len(root_list), span=span,
                shape=(H, W), reach=reach, labels=net_labels)
    return u, net, meta


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fig", action="store_true")
    a = ap.parse_args(argv)
    if a.freeze:
        u, _, _ = generate(1)
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        np.savez_compressed(GOLDEN, u=u)
        print("frozen ->", GOLDEN)
    if a.check:
        u, _, _ = generate(1)
        ref = np.load(GOLDEN)["u"]
        d = float(np.abs(u - ref).max())
        print(f"max |u - golden| = {d:.2e}  ->",
              "OK (zero unchanged)" if d < 1e-9 else
              "CHANGED -- parameter zero has moved!")
        return 0 if d < 1e-9 else 1
    if a.fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(3, 3, figsize=(12, 12.4))
        for i, sd in enumerate(range(1, 10)):
            u, _, _ = generate(sd)
            aa = ax[i // 3, i % 3]
            aa.contour(u, levels=np.linspace(0.02, 0.98, 18),
                       colors="#4a3a20", linewidths=0.7, origin="upper")
            aa.set_xticks([]); aa.set_yticks([]); aa.set_aspect("equal")
            aa.set_title(f"seed {sd}", fontsize=10)
        f.suptitle("baseline zero (frozen)", fontsize=13)
        f.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(os.path.dirname(GOLDEN), "..", "baseline_zero.png")
        f.savefig(out, dpi=110)
        print("saved", os.path.normpath(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
