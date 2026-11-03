import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

REC = 0.244097856927
SCALE = 1e-4

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t_deadline=None, radius_init=1e-3):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _get_duals(x, radius=5e-4):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    c = (x @ x.T)[iu]
    m = float(c.max())
    sel = c > m - 1e-9
    I, J = iu[0][sel], iu[1][sel]
    csel = c[sel]
    p = len(I)
    if p == 0:
        return I, J, csel, np.zeros(0)
    if p > 1500:
        # keep tightest 1500 for LP stability
        order = np.argsort(-csel)[:1500]
        I, J, csel = I[order], J[order], csel[order]
        p = 1500
    nv = n * d + 1
    ar = np.arange(d)
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
    vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
    obj = np.zeros(nv); obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
    except Exception:
        return I, J, csel, np.zeros(p)
    duals = None
    try:
        if hasattr(res, "ineqlin") and res.ineqlin is not None:
            mg = getattr(res.ineqlin, "marginals", None)
            if mg is not None:
                duals = np.asarray(mg, dtype=float).ravel()
            elif isinstance(res.ineqlin, dict) and "marginals" in res.ineqlin:
                duals = np.asarray(res.ineqlin["marginals"], dtype=float).ravel()
        if duals is None and isinstance(res, dict):
            pass
        # alternative attribute names
        if duals is None and hasattr(res, "marginals"):
            duals = np.asarray(res.marginals, dtype=float).ravel()
            if duals.size != p:
                duals = None
        if duals is None and hasattr(res, "ineqlin_marginals"):
            duals = np.asarray(res.ineqlin_marginals, dtype=float).ravel()
    except Exception:
        duals = None
    if duals is None or duals.size != p or not np.all(np.isfinite(duals)):
        # fallback: use slack rank as proxy (larger slack ~ lower stress)
        # slack = m - csel, larger slack => lower stress
        duals = -(m - csel)  # most negative = largest slack; abs small? use this ordering
        # we want |duals| small = low stress; here make tightest have larger |duals|
        # invert: tightest (slack~0) should look high stress
        duals = -(1.0 / (1.0 + (m - csel) / 1e-12))
    duals = np.asarray(duals, dtype=float).ravel()
    duals[~np.isfinite(duals)] = 0.0
    return I, J, csel, duals

def _adam_block(Xfrozen, S, t_target, steps, lr, rng_seed=0):
    # Xfrozen: (n,d) normalized, frozen reference; S: list of indices to move
    n, d = Xfrozen.shape
    k = len(S)
    Y = Xfrozen[S].copy()
    m1 = np.zeros_like(Y)
    m2 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # map global -> local for S-S handling not needed; use full matrix each step
    Xfull = Xfrozen.copy()
    for s in range(1, steps + 1):
        Xfull[S] = Y
        # dots between moving points and all points
        D = Y @ Xfull.T  # k x n
        # mask self
        for a, g in enumerate(S):
            D[a, g] = -10.0
        Z = (D - t_target) / SCALE
        # sigmoid stable
        Sig = np.empty_like(Z)
        pos = Z >= 0
        # sigmoid(z)=1/(1+exp(-z))
        Sig[pos] = 1.0 / (1.0 + np.exp(-Z[pos]))
        q = ~pos
        e = np.exp(Z[q])
        Sig[q] = e / (1.0 + e)
        # gradient: (Sig @ Xfull)/SCALE
        G = (Sig @ Xfull) / SCALE
        # tangent projection
        dot = np.sum(G * Y, axis=1, keepdims=True)
        Gt = G - dot * Y
        # normalize huge grads for stability
        gn = np.linalg.norm(Gt)
        if not np.isfinite(gn) or gn == 0:
            break
        # Adam update
        m1 = b1 * m1 + (1 - b1) * Gt
        m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
        m1h = m1 / (1 - b1 ** s)
        m2h = m2 / (1 - b2 ** s)
        step = lr * m1h / (np.sqrt(m2h) + eps)
        # clip step norm per row to avoid explosion
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        cap = 0.05
        f = np.minimum(1.0, cap / (sn + 1e-12))
        # decay lr slightly
        Y = Y - step * f / max(f.min(), 1e-12) * 1.0
        # simpler: Y = Y - step, then clip handled above; redo cleanly:
        # (above already scaled) renormalize
        Y = _norm(Y)
        Xfull[S] = Y
    return _norm(Y)

def solve():
    t0 = time.time()
    DEADLINE = t0 + 150.0
    rng = np.random.default_rng(12345)
    # load seeds
    bestX = None
    bestV = 10.0
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (95, 13):
                a = _norm(a)
                v = _max_cos(a)
                if v < bestV:
                    bestV = v
                    bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        z = rng.standard_normal((95, 13))
        bestX = _norm(z)
        bestV = _max_cos(bestX)
    recordX = bestX.copy()
    recordV = float(bestV)

    # one LP polish (directive)
    sec = min(12.0, max(6.0, DEADLINE - time.time() - 100.0))
    if sec > 0.5 and time.time() < DEADLINE - 20:
        bestX, bestV = _lp_polish(bestX, sec, t_deadline=DEADLINE - 20)
        if bestV < recordV:
            recordX, recordV = bestX.copy(), float(bestV)

    # extract dual marginals for slack<1e-9
    try:
        I, J, csel, duals = _get_duals(bestX)
    except Exception:
        I, J, csel, duals = np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0), np.zeros(0)

    n, d = bestX.shape
    if len(duals) < 10:
        # fallback: use tightest pairs as contacts with dummy duals
        iu = np.triu_indices(n, 1)
        c = (bestX @ bestX.T)[iu]
        m = float(c.max())
        ord_ = np.argsort(-c)[:60]
        I, J = iu[0][ord_], iu[1][ord_]
        duals = -np.arange(len(I), dtype=float)  # dummy
    absd = np.abs(duals)
    order = np.argsort(absd)  # ascending: low stress first
    Is, Js = I[order], J[order]

    targets = [0.2438, 0.2433, 0.2425]
    trial = 0
    # repeatedly choose 10-30 low-stress contacts involving 8-20 points
    while time.time() < DEADLINE - 12:
        trial += 1
        if trial > 40:
            break
        nc = 10 + (trial * 7) % 21  # 10..30 deterministic cycle
        # sliding window over low-stress list for diversity
        start_idx = (trial * 13) % max(1, len(Is) - nc)
        Ic = Is[start_idx:start_idx + nc]
        Jc = Js[start_idx:start_idx + nc]
        pts = sorted(set(Ic.tolist()) | set(Jc.tolist()))
        # enforce 8-20 points
        if len(pts) < 8:
            # expand window
            extra = 8 - len(pts)
            Ic2 = Is[start_idx:start_idx + nc + extra * 3]
            Jc2 = Js[start_idx:start_idx + nc + extra * 3]
            pts = sorted(set(Ic2.tolist()) | set(Jc2.tolist()))
            Ic, Jc = Ic2, Jc2
        if len(pts) > 20:
            # subsample deterministically to 16 points
            sel_idx = rng.choice(len(pts), size=16, replace=False)
            # use rng but deterministic sequence; ok (seeded)
            keep = set(np.array(pts)[np.sort(sel_idx)].tolist())
            # keep contacts fully inside keep; if too few, just take keep as S
            pts = sorted(keep)
        S = np.array(pts, dtype=int)
        if len(S) < 4 or len(S) > 22:
            continue
        # feasibility: stepwise targets with Adam, frozen rest
        Xcur = bestX.copy()
        ok = True
        for ti, t in enumerate(targets):
            if time.time() > DEADLINE - 10:
                ok = False
                break
            steps = 700 if len(S) <= 14 else 450
            lr = 0.008 if ti == 0 else 0.005
            try:
                Y = _adam_block(Xcur, S, t, steps, lr)
            except Exception:
                ok = False
                break
            Xcur[S] = Y
            # quick check: if block max vs frozen exploded badly, shrink step (reject)
            vblk = _max_cos(Xcur)
            if vblk > bestV + 0.03 and ti == 0:
                # too disruptive, keep anyway for LP to fix but limit
                pass
        if not ok:
            continue
        # unfreeze all, LP-polish
        remain = DEADLINE - time.time()
        if remain < 4:
            break
        psec = min(5.0, remain - 2.0)
        # small random kick to help LP leave vertex (deterministic tiny noise on S)
        # deliberately deactivate: perturb S slightly along tangent before polish
        # (helps escape same basin)
        amp = 0.002
        noise = rng.standard_normal((len(S), d)) * amp
        # project noise to tangent
        dotn = np.sum(noise * Xcur[S], axis=1, keepdims=True)
        noise -= dotn * Xcur[S]
        Xcur[S] = _norm(Xcur[S] + noise)
        Xpol, Vpol = _lp_polish(Xcur, psec, t_deadline=DEADLINE - 2)
        # keep only if beats REC (directive); also track global best
        if Vpol < REC and Vpol < recordV - 1e-12:
            recordX, recordV = Xpol.copy(), float(Vpol)
            bestX = Xpol.copy()
            bestV = float(Vpol)
            # refresh duals after improvement
            try:
                I, J, csel, duals = _get_duals(bestX)
                absd = np.abs(duals)
                order = np.argsort(absd)
                Is, Js = I[order], J[order]
            except Exception:
                pass
        elif Vpol < bestV - 1e-9:
            # improved local basin but not record: move base to explore from there occasionally
            bestX, bestV = Xpol.copy(), float(Vpol)
        # else discard (keep old bestX as base every other trial to diversify)
        if trial % 4 == 0 and recordV < bestV:
            bestX = recordX.copy()
            bestV = float(recordV)

    # final polish of record if time remains
    if time.time() < DEADLINE - 2 and recordX is not None:
        try:
            Xf, Vf = _lp_polish(recordX, max(0.5, DEADLINE - time.time() - 0.5), t_deadline=DEADLINE - 0.3)
            if Vf < recordV:
                recordX, recordV = Xf, Vf
        except Exception:
            pass
    # always fall back to record/best (never worse than loaded best)
    out = recordX / np.linalg.norm(recordX, axis=1, keepdims=True)
    # tiny shrink guard against NaN/zero
    out = np.where(np.isfinite(out), out, 0.0)
    nr = np.linalg.norm(out, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    out = out / nr
    return out
