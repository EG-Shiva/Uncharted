import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _build_tables(m, speeds):
    s1, s2, s3 = speeds
    ks = np.arange(m)
    t1 = 2 * np.pi * s1 * ks / m
    t2 = 2 * np.pi * s2 * ks / m
    t3 = 2 * np.pi * s3 * ks / m
    return (np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2),
            np.cos(t3), np.sin(t3))


def _expand(reps, m, tabs, pole):
    c1, s1, c2, s2, c3, s3 = tabs
    n_orb = reps.shape[0]
    r0 = reps[:, 0:1]
    r1 = reps[:, 1:2]
    r2 = reps[:, 2:3]
    r3 = reps[:, 3:4]
    r4 = reps[:, 4:5]
    r5 = reps[:, 5:6]
    r6 = reps[:, 6:7]
    a0 = r0 * c1 - r1 * s1
    b0 = r0 * s1 + r1 * c1
    a2 = r2 * c2 - r3 * s2
    b2 = r2 * s2 + r3 * c2
    a4 = r4 * c3 - r5 * s3
    b4 = r4 * s3 + r5 * c3
    # (n_orb, m, 7)
    XX = np.empty((n_orb, m, 7), dtype=np.float64)
    XX[:, :, 0] = a0
    XX[:, :, 1] = b0
    XX[:, :, 2] = a2
    XX[:, :, 3] = b2
    XX[:, :, 4] = a4
    XX[:, :, 5] = b4
    XX[:, :, 6] = r6  # broadcast (n_orb,1) -> (n_orb,m)
    X = XX.reshape(n_orb * m, 7)
    if pole:
        p = np.zeros((1, 7), dtype=np.float64)
        p[0, 6] = 1.0
        X = np.vstack([X, p])
    # renorm for safety
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _eval_reps(reps, m, tabs, pole):
    X = _expand(reps, m, tabs, pole)
    return _max_cos(X)


def _hillclimb(m, speeds, n_orb, pole, secs, rng, t_dead):
    tabs = _build_tables(m, speeds)
    # init
    best_reps = rng.standard_normal(size=(n_orb, 7))
    best_reps /= np.linalg.norm(best_reps, axis=1, keepdims=True)
    best_val = _eval_reps(best_reps, m, tabs, pole)
    # try a few random inits quickly
    for _ in range(8):
        if time.time() >= t_dead:
            break
        R = rng.standard_normal(size=(n_orb, 7))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        v = _eval_reps(R, m, tabs, pole)
        if v < best_val:
            best_val = v
            best_reps = R.copy()
    t_end = min(time.time() + secs, t_dead)
    sigma = 0.12
    succ = 0
    total = 0
    it = 0
    cur_reps = best_reps.copy()
    cur_val = best_val
    while time.time() < t_end:
        it += 1
        # batch of 1 proposal (fast python loop)
        prop = cur_reps + sigma * rng.standard_normal(size=cur_reps.shape)
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        v = _eval_reps(prop, m, tabs, pole)
        total += 1
        if v < cur_val:
            cur_reps = prop
            cur_val = v
            succ += 1
            if v < best_val:
                best_val = v
                best_reps = prop.copy()
        # adapt sigma every 200
        if total >= 200:
            rate = succ / max(1, total)
            if rate > 0.25:
                sigma = min(0.4, sigma * 1.25)
            elif rate < 0.08:
                sigma = max(0.005, sigma * 0.7)
            else:
                sigma *= 0.98
            succ = 0
            total = 0
        # occasional restart from best with smaller kick
        if it % 1500 == 0 and time.time() < t_end:
            # shake best
            prop2 = best_reps + (sigma * 0.7) * rng.standard_normal(size=best_reps.shape)
            prop2 /= np.linalg.norm(prop2, axis=1, keepdims=True)
            v2 = _eval_reps(prop2, m, tabs, pole)
            if v2 < cur_val + 0.005:
                cur_reps = prop2
                cur_val = v2
        if it % 50 == 0 and time.time() >= t_end:
            break
    return best_reps, best_val, tabs


def _reps_adam(reps, m, tabs, pole, secs, rng, t_dead):
    # smooth Adam/SGD on representatives with annealed tau
    n_orb = reps.shape[0]
    c1, s1, c2, s2, c3, s3 = tabs
    cur = reps.copy()
    cur /= np.linalg.norm(cur, axis=1, keepdims=True)
    best = cur.copy()
    best_val = _eval_reps(cur, m, tabs, pole)
    vel = np.zeros_like(cur)
    mu = 0.85
    t_end = min(time.time() + secs, t_dead)
    it = 0
    while time.time() < t_end:
        it += 1
        # schedule tau: 0.04 -> 0.004
        frac = 1.0 if t_end <= time.time() else 1.0
        # simple exponential decay with iteration
        tau = max(0.004, 0.05 * (0.995 ** it))
        lr = 0.25 * (0.9995 ** it) + 0.02
        X = _expand(cur, m, tabs, pole)
        N = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(np.max(G))
        E = np.exp((G - mx) / tau)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        # grad full = (E @ X)/S
        GF = (E @ X) / S
        # map to reps grad: reshape non-pole part
        nfull = n_orb * m
        Gorb = GF[:nfull].reshape(n_orb, m, 7)
        ga0 = Gorb[:, :, 0]
        gb0 = Gorb[:, :, 1]
        ga2 = Gorb[:, :, 2]
        gb2 = Gorb[:, :, 3]
        ga4 = Gorb[:, :, 4]
        gb4 = Gorb[:, :, 5]
        g6 = Gorb[:, :, 6]
        rg = np.empty_like(cur)
        rg[:, 0] = np.sum(ga0 * c1 + gb0 * s1, axis=1)
        rg[:, 1] = np.sum(-ga0 * s1 + gb0 * c1, axis=1)
        rg[:, 2] = np.sum(ga2 * c2 + gb2 * s2, axis=1)
        rg[:, 3] = np.sum(-ga2 * s2 + gb2 * c2, axis=1)
        rg[:, 4] = np.sum(ga4 * c3 + gb4 * s3, axis=1)
        rg[:, 5] = np.sum(-ga4 * s3 + gb4 * c3, axis=1)
        rg[:, 6] = np.sum(g6, axis=1)
        # tangent projection
        dot = np.sum(rg * cur, axis=1, keepdims=True)
        rg = rg - dot * cur
        # normalize grad scale
        nrm = np.linalg.norm(rg) + 1e-12
        if nrm > 1e3:
            rg *= (1e3 / nrm)
        vel = mu * vel - lr * rg
        cur = cur + vel
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        if it % 10 == 0:
            v = _eval_reps(cur, m, tabs, pole)
            if v < best_val:
                best_val = v
                best = cur.copy()
        if it % 20 == 0 and time.time() >= t_end:
            break
    return best, best_val


def _full_anneal(X0, secs, t_dead, rng):
    X = X0.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    best = X.copy()
    best_val = _max_cos(X)
    n, d = X.shape
    t_end = min(time.time() + secs, t_dead)
    it = 0
    while time.time() < t_end:
        it += 1
        tau = max(0.0025, 0.03 * (0.996 ** it))
        lr = 0.6 * (0.999 ** it) + 0.05
        # add tiny noise early to break symmetry
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(np.max(G))
        # weights
        E = np.exp((G - mx) / tau)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        GF = (E @ X) / S
        # tangent
        dot = np.sum(GF * X, axis=1, keepdims=True)
        GT = GF - dot * X
        # clip
        gn = np.linalg.norm(GT)
        if gn > 5.0:
            GT *= (5.0 / gn)
        X = X - lr * GT
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        if it % 15 == 0:
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best = X.copy()
        if it % 30 == 0 and time.time() >= t_end:
            break
    return best, best_val


def _lp_polish(X0, secs, t_dead):
    X = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_X = X.copy()
    radius = 5e-4
    t_end = min(time.time() + secs, t_dead)
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        remain = t_end - time.time()
        if remain < 0.5:
            break
        G = X @ X.T
        c = G[iuI, iuJ]
        m = float(np.max(c))
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 2200:
            idx = np.where(sel)[0]
            cv = c[idx]
            part = np.argpartition(cv, -2200)[-2200:]
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[idx[part]] = True
            sel = sel2
            cnt = 2200
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        crow = cols.reshape(p, 2 * d + 1)
        vrow = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        crow[:, :d] = I[:, None] * d + ar
        crow[:, d:2 * d] = J[:, None] * d + ar
        crow[:, 2 * d] = n * d
        vrow[:, :d] = X[J]
        vrow[:, d:2 * d] = X[I]
        vrow[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            tl = max(0.5, min(4.0, remain - 0.2))
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl, "presolve": True})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y
            best = v
            best_X = y.copy()
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return best_X, best


def solve():
    t0 = time.time()
    t_dead = t0 + 165.0
    rng = np.random.default_rng(0)
    # load incumbent
    x_best = None
    v_best = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if not np.all(np.isfinite(a)):
                    continue
                v = _max_cos(a)
                if v < v_best:
                    v_best = v
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng.standard_normal(size=(323, 7))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        v_best = _max_cos(x_best)
    # symmetric configs: 323 = 14*23+1 and 17*19
    cfgs = [
        (23, (1, 5, 11), 14, True),
        (23, (1, 2, 4), 14, True),
        (23, (1, 3, 7), 14, True),
        (19, (1, 4, 9), 17, False),
        (19, (1, 2, 7), 17, False),
    ]
    sym_best_val = np.inf
    sym_best_X = None
    sym_best_info = None
    # budget: ~7s hillclimb each if time allows
    for (m, speeds, n_orb, pole) in cfgs:
        if time.time() > t_dead - 40:
            break
        try:
            R, v, tabs = _hillclimb(m, speeds, n_orb, pole, 7.0, rng, t_dead - 35)
        except Exception:
            continue
        if v < sym_best_val:
            sym_best_val = v
            sym_best_info = (m, speeds, n_orb, pole, tabs, R.copy())
    # refine best symmetric with Adam on reps
    if sym_best_info is not None and time.time() < t_dead - 35:
        m, speeds, n_orb, pole, tabs, R = sym_best_info
        try:
            R2, v2 = _reps_adam(R, m, tabs, pole, 8.0, rng, t_dead - 30)
            if v2 < sym_best_val:
                sym_best_val = v2
                R = R2
                sym_best_info = (m, speeds, n_orb, pole, tabs, R)
        except Exception:
            pass
        try:
            sym_best_X = _expand(sym_best_info[5], sym_best_info[0], sym_best_info[4], sym_best_info[3])
        except Exception:
            sym_best_X = None
    # break symmetry with full anneal + LP polish
    cand_X = x_best.copy()
    cand_v = v_best
    if sym_best_X is not None:
        # full anneal to break symmetry
        try:
            secs = 18.0 if (t_dead - time.time()) > 45 else max(2.0, (t_dead - time.time()) * 0.4)
            XA, vA = _full_anneal(sym_best_X, secs, t_dead - 12, rng)
            if vA < cand_v:
                cand_v = vA
                cand_X = XA.copy()
            else:
                # even if not better, keep annealed version for LP if close
                # only keep if within 0.02 to save LP time
                if vA < v_best + 0.02:
                    # polish it anyway, may cross over
                    pass
                XA_keep = XA
                # LP polish the annealed symmetric result regardless (may improve a lot)
                try:
                    secs2 = max(2.0, min(14.0, t_dead - time.time() - 6))
                    if secs2 > 1.0:
                        XL, vL = _lp_polish(XA_keep, secs2 / 2, t_dead - 5)
                        if vL < cand_v:
                            cand_v = vL
                            cand_X = XL.copy()
                except Exception:
                    pass
                # also LP polish XA (best annealed) already done; skip duplicate below
                XA_done = True
                raise StopIteration
            # LP polish annealed symmetric
            try:
                secs2 = max(2.0, min(14.0, t_dead - time.time() - 6))
                if secs2 > 1.0 and time.time() < t_dead - 6:
                    XL, vL = _lp_polish(XA, secs2, t_dead - 4)
                    if vL < cand_v:
                        cand_v = vL
                        cand_X = XL.copy()
            except Exception:
                pass
        except StopIteration:
            pass
        except Exception:
            pass
    # final squeeze of incumbent if time remains and we haven't beaten it
    if time.time() < t_dead - 8:
        try:
            # small kick + anneal starting from current global best (incumbent side)
            # only if symmetric path didn't win
            if cand_v >= v_best - 1e-12:
                remain = t_dead - time.time()
                # quick anneal from incumbent with tiny noise to change basin slightly
                Xn = x_best + 0.002 * rng.standard_normal(size=x_best.shape)
                Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
                secs = min(10.0, max(2.0, remain - 6))
                XA2, vA2 = _full_anneal(Xn, secs, t_dead - 5, rng)
                if vA2 < cand_v:
                    cand_v = vA2
                    cand_X = XA2.copy()
                remain2 = t_dead - time.time()
                if remain2 > 4:
                    XL2, vL2 = _lp_polish(cand_X if cand_v < v_best else x_best, min(10.0, remain2 - 1), t_dead - 1)
                    if vL2 < cand_v:
                        cand_v = vL2
                        cand_X = XL2.copy()
            else:
                # we already beat incumbent via symmetric path; polish winner a bit more
                remain2 = t_dead - time.time()
                if remain2 > 4:
                    XL3, vL3 = _lp_polish(cand_X, min(10.0, remain2 - 1), t_dead - 1)
                    if vL3 < cand_v:
                        cand_v = vL3
                        cand_X = XL3.copy()
        except Exception:
            pass
    # choose final
    try:
        v_cand = _max_cos(cand_X)
    except Exception:
        cand_X = x_best
        v_cand = v_best
    if not np.all(np.isfinite(cand_X)):
        cand_X = x_best
    cand_X = cand_X / np.linalg.norm(cand_X, axis=1, keepdims=True)
    # safety: return best known
    try:
        if _max_cos(cand_X) > v_best + 1e-12:
            # if our exploration failed, return incumbent (guarantees no regression)
            return x_best
    except Exception:
        return x_best
    return cand_X
