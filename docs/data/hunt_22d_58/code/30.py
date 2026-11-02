import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _adam_pump(X, t, beta0, beta1, steps, lr0, lr1, rng, deadline, noise=0.0):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    bestX = X.copy()
    bestv = _max_cos(X)
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        if time.time() >= deadline:
            break
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # restrict to pairs above t-0.01
        Z = G - t
        # mask for computation
        # softplus and sigmoid stable
        S = beta * Z
        # f_sp, sig
        F = np.zeros_like(G)
        Sig = np.zeros_like(G)
        # large positive
        m1pos = S > 20
        m1neg = S < -20
        mid = ~(m1pos | m1neg)
        F[m1pos] = Z[m1pos]
        Sig[m1pos] = 1.0
        # neg ~ 0
        # mid
        if np.any(mid):
            eS = np.exp(S[mid])
            F[mid] = np.log1p(eS) / beta
            Sig[mid] = eS / (1.0 + eS)
        # zero out far-below pairs and diagonal
        keep = Z > -0.01
        F[~keep] = 0.0
        Sig[~keep] = 0.0
        np.fill_diagonal(F, 0.0)
        np.fill_diagonal(Sig, 0.0)
        W = 2.0 * F * Sig
        # symmetrize (already symmetric)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m1 = b1 * m1 + (1 - b1) * Gt
        m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** (s + 1)
        c2 = 1.0 - b2 ** (s + 1)
        step = lr * (m1 / c1) / (np.sqrt(m2 / c2) + eps)
        # retangent
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        over = sn > 0.05
        step = np.where(over, step * (0.05 / sn), step)
        X = X - step
        if noise > 0:
            X = X + noise * (1.0 - f) * rng.standard_normal(X.shape) * 0.002
        X = _norm(X)
        # track true max occasionally
        if (s % 50 == 49) or (s == steps - 1):
            v = _max_cos(X)
            if v < bestv:
                bestv = v
                bestX = X.copy()
    return bestX, bestv

def _lp_polish(X, seconds, deadline, stress=None):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    radius = 8e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    # stress weights dict for secondary objective: if provided use to filter
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        it += 1
        if it > 25:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 600:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 600:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
        # drop / penalise contacts with unfavourable stress:
        # use stress weights (dual approx or adam weights); drop lowest quartile
        if stress is not None and len(sel) > 100:
            try:
                w = stress[sel]
                thr = np.quantile(w, 0.25)
                keepm = w >= thr
                # always keep the top 20 worst pairs regardless
                keepm[:20] = True
                if np.sum(keepm) >= 50:
                    sel = sel[keepm]
            except Exception:
                pass
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        # secondary objective: tiny weighted sum via perturbation of obj using stress
        # we keep primary pure for stability; secondary handled by filtering above
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        # try to extract dual multipliers as new stress weights
        new_stress = None
        try:
            if hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
                marg = np.asarray(res.ineqlin.marginals, dtype=float)
                # duals for A_ub (positive for active)
                full = np.zeros(c.shape[0])
                # map back: sel positions
                # marg corresponds to current p constraints
                full[sel] = np.maximum(0, marg[:p])
                new_stress = full
        except Exception:
            new_stress = None
        if new_stress is not None:
            stress = new_stress
        if v < best - 1e-15:
            X = y
            best = v
            bestX = y.copy()
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return bestX, best

def _hole_replace(X, victims, rng, n_cand=2000):
    n, d = X.shape
    X2 = X.copy()
    for v in victims:
        mask = np.ones(n, dtype=bool)
        mask[v] = False
        Y = X2[mask]
        C = rng.standard_normal((n_cand, d))
        C = _norm(C)
        dots = C @ Y.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        X2[v] = C[bi]
    return X2

def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    rng = np.random.default_rng(0)
    n, d = 58, 22
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = _norm(v)
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        best = _norm(z)
        best_val = _max_cos(best)
    incumbent = best.copy()
    inc_val = best_val
    # initial quick LP squeeze to get true local value (short)
    try:
        if DEADLINE - time.time() > 20:
            best, best_val = _lp_polish(best, 2.0, DEADLINE - 8)
            if best_val < inc_val:
                incumbent = best.copy(); inc_val = best_val
    except Exception:
        pass
    # feasibility pump rounds with decreasing targets
    # targets slightly below incumbent
    deltas = [0.00030, 0.00020, 0.00012, 0.00006]
    sigmas = [0.04, 0.09, 0.025, 0.14, 0.06, 0.02]
    stress = None
    rnd = 0
    # precompute iu for degree checks
    while time.time() < DEADLINE - 12:
        rnd += 1
        if time.time() >= DEADLINE - 12:
            break
        # pick target t slightly below current best
        delta = deltas[(rnd - 1) % len(deltas)]
        # after improvements, keep delta modest
        t = best_val - delta
        # don't go absurdly low: clamp to best_val-0.0008
        # alternate sigma
        sigma = sigmas[(rnd - 1) % len(sigmas)]
        # start: restart from incumbent (best) with perturbation
        X = incumbent + sigma * rng.standard_normal((n, d))
        X = _norm(X)
        # every 4th round: hole-reinsert worst points to force topology change
        if rnd % 4 == 0:
            try:
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                # worst pairs
                cnt = (G > t).sum(axis=1)
                # victims: highest degree points
                nv = 2 if rnd % 8 == 0 else 1
                victims = np.argsort(-cnt)[:nv]
                X = _hole_replace(X, victims, rng, n_cand=1500)
            except Exception:
                pass
        # anneal beta upward
        beta0, beta1 = 150.0, 2500.0
        steps = 700 if rnd % 4 != 0 else 1000
        # leave time for LP
        pump_deadline = min(DEADLINE - 10, time.time() + 6.0)
        try:
            Xp, vp = _adam_pump(X, t, beta0, beta1, steps, 0.006, 0.0012, rng, pump_deadline)
        except Exception:
            continue
        # count violations above t and above old max
        try:
            Gp = Xp @ Xp.T
            np.fill_diagonal(Gp, -2.0)
            nviol_t = int((Gp > t).sum() // 2)
            # contacts per point at t
            deg = (Gp > t).sum(axis=1)
            mindeg = int(deg.min())
            mediandeg = float(np.median(deg))
        except Exception:
            nviol_t = 10**9; mindeg = 99; mediandeg = 99
        # update global best on true max
        if vp < best_val - 1e-12:
            best = Xp.copy(); best_val = vp
            incumbent = Xp.copy(); inc_val = vp
            # t-feasible found (or improvement): LP-polish it and lower t
            try:
                lp_sec = min(3.0, max(0.5, (DEADLINE - 8 - time.time()) / 4))
                if lp_sec > 0.4 and time.time() < DEADLINE - 10:
                    # build stress from current violations for LP secondary
                    iu = np.triu_indices(n, 1)
                    c0 = Gp[iu]
                    # adam-like stress
                    S0 = np.clip(800.0 * (c0 - t), -20, 20)
                    sig0 = 1.0 / (1.0 + np.exp(-S0))
                    f0 = np.where(S0 > 20, c0 - t, np.log1p(np.exp(np.clip(S0, -20, 20))) / 800.0)
                    w0 = 2 * f0 * sig0
                    w0[c0 <= t - 0.01] = 0.0
                    stress = w0
                    Xl, vl = _lp_polish(Xp, lp_sec, DEADLINE - 8, stress=stress)
                    if vl < best_val:
                        best = Xl.copy(); best_val = vl
                        incumbent = Xl.copy(); inc_val = vl
            except Exception:
                pass
        else:
            # restart logic: if round created markedly fewer contacts (de-jammed),
            # keep it as next exploration seed instead of discarding
            if mindeg < 12 or (mediandeg - mindeg) > 18:
                # de-jammed topology found: explore from here a bit with LP alternation
                # only if not much worse
                if vp < best_val + 0.004:
                    # short LP with stress weights to try collective rearrangement
                    try:
                        if time.time() < DEADLINE - 12 and nviol_t < 120:
                            iu = np.triu_indices(n, 1)
                            c0 = Gp[iu]
                            S0 = np.clip(800.0 * (c0 - t), -20, 20)
                            sig0 = 1.0 / (1.0 + np.exp(-S0))
                            f0 = np.where(S0 > 20, c0 - t, np.log1p(np.exp(np.clip(S0, -20, 20))) / 800.0)
                            w0 = 2 * f0 * sig0
                            w0[c0 <= t - 0.01] = 0.0
                            Xl2, vl2 = _lp_polish(Xp, 1.0, DEADLINE - 8, stress=w0)
                            if vl2 < best_val:
                                best = Xl2.copy(); best_val = vl2
                                incumbent = Xl2.copy(); inc_val = vl2
                            elif vl2 < best_val + 0.002:
                                # keep de-jammed seed for next round start
                                incumbent_alt = Xl2.copy()
                                # use it once: pump from it next round by temporarily setting incumbent
                                # only occasionally to avoid drift
                                if rnd % 3 == 0:
                                    incumbent = incumbent_alt
                    except Exception:
                        pass
                # else discard (restart from incumbent automatically next loop)
            # else: restart from incumbent (do nothing)
        # occasional LP alternation even without improvement to update dual stress
        if rnd % 6 == 0 and time.time() < DEADLINE - 12:
            try:
                Xl3, vl3 = _lp_polish(incumbent, 0.8, DEADLINE - 8, stress=stress)
                if vl3 < best_val:
                    best = Xl3.copy(); best_val = vl3
                    incumbent = Xl3.copy(); inc_val = vl3
            except Exception:
                pass
        if rnd > 40:
            break
    # final LP polish of best under remaining budget
    try:
        remain = DEADLINE - 4 - time.time()
        if remain > 0.5:
            sec = min(remain, 8.0)
            best, best_val = _lp_polish(best, sec, DEADLINE - 2, stress=stress)
    except Exception:
        pass
    best = _norm(best)
    return best
