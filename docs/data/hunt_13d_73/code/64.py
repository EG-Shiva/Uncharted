import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, best, t_deadline, r_init=6e-4):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < t_deadline and radius > 1e-14:
        Gfull = (X @ X.T)[iu0]
        m = float(np.max(Gfull))
        thresh = max(22.0 * radius, 1e-9)
        sel = Gfull > m - thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 1000:
            thr = np.partition(Gfull, -1000)[-1000]
            sel = Gfull >= thr
            idx = np.where(sel)[0]
            if len(idx) > 1100:
                order = np.argsort(-Gfull[idx])[:1100]
                mask = np.zeros_like(sel)
                mask[idx[order]] = True
                sel = mask
        I = iu0[0][sel]; J = iu0[1][sel]
        csub = Gfull[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            X = y; best = v
            radius = min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
    return X, best

def _quasi_samples(nq, d, seed=0):
    try:
        from scipy.stats import qmc, norm
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        # 8192 = 2^13
        u = sampler.random_base2(m=13)
        if u.shape[0] > nq:
            u = u[:nq]
        elif u.shape[0] < nq:
            extra = sampler.random(nq - u.shape[0])
            u = np.vstack([u, extra])
        u = np.clip(u, 1e-12, 1 - 1e-12)
        z = norm.ppf(u)
        z = np.asarray(z, dtype=np.float64)
        nr = np.linalg.norm(z, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        z = z / nr
        bad = ~np.all(np.isfinite(z), axis=1)
        if np.any(bad):
            rng = np.random.RandomState(seed)
            z[bad] = rng.randn(int(np.sum(bad)), d)
            z[bad] /= np.linalg.norm(z[bad], axis=1, keepdims=True)
        return z[:nq]
    except Exception:
        rng = np.random.RandomState(seed)
        z = rng.randn(nq, d)
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        return z

def _triple_lse_opt(Yinit, Core, alpha, t_deadline):
    # Yinit (3,d) unit, Core (70,d)
    d = Core.shape[1]
    x0 = Yinit.ravel().copy()
    def fun_grad(x):
        Y = x.reshape(3, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        U = Y / nrm
        Gtc = U @ Core.T  # 3x70
        c01 = float(U[0] @ U[1]); c02 = float(U[0] @ U[2]); c12 = float(U[1] @ U[2])
        c = np.concatenate([Gtc.ravel(), np.array([c01, c02, c12])])
        m = float(np.max(c))
        e = np.exp(alpha * (c - m))
        s = float(np.sum(e))
        if not np.isfinite(s) or s <= 0:
            return float(m), np.zeros_like(x)
        lse = m + np.log(s) / alpha
        w = e / s
        Wtc = w[:210].reshape(3, 70)
        wtt = w[210:]
        dLdU = Wtc @ Core  # 3xd
        dLdU[0] += wtt[0] * U[1] + wtt[1] * U[2]
        dLdU[1] += wtt[0] * U[0] + wtt[2] * U[2]
        dLdU[2] += wtt[1] * U[0] + wtt[2] * U[1]
        # backprop through normalize
        dLdx = np.zeros_like(Y)
        for k in range(3):
            uk = U[k]; gk = dLdU[k]
            nk = float(nrm[k, 0])
            dLdx[k] = (gk - float(gk @ uk) * uk) / max(nk, 1e-12)
        return float(lse), dLdx.ravel()
    try:
        res = minimize(lambda x: fun_grad(x), x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 60, "maxfun": 150, "ftol": 1e-12, "gtol": 1e-7})
    except Exception:
        return Yinit, np.inf
    if res.x is None or not np.all(np.isfinite(res.x)):
        return Yinit, np.inf
    Y = res.x.reshape(3, d)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    if time.time() > t_deadline:
        pass
    # exact triple-related max
    Gtc = Y @ Core.T
    c01 = float(Y[0] @ Y[1]); c02 = float(Y[0] @ Y[2]); c12 = float(Y[1] @ Y[2])
    f = float(max(np.max(Gtc), c01, c02, c12))
    return Y, f

def solve():
    t0 = time.time()
    DEADLINE = t0 + 176.0
    SURGERY_END = t0 + 110.0
    LP_DEADLINE = DEADLINE - 3.0
    n, d = 73, 13
    # load and safeguard incumbent
    Xbest = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(np.float64)
            if v.shape != (n, d):
                continue
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            if not np.all(np.isfinite(v)):
                continue
            c = _max_cos(v)
            if c < best_val:
                best_val = c; Xbest = v.copy()
        except Exception:
            continue
    if Xbest is None:
        rng0 = np.random.RandomState(0)
        Xbest = rng0.randn(n, d)
        Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
        best_val = _max_cos(Xbest)
    Xinc = Xbest.copy(); vinc = float(best_val)
    rng = np.random.RandomState(12345)

    # active pair + near-contact degrees at slack 1e-5
    try:
        G = Xbest @ Xbest.T
        np.fill_diagonal(G, -2.0)
        m0 = float(np.max(G))
        flat = int(np.argmax(G))
        a = flat // n; b = flat % n
        if a > b:
            a, b = b, a
        slack = 1e-5
        deg = np.sum(G > m0 - slack, axis=1)  # includes symmetric? diag is -2 so fine
        order_low = np.argsort(deg, kind="stable")  # ascending
    except Exception:
        a, b = 4, 22
        deg = np.zeros(n); order_low = np.arange(n)
        m0 = best_val

    # build triple candidates from active pair + lowest-degree points
    lows = [int(x) for x in order_low if int(x) != a and int(x) != b][:8]
    cand_triples = []
    def add_t(t):
        t = tuple(sorted(t))
        if t not in cand_triples and len(set(t)) == 3:
            cand_triples.append(t)
    if len(lows) >= 1:
        add_t((a, b, lows[0]))
    if len(lows) >= 2:
        add_t((a, b, lows[1]))
        add_t((a, lows[0], lows[1]))
        add_t((b, lows[0], lows[1]))
    if len(lows) >= 3:
        add_t((a, lows[0], lows[2]))
        add_t((b, lows[1], lows[2]))
        add_t((a, b, lows[2]))
        add_t((lows[0], lows[1], lows[2]))
    if len(lows) >= 4:
        add_t((a, lows[2], lows[3]))
        add_t((b, lows[0], lows[3]))
    cand_triples = cand_triples[:10]
    if len(cand_triples) == 0:
        cand_triples = [(a, b, int(order_low[0]))]

    # 8192 quasi-random samples once
    Q = _quasi_samples(8192, d, seed=0)

    alphas = [300.0, 800.0, 2000.0, 5000.0]
    max_restarts = 20
    restarts_done = 0
    best_triple = None; best_triple_S = None; best_triple_f = float(best_val)

    for S in cand_triples:
        if time.time() > SURGERY_END or restarts_done >= max_restarts:
            break
        if time.time() > DEADLINE - 40:
            break
        S = tuple(int(x) for x in S)
        mask = np.ones(n, dtype=bool); mask[list(S)] = False
        Core = Xbest[mask]  # 70xd
        # greedy seeding with Q
        try:
            s_core = np.max(Q @ Core.T, axis=1)  # 8192
            i1 = int(np.argmin(s_core))
            y1 = Q[i1]
            s2 = np.maximum(s_core, Q @ y1)
            # avoid re-pick same
            s2[i1] = 2.0
            i2 = int(np.argmin(s2))
            y2 = Q[i2]
            s3 = np.maximum(s2, np.maximum(Q @ y1, Q @ y2))
            s3[i1] = 2.0; s3[i2] = 2.0
            i3 = int(np.argmin(s3))
            y3 = Q[i3]
            Y0 = np.vstack([y1, y2, y3])
        except Exception:
            Y0 = Xbest[list(S)].copy()
        # number of restarts for this triple: 3 (or fewer if budget low)
        nrest = min(3, max_restarts - restarts_done)
        for r in range(nrest):
            if time.time() > SURGERY_END or time.time() > DEADLINE - 40:
                break
            alpha = alphas[(restarts_done) % len(alphas)]
            if r == 0:
                Yinit = Y0.copy()
            else:
                # small random tangent perturbation of Y0
                pert = rng.randn(3, d) * 0.02
                Yinit = Y0 + pert
                Yinit /= np.linalg.norm(Yinit, axis=1, keepdims=True)
            restarts_done += 1
            try:
                Yopt, f = _triple_lse_opt(Yinit, Core, alpha, SURGERY_END)
            except Exception:
                continue
            if not np.all(np.isfinite(Yopt)):
                continue
            if f < best_triple_f - 1e-12:
                # verify full exact maximum with insertion
                Xcand = Xbest.copy()
                Xcand[list(S)] = Yopt
                Xcand /= np.linalg.norm(Xcand, axis=1, keepdims=True)
                try:
                    vfull = _max_cos(Xcand)
                except Exception:
                    continue
                if vfull < best_triple_f:
                    best_triple_f = float(vfull)
                    best_triple = Yopt.copy()
                    best_triple_S = S
            if time.time() > SURGERY_END:
                break

    # insert best triple only if full exact maximum improves
    if best_triple is not None and best_triple_S is not None and best_triple_f < best_val - 1e-12:
        Xcand = Xbest.copy()
        Xcand[list(best_triple_S)] = best_triple
        Xcand /= np.linalg.norm(Xcand, axis=1, keepdims=True)
        try:
            v = _max_cos(Xcand)
            if v < best_val and np.all(np.isfinite(Xcand)):
                Xbest = Xcand; best_val = v
        except Exception:
            pass

    # reserve final ~30s for one LP polish; if no time, skip
    t_now = time.time()
    # ensure we reserve final 30s meaning polish gets at most 30s and we stop surgery by 110s (done)
    if t_now < LP_DEADLINE - 1.0:
        try:
            Xp, vp = _lp_polish(Xbest, float(best_val), LP_DEADLINE, r_init=6e-4)
            if np.all(np.isfinite(Xp)) and vp < best_val:
                Xbest, best_val = Xp, vp
        except Exception:
            pass

    # safeguard: if somehow worse than incumbent, return incumbent
    try:
        vfinal = _max_cos(Xbest)
    except Exception:
        return Xinc
    if not np.isfinite(vfinal) or vfinal > vinc + 1e-12:
        return Xinc
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
