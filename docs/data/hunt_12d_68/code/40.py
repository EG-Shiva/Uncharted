import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_perturb(X, sigma, rng):
    if sigma <= 0:
        return X.copy()
    N = rng.standard_normal(X.shape)
    N -= np.sum(N * X, axis=1, keepdims=True) * X
    Y = X + sigma * N
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _hinge_opt(x0, tau, maxiter, ftol=1e-14, gtol=1e-7):
    n, d = x0.shape
    IU = np.triu_indices(n, 1)
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        G = X @ X.T
        # hinge residual upper triangle
        g = G[IU] - tau
        e = g > 0
        # F
        F = float(np.sum(g[e] * g[e]))
        # gradient: P = 2*relu
        # build full symmetric E
        E = np.zeros((n, n))
        w = np.zeros_like(g)
        w[e] = 2.0 * g[e]
        E[IU] = w
        E = E + E.T
        GX = E @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                              'maxfun': maxiter * 3, 'maxcor': 25})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y, float(r.fun)
    except Exception:
        return x0, 1e99

def _lp_polish(x, deadline, cap_p=2000, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _lse_opt(x0, beta, maxiter):
    n, d = x0.shape
    IU = np.triu_indices(n, 1)
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n)); P[IU] = w; P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7,
                              'maxfun': maxiter * 2})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    explore_end = t0 + 132.0
    rng = np.random.default_rng(7)
    n, d = 68, 12
    bestX = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    s = _max_cos(a)
                    if s < best:
                        best = s
                        bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best = _max_cos(bestX)
    incumbent = bestX.copy()

    taus = [0.2198, 0.2195, 0.2192, 0.2188, 0.2184, 0.2180, 0.2176, 0.2172]
    tau_idx = 0
    # if incumbent already below first tau, advance
    while tau_idx < len(taus) and best <= taus[tau_idx] + 1e-12:
        tau_idx += 1
    if tau_idx >= len(taus):
        taus.append(best - 0.0003)

    sigmas = [0.0, 0.01, 0.02, 0.03, 0.05, 0.015, 0.04]
    attempt = 0
    stall = 0
    cur_tau = taus[min(tau_idx, len(taus)-1)]
    # candidate pool for final polish
    pool = [(best, bestX.copy())]

    while time.time() < explore_end and tau_idx < len(taus):
        tau = taus[tau_idx]
        sigma = sigmas[attempt % len(sigmas)]
        # after stall, force larger kicks
        if stall >= 6:
            sigma = float(rng.choice([0.03, 0.05, 0.07]))
        if attempt == 0:
            x_start = bestX.copy()
        else:
            # start from current best (lowest max) with tangent kick
            base = bestX.copy()
            # occasionally start from incumbent to diversify
            if attempt % 9 == 8:
                base = incumbent.copy()
            x_start = _tangent_perturb(base, sigma if sigma > 0 else 0.02, rng)
        # hinge minimise; fewer iters for speed, more when close
        maxiter = 350 if stall < 4 else 500
        x_opt, F = _hinge_opt(x_start, tau, maxiter)
        v = _max_cos(x_opt)
        attempt += 1
        if v < best - 1e-12:
            best = v
            bestX = x_opt.copy()
            pool.append((best, bestX.copy()))
            pool = sorted(pool, key=lambda t: t[0])[:4]
            stall = 0
        else:
            stall += 1
        # feasibility check
        if v <= tau + 1e-12:
            # feasible: lower tau, reset stall, continue from this point
            tau_idx += 1
            stall = 0
            # keep bestX as feasible point (already)
            if tau_idx >= len(taus):
                # extend further down
                taus.append(tau - 0.0004)
            continue
        # if hinge value very small but max still above tau (few violators),
        # keep x_opt as base for next try even if not global best? use it
        # to rearrange active graph: occasionally adopt it
        if F < 1e-10 and rng.random() < 0.3:
            bestX = x_opt.copy()
            # don't update best (max didn't improve) but move basin
            stall = max(0, stall - 1)
        # anti-stall: if many attempts without feasibility, re-perturb strongly
        if stall >= 10:
            # big random tangent jump from best to new basin, keep same tau
            bestX = _tangent_perturb(bestX, 0.08, rng)
            stall = 0
        # time guard for inner loop
        if time.time() >= explore_end:
            break

    # final squeeze: LP polish + LSE on pool members
    deadline_lp = T_END - 2.0
    # sort pool best first
    pool = sorted(pool, key=lambda t: t[0])
    # also add incumbent-based hinge best
    polished = []
    for v, x in pool[:3]:
        if time.time() > deadline_lp - 1.0:
            break
        # quick LSE quench at high beta to equalise before LP
        try:
            xq = _lse_opt(x, 600.0, 120)
            vq = _max_cos(xq)
            if vq < v:
                x, v = xq, vq
        except Exception:
            pass
        xp = _lp_polish(x, min(deadline_lp, time.time() + 9.0))
        vp = _max_cos(xp)
        polished.append((vp, xp))
    if polished:
        polished.sort(key=lambda t: t[0])
        bv, bx = polished[0]
        if bv < best:
            best, bestX = bv, bx
    else:
        # at least polish bestX
        bestX = _lp_polish(bestX, deadline_lp)
        best = _max_cos(bestX)

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # tiny shrink guard (no-op, ensures finite)
    bestX = np.where(np.isfinite(bestX), bestX, rng.standard_normal((n, d)))
    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
