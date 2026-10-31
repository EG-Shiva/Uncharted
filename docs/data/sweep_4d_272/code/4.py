import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _contact_counts(x, thresh=1e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = g.max()
    c = (g > m - thresh).sum(axis=1)
    # count each undirected once? above counts ordered; divide by 1 (row count is undirected degree)
    return c, m, g

def _lp_polish(x, deadline):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _anneal_lbfgs(x_init, taus, maxiters, t0, time_limit):
    n, d = x_init.shape
    y = _normalize(x_init).ravel().copy()
    for tau, mit in zip(taus, maxiters):
        if time.time() - t0 > time_limit:
            break
        def fun(yflat, _tau=tau, _n=n, _d=d):
            Y = yflat.reshape(_n, _d)
            norms = np.linalg.norm(Y, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X = Y / norms
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            Gmax = float(G.max())
            # exp weights
            E = np.exp((G - Gmax) / _tau)
            np.fill_diagonal(E, 0.0)
            Esum = float(E.sum())
            if Esum <= 0:
                Esum = 1e-300
            S = Esum * 0.5
            loss = Gmax + _tau * np.log(max(S, 1e-300))
            W = E / Esum
            GradX = 2.0 * (W @ X)
            dot = (GradX * X).sum(axis=1, keepdims=True)
            GradY = (GradX - dot * X) / norms
            return float(loss), GradY.ravel()
        try:
            res = minimize(fun, y, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(mit), "ftol": 1e-14, "gtol": 1e-07, "maxfun": int(mit*10)})
            y = res.x
        except Exception:
            break
    X = _normalize(y.reshape(n, d))
    return X


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(12345)
    try:
        x_best0 = np.load("best.npy")
    except Exception:
        x_best0 = np.load("record.npy")
    x_best0 = _normalize(np.asarray(x_best0, dtype=np.float64))
    if x_best0.shape != (272, 4):
        # fallback: random
        x_best0 = _normalize(rng.standard_normal((272, 4)))
    base_val = _max_cos(x_best0)
    best_x = x_best0.copy()
    best_val = base_val

    counts, _, _ = _contact_counts(best_x, thresh=1e-7)
    rattlers = np.where(counts == 0)[0]
    # sort by contacts descending for overload info
    order_desc = np.argsort(-counts)

    candidates = []

    # annealing schedule
    taus = [0.03, 0.015, 0.008, 0.004, 0.002]
    maxiters = [60, 60, 80, 100, 120]

    # trial sigmas: (global_sigma, rattler_resample, block_shake)
    trial_defs = [
        {"sigma": 0.02, "resample_rattlers": False, "block": None},
        {"sigma": 0.05, "resample_rattlers": True, "block": None},
        {"sigma": 0.08, "resample_rattlers": True, "block": None},
        {"sigma": 0.03, "resample_rattlers": False, "block": 24},
        {"sigma": 0.12, "resample_rattlers": True, "block": None},
        {"sigma": 0.06, "resample_rattlers": False, "block": 40},
    ]

    TRIAL_BUDGET_END = t0 + 110.0
    for ti, td in enumerate(trial_defs):
        if time.time() > TRIAL_BUDGET_END or time.time() > HARD_END - 50:
            break
        sigma = td["sigma"]
        x0 = best_x + sigma * rng.standard_normal(best_x.shape)
        if td["resample_rattlers"] and len(rattlers) > 0:
            # randomize rattlers fully
            for r in rattlers:
                x0[r] = rng.standard_normal(4)
            # also randomize a few most overloaded points a bit more
            for idx in order_desc[:4]:
                x0[idx] += 0.15 * rng.standard_normal(4)
        if td["block"] is not None:
            k = int(td["block"])
            sel = rng.choice(272, size=k, replace=False)
            x0[sel] += 0.25 * rng.standard_normal((k, 4))
        x0 = _normalize(x0)
        # quick different taus per trial: shift slightly for diversity
        scale = 1.0 + 0.15 * np.sin(ti * 2.1)
        taus_t = [t * scale for t in taus]
        x_opt = _anneal_lbfgs(x0, taus_t, maxiters, t0, TRIAL_BUDGET_END - t0)
        v = _max_cos(x_opt)
        candidates.append((v, x_opt.copy()))
        if v < best_val:
            best_val = v
            best_x = x_opt.copy()
        # also try a pure random-rattler greedy improvement: move worst point
        # (cheap, no extra global cost) - relocate current worst pair's one end randomly a few times
        # do 200 greedy tries on rattlers only if time allows
        if time.time() < TRIAL_BUDGET_END - 5:
            # local rattler hill-climb on current opt
            xc = x_opt.copy()
            vc = v
            for _ in range(300):
                if time.time() > TRIAL_BUDGET_END:
                    break
                # find worst pair
                G = xc @ xc.T
                np.fill_diagonal(G, -2.0)
                ij = np.unravel_index(int(G.argmax()), G.shape)
                # pick one end at random to move slightly
                mv = ij[rng.integers(2)]
                prop = xc.copy()
                prop[mv] = _normalize((xc[mv] + 0.02 * rng.standard_normal(4)).reshape(1, -1))[0]
                vp = _max_cos(prop)
                if vp < vc:
                    xc, vc = prop, vp
            if vc < v:
                candidates.append((vc, xc.copy()))
                if vc < best_val:
                    best_val = vc
                    best_x = xc.copy()

    # sort candidates, polish top few
    candidates.sort(key=lambda kv: kv[0])
    # always include original
    candidates.append((base_val, x_best0.copy()))
    candidates.sort(key=lambda kv: kv[0])

    polished_best_x = best_x
    polished_best_val = best_val
    # polish top 2 distinct candidates with remaining time
    seen = []
    for v, xc in candidates[:4]:
        if time.time() > HARD_END - 8:
            break
        # skip near-duplicates
        if any(_max_cos((xc + s) / 2) < 1e-12 for s in seen):
            pass
        seen.append(xc)
        # give each ~ remaining/ (remaining candidates) but at least 8s
        deadline = min(HARD_END, time.time() + max(12.0, (HARD_END - time.time()) / 2.5))
        xp, vp = _lp_polish(xc, deadline)
        if vp < polished_best_val:
            polished_best_val = vp
            polished_best_x = xp.copy()
        # early exit if we beat record comfortably
    # final safety: if polishing failed to beat, return best annealed normalized
    out = _normalize(polished_best_x)
    # strict feasibility: no NaN, no zero rows
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    out = out / nrm
    return out
