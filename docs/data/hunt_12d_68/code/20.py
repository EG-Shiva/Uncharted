import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.25:
            break
        c = (x @ x.T)[iu0]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu0[0][sel], iu0[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            radius *= 0.5
            continue
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def worst_index(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return int(np.argmax(G.max(axis=1)))

def contact_counts(x, delta=0.002):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    return (G > m - delta).sum(axis=1), m

def hole_for_rest(x_rest, rng, ncand=8000, top_rand=False):
    d = x_rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ x_rest.T).max(axis=1)
    if top_rand and rng.random() < 0.35:
        top = np.argsort(scores)[:12]
        return cand[top[int(rng.integers(0, len(top)))]]
    return cand[int(np.argmin(scores))]

def make_fg(n, d, IU0, beta):
    def fg(yf):
        Y = yf.reshape(n, d)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        G = X @ X.T
        g = G[IU0]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        S = float(e.sum()) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[IU0] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def run_lbfgs(x0, beta, maxiter):
    n, d = x0.shape
    IU0 = np.triu_indices(n, 1)
    fg = make_fg(n, d, IU0, beta)
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': maxiter * 3, 'maxcor': 12})
        Y = r.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def short_refine(x, big=False):
    if big:
        x = run_lbfgs(x, 150.0, 150)
        x = run_lbfgs(x, 650.0, 180)
    else:
        x = run_lbfgs(x, 150.0, 110)
        x = run_lbfgs(x, 600.0, 110)
    return x

def solve():
    t0 = time.time()
    TLIM = 172.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        x_init = rng0.standard_normal((68, 12))
        x_init /= np.linalg.norm(x_init, axis=1, keepdims=True)
    else:
        bv = 1e9
        x_init = cands[0]
        for c in cands:
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = max_cos(c)
            if v < bv:
                bv = v
                x_init = c.copy()
    x_init = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
    best_val = max_cos(x_init)
    x_best = x_init.copy()
    n, d = x_best.shape
    rng = np.random.default_rng(987654321)
    pop_x = [x_best.copy()]
    pop_v = [best_val]
    sigmas = [0.004, 0.012, 0.03, 0.08, 0.18, 0.35]
    explore_end = t0 + TLIM - 30.0
    trial = 0
    while time.time() < explore_end and trial < 600:
        trial += 1
        # pick base: mostly best, sometimes other pop member
        if len(pop_x) > 1 and rng.random() < 0.22:
            bi = int(rng.integers(0, len(pop_x)))
            x = pop_x[bi].copy()
        else:
            x = x_best.copy()
        u = rng.random()
        if u < 0.26:
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.46:
            # block re-init around random center (nearest neighbours)
            c0 = int(rng.integers(0, n))
            dots = x @ x[c0]
            order = np.argsort(-dots)
            k = int(rng.integers(4, 13))
            idx = order[:k]
            fresh = rng.standard_normal((k, d))
            fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
            x[idx] = fresh
            # small jitter rest
            x = x + 0.002 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.68:
            # worst / overloaded hole reinsertion 1-3 points
            nw = int(rng.integers(1, 4))
            for _ in range(nw):
                if rng.random() < 0.5:
                    wi = worst_index(x)
                else:
                    cnts, _ = contact_counts(x, delta=0.002)
                    wi = int(np.argmax(cnts + 0.15 * rng.standard_normal(n)))
                mask = np.ones(n, dtype=bool); mask[wi] = False
                hole = hole_for_rest(x[mask], rng, ncand=8000, top_rand=True)
                x[wi] = hole
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif u < 0.83:
            # random cap kill
            uvec = rng.standard_normal(d); uvec /= np.linalg.norm(uvec)
            dots = x @ uvec
            k = int(rng.integers(3, 9))
            idx = np.argsort(-dots)[:k]
            fresh = rng.standard_normal((k, d))
            fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
            x[idx] = fresh
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # mild shake of low-contact subset + global tiny noise
            cnts, _ = contact_counts(x, delta=0.0015)
            order = np.argsort(cnts)  # low contacts first
            k = int(rng.integers(2, 7))
            idx = order[:k]
            x[idx] = x[idx] + 0.25 * rng.standard_normal((k, d))
            x = x + 0.006 * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        # smooth
        big = (u < 0.46) or (trial % 7 == 0)
        x = short_refine(x, big=big)
        if time.time() > explore_end:
            v = max_cos(x)
        else:
            v = max_cos(x)
            # promising -> extra sharpening
            if v < best_val + 0.004 and rng.random() < 0.6:
                x = run_lbfgs(x, 1500.0, 150)
                v = max_cos(x)
        if v < best_val - 1e-12:
            # squeeze to true local optimum with short LP
            if time.time() < explore_end + 6.0:
                try:
                    xl = lp_polish(x, 1.6)
                    vl = max_cos(xl)
                    if vl < v:
                        x, v = xl, vl
                except Exception:
                    pass
            best_val = v
            x_best = x.copy()
            pop_x.append(x.copy()); pop_v.append(v)
            # keep pop small, best 5
            if len(pop_x) > 5:
                o = np.argsort(pop_v)[:5]
                pop_x = [pop_x[i] for i in o]
                pop_v = [pop_v[i] for i in o]
        elif v < best_val + 0.0015:
            # keep diverse near-best in population
            already = False
            for px in pop_x:
                if np.linalg.norm(px - x) < 1e-3 or np.linalg.norm(px + x) < 1e-3:
                    already = True
                    break
            if not already and len(pop_x) < 5:
                pop_x.append(x.copy()); pop_v.append(v)
    # final polish top candidates
    o = np.argsort(pop_v)
    pop_x = [pop_x[i] for i in o]
    pop_v = [pop_v[i] for i in o]
    # also include x_best
    final_cands = pop_x[:2]
    time_left = TLIM - (time.time() - t0)
    per = max(4.0, (time_left - 2.0) / max(1, len(final_cands)))
    best_f = x_best.copy(); best_fv = best_val
    for xc in final_cands:
        if time.time() - t0 > TLIM - 3.0:
            break
        x = xc.copy()
        try:
            x = run_lbfgs(x, 350.0, 300)
            x = run_lbfgs(x, 900.0, 300)
            x = run_lbfgs(x, 2200.0, 250)
        except Exception:
            pass
        try:
            x = lp_polish(x, min(per, 12.0))
        except Exception:
            pass
        v = max_cos(x)
        if v < best_fv:
            best_fv = v
            best_f = x.copy()
    best_f /= np.linalg.norm(best_f, axis=1, keepdims=True)
    return best_f
