import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

TRI_I = None
TRI_J = None

def lp_polish(x, deadline, radius0=3e-3, cap=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:cap]
            msel = np.zeros_like(sel, dtype=bool)
            msel[idx[order]] = True
            sel = msel
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def lse_stage(x_init, T, maxiter, iu0, iu1):
    n, d = x_init.shape
    Y0 = x_init.ravel().copy()
    lam = 1.0
    def fun(Yflat):
        Y = Yflat.reshape(n, d)
        r = np.linalg.norm(Y, axis=1)
        r = np.maximum(r, 1e-12)
        X = Y / r[:, None]
        G = X @ X.T
        g = G[iu0, iu1]
        gmax = float(g.max())
        e = np.exp((g - gmax) / T)
        S = float(e.sum())
        f = gmax + T * np.log(S + 1e-300)
        w = e / (S + 1e-300)
        W = np.zeros((n, n))
        W[iu0, iu1] = w
        W = W + W.T
        grad_X = W @ X
        dot = (grad_X * X).sum(axis=1, keepdims=True)
        grad_Y = (grad_X - dot * X) / r[:, None]
        diff = (r - 1.0)[:, None]
        f += lam * float(np.sum(diff ** 2))
        grad_Y += 2.0 * lam * diff * X
        return f, grad_Y.ravel()
    try:
        res = minimize(fun, Y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12,
                                "gtol": 1e-6, "maxfun": maxiter * 8, "maxcor": 10})
        Yopt = res.x.reshape(n, d)
    except Exception:
        return x_init
    rn = np.linalg.norm(Yopt, axis=1, keepdims=True)
    rn[rn == 0] = 1.0
    return Yopt / rn

def row_info(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    return rm, G

def greedy_holes(x, drop_idx, rng, S=20000):
    n, d = x.shape
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[drop_idx] = False
    others = x[keep_mask]
    samp = rng.standard_normal((S, d))
    samp /= np.linalg.norm(samp, axis=1, keepdims=True)
    # incremental greedy for each dropped point
    placed = []
    out = x.copy()
    for k, idx in enumerate(drop_idx):
        if len(placed) > 0:
            pl = np.stack(placed, axis=0)
            all_o = np.concatenate([others, pl], axis=0)
        else:
            all_o = others
        scores = samp @ all_o.T
        s_max = scores.max(axis=1)
        bi = int(np.argmin(s_max))
        best_v = samp[bi]
        out[idx] = best_v
        placed.append(best_v)
    return out

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)
    n_dim = (173, 5)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == n_dim:
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal(n_dim)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    x_best = None
    best = 2.0
    for a in cands:
        v = max_cos(a)
        if v < best:
            best = v
            x_best = a.copy()
    n, d = n_dim
    iu0, iu1 = np.triu_indices(n, 1)
    x_cur = x_best.copy()
    cur_val = best
    basin_T = 0.0025
    it = 0
    # quick ensure local opt if time allows (very short)
    if time.time() - t0 < BUDGET - 20:
        x_best, best = lp_polish(x_best, time.time() + 2.0, radius0=1e-3)
        x_cur = x_best.copy(); cur_val = best

    while True:
        remain = BUDGET - (time.time() - t0)
        if remain < 7.0:
            break
        it += 1
        mode = it % 7
        x_try = x_cur.copy()
        if mode == 0:
            sigma = float(rng.choice([0.008, 0.02, 0.05, 0.12, 0.25]))
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = lse_stage(x_try, 0.02, 60, iu0, iu1)
            x_try = lse_stage(x_try, 0.005, 60, iu0, iu1)
        elif mode == 1:
            rm, _ = row_info(x_try)
            k = 14 + int(rng.integers(0, 8))
            worst = np.argsort(rm)[-k:]
            sigma = 0.15 + 0.12 * float(rng.random())
            x_try[worst] += sigma * rng.standard_normal((k, d))
            xn = np.linalg.norm(x_try[worst], axis=1, keepdims=True)
            xn[xn == 0] = 1.0
            x_try[worst] /= xn
            x_try = lse_stage(x_try, 0.025, 50, iu0, iu1)
            x_try = lse_stage(x_try, 0.004, 70, iu0, iu1)
        elif mode == 2:
            rm, G = row_info(x_try)
            # overloaded by contact count within 0.004 of global max
            m = float(rm.max())
            counts = (G > m - 0.004).sum(axis=1)
            # pick worst 2 to reinsert into holes
            worst2 = np.argsort(rm)[-2:]
            # also include a high-count point sometimes
            if rng.random() < 0.5:
                worst2 = np.append(worst2, np.argmax(counts))
                worst2 = np.unique(worst2)
            x_try = greedy_holes(x_try, worst2, rng, S=20000)
            # jitter neighbours of reinserted
            for wi in worst2:
                dots = x_try @ x_try[int(wi)]
                nbrs = np.argsort(dots)[::-1][1:7]
                x_try[nbrs] += 0.06 * rng.standard_normal((len(nbrs), d))
                x_try[nbrs] /= np.linalg.norm(x_try[nbrs], axis=1, keepdims=True)
            x_try = lse_stage(x_try, 0.015, 60, iu0, iu1)
            x_try = lse_stage(x_try, 0.0035, 70, iu0, iu1)
        elif mode == 3:
            sigma = 0.05 + 0.03 * float(rng.random())
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = lse_stage(x_try, 0.03, 50, iu0, iu1)
            x_try = lse_stage(x_try, 0.006, 60, iu0, iu1)
        elif mode == 4:
            k = int(rng.integers(5, 16))
            idx = rng.choice(n, size=k, replace=False)
            rnd = rng.standard_normal((k, d))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            x_try[idx] = rnd
            x_try = lse_stage(x_try, 0.03, 60, iu0, iu1)
            x_try = lse_stage(x_try, 0.005, 60, iu0, iu1)
        elif mode == 5:
            # rattler + neighbour surgery
            rm, G = row_info(x_try)
            m = float(rm.max())
            # rattlers: far below max
            ratt = np.where(rm < m - 0.01)[0]
            if len(ratt) == 0:
                ratt = np.argsort(rm)[:3]
            # randomize rattlers
            for r in ratt[:4]:
                v = rng.standard_normal(d)
                v /= np.linalg.norm(v)
                x_try[int(r)] = v
            # shake most loaded
            counts = (G > m - 0.003).sum(axis=1)
            top = np.argsort(counts)[-6:]
            x_try[top] += 0.10 * rng.standard_normal((len(top), d))
            x_try[top] /= np.linalg.norm(x_try[top], axis=1, keepdims=True)
            x_try = lse_stage(x_try, 0.018, 60, iu0, iu1)
            x_try = lse_stage(x_try, 0.004, 70, iu0, iu1)
        else:
            sigma = 0.28
            x_try = x_try + sigma * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_try = lse_stage(x_try, 0.04, 50, iu0, iu1)
            x_try = lse_stage(x_try, 0.006, 60, iu0, iu1)

        lp_budget = min(1.6, remain - 3.0)
        if lp_budget < 0.6:
            break
        x_pol, v_pol = lp_polish(x_try, time.time() + lp_budget, radius0=3e-3)
        if v_pol < best - 1e-12:
            x_best, best = x_pol, v_pol
            # move current to new best
            x_cur, cur_val = x_pol.copy(), v_pol
        else:
            # Metropolis move of current (explore even if worse)
            if v_pol < cur_val or rng.random() < np.exp((cur_val - v_pol) / max(basin_T, 1e-6)):
                x_cur, cur_val = x_pol.copy(), v_pol
        # decay basin temperature slowly
        basin_T *= 0.9985
        basin_T = max(basin_T, 0.00025)
        # occasional restart current from best to avoid drift too far
        if it % 25 == 0 and rng.random() < 0.5:
            # pull back if current much worse than best
            if cur_val > best + 0.004:
                x_cur, cur_val = x_best.copy(), best

    remain = BUDGET - (time.time() - t0)
    if remain > 2.0:
        x_best, best = lp_polish(x_best, time.time() + remain - 0.6, radius0=5e-4)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
