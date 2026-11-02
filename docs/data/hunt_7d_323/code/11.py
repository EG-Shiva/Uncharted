import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _per_point_worst(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G


def _gd_polish(x, steps, lr, beta, delta, mu, rng=None):
    n, d = x.shape
    v = np.zeros_like(x)
    for it in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # weights: exp(beta*(G-m)) for pairs within delta of max
        # compute in stable way
        S = G - m  # <=0
        mask = S > -delta
        W = np.zeros_like(G)
        # exp for masked only
        W[mask] = np.exp(beta * S[mask])
        np.fill_diagonal(W, 0.0)
        # gradient of weighted sum: Grad = W @ x (symmetric, factor 2 absorbed in lr)
        Grad = W @ x
        # tangent projection
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        # normalize step by mean norm to keep lr stable
        gn = np.linalg.norm(Gt) / n + 1e-12
        # scale lr
        step = lr / (1.0 + gn)
        v = mu * v + step * Gt
        x = x - v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _greedy_hole(x, idx_list, rng, n_cand=20000):
    n, d = x.shape
    for k in idx_list:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(x, k, axis=0)
        dots = cand @ others.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        x[k] = cand[bi]
    return x


def _lp_polish(x, seconds, init_radius=4e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 3000:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -3000)[-3000:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 3000
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
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
        vrow[:, :d] = x[J]
        vrow[:, d:2 * d] = x[I]
        vrow[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
    rng = np.random.default_rng(12345)
    # load start
    best_x = None
    best_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        best_x = a
        best_v = _max_cos(a)
    cur_x = best_x.copy()
    cur_v = best_v

    # quick initial GD to settle numerical noise (cheap)
    # hopping budget until ~130s, reserve ~35s for LP
    hop_deadline = t0 + 130.0
    lp_budget = 32.0

    it = 0
    # annealing temperature for accepting worse basins
    while time.time() - t0 < hop_deadline - 2.0:
        it += 1
        frac = (time.time() - t0) / max(hop_deadline - t0, 1.0)
        temp = 0.0015 * (1.0 - frac)  # allow ~1e-3 uphill early
        # pick strategy
        mode = it % 5
        trial = cur_x.copy()
        w, G = _per_point_worst(trial)
        order = np.argsort(-w)
        if mode == 0:
            # kick top hubs hard
            k = int(rng.integers(2, 5))
            idx = order[:k]
            sig = 0.08 + 0.10 * rng.random()
            trial[idx] += sig * rng.standard_normal(size=(k, 7))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            # greedy hole for one of them sometimes
            if rng.random() < 0.3:
                trial = _greedy_hole(trial, [int(idx[0])], rng, n_cand=15000)
            steps, lr, beta, delta, mu = 600, 0.015, 220.0, 0.015, 0.85
        elif mode == 1:
            # global jitter + medium polish
            sig = 0.008 + 0.015 * rng.random()
            trial += sig * rng.standard_normal(size=trial.shape)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            steps, lr, beta, delta, mu = 500, 0.012, 280.0, 0.012, 0.9
        elif mode == 2:
            # relocate worst single point to hole + polish neighbors
            k = int(order[0])
            trial = _greedy_hole(trial, [k], rng, n_cand=25000)
            # jitter its close neighbors
            gk = G[k].copy()
            gk[k] = -10
            nbr = np.argsort(-gk)[:6]
            trial[nbr] += 0.03 * rng.standard_normal(size=(len(nbr), 7))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            steps, lr, beta, delta, mu = 700, 0.015, 200.0, 0.02, 0.85
        elif mode == 3:
            # cluster shake: pick random seed, shake its contact cluster
            s = int(rng.integers(0, 323))
            gs = G[s].copy()
            # contacts = top dots
            m = float(gs.max())
            cl = np.where(gs > m - 0.03)[0]
            if len(cl) < 3:
                cl = np.argsort(-gs)[:8]
            else:
                cl = cl[:12]
            trial[cl] += 0.06 * rng.standard_normal(size=(len(cl), 7))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            steps, lr, beta, delta, mu = 600, 0.014, 250.0, 0.015, 0.88
        else:
            # mild kick of random 4-10 points
            k = int(rng.integers(4, 11))
            idx = rng.choice(323, size=k, replace=False)
            trial[idx] += 0.05 * rng.standard_normal(size=(k, 7))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            steps, lr, beta, delta, mu = 500, 0.012, 300.0, 0.010, 0.9

        # anneal beta/lr slightly with frac
        beta = beta * (1.0 + 0.5 * frac)
        try:
            trial = _gd_polish(trial, steps, lr, beta, delta, mu)
        except Exception:
            continue
        # quick local hill refine on worst point (tiny)
        v = _max_cos(trial)
        if v < cur_v or (v < cur_v + temp and rng.random() < 0.25):
            cur_x, cur_v = trial, v
            if v < best_v - 1e-12:
                best_v = v
                best_x = trial.copy()
        # also track best even without acceptance
        elif v < best_v - 1e-12:
            best_v = v
            best_x = trial.copy()
            # jump to new best occasionally
            if rng.random() < 0.5:
                cur_x, cur_v = trial, v
        # time check for early excellent improvement: keep hopping
        if time.time() - t0 > hop_deadline - 2.0:
            break
        # safety: if approaching total limit, break
        if time.time() - t0 > TIME_LIMIT - 40.0:
            break

    # final polish of best with remaining time
    remain = TIME_LIMIT - (time.time() - t0) - 2.0
    if remain < 5.0:
        remain = 5.0
    remain = min(remain, lp_budget + 8.0)
    # short GD settle then LP
    try:
        if time.time() - t0 < TIME_LIMIT - remain - 1.0:
            best_x = _gd_polish(best_x, 800, 0.008, 350.0, 0.008, 0.9)
            bv = _max_cos(best_x)
            if bv < best_v:
                best_v = bv
    except Exception:
        pass
    try:
        t_lp = max(4.0, min(remain, TIME_LIMIT - (time.time() - t0) - 1.0))
        if t_lp > 1.0:
            y = _lp_polish(best_x, t_lp, init_radius=3e-3)
            yv = _max_cos(y)
            if yv < best_v:
                best_x, best_v = y, yv
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
