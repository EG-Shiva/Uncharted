import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_N = 173
_D = 5

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        C = (X @ X.T)[iu0, iu1]
        m = float(C.max())
        sel = C > m - max(20.0 * radius, 1e-9)
        if int(sel.sum()) > 3500:
            idx = np.where(sel)[0]
            order = np.argsort(C[idx])[-3500:]
            m2 = np.zeros_like(sel)
            m2[idx[order]] = True
            sel = m2
        I = iu0[sel]; J = iu1[sel]
        c = C[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _void_fill(rem, k, rng, n_cand=4000):
    d = rem.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s_max = (cand @ rem.T).max(axis=1)
    chosen = []
    for _ in range(k):
        idx = int(np.argmin(s_max))
        pt = cand[idx].copy()
        chosen.append(pt)
        upd = cand @ pt
        np.maximum(s_max, upd, out=s_max)
        s_max[idx] = 2.0
    return np.stack(chosen, axis=0)

def _anneal(X0, rng, iu0, iu1, alphas, iters_list, lrs, deadline=None):
    n, d = X0.shape
    X = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    m1 = np.zeros_like(X)
    v2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    for alpha, iters, lr in zip(alphas, iters_list, lrs):
        for _ in range(int(iters)):
            if deadline is not None and step % 200 == 0 and time.time() > deadline:
                return X
            step += 1
            G = X @ X.T
            s = G[iu0, iu1]
            mx = float(s.max())
            e = np.exp(alpha * (s - mx))
            se = float(e.sum())
            if se == 0 or not np.isfinite(se):
                continue
            w = e / se
            W = np.zeros((n, n), dtype=np.float64)
            W[iu0, iu1] = w
            W = W + W.T
            grad = W @ X
            dot = (grad * X).sum(axis=1, keepdims=True)
            gt = grad - dot * X
            m1 = b1 * m1 + (1.0 - b1) * gt
            v2 = b2 * v2 + (1.0 - b2) * (gt * gt)
            bc1 = 1.0 - b1 ** step
            bc2 = 1.0 - b2 ** step
            mh = m1 / bc1
            vh = v2 / bc2
            D = mh / (np.sqrt(vh) + eps)
            dpro = (D * X).sum(axis=1, keepdims=True)
            Dt = D - dpro * X
            # clip extreme direction
            Xn = X - lr * Dt
            nn = np.linalg.norm(Xn, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            X = Xn / nn
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(12345)
    n, d = _N, _D
    iu0, iu1 = np.triu_indices(n, 1)

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(an)
                if v < best_val:
                    best_val = v
                    best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    def per_point_max(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1), G

    trial = 0
    # quick ensure best is polished (it already is, skip long)
    while True:
        rem = deadline - time.time()
        if rem < 9.0:
            break
        trial += 1
        mode = trial % 10
        y = best.copy()
        if mode == 1:
            y = y + 0.02 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            al = [30.0, 80.0, 180.0, 350.0]
            it = [250, 250, 300, 300]
            lr = [0.010, 0.008, 0.005, 0.003]
        elif mode == 2:
            y = y + 0.10 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            al = [20.0 + 20*rng.random(), 70.0 + 30*rng.random(), 150.0+80*rng.random(), 300.0+150*rng.random()]
            it = [250, 250, 350, 350]
            lr = [0.012, 0.008, 0.005, 0.003]
        elif mode == 3:
            y = y + 0.30 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            al = [20.0, 60.0, 150.0, 320.0]
            it = [300, 300, 350, 350]
            lr = [0.015, 0.009, 0.005, 0.003]
        elif mode == 4 or mode == 5:
            # cap removal + void fill
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            proj = y @ u
            k = int(rng.integers(4, 9))
            victim = np.argsort(-proj)[:k]
            mask = np.ones(n, dtype=bool); mask[victim] = False
            rem_pts = y[mask]
            try:
                newp = _void_fill(rem_pts, k, rng, n_cand=4000)
            except Exception:
                newp = rng.standard_normal((k, d)); newp /= np.linalg.norm(newp, axis=1, keepdims=True)
            y[victim] = newp
            y = y + (0.005 + 0.01*rng.random()) * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            al = [30.0, 90.0, 200.0, 400.0]
            it = [250, 250, 300, 300]
            lr = [0.010, 0.007, 0.005, 0.0025]
        elif mode == 6 or mode == 7:
            # overload removal
            mi, _ = per_point_max(y)
            order = np.argsort(-mi)
            k = int(rng.integers(4, 9))
            victim = order[:k]
            # add neighbours of overloaded a bit of shake too
            mask = np.ones(n, dtype=bool); mask[victim] = False
            rem_pts = y[mask]
            try:
                newp = _void_fill(rem_pts, k, rng, n_cand=4000)
            except Exception:
                newp = rng.standard_normal((k, d)); newp /= np.linalg.norm(newp, axis=1, keepdims=True)
            y[victim] = newp
            y = y + 0.01 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            al = [25.0+20*rng.random(), 80.0+30*rng.random(), 180.0+60*rng.random(), 380.0]
            it = [250, 250, 300, 300]
            lr = [0.011, 0.008, 0.005, 0.003]
        elif mode == 8:
            # patch restart: replace cap region with random spread + repel-ish noise
            u = rng.standard_normal(d); u /= np.linalg.norm(u)
            proj = y @ u
            thresh = np.sort(proj)[-12:]
            # remove points with proj > quantile
            q = float(np.quantile(proj, 0.93 + 0.04*rng.random()))
            victim = np.where(proj >= q)[0]
            if len(victim) < 2:
                victim = np.argsort(-proj)[:6]
            if len(victim) > 14:
                victim = victim[:14]
            k = len(victim)
            mask = np.ones(n, dtype=bool); mask[victim] = False
            rem_pts = y[mask]
            try:
                newp = _void_fill(rem_pts, k, rng, n_cand=4000)
            except Exception:
                newp = rng.standard_normal((k, d)); newp /= np.linalg.norm(newp, axis=1, keepdims=True)
            y[victim] = newp
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            al = [20.0, 70.0, 170.0, 350.0]
            it = [300, 300, 350, 350]
            lr = [0.013, 0.008, 0.005, 0.003]
        else:  # mode 0,9 : fully random fresh basin
            y = rng.standard_normal((n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            al = [15.0, 50.0, 130.0, 300.0]
            it = [350, 350, 400, 400]
            lr = [0.015, 0.010, 0.006, 0.003]

        # anneal with inner deadline to leave time for polish
        inner_dl = deadline - 7.0
        y2 = _anneal(y, rng, iu0, iu1, al, it, lr, deadline=inner_dl)
        v_raw = _max_cos(y2)
        # only LP-polish if promising (within 0.025 of best) to save time
        # also polish occasional random to explore
        do_polish = (v_raw < best_val + 0.025) or (trial % 12 == 0)
        if not do_polish:
            continue
        rem2 = deadline - time.time()
        if rem2 < 8.0:
            break
        pt = 1.1 if rem2 > 20 else max(0.5, rem2 - 7.0)
        pt = min(pt, rem2 - 6.5)
        if pt < 0.4:
            continue
        try:
            y3 = _lp_polish(y2, pt)
        except Exception:
            continue
        v3 = _max_cos(y3)
        if v3 < best_val - 1e-12:
            best_val = v3
            best = y3.copy()
            # extra polish immediately on improvement if time allows
            rem3 = deadline - time.time()
            if rem3 > 10:
                try:
                    y4 = _lp_polish(best, min(2.0, rem3 - 7.0))
                    v4 = _max_cos(y4)
                    if v4 < best_val:
                        best_val = v4
                        best = y4
                except Exception:
                    pass

    # final polish with remaining time
    rem = deadline - time.time()
    if rem > 1.5:
        try:
            best = _lp_polish(best, max(0.5, rem - 0.8))
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
