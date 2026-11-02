import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, m, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > m - tol, axis=1)

def _lp_polish(x, deadline, init_radius=5e-3, cap=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            cv = c[idx]
            part = np.argpartition(cv, -cap)[-cap:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = cap
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cm = cols.reshape(p, 2 * d + 1)
        Vm = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cm[:, :d] = I[:, None] * d + ar
        Cm[:, d:2 * d] = J[:, None] * d + ar
        Cm[:, 2 * d] = n * d
        Vm[:, :d] = x[J]
        Vm[:, d:2 * d] = x[I]
        Vm[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x

def _smooth(x, steps, tau0, tau1, lr0, lr1, mu, deadline, iuI, iuJ):
    n, d = x.shape
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    best = float(G0.max())
    best_x = x.copy()
    W = np.empty((n, n), dtype=np.float64)
    for s in range(steps):
        if time.time() >= deadline:
            break
        frac = s / max(1, steps - 1)
        tau = tau0 + (tau1 - tau0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        e = np.exp((c - m) / tau)
        S = float(e.sum())
        if S == 0 or not np.isfinite(S):
            break
        e /= S
        W.fill(0.0)
        W[iuI, iuJ] = e
        W[iuJ, iuI] = e
        Wx = W @ x
        dot = np.sum(Wx * x, axis=1, keepdims=True)
        Gt = Wx - dot * x
        vel = mu * vel + Gt
        x = x - lr * vel
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
        if m < best:
            best = m
            best_x = x.copy()
    return best_x, best

def _hole_for(k, y, rng, n_cand=8000):
    n, d = y.shape
    others = np.delete(y, k, axis=0)
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    return cand[bi]

def solve():
    t0 = time.time()
    HARD = t0 + 168.0
    EXPLORE_END = t0 + 138.0
    rng = np.random.default_rng(20261101)
    x0 = None; b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < b0:
                        b0 = v; x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        x0 = rng.standard_normal(size=(323, 7))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        b0 = _max_cos(x0)
    x_best = x0.copy(); best = float(b0)
    n, d = x_best.shape
    iuI, iuJ = np.triu_indices(n, 1)
    it = 0; no_imp = 0
    while time.time() < EXPLORE_END and time.time() < HARD - 10:
        mode = it % 5
        x = x_best.copy()
        if mode == 0:
            sig = 0.020 * (1.0 + 0.07 * min(no_imp, 12))
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            dl = min(time.time() + 0.45, HARD - 8)
            x, _ = _smooth(x, 550, 0.014, 0.0025, 4.0, 0.6, 0.8, dl, iuI, iuJ)
        elif mode == 1:
            G = x @ x.T; np.fill_diagonal(G, -2.0)
            w = G.max(axis=1)
            k = 26
            idx = np.argpartition(w, -k)[-k:]
            x[idx] += rng.standard_normal(size=(k, d)) * 0.065
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            dl = min(time.time() + 0.40, HARD - 8)
            x, _ = _smooth(x, 450, 0.011, 0.0022, 3.2, 0.5, 0.78, dl, iuI, iuJ)
        elif mode == 2:
            m = _max_cos(x)
            cc = _contacts(x, m)
            hubs = np.argsort(-cc)[:3]
            for h in hubs:
                x[int(h)] = _hole_for(int(h), x, rng, n_cand=8000)
            x += rng.standard_normal(size=x.shape) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            dl = min(time.time() + 0.45, HARD - 8)
            x, _ = _smooth(x, 500, 0.010, 0.0022, 3.0, 0.5, 0.8, dl, iuI, iuJ)
        elif mode == 3:
            m = _max_cos(x)
            cc = _contacts(x, m)
            ratt = np.where(cc == 0)[0].tolist()
            order = np.argsort(-cc)
            torep = [int(r) for r in ratt[:4]]
            for o in order:
                if len(torep) >= 5:
                    break
                if int(o) not in torep:
                    torep.append(int(o))
            if not torep:
                G = x @ x.T; np.fill_diagonal(G, -2.0)
                w = G.max(axis=1)
                torep = np.argpartition(w, -5)[-5:].tolist()
            for k in torep:
                x[int(k)] = _hole_for(int(k), x, rng, n_cand=7000)
            x += rng.standard_normal(size=x.shape) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            dl = min(time.time() + 0.40, HARD - 8)
            x, _ = _smooth(x, 450, 0.009, 0.002, 2.8, 0.45, 0.8, dl, iuI, iuJ)
        else:
            G = x @ x.T; np.fill_diagonal(G, -2.0)
            c = G[iuI, iuJ]
            top = np.argpartition(c, -10)[-10:]
            for t in top:
                i = int(iuI[t]); j = int(iuJ[t])
                di = x[i] - x[j]
                di = di - np.dot(di, x[i]) * x[i]
                dj = -di
                ni = np.linalg.norm(di)
                if ni > 1e-12:
                    x[i] = x[i] + 0.055 * di / ni
                    dj2 = x[j] - x[i]
                    dj2 = dj2 - np.dot(dj2, x[j]) * x[j]
                    nj = np.linalg.norm(dj2)
                    if nj > 1e-12:
                        x[j] = x[j] + 0.055 * dj2 / nj
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            dl = min(time.time() + 0.40, HARD - 8)
            x, _ = _smooth(x, 450, 0.008, 0.002, 2.5, 0.4, 0.75, dl, iuI, iuJ)
        v = _max_cos(x)
        improved = (v < best - 1e-12)
        do_lp = False
        if improved:
            do_lp = True
        elif (it % 5 == 2 or it % 5 == 3) and (it % 10 == 2):
            do_lp = True
        elif no_imp >= 8 and (it % 4 == 0):
            do_lp = True
        if do_lp and time.time() < HARD - 10:
            lp_dead = min(time.time() + 1.1, HARD - 8, EXPLORE_END + 6)
            if v < best:
                lp_dead = min(lp_dead + 0.5, HARD - 8)
            try:
                x = _lp_polish(x, deadline=lp_dead, init_radius=0.006)
            except Exception:
                pass
            v = _max_cos(x)
            if v < best - 1e-12 and np.all(np.isfinite(x)):
                best = v; x_best = x.copy(); no_imp = 0
            else:
                if v < best - 1e-12:
                    best = v; x_best = x.copy(); no_imp = 0
                else:
                    no_imp += 1
        else:
            if improved and np.all(np.isfinite(x)):
                best = v; x_best = x.copy(); no_imp = 0
            else:
                no_imp += 1
        it += 1
        if time.time() >= HARD - 9:
            break
    if time.time() < HARD - 2:
        try:
            dl = min(HARD - 1.0, time.time() + max(3.0, HARD - 1.0 - time.time()))
            x_best = _lp_polish(x_best, deadline=dl, init_radius=2.5e-3, cap=2500)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    for i in range(n):
        r = x_best[i]
        if not np.all(np.isfinite(r)) or np.linalg.norm(r) < 1e-12:
            v = rng.standard_normal(size=(d,))
            v /= np.linalg.norm(v)
            x_best[i] = v
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
