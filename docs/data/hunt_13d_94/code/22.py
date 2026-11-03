import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = cmat.max()
        thr = max(30 * radius, 5e-10)
        sel = cmat > m - thr
        if np.count_nonzero(sel) > 3500:
            idx = np.where(sel)[0]
            order = np.argsort(cmat[idx])[-3500:]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        c_sel = cmat[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _smooth(x, steps, k0, k1, lr0, mu, rng, jitter=0.0):
    n, d = x.shape
    if jitter > 0:
        x = x + jitter * rng.standard_normal(size=x.shape)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    v = np.zeros_like(x)
    eye = np.eye(n)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        m = np.max(G - eye * 2.0)
        E = np.exp(k * (G - m))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            break
        Gr = E @ x / S
        dot = (Gr * x).sum(axis=1, keepdims=True)
        Gt = Gr - dot * x
        v = mu * v + Gt
        x = x - lr * v
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x = x / nr
    return x

def _reinsert_vec(others, rng, n_cand=4000):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = (cand @ others.T).max(axis=1)
    c = cand[np.argmin(sc)].copy()
    for it in range(120):
        dots = others @ c
        m = dots.max()
        w = np.exp(60.0 * (dots - m))
        w /= w.sum()
        grad = w @ others
        gt = grad - (grad @ c) * c
        step = 0.12 * (1.0 - 0.5 * it / 120.0)
        c = c - step * gt
        nrm = np.linalg.norm(c)
        if nrm == 0 or not np.isfinite(nrm):
            break
        c /= nrm
    return c

def solve():
    T0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        xb_n = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xb_n):
            xb = xr.copy()
    except Exception:
        pass
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    cur_x = xb.copy()
    n, d = cur_x.shape

    def worst_scores(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return G.max(axis=1)

    def contact_deg(x, tol=1e-3):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        return (G > m - tol).sum(axis=1), G.max(axis=1), G

    it = 0
    HOP_END = T0 + 130.0
    while time.time() < HOP_END and time.time() - T0 < BUDGET - 45:
        it += 1
        deg, sc, Gfull = contact_deg(cur_x)
        y = cur_x.copy()
        mode = it % 8
        if mode == 0:
            # collective cap shift: push a random cap together
            u = rng.standard_normal(size=d); u /= np.linalg.norm(u)
            proj = y @ u
            k = int(rng.integers(8, 22))
            idx = np.argsort(proj)[-k:]
            amp = float(rng.uniform(0.04, 0.14))
            y[idx] += amp * u[None, :]
            y[idx] /= np.linalg.norm(y[idx], axis=1, keepdims=True)
            y = _smooth(y, 550, 18.0, 350.0, 0.5, 0.90, rng)
        elif mode == 1:
            # break top edges apart explicitly then smooth
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            iu0 = np.triu_indices(n, 1)
            c = G[iu0]
            top = np.argsort(c)[-6:]
            step = float(rng.uniform(0.03, 0.09))
            for t in top:
                a = iu0[0][t]; b = iu0[1][t]
                # push a away from b tangentially
                dvec = y[a] - (y[a] @ y[b]) * y[b]
                evec = y[b] - (y[b] @ y[a]) * y[a]
                dn = np.linalg.norm(dvec); en = np.linalg.norm(evec)
                if dn > 1e-12:
                    y[a] += step * dvec / dn * rng.uniform(0.5, 1.5)
                    y[a] /= np.linalg.norm(y[a])
                if en > 1e-12:
                    y[b] += step * evec / en * rng.uniform(0.5, 1.5)
                    y[b] /= np.linalg.norm(y[b])
            y = _smooth(y, 600, 25.0, 500.0, 0.55, 0.88, rng)
        elif mode == 2:
            # reinsert worst point with many candidates
            wi = int(np.argmax(sc))
            others = np.delete(y, wi, axis=0)
            y[wi] = _reinsert_vec(others, rng, n_cand=5000)
            y = _smooth(y, 500, 30.0, 450.0, 0.6, 0.86, rng)
        elif mode == 3:
            # hub + neighbours joint jitter (jammed core)
            hub = int(np.argmax(deg))
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            nbrs = np.where(G[hub] > m - 2e-3)[0]
            if len(nbrs) > 8:
                nbrs = nbrs[np.argsort(G[hub][nbrs])[-8:]]
            y[hub] += 0.10 * rng.standard_normal(size=d)
            y[hub] /= np.linalg.norm(y[hub])
            for nb in nbrs:
                y[nb] += 0.05 * rng.standard_normal(size=d)
                y[nb] /= np.linalg.norm(y[nb])
            y = _smooth(y, 550, 22.0, 400.0, 0.5, 0.90, rng)
        elif mode == 4:
            # low-degree (loose) point reinsertion
            order = np.argsort(deg)
            wi = int(order[0] if rng.random() < 0.7 else order[1 % n])
            others = np.delete(y, wi, axis=0)
            y[wi] = _reinsert_vec(others, rng, n_cand=4000)
            y = _smooth(y, 450, 35.0, 400.0, 0.6, 0.85, rng)
        elif mode == 5:
            # global jitter + low-k exploration
            y = _smooth(y, 650, 15.0, 280.0, 0.45, 0.90, rng, jitter=0.010)
        elif mode == 6:
            # two-point reinsertion
            order = np.argsort(sc)[-2:]
            for w in order:
                others = np.delete(y, int(w), axis=0)
                y[int(w)] = _reinsert_vec(others, rng, n_cand=2500)
            y = _smooth(y, 500, 28.0, 420.0, 0.55, 0.87, rng)
        else:
            # random subset shake
            k = int(rng.integers(3, 7))
            picks = rng.choice(n, size=k, replace=False)
            amp = float(rng.uniform(0.04, 0.10))
            for p in picks:
                y[p] += amp * rng.standard_normal(size=d)
                y[p] /= np.linalg.norm(y[p])
            y = _smooth(y, 500, 25.0, 380.0, 0.55, 0.88, rng)

        v_smooth = _max_cos(y)
        cur_val = _max_cos(cur_x)
        if v_smooth > best_val + 0.012 and v_smooth > cur_val + 0.004:
            if rng.random() < 0.12:
                cur_x = y.copy()
            continue
        remain = BUDGET - (time.time() - T0)
        if remain < 12:
            break
        short_dl = time.time() + min(2.2, max(1.0, remain - 35))
        y2, v2 = _lp_polish_deadline(y, short_dl)
        if v2 < best_val - 1e-12:
            x_best = y2.copy()
            best_val = v2
            cur_x = y2.copy()
        elif v2 < cur_val:
            cur_x = y2.copy()
        else:
            if rng.random() < 0.30:
                cur_x = x_best.copy()
                k = int(rng.integers(1, 4))
                picks = rng.choice(n, size=k, replace=False)
                for p in picks:
                    cur_x[p] += 0.05 * rng.standard_normal(size=d)
                    cur_x[p] /= np.linalg.norm(cur_x[p])
                cur_x = _smooth(cur_x, 300, 25.0, 300.0, 0.5, 0.88, rng)

    remain = BUDGET - (time.time() - T0)
    if remain > 5:
        x_best, best_val = _lp_polish_deadline(x_best, time.time() + remain - 1.0)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
