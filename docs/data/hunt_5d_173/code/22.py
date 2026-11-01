import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    if seconds <= 0.05:
        return x
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        iuc = G[iu0]
        m = float(iuc.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = iuc > thresh
        if int(sel.sum()) > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(iuc[idx])[-3000:]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = iuc[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        cols = np.concatenate([
            (I[:, None] * d + np.arange(d)),
            (J[:, None] * d + np.arange(d)),
            np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        rows2 = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows2, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
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
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_flow(v0, steps, T0, T1, lr0, lr1, beta, noise, rng, iu):
    n, d = v0.shape
    v = v0.copy()
    vel = np.zeros_like(v)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = v @ v.T
        c = G[iu]
        m = float(c.max())
        # softmax weights over upper triangle
        e = np.exp(T * (c - m))
        Zs = e.sum()
        if Zs == 0 or not np.isfinite(Zs):
            break
        w = e / Zs
        W = np.zeros((n, n), dtype=np.float64)
        W[iu] = w
        W = W + W.T
        Grad = W @ v
        dot = (Grad * v).sum(axis=1, keepdims=True)
        Gt = Grad - dot * v
        vel = beta * vel + (1.0 - beta) * Gt
        v = v - lr * vel
        if noise > 0:
            v = v + noise * rng.standard_normal(size=v.shape)
        vn = np.linalg.norm(v, axis=1, keepdims=True)
        vn[vn == 0] = 1.0
        v = v / vn
    return v

def _refine_hole(p0, others, T, steps, lr, beta):
    p = p0.copy()
    nrm = np.linalg.norm(p)
    if nrm == 0:
        p = np.zeros_like(p); p[0] = 1.0
    else:
        p = p / nrm
    vel = np.zeros_like(p)
    for _ in range(steps):
        c = others @ p
        m = float(c.max())
        e = np.exp(T * (c - m))
        Z = e.sum()
        if Z == 0 or not np.isfinite(Z):
            break
        w = e / Z
        grad = w @ others
        gt = grad - float(grad @ p) * p
        vel = beta * vel + (1.0 - beta) * gt
        p = p - lr * vel
        nrm = np.linalg.norm(p)
        if nrm == 0:
            break
        p = p / nrm
    return p

def _find_hole(others, rng, n_sample=30000, n_cand=6):
    n, d = others.shape
    S = rng.standard_normal(size=(n_sample, d))
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    scores = S @ others.T
    mx = scores.max(axis=1)
    idx = np.argsort(mx)[:n_cand]
    best_p = None
    best_s = 1e9
    for k in idx:
        p0 = S[k]
        for T, steps, lr in ((120.0, 220, 0.35), (400.0, 150, 0.15)):
            p0 = _refine_hole(p0, others, T, steps, lr, 0.9)
        s = float((others @ p0).max())
        if s < best_s:
            best_s = s
            best_p = p0
    return best_p, best_s

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(12345)
    n, d = 173, 5
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(an)):
                    vv = _max_cos(an)
                    if vv < best_val:
                        best_val = vv
                        best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    iu = np.triu_indices(n, 1)

    def contacts(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cnt = (G >= m - 1e-9).sum(axis=1)
        return m, cnt, G

    # short initial snap
    rem = deadline - time.time()
    if rem > 8:
        best = _lp_polish(best, min(2.0, rem - 6.0))
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    Temp = 0.0009
    trial = 0
    while True:
        now = time.time()
        rem = deadline - now
        if rem < 7.0:
            break
        pt = 0.85
        if rem < 25:
            pt = max(0.5, rem - 6.0)
        pt = min(pt, rem - 6.0)
        if pt < 0.45:
            break
        m_cur, cnt_cur, G_cur = contacts(cur)
        order_desc = np.argsort(-cnt_cur)
        order_asc = np.argsort(cnt_cur)
        mode = trial % 8
        y = None
        v = 1e9
        if mode in (0, 1, 2, 3, 6):
            y = cur.copy()
            if mode == 0:
                # jammed kick + random
                k_rand = rng.choice(n, 5, replace=False)
                subset = np.unique(np.concatenate([order_desc[:3], k_rand]))
                y[subset] += 0.07 * rng.standard_normal(size=(len(subset), d))
            elif mode == 1:
                y += 0.014 * rng.standard_normal(size=(n, d))
            elif mode == 2:
                victim = int(order_desc[0])
                y[victim] = rng.standard_normal(size=(d,))
                neigh = np.where(G_cur[victim] > m_cur - 0.03)[0]
                neigh = neigh[neigh != victim]
                if len(neigh) > 0:
                    y[neigh] += 0.05 * rng.standard_normal(size=(len(neigh), d))
            elif mode == 3:
                ratt = np.where(cnt_cur == 0)[0]
                k_rand = rng.choice(n, 9, replace=False)
                subset = np.unique(np.concatenate([ratt, k_rand])) if len(ratt) else k_rand
                y[subset] += 0.10 * rng.standard_normal(size=(len(subset), d))
            else:  # mode 6 broad
                subset = rng.choice(n, 22, replace=False)
                y[subset] += 0.045 * rng.standard_normal(size=(len(subset), d))
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            yn[yn == 0] = 1.0
            y = y / yn
            # annealed softmax flow
            if rem > 40:
                steps = 650
            elif rem > 15:
                steps = 450
            else:
                steps = 280
            if mode == 1:
                T0, T1, lr0, lr1, beta, noise = 30.0, 350.0, 0.30, 0.05, 0.90, 0.0
            elif mode == 2:
                T0, T1, lr0, lr1, beta, noise = 25.0, 500.0, 0.35, 0.04, 0.88, 0.0
            else:
                T0, T1, lr0, lr1, beta, noise = 22.0, 420.0, 0.32, 0.05, 0.90, 0.0
            y = _smooth_flow(y, steps, T0, T1, lr0, lr1, beta, noise, rng, iu)
            y = _lp_polish(y, pt)
            v = _max_cos(y)
        elif mode == 4 or mode == 7:
            # hole reinsertion surgery on worst point
            y = cur.copy()
            # try worst 1 point
            victim = int(order_desc[trial % 3])
            others = np.delete(cur, victim, axis=0)
            hole, _ = _find_hole(others, rng, n_sample=26000, n_cand=6)
            yn = np.linalg.norm(hole)
            if yn == 0 or not np.isfinite(yn):
                hole = rng.standard_normal(size=(d,))
                hole = hole / np.linalg.norm(hole)
            y[victim] = hole
            # small shake of old neighbours to open topology
            neigh = np.where(G_cur[victim] > m_cur - 0.03)[0]
            neigh = neigh[neigh != victim]
            if len(neigh) > 0:
                pick = neigh[:min(len(neigh), 4)]
                y[pick] += 0.02 * rng.standard_normal(size=(len(pick), d))
                y = y / np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_flow(y, 260, 40.0, 500.0, 0.22, 0.04, 0.90, 0.0, rng, iu)
            y = _lp_polish(y, pt)
            v = _max_cos(y)
        else:  # mode 5: aggressive global shake + flow (basin jump)
            y = cur.copy()
            subset = rng.choice(n, 40, replace=False)
            y[subset] += 0.09 * rng.standard_normal(size=(len(subset), d))
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_flow(y, 500, 15.0, 300.0, 0.35, 0.06, 0.88, 0.0, rng, iu)
            y = _lp_polish(y, pt)
            v = _max_cos(y)
        # Metropolis accept to cur, greedy for best
        if v < best_val:
            best = y
            best_val = v
            cur = y
            cur_val = v
        else:
            if v < cur_val or rng.random() < np.exp(-(v - cur_val) / Temp):
                cur = y
                cur_val = v
        trial += 1
        # slow anneal of Temp
        Temp = max(0.00025, Temp * 0.9995)

    rem = deadline - time.time()
    if rem > 1.3:
        best = _lp_polish(best, rem - 0.7)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
