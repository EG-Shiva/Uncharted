import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _worst(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _ccounts(x, m, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > m - tol, axis=1)

def _lp_polish(x, deadline, init_radius=4e-3, max_it=30):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < max_it:
        it += 1
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 2000:
            idx = np.where(sel)[0]
            cv = c[idx]
            part = np.argpartition(cv, -2000)[-2000:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 2000
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cm = cols.reshape(p, 2 * d + 1)
        Vm = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cm[:, :d] = I[:, None] * d + ar
        Cm[:, d:2*d] = J[:, None] * d + ar
        Cm[:, 2*d] = n * d
        Vm[:, :d] = x[J]
        Vm[:, d:2*d] = x[I]
        Vm[:, 2*d] = -1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        tl = float(deadline - time.time() - 0.15)
        if tl < 0.2:
            break
        tl = min(tl, 3.0)
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0 or res.x is None:
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
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _anneal(x, steps, tau0, tau1, lr0, lr1, mu, deadline, topk=6000):
    n, d = x.shape
    iuI, iuJ = np.triu_indices(n, 1)
    vel = np.zeros_like(x)
    best = _max_cos(x)
    bx = x.copy()
    for s in range(steps):
        if time.time() >= deadline:
            break
        frac = s / max(1, steps - 1)
        tau = tau0 + (tau1 - tau0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        # truncated softmax: only pairs within cut
        cut = m - 6.0 * tau
        # keep at most topk
        if c.size > topk:
            # quick threshold via partition
            thr = np.partition(c, -topk)[-topk]
            if thr > cut:
                cut = thr
        mask = c > cut
        if not np.any(mask):
            mask = c > (m - 1e-12)
        cm = c[mask] - m
        e = np.exp(cm / max(tau, 1e-12))
        S = float(e.sum())
        if not np.isfinite(S) or S == 0:
            break
        w = e / S
        Im = iuI[mask]; Jm = iuJ[mask]
        Wx = np.zeros_like(x)
        np.add.at(Wx, Im, w[:, None] * x[Jm])
        np.add.at(Wx, Jm, w[:, None] * x[Im])
        dot = np.sum(Wx * x, axis=1, keepdims=True)
        Gt = Wx - dot * x
        vel = mu * vel + Gt
        x = x - lr * vel
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
        if (s % 60 == 59) or (s == steps - 1):
            v = _max_cos(x)
            if v < best:
                best = v; bx = x.copy()
    return bx

def _reinsert(y, lst, rng, n_cand=8000):
    n, d = y.shape
    for k in lst:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(y, int(k), axis=0)
        dots = cand @ others.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        y[int(k)] = cand[bi]
    for k in lst:
        k = int(k)
        others = np.delete(y, k, axis=0)
        cur = y[k].copy()
        curw = float((others @ cur).max())
        tr = rng.standard_normal(size=(400, d)) * 0.04 + cur[None, :]
        tr /= np.linalg.norm(tr, axis=1, keepdims=True)
        dw = (tr @ others.T).max(axis=1)
        bi = int(np.argmin(dw))
        if dw[bi] < curw:
            y[k] = tr[bi]
    return y

def solve():
    t0 = time.time()
    HARD = t0 + 165.0
    EXPLORE_END = t0 + 148.0
    rng = np.random.default_rng(7)
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
    xb = x0.copy(); best = float(b0)
    n, d = xb.shape
    it = 0; stall = 0
    while time.time() < EXPLORE_END and time.time() < HARD - 10:
        x = xb.copy()
        mode = it % 5
        try:
            if mode == 0:
                sig = 0.02 * (1.0 + 0.06 * min(stall, 10))
                x = x + rng.standard_normal(size=x.shape) * sig
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                dl = min(time.time() + 0.55, HARD - 6)
                x = _anneal(x, steps=420, tau0=0.014, tau1=0.0025, lr0=3.5, lr1=0.5, mu=0.8, deadline=dl)
                lp_dead = min(time.time() + 1.8, EXPLORE_END, HARD - 6)
                x = _lp_polish(x, deadline=lp_dead, init_radius=0.005)
            elif mode == 1:
                w = _worst(x)
                k = 28
                idx = np.argpartition(w, -k)[-k:]
                x[idx] += rng.standard_normal(size=(k, d)) * 0.065
                x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
                dl = min(time.time() + 0.5, HARD - 6)
                x = _anneal(x, steps=380, tau0=0.011, tau1=0.0022, lr0=3.0, lr1=0.45, mu=0.75, deadline=dl)
                lp_dead = min(time.time() + 1.8, EXPLORE_END, HARD - 6)
                x = _lp_polish(x, deadline=lp_dead, init_radius=0.006)
            elif mode == 2:
                m = best
                cc = _ccounts(x, m, tol=1e-9)
                ratt = np.where(cc == 0)[0].tolist()
                order = np.argsort(cc)  # ascending: rattlers/low first
                torep = []
                for o in order:
                    if len(torep) >= 5:
                        break
                    torep.append(int(o))
                # also include a hub to break overload
                hub = int(np.argmax(cc))
                if hub not in torep and len(torep) >= 1:
                    torep[-1] = hub
                x = _reinsert(x, torep, rng, n_cand=8000)
                x += rng.standard_normal(size=x.shape) * 0.003
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                dl = min(time.time() + 0.5, HARD - 6)
                x = _anneal(x, steps=380, tau0=0.009, tau1=0.0022, lr0=2.8, lr1=0.45, mu=0.8, deadline=dl)
                lp_dead = min(time.time() + 2.0, EXPLORE_END, HARD - 6)
                x = _lp_polish(x, deadline=lp_dead, init_radius=0.006)
            elif mode == 3:
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                iuI, iuJ = np.triu_indices(n, 1)
                c = G[iuI, iuJ]
                top = np.argpartition(c, -10)[-10:]
                for t in top:
                    i = int(iuI[t]); j = int(iuJ[t])
                    di = x[i] - x[j]
                    di = di - np.dot(di, x[i]) * x[i]
                    dj = -di
                    # for j use its own tangent (approx)
                    dj2 = x[j] - x[i]
                    dj2 = dj2 - np.dot(dj2, x[j]) * x[j]
                    ni = np.linalg.norm(di); nj = np.linalg.norm(dj2)
                    if ni > 1e-12:
                        x[i] = x[i] + 0.055 * di / ni
                    if nj > 1e-12:
                        x[j] = x[j] + 0.055 * dj2 / nj
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                dl = min(time.time() + 0.5, HARD - 6)
                x = _anneal(x, steps=360, tau0=0.008, tau1=0.002, lr0=2.5, lr1=0.4, mu=0.75, deadline=dl)
                lp_dead = min(time.time() + 1.8, EXPLORE_END, HARD - 6)
                x = _lp_polish(x, deadline=lp_dead, init_radius=0.005)
            else:
                w = _worst(x)
                k = 7
                idx = np.argpartition(w, -k)[-k:].tolist()
                x = _reinsert(x, [int(v) for v in idx], rng, n_cand=7000)
                x += rng.standard_normal(size=x.shape) * 0.004
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                dl = min(time.time() + 0.45, HARD - 6)
                x = _anneal(x, steps=320, tau0=0.009, tau1=0.0022, lr0=2.8, lr1=0.45, mu=0.8, deadline=dl)
                lp_dead = min(time.time() + 1.9, EXPLORE_END, HARD - 6)
                x = _lp_polish(x, deadline=lp_dead, init_radius=0.005)
        except Exception:
            it += 1
            continue
        try:
            v = _max_cos(x)
        except Exception:
            it += 1
            continue
        if np.all(np.isfinite(x)) and v < best - 1e-12:
            best = v; xb = x.copy(); stall = 0
        else:
            stall += 1
        it += 1
        if time.time() >= HARD - 9:
            break
    if time.time() < HARD - 2:
        try:
            dl = min(HARD - 1.0, time.time() + max(2.0, HARD - 1.0 - time.time()))
            xb = _lp_polish(xb, deadline=dl, init_radius=1.5e-3, max_it=30)
        except Exception:
            pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    xb[~np.isfinite(xb)] = 0.0
    nn = np.linalg.norm(xb, axis=1, keepdims=True)
    for i in range(n):
        if not np.isfinite(nn[i, 0]) or nn[i, 0] < 1e-12:
            v = rng.standard_normal(size=(d,))
            v /= np.linalg.norm(v)
            xb[i] = v
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
