import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_IU_I = None
_IU_J = None

def _get_iu(n):
    global _IU_I, _IU_J
    if _IU_I is None or len(_IU_I) != n * (n - 1) // 2:
        _IU_I, _IU_J = np.triu_indices(n, 1)
    return _IU_I, _IU_J

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _worst_per_point(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _contact_counts(x, m, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > m - tol, axis=1)

def _lp_polish(x, deadline, init_radius=5e-3, cap=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    iuI, iuJ = _get_iu(n)
    best = _max_cos(x)
    radius = float(init_radius)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < 60:
        it += 1
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(25.0 * radius, 2e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -cap)[-cap:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = cap
        I = iuI[sel]
        J = iuJ[sel]
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
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
            x = y
            best = v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
        if time.time() >= deadline:
            break
    return x

def _fast_anneal(x, steps, tau0, tau1, lr0, lr1, mu, w0, w1, deadline, refresh=20, cap=5000):
    n, d = x.shape
    iuI, iuJ = _get_iu(n)
    vel = np.zeros_like(x)
    best = _max_cos(x)
    best_x = x.copy()
    I = None
    J = None
    w = None
    for s in range(steps):
        if s % refresh == 0:
            if time.time() >= deadline:
                break
            frac = s / max(1, steps - 1)
            tau = tau0 + (tau1 - tau0) * frac
            width = w0 + (w1 - w0) * frac
            G = x @ x.T
            c = G[iuI, iuJ]
            m = float(c.max())
            thresh = m - width
            sel = c > thresh
            cnt = int(sel.sum())
            if cnt == 0:
                # fall back to top 500
                k = 500
                idx = np.argpartition(c, -k)[-k:]
                I = iuI[idx]; J = iuJ[idx]
                cs = c[idx]
                e = np.exp((cs - m) / max(tau, 1e-12))
                S = float(e.sum())
                if S == 0 or not np.isfinite(S):
                    w = np.full(k, 1.0 / k)
                else:
                    w = e / S
            else:
                if cnt > cap:
                    idx = np.where(sel)[0]
                    cidx = c[idx]
                    part = np.argpartition(cidx, -cap)[-cap:]
                    sidx = idx[part]
                    I = iuI[sidx]; J = iuJ[sidx]
                    cs = c[sidx]
                else:
                    I = iuI[sel]; J = iuJ[sel]
                    cs = c[sel]
                e = np.exp((cs - m) / max(tau, 1e-12))
                S = float(e.sum())
                if S == 0 or not np.isfinite(S):
                    w = np.full(len(cs), 1.0 / max(1, len(cs)))
                else:
                    w = e / S
            if s % 100 == 0 and s > 0:
                v = _max_cos(x)
                if v < best:
                    best = v
                    best_x = x.copy()
        frac = s / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * frac
        Wx = np.zeros_like(x)
        np.add.at(Wx, I, w[:, None] * x[J])
        np.add.at(Wx, J, w[:, None] * x[I])
        dot = np.sum(Wx * x, axis=1, keepdims=True)
        Gt = Wx - dot * x
        vel = mu * vel + Gt
        x = x - lr * vel
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
        if (s % 100 == 99):
            if time.time() >= deadline:
                break
            v = _max_cos(x)
            if v < best:
                best = v
                best_x = x.copy()
    v = _max_cos(x)
    if v < best:
        return x
    return best_x

def _greedy_reinsert(y, to_replace, rng, n_cand=20000):
    n, d = y.shape
    for k in to_replace:
        k = int(k)
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(y, k, axis=0)
        dots = cand @ others.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        y[k] = cand[bi]
    for k in to_replace:
        k = int(k)
        others = np.delete(y, k, axis=0)
        cur = y[k].copy()
        cur_w = float((others @ cur).max())
        trial = rng.standard_normal(size=(500, d)) * 0.04 + cur[None, :]
        trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        dw = (trial @ others.T).max(axis=1)
        bi = int(np.argmin(dw))
        if dw[bi] < cur_w:
            y[k] = trial[bi]
    return y

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    EXPLORE_END = t0 + 132.0
    rng = np.random.default_rng(7)
    x0 = None
    b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < b0:
                        b0 = v
                        x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x0 = a
        b0 = _max_cos(a)
    x_best = x0.copy()
    best = float(b0)
    n, d = x_best.shape
    it = 0
    no_improve = 0
    while time.time() < EXPLORE_END and time.time() < HARD - 10:
        mode = it % 4
        x = x_best.copy()
        if mode == 0:
            sig = 0.016 * (1.0 + 0.07 * min(no_improve, 10))
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            dl = min(time.time() + 0.55, HARD - 6)
            x = _fast_anneal(x, steps=600, tau0=0.014, tau1=0.0025, lr0=4.0, lr1=0.6, mu=0.8, w0=0.03, w1=0.008, deadline=dl)
            lp_dead = min(time.time() + 2.0, EXPLORE_END, HARD - 6)
            x = _lp_polish(x, deadline=lp_dead, init_radius=0.006)
        elif mode == 1:
            m = best
            cc = _contact_counts(x, m, tol=2e-7)
            hubs = np.argsort(-cc)[:3].tolist()
            # neighbors of hubs with high dot
            G = x @ x.T
            cluster = set(int(h) for h in hubs)
            for h in hubs:
                h = int(h)
                row = G[h]
                nbrs = np.where(row > m - 5e-3)[0]
                for nb in nbrs[:12]:
                    cluster.add(int(nb))
            cluster = list(cluster)[:28]
            amp = 0.055
            x[cluster] += rng.standard_normal(size=(len(cluster), d)) * amp
            x[cluster] /= np.linalg.norm(x[cluster], axis=1, keepdims=True)
            dl = min(time.time() + 0.55, HARD - 6)
            x = _fast_anneal(x, steps=550, tau0=0.012, tau1=0.0022, lr0=3.2, lr1=0.5, mu=0.78, w0=0.025, w1=0.007, deadline=dl)
            lp_dead = min(time.time() + 2.0, EXPLORE_END, HARD - 6)
            x = _lp_polish(x, deadline=lp_dead, init_radius=0.006)
        elif mode == 2:
            m = best
            cc = _contact_counts(x, m, tol=1e-7)
            ratt = np.where(cc == 0)[0].tolist()
            wpp = _worst_per_point(x)
            order = np.argsort(-wpp).tolist()
            torep = []
            for r in ratt:
                torep.append(int(r))
            for o in order:
                if len(torep) >= 6:
                    break
                if int(o) not in torep:
                    torep.append(int(o))
            torep = torep[:6]
            if len(torep) == 0:
                torep = np.argpartition(wpp, -6)[-6:].tolist()
            x = _greedy_reinsert(x, [int(v) for v in torep], rng, n_cand=20000)
            x += rng.standard_normal(size=x.shape) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            dl = min(time.time() + 0.5, HARD - 6)
            x = _fast_anneal(x, steps=500, tau0=0.010, tau1=0.0022, lr0=3.0, lr1=0.5, mu=0.8, w0=0.022, w1=0.007, deadline=dl)
            lp_dead = min(time.time() + 2.2, EXPLORE_END, HARD - 6)
            x = _lp_polish(x, deadline=lp_dead, init_radius=0.007)
        else:
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            iuI, iuJ = _get_iu(n)
            c = G[iuI, iuJ]
            top = np.argpartition(c, -10)[-10:]
            for t in top:
                i = int(iuI[t]); j = int(iuJ[t])
                di = x[i] - x[j]
                di = di - np.dot(di, x[i]) * x[i]
                dj = -di - np.dot(-di, x[j]) * x[j] if False else (-(x[i] - x[j]))
                # push apart
                ni = np.linalg.norm(di)
                if ni > 1e-12:
                    x[i] = x[i] + 0.05 * di / ni
                dj2 = x[j] - x[i]
                dj2 = dj2 - np.dot(dj2, x[j]) * x[j]
                nj = np.linalg.norm(dj2)
                if nj > 1e-12:
                    x[j] = x[j] + 0.05 * dj2 / nj
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            dl = min(time.time() + 0.5, HARD - 6)
            x = _fast_anneal(x, steps=500, tau0=0.008, tau1=0.002, lr0=2.6, lr1=0.4, mu=0.75, w0=0.020, w1=0.006, deadline=dl)
            lp_dead = min(time.time() + 2.0, EXPLORE_END, HARD - 6)
            x = _lp_polish(x, deadline=lp_dead, init_radius=0.005)
        try:
            v = _max_cos(x)
        except Exception:
            v = np.inf
        if np.all(np.isfinite(x)) and v < best - 1e-12:
            best = v
            x_best = x
            no_improve = 0
        else:
            no_improve += 1
        it += 1
        if time.time() >= HARD - 10:
            break
    if time.time() < HARD - 2:
        try:
            dl = min(HARD - 1.0, time.time() + max(3.0, HARD - 1.0 - time.time()))
            x_best = _lp_polish(x_best, deadline=dl, init_radius=2e-3, cap=2500)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best[~np.isfinite(x_best)] = 0.0
    nn = np.linalg.norm(x_best, axis=1, keepdims=True)
    for i in range(n):
        if not np.isfinite(nn[i, 0]) or nn[i, 0] < 1e-12:
            vv = rng.standard_normal(size=(d,))
            vv /= np.linalg.norm(vv)
            x_best[i] = vv
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
