import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _worst_per_point(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _contact_counts(x, m, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > m - tol, axis=1)

def _lp_polish(x, deadline, init_radius=5e-3):
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
        if cnt > 2500:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -2500)[-2500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 2500
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
        Cm[:, d:2*d] = J[:, None] * d + ar
        Cm[:, 2*d] = n * d
        Vm[:, :d] = x[J]
        Vm[:, d:2*d] = x[I]
        Vm[:, 2*d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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

def _smooth_descent(x, steps, tau0, tau1, lr0, lr1, mu, deadline):
    n, d = x.shape
    iuI, iuJ = np.triu_indices(n, 1)
    vel = np.zeros_like(x)
    best = _max_cos(x)
    best_x = x.copy()
    for s in range(steps):
        if time.time() >= deadline:
            break
        frac = s / max(1, steps - 1)
        tau = tau0 + (tau1 - tau0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        # normalized softmax weights
        e = np.exp((c - m) / tau)
        S = float(e.sum())
        if S == 0 or not np.isfinite(S):
            break
        w = e / S
        # build symmetric weight matrix action without full matrix
        # grad = W @ x where W has w on (I,J) and (J,I)
        Wx = np.zeros_like(x)
        # accumulate: for efficiency use bincount-like via np.add.at
        # w* x[J] contributes to grad[I], w*x[I] to grad[J]
        # use vectorized scatter
        np.add.at(Wx, iuI, w[:, None] * x[iuJ])
        np.add.at(Wx, iuJ, w[:, None] * x[iuI])
        # tangent projection
        dot = np.sum(Wx * x, axis=1, keepdims=True)
        Gt = Wx - dot * x
        # clip extreme
        gn = np.linalg.norm(Gt, axis=1, keepdims=True)
        # optional clip not needed
        vel = mu * vel + Gt
        x = x - lr * vel
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        xn[xn == 0] = 1.0
        x = x / xn
        # track best every 50 steps to save matmul
        if (s % 50 == 49) or (s == steps - 1):
            v = _max_cos(x)
            if v < best:
                best = v
                best_x = x.copy()
    return best_x

def _greedy_reinsert(y, to_replace, rng, n_cand=16000):
    n, d = y.shape
    fixed_mask = np.ones(n, dtype=bool)
    fixed_mask[to_replace] = False
    X_fixed = y[fixed_mask]
    for k in to_replace:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        if len(to_replace) > 1:
            # include already reinserted points: rebuild fixed set
            # cheap: use current y without k
            others = np.delete(y, k, axis=0)
            dots = cand @ others.T
        else:
            dots = cand @ X_fixed.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        y[k] = cand[bi]
    # tiny local hill climb on replaced
    for k in to_replace:
        others = np.delete(y, k, axis=0)
        cur = y[k].copy()
        cur_w = float((others @ cur).max())
        trial = rng.standard_normal(size=(600, d)) * 0.04 + cur[None, :]
        trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        dw = (trial @ others.T).max(axis=1)
        bi = int(np.argmin(dw))
        if dw[bi] < cur_w:
            y[k] = trial[bi]
    return y

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    EXPLORE_END = t0 + 142.0
    rng = np.random.default_rng(12345)
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
    while time.time() < EXPLORE_END and time.time() < HARD - 8:
        s = it % 6
        x = x_best.copy()
        if s == 0:
            sig = 0.018 * (1.0 + 0.08 * min(no_improve, 10))
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            dl = min(time.time() + 0.6, HARD - 5)
            x = _smooth_descent(x, steps=700, tau0=0.015, tau1=0.003, lr0=4.0, lr1=0.6, mu=0.8, deadline=dl)
            lp_dead = min(time.time() + 2.2, EXPLORE_END, HARD - 5)
            x = _lp_polish(x, deadline=lp_dead, init_radius=0.006)
        elif s == 1:
            w = _worst_per_point(x)
            k = 32
            idx = np.argpartition(w, -k)[-k:]
            x[idx] += rng.standard_normal(size=(k, d)) * 0.07
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            dl = min(time.time() + 0.5, HARD - 5)
            x = _smooth_descent(x, steps=500, tau0=0.012, tau1=0.0025, lr0=3.0, lr1=0.5, mu=0.75, deadline=dl)
            lp_dead = min(time.time() + 2.2, EXPLORE_END, HARD - 5)
            x = _lp_polish(x, deadline=lp_dead, init_radius=0.007)
        elif s == 2:
            m = best
            cc = _contact_counts(x, m, tol=1e-9)
            ratt = np.where(cc == 0)[0].tolist()
            order = np.argsort(-cc)
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
                torep = np.argpartition(_worst_per_point(x), -6)[-6:].tolist()
            x = _greedy_reinsert(x, [int(v) for v in torep], rng, n_cand=16000)
            x += rng.standard_normal(size=x.shape) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            dl = min(time.time() + 0.5, HARD - 5)
            x = _smooth_descent(x, steps=500, tau0=0.010, tau1=0.0025, lr0=3.0, lr1=0.5, mu=0.8, deadline=dl)
            lp_dead = min(time.time() + 2.6, EXPLORE_END, HARD - 5)
            x = _lp_polish(x, deadline=lp_dead, init_radius=0.007)
        elif s == 3:
            k = 24
            idx = rng.choice(n, size=k, replace=False)
            x[idx] += rng.standard_normal(size=(k, d)) * 0.055
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            dl = min(time.time() + 0.5, HARD - 5)
            x = _smooth_descent(x, steps=600, tau0=0.014, tau1=0.003, lr0=3.5, lr1=0.5, mu=0.8, deadline=dl)
            lp_dead = min(time.time() + 2.2, EXPLORE_END, HARD - 5)
            x = _lp_polish(x, deadline=lp_dead, init_radius=0.006)
        elif s == 4:
            # break worst pairs apart explicitly
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            iuI, iuJ = np.triu_indices(n, 1)
            c = G[iuI, iuJ]
            top = np.argpartition(c, -12)[-12:]
            pushed = set()
            for t in top:
                i = int(iuI[t]); j = int(iuJ[t])
                pushed.add(i); pushed.add(j)
                # push i and j apart along tangent
                di = x[i] - x[j]
                dj = -di
                di = di - np.dot(di, x[i]) * x[i]
                dj = dj - np.dot(dj, x[j]) * x[j]
                ni = np.linalg.norm(di); nj = np.linalg.norm(dj)
                if ni > 1e-12:
                    x[i] = x[i] + 0.06 * di / ni
                if nj > 1e-12:
                    x[j] = x[j] + 0.06 * dj / nj
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            dl = min(time.time() + 0.5, HARD - 5)
            x = _smooth_descent(x, steps=500, tau0=0.008, tau1=0.002, lr0=2.5, lr1=0.4, mu=0.75, deadline=dl)
            lp_dead = min(time.time() + 2.2, EXPLORE_END, HARD - 5)
            x = _lp_polish(x, deadline=lp_dead, init_radius=0.005)
        else:
            w = _worst_per_point(x)
            k = 8
            idx = np.argpartition(w, -k)[-k:].tolist()
            x = _greedy_reinsert(x, [int(v) for v in idx], rng, n_cand=12000)
            x += rng.standard_normal(size=x.shape) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            dl = min(time.time() + 0.4, HARD - 5)
            x = _smooth_descent(x, steps=400, tau0=0.010, tau1=0.0025, lr0=3.0, lr1=0.5, mu=0.8, deadline=dl)
            lp_dead = min(time.time() + 2.4, EXPLORE_END, HARD - 5)
            x = _lp_polish(x, deadline=lp_dead, init_radius=0.006)
        v = _max_cos(x)
        if v < best - 1e-12 and np.all(np.isfinite(x)):
            best = v
            x_best = x
            no_improve = 0
        else:
            no_improve += 1
        it += 1
        if time.time() >= HARD - 8:
            break
    # final squeeze
    if time.time() < HARD - 2:
        try:
            dl = min(HARD - 1.0, time.time() + max(2.0, HARD - 1.0 - time.time()))
            # use small radius for tight convergence
            x_best = _lp_polish(x_best, deadline=dl, init_radius=2e-3)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # sanitize
    x_best[~np.isfinite(x_best)] = 0.0
    nn = np.linalg.norm(x_best, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    # replace any zero rows with random
    for i in range(n):
        if not np.isfinite(nn[i, 0]) or nn[i, 0] < 1e-12:
            v = rng.standard_normal(size=(d,))
            v /= np.linalg.norm(v)
            x_best[i] = v
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
