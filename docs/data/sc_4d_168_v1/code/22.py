import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _lp_polish_deadline(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        iu_c = G[iu]
        m = float(np.max(iu_c))
        sel = iu_c > m - max(20 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        csel = iu_c[sel]
        p = len(I)
        if p == 0:
            break
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
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    SEARCH_END = t0 + 138.0
    rng = np.random.RandomState(20261031)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(168, 4)
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr)
        xb = _norm(xb)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        xb = _norm(xb)
    n, d = 168, 4
    if xb.shape != (168, 4):
        xb = _norm(rng.randn(168, 4))
    xb = _norm(xb)
    best_val = _max_cos(xb)
    x_best = xb.copy()
    x_cur = xb.copy()
    cur_val = best_val

    def refine(x_init, steps, beta0=30.0, beta1=600.0, lr0=0.016, lr1=0.0008, mu=0.88):
        x = x_init.copy()
        vel = np.zeros_like(x)
        lb0 = np.log(beta0); lb1 = np.log(beta1)
        for s in range(steps):
            f = s / max(1, steps - 1)
            beta = float(np.exp(lb0 + (lb1 - lb0) * f))
            lr = lr0 * (1.0 - f) + lr1 * f
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
            np.fill_diagonal(E, 0.0)
            rs = np.sum(E, axis=1, keepdims=True) + 1e-12
            grad = (E @ x) / rs
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            # normalize scale for stability
            sc = np.mean(np.linalg.norm(gt, axis=1)) + 1e-12
            gt = gt / sc
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x = _norm(x)
            if time.time() > SEARCH_END:
                break
        return x

    def refine_single(x0, Xref, steps=100, beta0=40.0, beta1=250.0, lr0=0.09, lr1=0.006):
        s = x0 / (np.linalg.norm(x0) + 1e-30)
        lb0 = np.log(beta0); lb1 = np.log(beta1)
        for it in range(steps):
            f = it / max(1, steps - 1)
            beta = float(np.exp(lb0 + (lb1 - lb0) * f))
            lr = lr0 * (1.0 - f) + lr1 * f
            dots = Xref @ s
            cm = float(np.max(dots))
            w = np.exp(np.clip(beta * (dots - cm), -25.0, 0.0))
            w /= (np.sum(w) + 1e-30)
            g = w @ Xref
            gt = g - np.dot(g, s) * s
            gn = np.linalg.norm(gt) + 1e-12
            gt = gt / gn
            s = s - lr * gt
            ns = np.linalg.norm(s)
            if ns < 1e-12:
                s = rng.randn(4); ns = np.linalg.norm(s)
            s = s / ns
        return s

    def best_void(Xref, n_sample=10000):
        C = rng.randn(n_sample, 4)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
        D = C @ Xref.T
        m = D.max(axis=1)
        return C[int(np.argmin(m))]

    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 1.0:
            break
        # choose base: mostly best, sometimes cur to walk
        if trial % 5 == 0 and cur_val < best_val + 0.004:
            base = x_cur
        else:
            base = x_best
        typ = trial % 10
        x = base.copy()
        steps = 500
        if typ == 0:
            x = x + 0.012 * rng.randn(n, d)
            x = _norm(x)
            steps = 450
            x = refine(x, steps, 25.0, 500.0, 0.014, 0.0007)
        elif typ == 1:
            x = x + 0.045 * rng.randn(n, d)
            x = _norm(x)
            steps = 600
            x = refine(x, steps, 30.0, 700.0, 0.016, 0.0007)
        elif typ == 2:
            x = x + 0.12 * rng.randn(n, d)
            x = _norm(x)
            steps = 700
            x = refine(x, steps, 20.0, 500.0, 0.018, 0.0008)
        elif typ in (3, 4, 5):
            # contact-guided surgery: move overloaded points to voids
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = np.sum(G > m - 0.004, axis=1)
            k_move = int(rng.choice([2, 3, 4, 5, 6]))
            if typ == 5:
                k_move = int(rng.choice([1, 2, 3]))
            order = np.argsort(-deg, kind="stable")
            victims = order[:k_move].copy()
            # occasionally displace a rattler too
            if rng.rand() < 0.3:
                contacts = np.sum(G > m - 1e-7, axis=1)
                ratt = np.where(contacts == 0)[0]
                if len(ratt) > 0:
                    victims[-1] = rng.choice(ratt)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            X_keep = x[keep_mask]
            new_pts = []
            for s_i in range(k_move):
                Xref = X_keep if len(new_pts) == 0 else np.vstack([X_keep, np.array(new_pts)])
                c0 = best_void(Xref, n_sample=10000)
                xn = refine_single(c0, Xref, steps=110, beta0=45.0, beta1=280.0)
                new_pts.append(xn)
            x[victims] = np.array(new_pts)
            x = x + 0.006 * rng.randn(n, d)
            x = _norm(x)
            steps = 550 if typ != 5 else 450
            b1 = 800.0 if typ == 4 else 600.0
            x = refine(x, steps, 30.0, b1, 0.015, 0.0007)
        elif typ == 6:
            # cluster kick: pick random center + nearest neighbours
            c = rng.randint(n)
            G = x @ x.T
            nbrs = np.argsort(-G[c])[:int(rng.choice([4, 6, 8]))]
            x[nbrs] = x[nbrs] + 0.15 * rng.randn(len(nbrs), d)
            x = _norm(x)
            x = refine(x, 600, 25.0, 600.0, 0.016, 0.0007)
        elif typ == 7:
            # sharp high-beta quench from small shake
            x = x + 0.02 * rng.randn(n, d)
            x = _norm(x)
            x = refine(x, 500, 80.0, 1500.0, 0.010, 0.0005, mu=0.85)
        elif typ == 8:
            # random void re-insert of 1-2 random victims
            k_move = int(rng.choice([1, 2, 3]))
            victims = rng.choice(n, size=k_move, replace=False)
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            X_keep = x[keep_mask]
            new_pts = []
            for s_i in range(k_move):
                Xref = X_keep if len(new_pts) == 0 else np.vstack([X_keep, np.array(new_pts)])
                c0 = best_void(Xref, n_sample=12000)
                xn = refine_single(c0, Xref, steps=130, beta0=35.0, beta1=300.0)
                new_pts.append(xn)
            x[victims] = np.array(new_pts)
            x = _norm(x + 0.004 * rng.randn(n, d))
            x = refine(x, 550, 30.0, 700.0, 0.014, 0.0006)
        else:
            # far jump then smooth anneal
            x = x + 0.25 * rng.randn(n, d)
            x = _norm(x)
            x = refine(x, 750, 15.0, 400.0, 0.020, 0.0009)
        v = _max_cos(x)
        # update walk state
        if v < cur_val:
            x_cur, cur_val = x.copy(), v
        elif v < cur_val + 0.003 and rng.rand() < 0.15:
            x_cur, cur_val = x.copy(), v
        if trial % 25 == 0 and cur_val > best_val + 0.005:
            x_cur, cur_val = x_best.copy(), best_val
        # opportunistic LP squeeze if promising
        if v < best_val + 0.0008 and time.time() < SEARCH_END - 3.0:
            # only squeeze if actually better or very close to best
            if v < best_val or (v < best_val + 0.00015 and (trial % 3 == 0)):
                dl = min(time.time() + 1.4, SEARCH_END)
                xs, vs = _lp_polish_deadline(x, dl)
                if vs < best_val:
                    x_best, best_val = xs.copy(), vs
                    x_cur, cur_val = xs.copy(), vs
                elif vs < cur_val:
                    x_cur, cur_val = xs.copy(), vs
        else:
            if v < best_val:
                # GD alone beat best (LP would only help); record it
                x_best, best_val = x.copy(), v
                x_cur, cur_val = x.copy(), v
    # final squeeze on best
    try:
        x_best, best_val = _lp_polish_deadline(x_best, HARD_END - 1.0)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
