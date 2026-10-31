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
    SEARCH_END = t0 + 152.0
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
    if xb.shape != (168, 4):
        xb = _norm(rng.randn(168, 4))
    xb = _norm(xb)
    best_val = _max_cos(xb)
    x_best = xb.copy()

    def refine(x_init, steps, beta0=30.0, beta1=600.0, lr0=0.016, lr1=0.0008, mu=0.88):
        x = x_init.copy()
        vel = np.zeros_like(x)
        lb0 = np.log(beta0); lb1 = np.log(beta1)
        for s in range(steps):
            if time.time() > SEARCH_END:
                break
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
            sc = np.mean(np.linalg.norm(gt, axis=1)) + 1e-12
            gt = gt / sc
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x = _norm(x)
        return x

    def refine_single(x0, Xref, steps=90, beta0=40.0, beta1=250.0, lr0=0.09, lr1=0.006):
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

    def best_void_raw(Xref, n_sample=8000):
        C = rng.randn(n_sample, 4)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
        D = C @ Xref.T
        m = D.max(axis=1)
        return C[int(np.argmin(m))]

    def contact_load(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        cnt = np.sum(G > m - 1e-3, axis=1)
        exc = np.sum(np.clip(G - 0.80, 0.0, None), axis=1)
        load = cnt.astype(float) + exc
        return load, m

    def insert_hole(Xcur, n_sample=8000, s_steps=90):
        c0 = best_void_raw(Xcur, n_sample=n_sample)
        co = refine_single(c0, Xcur, steps=s_steps)
        return np.vstack([Xcur, co[None, :]])

    cycle = 0
    while time.time() < SEARCH_END:
        cycle += 1
        if time.time() > SEARCH_END - 2.0:
            break
        base = x_best.copy()
        if cycle % 2 == 1:
            # OVERSHOOT branch: N=169..171 then delete
            k_add = int(rng.choice([1, 2, 3, 1, 2, 3, 2]))
            if time.time() > SEARCH_END - 8.0:
                k_add = 1
            Xext = base.copy()
            for _ in range(k_add):
                Xext = insert_hole(Xext, n_sample=9000, s_steps=100)
            # refine enlarged hard (worse max is expected)
            steps_ext = 650 if k_add <= 2 else 750
            # vary annealing per cycle
            if cycle % 4 == 1:
                Xext = refine(Xext, steps_ext, 25.0, 550.0, 0.015, 0.0007, 0.88)
            else:
                Xext = refine(Xext, steps_ext, 30.0, 800.0, 0.016, 0.0007, 0.88)
            load, _ = contact_load(Xext)
            order = np.argsort(-load, kind="stable")
            victims = order[:k_add]
            keep = np.ones(Xext.shape[0], dtype=bool)
            keep[victims] = False
            X168 = _norm(Xext[keep])
            # refine resulting 168 with LSE
            X168 = refine(X168, 450, 30.0, 700.0, 0.014, 0.0006, 0.88)
            v = _max_cos(X168)
            if v < best_val + 0.0008 and time.time() < SEARCH_END - 2.5:
                do_lp = False
                if v < best_val:
                    do_lp = True
                elif v < best_val + 0.00012 and (cycle % 2 == 1):
                    do_lp = True
                if do_lp:
                    dl = min(time.time() + 1.3, SEARCH_END)
                    try:
                        Xs, vs = _lp_polish_deadline(X168, dl)
                    except Exception:
                        Xs, vs = X168, v
                    if vs < best_val:
                        x_best, best_val = Xs.copy(), vs
                    elif vs < v:
                        # keep descendant for pressure even if not best? only best seeds per spec
                        if v < best_val:
                            x_best, best_val = Xs.copy(), vs
                else:
                    if v < best_val:
                        x_best, best_val = X168.copy(), v
            else:
                if v < best_val:
                    x_best, best_val = X168.copy(), v
        else:
            # UNDERSHOOT branch: delete 8-18, polish small, regrow
            k_del = int(rng.choice([8, 10, 12, 14, 16, 18, 9, 11, 13]))
            if time.time() > SEARCH_END - 10.0:
                k_del = 8
            load, _ = contact_load(base)
            order = np.argsort(-load, kind="stable")
            # add slight randomness: swap a couple victims with random high-load
            victims = order[:k_del].copy()
            if rng.rand() < 0.4:
                # replace last victim by random among top 30
                cand = order[:30]
                victims[-1] = cand[rng.randint(len(cand))]
            keep = np.ones(168, dtype=bool)
            keep[victims] = False
            Xsmall = base[keep].copy()
            # polish small to its own optimum
            Xsmall = refine(Xsmall, 500, 25.0, 600.0, 0.016, 0.0007, 0.88)
            # optionally extra quench
            if rng.rand() < 0.3:
                Xsmall = refine(Xsmall, 150, 80.0, 1200.0, 0.009, 0.0005, 0.85)
            batch = int(rng.choice([2, 3, 4, 2, 3]))
            Xcur = Xsmall.copy()
            n_need = k_del
            for ii in range(n_need):
                # deeper samples for later holes
                ns = 9000 if (ii % batch == 0) else 6000
                Xcur = insert_hole(Xcur, n_sample=ns, s_steps=90)
                end_batch = ((ii + 1) % batch == 0) or (ii == n_need - 1)
                if end_batch:
                    # short joint refinement
                    Xcur = refine(Xcur, 160, 30.0, 350.0, 0.013, 0.0010, 0.88)
            # longer final joint refinement
            Xcur = refine(Xcur, 450, 30.0, 700.0, 0.014, 0.0006, 0.88)
            v = _max_cos(Xcur)
            if v < best_val + 0.0008 and time.time() < SEARCH_END - 2.5:
                do_lp = False
                if v < best_val:
                    do_lp = True
                elif v < best_val + 0.00012 and (cycle % 3 == 0):
                    do_lp = True
                if do_lp:
                    dl = min(time.time() + 1.3, SEARCH_END)
                    try:
                        Xs, vs = _lp_polish_deadline(Xcur, dl)
                    except Exception:
                        Xs, vs = Xcur, v
                    if vs < best_val:
                        x_best, best_val = Xs.copy(), vs
                else:
                    if v < best_val:
                        x_best, best_val = Xcur.copy(), v
            else:
                if v < best_val:
                    x_best, best_val = Xcur.copy(), v
    try:
        x_best, best_val = _lp_polish_deadline(x_best, HARD_END - 1.0)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
