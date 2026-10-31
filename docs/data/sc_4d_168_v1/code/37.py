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

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        C = (x @ x.T)[iu]
        m = float(np.max(C))
        sel = C > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
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
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-C[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    SEARCH_END = t0 + 141.0
    rng = np.random.RandomState(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and getattr(c, "shape", None) == (168, 4):
            c = np.asarray(c, dtype=float)
            if np.all(np.isfinite(c)):
                c = _norm(c)
                cands.append(c)
    if len(cands) == 0:
        x_best = _norm(rng.randn(168, 4))
    else:
        vals = [_max_cos(c) for c in cands]
        x_best = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(x_best)

    def refine_full(x_init, steps, mu=0.9, beta0=25.0, beta1=450.0, lr0=0.012, lr1=0.0007):
        x = x_init.copy()
        vel = np.zeros_like(x)
        for s in range(steps):
            if s % 150 == 0 and s > 0 and time.time() >= SEARCH_END:
                break
            frac = s / max(1, steps - 1)
            beta = beta0 * (beta1 / beta0) ** frac
            lr = lr0 * (1.0 - frac) + lr1 * frac
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -20.0, 0.0))
            np.fill_diagonal(E, 0.0)
            grad = E @ x
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x = _norm(x)
        return x

    def batch_refine_cands(cand, Xcur, iters=60, b0=40.0, b1=260.0, l0=0.09, l1=0.006):
        c = cand.copy()
        c = _norm(c)
        lb0 = np.log(b0); lb1 = np.log(b1)
        for it in range(iters):
            f = it / max(1, iters - 1)
            beta = float(np.exp(lb0 + (lb1 - lb0) * f))
            lr = l0 * (1.0 - f) + l1 * f
            D = c @ Xcur.T
            cm = D.max(axis=1, keepdims=True)
            W = np.exp(np.clip(beta * (D - cm), -20.0, 0.0))
            W /= (W.sum(axis=1, keepdims=True) + 1e-30)
            grad = W @ Xcur
            dot = np.sum(grad * c, axis=1, keepdims=True)
            gt = grad - dot * c
            c = c - lr * gt
            c = _norm(c)
        return c

    def deepest_hole(Xcur, ns=8000, top=6, iters=60):
        S = rng.randn(ns, 4)
        S = _norm(S)
        D = S @ Xcur.T
        worst = D.max(axis=1)
        idx = np.argpartition(worst, top)[:top]
        cand = S[idx]
        cand2 = batch_refine_cands(cand, Xcur, iters=iters)
        Df = cand2 @ Xcur.T
        wf = Df.max(axis=1)
        return cand2[int(np.argmin(wf))]

    def contact_loads(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        deg = np.sum(G > cmax - 1e-3, axis=1).astype(float)
        exc = np.sum(np.maximum(G - 0.80, 0.0), axis=1)
        return deg + exc, cmax, G

    cycle = 0
    # precompute alternating schedule
    while time.time() < SEARCH_END:
        cycle += 1
        if time.time() > SEARCH_END - 2.0:
            break
        if cycle % 2 == 1:
            # OVERSHOOT branch
            add_k = 1 + ((cycle // 2) % 3)  # cycles 1,2,3
            try:
                X = x_best.copy()
                for _ in range(add_k):
                    if time.time() > SEARCH_END - 1.0:
                        break
                    s = deepest_hole(X, ns=8000, top=6, iters=60)
                    X = np.vstack([X, s])
                    X = refine_full(X, steps=160, beta0=30.0, beta1=220.0, lr0=0.010, lr1=0.001)
                if X.shape[0] != 168 + add_k:
                    continue
                X = refine_full(X, steps=700, beta0=25.0, beta1=450.0, lr0=0.012, lr1=0.0007)
                loads, _, _ = contact_loads(X)
                # delete point with largest load (tie-break random)
                jitter = 1e-9 * rng.rand(X.shape[0])
                del_i = int(np.argmax(loads + jitter))
                mask = np.ones(X.shape[0], dtype=bool)
                mask[del_i] = False
                X168 = X[mask]
                X168 = refine_full(X168, steps=700, beta0=25.0, beta1=450.0, lr0=0.012, lr1=0.0007)
                v = _max_cos(X168)
                improved = False
                if v < best_val:
                    best_val = v
                    x_best = X168.copy()
                    improved = True
                # LP polish on improvement or occasional near-miss
                do_lp = False
                budget = 0.0
                if improved:
                    do_lp = True; budget = 2.2
                elif v < best_val + 0.001 and (cycle % 4 == 1):
                    do_lp = True; budget = 0.8
                if do_lp and time.time() < SEARCH_END - 0.5:
                    try:
                        xp = _lp_polish(X168, min(time.time() + budget, SEARCH_END + 4.0, HARD_END - 6.0))
                        vp = _max_cos(xp)
                        if vp < best_val:
                            best_val = vp
                            x_best = xp.copy()
                    except Exception:
                        pass
            except Exception:
                continue
        else:
            # UNDERSHOOT branch
            k_del = 8 + ((cycle // 2) % 11)  # 8..18
            group = 2 + ((cycle // 2) % 3)  # 2,3,4 insertions per short refine
            try:
                loads, _, _ = contact_loads(x_best)
                jitter = 1e-9 * rng.rand(168)
                del_idx = np.argsort(-(loads + jitter), kind="stable")[:k_del]
                mask = np.ones(168, dtype=bool)
                mask[del_idx] = False
                Xcur = x_best[mask].copy()  # 150..160
                Xcur = refine_full(Xcur, steps=500, beta0=25.0, beta1=400.0, lr0=0.012, lr1=0.0008)
                # grow back
                fail = False
                while Xcur.shape[0] < 168:
                    if time.time() > SEARCH_END - 1.0:
                        fail = True
                        break
                    for _ in range(group):
                        if Xcur.shape[0] >= 168:
                            break
                        s = deepest_hole(Xcur, ns=8000, top=6, iters=60)
                        Xcur = np.vstack([Xcur, s])
                    Xcur = refine_full(Xcur, steps=200, beta0=30.0, beta1=250.0, lr0=0.010, lr1=0.001)
                if fail:
                    continue
                Xcur = refine_full(Xcur, steps=700, beta0=25.0, beta1=450.0, lr0=0.012, lr1=0.0007)
                v = _max_cos(Xcur)
                improved = False
                if v < best_val:
                    best_val = v
                    x_best = Xcur.copy()
                    improved = True
                do_lp = False
                budget = 0.0
                if improved:
                    do_lp = True; budget = 2.2
                elif v < best_val + 0.001 and (cycle % 4 == 0):
                    do_lp = True; budget = 0.8
                if do_lp and time.time() < SEARCH_END - 0.5:
                    try:
                        xp = _lp_polish(Xcur, min(time.time() + budget, SEARCH_END + 4.0, HARD_END - 6.0))
                        vp = _max_cos(xp)
                        if vp < best_val:
                            best_val = vp
                            x_best = xp.copy()
                    except Exception:
                        pass
            except Exception:
                continue

    if time.time() < HARD_END - 2.0:
        try:
            rem = min(HARD_END - 1.0, time.time() + max(6.0, HARD_END - time.time() - 1.0))
            x_best = _lp_polish(x_best, rem)
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
