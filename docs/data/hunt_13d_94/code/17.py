import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_relax(X, steps, a0, a1, lr0, lr1, deadline=None):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        if deadline is not None and (s & 127) == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        W = np.exp(alpha * (G - mv))
        np.fill_diagonal(W, 0.0)
        # normalize to keep scale stable
        S = W.sum()
        if not np.isfinite(S) or S == 0:
            break
        W = W / S
        grad = W @ X
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        m1 = b1 * m1 + (1 - b1) * gt
        m2 = b2 * m2 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        mh = m1 / bc1
        vh = m2 / bc2
        X = X - lr * mh / (np.sqrt(vh) + eps)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    return X

def _reinsert_best(others, rng, n_cand=8000):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_cand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ others.T).max(axis=1)
    return cand[int(np.argmin(scores))].copy()

def _refine_single(c, others, steps=250, kk=60.0, step0=0.2):
    c = c / np.linalg.norm(c)
    for it in range(steps):
        dots = others @ c
        m = dots.max()
        w = np.exp(kk * (dots - m))
        sw = w.sum()
        if not np.isfinite(sw) or sw == 0:
            break
        w = w / sw
        grad = w @ others
        gt = grad - (grad @ c) * c
        st = step0 * (1.0 - 0.6 * it / steps)
        c = c - st * gt
        n = np.linalg.norm(c)
        if n == 0 or not np.isfinite(n):
            break
        c = c / n
    return c

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
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

def solve():
    T0 = time.time()
    BUDGET = 176.0
    HOP_END = T0 + 112.0
    rng = np.random.default_rng(1)
    # load start
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(94, 13))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    n, d = best.shape
    cur = best.copy()
    cur_val = best_val
    global_best = best.copy()
    global_val = best_val
    # elite archive (val, array)
    elites = [(global_val, global_best.copy())]

    def push_elite(v, X):
        elites.append((v, X.copy()))
        elites.sort(key=lambda t: t[0])
        del elites[6:]

    def contact_counts(X, tol=1e-3):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cnt = (G > m - tol).sum(axis=1)
        worst = G.max(axis=1)
        return m, cnt, worst

    it = 0
    sigmas = [0.004, 0.01, 0.02, 0.04, 0.08]
    while time.time() < HOP_END:
        it += 1
        elapsed = time.time() - T0
        frac = elapsed / max(1e-9, (HOP_END - T0))
        Temp = 0.008 * (1.0 - 0.85 * frac) + 1e-4
        Y = cur.copy()
        mode = it % 7
        _, cnt, worst = contact_counts(Y)
        order_worst = np.argsort(-worst)
        order_loose = np.argsort(cnt)
        order_jam = np.argsort(-cnt)
        if mode == 0:
            # single worst reinsert
            v = int(order_worst[0])
            others = np.delete(Y, v, axis=0)
            c = _reinsert_best(others, rng, n_cand=8000)
            c = _refine_single(c, others, steps=250, kk=80.0, step0=0.25)
            Y[v] = c
            Y = _adam_relax(Y, steps=700, a0=30.0, a1=280.0, lr0=0.008, lr1=0.0015, deadline=HOP_END)
        elif mode == 1:
            # double reinsert: worst + jammed/loose
            v1 = int(order_worst[0])
            if rng.random() < 0.5:
                v2 = int(order_loose[rng.integers(0, 10)])
            else:
                v2 = int(order_jam[rng.integers(0, 10)])
            if v2 == v1:
                v2 = int(order_worst[1])
            for v in (v1, v2):
                others = np.delete(Y, v, axis=0)
                c = _reinsert_best(others, rng, n_cand=6000)
                c = _refine_single(c, others, steps=200, kk=60.0, step0=0.25)
                Y[v] = c
            Y = _adam_relax(Y, steps=900, a0=25.0, a1=300.0, lr0=0.009, lr1=0.0012, deadline=HOP_END)
        elif mode == 2:
            # triple fresh random replacement in a cap
            u = rng.standard_normal(size=(d,))
            u /= np.linalg.norm(u)
            proj = Y @ u
            k = int(rng.integers(2, 5))
            victims = np.argsort(-proj)[:k]
            for v in victims:
                Y[v] = rng.standard_normal(size=(d,))
                Y[v] /= np.linalg.norm(Y[v])
            # quick single-refine each victim
            for v in victims:
                others = np.delete(Y, v, axis=0)
                Y[v] = _refine_single(Y[v], others, steps=150, kk=50.0, step0=0.2)
            Y = _adam_relax(Y, steps=1100, a0=20.0, a1=320.0, lr0=0.010, lr1=0.0010, deadline=HOP_END)
        elif mode == 3:
            # global gaussian shake small
            sigma = float(sigmas[rng.integers(0, len(sigmas))])
            Y = Y + sigma * rng.standard_normal(size=Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _adam_relax(Y, steps=700, a0=40.0, a1=300.0, lr0=0.006, lr1=0.0010, deadline=HOP_END)
        elif mode == 4:
            # subset shake of loose+jammed points
            k = int(rng.integers(3, 7))
            victims = set()
            while len(victims) < k:
                if rng.random() < 0.5:
                    victims.add(int(order_loose[rng.integers(0, 12)]))
                else:
                    victims.add(int(order_jam[rng.integers(0, 12)]))
            sigma = 0.05 + 0.10 * rng.random()
            for v in victims:
                Y[v] = Y[v] + sigma * rng.standard_normal(size=(d,))
                Y[v] /= np.linalg.norm(Y[v])
            Y = _adam_relax(Y, steps=900, a0=30.0, a1=300.0, lr0=0.008, lr1=0.0012, deadline=HOP_END)
        elif mode == 5:
            # large global shake + long anneal
            sigma = 0.06 + 0.06 * rng.random()
            Y = Y + sigma * rng.standard_normal(size=Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _adam_relax(Y, steps=1600, a0=15.0, a1=350.0, lr0=0.012, lr1=0.0008, deadline=HOP_END)
        else:
            # random single point teleport to best hole
            v = int(rng.integers(0, n))
            others = np.delete(Y, v, axis=0)
            c = _reinsert_best(others, rng, n_cand=12000)
            c = _refine_single(c, others, steps=300, kk=100.0, step0=0.2)
            Y[v] = c
            Y = _adam_relax(Y, steps=600, a0=50.0, a1=320.0, lr0=0.005, lr1=0.0010, deadline=HOP_END)

        v = _max_cos(Y)
        if v < global_val:
            global_val = v
            global_best = Y.copy()
            push_elite(v, Y)
        elif v < elites[-1][0] + 5e-4 and len(elites) < 6 and rng.random() < 0.2:
            push_elite(v, Y)
        # annealing acceptance for chain
        if v < cur_val:
            cur, cur_val = Y, v
        else:
            if rng.random() < np.exp((cur_val - v) / max(1e-9, Temp)):
                cur, cur_val = Y, v
        # occasional restart from global best
        if it % 25 == 0 and rng.random() < 0.4:
            cur, cur_val = global_best.copy(), global_val
        if time.time() > HOP_END:
            break

    # final LP polish of elites + current best
    cands = []
    seen = []
    for vv, XX in elites:
        cands.append((vv, XX))
    cands.append((cur_val, cur))
    # deduplicate roughly and sort
    cands.sort(key=lambda t: t[0])
    keep = cands[:4]
    total_end = T0 + BUDGET - 2.0
    remaining = total_end - time.time()
    per = remaining / max(1, len(keep)) if len(keep) > 0 else remaining
    per = min(max(per, 5.0), 25.0)
    final_best = global_best
    final_val = global_val
    for vv, XX in keep:
        if time.time() > total_end - 3.0:
            break
        dl = min(time.time() + per, total_end)
        try:
            P, pv = _lp_polish(XX.copy(), dl)
        except Exception:
            continue
        if pv < final_val:
            final_val = pv
            final_best = P.copy()
    final_best = final_best / np.linalg.norm(final_best, axis=1, keepdims=True)
    return final_best
