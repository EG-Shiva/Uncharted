import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal(x, steps, beta0, beta1, lr0, mu=0.85):
    n = x.shape[0]
    x = _norm(x.copy())
    v = np.zeros_like(x)
    if beta1 > beta0:
        lb = np.log(beta1 / beta0)
    else:
        lb = 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        v = mu * v + Gt
        vd = (v * x).sum(axis=1, keepdims=True)
        v = v - vd * x
        x = x - lr * v
        x = _norm(x)
    return x

def _lp_deadline(x, deadline, r0=1e-3):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = c.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if not np.any(sel):
            break
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = I[:, None] * d + np.arange(d)
        cj = J[:, None] * d + np.arange(d)
        cols = np.concatenate([ci, cj, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            best = vv; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb

def _contacts(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    C = G > (m - 1e-9)
    deg = C.sum(axis=1)
    per = np.maximum(G.max(axis=1), -2.0)
    return m, deg, per, G

def _sample_holes(Xf, rng, n_holes, n_samples=6000):
    S = rng.standard_normal((n_samples, 4))
    S = _norm(S)
    C = S @ Xf.T
    score = C.max(axis=1)
    order = np.argsort(score)
    picked = []
    for idx in order:
        s = S[idx]
        ok = True
        for q in picked:
            if float(s @ q) > 0.92:
                ok = False; break
        if ok:
            picked.append(s)
        if len(picked) >= n_holes:
            break
    while len(picked) < n_holes:
        picked.append(S[order[len(picked)]])
    return np.array(picked)

def solve():
    t0 = time.time()
    TLIM = 172.0
    rng = np.random.default_rng(777001)
    n, d = 292, 4

    def _load(p):
        try:
            a = np.load(p)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nn = np.linalg.norm(a, axis=1, keepdims=True); nn[nn == 0] = 1.0
                return a / nn
        except Exception:
            pass
        return None

    best = _load("best.npy")
    rec = _load("record.npy")
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
    if rec is not None and _max_cos(rec) < _max_cos(best):
        best = rec.copy()
    best_val = _max_cos(best)

    # initial squeeze
    if time.time() - t0 < 150:
        dl = min(t0 + 12.0, t0 + TLIM - 30)
        if time.time() < dl:
            cand = _lp_deadline(best, dl)
            v = _max_cos(cand)
            if v < best_val:
                best, best_val = cand, v

    it = 0
    # reserve final polish
    while time.time() - t0 < TLIM - 22.0:
        it += 1
        mode = it % 5
        base = best.copy()
        m, deg, per, G = _contacts(base)
        order_high = np.argsort(-deg)
        order_low = np.argsort(deg)
        order_worst = np.argsort(-per)
        try:
            if mode == 0:
                # rattler / low-degree to hole
                k = 2 if (deg <= 2).sum() >= 2 else 1
                idx = list(order_low[:k])
                keep = [i for i in range(n) if i not in set(idx)]
                Xf = base[np.array(keep)]
                holes = _sample_holes(Xf, rng, len(idx), n_samples=6000)
                Xc = base.copy()
                for ii, h in zip(idx, holes):
                    Xc[ii] = h
                Xc = _anneal(Xc, 350, 60.0, 900.0, 0.035, 0.85)
                dl = min(time.time() + 4.0, t0 + TLIM - 22.0)
                Xc = _lp_deadline(Xc, dl)
            elif mode == 1:
                # overload relief: 1 high + 1 low to holes
                idx = [int(order_high[0]), int(order_low[0])]
                if idx[0] == idx[1]:
                    idx[1] = int(order_low[1])
                keep = [i for i in range(n) if i not in set(idx)]
                Xf = base[np.array(keep)]
                holes = _sample_holes(Xf, rng, 2, n_samples=6000)
                Xc = base.copy()
                for ii, h in zip(idx, holes):
                    Xc[ii] = h
                # slight shake of neighbours of high point to break jam
                dists = G[idx[0]].copy()
                dists[idx[0]] = -2
                nbr = np.argsort(-dists)[:4]
                Xc[nbr] += 0.004 * rng.standard_normal((len(nbr), 4))
                Xc = _norm(Xc)
                Xc = _anneal(Xc, 350, 60.0, 900.0, 0.035, 0.85)
                dl = min(time.time() + 4.0, t0 + TLIM - 22.0)
                Xc = _lp_deadline(Xc, dl)
            elif mode == 2:
                # worst-pair subset kick
                k = 5
                idx = list(order_worst[:k])
                Xc = base.copy()
                Xc[np.array(idx)] = _norm(rng.standard_normal((k, 4)))
                Xc = _anneal(Xc, 400, 50.0, 800.0, 0.04, 0.85)
                dl = min(time.time() + 4.0, t0 + TLIM - 22.0)
                Xc = _lp_deadline(Xc, dl)
            elif mode == 3:
                # global small shake
                sig = 0.008 if (it % 10 < 7) else 0.016
                Xc = base + sig * rng.standard_normal((n, d))
                Xc = _norm(Xc)
                Xc = _anneal(Xc, 300, 80.0, 1000.0, 0.03, 0.85)
                dl = min(time.time() + 3.5, t0 + TLIM - 22.0)
                Xc = _lp_deadline(Xc, dl)
            else:
                # cluster transplant: pick a high point and move it + random to holes
                h0 = int(order_high[rng.integers(0, 8)])
                r1 = int(rng.integers(0, n))
                idx = [h0, r1] if r1 != h0 else [h0, int(order_low[0])]
                keep = [i for i in range(n) if i not in set(idx)]
                Xf = base[np.array(keep)]
                holes = _sample_holes(Xf, rng, 2, n_samples=8000)
                Xc = base.copy()
                for ii, h in zip(idx, holes):
                    Xc[ii] = h
                Xc = _anneal(Xc, 350, 60.0, 900.0, 0.035, 0.85)
                dl = min(time.time() + 4.0, t0 + TLIM - 22.0)
                Xc = _lp_deadline(Xc, dl)
        except Exception:
            continue
        try:
            v = _max_cos(Xc)
        except Exception:
            continue
        if v < best_val and np.all(np.isfinite(Xc)):
            best, best_val = Xc, v
        # early exit if time low
        if time.time() - t0 >= TLIM - 22.0:
            break

    # final polish of best
    try:
        dl = t0 + TLIM - 1.0
        best = _lp_deadline(best, dl)
        best_val = _max_cos(best)
    except Exception:
        pass
    best = _norm(np.asarray(best, dtype=float))
    return best
