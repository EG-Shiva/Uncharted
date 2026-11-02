import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X, steps=250, T0=0.02, T1=0.0025, lr0=0.04, lr1=0.004):
    n, d = X.shape
    x = X.copy()
    best = _max_cos(x)
    best_x = x.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-12
        C = (W @ x) / S
        dot = (C * x).sum(axis=1, keepdims=True)
        Ct = C - dot * x
        x = x - lr * Ct
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm < 1e-12] = 1.0
        x /= nrm
        if (s & 63) == 63 or s == steps - 1:
            v = _max_cos(x)
            if v < best:
                best = v
                best_x = x.copy()
    return best_x, best

def _best_hole(Xkeep, rng, S=3000):
    d = Xkeep.shape[1]
    R = rng.standard_normal((S, d))
    R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-16)
    D = R @ Xkeep.T
    m = D.max(axis=1)
    i = int(np.argmin(m))
    return R[i].copy()

def _whiten(X):
    n, d = X.shape
    M = (X.T @ X) / n
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, 1e-6)
    target = 1.0 / d
    sc = np.sqrt(target / w)
    Wm = (V * sc[None, :]) @ V.T
    Y = X @ Wm
    Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-16)
    return Y

def _lp_polish(x, deadline, r_init=0.008, r_max=0.04, cap=2200, max_it=40):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    it = 0
    while time.time() + 0.5 < deadline and radius > 1e-13 and it < max_it:
        it += 1
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y; best = v; best_x = y.copy()
            radius = min(radius * 1.5, r_max)
        else:
            radius *= 0.5
    return best_x, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    rng = np.random.default_rng(0)
    n, d = 78, 15
    cand = []
    for f in ("best.npy", "record.npy"):
        try:
            a = np.load(f).astype(float)
            if a.shape == (n, d):
                a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-16
                cand.append(a)
        except Exception:
            pass
    if not cand:
        a = rng.standard_normal((n, d)); a /= np.linalg.norm(a, axis=1, keepdims=True)
        cand.append(a)
    base = min(cand, key=_max_cos).copy()
    v_base = _max_cos(base)
    best_x = base.copy(); best_v = v_base
    elites = [(best_v, best_x.copy())]

    def consider(Xq, vq):
        global_best = elites
        if vq < best_v - 1e-12:
            pass
        # keep top 4
        elites.append((vq, Xq.copy()))
        elites.sort(key=lambda t: t[0])
        del elites[4:]

    # quick LP squeeze of base is skipped; go straight to diverse quench screening
    NTRIAL = 160
    for trial in range(NTRIAL):
        if time.time() > HARD_END - 60:
            break
        mode = trial % 10
        Xt = base.copy()
        if mode == 0:
            Xt = Xt + 0.01 * rng.standard_normal((n, d))
        elif mode == 1:
            Xt = Xt + 0.05 * rng.standard_normal((n, d))
        elif mode == 2:
            k = 14
            sel = rng.choice(n, size=k, replace=False)
            N = 0.004 * rng.standard_normal((n, d))
            N[sel] += 0.22 * rng.standard_normal((k, d))
            Xt = Xt + N
        elif mode == 3:
            # kill worst personal maxima, reinsert into holes
            G = Xt @ Xt.T; np.fill_diagonal(G, -2.0)
            pm = G.max(axis=1)
            order = np.argsort(-pm)
            k = 4 + (trial % 3)
            repl = order[:k]
            keep = np.ones(n, dtype=bool); keep[repl] = False
            for idx in repl:
                km = np.ones(n, dtype=bool); km[repl[repl != idx]] = False
                # use current Xt keep part (already replaced holes stay)
                Xkeep = Xt[km]
                Xt[idx] = _best_hole(Xkeep, rng, S=2500)
            m2 = np.where(keep)[0]
            Xt[m2] += 0.004 * rng.standard_normal((len(m2), d))
        elif mode == 4:
            # whiten to isotropize then jitter
            Xt = _whiten(Xt)
            Xt = Xt + (0.02 + 0.03 * ((trial // 10) % 3)) * rng.standard_normal((n, d))
        elif mode == 5:
            # coordinate sign-flip block mixing: big topology change
            cols = rng.choice(d, size=rng.integers(3, 8), replace=False)
            rows = rng.choice(n, size=n // 2, replace=False)
            Xt[rows][:, cols] *= -1.0
            Xt += 0.015 * rng.standard_normal((n, d))
        elif mode == 6:
            # interpolate with random
            R = rng.standard_normal((n, d)); R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-16
            a = 0.25 + 0.1 * (trial % 4)
            Xt = (1 - a) * Xt + a * R
        elif mode == 7:
            # fully random start (few)
            if trial % 20 == 7:
                Xt = rng.standard_normal((n, d))
            else:
                Xt = Xt + 0.10 * rng.standard_normal((n, d))
        elif mode == 8:
            # orthogonal mix of two halves
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            half = n // 2
            Xt[:half] = Xt[:half] @ Q.T + 0.01 * rng.standard_normal((half, d))
            Xt[half:] += 0.01 * rng.standard_normal((n - half, d))
        else:
            # heavy global shake
            Xt = Xt + 0.16 * rng.standard_normal((n, d))
        Xt /= (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-16)
        # fast screen quench
        steps = 220 if mode in (3, 5) else 180
        T0 = 0.022 if mode in (2, 6, 9) else 0.016
        try:
            Xq, vq = _quench(Xt, steps=steps, T0=T0, T1=0.0025, lr0=0.045, lr1=0.004)
        except Exception:
            continue
        if vq < elites[-1][0] - 1e-12 or (len(elites) < 4 and vq < 0.19):
            consider(Xq, vq)
        if vq < best_v - 1e-12:
            best_v = vq; best_x = Xq.copy()

    # refine elites with longer quench
    elites_sorted = sorted(elites, key=lambda t: t[0])[:4]
    # also include base
    refined = []
    for ev, ex in elites_sorted:
        if time.time() > HARD_END - 55:
            break
        try:
            Xq, vq = _quench(ex, steps=500, T0=0.010, T1=0.0012, lr0=0.02, lr1=0.003)
        except Exception:
            continue
        refined.append((vq, Xq))
        if vq < best_v:
            best_v = vq; best_x = Xq.copy()
    if refined:
        refined.sort(key=lambda t: t[0])
        order_best = [r[1] for r in refined[:3]]
    else:
        order_best = [best_x]

    # LP squeeze top candidates with remaining time
    for bi, bx in enumerate(order_best):
        budget_end = HARD_END - (8 if bi < len(order_best) - 1 else 1.0)
        if time.time() + 2.0 > budget_end:
            break
        try:
            Xp, vp = _lp_polish(bx, min(budget_end, time.time() + 22.0),
                                r_init=0.010 if bi == 0 else 0.006, r_max=0.04)
        except Exception:
            continue
        if vp < best_v:
            best_v = vp; best_x = Xp.copy()

    # final micro polish: small quench + tiny LP
    try:
        if time.time() < HARD_END - 8:
            Xq, vq = _quench(best_x, steps=200, T0=0.004, T1=0.0008, lr0=0.008, lr1=0.002)
            if vq < best_v:
                best_v = vq; best_x = Xq.copy()
        if time.time() < HARD_END - 1.0:
            Xp, vp = _lp_polish(best_x, HARD_END - 0.4, r_init=0.002, r_max=0.008, max_it=30)
            if vp < best_v:
                best_v = vp; best_x = Xp.copy()
    except Exception:
        pass
    best_x = best_x / (np.linalg.norm(best_x, axis=1, keepdims=True) + 1e-16)
    if not np.all(np.isfinite(best_x)):
        best_x = base
    return best_x
