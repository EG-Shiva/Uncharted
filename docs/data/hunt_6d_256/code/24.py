import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_lse(y0, steps, T0, T1, lr0, lr1):
    y = y0.copy()
    n, d = y.shape
    m = np.zeros_like(y)
    v = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    best = _max_cos(y)
    best_x = y.copy()
    b1p, b2p = 1.0, 1.0
    for s in range(1, steps + 1):
        f = s / max(steps, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        Grad = (E @ y) / S
        dot = np.sum(Grad * y, axis=1, keepdims=True)
        Gt = Grad - dot * y
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * Gt
        v *= b2
        v += (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * y, axis=1, keepdims=True)
        step = step - sdot * y
        y = y - step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        if s % 150 == 0 or s == steps:
            val = _max_cos(y)
            if val < best:
                best = val
                best_x = y.copy()
    return best_x, best

def _lp_polish(x, deadline, r0=8e-4, max_pairs=5000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        win = max(12.0 * radius, 3e-6)
        if radius < 3e-7:
            win = max(12.0 * radius, 3e-7)
        sel = c > (m - win)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > max_pairs:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:max_pairs]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        val = _max_cos(y)
        if val < best:
            x, best = y, val
            radius = min(radius * 1.4, 0.01)
        else:
            radius *= 0.5
    return x, best

def _refine_single(x, i, cand, other_max, best, rng, nsteps=250):
    d = x.shape[1]
    others = np.delete(x, i, axis=0)
    cur = cand.copy()
    cur_dot = float((others @ cur).max())
    cur_glob = cur_dot if cur_dot > other_max else other_max
    sigma = 0.02
    for _ in range(nsteps):
        r = rng.normal(size=d)
        r = r - float(np.dot(r, cur)) * cur
        nr = float(np.linalg.norm(r))
        if nr < 1e-12:
            continue
        r /= nr
        prop = cur + sigma * r
        prop /= np.linalg.norm(prop)
        pd = float((others @ prop).max())
        pg = pd if pd > other_max else other_max
        if pg < cur_glob - 1e-14:
            cur = prop
            cur_dot = pd
            cur_glob = pg
        else:
            sigma *= 0.995
            if sigma < 1e-5:
                sigma = 1e-5
    return cur, cur_glob, cur_dot

def _greedy_sweep(x, best, rng, deadline, batch=3072, top_k=40):
    n, d = x.shape
    while time.time() < deadline:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        prow = G.max(axis=1)
        order = np.argsort(-prow)
        any_imp = False
        for kk in range(top_k):
            if time.time() >= deadline:
                break
            i = int(order[kk])
            cur = float(prow[i])
            if cur < best - 2e-3:
                break
            Gm = G.copy()
            Gm[i, :] = -2.0
            Gm[:, i] = -2.0
            other_max = float(Gm.max())
            if other_max >= best - 1e-13:
                continue
            # need local < best
            B1 = batch // 2
            B2 = batch - B1
            Zg = rng.normal(size=(B1, d))
            Zg /= np.linalg.norm(Zg, axis=1, keepdims=True)
            # local perturbations with mixed scales
            base = np.repeat(x[i:i+1], B2, axis=0)
            sig = np.empty(B2)
            half = B2 // 2
            sig[:half] = 0.03
            sig[half:] = 0.15
            pert = rng.normal(size=(B2, d))
            # project not needed, just add then normalize
            C = np.vstack([Zg, base + sig[:, None] * pert])
            Cn = np.linalg.norm(C, axis=1, keepdims=True)
            Cn[Cn == 0] = 1.0
            C = C / Cn
            D = C @ x.T
            D[:, i] = -2.0
            rm = D.max(axis=1)
            pick = int(np.argmin(rm))
            bloc = float(rm[pick])
            glob = bloc if bloc > other_max else other_max
            if glob < best - 1e-13:
                # refine
                rc, rglob, _ = _refine_single(x, i, C[pick], other_max, best, rng, nsteps=200)
                if rglob < glob:
                    glob = rglob
                    C[pick] = rc
                x[i] = C[pick]
                best = glob
                any_imp = True
                # update G row/col in place for remaining ks
                Gi = x @ x[i]
                G[i, :] = Gi
                G[:, i] = Gi
                G[i, i] = -2.0
                prow[i] = float(np.delete(Gi, i).max())
        if not any_imp:
            break
    return x, best

def solve():
    t0 = time.time()
    D1 = t0 + 100.0
    D2 = t0 + 140.0
    D3 = t0 + 168.0
    rng = np.random.default_rng(20261103)
    x_best = None
    for fn in ("best.npy", "record.npy"):
        try:
            x_best = np.load(fn)
            break
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(256, 6))
    x_best = np.asarray(x_best, dtype=float)
    if x_best.shape != (256, 6):
        tmp = rng.normal(size=(256, 6))
        try:
            r = min(x_best.shape[0], 256)
            cc = min(x_best.shape[1], 6)
            tmp[:r, :cc] = x_best[:r, :cc]
        except Exception:
            pass
        x_best = tmp
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    best = _max_cos(x_best)
    n, d = 256, 6
    cycle = 0
    while time.time() < D1:
        cycle += 1
        rem = D1 - time.time()
        if rem < 2.0:
            break
        x = x_best.copy()
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        worst = G.max(axis=1)
        order = np.argsort(-worst)
        mode = cycle % 5
        if mode == 1:
            # void reinsert 2 worst
            for r in order[:2]:
                Z = rng.normal(size=(12000, 6))
                Z /= np.linalg.norm(Z, axis=1, keepdims=True)
                S = Z @ x.T
                S[:, int(r)] = -2.0
                m2 = S.max(axis=1)
                pick = int(np.argmin(m2))
                x[int(r)] = Z[pick]
            sel = order[:6]
            x[sel] += 0.04 * rng.normal(size=(6, 6))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, l0, l1 = 650, 0.02, 0.0007, 0.010, 0.003
        elif mode == 2:
            c0 = int(rng.integers(n))
            dots = x @ x[c0]
            nearest = np.argsort(-dots)[:12]
            x[nearest] += 0.35 * rng.normal(size=(12, 6))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, l0, l1 = 700, 0.025, 0.0008, 0.012, 0.004
        elif mode == 3:
            x += 0.03 * rng.normal(size=x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, l0, l1 = 550, 0.015, 0.0005, 0.008, 0.002
        elif mode == 4:
            pick = rng.choice(n, size=16, replace=False)
            x[pick] += 0.30 * rng.normal(size=(16, 6))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, l0, l1 = 650, 0.02, 0.0007, 0.011, 0.003
        else:
            pick = rng.choice(n, size=32, replace=False)
            x[pick] += 0.50 * rng.normal(size=(32, 6))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps, T0, T1, l0, l1 = 750, 0.03, 0.001, 0.013, 0.004
        if rem < 25:
            steps = min(steps, 350)
        xb, vb = _adam_lse(x, steps, T0, T1, l0, l1)
        # quick greedy snap on worst
        if time.time() < D1 - 1.0:
            try:
                xb, vb2 = _greedy_sweep(xb, vb, rng, min(time.time() + 1.2, D1), batch=2048, top_k=20)
                vb = vb2
            except Exception:
                pass
        if vb < best - 1e-12 and np.all(np.isfinite(xb)):
            x_best = xb
            best = vb
    # intensive greedy
    if time.time() < D2:
        try:
            x_best, best = _greedy_sweep(x_best, best, rng, D2, batch=4096, top_k=40)
        except Exception:
            pass
    # LP polish
    if time.time() < D3:
        try:
            x_best, best = _lp_polish(x_best, min(D3 - 4.0, time.time() + (D3 - time.time()) * 0.7), r0=8e-4)
        except Exception:
            pass
    # final greedy + micro LP
    if time.time() < D3:
        try:
            x_best, best = _greedy_sweep(x_best, best, rng, D3, batch=4096, top_k=40)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = np.where(np.isfinite(x_best), x_best, 0.0)
    nr = np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
