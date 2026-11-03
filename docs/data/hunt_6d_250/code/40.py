import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _adam(X0, steps, b0, b1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1m = 0.9
    b2m = 0.999
    eps = 1e-8
    t = 0
    lb0 = np.log(b0)
    lb1 = np.log(b1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = float(np.max(G))
        G -= cm
        G *= beta
        np.clip(G, -30.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G))
        if S < 1e-100 or not np.isfinite(S):
            continue
        GX = G @ X
        GX *= (2.0 / S)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GX -= dot * X
        t += 1
        m *= b1m
        m += (1.0 - b1m) * GX
        v *= b2m
        v += (1.0 - b2m) * (GX * GX)
        c1 = 1.0 - (b1m ** t)
        c2 = 1.0 - (b2m ** t)
        denom = np.sqrt(v / c2) + eps
        step = (lr / c1) * m / denom
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 5.0 * lr + 1e-12
        sc = np.minimum(1.0, cap / sn)
        step *= sc
        X -= step
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
        md = np.sum(m * X, axis=1, keepdims=True)
        m -= md * X
    return X


def _lp_polish(x, deadline, k_top=5000, r0=2.0e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline - 1.0 and radius > 1e-14:
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        m = float(np.max(c_all))
        sel = c_all > m - max(20.0 * radius, 2e-6)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > k_top:
            idx = np.argpartition(-c_all, k_top)[:k_top]
            I = iu0[idx]
            J = iu1[idx]
            c_sel = c_all[idx]
        else:
            I = iu0[sel]
            J = iu1[sel]
            c_sel = c_all[sel]
        p = I.shape[0]
        nv = n * d + 1
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n),
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
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        vv = _max_cos(y)
        if vv < best - 1e-12:
            x, best = y, vv
            radius = min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
        if time.time() > deadline - 1.5:
            break
    return x


def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    HOP_END = t0 + 132.0
    rng = np.random.default_rng(0)
    n, d = 250, 6
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = z / np.linalg.norm(z, axis=1, keepdims=True)
                if np.all(np.isfinite(z)):
                    cands.append(z)
        except Exception:
            pass
    if not cands:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)

    # initial collective anneal to balance plateau
    try:
        cur = _adam(best, 1200, 2000.0, 40000.0, 0.004, 0.0003)
        v = _max_cos(cur)
        if v < best_val:
            best, best_val = cur, v
    except Exception:
        pass

    iu0, iu1 = np.triu_indices(n, 1)
    trial = 0
    while time.time() < HOP_END and trial < 300:
        trial += 1
        mode = trial % 5
        Y = best.copy()
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            gv = G[iu0, iu1]
            kk = int(np.argmax(gv))
            wa, wb = int(iu0[kk]), int(iu1[kk])
            rowmax = G.max(axis=1)
        except Exception:
            wa, wb = 0, 1
            rowmax = np.zeros(n)
        if mode == 0:
            k = int(3 + rng.integers(0, 10))
            idx = np.argsort(-rowmax)[:k]
            sigma = 0.20 + 0.15 * float(rng.random())
            Y[idx] += rng.standard_normal((k, d)) * sigma
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps, b0, b1, lr0 = 700, 1500.0, 25000.0, 0.004
        elif mode == 1:
            k = int(8 + rng.integers(0, 8))
            # neighbourhood of bottleneck pair
            sc = G[wa] + G[wb]
            idx = np.argsort(-sc)[:k]
            if wa not in idx:
                idx[0] = wa
            if wb not in idx:
                idx[1 % len(idx)] = wb
            sigma = 0.12 + 0.12 * float(rng.random())
            Y[idx] += rng.standard_normal((k, d)) * sigma
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps, b0, b1, lr0 = 700, 2500.0, 35000.0, 0.0035
        elif mode == 2:
            # global jitter + random block big kick
            Y += rng.standard_normal((n, d)) * 0.012
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            k = int(6 + rng.integers(0, 14))
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.30 + 0.20 * float(rng.random())
            Y[idx] += rng.standard_normal((k, d)) * sigma
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps, b0, b1, lr0 = 800, 2000.0, 30000.0, 0.004
        elif mode == 3:
            # void regrowth of worst block
            k = int(10 + rng.integers(0, 15))
            sel = np.argsort(-rowmax)[:k]
            mask = np.ones(n, dtype=bool)
            mask[sel] = False
            Others = Y[mask]
            Ns = 6000
            R = rng.standard_normal((Ns, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-16
            D = R @ Others.T
            md = D.max(axis=1)
            order = np.argsort(md)
            picked = []
            for oi in order:
                cand = R[oi]
                ok = True
                for pc in picked:
                    if float(cand @ pc) > 0.85:
                        ok = False
                        break
                if not ok:
                    continue
                picked.append(cand)
                if len(picked) >= k:
                    break
            for ti, sidx in enumerate(sel):
                if ti < len(picked):
                    Y[sidx] = picked[ti]
                else:
                    vv = rng.standard_normal(d)
                    vv /= np.linalg.norm(vv) + 1e-16
                    Y[sidx] = vv
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps, b0, b1, lr0 = 900, 1500.0, 30000.0, 0.004
        else:
            # coherent cap shift + noise
            u = rng.standard_normal(d)
            u /= np.linalg.norm(u) + 1e-16
            dots = best @ u
            k = int(12 + rng.integers(0, 14))
            thr = np.sort(dots)[-k]
            sel = np.where(dots >= thr)[0]
            w = rng.standard_normal(d)
            w -= float(w @ u) * u
            wn = float(np.linalg.norm(w))
            if wn < 1e-12:
                w = np.zeros(d)
                w[0] = 1.0
            else:
                w /= wn
            shift = 0.08 + 0.08 * float(rng.random())
            Y[sel] += w * shift + rng.standard_normal((sel.shape[0], d)) * 0.05
            Y += rng.standard_normal((n, d)) * 0.008
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps, b0, b1, lr0 = 700, 2000.0, 40000.0, 0.0035
        if time.time() > HOP_END:
            break
        try:
            Y2 = _adam(Y, steps, b0, b1, lr0, 0.0002)
        except Exception:
            continue
        try:
            v = _max_cos(Y2)
        except Exception:
            continue
        if v < best_val - 1e-12:
            best, best_val = Y2, v
            # opportunistic squeeze on improvement if time allows
            if time.time() < HOP_END - 5.0:
                try:
                    Y3 = _adam(best, 400, 10000.0, 80000.0, 0.0015, 0.0001)
                    v3 = _max_cos(Y3)
                    if v3 < best_val:
                        best, best_val = Y3, v3
                except Exception:
                    pass
        if time.time() > HARD - 30.0:
            break

    # final high-beta squeeze
    if time.time() < HARD - 28.0:
        try:
            Yf = _adam(best, 1000, 8000.0, 80000.0, 0.002, 0.0001)
            vf = _max_cos(Yf)
            if vf < best_val:
                best, best_val = Yf, vf
        except Exception:
            pass
    # wide-window LP to tie plateau
    if time.time() < HARD - 3.0:
        try:
            deadline = min(HARD - 1.0, time.time() + 24.0)
            Yl = _lp_polish(best, deadline, k_top=5000, r0=2.0e-3)
            vl = _max_cos(Yl)
            if vl < best_val:
                best, best_val = Yl, vl
            # one more adam after LP
            if time.time() < HARD - 6.0:
                try:
                    Yf2 = _adam(best, 500, 15000.0, 100000.0, 0.001, 0.0001)
                    v2 = _max_cos(Yf2)
                    if v2 < best_val:
                        best, best_val = Yf2, v2
                except Exception:
                    pass
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
