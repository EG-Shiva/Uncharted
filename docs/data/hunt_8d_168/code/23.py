import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_quench(X, steps, beta0, beta1, lr0, lr1, rownorm):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        if rownorm:
            S = W.sum(axis=1, keepdims=True) + 1e-12
            Grad = (W @ X) / S
        else:
            Z = W.sum()
            if Z == 0:
                continue
            Grad = (W @ X) / Z
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** t; bc2 = 1.0 - b2 ** t
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _hard_polish(X, steps=300, lr0=0.004, lr1=0.0002, tol0=3e-4, tol1=3e-6):
    n, d = X.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tol = tol0 + (tol1 - tol0) * frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        M = (G >= (m - tol)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        cnt = M.sum(axis=1, keepdims=True)
        active = (cnt[:, 0] > 0.5)
        if not np.any(active):
            continue
        Grad = (M @ X) / np.maximum(cnt, 1.0)
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        step = np.zeros_like(Gt)
        step[active] = Gt[active] / nn[active]
        X = X - lr * step
        nrm = np.sqrt((X * X).sum(axis=1, keepdims=True))
        nrm[nrm == 0] = 1.0
        X /= nrm
    return X

def _refine_hole(Y, p, iters=200, beta=90.0, lr0=0.12, lr1=0.004):
    p = p / np.linalg.norm(p)
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        g = Y @ p
        m = g.max()
        w = np.exp(beta * (g - m))
        s = w.sum() + 1e-12
        A = (w @ Y) / s
        At = A - float(A @ p) * p
        p = p - lr * At
        nr = np.linalg.norm(p)
        if nr == 0:
            break
        p /= nr
    return p

def _find_holes(Y, k, rng, d, n_probe=14000):
    P = rng.randn(n_probe, d)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    CH = 4000
    bs = np.empty(n_probe)
    for a in range(0, n_probe, CH):
        b = min(a + CH, n_probe)
        D = P[a:b] @ Y.T
        bs[a:b] = D.max(axis=1)
    order = np.argsort(bs)
    sel = []
    for idx in order:
        if len(sel) >= k:
            break
        p = P[idx]
        ok = True
        for q in sel:
            if abs(float(p @ q)) > 0.90:
                ok = False
                break
        if ok:
            sel.append(p.copy())
    j = 0
    while len(sel) < k:
        sel.append(P[order[j % len(order)]].copy())
        j += 1
    return sel

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 174.0
    LP_FINAL = 30.0
    explore_end = t0 + BUDGET - LP_FINAL - 1.0
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (168, 8):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best = a.copy()
        except Exception:
            pass
    if best is None:
        rng0 = np.random.RandomState(0)
        best = rng0.randn(168, 8)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    n, d = best.shape
    cur = best.copy()
    cur_val = best_val
    elite = [(best_val, best.copy())]
    rng = np.random.RandomState(12345)
    iu0 = np.triu_indices(n, 1)
    T0 = 3e-9
    T1 = 1e-11
    it = 0
    while time.time() < explore_end:
        it += 1
        elapsed = time.time() - t0
        frac_g = elapsed / max(1e-9, (explore_end - t0))
        Temp = T0 * (T1 / T0) ** frac_g
        # choose base: mostly chain, sometimes elite
        if rng.rand() < 0.30 and len(elite) > 1:
            bi = rng.randint(len(elite))
            # bias to better
            if rng.rand() < 0.6:
                bi = 0
            base = elite[bi][1]
            base_val = elite[bi][0]
        else:
            base = cur
            base_val = cur_val
        # occasional reset of wandering chain
        if cur_val > best_val + 8e-4 and rng.rand() < 0.25:
            base = best
            base_val = best_val
        r = rng.rand()
        try:
            if r < 0.34:
                sigma = [0.004, 0.01, 0.022, 0.045, 0.085][rng.randint(5)]
                X = base + rng.randn(n, d) * sigma
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                steps = 600 + int(rng.randint(900))
                beta0 = 25.0 + rng.rand() * 45.0
                beta1 = 250.0 + rng.rand() * 550.0
                lr0 = 0.004 + rng.rand() * 0.010
                lr1 = lr0 * (0.07 + rng.rand() * 0.12)
                rownorm = (rng.rand() < 0.5)
                Xq = _adam_quench(X, steps, beta0, beta1, lr0, lr1, rownorm)
                if rng.rand() < 0.5:
                    Xq = _hard_polish(Xq, steps=250, lr0=0.003, lr1=0.0002)
            elif r < 0.66:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                rmax = G.max(axis=1)
                m0 = G.max()
                cnt = (G >= m0 - 1e-7).sum(axis=1)
                k = 1 + int(rng.randint(3))
                u = rng.rand()
                if u < 0.45:
                    victims = np.argsort(rmax)[-k:]
                elif u < 0.75:
                    victims = np.argsort(cnt)[:k]
                    # mix: if rattlers exist prefer them sometimes
                    if rng.rand() < 0.5:
                        ratt = np.where(cnt == 0)[0]
                        if len(ratt) >= k:
                            victims = rng.choice(ratt, size=k, replace=False)
                else:
                    victims = rng.randint(n, size=k)
                X = base.copy()
                alive_mask = np.ones(n, dtype=bool)
                alive_mask[victims] = False
                Y_alive = X[alive_mask]
                holes = _find_holes(Y_alive, k, rng, d)
                for tii, idx in enumerate(victims):
                    p0 = holes[tii]
                    beta_h = 60.0 + rng.rand() * 80.0
                    p_ref = _refine_hole(Y_alive, p0, iters=200, beta=beta_h)
                    # small jitter
                    p_ref = p_ref + rng.randn(d) * 1e-3
                    p_ref /= np.linalg.norm(p_ref)
                    X[idx] = p_ref
                    Y_alive = X[np.ones(n, dtype=bool)]  # not needed, keep simple
                    # update alive for next hole diversity: include new point
                    # (recompute lazily by using X with placeholder)
                steps = 800 + int(rng.randint(900))
                beta0 = 30.0 + rng.rand() * 40.0
                beta1 = 300.0 + rng.rand() * 500.0
                lr0 = 0.004 + rng.rand() * 0.008
                rownorm = (rng.rand() < 0.5)
                Xq = _adam_quench(X, steps, beta0, beta1, lr0, lr0 * 0.12, rownorm)
                if rng.rand() < 0.4:
                    Xq = _hard_polish(Xq, steps=300, lr0=0.004, lr1=0.00015)
            elif r < 0.83:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                flat = G[iu0]
                K = min(40, flat.size)
                top = np.argpartition(flat, -K)[-K:]
                ti = top[rng.randint(K)]
                a = int(iu0[0][ti]); b = int(iu0[1][ti])
                X = base.copy()
                amp = 0.015 + rng.rand() * 0.06
                for (pp, qq) in ((a, b), (b, a)):
                    dot = float(X[pp] @ X[qq])
                    tang = X[qq] - dot * X[pp]
                    nn = np.linalg.norm(tang)
                    if nn > 1e-12:
                        tang /= nn
                        X[pp] = X[pp] - amp * tang
                        X[pp] /= np.linalg.norm(X[pp])
                G2 = X @ X.T
                np.fill_diagonal(G2, -2.0)
                for pp in (a, b):
                    nbrs = np.where(G2[pp] >= G2[pp].max() - 5e-4)[0][:6]
                    for nb in nbrs:
                        if nb == pp:
                            continue
                        X[nb] = X[nb] + rng.randn(d) * 0.0025
                        X[nb] /= np.linalg.norm(X[nb])
                steps = 650 + int(rng.randint(700))
                beta0 = 28.0 + rng.rand() * 40.0
                beta1 = 280.0 + rng.rand() * 500.0
                lr0 = 0.003 + rng.rand() * 0.007
                rownorm = (rng.rand() < 0.5)
                Xq = _adam_quench(X, steps, beta0, beta1, lr0, lr0 * 0.1, rownorm)
            else:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                m0 = G.max()
                cnt = (G >= m0 - 1e-7).sum(axis=1)
                hub = int(np.argmax(cnt + rng.randn(n) * 0.6))
                nbrs = np.where(G[hub] >= m0 - 2e-3)[0]
                if len(nbrs) > 8:
                    nbrs = nbrs[np.argsort(G[hub, nbrs])[-8:]]
                X = base.copy()
                sig = 0.018 + rng.rand() * 0.045
                sel = np.concatenate([[hub], nbrs])
                X[sel] += rng.randn(len(sel), d) * sig
                X[sel] /= np.linalg.norm(X[sel], axis=1, keepdims=True)
                # also shake a random rattler if exists
                if rng.rand() < 0.5:
                    ratt = np.where(cnt == 0)[0]
                    if len(ratt) > 0:
                        ri = ratt[rng.randint(len(ratt))]
                        X[ri] = rng.randn(d)
                        X[ri] /= np.linalg.norm(X[ri])
                steps = 750 + int(rng.randint(800))
                beta0 = 25.0 + rng.rand() * 40.0
                beta1 = 260.0 + rng.rand() * 500.0
                lr0 = 0.004 + rng.rand() * 0.008
                rownorm = (rng.rand() < 0.5)
                Xq = _adam_quench(X, steps, beta0, beta1, lr0, lr0 * 0.12, rownorm)
        except Exception:
            continue
        try:
            vq = _max_cos(Xq)
        except Exception:
            continue
        if not np.isfinite(vq):
            continue
        if vq < best_val - 1e-15:
            best_val = vq
            best = Xq.copy()
            cur = Xq.copy()
            cur_val = vq
            elite.append((vq, Xq.copy()))
            elite = sorted(elite, key=lambda z: z[0])[:6]
        elif vq < cur_val:
            cur = Xq.copy()
            cur_val = vq
            # insert into elite if good enough
            if len(elite) < 6 or vq < elite[-1][0]:
                elite.append((vq, Xq.copy()))
                elite = sorted(elite, key=lambda z: z[0])[:6]
        else:
            if Temp > 0 and rng.rand() < np.exp(-(vq - cur_val) / Temp):
                cur = Xq.copy()
                cur_val = vq
        if time.time() + 0.8 > explore_end:
            break
    remain = (t0 + BUDGET) - time.time() - 0.4
    if remain > 4 and best is not None:
        elite = sorted(elite, key=lambda z: z[0])[:3]
        # ensure best first
        if elite[0][0] > best_val:
            elite = [(best_val, best.copy())] + elite
            elite = elite[:3]
        per = remain / len(elite)
        for (_, Xe) in list(elite):
            if time.time() + 2 > t0 + BUDGET - 0.3:
                break
            try:
                Yp = _lp_polish(Xe, seconds=min(per, (t0 + BUDGET - 0.3) - time.time()))
                vp = _max_cos(Yp)
                if vp < best_val:
                    best_val = vp
                    best = Yp.copy()
            except Exception:
                continue
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
