import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam(X0, steps, T0, T1, lr0, lr1, rng, noise0=0.0, idx=None, t_dead=None):
    X = X0.copy()
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(np.max(G0))
    best = X.copy()
    lT = np.log(T1 / T0)
    llr = np.log(lr1 / lr0)
    b1p = 1.0
    b2p = 1.0
    for t in range(1, steps + 1):
        if (t & 63) == 0 and t_dead is not None and time.time() > t_dead:
            break
        frac = t / float(steps)
        T = T0 * np.exp(lT * frac)
        lr = lr0 * np.exp(llr * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = float(np.max(G))
        if cm < bv:
            bv = cm
            best = X.copy()
        S = (G - cm) / T
        np.maximum(S, -15.0, out=S)
        W = np.exp(S)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - proj * X
        b1p *= b1
        b2p *= b2
        # m1 = b1*m1+(1-b1)*Gt etc in place
        m1 *= b1
        m1 += (1.0 - b1) * Gt
        v1 *= b2
        v1 += (1.0 - b2) * (Gt * Gt)
        mhat_d = 1.0 - b1p
        vhat_d = 1.0 - b2p
        # step = lr*mhat/(sqrt(vhat)+eps)
        step = m1 / mhat_d
        step = step / (np.sqrt(v1 / vhat_d) + eps)
        step *= lr
        if idx is None:
            X -= step
            if noise0 > 0:
                amp = noise0 * (1.0 - frac)
                if amp > 0:
                    nz = rng.randn(X.shape[0], 4) * amp
                    nz -= np.sum(nz * X, axis=1, keepdims=True) * X
                    X += nz
            X = _norm(X)
        else:
            X[idx] -= step[idx]
            if noise0 > 0:
                amp = noise0 * (1.0 - frac)
                if amp > 0:
                    nz = rng.randn(len(idx), 4) * amp
                    Xi = X[idx]
                    nz -= np.sum(nz * Xi, axis=1, keepdims=True) * Xi
                    X[idx] += nz
            X[idx] = _norm(X[idx])
    return best, bv

def _greedy_cap(cur_fixed_pool, fixed, k, rng, npool=8000):
    P = rng.randn(npool, 4)
    P = _norm(P)
    # scores vs fixed
    D = P @ fixed.T
    scores = np.max(D, axis=1)
    chosen = []
    for _ in range(k):
        j = int(np.argmin(scores))
        c = P[j].copy()
        chosen.append(c)
        d = P @ c
        # max with existing scores
        np.maximum(scores, d, out=scores)
        scores[j] = 2.0
    return np.array(chosen)

def _find_hole(fixed, rng, ncand=15000):
    R = _norm(rng.randn(ncand, 4))
    D = R @ fixed.T
    vals = np.max(D, axis=1)
    i = int(np.argmin(vals))
    h = R[i].copy()
    # quick tangential refine away from closest
    v = float(vals[i])
    for _ in range(120):
        dots = fixed @ h
        j = int(np.argmax(dots))
        g = fixed[j] - dots[j] * h
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        g /= ng
        improved = False
        for st in (0.03, 0.01, 0.003):
            yn = h - st * g
            yn /= np.linalg.norm(yn)
            vn = float(np.max(fixed @ yn))
            if vn < v:
                h, v = yn, vn
                improved = True
                break
        if not improved:
            break
    return h, v

def _lp_polish(X, seconds, t_dead):
    X = _norm(X.copy())
    bv = _max_cos(X)
    bx = X.copy()
    n, d = X.shape
    radius = 1.5e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if time.time() > t_dead:
            break
        if radius < 1e-13:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(np.max(c))
        thr = max(20.0 * radius, 1e-9)
        sel = c > m - thr
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            # keep largest 2500
            idx = np.argpartition(-c, 2500)[:2500]
            msk = np.zeros_like(sel)
            msk[idx] = True
            sel = msk & sel
            # if still weird, just take top 2500
            if int(np.sum(sel)) < 500:
                msk = np.zeros_like(sel)
                msk[idx] = True
                sel = msk
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols[:, 2 * d] = n * d
        vals = np.empty((p, 2 * d + 1), dtype=float)
        vals[:, :d] = X[J]
        vals[:, d:2 * d] = X[I]
        vals[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals.ravel(), (rows, cols.ravel())), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < bv:
            X, bv = y, v
            bx = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bx, bv

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 132.0
    rng = np.random.RandomState(7)
    n = 163
    best = None
    bv = 2.0
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, 4) and np.all(np.isfinite(v)):
                v = _norm(v)
                c = _max_cos(v)
                if c < bv:
                    bv = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.randn(n, 4))
        bv = _max_cos(best)
    best_x = best.copy()
    best_val = bv
    trial = 0
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 38:
            break
        trial += 1
        mode = trial % 5
        try:
            if mode == 0:
                sig = [0.03, 0.08, 0.18][(trial // 5) % 3]
                cur = best_x + rng.randn(n, 4) * sig
                cur = _norm(cur)
                cur_b, cur_v = _adam(cur, steps=3200, T0=0.012, T1=0.0006, lr0=0.014, lr1=0.0012,
                                     rng=rng, noise0=0.002, t_dead=SEARCH_END)
            elif mode == 1:
                k = 8 + (trial % 8)  # 8..15
                u = rng.randn(4); u /= np.linalg.norm(u)
                dots = best_x @ u
                idx = np.argsort(dots)[-k:]
                fixed = np.delete(best_x, idx, axis=0)
                ch = _greedy_cap(None, fixed, k, rng, npool=8000)
                cur = best_x.copy()
                cur[idx] = ch + rng.randn(k, 4) * 0.004
                cur = _norm(cur)
                # frozen phase
                fz_b, fz_v = _adam(cur, steps=1200, T0=0.010, T1=0.0012, lr0=0.012, lr1=0.002,
                                  rng=rng, noise0=0.0, idx=idx, t_dead=SEARCH_END)
                cur_b, cur_v = _adam(fz_b, steps=2800, T0=0.010, T1=0.0005, lr0=0.013, lr1=0.001,
                                     rng=rng, noise0=0.001, t_dead=SEARCH_END)
            elif mode == 2:
                k = 24 + (trial % 17)  # 24..40
                # cluster around random point + neighbours
                c0 = trial % n
                d2 = np.sum((best_x - best_x[c0]) ** 2, axis=1)
                idx = np.argsort(d2)[:k]
                fixed = np.delete(best_x, idx, axis=0)
                ch = _greedy_cap(None, fixed, k, rng, npool=6000)
                cur = best_x.copy()
                cur[idx] = ch + rng.randn(k, 4) * 0.005
                cur = _norm(cur)
                fz_b, fz_v = _adam(cur, steps=1300, T0=0.013, T1=0.0010, lr0=0.014, lr1=0.002,
                                  rng=rng, noise0=0.0, idx=idx, t_dead=SEARCH_END)
                cur_b, cur_v = _adam(fz_b, steps=3000, T0=0.013, T1=0.0005, lr0=0.013, lr1=0.001,
                                     rng=rng, noise0=0.0015, t_dead=SEARCH_END)
            elif mode == 3:
                G = best_x @ best_x.T
                np.fill_diagonal(G, -2.0)
                m = float(np.max(G))
                deg = np.sum(G > m - 1e-9, axis=1)
                # overloaded top 4 + random 2
                hub = np.argsort(deg)[-4:]
                extra = rng.choice(n, size=2, replace=False)
                idx = np.unique(np.concatenate([hub, extra]))
                fixed0 = np.delete(best_x, idx, axis=0)
                cur = best_x.copy()
                f = fixed0.copy()
                for w in idx:
                    h, _ = _find_hole(f, rng, ncand=12000)
                    cur[w] = h
                    f = np.vstack([f, h.reshape(1, 4)])
                cur += rng.randn(n, 4) * 0.003
                cur = _norm(cur)
                cur_b, cur_v = _adam(cur, steps=3200, T0=0.009, T1=0.00045, lr0=0.012, lr1=0.001,
                                     rng=rng, noise0=0.001, t_dead=SEARCH_END)
            else:
                if (trial // 5) % 4 == 0:
                    cur = _norm(rng.randn(n, 4))
                    cur_b, cur_v = _adam(cur, steps=3800, T0=0.018, T1=0.0007, lr0=0.015, lr1=0.0012,
                                         rng=rng, noise0=0.003, t_dead=SEARCH_END)
                else:
                    # random rotation kick + jitter
                    cur = best_x + rng.randn(n, 4) * 0.10
                    cur = _norm(cur)
                    A = rng.randn(4, 4) * 0.06
                    A = A - A.T
                    Q, _ = np.linalg.qr(np.eye(4) + A)
                    cur = cur @ Q.T
                    cur = _norm(cur)
                    cur_b, cur_v = _adam(cur, steps=3000, T0=0.015, T1=0.0006, lr0=0.014, lr1=0.001,
                                         rng=rng, noise0=0.002, t_dead=SEARCH_END)
        except Exception:
            continue
        if cur_v < best_val - 1e-12:
            if cur_v < best_val - 2e-5 and time.time() < DEADLINE - 42:
                try:
                    pol, pv = _lp_polish(cur_b, seconds=2.0, t_dead=DEADLINE - 32)
                    if pv < best_val:
                        best_x, best_val = pol, pv
                    else:
                        best_x, best_val = cur_b, cur_v
                except Exception:
                    best_x, best_val = cur_b, cur_v
            else:
                best_x, best_val = cur_b, cur_v
            if best_val < 0.84:
                break
    remain = DEADLINE - time.time() - 1.5
    if remain > 3:
        try:
            px, pv = _lp_polish(best_x, seconds=min(remain, 30.0), t_dead=DEADLINE - 0.8)
            if pv < best_val:
                best_x, best_val = px, pv
            else:
                # keep squeezed version if not worse beyond tolerance
                if pv <= best_val + 5e-12:
                    best_x = px
        except Exception:
            pass
    best_x = _norm(best_x)
    return best_x
