import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, radius_init=1e-3, max_c=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > max_c:
            idx = np.argpartition(c, -max_c)[-max_c:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),
                           (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_refine(X, steps, T0, T1, lr0, lr1, mu=0.9, idx=None):
    n, d = X.shape
    if idx is None:
        idx_arr = np.arange(n)
    else:
        idx_arr = np.asarray(idx, dtype=int)
    k = len(idx_arr)
    V = np.zeros((k, d))
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = G.max()
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        Grad = W[idx_arr] @ X
        dot = np.sum(Grad * X[idx_arr], axis=1, keepdims=True)
        Gt = Grad - dot * X[idx_arr]
        V = mu * V + Gt
        X[idx_arr] = X[idx_arr] - lr * V
        X[idx_arr] /= np.linalg.norm(X[idx_arr], axis=1, keepdims=True)
    return X


def refine_k_joint(others, Xk, steps=350, T0=0.008, T1=0.0005, lr0=0.20, lr1=0.01, mu=0.85):
    Xk = Xk.copy()
    k = Xk.shape[0]
    V = np.zeros_like(Xk)

    def joint_max(Z):
        a = float((Z @ others.T).max())
        if k > 1:
            G = Z @ Z.T
            np.fill_diagonal(G, -2.0)
            b = float(G.max())
            return a if a > b else b
        return a

    best_val = joint_max(Xk)
    best_Xk = Xk.copy()
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        D = Xk @ others.T
        Gk = Xk @ Xk.T
        if k > 1:
            np.fill_diagonal(Gk, -10.0)
            mx = D.max()
            mg = Gk.max()
            mx = mx if mx > mg else mg
        else:
            mx = float(D.max())
            Gk = np.zeros((1, 1))
        Wd = np.exp((D - mx) / T)
        if k > 1:
            Wm = np.exp((Gk - mx) / T)
            np.fill_diagonal(Wm, 0.0)
            tot = Wd.sum() + Wm.sum()
            if not np.isfinite(tot) or tot == 0:
                break
            Wd /= tot
            Wm /= tot
            Grad = Wd @ others + Wm @ Xk
        else:
            tot = Wd.sum()
            if not np.isfinite(tot) or tot == 0:
                break
            Wd /= tot
            Grad = Wd @ others
        dot = np.sum(Grad * Xk, axis=1, keepdims=True)
        Gt = Grad - dot * Xk
        V = mu * V + Gt
        Xk = Xk - lr * V
        Xk /= np.linalg.norm(Xk, axis=1, keepdims=True)
        if (s % 10 == 0) or (s == steps - 1):
            v = joint_max(Xk)
            if v < best_val:
                best_val = v
                best_Xk = Xk.copy()
    return best_Xk, float(best_val)


def find_k_holes(others, rng, k, n_samples=3500):
    d = others.shape[1]
    C = rng.normal(size=(n_samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dm = (C @ others.T).max(axis=1)
    if k == 1:
        idx = np.argpartition(dm, 3)[:3]
        bv = 1e9
        bx = None
        for j in idx:
            X0 = C[int(j)][None, :].copy()
            RX, rv = refine_k_joint(others, X0, steps=300, T0=0.006, T1=0.0003,
                                   lr0=0.18, lr1=0.006, mu=0.8)
            if rv < bv:
                bv = rv
                bx = RX
        return bx
    else:
        top_n = min(40, n_samples)
        top_idx = np.argpartition(dm, top_n)[:top_n]
        topC = C[top_idx]
        n_trials = 3 if k == 2 else 2
        best_val = 1e9
        best_X = None
        for _ in range(n_trials):
            sel = rng.choice(top_n, size=k, replace=False)
            X0 = topC[sel].copy()
            X0 += 0.02 * rng.normal(size=X0.shape)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            RX, rv = refine_k_joint(others, X0, steps=350 if k == 2 else 420,
                                   T0=0.008, T1=0.0005, lr0=0.22, lr1=0.01, mu=0.85)
            if rv < best_val:
                best_val = rv
                best_X = RX
        return best_X


def solve():
    t0 = time.time()
    BUDGET = 168.0
    EXPL_T = 120.0
    rng = np.random.default_rng(0)
    n, d = 95, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (95, 13):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (95, 13):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        tmp = rng.normal(size=(95, 13))
        cands.append(tmp / np.linalg.norm(tmp, axis=1, keepdims=True))
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    cur = x_best.copy()
    cur_val = best_val

    it = 0
    while time.time() - t0 < EXPL_T:
        it += 1
        if time.time() - t0 > BUDGET - 38:
            break
        remain = EXPL_T - (time.time() - t0)
        if remain < 1.0:
            break
        r = rng.random()
        try:
            if r < 0.45:
                # joint worst-point hole reinsertion, k=1..3
                rr = rng.random()
                if rr < 0.50:
                    k = 1
                elif rr < 0.80:
                    k = 2
                else:
                    k = 3
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                pm = G.max(axis=1)
                order = np.argsort(-pm)
                idx = [int(order[0])]
                if k > 1:
                    pool = order[1:12]
                    for _ in range(k - 1):
                        if rng.random() < 0.6:
                            c = int(pool[int(rng.integers(0, len(pool)))])
                            t = 0
                            while c in idx and t < 6:
                                c = int(pool[int(rng.integers(0, len(pool)))])
                                t += 1
                            if c in idx:
                                c = int(rng.integers(0, n))
                                while c in idx:
                                    c = int(rng.integers(0, n))
                            idx.append(c)
                        else:
                            c = int(rng.integers(0, n))
                            while c in idx:
                                c = int(rng.integers(0, n))
                            idx.append(c)
                else:
                    if rng.random() < 0.20:
                        idx = [int(order[int(rng.integers(0, 10))])]
                    if rng.random() < 0.12:
                        idx = [int(rng.integers(0, n))]
                idx_arr = np.array(idx, dtype=int)
                mask = np.ones(n, dtype=bool)
                mask[idx_arr] = False
                others = cur[mask]
                ns = 4000 if k == 1 else 3000
                Xk = find_k_holes(others, rng, k, n_samples=ns)
                if Xk is None:
                    continue
                Xt = cur.copy()
                Xt[idx_arr] = Xk
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                v0 = max_cos(Xt)
                if v0 > cur_val + 0.07:
                    continue
                if v0 < cur_val + 0.05:
                    Xt = smooth_refine(Xt, steps=320, T0=0.008, T1=0.0008,
                                       lr0=0.12, lr1=0.01, mu=0.9, idx=None)
                    v0 = max_cos(Xt)
                    if v0 > cur_val + 0.07:
                        continue
                else:
                    continue
                Xp = lp_polish(Xt, seconds=min(1.5, max(0.7, remain * 0.10)), radius_init=0.008)
                v = max_cos(Xp)
                if v < best_val - 1e-12:
                    x_best = Xp.copy(); best_val = v
                    cur = Xp.copy(); cur_val = v
                elif v < cur_val:
                    cur = Xp.copy(); cur_val = v
                else:
                    if rng.random() < 0.05 and v < cur_val + 0.01:
                        cur = Xp.copy(); cur_val = v
                    if rng.random() < 0.10:
                        cur = x_best.copy(); cur_val = best_val
            elif r < 0.75:
                # coherent cap / cluster rearrangement around worst point
                G = cur @ cur.T
                np.fill_diagonal(G, -2.0)
                pm = G.max(axis=1)
                order = np.argsort(-pm)
                if rng.random() < 0.80:
                    seed = int(order[int(rng.integers(0, 5))])
                else:
                    seed = int(rng.integers(0, n))
                k = int(rng.integers(7, 15))
                ds = cur @ cur[seed]
                block = np.argsort(-ds)[:k]
                Xt = cur.copy()
                if rng.random() < 0.5:
                    # coherent shift: move cap together
                    shift = 0.16 * rng.normal(size=(d,))
                    Xt[block] += shift + 0.07 * rng.normal(size=(k, d))
                else:
                    Xt[block] += 0.22 * rng.normal(size=(k, d))
                Xt[block] /= np.linalg.norm(Xt[block], axis=1, keepdims=True)
                Xt = smooth_refine(Xt, steps=520, T0=0.014, T1=0.001,
                                   lr0=0.28, lr1=0.015, mu=0.9, idx=block)
                v0 = max_cos(Xt)
                if v0 > cur_val + 0.08:
                    continue
                # light global settle
                Xt = smooth_refine(Xt, steps=180, T0=0.005, T1=0.0008,
                                   lr0=0.08, lr1=0.008, mu=0.9, idx=None)
                v0 = max_cos(Xt)
                if v0 > cur_val + 0.07:
                    continue
                Xp = lp_polish(Xt, seconds=min(1.8, max(0.9, remain * 0.13)), radius_init=0.008)
                v = max_cos(Xp)
                if v < best_val - 1e-12:
                    x_best = Xp.copy(); best_val = v
                    cur = Xp.copy(); cur_val = v
                elif v < cur_val:
                    cur = Xp.copy(); cur_val = v
                else:
                    if rng.random() < 0.06 and v < cur_val + 0.008:
                        cur = Xp.copy(); cur_val = v
                    if rng.random() < 0.12:
                        cur = x_best.copy(); cur_val = best_val
            else:
                # global shake + smooth
                sig = float(rng.choice(np.array([0.03, 0.07, 0.15, 0.28])))
                Xt = cur + sig * rng.normal(size=(n, d))
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                Xt = smooth_refine(Xt, steps=480, T0=0.013, T1=0.001,
                                   lr0=0.15, lr1=0.010, mu=0.9, idx=None)
                v0 = max_cos(Xt)
                if v0 > cur_val + 0.08:
                    continue
                Xp = lp_polish(Xt, seconds=min(1.5, max(0.8, remain * 0.12)), radius_init=0.006)
                v = max_cos(Xp)
                if v < best_val - 1e-12:
                    x_best = Xp.copy(); best_val = v
                    cur = Xp.copy(); cur_val = v
                elif v < cur_val:
                    cur = Xp.copy(); cur_val = v
                else:
                    if rng.random() < 0.05 and v < cur_val + 0.01:
                        cur = Xp.copy(); cur_val = v
                    if rng.random() < 0.12:
                        cur = x_best.copy(); cur_val = best_val
        except Exception:
            continue

    remain = BUDGET - (time.time() - t0)
    final_sec = float(np.clip(remain - 2.0, 5.0, 40.0))
    if final_sec > 1.0:
        try:
            x_best = lp_polish(x_best, seconds=final_sec * 0.6, radius_init=1e-3)
            rest = BUDGET - (time.time() - t0) - 1.0
            if rest > 2.0:
                x_best = lp_polish(x_best, seconds=min(rest, 18.0), radius_init=5e-4)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
