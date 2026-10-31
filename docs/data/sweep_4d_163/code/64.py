import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _nrm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth(x, iters, T0, T1, lr0, lr1):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _void_vec(x_fixed, rng, n_samp=1200):
    d = x_fixed.shape[1]
    cand = rng.normal(size=(n_samp, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sc = cand @ x_fixed.T
    mx = sc.max(axis=1)
    i = int(np.argmin(mx))
    v = cand[i].copy()
    for _ in range(18):
        dots = x_fixed @ v
        dm = float(dots.max())
        w = np.exp(50.0 * (dots - dm))
        g = w @ x_fixed
        gt = g - float(np.dot(g, v)) * v
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-12:
            break
        gt /= nrm
        v = v - 0.06 * gt
        v /= np.linalg.norm(v)
    return v

_IU = None
def _lp_polish(x, seconds, r0=2e-3, cap=1500):
    global _IU
    x = _nrm(x)
    n, d = x.shape
    if _IU is None or _IU[0].shape[0] != n * (n - 1) // 2:
        _IU = np.triu_indices(n, 1)
    iu = _IU
    best = _max_cos(x)
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = c_all >= kth
            cnt2 = int(np.count_nonzero(sel))
            if cnt2 > cap + 400:
                ids = np.where(sel)[0]
                keep = ids[::2][:cap]
                ns = np.zeros_like(sel, dtype=bool)
                ns[keep] = True
                sel = ns
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _nrm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD = 170.0
    LP_TIME = 28.0
    rng = np.random.default_rng(123456)
    n, d = 163, 4
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            cc = np.asarray(np.load(fn), dtype=float)
            if cc.shape == (n, d):
                cc = np.nan_to_num(cc, nan=0.0, posinf=0.0, neginf=0.0)
                cc = _nrm(cc)
                if np.all(np.isfinite(cc)):
                    cands.append(cc)
        except Exception:
            pass
    if not cands:
        z = rng.normal(size=(n, d))
        cands.append(_nrm(z))
    best = None; best_val = 1e9
    for cc in cands:
        v = _max_cos(cc)
        if v < best_val:
            best_val = v; best = cc.copy()
    cur = best.copy(); cur_val = best_val
    # hopping phase
    hop_end = HARD - LP_TIME - 2.0
    it = 0
    Temp = 0.0025
    while time.time() - t0 < hop_end:
        it += 1
        mode = it % 8
        Y = cur.copy()
        # analyse current
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        rowmax = G.max(axis=1)
        order_worst = np.argsort(-rowmax)
        # degrees with tight threshold
        deg = (G > mcur - 2e-4).sum(axis=1)
        order_hub = np.argsort(-deg)
        # rattlers: far from max
        ratt = np.where(rowmax < mcur - 5e-4)[0]
        try:
            if mode == 0:
                sig = 0.07
                Y = Y + rng.normal(scale=sig, size=Y.shape)
                Y = _nrm(Y)
                Y = _smooth(Y, 180, 0.018, 0.002, 0.07, 0.012)
            elif mode == 1:
                k = 4
                idx = order_worst[:k]
                fixed = np.delete(Y, idx, axis=0)
                for j in idx:
                    Y[j] = _void_vec(fixed, rng, n_samp=1200)
                    fixed = np.vstack([fixed, Y[j][None, :]])
                Y = _nrm(Y)
                Y = _smooth(Y, 220, 0.02, 0.0018, 0.07, 0.01)
            elif mode == 2:
                idx = list(order_worst[:2])
                if len(ratt) > 0:
                    for r in ratt[:3]:
                        if r not in idx:
                            idx.append(int(r))
                idx = np.array(idx[:5], dtype=int)
                fixed = np.delete(Y, idx, axis=0)
                for j in idx:
                    Y[j] = _void_vec(fixed, rng, n_samp=1400)
                    fixed = np.vstack([fixed, Y[j][None, :]])
                # tiny shake rest
                mask = np.ones(n, bool); mask[idx] = False
                Y[mask] = Y[mask] + rng.normal(scale=0.008, size=(int(mask.sum()), d))
                Y = _nrm(Y)
                Y = _smooth(Y, 220, 0.016, 0.0015, 0.06, 0.01)
            elif mode == 3:
                k = 5
                idx = order_hub[:k]
                fixed = np.delete(Y, idx, axis=0)
                for j in idx:
                    Y[j] = _void_vec(fixed, rng, n_samp=1200)
                    fixed = np.vstack([fixed, Y[j][None, :]])
                Y = _nrm(Y)
                Y = _smooth(Y, 200, 0.022, 0.002, 0.075, 0.012)
            elif mode == 4:
                # patch around worst point
                i0 = int(order_worst[0])
                gi = G[i0]
                nb = np.argsort(-gi)[:5]
                idx = np.unique(np.concatenate([[i0], nb]))[:6]
                fixed = np.delete(Y, idx, axis=0)
                for j in idx:
                    Y[int(j)] = _void_vec(fixed, rng, n_samp=1000)
                    fixed = np.vstack([fixed, Y[int(j)][None, :]])
                Y = _nrm(Y)
                Y = _smooth(Y, 240, 0.02, 0.0015, 0.07, 0.009)
            elif mode == 5:
                k = 6
                idx = rng.choice(n, size=k, replace=False)
                fixed = np.delete(Y, idx, axis=0)
                for j in idx:
                    Y[int(j)] = _void_vec(fixed, rng, n_samp=1000)
                    fixed = np.vstack([fixed, Y[int(j)][None, :]])
                Y = _nrm(Y)
                Y = _smooth(Y, 200, 0.018, 0.002, 0.065, 0.011)
            elif mode == 6:
                sig = 0.13
                Y = Y + rng.normal(scale=sig, size=Y.shape)
                Y = _nrm(Y)
                Y = _smooth(Y, 260, 0.022, 0.0018, 0.075, 0.01)
            else:
                # shake overloaded neighbourhood lightly
                sig = 0.03
                Y = Y + rng.normal(scale=sig, size=Y.shape)
                Y = _nrm(Y)
                Y = _smooth(Y, 160, 0.012, 0.0012, 0.05, 0.008)
        except Exception:
            continue
        if not np.all(np.isfinite(Y)):
            continue
        v = _max_cos(Y)
        if v < best_val:
            best = Y.copy(); best_val = v
            cur = Y.copy(); cur_val = v
        else:
            # Metropolis to change basin
            if v < cur_val + 1e-4 or rng.random() < np.exp(-(v - cur_val) / max(Temp, 1e-6)):
                cur = Y.copy(); cur_val = v
            # occasionally reset to best
            if it % 25 == 0:
                cur = best.copy(); cur_val = best_val
        # decay temp
        Temp = max(0.0004, Temp * 0.998)
        # early stop if time
        if time.time() - t0 > hop_end:
            break
    # final long smooth on best before LP
    try:
        best = _smooth(best, 250, 0.01, 0.001, 0.04, 0.008)
        best_val = _max_cos(best)
    except Exception:
        pass
    remain = HARD - (time.time() - t0)
    lp_t = float(min(LP_TIME, max(6.0, remain - 1.5)))
    if lp_t > 3 and time.time() - t0 < HARD - 2:
        try:
            best = _lp_polish(best, lp_t, r0=2e-3, cap=1500)
            best_val = _max_cos(best)
        except Exception:
            pass
    # micro shake + short smooth if time left
    if HARD - (time.time() - t0) > 8:
        try:
            Y = best + rng.normal(scale=0.0025, size=best.shape)
            Y = _nrm(Y)
            Y = _smooth(Y, 120, 0.006, 0.001, 0.03, 0.007)
            v = _max_cos(Y)
            if v < best_val:
                best = Y; best_val = v
        except Exception:
            pass
        r2 = float(min(8.0, HARD - (time.time() - t0) - 0.8))
        if r2 > 2:
            try:
                best = _lp_polish(best, r2, r0=7e-4, cap=1500)
            except Exception:
                pass
    best = _nrm(best)
    return best
