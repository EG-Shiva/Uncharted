import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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

def get_duals(x, radius=0.002):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    c = (x @ x.T)[iu]
    m = float(c.max())
    sel = c > m - 1e-9
    I, J = iu[0][sel], iu[1][sel]
    csel = c[sel]
    p = len(I)
    if p == 0:
        return I, J, None, m
    nv = n * d + 1
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
    vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
    A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
    obj = np.zeros(nv); obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
    except Exception:
        return I, J, None, m
    if res.status != 0:
        return I, J, None, m
    marg = None
    try:
        marg = np.asarray(res.ineqlin.marginals, dtype=float).ravel()
    except Exception:
        try:
            marg = np.asarray(res["ineqlin"]["marginals"], dtype=float).ravel()
        except Exception:
            marg = None
    if marg is not None and len(marg) != p:
        try:
            marg = np.asarray(marg).ravel()
            if len(marg) != p:
                marg = None
        except Exception:
            marg = None
    return I, J, marg, m

def surgery_adam(X0, S_arr, targets, steps_per=700, lr=0.004, scale=1e-4):
    n, d = X0.shape
    s = len(S_arr)
    g2s = np.full(n, -1, dtype=int)
    g2s[S_arr] = np.arange(s)
    iu0, iu1 = np.triu_indices(n, 1)
    mask = np.isin(iu0, S_arr) | np.isin(iu1, S_arr)
    PI = iu0[mask]; PJ = iu1[mask]
    sI = g2s[PI]; sJ = g2s[PJ]
    mI = sI >= 0; mJ = sJ >= 0
    Xcur = X0.copy()
    Y = X0[S_arr].copy()
    m = np.zeros((s, d)); v = np.zeros((s, d))
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    cnt = 0
    for t in targets:
        for it in range(steps_per):
            cnt += 1
            Xcur[S_arr] = Y
            Xi = Xcur[PI]; Xj = Xcur[PJ]
            dots = np.sum(Xi * Xj, axis=1)
            z = (dots - t) / scale
            sig = np.empty_like(z)
            pos = z >= 0
            sig[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
            ez = np.exp(z[~pos])
            sig[~pos] = ez / (1.0 + ez)
            w = sig / scale
            WJ = w[:, None] * Xcur[PJ]
            WI = w[:, None] * Xcur[PI]
            Gs = np.zeros((s, d))
            if np.any(mI):
                np.add.at(Gs, sI[mI], WJ[mI])
            if np.any(mJ):
                np.add.at(Gs, sJ[mJ], WI[mJ])
            dot = np.sum(Gs * Y, axis=1, keepdims=True)
            Gt = Gs - dot * Y
            m = beta1 * m + (1 - beta1) * Gt
            v = beta2 * v + (1 - beta2) * (Gt * Gt)
            mhat = m / (1 - beta1 ** cnt)
            vhat = v / (1 - beta2 ** cnt)
            delta = lr * mhat / (np.sqrt(vhat) + eps)
            # clip extreme step
            dn = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12
            cap = 0.02
            fac = np.minimum(1.0, cap / (dn + 1e-12))
            delta = delta * fac
            Y = Y - delta
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            if it % 200 == 199:
                Xcur[S_arr] = Y
                md = float(np.sum(Xcur[PI] * Xcur[PJ], axis=1).max())
                if md < t + 2e-5:
                    break
    Xcur[S_arr] = Y
    Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
    return Xcur

def solve():
    t0 = time.time()
    DEADLINE = t0 + 150.0
    HARD = t0 + 172.0
    rng = np.random.default_rng(0)
    RECORD = 0.244097856927
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    base = None
    if xb is not None and xb.shape == (95, 13):
        base = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    elif xr is not None and xr.shape == (95, 13):
        base = xr / np.linalg.norm(xr, axis=1, keepdims=True)
    else:
        tmp = rng.normal(size=(95, 13))
        base = tmp / np.linalg.norm(tmp, axis=1, keepdims=True)
    fallback = base.copy()
    if xr is not None and xr.shape == (95, 13):
        fallback = xr / np.linalg.norm(xr, axis=1, keepdims=True)
    best_val = max_cos(base)
    x_best = base.copy()
    # one LP polish
    remain = DEADLINE - time.time()
    if remain > 4:
        x_best = lp_polish(x_best, min(4.0, remain - 1.0), t_deadline=DEADLINE - 1.0)
        v = max_cos(x_best)
        if v < best_val:
            best_val = v
        else:
            # keep original if polish worsened (should not)
            pass
    # extract duals
    try:
        CI, CJ, marg, mcur = get_duals(x_best, radius=0.002)
    except Exception:
        CI, CJ, marg, mcur = None, None, None, best_val
    if CI is None or len(CI) == 0:
        # fallback: define contacts by slack
        g = x_best @ x_best.T
        np.fill_diagonal(g, -2.0)
        mcur = float(g.max())
        iu0, iu1 = np.triu_indices(95, 1)
        c = g[iu0, iu1]
        sel = c > mcur - 1e-9
        CI, CJ = iu0[sel], iu1[sel]
        marg = None
    p = len(CI)
    if marg is None:
        marg = np.zeros(p)
        # use random order
        order = rng.permutation(p)
    else:
        order = np.argsort(np.abs(marg))  # low stress first
    low_n = min(max(150, p // 2), p)
    low_pool = order[:low_n]
    targets = [0.2438, 0.2433, 0.2425]
    attempt = 0
    # surgery loop
    while time.time() < DEADLINE - 6.0 and attempt < 25:
        attempt += 1
        if time.time() > HARD - 30:
            break
        kcon = int(rng.integers(10, 31))
        # sample kcon from low_pool with sliding offset for diversity
        # mix random sample and ordered slice
        if rng.random() < 0.5:
            start_idx = (attempt * 37) % max(1, low_n - kcon)
            chosen = low_pool[start_idx:start_idx + kcon]
            # add a few random ones
            if len(chosen) < kcon:
                extra = rng.choice(low_pool, size=kcon - len(chosen), replace=False)
                chosen = np.concatenate([chosen, extra])
        else:
            chosen = rng.choice(low_pool, size=min(kcon, low_n), replace=False)
        selI = CI[chosen]; selJ = CJ[chosen]
        S = np.unique(np.concatenate([selI, selJ]))
        # enforce 8-20 points
        tries = 0
        while len(S) < 8 and tries < 10:
            tries += 1
            need = 8 - len(S)
            add = rng.choice(low_pool, size=min(need + 2, low_n), replace=False)
            selI2 = CI[add]; selJ2 = CJ[add]
            chosen = np.unique(np.concatenate([chosen, add]))
            # limit to 30 contacts
            if len(chosen) > 30:
                chosen = rng.choice(chosen, size=30, replace=False)
            selI = CI[chosen]; selJ = CJ[chosen]
            S = np.unique(np.concatenate([selI, selJ]))
        tries = 0
        while len(S) > 20 and tries < 20:
            tries += 1
            # drop random contact
            drop = int(rng.integers(0, len(chosen)))
            chosen = np.delete(chosen, drop)
            if len(chosen) < 10:
                break
            selI = CI[chosen]; selJ = CJ[chosen]
            S = np.unique(np.concatenate([selI, selJ]))
        if len(S) < 8 or len(S) > 22 or len(chosen) < 10:
            continue
        # frozen feasibility
        steps = 600 if attempt < 8 else 450
        # adapt lr slightly per attempt
        lr = 0.004 if attempt % 2 == 0 else 0.006
        try:
            Xmid = surgery_adam(x_best, S, targets, steps_per=steps, lr=lr, scale=1e-4)
        except Exception:
            continue
        if time.time() > DEADLINE - 5.0:
            break
        # unfreeze LP polish short
        try:
            Xpol = lp_polish(Xmid, 2.2, t_deadline=DEADLINE - 1.0)
        except Exception:
            continue
        try:
            nv = max_cos(Xpol)
        except Exception:
            continue
        if nv < best_val - 1e-12 and nv < RECORD:
            x_best = Xpol
            best_val = nv
            # refresh duals around new basin
            try:
                CI2, CJ2, marg2, mcur2 = get_duals(x_best, radius=0.002)
                if len(CI2) > 0:
                    CI, CJ, marg = CI2, CJ2, marg2
                    p = len(CI)
                    if marg is None:
                        order = rng.permutation(p)
                        low_n = min(max(150, p // 2), p)
                        low_pool = order[:low_n]
                    else:
                        order = np.argsort(np.abs(marg))
                        low_n = min(max(150, p // 2), p)
                        low_pool = order[:low_n]
            except Exception:
                pass
            # early exit if big win
            if best_val < 0.2425:
                break
        # also allow neutral moves that change topology but keep record? per spec keep only if beats record, so discard
        if time.time() > DEADLINE:
            break
    # final short polish if time permits and we improved, else return fallback/best
    if best_val < RECORD:
        # extra squeeze
        if time.time() < HARD - 8:
            try:
                x_best = lp_polish(x_best, min(5.0, HARD - time.time() - 2.0))
                best_val = max_cos(x_best)
            except Exception:
                pass
        if best_val < RECORD:
            return (x_best / np.linalg.norm(x_best, axis=1, keepdims=True))
    # fall back to record/best (must return valid)
    ret = x_best if max_cos(x_best) <= max_cos(fallback) else fallback
    ret = ret / np.linalg.norm(ret, axis=1, keepdims=True)
    return ret
