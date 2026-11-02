import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _feas_stage(X0, t, steps, lr0, lr1, beta_w=250.0):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    # log schedule
    import math
    ll0 = math.log(lr0)
    ll1 = math.log(lr1)
    for k in range(steps):
        f = k / max(1, steps - 1)
        lr = math.exp(ll0 + (ll1 - ll0) * f)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        me = G.max(axis=1)  # (n,)
        s = 80.0 * (me - t)
        sc = np.clip(s, -60.0, 60.0)
        # softplus stable
        sp = np.where(s > 20.0, s, np.log1p(np.exp(sc)))
        # for s>20 sp=s correct; for very negative sc exp->0
        # fix where s>20 sc clipped but sp overwritten, fine
        sig = 1.0 / (1.0 + np.exp(-sc))
        # for large s sig->1
        sig = np.where(s > 20.0, 1.0, sig)
        pos = t - 0.002 - me
        pos = np.where(pos > 0, pos, 0.0)
        A = 2.0 * sp * sig * 80.0 - 2.0 * pos
        # softmax weights for direction
        E = np.exp(beta_w * (G - me[:, None]))
        np.fill_diagonal(E, 0.0)
        S = E.sum(axis=1, keepdims=True)
        S[S <= 0] = 1.0
        W = E / S
        WX = W @ X  # (n,d)
        AW = A[:, None] * W
        Gr = A[:, None] * WX + (AW.T @ X)
        # tangent projection
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        # adam
        b1t *= b1
        b2t *= b2
        m1 *= b1
        m1 += (1.0 - b1) * Gt
        vv *= b2
        vv += (1.0 - b2) * (Gt * Gt)
        mhat = m1 / (1.0 - b1t)
        vhat = vv / (1.0 - b2t)
        X -= lr * mhat / (np.sqrt(vhat) + eps)
        # reproject
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _lp_brief(X, t_end, max_iters=4):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 6e-4
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() < t_end - 0.4 and it < max_iters and radius > 1e-12:
        it += 1
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        if int(sel.sum()) > 1200:
            idx = np.argpartition(-c, 1200)[:1200]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.4, 0.01)
        else:
            radius *= 0.5
    return X

def solve():
    import numpy as np
    import time
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    def normed(a):
        a = np.asarray(a, dtype=float)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        return a
    cands = []
    if xb is not None and xb.shape == (76, 15):
        try:
            cands.append(normed(xb))
        except Exception:
            pass
    if xr is not None and xr.shape == (76, 15):
        try:
            cands.append(normed(xr))
        except Exception:
            pass
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        X = rng0.normal(size=(76, 15))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X
    best = cands[0]
    bv = _max_cos(best)
    for c in cands[1:]:
        v = _max_cos(c)
        if v < bv:
            best, bv = c.copy(), v
    Xcur = best.copy()
    cur_max = bv
    # continuation targets: start at bv, step down
    # use steps 0.0004 then 0.00025 fine-tuning
    step_list = [0.0004, 0.0004, 0.0004, 0.0003, 0.00025, 0.00025, 0.00025, 0.0002]
    t = float(bv)
    # first equilibrate at current t without demanding improvement
    # stages
    for si, st in enumerate(step_list):
        if time.time() > deadline - 25:
            break
        t_next = t - st
        # adam stage warm-started from Xcur
        # fewer steps early, more later if close
        steps = 1400
        if time.time() > deadline - 60:
            steps = 800
        lr0, lr1 = 0.004, 0.00035
        # slight schedule variation per stage for determinism/diversity
        if si % 2 == 1:
            lr0 = 0.003
        Xtry = _feas_stage(Xcur, t_next, steps, lr0, lr1, beta_w=250.0)
        v_try = _max_cos(Xtry)
        # brief LP polish if close to target
        if v_try <= t_next + 6e-4 and time.time() < deadline - 25:
            t_end = min(deadline - 2.0, time.time() + 4.0)
            try:
                Xpol = _lp_brief(Xtry, t_end, max_iters=3)
                v_pol = _max_cos(Xpol)
                if v_pol < v_try:
                    Xtry, v_try = Xpol, v_pol
            except Exception:
                pass
        # acceptance: max <= t_next + 1e-5
        if v_try <= t_next + 1e-5:
            Xcur = Xtry
            cur_max = v_try
            t = t_next
            if v_try < bv - 1e-12:
                best, bv = Xtry.copy(), v_try
        else:
            # try one retry with smaller lr / more steps at same target?
            # only if v_try improved over cur and is close
            if v_try < cur_max and v_try <= t_next + 4e-4:
                # accept as new cur but keep t (don't lower fully)
                Xcur = Xtry
                cur_max = v_try
                t = float(v_try)  # reset target to achieved level
                if v_try < bv:
                    best, bv = Xtry.copy(), v_try
            else:
                # shrink step and retry once
                t_half = t - st * 0.5
                if time.time() > deadline - 25:
                    break
                Xtry2 = _feas_stage(Xcur, t_half, 800, 0.002, 0.0003, beta_w=300.0)
                v2 = _max_cos(Xtry2)
                if v2 <= t_half + 1e-5:
                    Xcur, cur_max, t = Xtry2, v2, t_half
                    if v2 < bv:
                        best, bv = Xtry2.copy(), v2
                elif v2 < bv:
                    best, bv = Xtry2.copy(), v2
                    # do not advance t
                # else stay
    # final short LP polish on best if time remains
    remain = deadline - time.time()
    if remain > 7:
        secs = min(remain - 2.0, 18.0)
        t_end = time.time() + secs
        try:
            Xp = _lp_brief(best, t_end, max_iters=6)
            vp = _max_cos(Xp)
            if vp < bv:
                best, bv = Xp, vp
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    if not np.all(np.isfinite(best)):
        best = cands[0]
    return best
