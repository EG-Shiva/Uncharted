import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())

def _lp_brief(X, t_end, max_iters=4, radius=5e-4):
    # very brief trust-region LP, capped constraints
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X, _max_cos(X)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    iu = np.triu_indices(n, 1)
    it = 0
    while it < max_iters and time.time() < t_end - 0.5 and radius > 1e-12:
        it += 1
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        if int(sel.sum()) > 1000:
            idx = np.argpartition(-c, 1000)[:1000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        if int(sel.sum()) == 0:
            break
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 3.0})
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
            radius = min(radius * 1.5, 0.005)
        else:
            radius *= 0.5
    return X, best

def _optimize_Ft(X, t, steps, lr, B, a, delta, t_deadline, track_best_fn):
    n, d = X.shape
    m_ad = np.zeros_like(X)
    v_ad = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    thresh = t - delta
    for k in range(steps):
        if time.time() > t_deadline - 1.0:
            break
        # decay lr slightly
        frac = k / max(1, steps - 1)
        lr_k = lr * (1.0 - 0.5 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mi = G.max(axis=1)  # (n,)
        # weights for smooth lse
        # E = exp(B*(G - mi[:,None]))
        D = B * (G - mi[:, None])
        # clip lower to avoid underflow warning (not needed)
        E = np.exp(D, out=np.empty_like(G))
        # diagonal is 0 already
        S = E.sum(axis=1)  # (n,)
        S = np.maximum(S, 1e-300)
        lse = mi + np.log(S) / B
        # track global best using true max
        gmax = float(mi.max())
        track_best_fn(X, gmax)
        u = a * (lse - t)
        # softplus stable
        sp = np.where(u > 20, u, np.log1p(np.exp(np.clip(u, -60, 20))))
        sig = 1.0 / (1.0 + np.exp(-np.clip(u, -60, 60)))
        d1 = 2.0 * sp * sig * a
        diff2 = lse - thresh
        d2 = np.where(diff2 < 0, 2.0 * diff2, 0.0)
        g = d1 + d2  # (n,)
        W = E / S[:, None]
        A = g[:, None] * W
        Grad = (A + A.T) @ X
        Gt = Grad - (np.sum(Grad * X, axis=1, keepdims=True)) * X
        # Adam
        b1t *= b1; b2t *= b2
        m_ad *= b1; m_ad += (1 - b1) * Gt
        v_ad *= b2; v_ad += (1 - b2) * (Gt * Gt)
        mhat = m_ad / (1 - b1t)
        vhat = v_ad / (1 - b2t)
        X = X - lr_k * mhat / (np.sqrt(vhat) + eps)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        if not np.all(np.isfinite(X)):
            break
    return X

def solve():
    import time
    t0 = time.time()
    deadline = t0 + 170.0
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
    best_val = 1e9
    best = cands[0]
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    Xbest = best.copy()
    Gbest = best_val

    # tracker closure
    def make_tracker():
        def track(X, gmax):
            nonlocal Xbest, Gbest
            if gmax < Gbest - 1e-13 and np.all(np.isfinite(X)):
                # copy normalized
                Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
                # re-evaluate to be safe
                v = _max_cos(Xn)
                if v < Gbest:
                    Gbest = v
                    Xbest = Xn.copy()
        return track
    tracker = make_tracker()

    t_start = 0.16826349
    # if current best already below start, start from it
    if Gbest < t_start:
        t_start = Gbest - 1e-9
    step_list = [0.00025] * 8  # per spec 0.00025-0.0005
    # use B=300, a=80, delta=0.002 per spec
    B, a, delta = 300.0, 80.0, 0.002
    Xcur = best.copy()
    rng = np.random.default_rng(0)
    for stage, st in enumerate(step_list):
        if time.time() > deadline - 25:
            break
        t_cur = t_start - stage * st
        # warm start: add tiny noise after first stage to help topology change (deterministic)
        if stage > 0:
            Xcur = Xcur + 0.0015 * rng.normal(size=Xcur.shape)
            Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
        # steps: more on early stages
        steps = 2200 if stage < 3 else 1800
        # shrink steps if low on time
        if time.time() > deadline - 45:
            steps = 900
        lr = 0.003 if stage < 4 else 0.002
        Xcur = _optimize_Ft(Xcur, t_cur, steps, lr, B, a, delta, deadline - 12, tracker)
        # brief LP polish for acceptance test
        if time.time() < deadline - 14:
            try:
                Xp, Vp = _lp_brief(Xcur, deadline - 8, max_iters=2, radius=4e-4)
            except Exception:
                Xp, Vp = Xcur, _max_cos(Xcur)
        else:
            Xp, Vp = Xcur, _max_cos(Xcur)
        # acceptance: max_i m_i <= t + 1e-5
        if Vp <= t_cur + 1e-5:
            # accept stage: keep polished
            Xcur = Xp
            # tracker already updated if better; also force update check
            tracker(Xcur, Vp)
        else:
            # not accepted: still keep Xp as warm start if it is not much worse,
            # otherwise revert to Xbest to avoid drift
            if Vp < _max_cos(Xcur) + 1e-12:
                Xcur = Xp
            # if drifted too high, pull back toward best
            cur_v = _max_cos(Xcur)
            if cur_v > Gbest + 0.002:
                Xcur = 0.7 * Xcur + 0.3 * Xbest
                Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
            tracker(Xcur, _max_cos(Xcur))
        # early stop if we improved notably
        if Gbest < 0.1675 and time.time() > deadline - 30:
            break

    # final squeeze: direct global soft-max descent on best
    if time.time() < deadline - 12:
        Xf = Xbest.copy()
        # short global lse annealing
        m_ad = np.zeros_like(Xf); v_ad = np.zeros_like(Xf)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1t, b2t = 1.0, 1.0
        nsteps = 1500
        if time.time() > deadline - 30:
            nsteps = 600
        for k in range(nsteps):
            if time.time() > deadline - 6:
                break
            frac = k / max(1, nsteps - 1)
            beta = 120.0 * (400.0 / 120.0) ** frac
            lr_k = 0.002 * (0.0003 / 0.002) ** frac
            G = Xf @ Xf.T
            np.fill_diagonal(G, -10.0)
            gm = float(G.max())
            E = np.exp(beta * (G - gm))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if S == 0 or not np.isfinite(S):
                break
            E /= S
            Gr = 2.0 * (E @ Xf)
            Gt = Gr - (np.sum(Gr * Xf, axis=1, keepdims=True)) * Xf
            b1t *= b1; b2t *= b2
            m_ad *= b1; m_ad += (1 - b1) * Gt
            v_ad *= b2; v_ad += (1 - b2) * (Gt * Gt)
            mhat = m_ad / (1 - b1t); vhat = v_ad / (1 - b2t)
            Xf -= lr_k * mhat / (np.sqrt(vhat) + eps)
            Xf /= np.linalg.norm(Xf, axis=1, keepdims=True)
            if k % 150 == 0:
                tracker(Xf, _max_cos(Xf))
        tracker(Xf, _max_cos(Xf))

    # final brief LP if time remains
    if time.time() < deadline - 8:
        try:
            Xp, Vp = _lp_brief(Xbest, deadline - 2, max_iters=3, radius=3e-4)
            if Vp < Gbest and np.all(np.isfinite(Xp)):
                Xbest, Gbest = Xp, Vp
        except Exception:
            pass

    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    if not np.all(np.isfinite(Xbest)):
        Xbest = best
    return Xbest
