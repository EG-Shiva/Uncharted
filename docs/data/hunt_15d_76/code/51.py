import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _tangent_bases(X, rng):
    n, d = X.shape
    dm = d - 1
    Bs = np.empty((n, d, dm), dtype=np.float64)
    for i in range(n):
        x = X[i]
        R = rng.standard_normal(size=(d, dm))
        # project out radial
        R = R - np.outer(x, (x @ R))
        Q, _ = np.linalg.qr(R)
        # ensure tangent (fix sign drift)
        Q = Q - np.outer(x, (x @ Q))
        Q, _ = np.linalg.qr(Q)
        Bs[i] = Q
    return Bs

def _floppy_modes(X, slack=1e-7, nmodes=20, pmax=1200):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu0 = np.triu_indices(n, 1)
    c = G[iu0]
    order = np.argsort(-c)  # tightest first
    sel = []
    for k in order:
        if c[k] >= m - slack:
            sel.append(k)
            if len(sel) >= pmax:
                break
        else:
            break
    if len(sel) == 0:
        return None, None, m, None
    sel = np.array(sel, dtype=int)
    I = iu0[0][sel]; J = iu0[1][sel]
    cc = c[sel]
    p = len(sel)
    rng = np.random.default_rng(0)
    Bs = _tangent_bases(X, rng)
    dm = d - 1
    Nt = n * dm
    Jt = np.zeros((p, Nt), dtype=np.float64)
    # blocks: (x_j - c*x_i) @ B_i
    for k in range(p):
        i = int(I[k]); j = int(J[k]); ck = float(cc[k])
        vi = X[j] - ck * X[i]
        vj = X[i] - ck * X[j]
        Jt[k, i*dm:(i+1)*dm] = vi @ Bs[i]
        Jt[k, j*dm:(j+1)*dm] = vj @ Bs[j]
    # dense SVD, smallest vectors = tail of Vt
    try:
        _, S, Vt = np.linalg.svd(Jt, full_matrices=False)
    except Exception:
        return None, Bs, m, (I, J, cc)
    # take lowest nmodes
    if Vt.shape[0] >= nmodes:
        modes = Vt[-nmodes:, :]
    else:
        modes = Vt
    return modes, Bs, m, (I, J, cc)

def _modes_to_disp(modes_sel, Bs, coeffs):
    # modes_sel: (k,Nt), coeffs: (k,) -> D (n,d)
    n, d, dm = Bs.shape[0], Bs.shape[1], Bs.shape[2]
    Nt = n * dm
    combo = coeffs @ modes_sel  # (Nt,)
    D = np.empty((n, d), dtype=np.float64)
    for i in range(n):
        seg = combo[i*dm:(i+1)*dm]
        D[i] = Bs[i] @ seg
    return D

def _anneal_lse_adam(X, steps, b0, b1, lr, t0, deadline):
    n, d = X.shape
    Mt = np.zeros_like(X)
    Vt = np.zeros_like(X)
    b1m, b2m = 0.9, 0.999
    eps = 1e-8
    for s in range(steps):
        if (s & 31) == 0 and time.time() - t0 > deadline:
            break
        f = s / max(1, steps - 1)
        beta = b0 + (b1 - b0) * f
        T = 1.0 / beta
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        E = np.exp((G - mx) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.finite(S) or S <= 0:
            continue
        Gr = (2.0 / S) * (E @ X)
        dot = (Gr * X).sum(axis=1, keepdims=True)
        Gt = Gr - dot * X
        Mt = b1m * Mt + (1.0 - b1m) * Gt
        Vt = b2m * Vt + (1.0 - b2m) * (Gt * Gt)
        mhat = Mt / (1.0 - b1m ** (s + 1))
        vhat = Vt / (1.0 - b2m ** (s + 1))
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # clip per-point step to avoid blowup
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        # limit max per-point move to 0.05
        scale = np.minimum(1.0, 0.05 / sn)
        step = step * scale
        X = X - step
        X = _norm(X)
    return X

def _lp_polish(X, seconds, t0_global, deadline_global):
    # trust-region LP, capped by seconds and global deadline
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return X, _max_cos(X)
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    start = time.time()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() - t0_global < deadline_global:
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        c = Gfull[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) == 0:
            radius *= 0.5
            continue
        # cap active size for speed
        idx = np.where(sel)[0]
        if len(idx) > 1500:
            # keep tightest 1500
            o = np.argsort(-c[idx])[:1500]
            idx = idx[o]
        I = iu[0][idx]; J = iu[1][idx]
        csel = c[idx]
        p = len(idx)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
        try:
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
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X, best

def solve():
    t0 = time.time()
    DEADLINE = 165.0
    LP_BUDGET = 20.0
    lp_used_start = time.time()
    lp_used = 0.0
    rng = np.random.default_rng(42)
    try:
        X0 = np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            X0 = np.load("record.npy").astype(np.float64)
        except Exception:
            X0 = rng.standard_normal(size=(76, 15))
    if X0.shape != (76, 15):
        X0 = rng.standard_normal(size=(76, 15))
    X0 = _norm(X0)
    best = _max_cos(X0)
    Xbest = X0.copy()

    # contact-Jacobian floppy modes
    try:
        modes, Bs, m0, act = _floppy_modes(Xbest, slack=1e-7, nmodes=20, pmax=1200)
    except Exception:
        modes, Bs = None, None
    if modes is None or Bs is None:
        # fallback: no modes, just anneal
        X1 = _anneal_lse_adam(Xbest.copy(), 1500, 40, 400, 0.008, t0, DEADLINE - 25)
        v1 = _max_cos(X1)
        if v1 < best:
            Xbest, best = X1, v1
        Xbest = _norm(Xbest)
        return Xbest

    nmodes = modes.shape[0]
    # rattlers: contacts with 1e-7 slack
    try:
        G0 = Xbest @ Xbest.T
        np.fill_diagonal(G0, -2.0)
        mm = float(G0.max())
        cnt = (G0 >= mm - 1e-7).sum(axis=1)
        rattlers = np.where(cnt == 0)[0]
    except Exception:
        rattlers = np.array([], dtype=int)

    amps = [0.03, 0.06, 0.12, 0.24]
    kchoices = [5, 8, 12, 20]
    # precompute tail ordering: modes already lowest-last; modes[-k:] are lowest k
    trial = 0
    max_trials = 40
    while trial < max_trials and time.time() - t0 < DEADLINE - 8:
        k = kchoices[trial % len(kchoices)]
        k = min(k, nmodes)
        amp = amps[(trial // 1) % len(amps)]
        # random combo of lowest k
        modes_sel = modes[-k:, :]
        coeffs = rng.standard_normal(size=(k,))
        nrm = float(np.linalg.norm(coeffs)) + 1e-12
        coeffs /= nrm
        D = _modes_to_disp(modes_sel, Bs, coeffs)
        # normalize D to unit Fro then scale to amp
        dn = float(np.linalg.norm(D)) + 1e-12
        D = D / dn * amp
        X = Xbest + D
        # optional larger motion to rattlers
        if len(rattlers) > 0 and (trial % 2 == 0):
            extra = 0.10 if (trial % 4 == 0) else 0.05
            for ri in rattlers:
                g = rng.standard_normal(size=(15,))
                g = g - (g @ X[ri]) * X[ri]
                gn = float(np.linalg.norm(g)) + 1e-12
                g /= gn
                X[ri] = X[ri] + extra * g
        X = _norm(X)
        # fast LSE-Adam anneal beta 40->400 ~1500 steps
        # use slightly fewer steps for large amp trials to save time
        steps = 1500 if (time.time() - t0 < DEADLINE - 40) else 800
        lr = 0.008
        X = _anneal_lse_adam(X, steps, 40, 400, lr, t0, DEADLINE - 5)
        v = _max_cos(X)
        # decide LP polish: only if promising and budget left
        polished = False
        if v < best + 0.002 and lp_used < LP_BUDGET and time.time() - t0 < DEADLINE - 10:
            sec = min(4.0, LP_BUDGET - lp_used, DEADLINE - 5 - (time.time() - t0))
            if sec > 0.5:
                s_lp = time.time()
                Xp, vp = _lp_polish(X, sec, t0, DEADLINE - 2)
                lp_used += time.time() - s_lp
                X, v = Xp, vp
                polished = True
        if v < best:
            best = v
            Xbest = X.copy()
            # refresh modes around new best if big improvement? keep old for speed
            # update rattlers
            try:
                G1 = Xbest @ Xbest.T
                np.fill_diagonal(G1, -2.0)
                mm1 = float(G1.max())
                cnt1 = (G1 >= mm1 - 1e-7).sum(axis=1)
                rattlers = np.where(cnt1 == 0)[0]
            except Exception:
                pass
        trial += 1
        # time-check already inside anneal every 32 iters

    # final capped LP polish on best if time and budget remain
    if time.time() - t0 < DEADLINE - 5 and lp_used < LP_BUDGET:
        sec = min(LP_BUDGET - lp_used, DEADLINE - 2 - (time.time() - t0), 8.0)
        if sec > 0.5:
            try:
                Xp, vp = _lp_polish(Xbest, sec, t0, DEADLINE - 1)
                if vp < best:
                    Xbest, best = Xp, vp
            except Exception:
                pass

    Xbest = _norm(np.nan_to_num(Xbest))
    return Xbest
