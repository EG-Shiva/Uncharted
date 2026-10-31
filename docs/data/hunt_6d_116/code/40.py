import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _contacts_set(X, tol=1e-6):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    sel = np.where(c > m - tol)[0]
    s = set(zip((iu[0][sel]).tolist(), (iu[1][sel]).tolist()))
    return s, m

def _jaccard(a, b):
    if len(a) == 0 and len(b) == 0:
        return 1.0
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(union, 1)

def _tangent_bases(X):
    n, d = X.shape
    assert d == 6
    B = np.empty((n, d, 5), dtype=np.float64)
    I6 = np.eye(6)
    for i in range(n):
        x = X[i]
        k = int(np.argmax(np.abs(x)))
        idx = [j for j in range(6) if j != k]
        P = I6[:, idx].copy()  # 6x5
        # project to tangent: P - x*(x^T P)
        xtP = x @ P  # 5
        P = P - np.outer(x, xtP)
        Q, _ = np.linalg.qr(P)
        # Q is 6x5 orthonormal, orthogonal to x (up to sign)
        B[i] = Q
    return B

def _build_R(X, B, pairs):
    n = X.shape[0]
    p = len(pairs)
    R = np.zeros((p, n * 5), dtype=np.float64)
    for r, (i, j) in enumerate(pairs):
        Bi = B[i]  # 6x5
        Bj = B[j]
        ci = X[j] @ Bi  # 5
        cj = X[i] @ Bj
        R[r, i * 5:(i + 1) * 5] = ci
        R[r, j * 5:(j + 1) * 5] = cj
    return R

def _feas_adam(X0, t_target, steps, lr0, lr1):
    X = _norm(X0.copy())
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    n = X.shape[0]
    for it in range(1, steps + 1):
        f = it / float(steps)
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        # hinge
        H = G - t_target
        H[H < 0] = 0.0
        np.fill_diagonal(H, 0.0)
        H *= 2.0
        Gr = H @ X
        proj = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - proj * X
        b1p *= b1; b2p *= b2
        m *= b1; m += (1 - b1) * Gt
        v *= b2; v += (1 - b2) * Gt * Gt
        mh = m / (1 - b1p); vh = v / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        step = step - np.sum(step * X, axis=1, keepdims=True) * X
        X = X - step
        X = _norm(X)
    return X

def _softmax_adam(X0, steps, T0, T1, lr0, lr1):
    X = _norm(X0.copy())
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for it in range(1, steps + 1):
        f = it / float(steps)
        T = T0 * (T1 / T0) ** f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(np.max(G))
        E = np.exp((G - mx) / T)
        np.fill_diagonal(E, 0.0)
        s = float(np.sum(E))
        if not np.isfinite(s) or s <= 0:
            break
        E /= s
        Gr = E @ X
        proj = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - proj * X
        b1p *= b1; b2p *= b2
        m *= b1; m += (1 - b1) * Gt
        v *= b2; v += (1 - b2) * Gt * Gt
        mh = m / (1 - b1p); vh = v / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        step = step - np.sum(step * X, axis=1, keepdims=True) * X
        X = X - step
        X = _norm(X)
    return X

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.3:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1400:
            o = np.argsort(c[idx])[-1400:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool); sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]; cs = c[sel]
        p = len(I); nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            best = vv; bx = y.copy(); X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if vv <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def solve():
    t0 = time.time()
    HARD = t0 + 171.0
    rng = np.random.default_rng(0)
    # load incumbents
    cands = []
    try:
        xb = np.load("best.npy").astype(np.float64)
        xb = _norm(xb)
        if xb.shape == (116, 6) and np.all(np.isfinite(xb)):
            cands.append(xb)
    except Exception:
        pass
    try:
        xr = np.load("record.npy").astype(np.float64)
        xr = _norm(xr)
        if xr.shape == (116, 6) and np.all(np.isfinite(xr)):
            cands.append(xr)
    except Exception:
        pass
    if len(cands) == 0:
        Xb = _norm(rng.normal(size=(116, 6)))
    else:
        vs = [_maxcos(c) for c in cands]
        Xb = cands[int(np.argmin(vs))].copy()
    bestV = _maxcos(Xb)
    bestX = Xb.copy()
    # quick initial LP to jam (short)
    if HARD - time.time() > 20:
        Xp, vp = _lp_polish(bestX, min(time.time() + 4.0, HARD - 15.0))
        if vp < bestV:
            bestV, bestX = vp, Xp.copy()
    base_contacts, _ = _contacts_set(bestX, tol=1e-6)
    # if too few contacts at 1e-6, loosen slightly to get usable R
    tol = 1e-6
    if len(base_contacts) < 200:
        tol = 5e-6
        base_contacts, _ = _contacts_set(bestX, tol=tol)
    # main soft-mode surgery loop
    t_target = 0.5958
    trial = 0
    # precompute list of RMS targets to cycle
    while time.time() < HARD - 8.0:
        trial += 1
        # rebuild contact graph of current best periodically
        if trial % 4 == 1:
            base_contacts, _ = _contacts_set(bestX, tol=1e-6)
            if len(base_contacts) < 100:
                base_contacts, _ = _contacts_set(bestX, tol=5e-6)
        Xcur = bestX.copy()
        # build rigidity matrix
        try:
            B = _tangent_bases(Xcur)
        except Exception:
            # fallback random kick
            Xn = _norm(Xcur + rng.normal(scale=0.05, size=Xcur.shape))
            Xf = _feas_adam(Xn, t_target, 600, 0.02, 0.005)
            Xs = _softmax_adam(Xf, 700, 0.015, 0.001, 0.012, 0.002)
            v = _maxcos(Xs)
            if v < bestV:
                Xp2, vp2 = _lp_polish(Xs, min(time.time() + 3.0, HARD - 5.0))
                if vp2 < bestV:
                    bestV, bestX = vp2, Xp2.copy()
            continue
        clist = sorted(list(base_contacts))
        if len(clist) < 20:
            # not jammed: random kick
            D = rng.normal(size=Xcur.shape)
            D = D - np.sum(D * Xcur, axis=1, keepdims=True) * Xcur
            rms = float(np.sqrt(np.mean(np.sum(D * D, axis=1))))
            D *= (0.1 / max(rms, 1e-12))
            Xkick = _norm(Xcur + D)
        else:
            pairs = clist
            # cap pairs for R to top 800 closest to max to bound size
            if len(pairs) > 800:
                G = Xcur @ Xcur.T
                vals = [float(G[i, j]) for (i, j) in pairs]
                o = np.argsort(vals)[-800:]
                pairs = [pairs[k] for k in o]
            try:
                R = _build_R(Xcur, B, pairs)
            except Exception:
                R = None
            Dtan = None
            if R is not None and R.shape[0] >= 10:
                try:
                    if R.shape[0] < 580:
                        U, S, Vh = np.linalg.svd(R, full_matrices=True)
                        # Vh 580x580, rows are right singular vectors; smallest last
                        nsoft = int(20 + (trial % 3) * 10)  # 20,30,40 cycle
                        nsoft = min(nsoft, 580)
                        Vsmall = Vh[-nsoft:, :].T  # 580 x nsoft
                    else:
                        U, S, Vh = np.linalg.svd(R, full_matrices=False)
                        nsoft = int(20 + (trial % 3) * 10)
                        nsoft = min(nsoft, Vh.shape[0])
                        Vsmall = Vh[-nsoft:, :].T
                    coeff = rng.normal(size=(Vsmall.shape[1],))
                    u = Vsmall @ coeff  # 580
                    Umat = u.reshape(Xcur.shape[0], 5)
                    Dtan = np.empty_like(Xcur)
                    for i in range(Xcur.shape[0]):
                        Dtan[i] = B[i] @ Umat[i]
                except Exception:
                    Dtan = None
            if Dtan is None:
                Dtan = rng.normal(size=Xcur.shape)
                Dtan = Dtan - np.sum(Dtan * Xcur, axis=1, keepdims=True) * Xcur
            # scale to RMS target 0.05-0.25
            rms_targets = [0.05, 0.09, 0.14, 0.20, 0.25]
            rms_t = rms_targets[trial % len(rms_targets)]
            # add small random jitter to target
            rms_t *= (0.85 + 0.3 * rng.random())
            rms = float(np.sqrt(np.mean(np.sum(Dtan * Dtan, axis=1))))
            if rms < 1e-12:
                Dtan = rng.normal(size=Xcur.shape)
                Dtan = Dtan - np.sum(Dtan * Xcur, axis=1, keepdims=True) * Xcur
                rms = float(np.sqrt(np.mean(np.sum(Dtan * Dtan, axis=1))))
            Dtan *= (rms_t / max(rms, 1e-12))
            # self-stress / high-force separation: top 5-10 pairs by cosine
            try:
                G = Xcur @ Xcur.T
                np.fill_diagonal(G, -2.0)
                iu = np.triu_indices(116, 1)
                c = G[iu]
                k = 5 + (trial % 6)  # 5..10
                top = np.argsort(c)[-k:]
                alpha = 0.03 + 0.02 * rng.random()
                Dsep = np.zeros_like(Xcur)
                for tix in top:
                    i = int(iu[0][tix]); j = int(iu[1][tix])
                    # tangent direction away
                    vi = Xcur[j] - float(Xcur[j] @ Xcur[i]) * Xcur[i]
                    vj = Xcur[i] - float(Xcur[i] @ Xcur[j]) * Xcur[j]
                    ni = float(np.linalg.norm(vi)); nj = float(np.linalg.norm(vj))
                    if ni > 1e-12:
                        Dsep[i] -= alpha * vi / ni
                    if nj > 1e-12:
                        Dsep[j] -= alpha * vj / nj
                Dtot = Dtan + Dsep
            except Exception:
                Dtot = Dtan
            Xkick = _norm(Xcur + Dtot)
        # feasibility phase: hinge to t_target
        feas_steps = 600 + (trial % 2) * 200  # 600 or 800
        # time guard: reduce steps if late
        if time.time() > HARD - 30:
            feas_steps = 500
        Xf = _feas_adam(Xkick, t_target, feas_steps, 0.02, 0.005)
        # topology check before expensive softmax? compute contacts
        try:
            cf, _ = _contacts_set(Xf, tol=1e-6)
            jac = _jaccard(base_contacts, cf)
        except Exception:
            jac = 0.0
        # softmax anneal
        soft_steps = 700
        if time.time() > HARD - 30:
            soft_steps = 500
        Xs = _softmax_adam(Xf, soft_steps, 0.015, 0.001, 0.012, 0.002)
        vf = _maxcos(Xs)
        try:
            cs, _ = _contacts_set(Xs, tol=1e-6)
            jac2 = _jaccard(base_contacts, cs)
        except Exception:
            jac2 = 0.0
        # accept basin only if materially different OR clearly better
        # if same basin and not better than best, skip LP to save time
        same_basin = (jac2 > 0.9)
        if same_basin and vf >= bestV - 1e-7:
            # slide back: skip LP polish, continue to next kick
            continue
        # short LP polish for promising / novel basins
        # only polish if within threshold to avoid wasting time
        if vf > bestV + 0.004 and jac2 < 0.9:
            # still allow one polish rarely? skip if far
            if trial % 3 != 0:
                continue
        budget = 2.5
        if vf < bestV:
            budget = 4.0
        if time.time() > HARD - 12:
            budget = min(budget, HARD - 6.0 - time.time())
            if budget < 0.8:
                # no time for LP, just track
                if vf < bestV:
                    bestV, bestX = vf, Xs.copy()
                    base_contacts = cs
                continue
        deadline = min(time.time() + budget, HARD - 5.0)
        Xp2, vp2 = _lp_polish(Xs, deadline)
        if vp2 < bestV:
            bestV, bestX = vp2, Xp2.copy()
            base_contacts = cs if 'cs' in locals() else base_contacts
            # update base contacts to new jammed state
            try:
                base_contacts, _ = _contacts_set(bestX, tol=1e-6)
            except Exception:
                pass
        # also track unpolished if LP failed but softmax good
        if vf < bestV:
            bestV, bestX = vf, Xs.copy()
            try:
                base_contacts, _ = _contacts_set(bestX, tol=1e-6)
            except Exception:
                pass
    bestX = _norm(bestX)
    return bestX
