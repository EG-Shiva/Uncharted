import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _normed(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, best=None, tol=1e-7):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    if best is None:
        best = float(G.max())
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    sel = c > best - tol
    I = iu0[sel]; J = iu1[sel]
    deg = np.zeros(n, dtype=int)
    np.add.at(deg, I, 1)
    np.add.at(deg, J, 1)
    return I, J, deg, best, c

def _lp_polish_deadline(x, deadline, r0=1e-3):
    x = _normed(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    xb = x.copy()
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        if not np.any(sel):
            break
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _normed(y)
        try:
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            best = vv; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb

def _smooth_opt(X0, alpha, maxiter):
    n = X0.shape[0]
    yflat = X0.ravel().copy()
    def fg(yf):
        Y = yf.reshape(n, 4)
        nr = np.sqrt((Y * Y).sum(axis=1, keepdims=True))
        nr = np.maximum(nr, 1e-12)
        S = Y / nr
        G = S @ S.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp(alpha * (G - m))
        np.fill_diagonal(E, 0.0)
        Ssum = float(E.sum()) * 0.5 + 1e-300
        lse = m + np.log(Ssum) / alpha
        W = E / Ssum
        gradS = W @ S
        dot = (gradS * S).sum(axis=1, keepdims=True)
        gradY = (gradS - dot * S) / nr
        return float(lse), gradY.ravel()
    try:
        res = minimize(fg, yflat, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-6, 'maxfun': maxiter * 10})
        yf = res.x
    except Exception:
        yf = yflat
    Y = yf.reshape(n, 4)
    return _normed(Y)

def _hole_for_rattler(X, exclude, rng, ncan=6000):
    n = X.shape[0]
    mask = np.ones(n, dtype=bool)
    mask[np.asarray(exclude, dtype=int)] = False
    Xk = X[mask]
    C = rng.standard_normal((ncan, 4))
    C = _normed(C)
    G = C @ Xk.T
    mx = G.max(axis=1)
    bi = int(np.argmin(mx))
    return C[bi]

def solve():
    t0 = time.time()
    TLIM = 172.0
    deadline = t0 + TLIM
    rng = np.random.default_rng(292417)
    n, d = 292, 4
    xb = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (292, 4):
                a = _normed(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    xb = a.copy()
        except Exception:
            continue
    if xb is None:
        xb = _normed(rng.standard_normal((n, d)))
        best_val = _max_cos(xb)
    x_best = xb.copy()
    # quick settle of incumbent (short)
    if time.time() < deadline - 20:
        try:
            x_best = _lp_polish_deadline(x_best, min(deadline - 15, time.time() + 4.0))
            best_val = _max_cos(x_best)
        except Exception:
            pass
    iu0, iu1 = np.triu_indices(n, 1)
    ntrials = 60
    for trial in range(ntrials):
        if time.time() > deadline - 18.0:
            break
        try:
            I, J, deg, bv, _ = _contacts(x_best, best_val, tol=2e-7)
        except Exception:
            deg = np.full(n, 6)
            bv = best_val
        rattlers = np.where(deg == 0)[0].tolist()
        low = np.where(deg <= 3)[0].tolist()
        high = np.where(deg >= 7)[0].tolist()
        base = x_best.copy()
        mode = trial % 4
        X0 = base.copy()
        try:
            if mode == 0 and len(rattlers) > 0:
                # transplant rattler(s) to largest hole + kick hubs
                for r in rattlers[:2]:
                    h = _hole_for_rattler(base, [r], rng, ncan=6000)
                    X0[r] = h
                # kick a few high-degree points hard
                if len(high) > 0:
                    k = min(8, len(high))
                    pick = rng.choice(np.array(high), size=k, replace=False)
                    pert = rng.standard_normal((len(pick), d))
                    pert -= (pert * X0[pick]).sum(axis=1, keepdims=True) * X0[pick]
                    pn = np.linalg.norm(pert, axis=1, keepdims=True) + 1e-12
                    pert = pert / pn * 0.10
                    X0[pick] = _normed(X0[pick] + pert)
                # small global jitter
                pert = rng.standard_normal((n, d))
                pert -= (pert * X0).sum(axis=1, keepdims=True) * X0
                pn = np.linalg.norm(pert, axis=1, keepdims=True) + 1e-12
                X0 = _normed(X0 + pert / pn * 0.008)
            elif mode == 1:
                # degree-balanced kick: hubs + randoms
                k = 12
                cand = []
                if len(high) > 0:
                    cand += rng.choice(np.array(high), size=min(6, len(high)), replace=False).tolist()
                cand += rng.choice(n, size=k, replace=False).tolist()
                cand = np.unique(np.array(cand, dtype=int))[:14]
                sigma = 0.06 if (trial % 8 < 4) else 0.12
                pert = rng.standard_normal((len(cand), d))
                pert -= (pert * X0[cand]).sum(axis=1, keepdims=True) * X0[cand]
                pn = np.linalg.norm(pert, axis=1, keepdims=True) + 1e-12
                pert = pert / pn * sigma
                X0[cand] = _normed(X0[cand] + pert)
            elif mode == 2:
                # global small perturbation, different basin nearby
                sigma = 0.018
                pert = rng.standard_normal((n, d))
                pert -= (pert * X0).sum(axis=1, keepdims=True) * X0
                pn = np.linalg.norm(pert, axis=1, keepdims=True) + 1e-12
                X0 = _normed(X0 + pert / pn * sigma)
            else:
                # low-degree / random subset medium kick + hole nudge for rattler
                k = 10
                pool = low + rattlers
                if len(pool) < 4:
                    pool = list(range(n))
                pick = rng.choice(np.array(pool), size=min(k, len(pool)), replace=False)
                # also pick some hubs to relieve
                if len(high) > 0:
                    pick2 = rng.choice(np.array(high), size=min(5, len(high)), replace=False)
                    pick = np.unique(np.concatenate([pick, pick2]))
                pert = rng.standard_normal((len(pick), d))
                pert -= (pert * X0[pick]).sum(axis=1, keepdims=True) * X0[pick]
                pn = np.linalg.norm(pert, axis=1, keepdims=True) + 1e-12
                pert = pert / pn * 0.09
                X0[pick] = _normed(X0[pick] + pert)
        except Exception:
            X0 = base.copy()
        # two-stage smooth sharpening
        try:
            if time.time() > deadline - 16:
                break
            X1 = _smooth_opt(X0, 600.0, 40)
            if time.time() > deadline - 15:
                # evaluate and continue
                v1 = _max_cos(X1)
                if v1 < best_val:
                    best_val = v1; x_best = X1.copy()
                break
            X2 = _smooth_opt(X1, 2200.0, 60)
        except Exception:
            continue
        try:
            v = _max_cos(X2)
        except Exception:
            continue
        if v < best_val:
            best_val = v
            x_best = X2.copy()
            # immediate LP squeeze on improvement
            if time.time() < deadline - 12:
                try:
                    Xp = _lp_polish_deadline(x_best, min(deadline - 10, time.time() + 4.0))
                    vp = _max_cos(Xp)
                    if vp < best_val:
                        best_val = vp; x_best = Xp.copy()
                except Exception:
                    pass
        elif v < best_val + 1.2e-05 and time.time() < deadline - 20:
            # near miss: LP polish to different local optimum, keep if better
            try:
                Xp = _lp_polish_deadline(X2, time.time() + 2.5)
                vp = _max_cos(Xp)
                if vp < best_val:
                    best_val = vp; x_best = Xp.copy()
            except Exception:
                pass
        # else discard
    # final polish
    try:
        if time.time() < deadline - 2.0:
            x_best = _lp_polish_deadline(x_best, min(deadline - 0.5, time.time() + 12.0))
    except Exception:
        pass
    x_best = _normed(x_best)
    return x_best
