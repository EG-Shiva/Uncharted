import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _lp_polish(x, deadline, r0=1.2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2800:
            idx = np.argsort(c[sel])[-2800:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _balanced_quench(X, steps, b0, b1, lr0, lr1, mu=0.85, lam=0.12, bl_frac=0.25):
    n, d = X.shape
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    best_m = _max_cos(X)
    best = X.copy()
    V = np.zeros_like(X)
    one_mu = 1.0 - mu
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best = X.copy()
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        Sg = E.sum() / max(n, 1) + 1e-30
        Grad_g = (E @ X) / Sg
        bl = max(beta * bl_frac, 8.0)
        mx = G.max(axis=1, keepdims=True)
        W = np.exp(bl * (G - mx))
        np.fill_diagonal(W, 0.0)
        Rs = W.sum(axis=1, keepdims=True) + 1e-30
        Grad_l = (W @ X) / Rs
        Grad = Grad_g + lam * Grad_l
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        scale = np.minimum(1.0, 3.0 / (nn + 1e-12))
        Gt = Gt * scale
        V = mu * V + one_mu * Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return best, best_m

def _hard_focus(X, steps, lr0=0.003, lr1=0.00012, tol0=2e-4, tol1=2e-6):
    X = X.copy()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    best_m = _max_cos(X)
    best = X.copy()
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tol = tol0 + (tol1 - tol0) * frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best = X.copy()
        M = (G > (m - tol)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        cnt = M.sum(axis=1, keepdims=True) + 1e-12
        active = (cnt[:, 0] > 0.5)
        if not np.any(active):
            continue
        Grad = (M @ X) / cnt
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        step = np.zeros_like(Gt)
        step[active] = Gt[active] / nn[active]
        X = X - lr * step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return best

def _find_holes(Xkeep, k, rng, n_sample=20000):
    d = Xkeep.shape[1]
    cur = Xkeep.copy()
    out = []
    for _ in range(k):
        S = rng.randn(n_sample, d)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        scores = (S @ cur.T).max(axis=1)
        # encourage separation among new holes by penalizing closeness to already picked
        best_idx = int(np.argmin(scores))
        out.append(S[best_idx])
        cur = np.vstack([cur, S[best_idx:best_idx + 1]])
    return np.array(out)

def solve():
    t0 = time.time()
    BUDGET = 175.0
    deadline_final = t0 + BUDGET
    n, d = 168, 8
    rng = np.random.RandomState(12345)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (n, d):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va
                best = a.copy()
        except Exception:
            continue
    if best is None:
        best = rng.randn(n, d)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    incumbent = best.copy()
    inc_val = best_val
    base = best.copy()

    # quick LP squeeze on incumbent (baseline)
    try:
        tmp = _lp_polish(incumbent, min(deadline_final - 100.0, time.time() + 12.0))
        v = _max_cos(tmp)
        if v < inc_val:
            incumbent = tmp
            inc_val = v
    except Exception:
        pass

    # diversification trials; reserve ~25s for final squeeze
    sigmas = [0.02, 0.05, 0.10, 0.03, 0.08, 0.15, 0.04, 0.06]
    betas = [(60.0, 900.0), (120.0, 1500.0), (40.0, 600.0), (200.0, 2500.0)]
    trial = 0
    while time.time() < deadline_final - 30.0 and trial < 20:
        trial += 1
        sigma = sigmas[(trial - 1) % len(sigmas)]
        b0, b1 = betas[(trial - 1) % len(betas)]
        # start point
        try:
            if trial % 3 == 0:
                # ejection trial: remove worst points, insert holes
                per = _per_max(incumbent)
                k = 4 if (trial % 6 == 0) else 3
                worst = np.argsort(per)[-k:]
                mask = np.ones(n, dtype=bool)
                mask[worst] = False
                Xkeep = incumbent[mask]
                holes = _find_holes(Xkeep, k, rng, n_sample=16000)
                X0 = np.empty_like(incumbent)
                X0[mask] = Xkeep
                X0[~mask] = holes
                # small jitter all
                jit = rng.randn(n, d) * 0.005
                jit -= (jit * X0).sum(axis=1, keepdims=True) * X0
                X0 = X0 + jit
                X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            else:
                X0 = base.copy()
                # mix incumbent and base for diversity
                if trial % 2 == 0:
                    X0 = incumbent.copy()
                jit = rng.randn(n, d) * sigma
                jit -= (jit * X0).sum(axis=1, keepdims=True) * X0
                X0 = X0 + jit
                X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        except Exception:
            continue
        # quench
        try:
            steps = 450 if time.time() < deadline_final - 60 else 250
            lr0 = 0.02 + (trial % 3) * 0.01
            lr1 = 0.0015
            Xq, _ = _balanced_quench(X0, steps, b0, b1, lr0, lr1, mu=0.85, lam=0.12)
        except Exception:
            continue
        try:
            Xq = _hard_focus(Xq, steps=120)
        except Exception:
            pass
        # short LP
        try:
            Xp = _lp_polish(Xq, min(deadline_final - 25.0, time.time() + 4.0), r0=8e-4)
        except Exception:
            continue
        try:
            v = _max_cos(Xp)
        except Exception:
            continue
        if v < inc_val:
            incumbent = Xp.copy()
            inc_val = v
        # early stop if clearly record-beating and time low? keep going
        if time.time() > deadline_final - 30.0:
            break

    # final long squeeze
    try:
        remain = deadline_final - time.time() - 1.0
        if remain > 3.0:
            Xf = _lp_polish(incumbent, time.time() + remain, r0=1.2e-3)
            v = _max_cos(Xf)
            if v < inc_val:
                incumbent = Xf
                inc_val = v
    except Exception:
        pass
    incumbent = incumbent / np.linalg.norm(incumbent, axis=1, keepdims=True)
    return incumbent
