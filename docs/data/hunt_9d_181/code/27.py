import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, budget, init_radius=8e-3):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = init_radius
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < budget and radius > 1e-14:
        if time.time() - t0 + 0.25 > budget and radius < 5e-4:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            kth = np.partition(c_all, -3000)[-3000]
            sel = c_all >= kth
            cnt = int(sel.sum())
            if cnt > 3500:
                idx = np.where(sel)[0]
                order = np.argsort(-c_all[idx])[:3500]
                mask = np.zeros_like(c_all, dtype=bool)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_anneal_full(x, iters=250, p_start=25, p_end=70, lr=0.04):
    n, d = x.shape
    for it in range(iters):
        frac = it / max(1, iters - 1)
        p = p_start + (p_end - p_start) * frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        Gp = np.where(G > 0, G, 0.0)
        m = Gp.max()
        if m <= 1e-12:
            break
        W = (Gp / m) ** (p - 1)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_t = grad - dot * x
        nrm = np.mean(np.linalg.norm(grad_t, axis=1)) + 1e-12
        step = lr * (1.0 - 0.5 * frac) / nrm
        x = x - step * grad_t
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

# ---- symmetry constrained part: cyclic shifts on R9 ----
_D = 9
_K = 20  # 20*9+1=181
# index map Idx[s,j]=(j-s)%9
_IDX = (np.arange(_D)[None, :] - np.arange(_D)[:, None]) % _D  # shape (9,9): row s

def _expand(R):
    K = R.shape[0]
    N = K * 9 + 1
    X = np.empty((N, _D), dtype=np.float64)
    for k in range(K):
        X[k*9:(k+1)*9] = R[k][_IDX]
    X[-1] = 1.0 / 3.0
    return X

def _sym_optimize(rng, steps=2200, T0=0.04, T1=0.0018, lr0=0.09, lr1=0.012, mu=0.7):
    K = _K
    R = rng.standard_normal((K, _D))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    V = np.zeros_like(R)
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    bestR = R.copy()
    # quick eval
    Xb = _expand(R)
    best_val = _max_cos(Xb)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        X = _expand(R)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        # softmax weights
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        # gradient on full config
        GX = W @ X  # (N,d)
        # pullback to representatives (exclude fixed point row)
        Gorb = GX[:-1].reshape(K, 9, _D)  # [k,s,j]
        gR = np.zeros_like(R)
        for sh in range(9):
            gR += np.roll(Gorb[:, sh, :], shift=-sh, axis=1)
        # normalize scale by mean weight to keep step stable
        # tangent projection
        dot = np.sum(gR * R, axis=1, keepdims=True)
        gT = gR - dot * R
        nm = np.mean(np.linalg.norm(gT, axis=1)) + 1e-12
        gT /= nm
        V = mu * V + (1.0 - mu) * gT
        # normalize velocity
        # R update
        R = R - lr * V
        nrm = np.linalg.norm(R, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        R /= nrm
        # track best by true max every 200 steps + end
        if (s % 200 == 199) or (s == steps - 1):
            Xc = _expand(R)
            v = _max_cos(Xc)
            if v < best_val:
                best_val = v
                bestR = R.copy()
    return bestR, best_val

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(0)
    # 1) symmetry search
    try:
        Rsym, vsym = _sym_optimize(rng, steps=2200)
        Xsym = _expand(Rsym)
    except Exception:
        Rsym = None
        Xsym = None
        vsym = np.inf
    # 2) load fallback codes
    cands = []
    if Xsym is not None:
        try:
            Xn = Xsym / np.linalg.norm(Xsym, axis=1, keepdims=True)
            if np.all(np.finite(Xn)):
                cands.append(Xn)
        except Exception:
            pass
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        tmp = rng.standard_normal((181, 9))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    xb = cands[bi].copy()
    bb = float(vals[bi])
    # identify symmetric seed (first cand if present)
    has_sym = (Xsym is not None) and len(cands) > 0
    # base seed for perturbations: prefer symmetric seed even if slightly worse
    if Xsym is not None:
        Xseed = (Xsym / np.linalg.norm(Xsym, axis=1, keepdims=True)).copy()
        vseed = _max_cos(Xseed)
    else:
        Xseed = xb.copy()
        vseed = bb
    # 3) break symmetry: graded perturbations + smooth + LP
    scales = [0.0, 1e-4, 5e-4, 1e-3, 3e-3, 1e-2, 3e-3, 1e-3]
    # LP budgets per scale (seconds)
    lp_budgets = [5.0, 4.0, 5.0, 5.0, 6.0, 7.0, 5.0, 4.0]
    init_rs =   [0.008, 0.004, 0.006, 0.010, 0.015, 0.025, 0.015, 0.010]
    for si, sig in enumerate(scales):
        rem = BUDGET - (time.time() - t0)
        if rem < 4.0:
            break
        y = Xseed.copy()
        if sig > 0:
            # deterministic noise per scale (continue rng stream)
            y = y + rng.standard_normal(y.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        # short smooth to spread load (cheaper for small sig)
        try:
            if sig == 0.0:
                y = _smooth_anneal_full(y, iters=200, p_start=30, p_end=80, lr=0.03)
            else:
                y = _smooth_anneal_full(y, iters=300, p_start=25, p_end=70, lr=0.04)
        except Exception:
            pass
        rem2 = BUDGET - (time.time() - t0)
        if rem2 < 2.5:
            break
        b = min(lp_budgets[si], rem2 - 1.0)
        if b < 0.8:
            b = max(0.8, rem2 - 0.5)
        try:
            y2 = _lp_polish(y, budget=float(b), init_radius=init_rs[si])
            v = _max_cos(y2)
        except Exception:
            continue
        if v < bb - 1e-14:
            xb = y2.copy(); bb = v
            # update seed to new best for further perturbations? keep original symmetric seed
            # but also allow hopping from new best for large scales: keep Xseed as is per spec
            # do extra tight polish if record-like improvement and time allows
            rem3 = BUDGET - (time.time() - t0)
            if rem3 > 9.0:
                try:
                    y3 = _lp_polish(xb, budget=1.5, init_radius=1e-3)
                    vv = _max_cos(y3)
                    if vv < bb:
                        xb = y3; bb = vv
                except Exception:
                    pass
        # also if this perturbation of symmetric seed beats symmetric seed, update seed chain slightly?
        # keep Xseed fixed to preserve basin exploration (per assignment)
    # 4) ensure we at least polish the loaded best if symmetric path failed to beat it
    rem = BUDGET - (time.time() - t0)
    # quick final squeeze on global best
    if rem > 3.0:
        try:
            yf = _lp_polish(xb, budget=min(rem - 0.6, 6.0), init_radius=1e-3)
            vf = _max_cos(yf)
            if vf < bb:
                xb = yf; bb = vf
        except Exception:
            pass
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return xb
