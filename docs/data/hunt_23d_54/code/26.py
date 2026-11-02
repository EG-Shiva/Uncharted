import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _softmax_anneal(X, steps, k0, k1, lr0, lr1, w0, w1, mu=0.85, t0=None, deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lk0, lk1 = np.log(k0), np.log(k1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        f = s / max(steps - 1, 1)
        k = np.exp(lk0 + (lk1 - lk0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        w = w0 + (w1 - w0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        sel = G > (m - w)
        W = np.zeros_like(G)
        # stable exp: G-m in [-w,0]
        W[sel] = np.exp(k * (G[sel] - m))
        np.fill_diagonal(W, 0.0)
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        an = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
        V = mu * V + Ft / an
        vn = float(np.sqrt(np.sum(V * V) / n)) + 1e-12
        step = lr * V / max(vn, 1.0)
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _bm_homotopy(Y0, t, steps, lr0, lr1, mu=0.85, deadline=None):
    Y = Y0.copy()
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    n = Y.shape[0]
    V = np.zeros_like(Y)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    best_res = 1e9
    best_Y = Y.copy()
    for s in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        f = s / max(steps - 1, 1)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = Y @ Y.T
        # violation matrix
        W = G - t
        W[W < 0] = 0.0
        np.fill_diagonal(W, 0.0)
        res = float(W.max()) if W.size else 0.0
        if res < best_res:
            best_res = res
            best_Y = Y.copy()
            if best_res <= 0:
                break
        if res <= 1e-12:
            # feasible: tiny random walk to keep rank full then continue
            pass
        F = W @ Y
        dot = np.sum(F * Y, axis=1, keepdims=True)
        Ft = F - dot * Y
        an = float(np.sqrt(np.sum(Ft * Ft) / n)) + 1e-12
        if an < 1e-15:
            # no gradient (feasible); add tiny noise to explore
            Y = Y + np.random.randn(*Y.shape) * 1e-4
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            V *= 0.5
            continue
        V = mu * V + Ft / an
        vn = float(np.sqrt(np.sum(V * V) / n)) + 1e-12
        Y = Y - lr * V / max(vn, 1.0)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Y /= nr
    return best_Y, best_res

def _gram_ap(G0, t, iters, deadline=None):
    G = G0.copy()
    G = (G + G.T) * 0.5
    n = G.shape[0]
    r = 23
    iu = np.triu_indices(n, 1)
    best_res = 1e9
    best_X = None
    # Dykstra-style corrections (two auxiliaries)
    P = np.zeros_like(G)
    Q = np.zeros_like(G)
    for k in range(iters):
        if deadline is not None and time.time() > deadline:
            break
        # --- project onto rank-r PSD (with Dykstra correction P) ---
        S = G + P
        S = (S + S.T) * 0.5
        w, V = np.linalg.eigh(S)
        Vc = V[:, -r:]  # largest r
        lam = w[-r:]
        lam_pos = np.maximum(lam, 0.0)
        Gr = (Vc * lam_pos) @ Vc.T
        Gr = (Gr + Gr.T) * 0.5
        P = S - Gr  # Dykstra update ~ residual
        # damp P to avoid blowup
        P *= 0.85
        # --- project onto diag/box (with correction Q) ---
        S2 = Gr + Q
        S2 = (S2 + S2.T) * 0.5
        Gb = np.minimum(S2, t)
        np.fill_diagonal(Gb, 1.0)
        Q = (S2 - Gb) * 0.85
        G = Gb
        if (k % 150 == 0) or (k == iters - 1):
            # infeasibility residual of rank projection
            off = Gr[iu].max()
            diag_err = float(np.abs(np.diag(Gr) - 1.0).max())
            tail = float(np.sum(np.maximum(-w[:-r], 0.0)) + np.sum(np.maximum(w[:-r], 0.0)) if n > r else 0.0)
            viol = max(0.0, float(off) - t)
            res = viol + 0.1 * diag_err
            # extract points from Gr
            sq = np.sqrt(np.maximum(lam_pos, 0.0))
            # avoid zero scale
            Y = Vc * sq[None, :]
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr[nr < 1e-12] = 1.0
            Xn = Y / nr
            # guard NaN
            if np.all(np.isfinite(Xn)):
                v = _max_cos(Xn)
                # score residual by actual max too
                if viol < best_res:
                    best_res = viol
                    best_X = Xn.copy()
    return best_X, best_res

def _lp_polish_short(X, seconds, deadline=None):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        it += 1
        if it > 40:
            break
        Gf = X @ X.T
        c_all = Gf[iu]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 600:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:600]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
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
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 3.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t_start = time.time()
    DEADLINE = t_start + 170.0
    rng = np.random.RandomState(0)
    np.random.seed(0)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()
    cands = []

    targets = [0.04755, 0.04745, 0.04730, 0.04700]
    # ---- Phase A: Gram alternating projections (SDP-style) ----
    for ti, t in enumerate(targets):
        if time.time() > DEADLINE - 90:
            break
        for rs in range(2):
            if time.time() > DEADLINE - 90:
                break
            if rs == 0:
                X0 = x_best + rng.randn(n, d) * (0.004 + 0.004 * ti)
            else:
                X0 = rng.randn(n, d) * 1.0
                # bias toward best subspace slightly
                X0 = 0.7 * X0 + 0.3 * (rng.randn(d, d) @ rng.randn(d, d)).mean(axis=0, keepdims=True) * 0.0 + 0.0
                X0 = rng.randn(n, d)
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            G0 = X0 @ X0.T
            iters = 900 if rs == 0 else 600
            Xa, res_a = _gram_ap(G0, t, iters=iters, deadline=DEADLINE - 85)
            if Xa is not None and np.all(np.isfinite(Xa)):
                va = _max_cos(Xa)
                cands.append((va, Xa.copy(), t, res_a))
                if va < best:
                    best, x_best = va, Xa.copy()
            # ---- Phase B: Burer-Monteiro Y Y^T penalty homotopy for same t ----
            Y0 = Xa if Xa is not None else X0
            Yb, res_b = _bm_homotopy(Y0, t, steps=1200, lr0=0.02, lr1=0.0008, mu=0.85, deadline=DEADLINE - 85)
            if np.all(np.isfinite(Yb)):
                Yb /= np.linalg.norm(Yb, axis=1, keepdims=True)
                vb = _max_cos(Yb)
                cands.append((vb, Yb.copy(), t, res_b))
                if vb < best:
                    best, x_best = vb, Yb.copy()

    # ---- Phase C: descending homotopy chain from best ----
    if time.time() < DEADLINE - 60:
        Yc = x_best.copy()
        chain = [best - 5e-05, best - 1e-04, best - 2e-04, 0.04745, 0.04730]
        # keep only values below current best and sensible
        chain = [c for c in chain if c < best and c > 0.040]
        # dedupe sorted descending
        chain = sorted(set(chain), reverse=True)[:4]
        for t in chain:
            if time.time() > DEADLINE - 55:
                break
            Yc, res_c = _bm_homotopy(Yc, t, steps=900, lr0=0.015, lr1=0.0006, mu=0.86, deadline=DEADLINE - 50)
            Yc /= np.linalg.norm(Yc, axis=1, keepdims=True)
            vc = _max_cos(Yc)
            cands.append((vc, Yc.copy(), t, res_c))
            if vc < best:
                best, x_best = vc, Yc.copy()

    # ---- Phase D: tangential softmax refinement of top candidates ----
    cands = sorted(cands, key=lambda z: z[0])[:4]
    # always include incumbent
    cands.append((best, x_best.copy(), best, 0.0))
    # dedupe to top 4 unique
    seen = []
    for va, Xa, tt, rr in sorted(cands, key=lambda z: z[0]):
        if time.time() > DEADLINE - 35:
            break
        if not np.all(np.isfinite(Xa)):
            continue
        Xs = _softmax_anneal(Xa.copy(), steps=500, k0=120, k1=2500, lr0=0.010, lr1=0.0007,
                             w0=0.045, w1=0.005, mu=0.85, deadline=DEADLINE - 30)
        vs = _max_cos(Xs)
        if vs < best:
            best, x_best = vs, Xs.copy()
        seen.append((vs, Xs))
        # second sharper anneal if promising (within 3e-4 of best)
        if vs < best + 3e-04 and time.time() < DEADLINE - 30:
            Xs2 = _softmax_anneal(Xs.copy(), steps=350, k0=400, k1=4000, lr0=0.004, lr1=0.0003,
                                  w0=0.015, w1=0.003, mu=0.88, deadline=DEADLINE - 25)
            vs2 = _max_cos(Xs2)
            if vs2 < best:
                best, x_best = vs2, Xs2.copy()

    # ---- Phase E: one short LP trust-region polish on the champion ----
    if time.time() < DEADLINE - 8:
        remain = DEADLINE - time.time() - 2.0
        secs = min(max(remain, 3.0), 14.0)
        try:
            Xp = _lp_polish_short(x_best.copy(), seconds=secs, deadline=DEADLINE - 1.0)
            vp = _max_cos(Xp)
            if vp < best and np.all(np.isfinite(Xp)):
                best, x_best = vp, Xp.copy()
        except Exception:
            pass

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # final safety: if somehow worse than loaded best, return loaded best
    try:
        x0 = np.load("best.npy").astype(float)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        if _max_cos(x0) < _max_cos(x_best):
            x_best = x0.copy()
    except Exception:
        pass
    return x_best
