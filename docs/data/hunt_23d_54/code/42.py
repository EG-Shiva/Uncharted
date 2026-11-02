import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _build_X(S, sel, Dmat):
    a, b, c = S[0], S[1], S[2]
    A = a[Dmat]
    B = b[Dmat]
    Cfull = c[Dmat]
    Csel = Cfull[sel]
    return np.vstack([A, B, Csel]), A, B, Cfull


def _greedy_sel(A, B, Cfull):
    # greedily pick 8 shifts minimizing max cosine
    sel = []
    rem = list(range(Cfull.shape[0]))
    for _ in range(8):
        best_v = 1e9
        best_k = -1
        for k in rem:
            if k in sel:
                continue
            cand_idx = sel + [k]
            Xc = np.vstack([A, B, Cfull[cand_idx]])
            G = Xc @ Xc.T
            np.fill_diagonal(G, -2.0)
            v = float(np.max(G))
            if v < best_v:
                best_v = v
                best_k = k
        sel.append(best_k)
    return np.array(sorted(sel), dtype=int)


def _optimize_seed(S_init, sel_init, Dmat, Cmat, steps, lr=0.008):
    N = Dmat.shape[0]
    ar = np.arange(N)
    S = S_init.copy()
    sel = sel_init.copy()
    m = np.zeros_like(S)
    v = np.zeros_like(S)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # initial greedy
    try:
        _, A0, B0, C0 = _build_X(S, sel, Dmat)
        sel = _greedy_sel(A0, B0, C0)
    except Exception:
        pass
    for s in range(steps):
        # mid reselection
        if s == 1000 and steps > 1200:
            try:
                _, A1, B1, C1 = _build_X(S, sel, Dmat)
                sel = _greedy_sel(A1, B1, C1)
                m[:] = 0
                v[:] = 0
            except Exception:
                pass
        frac = s / max(steps - 1, 1)
        beta = 50.0 * (800.0 / 50.0) ** frac
        X, _, _, _ = _build_X(S, sel, Dmat)
        # Gram and weights
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(np.max(G))
        # stable exp
        E = np.exp(beta * (G - gm))
        np.fill_diagonal(E, 0.0)
        Esum = float(np.sum(E))
        if Esum <= 0:
            Esum = 1.0
        Gr = (E @ X) * (2.0 / Esum)
        # split grads
        GA = Gr[0:23]
        GB = Gr[23:46]
        GC = Gr[46:54]
        # map to seed grads
        grad_a = np.take_along_axis(GA, Cmat, axis=1).sum(axis=0)
        grad_b = np.take_along_axis(GB, Cmat, axis=1).sum(axis=0)
        idxC = (sel[:, None] + ar[None, :]) % N
        # gather
        rolled = GC[np.arange(GC.shape[0])[:, None], idxC]
        grad_c = rolled.sum(axis=0)
        grad = np.vstack([grad_a, grad_b, grad_c])
        # Riemannian projection
        dot = np.sum(grad * S, axis=1, keepdims=True)
        gt = grad - dot * S
        m = b1 * m + (1 - b1) * gt
        v = b2 * v + (1 - b2) * (gt * gt)
        bc1 = 1 - b1 ** (s + 1)
        bc2 = 1 - b2 ** (s + 1)
        mh = m / bc1
        vh = v / bc2
        step = lr * mh / (np.sqrt(vh) + eps)
        # keep tangent part only
        sdot = np.sum(step * S, axis=1, keepdims=True)
        step_t = step - sdot * S
        S = S - step_t
        nr = np.linalg.norm(S, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        S /= nr
    Xf, _, _, _ = _build_X(S, sel, Dmat)
    return S, sel, Xf


def _smooth(X_init, steps, beta0, beta1, lr0, lr1, width0, width1):
    X = X_init.copy()
    n = X.shape[0]
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f if beta0 > 0 else beta1
        lr = lr0 * (lr1 / lr0) ** f if steps > 1 else lr0
        w = width0 + (width1 - width0) * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(np.max(G))
        # window to sparsify
        D = G - gm
        W = np.zeros_like(G)
        selm = D > -w
        if np.any(selm):
            W[selm] = np.exp(beta * D[selm])
        np.fill_diagonal(W, 0.0)
        F = W @ X
        dot = np.sum(F * X, axis=1, keepdims=True)
        Ft = F - dot * X
        deg = np.sum(W, axis=1, keepdims=True) + 1.0
        X = X - lr * Ft / deg
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X


def _lp_polish(X, seconds, t0=None, budget=1e9):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and (time.time() - t0) > budget:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        c_all = Gfull[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 800:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:800]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(0)
    n, d, N = 54, 23, 23
    ar = np.arange(N)
    Dmat = (ar[None, :] - ar[:, None]) % N
    Cmat = (ar[:, None] + ar[None, :]) % N

    # load fallback best/record
    try:
        xb0 = np.load("best.npy").astype(float)
        xb0 /= np.linalg.norm(xb0, axis=1, keepdims=True)
        if xb0.shape != (54, 23):
            raise ValueError("shape")
    except Exception:
        xb0 = rng.randn(54, 23)
        xb0 /= np.linalg.norm(xb0, axis=1, keepdims=True)
    best_fallback = _max_cos(xb0)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (54, 23):
            vr = _max_cos(xr)
            if vr < best_fallback:
                best_fallback = vr
                xb0 = xr.copy()
    except Exception:
        pass

    cand_X = []
    cand_v = []
    # keep fallback as candidate
    cand_X.append(xb0.copy())
    cand_v.append(best_fallback)

    # Phase 1: symmetry search, strict ~105s
    PHASE1 = 105.0
    ntrials = 0
    STEPS = 2000
    # try at least 30 triples if time allows
    while time.time() - t0 < PHASE1 and ntrials < 34:
        # adapt steps if little time left
        remain = PHASE1 - (time.time() - t0)
        trials_left = 30 - ntrials
        if trials_left > 0 and remain / trials_left < 2.0:
            STEPS_use = 800
        else:
            STEPS_use = STEPS
        if remain < 1.5:
            break
        S_init = rng.randn(3, N)
        S_init /= np.linalg.norm(S_init, axis=1, keepdims=True)
        sel_init = np.array(sorted(rng.choice(N, 8, replace=False)), dtype=int)
        try:
            S_opt, sel_opt, Xf = _optimize_seed(S_init, sel_init, Dmat, Cmat, STEPS_use)
            Xf /= np.linalg.norm(Xf, axis=1, keepdims=True)
            vf = _max_cos(Xf)
            cand_X.append(Xf)
            cand_v.append(vf)
        except Exception:
            pass
        ntrials += 1
        # ensure at least 30 attempts: if STEPS_use large and time runs out, reduce steps
        if ntrials >= 30:
            break
    # keep best few distinct symmetry candidates (excluding fallback at idx0)
    order = np.argsort(np.array(cand_v))
    # pick top symmetry ones
    sym_best = []
    for idx in order:
        if idx == 0:
            continue
        sym_best.append(cand_X[idx])
        if len(sym_best) >= 3:
            break
    if len(sym_best) == 0 and len(cand_X) > 1:
        sym_best = [cand_X[order[0]]]
    # if no symmetry candidate (timeout immediately), use fallback perturbed
    if len(sym_best) == 0:
        sym_best = [xb0.copy()]

    # Phase 2: gradual symmetry breaking
    # budget until BUDGET-20
    release_scales = [1e-3, 3e-3, 1e-2]
    pool_X = list(sym_best)
    pool_v = [_max_cos(x) for x in pool_X]
    # also include fallback in pool for release? keep one slot
    # iterate releases
    for scale in release_scales:
        if time.time() - t0 > BUDGET - 22.0:
            break
        new_pool_X = []
        new_pool_v = []
        # for each current best, perturb and smooth
        # keep pool size <=3
        idxs = np.argsort(np.array(pool_v))[:3]
        for ii in idxs:
            Xb = pool_X[ii]
            # two noise draws per scale for best, one for others
            ndraw = 2 if ii == idxs[0] else 1
            for _ in range(ndraw):
                if time.time() - t0 > BUDGET - 22.0:
                    break
                Xn = Xb + rng.randn(n, d) * scale
                Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
                try:
                    Xs = _smooth(Xn, steps=450, beta0=60.0, beta1=1200.0,
                                 lr0=0.02, lr1=0.002, width0=0.06, width1=0.008)
                except Exception:
                    continue
                vs = _max_cos(Xs)
                new_pool_X.append(Xs)
                new_pool_v.append(vs)
        # merge
        pool_X = pool_X + new_pool_X
        pool_v = pool_v + new_pool_v
        # prune to 3 distinct best (distinct by value diff)
        o = np.argsort(np.array(pool_v))
        pruned_X, pruned_v = [], []
        for j in o:
            vv = pool_v[j]
            keep = True
            for pv in pruned_v:
                if abs(vv - pv) < 5e-6:
                    # check Frobenius closeness roughly via value only; allow duplicate values but limit
                    pass
            pruned_X.append(pool_X[j])
            pruned_v.append(vv)
            if len(pruned_X) >= 4:
                break
        pool_X, pool_v = pruned_X, pruned_v

    # choose champion among pool + fallback
    all_X = pool_X + [xb0]
    all_v = pool_v + [best_fallback]
    bi = int(np.argmin(np.array(all_v)))
    x_best = all_X[bi].copy()
    best = float(all_v[bi])

    # Final 20s: one trust-region LP polish
    left = BUDGET - (time.time() - t0)
    lp_time = min(max(left - 1.0, 2.0), 20.0)
    if left > 2.5:
        try:
            x_best = _lp_polish(x_best, lp_time, t0, BUDGET)
            x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
            best = _max_cos(x_best)
        except Exception:
            pass
        # if polish worsened vs fallback, restore fallback polished? ensure best
        if best > best_fallback + 1e-12:
            # try polishing fallback briefly if time remains
            rem2 = BUDGET - (time.time() - t0)
            if rem2 > 2:
                try:
                    xf = _lp_polish(xb0, min(rem2 - 0.5, 5.0), t0, BUDGET)
                    vf = _max_cos(xf)
                    if vf < best:
                        x_best, best = xf, vf
                    else:
                        if best > best_fallback:
                            x_best = xb0.copy()
                except Exception:
                    x_best = xb0.copy()
            else:
                x_best = xb0.copy()
    else:
        if best > best_fallback:
            x_best = xb0.copy()

    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
