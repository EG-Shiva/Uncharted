import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _max_abs_proj(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, 0.0)
    return float(np.abs(G).max())

def _lp_polish_deadline(X, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = X @ X.T
        c = G[iu0, iu1]
        m = c.max()
        sel_thresh = m - max(20 * radius, 1e-9)
        sel = c > sel_thresh
        if np.count_nonzero(sel) == 0:
            break
        I = iu0[sel]; J = iu1[sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X, best

def _greedy_antipodal_proj(X):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, 2.0)
    iu0, iu1 = np.triu_indices(n, 1)
    order = np.argsort(G[iu0, iu1])  # most negative first
    used = np.zeros(n, dtype=bool)
    pairs = []
    for k in range(len(order)):
        i = int(iu0[order[k]]); j = int(iu1[order[k]])
        if (not used[i]) and (not used[j]):
            pairs.append((i, j))
            used[i] = True; used[j] = True
            if len(pairs) == n // 2:
                break
    # fallback for leftovers (should not happen for even n)
    rem = np.where(~used)[0]
    while len(rem) >= 2:
        pairs.append((int(rem[0]), int(rem[1])))
        rem = rem[2:]
    Y = np.empty((n // 2, X.shape[1]))
    for t, (a, b) in enumerate(pairs):
        v = X[a] - X[b]
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            v = np.random.randn(X.shape[1]); nv = np.linalg.norm(v)
        Y[t] = v / nv
    return Y

def _sym_batch_anneal(F, steps, beta0, beta1, lr0, lr1):
    B, m, d = F.shape
    m1 = np.zeros_like(F); v1 = np.zeros_like(F)
    b1, b2, eps = 0.9, 0.999, 1e-8
    Eye = np.eye(m)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = np.einsum('bmd,bnd->bmn', F, F)
        A = np.abs(G)
        # mask diag for max
        A[:, range(m), range(m)] = -1.0
        mval = A.max(axis=(1, 2))
        Epos = np.exp(beta * (G - mval[:, None, None]))
        Eneg = np.exp(beta * (-G - mval[:, None, None]))
        # zero diag
        Epos[:, range(m), range(m)] = 0.0
        Eneg[:, range(m), range(m)] = 0.0
        Z = (Epos + Eneg).sum(axis=(1, 2)) * 0.5 + 1e-30
        H = (Epos - Eneg) / (Z[:, None, None] * 2.0)
        # symmetrize (already symmetric)
        grad = 2.0 * np.einsum('bmn,bnd->bmd', H, F)
        dot = (grad * F).sum(axis=2, keepdims=True)
        gt = grad - dot * F
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1); bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        F = F - upd
        F /= np.linalg.norm(F, axis=2, keepdims=True) + 1e-12
    return F

def _full_batch_combined(F, steps, beta0, beta1, lr0, lr1, target, w_hinge=2.0, w_lse=0.3):
    B, n, d = F.shape
    m1 = np.zeros_like(F); v1 = np.zeros_like(F)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = np.einsum('bnd,bmd->bnm', F, F)
        # LSE grad
        Gtmp = G.copy()
        for b in range(B):
            np.fill_diagonal(Gtmp[b], -10.0)
        mval = Gtmp.max(axis=(1, 2))
        E = np.exp(beta * (Gtmp - mval[:, None, None]))
        for b in range(B):
            np.fill_diagonal(E[b], 0.0)
        Z = E.sum(axis=(1, 2)) * 0.5 + 1e-30
        grad_lse = np.einsum('bnm,bmd->bnd', E / Z[:, None, None], F)
        # hinge grad
        S = G - target
        S[S < 0] = 0.0
        for b in range(B):
            np.fill_diagonal(S[b], 0.0)
        grad_h = 2.0 * np.einsum('bnm,bmd->bnd', S, F)
        grad = w_lse * grad_lse + w_hinge * grad_h
        dot = (grad * F).sum(axis=2, keepdims=True)
        gt = grad - dot * F
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1); bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        F = F - upd
        F /= np.linalg.norm(F, axis=2, keepdims=True) + 1e-12
    return F

def _single_lse_adam(X, steps, beta0, beta1, lr0, lr1):
    n, d = X.shape
    Y = X.copy()
    m1 = np.zeros_like(Y); v1 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        mval = G.max()
        E = np.exp(beta * (G - mval))
        np.fill_diagonal(E, 0.0)
        Z = E.sum() * 0.5 + 1e-30
        grad = (E / Z) @ Y
        dot = (grad * Y).sum(axis=1, keepdims=True)
        gt = grad - dot * Y
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1); bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        Y = Y - upd
        Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
    return Y

def solve():
    t0 = time.time()
    HARD_END = t0 + 178.0
    SYM_BUDGET = 40.0
    MID_END = t0 + 150.0
    rng = np.random.RandomState(7)
    n, d, m = 94, 13, 47
    def load_norm(p):
        try:
            a = np.load(p)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (94, 13):
                return None
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                return None
            return a
        except Exception:
            return None
    xb = load_norm("best.npy")
    xr = load_norm("record.npy")
    cands = [a for a in [xb, xr] if a is not None]
    if not cands:
        a = rng.randn(n, d); a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands = [a]
    X_best = cands[0].copy()
    best_val = _max_cos(X_best)
    for c in cands[1:]:
        v = _max_cos(c)
        if v < best_val:
            best_val = v; X_best = c.copy()
    base_for_pair = X_best.copy()

    # symmetric inits: greedy projections + random
    Yp1 = _greedy_antipodal_proj(base_for_pair)
    if xr is not None:
        Yp2 = _greedy_antipodal_proj(xr)
    else:
        Yp2 = Yp1.copy()
    Bsym = 8
    Fsym = np.empty((Bsym, m, d))
    Fsym[0] = Yp1
    Fsym[1] = Yp2
    for b in range(2, Bsym):
        A = rng.randn(m, d)
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        Fsym[b] = A
    # anneal symmetric stage (two phases to respect time)
    try:
        Fsym = _sym_batch_anneal(Fsym, 700, 25.0, 200.0, 0.03, 0.006)
        if time.time() < t0 + SYM_BUDGET - 8:
            Fsym = _sym_batch_anneal(Fsym, 600, 200.0, 420.0, 0.006, 0.002)
    except Exception:
        pass
    # evaluate symmetric expansions
    sym_vals = []
    for b in range(Bsym):
        Y = Fsym[b] / np.linalg.norm(Fsym[b], axis=1, keepdims=True)
        Xs = np.concatenate([Y, -Y], axis=0)
        v = _max_cos(Xs)
        sym_vals.append(v)
    sym_vals = np.array(sym_vals)
    # symmetry breaking: tangent perturbations of several magnitudes
    sigmas = [0.0, 0.002, 0.008, 0.025]
    full_list = []
    # take all symmetric opts, perturb
    order_sym = np.argsort(sym_vals)
    # limit to top 6 symmetric to bound batch size
    keep = order_sym[:6]
    for b in keep:
        Y = Fsym[b] / np.linalg.norm(Fsym[b], axis=1, keepdims=True)
        Xs0 = np.concatenate([Y, -Y], axis=0)
        for sg in sigmas:
            if sg == 0.0:
                full_list.append(Xs0.copy())
            else:
                Nz = rng.randn(n, d)
                # tangent component
                dot = (Nz * Xs0).sum(axis=1, keepdims=True)
                Nt = Nz - dot * Xs0
                Xp = Xs0 + sg * Nt
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
                full_list.append(Xp)
    # add current best as safety member
    full_list.append(X_best.copy())
    # also add small perturbations of best to allow direct descent
    for sg in [0.005, 0.02]:
        Nz = rng.randn(n, d)
        dot = (Nz * X_best).sum(axis=1, keepdims=True)
        Nt = Nz - dot * X_best
        Xp = X_best + sg * Nt
        Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        full_list.append(Xp)
    Ffull = np.stack(full_list, axis=0)  # (B2,n,d)
    # below-record hinge target
    target = best_val - 0.0012
    # combined hinge+LSE batch
    rem = MID_END - time.time()
    if rem > 20 and Ffull.shape[0] > 0:
        steps = 750
        # adapt steps to batch size/time
        try:
            Ffull = _full_batch_combined(Ffull, steps, 60.0, 320.0, 0.012, 0.0025, target, w_hinge=2.0, w_lse=0.4)
        except Exception:
            pass
    # rank by exact max
    vals = np.array([_max_cos(Ffull[b]) for b in range(Ffull.shape[0])])
    order = np.argsort(vals)
    # try single refinements + LP on top few
    for idx in order[:4]:
        if time.time() > MID_END:
            break
        Xc = Ffull[idx] / np.linalg.norm(Ffull[idx], axis=1, keepdims=True)
        vc = _max_cos(Xc)
        # short LSE adam
        if time.time() < MID_END - 4:
            try:
                Xr = _single_lse_adam(Xc, 220, 120.0, 320.0, 0.007, 0.002)
                vr = _max_cos(Xr)
                if vr < vc:
                    Xc, vc = Xr, vr
            except Exception:
                pass
        if vc < best_val - 1e-12:
            X_best, best_val = Xc, vc
        # LP polish if promising
        if vc < best_val + 0.004:
            budget = 1.5
            if vc < best_val + 0.0005:
                budget = 2.5
            if vc < best_val - 1e-12:
                budget = 3.0
            endt = min(time.time() + budget, HARD_END - 0.5)
            if endt > time.time() + 0.2:
                try:
                    Xp, vp = _lp_polish_deadline(Xc, endt)
                    if vp < best_val:
                        X_best, best_val = Xp, vp
                except Exception:
                    pass
    if time.time() < HARD_END - 0.5:
        try:
            X_best, best_val = _lp_polish_deadline(X_best, HARD_END - 0.2)
        except Exception:
            pass
    X_best = X_best / np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
