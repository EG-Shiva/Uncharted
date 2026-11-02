import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contact_counts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = (G >= m - tol).sum(axis=1)
    return cnt, m, G

def _lp_polish(X, seconds, t_end=1e30):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        c = (X @ X.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
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
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _refine_candidate(c0, obstacles, alpha=90.0, steps=150, lr0=0.08, lr1=0.005):
    d = c0.shape[0]
    c = c0.copy()
    c /= np.linalg.norm(c)
    for k in range(steps):
        frac = k / max(steps - 1, 1)
        lr = lr0 * (1 - frac) + lr1 * frac
        dots = obstacles @ c  # (M,)
        M = dots.max()
        e = np.exp(alpha * (dots - M))
        S = e.sum()
        if S == 0 or not np.isfinite(S):
            break
        p = e / S
        g = p @ obstacles  # (d,)
        gt = g - (g @ c) * c
        c = c - lr * gt
        n = np.linalg.norm(c)
        if n == 0 or not np.isfinite(n):
            return c0 / np.linalg.norm(c0)
        c /= n
    return c

def _block_lbfgs(Y0, F, alpha=140.0, maxiter=250):
    m, d = Y0.shape
    Y = Y0.ravel().copy()
    iu0 = np.triu_indices(m, 1) if m > 1 else (np.array([], dtype=int), np.array([], dtype=int))
    has_f = F.shape[0] > 0
    def fun(Yflat):
        Ym = Yflat.reshape(m, d)
        rn = np.linalg.norm(Ym, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        Xb = Ym / rn
        if m > 1:
            G = Xb @ Xb.T
            di = G[iu0]
        else:
            di = np.zeros(0)
        if has_f:
            dc = (Xb @ F.T).ravel()
            if di.size:
                dots = np.concatenate([di, dc])
            else:
                dots = dc
        else:
            dots = di
        M = dots.max()
        e = np.exp(alpha * (dots - M))
        S = e.sum()
        loss = M + np.log(S) / alpha
        p = e / S
        # split p
        grad_Xb = np.zeros_like(Xb)
        off = 0
        if m > 1:
            nint = di.size
            pi = p[:nint]
            W = np.zeros((m, m))
            W[iu0] = pi
            W = W + W.T
            grad_Xb += W @ Xb
            off = nint
        if has_f:
            pc = p[off:].reshape(m, -1)
            grad_Xb += pc @ F
        dg = (grad_Xb * Xb).sum(axis=1, keepdims=True)
        grad_Xt = grad_Xb - dg * Xb
        grad_Y = (grad_Xt / rn).ravel()
        return float(loss), grad_Y.astype(float)
    try:
        res = minimize(fun, Y, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-7, "maxcor": 10})
        Y = res.x
    except Exception:
        pass
    Ym = Y.reshape(m, d)
    Ym /= np.linalg.norm(Ym, axis=1, keepdims=True)
    return Ym

def _adam_relax(X0, budget, s0=160.0, s1=320.0, lr=0.004, t_end=1e30):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X); v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    start = time.time()
    steps = 0
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < budget and time.time() < t_end:
        frac = 0.0
        # use time frac for schedule if budget known
        el = (time.time() - start) / max(budget, 1e-9)
        el = min(max(el, 0.0), 1.0)
        s = s0 + (s1 - s0) * el
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = G.max()
        dots = G[iu0]
        E = np.exp(s * (dots - cmax))
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            break
        P = E / S
        W = np.zeros((n, n))
        W[iu0] = P
        W = W + W.T
        Gt = W @ X
        dot = (Gt * X).sum(axis=1, keepdims=True)
        gt = Gt - dot * X
        m1 *= b1; m1 += (1 - b1) * gt
        v1 *= b2; v1 += (1 - b2) * (gt * gt)
        mhat = m1 / (1 - b1p); vhat = v1 / (1 - b2p)
        b1p *= b1; b2p *= b2
        X -= lr * mhat / (np.sqrt(vhat) + eps)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        steps += 1
        if steps > 400:
            break
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SURG_END = t0 + 150.0
    rng = np.random.default_rng(0)
    n, d = 73, 13
    try:
        Xb = np.load("best.npy")
    except Exception:
        try:
            Xb = np.load("record.npy")
        except Exception:
            Xb = rng.normal(size=(n, d))
    Xb = np.asarray(Xb, dtype=float)
    if Xb.shape != (n, d):
        tmp = rng.normal(size=(n, d))
        m0 = min(Xb.shape[0], n); m1 = min(Xb.shape[1], d)
        tmp[:m0, :m1] = Xb[:m0, :m1]
        Xb = tmp
    Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    best_val = _max_cos(Xb)
    X_best = Xb.copy()
    cap_cycle = [10, 12, 14, 16, 11, 13, 15, 10, 16, 12]
    surg = 0
    NCAND = 80000
    CHUNK = 10000
    while time.time() < SURG_END and time.time() < DEADLINE - 12.0:
        if time.time() > SURG_END:
            break
        mode = surg % 4
        k = cap_cycle[surg % len(cap_cycle)]
        # choose cap center
        try:
            cnt, mval, Gfull = _contact_counts(X_best, tol=1e-9)
        except Exception:
            cnt = np.zeros(n); mval = best_val
        try:
            if mode == 0:
                idx_c = int(np.argmin(cnt))
                center = X_best[idx_c].copy()
            elif mode == 1:
                idx_c = int(rng.integers(0, n))
                center = X_best[idx_c].copy()
            elif mode == 2:
                idx_c = int(np.argmax(cnt))
                center = X_best[idx_c].copy()
            else:
                # PCA outlier: top-2 PCs
                Xc = X_best - X_best.mean(axis=0, keepdims=True)
                try:
                    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                    P2 = (Xc @ Vt[:2].T)
                except Exception:
                    P2 = Xc[:, :2]
                med = np.median(P2, axis=0, keepdims=True)
                dists = np.linalg.norm(P2 - med, axis=1)
                idx_c = int(np.argmax(dists))
                center = X_best[idx_c].copy()
        except Exception:
            idx_c = int(rng.integers(0, n))
            center = X_best[idx_c].copy()
        dots_c = X_best @ center
        del_idx = np.argsort(-dots_c)[:k]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        F = X_best[keep_mask].copy()
        rebuilt = []
        # greedy rebuild one at a time
        ok = True
        for vi in range(k):
            if time.time() > SURG_END or time.time() > DEADLINE - 12.0:
                ok = False
                break
            if rebuilt:
                Rb = np.vstack(rebuilt)
                obstacles = np.vstack([F, Rb])
            else:
                obstacles = F
            # sample candidates in chunks, keep top few
            # keep best 8
            top_vals = None; top_vecs = None
            # we accumulate best across chunks
            best_chunk_vals = np.full(8, np.inf)
            best_chunk_vecs = np.zeros((8, d))
            for s in range(NCAND // CHUNK):
                C = rng.normal(size=(CHUNK, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                dots = C @ obstacles.T
                mv = dots.max(axis=1)
                # merge
                # combine current chunk with stored best
                # take 8 smallest among concat
                if s == 0:
                    bi = np.argsort(mv)[:8]
                    best_chunk_vals = mv[bi].copy()
                    best_chunk_vecs = C[bi].copy()
                else:
                    allv = np.concatenate([best_chunk_vals, mv])
                    # need vecs for mv part
                    # to avoid storing all C, find threshold
                    # get indices of 8 smallest
                    bi = np.argpartition(allv, 8)[:8]
                    # reconstruct vecs: first 8 are old, rest are new
                    new_vecs = np.zeros((8, d))
                    new_vals = np.zeros(8)
                    for t, ii in enumerate(bi):
                        if ii < 8:
                            new_vals[t] = best_chunk_vals[ii]
                            new_vecs[t] = best_chunk_vecs[ii]
                        else:
                            jj = ii - 8
                            new_vals[t] = mv[jj]
                            new_vecs[t] = C[jj]
                    # sort
                    o = np.argsort(new_vals)
                    best_chunk_vals = new_vals[o]
                    best_chunk_vecs = new_vecs[o]
            # refine best few (top 5) by local softmax
            refined = []
            for t in range(min(5, len(best_chunk_vecs))):
                try:
                    rc = _refine_candidate(best_chunk_vecs[t], obstacles, alpha=90.0, steps=120)
                except Exception:
                    rc = best_chunk_vecs[t]
                # score
                try:
                    mv2 = float((obstacles @ rc).max())
                except Exception:
                    mv2 = float('inf')
                refined.append((mv2, rc))
            refined.sort(key=lambda x: x[0])
            winner = refined[0][1]
            winner /= np.linalg.norm(winner)
            rebuilt.append(winner)
        if not ok or len(rebuilt) != k:
            surg += 1
            continue
        Rb = np.vstack(rebuilt)
        # block-only optimization with frozen obstacles
        try:
            Rb_opt = _block_lbfgs(Rb, F, alpha=140.0, maxiter=250)
        except Exception:
            Rb_opt = Rb
        X_new = X_best.copy()
        X_new[del_idx] = Rb_opt
        X_new /= np.linalg.norm(X_new, axis=1, keepdims=True)
        # unfreeze 1-second smooth relaxation
        try:
            remain1 = min(1.0, max(0.2, SURG_END - time.time(), DEADLINE - 12.0 - time.time()))
            if remain1 > 0.15:
                X_new = _adam_relax(X_new, budget=remain1, s0=160.0, s1=320.0, lr=0.004, t_end=DEADLINE - 9.0)
        except Exception:
            pass
        try:
            v_new = _max_cos(X_new)
        except Exception:
            surg += 1
            continue
        # conditional LP polish only if within 0.001 or better
        if v_new <= best_val + 0.001 + 1e-12:
            try:
                lp_budget = min(6.0, max(0.5, SURG_END - time.time(), DEADLINE - 8.0 - time.time()))
                if lp_budget > 0.4 and time.time() < DEADLINE - 8.0:
                    Xp = _lp_polish(X_new, seconds=lp_budget, t_end=min(SURG_END, DEADLINE - 2.0))
                    vp = _max_cos(Xp)
                    if vp < v_new:
                        X_new, v_new = Xp, vp
            except Exception:
                pass
        if v_new < best_val - 1e-12:
            X_best = X_new.copy()
            best_val = float(v_new)
        surg += 1
        if best_val < 0.19:
            break
    # final safety: ensure normalized
    X_best /= np.linalg.norm(X_best, axis=1, keepdims=True)
    return X_best
