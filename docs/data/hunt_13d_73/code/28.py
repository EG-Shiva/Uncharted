import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rowwise_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _contact_counts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    cm = float(G.max())
    return (G >= cm - tol).sum(axis=1), cm

def _pca_outlier(X):
    try:
        M = X - X.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(M, full_matrices=False)
        P = M @ Vt[:2].T
        Pc = P - P.mean(axis=0, keepdims=True)
        r = np.linalg.norm(Pc, axis=1)
        return int(np.argmax(r))
    except Exception:
        return int(np.argmax(np.linalg.norm(X - X.mean(axis=0), axis=1)))

def _lp_polish(X, seconds, t_end=1e30):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        cfull = X @ X.T
        c = cfull[iu0]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
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

def _block_lbfgs(F, B0, alpha=150.0, maxiter=200):
    k, d = B0.shape
    m = F.shape[0]
    iu_k = np.triu_indices(k, 1) if k > 1 else (np.array([], dtype=int), np.array([], dtype=int))
    has_bb = k > 1
    x0 = B0.ravel().copy()
    def fun(xf):
        B = xf.reshape(k, d)
        rn = np.linalg.norm(B, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        Bn = B / rn
        Dbf = Bn @ F.T
        if has_bb:
            Gbb = Bn @ Bn.T
            dbb = Gbb[iu_k]
            M = max(float(Dbf.max()), float(dbb.max()))
            e_bf = np.exp(alpha * (Dbf - M))
            e_bb = np.exp(alpha * (dbb - M))
            S = e_bf.sum() + e_bb.sum()
            loss = M + np.log(S) / alpha
            pbf = e_bf / S
            pbb = e_bb / S
            GB = pbf @ F
            W = np.zeros((k, k))
            W[iu_k] = pbb
            W = W + W.T
            GB = GB + W @ Bn
        else:
            M = float(Dbf.max())
            e_bf = np.exp(alpha * (Dbf - M))
            S = e_bf.sum()
            loss = M + np.log(S) / alpha
            pbf = e_bf / S
            GB = pbf @ F
        dg = (GB * Bn).sum(axis=1, keepdims=True)
        Gt = GB - dg * Bn
        grad = (Gt / rn).ravel()
        return float(loss), grad.astype(float)
    try:
        res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-7, "maxcor": 8})
        B = res.x.reshape(k, d)
    except Exception:
        B = B0
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return B

def _full_relax(X0, alpha=120.0, maxiter=40):
    n, d = X0.shape
    iu = np.triu_indices(n, 1)
    x0 = X0.ravel().copy()
    def fun(xf):
        Y = xf.reshape(n, d)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X = Y / rn
        G = X @ X.T
        dots = G[iu]
        M = float(dots.max())
        e = np.exp(alpha * (dots - M))
        S = float(e.sum())
        loss = M + np.log(S) / alpha
        p = e / S
        W = np.zeros((n, n))
        W[iu] = p
        W = W + W.T
        GX = W @ X
        dg = (GX * X).sum(axis=1, keepdims=True)
        Gt = GX - dg * X
        grad = (Gt / rn).ravel()
        return float(loss), grad.astype(float)
    try:
        res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-7, "maxcor": 8})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = X0.copy()
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def solve():
    t0 = time.time()
    SURGERY_END = t0 + 150.0
    DEADLINE = t0 + 172.0
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
        try:
            tmp[:m0, :m1] = Xb[:m0, :m1]
        except Exception:
            pass
        Xb = tmp
    Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    best_val = _max_cos(Xb)
    Xbest = Xb.copy()
    cap_sizes = [10, 12, 14, 16, 11, 13, 15]
    it = 0
    NCAND = 80000
    CHUNK = 20000
    while time.time() < SURGERY_END and time.time() < DEADLINE - 12.0:
        it += 1
        mode = it % 4
        k = cap_sizes[it % len(cap_sizes)]
        # choose cap center
        try:
            rm = _rowwise_max(Xbest)
            rattler = int(np.argmin(rm))
            cc, _ = _contact_counts(Xbest, tol=1e-7)
            highly = int(np.argmax(cc))
            out = _pca_outlier(Xbest)
            if mode == 0:
                ci = rattler
            elif mode == 1:
                ci = int(rng.integers(0, n))
            elif mode == 2:
                ci = highly
            else:
                ci = out
            center = Xbest[ci].copy()
            center /= np.linalg.norm(center)
            dots_c = Xbest @ center
            order = np.argsort(-dots_c)
            del_idx = order[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[del_idx] = False
            F = Xbest[keep_mask].copy()
        except Exception:
            continue
        # greedy rebuild
        rebuilt = []
        ok = True
        for v in range(k):
            if time.time() > SURGERY_END or time.time() > DEADLINE - 12.0:
                ok = False
                break
            if len(rebuilt) == 0:
                A = F
            else:
                A = np.vstack([F] + [r[None, :] for r in rebuilt])
            best_c = None
            best_s = 2.0
            topC = None
            # chunked sampling to bound memory
            gbest_s = np.full(3, 2.0)
            gbest_v = np.zeros((3, d))
            n_chunks = NCAND // CHUNK
            for _ch in range(n_chunks):
                try:
                    C = rng.normal(size=(CHUNK, d))
                    C /= np.linalg.norm(C, axis=1, keepdims=True)
                    D = C @ A.T
                    s = D.max(axis=1)
                    # top 3 in chunk
                    idx = np.argpartition(s, 3)[:3]
                    for t_ in range(3):
                        cand = C[idx[t_]]
                        sc = float(s[idx[t_]])
                        # insert into global top3
                        w = int(np.argmax(gbest_s))
                        if sc < gbest_s[w]:
                            gbest_s[w] = sc
                            gbest_v[w] = cand
                except Exception:
                    continue
            # refine best few by local random search (soft-max approx = max here)
            refined = []
            for b in range(3):
                c0 = gbest_v[b] / np.linalg.norm(gbest_v[b])
                try:
                    D0 = c0 @ A.T
                    s0 = float(D0.max())
                except Exception:
                    s0 = 2.0
                bc = c0; bs = s0
                for rnd, sig in ((0, 0.03), (1, 0.012)):
                    try:
                        P = bc[None, :] + sig * rng.normal(size=(2000, d))
                        P /= np.linalg.norm(P, axis=1, keepdims=True)
                        Dp = P @ A.T
                        sp = Dp.max(axis=1)
                        bi = int(np.argmin(sp))
                        if float(sp[bi]) < bs:
                            bs = float(sp[bi]); bc = P[bi]
                    except Exception:
                        pass
                refined.append((bs, bc))
            refined.sort(key=lambda z: z[0])
            best_s, best_c = refined[0][1], refined[0][0]  # careful swap
            # refined tuples are (bs,bc); above swapped names: fix
            bs_win, bc_win = refined[0]
            rebuilt.append(bc_win / np.linalg.norm(bc_win))
        if not ok:
            continue
        if len(rebuilt) != k:
            continue
        B0 = np.vstack(rebuilt)
        # block-only optimization with frozen obstacles
        try:
            B1 = _block_lbfgs(F, B0, alpha=150.0, maxiter=200)
        except Exception:
            B1 = B0
        # assemble full candidate
        try:
            Xtry = Xbest.copy()
            for pos, idx in enumerate(del_idx):
                Xtry[idx] = B1[pos]
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        except Exception:
            continue
        # unfreeze 1-sec smooth relaxation
        try:
            Xtry2 = _full_relax(Xtry, alpha=120.0, maxiter=35)
        except Exception:
            Xtry2 = Xtry
        try:
            v = _max_cos(Xtry2)
        except Exception:
            continue
        # consider LP polish only if within 0.001 or better
        if v <= best_val + 0.001 + 1e-12:
            # only if time allows
            if time.time() < SURGERY_END and time.time() < DEADLINE - 10.0:
                try:
                    Xp = _lp_polish(Xtry2, seconds=min(6.0, max(0.5, (DEADLINE - 3.0) - time.time())), t_end=min(SURGERY_END, DEADLINE - 2.0))
                    vp = _max_cos(Xp)
                    if vp < v:
                        Xtry2, v = Xp, vp
                except Exception:
                    pass
        if v < best_val - 1e-12:
            Xbest = Xtry2.copy()
            best_val = float(v)
            if best_val < 0.19:
                break
        # loop continues cycling centers/sizes
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
