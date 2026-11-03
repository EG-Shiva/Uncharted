import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _row_maxima(G):
    A = G.copy()
    np.fill_diagonal(A, -2.0)
    return np.max(A, axis=1)

def _count_close(G, m, tol=1e-6):
    iu0, iu1 = np.triu_indices(G.shape[0], 1)
    c = G[iu0, iu1]
    return int(np.sum(c > m - tol))

def _subset_lse_polish(Xbase, opt_idx, alpha, maxiter):
    # optimize only rows opt_idx, others frozen
    m = len(opt_idx)
    if m == 0:
        return Xbase.copy()
    d = Xbase.shape[1]
    N = Xbase.shape[0]
    opt_idx = np.asarray(opt_idx, dtype=int)
    frozen = Xbase.copy()
    V0 = Xbase[opt_idx].copy().ravel()
    def fun(vf):
        V = vf.reshape(m, d)
        nr = np.linalg.norm(V, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        Y = V / nr
        X = frozen.copy()
        X[opt_idx] = Y
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(np.max(G))
        E = np.exp(alpha * (G - mx))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5) + 1e-300
        L = mx + np.log(S) / alpha
        GX = (E @ X) / S
        # grad w.r.t V
        dot = np.sum(GX[opt_idx] * Y, axis=1, keepdims=True)
        GV = (GX[opt_idx] - dot * Y) / nr
        return float(L), GV.ravel().astype(float)
    try:
        res = minimize(fun, V0, method='L-BFGS-B', jac=True,
                       options={'maxiter': int(maxiter), 'ftol': 1e-20,
                                'gtol': 1e-9, 'maxfun': int(maxiter)*5, 'maxcor': 8})
        V = res.x.reshape(m, d)
        X = frozen.copy()
        X[opt_idx] = _norm_rows(V)
        return _norm_rows(X)
    except Exception:
        return Xbase.copy()

def _global_lse_polish(X0, alpha, maxiter):
    N, d = X0.shape
    U0 = X0.ravel().copy()
    def fun(uf):
        U = uf.reshape(N, d)
        nr = np.linalg.norm(U, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = U / nr
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(np.max(G))
        E = np.exp(alpha * (G - mx))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5) + 1e-300
        L = mx + np.log(S) / alpha
        GX = (E @ X) / S
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GU = (GX - dot * X) / nr
        return float(L), GU.ravel().astype(float)
    try:
        res = minimize(fun, U0, method='L-BFGS-B', jac=True,
                       options={'maxiter': int(maxiter), 'ftol': 1e-20,
                                'gtol': 1e-9, 'maxfun': int(maxiter)*5, 'maxcor': 8})
        U = res.x.reshape(N, d)
        return _norm_rows(U)
    except Exception:
        return X0.copy()

def solve():
    t0 = time.time()
    HARD = 172.0
    CYCLE_BUDGET = 150.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and np.shape(xb) == (99, 13):
        cands.append(_norm_rows(np.asarray(xb, float)))
    if xr is not None and np.shape(xr) == (99, 13):
        cands.append(_norm_rows(np.asarray(xr, float)))
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        cands.append(_norm_rows(rng0.standard_normal(size=(99, 13))))
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    Xcur = cands[bi].copy()
    mcur = float(vals[bi])
    Gcur = Xcur @ Xcur.T
    cntcur = _count_close(Gcur, mcur, 1e-6)
    best_X = Xcur.copy()
    best_m = mcur
    n, d = 99, 13
    NCAND = 50000
    MAXCYC = 20
    # deterministic master rng
    master = np.random.default_rng(20261103)
    # pre-drawn seeds for candidate clouds to stay deterministic
    cand_seeds = master.integers(0, 2**31 - 1, size=MAXCYC)
    for cyc in range(MAXCYC):
        if time.time() - t0 > CYCLE_BUDGET:
            break
        G = Xcur @ Xcur.T
        # argmax pair computationally
        A = G.copy()
        np.fill_diagonal(A, -2.0)
        flat = int(np.argmax(A))
        a = flat // n
        b = flat % n
        mchk = float(G[a, b])
        rowmax = _row_maxima(G)
        # one endpoint of controlling pair
        # alternate which endpoint, bias to larger second-best
        if (cyc % 2) == 0:
            rem0 = a
        else:
            rem0 = b
        # plateau neighbourhood within 1e-5
        plat = np.where(rowmax > mchk - 1e-5)[0]
        plat = [int(i) for i in plat if int(i) != rem0 and int(i) != a and int(i) != b]
        # number of additional points 2..6 cycling
        kadd = 2 + (cyc % 5)  # 2,3,4,5,6
        # sample preferentially from plateau
        rem = [int(rem0)]
        if len(plat) > 0:
            # weight by closeness to max
            w = np.array([rowmax[i] for i in plat], dtype=float)
            w = w - np.min(w) + 1e-12
            # sharper preference for top rows
            w = w + 0.2 * np.max(w)
            s = float(np.sum(w))
            if not np.isfinite(s) or s <= 0:
                prob = None
            else:
                prob = w / s
            rng_s = np.random.default_rng(int(cand_seeds[cyc]) % (2**31 - 1) + 1000 + cyc)
            kk = min(kadd, len(plat))
            try:
                chosen = rng_s.choice(len(plat), size=kk, replace=False, p=prob)
            except Exception:
                chosen = rng_s.choice(len(plat), size=kk, replace=False)
            for ci in chosen:
                rem.append(int(plat[int(ci)]))
        # if still need more (plateau too small), fill by top rowmax
        if len(rem) - 1 < kadd:
            order = np.argsort(-rowmax)
            for oi in order:
                oi = int(oi)
                if oi not in rem and oi != a and oi != b:
                    rem.append(oi)
                if len(rem) - 1 >= kadd:
                    break
        rem = sorted(set(rem))
        frozen_idx = np.array([i for i in range(n) if i not in set(rem)], dtype=int)
        Xf = Xcur[frozen_idx]
        # generate >=50000 random sphere candidates
        rng_c = np.random.default_rng(int(cand_seeds[cyc]) % (2**31 - 1))
        Cand = rng_c.standard_normal(size=(NCAND, d))
        Cand = _norm_rows(Cand)
        # greedy farthest insertion: minimize max dot with frozen+already inserted
        Df = Cand @ Xf.T  # (NCAND, nF)
        maxf = np.max(Df, axis=1)  # (NCAND,)
        inserted = []
        InsMat = np.zeros((0, d))
        best_vals = maxf.copy()
        # greedy loop
        used = np.zeros(NCAND, dtype=bool)
        for t in range(len(rem)):
            if t == 0:
                scores = maxf
            else:
                Di = Cand @ InsMat.T
                maxi = np.max(Di, axis=1)
                scores = np.maximum(maxf, maxi)
            scores = np.where(used, np.inf, scores)
            bi2 = int(np.argmin(scores))
            used[bi2] = True
            inserted.append(Cand[bi2].copy())
            InsMat = np.vstack([InsMat, Cand[bi2:bi2+1]]) if len(inserted) > 0 else Cand[bi2:bi2+1]
        Xtrial = Xcur.copy()
        for k, ri in enumerate(rem):
            Xtrial[ri] = inserted[k] / max(1e-12, np.linalg.norm(inserted[k]))
        Xtrial = _norm_rows(Xtrial)
        # optimize only inserted + neighbours within slack 5e-6
        Gt = Xtrial @ Xtrial.T
        mt = _max_cos(Xtrial)
        rmt = _row_maxima(Gt + np.eye(n) * -10.0 + np.eye(n) * 0.0)  # diag already? recompute properly
        # _row_maxima expects diag handling internally? Gt diag=1, so compute manually
        A2 = Gt.copy()
        np.fill_diagonal(A2, -2.0)
        rmt = np.max(A2, axis=1)
        neigh = set(rem)
        for i in range(n):
            if rmt[i] > mt - 5e-6:
                neigh.add(int(i))
        # also add any point with a pair > mt-5e-6 to an inserted point
        for ri in rem:
            row = Gt[ri].copy()
            row[ri] = -2.0
            hits = np.where(row > mt - 5e-6)[0]
            for h in hits:
                neigh.add(int(h))
        opt_idx = sorted(neigh)
        # cap size to keep fast: if huge, keep most critical 40
        if len(opt_idx) > 40:
            # keep all rem + top by rowmax
            rest = [i for i in opt_idx if i not in set(rem)]
            rest_sorted = sorted(rest, key=lambda i: -rmt[i])
            keep = set(rem) | set(rest_sorted[:40 - len(rem)])
            opt_idx = sorted(keep)
        Xopt = Xtrial.copy()
        # annealed LogSumExp on subset
        for alpha, mit in ((100.0, 50), (500.0, 50), (2000.0, 60)):
            if time.time() - t0 > CYCLE_BUDGET:
                break
            Xopt = _subset_lse_polish(Xopt, opt_idx, alpha, mit)
        # brief global polish
        if time.time() - t0 < CYCLE_BUDGET - 1.0:
            Xopt = _global_lse_polish(Xopt, 500.0, 25)
        Xopt = _norm_rows(Xopt)
        Gopt = Xopt @ Xopt.T
        mopt = _max_cos(Xopt)
        cntopt = _count_close(Gopt, mopt, 1e-6)
        # retain only if exact score improves or close-pair count materially decreases
        improved = False
        if mopt < mcur - 1e-12:
            improved = True
        elif mopt <= mcur + 1e-12 and cntopt <= cntcur - 20:
            improved = True
        elif mopt <= mcur + 1e-12 and cntopt < int(0.9 * cntcur) and cntopt < cntcur:
            improved = True
        if improved:
            Xcur = Xopt.copy()
            mcur = mopt
            cntcur = cntopt
            if mopt < best_m:
                best_m = mopt
                best_X = Xopt.copy()
        # else revert to checkpoint (do nothing)
        if time.time() - t0 > HARD - 8.0:
            break
    # final quick squeeze if time remains
    if time.time() - t0 < HARD - 6.0:
        try:
            Y = _global_lse_polish(best_X, 2000.0, 40)
            my = _max_cos(Y)
            if my < best_m:
                best_X = Y.copy()
                best_m = my
        except Exception:
            pass
    return _norm_rows(np.asarray(best_X, dtype=float))
