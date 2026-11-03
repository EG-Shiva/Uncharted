import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _e8_roots():
    pts = []
    # 112: two coords +-1
    for i in range(8):
        for j in range(i+1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8)
                    v[i] = si
                    v[j] = sj
                    pts.append(v)
    # 128: all +-1/2 even minus
    # iterate 0..255
    for mask in range(256):
        signs = np.ones(8)
        nm = 0
        for k in range(8):
            if (mask >> k) & 1:
                signs[k] = -1.0
                nm += 1
        if nm % 2 == 0:
            pts.append(0.5 * signs)
    A = np.array(pts, dtype=float)
    # norm sqrt(2) -> unit
    A = _norm(A)
    return A  # 240x8

def _greedy_delete(roots, seed, ndel=78):
    rng = np.random.default_rng(seed)
    idx = list(range(roots.shape[0]))
    for _ in range(ndel):
        X = roots[idx]
        n = len(idx)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        gmax = rowmax.max()
        # candidates attaining global max
        cand = np.where(rowmax >= gmax - 1e-12)[0]
        if len(cand) == 1:
            # add slight diversification: occasionally remove 2nd worst
            # check second max
            order = np.argsort(-rowmax)
            if rng.random() < 0.15:
                choice = int(order[1])
            else:
                choice = int(cand[0])
            del idx[choice]
            continue
        # contact count vs global max
        cnt = (G[cand][:, :] > gmax - 1e-12).sum(axis=1) if False else None
        # compute counts
        counts = np.array([int((G[c, :] > gmax - 1e-12).sum()) for c in cand])
        mc = counts.max()
        tied = cand[counts == mc]
        # randomised tie break; occasionally pick second-best tier for diversity
        if len(tied) > 1:
            pick = int(rng.choice(tied))
        else:
            # 10% pick among top-3 counts
            if rng.random() < 0.12 and len(cand) > 3:
                o = np.argsort(-counts)[:min(4, len(cand))]
                pick = int(cand[int(rng.choice(o))])
            else:
                pick = int(tied[0])
        del idx[pick]
    return roots[idx].copy()

def _anneal_lbfgs(X0, betas, maxiters, deadline=None):
    X = _norm(X0.copy())
    Y = X.copy()
    n, d = X.shape
    iu = np.triu_indices(n, 1)
    for bi, beta in enumerate(betas):
        mi = maxiters[bi] if bi < len(maxiters) else maxiters[-1]
        # closure
        def fun_flat(yf):
            Ym = yf.reshape(n, d)
            nr = np.linalg.norm(Ym, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            Xm = Ym / nr
            G = Xm @ Xm.T
            # use shift M
            # compute only upper triangle for stability but need full for grad
            np.fill_diagonal(G, -2.0)
            M = float(G.max())
            E = np.exp(beta * (G - M))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum() * 0.5)
            if S <= 0 or not np.finite(S):
                S = 1e-300
            f = M + np.log(S) / beta
            # row sums
            rs = E.sum(axis=1)  # includes both triangles
            # P unordered sums to 1: P_ij = E_ij/(2S)
            # grad_Xk = sum_j E_kj/(2S) * Xj
            W = E / (2.0 * S)
            Gx = W @ Xm
            # project to tangent and divide by r
            dot = (Gx * Xm).sum(axis=1, keepdims=True)
            Gt = Gx - dot * Xm
            r = nr.ravel()
            Gy = Gt / r[:, None]
            return float(f), Gy.ravel().astype(float)
        try:
            res = minimize(fun_flat, Y.ravel(), jac=True, method='L-BFGS-B',
                           options={'maxiter': int(mi), 'ftol': 1e-14, 'gtol': 1e-7, 'maxfun': int(mi*3+10)})
            Y = res.x.reshape(n, d)
            X = _norm(Y)
            Y = X.copy()
        except Exception:
            pass
        if deadline is not None and time.time() >= deadline:
            break
    return _norm(Y)

def _hole_replace(X, victims, rng, nsamp=8000):
    n, d = X.shape
    keep = np.ones(n, dtype=bool)
    keep[victims] = False
    Xk = X[keep]
    C = rng.standard_normal((nsamp, d))
    C = _norm(C)
    S = C @ Xk.T
    scores = S.max(axis=1)
    order = np.argsort(scores)
    picked = []
    for idx in order:
        v = C[idx]
        ok = True
        for q in picked:
            if float(v @ q) > 0.6:
                ok = False
                break
        if ok:
            picked.append(v)
            if len(picked) >= len(victims):
                break
    while len(picked) < len(victims):
        v = rng.standard_normal(d)
        v = v / np.linalg.norm(v)
        picked.append(v)
    Xn = X.copy()
    Xn[victims] = np.array(picked)
    return _norm(Xn)

def _lp_polish(X, seconds, deadline=None):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    t0 = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - t0 >= seconds:
            break
        if deadline is not None and now >= deadline:
            break
        if radius < 1e-14:
            break
        Gfull = X @ X.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        # cap for speed
        if p > 3000:
            # keep largest 3000
            cs = c[sel]
            k = np.argpartition(-cs, 3000)[:3000]
            I, J = I[k], J[k]
            cs = cs[k]
            p = 3000
        else:
            cs = c[sel]
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
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    np.random.seed(0)
    t0 = time.time()
    DEADLINE = t0 + 172.0
    roots = _e8_roots()
    # fallback best
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (162, 8):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(np.random.default_rng(0).standard_normal((162, 8)))
        best_val = _max_cos(best)
    base_best = best.copy()
    base_val = best_val
    # build E8 starts
    rng = np.random.default_rng(12345)
    starts = []
    # 6 greedy deletions
    for s in range(6):
        try:
            Xs = _greedy_delete(roots, seed=1000 + s)
            starts.append(Xs)
        except Exception:
            pass
    # 3 random subsets
    for s in range(3):
        idx = rng.choice(240, size=162, replace=False)
        starts.append(_norm(roots[idx].copy()))
    # small jitter to break exact symmetry
    for i in range(len(starts)):
        starts[i] = _norm(starts[i] + 1e-4 * np.random.default_rng(2000 + i).standard_normal((162, 8)))
    # staged screening
    scored = []
    for si, Xs in enumerate(starts):
        if time.time() >= DEADLINE - 45.0:
            break
        # light anneal first
        try:
            Xa = _anneal_lbfgs(Xs, betas=[30.0, 100.0, 300.0], maxiters=[120, 120, 150],
                               deadline=DEADLINE - 40.0)
        except Exception:
            continue
        v = _max_cos(Xa)
        # rattler / imbalance repair: if rattlers exist, swap them into holes and re-anneal briefly
        try:
            G = Xa @ Xa.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            cnt = (G > m - 1e-4).sum(axis=1)
            ratt = np.where(cnt == 0)[0]
            # also very low (<=2) as victims if many overloaded
            if len(ratt) > 0 and len(ratt) <= 6:
                r2 = np.random.default_rng(5000 + si)
                Xa2 = _hole_replace(Xa, sorted(ratt.tolist()), r2)
                Xa2 = _anneal_lbfgs(Xa2, betas=[100.0, 300.0], maxiters=[80, 100], deadline=DEADLINE - 40.0)
                v2 = _max_cos(Xa2)
                if v2 < v:
                    Xa, v = Xa2, v2
            elif (cnt > 22).sum() > 4:
                # swap 2 overloaded hubs into holes
                order_hi = np.argsort(-cnt)[:2]
                r2 = np.random.default_rng(6000 + si)
                Xa2 = _hole_replace(Xa, sorted(order_hi.tolist()), r2)
                Xa2 = _anneal_lbfgs(Xa2, betas=[100.0, 300.0], maxiters=[80, 100], deadline=DEADLINE - 40.0)
                v2 = _max_cos(Xa2)
                if v2 < v + 0.001:
                    # keep better of the two for further sharpening
                    if v2 < v:
                        Xa, v = Xa2, v2
        except Exception:
            pass
        scored.append((v, Xa))
        if v < best_val - 1e-12:
            best_val = v
            best = Xa.copy()
    # sharpen top candidates with beta 1000 + LP
    if scored:
        scored.sort(key=lambda t: t[0])
        for k in range(min(3, len(scored))):
            if time.time() >= DEADLINE - 18.0:
                break
            v0, X0c = scored[k]
            # only promising ones (within 0.004 of global best)
            if v0 > min(best_val, base_val) + 0.004:
                continue
            try:
                X1 = _anneal_lbfgs(X0c, betas=[1000.0], maxiters=[200], deadline=DEADLINE - 14.0)
                v1 = _max_cos(X1)
            except Exception:
                X1, v1 = X0c, v0
            if v1 < best_val - 1e-12:
                best_val = v1
                best = X1.copy()
            # LP polish if promising
            remain = DEADLINE - time.time()
            if remain > 8.0 and v1 < min(best_val, base_val) + 0.002:
                try:
                    Xp = _lp_polish(X1, seconds=min(4.0, remain - 3.0), deadline=DEADLINE - 2.0)
                    vp = _max_cos(Xp)
                    if vp < best_val:
                        best_val = vp
                        best = Xp.copy()
                except Exception:
                    pass
    # final safety: if E8 path did not beat base, polish base briefly
    if best_val >= base_val - 1e-12:
        # E8 did not win; ensure we return at least base polished a bit
        remain = DEADLINE - time.time()
        if remain > 4.0:
            try:
                Xp = _lp_polish(base_best, seconds=min(6.0, remain - 1.0), deadline=DEADLINE - 1.0)
                vp = _max_cos(Xp)
                if vp < best_val:
                    best = Xp
                    best_val = vp
                else:
                    best = base_best
                    best_val = base_val
            except Exception:
                best = base_best
        else:
            best = base_best
    else:
        # final polish of winner
        remain = DEADLINE - time.time()
        if remain > 3.0:
            try:
                Xp = _lp_polish(best, seconds=min(8.0, remain - 1.0), deadline=DEADLINE - 0.5)
                vp = _max_cos(Xp)
                if vp < best_val:
                    best = Xp
            except Exception:
                pass
    best = _norm(best)
    return best
