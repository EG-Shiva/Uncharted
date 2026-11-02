import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    n, d = 121, 8
    rng = np.random.default_rng(42)
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]

    # baselines
    Xfall = None
    Vfall = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < Vfall:
                        Vfall = float(v)
                        Xfall = a.copy()
        except Exception:
            pass
    if Xfall is None:
        Xfall = rng.standard_normal((n, d))
        Xfall /= np.linalg.norm(Xfall, axis=1, keepdims=True)
        Vfall = _max_cos(Xfall)

    def full_fun_factory(beta):
        b = float(beta)
        def fun(flat):
            X = flat.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[iu]
            m = float(c.max())
            e = np.exp(b * (c - m))
            s = float(e.sum())
            f = m + np.log(s) / b
            w = e / s
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(f), GX.ravel()
        return fun

    def lbfgs_full(Xinit, betas, iters):
        xf = Xinit.ravel().copy()
        for beta, mi in zip(betas, iters):
            if time.time() > DEADLINE - 45:
                break
            fun = full_fun_factory(float(beta))
            try:
                res = minimize(fun, xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20,
                                        "gtol": 1e-12, "maxfun": int(mi * 8)})
                xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        X = xf.reshape(n, d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def lp_polish(x, seconds, radius_init=4e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 2.0:
            it += 1
            if it > 90 or radius < 1e-14:
                break
            G = x @ x.T
            cf = G[iu]
            m = float(cf.max())
            thr = max(20.0 * radius, 2e-9)
            sel = cf > m - thr
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 2200:
                kth = np.sort(cf)[-2200]
                sel = cf >= kth
            c = cf[sel]
            I = iu[0][sel]; J = iu[1][sel]
            p = int(sel.sum())
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            valsA = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = y / np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.6, 0.02)
            else:
                radius *= 0.5
        return x, best

    # E8 roots (unit)
    sq2 = np.sqrt(2.0)
    E = []
    for i in range(8):
        for j in range(i + 1, 8):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(8); v[i] = s1; v[j] = s2
                    E.append(v / sq2)
    for mask in range(256):
        if bin(mask).count("1") % 2 == 1:
            continue
        v = np.array([ -0.5 if ((mask >> k) & 1) else 0.5 for k in range(8)]) / sq2
        E.append(v)
    E8 = np.array(E)
    E8 = E8 / np.linalg.norm(E8, axis=1, keepdims=True)

    def greedy_e8(seed):
        rs = np.random.default_rng(seed)
        order = rs.permutation(240)
        Es = E8[order]
        selmask = np.zeros(240, bool)
        curmax = Es @ Es[0]
        sel = [0]
        selmask[0] = True
        curmax[0] = 2.0
        for _ in range(120):
            cand = np.where(~selmask)[0]
            vals = curmax[cand]
            m = vals.min()
            ties = cand[vals <= m + 1e-12]
            pick = int(ties[rs.integers(len(ties))])
            sel.append(pick)
            selmask[pick] = True
            curmax[pick] = 2.0
            nd = Es @ Es[pick]
            curmax = np.maximum(curmax, nd)
            curmax[selmask] = 2.0
        return Es[sel].copy()

    # build initial pool de novo
    inits = []
    for s in range(6):
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        inits.append(X0)
    for sd in (101, 202):
        try:
            inits.append(greedy_e8(sd))
        except Exception:
            pass
    # one large perturbation of record to add diversity but not dominate
    try:
        Xp = Xfall + 0.6 * rng.standard_normal((n, d))
        Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        inits.append(Xp)
    except Exception:
        pass

    pool = []
    for X0 in inits:
        if time.time() > DEADLINE - 55:
            break
        Xopt = lbfgs_full(X0, [2000.0, 8000.0], [70, 70])
        v = _max_cos(Xopt)
        pool.append((v, Xopt))
    if not pool:
        return Xfall
    pool.sort(key=lambda t: t[0])
    # deep anneal top 2
    deep = []
    for k in range(min(2, len(pool))):
        if time.time() > DEADLINE - 45:
            break
        Xd = lbfgs_full(pool[k][1], [12000.0, 35000.0], [140, 140])
        deep.append((_max_cos(Xd), Xd))
    for v, X in pool[:3]:
        deep.append((v, X))
    deep.sort(key=lambda t: t[0])
    Xcur, Vcur = deep[0][1], deep[0][0]

    # LP polish best
    remain = DEADLINE - time.time() - 2.0
    if remain > 5:
        try:
            Xp2, Vp2 = lp_polish(Xcur, min(remain, 50.0))
            if Vp2 < Vcur:
                Xcur, Vcur = Xp2, Vp2
        except Exception:
            pass
    # fallback to lab best if we did not improve
    if Vfall < Vcur:
        return Xfall
    Xcur = Xcur / np.linalg.norm(Xcur, axis=1, keepdims=True)
    Xcur = np.nan_to_num(Xcur, nan=0.0, posinf=0.0, neginf=0.0)
    nn = np.linalg.norm(Xcur, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    Xcur /= nn
    return Xcur
