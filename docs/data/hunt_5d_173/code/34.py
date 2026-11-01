import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

_N = 173
_D = 5

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def _lp_polish(x, seconds):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _repel(x, alphas, iters_list, lrs, mu, decay):
    v = np.zeros_like(x)
    for alpha, iters, lr0 in zip(alphas, iters_list, lrs):
        lr = float(lr0)
        for _ in range(int(iters)):
            G = x @ x.T
            W = np.exp(alpha * (G - 0.75))
            np.fill_diagonal(W, 0.0)
            rs = W.sum(axis=1, keepdims=True) + 1e-300
            A = (W @ x) / rs
            dot = (A * x).sum(axis=1, keepdims=True)
            Ft = A - dot * x
            v = mu * v + (1.0 - mu) * Ft
            x = x - lr * v
            x = _norm(x)
            lr *= decay
    return x

def _build_X(p_flat, K, a, b, q, n_poles):
    p = np.asarray(p_flat, dtype=float).reshape(q, 4)
    u = p[:, 0]; vv = p[:, 1]; phi = p[:, 2]; psi = p[:, 3]
    h = np.cos(u)
    R = np.sin(u)
    # R>=0 for u in [0,pi]
    r12 = R * np.cos(vv)
    r34 = R * np.sin(vv)
    # ensure non-negative (vv in [0,pi/2])
    r12 = np.abs(r12); r34 = np.abs(r34)
    j = np.arange(K)
    ang1 = phi[:, None] + a * 2.0 * np.pi * j[None, :] / K
    ang2 = psi[:, None] + b * 2.0 * np.pi * j[None, :] / K
    x1 = r12[:, None] * np.cos(ang1)
    x2 = r12[:, None] * np.sin(ang1)
    x3 = r34[:, None] * np.cos(ang2)
    x4 = r34[:, None] * np.sin(ang2)
    hh = np.repeat(h[:, None], K, axis=1)
    Xo = np.stack([x1, x2, x3, x4, hh], axis=-1).reshape(q * K, 5)
    if n_poles == 1:
        Xo = np.vstack([Xo, np.array([[0., 0., 0., 0., 1.]])])
    elif n_poles == 2:
        Xo = np.vstack([Xo, np.array([[0., 0., 0., 0., 1.], [0., 0., 0., 0., -1.]])])
    # renormalize (numerical)
    Xo = _norm(Xo)
    return Xo

def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    rng = np.random.RandomState(12345)
    n, d = _N, _D

    try:
        xb = np.load("best.npy").astype(float)
        xb = _norm(xb) if xb.shape == (n, d) else None
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr) if xr.shape == (n, d) else None
    except Exception:
        xr = None
    base = xb if xb is not None else (xr if xr is not None else _norm(rng.standard_normal((n, d))))
    best_X = base.copy()
    best_v = _max_cos(best_X)

    # ---------- symmetric orbit search ----------
    templates = [
        (43, 1, 2, 4, 1),
        (43, 1, 3, 4, 1),
        (43, 2, 3, 4, 1),
        (86, 1, 2, 2, 1),
        (86, 1, 3, 2, 1),
        (57, 1, 2, 3, 2),
        (57, 1, 3, 3, 2),
        (19, 1, 2, 9, 2),
        (19, 1, 3, 9, 2),
    ]
    sym_best_X = None
    sym_best_v = 1e9
    phase1_end = t0 + 72.0
    # bounds for (u,v,phi,psi)
    for (K, a, b, q, npol) in templates:
        if time.time() > phase1_end or time.time() > deadline - 60.0:
            break
        assert q * K + npol == 173, "template mismatch"
        P = q * 4
        lo = np.zeros(P); hi = np.zeros(P)
        for i in range(q):
            lo[i*4+0] = 0.06; hi[i*4+0] = np.pi - 0.06
            lo[i*4+1] = 0.03; hi[i*4+1] = np.pi/2 - 0.03
            lo[i*4+2] = -np.pi; hi[i*4+2] = np.pi
            lo[i*4+3] = -np.pi; hi[i*4+3] = np.pi
        bnds = list(zip(lo, hi))
        n_rest = 2 if q > 4 else 3
        for rr in range(n_rest):
            if time.time() > phase1_end or time.time() > deadline - 60.0:
                break
            p0 = lo + (hi - lo) * rng.rand(P)
            # two-stage smooth-max: alpha 80 then 250
            for alpha, maxit in [(80.0, 120), (250.0, 120)]:
                iu0, iu1 = np.triu_indices(q*K+npol, 1)
                npair = len(iu0)
                def obj(pf, alpha=alpha):
                    try:
                        X = _build_X(pf, K, a, b, q, npol)
                    except Exception:
                        return 2.0
                    G = X @ X.T
                    c = G[iu0, iu1]
                    m = float(c.max())
                    # smooth max via mean (stable)
                    e = np.exp(np.clip(alpha*(c-m), -60, 0))
                    s = float(e.mean())
                    if s <= 0:
                        return m
                    return m + np.log(s)/alpha
                try:
                    res = minimize(obj, p0, method='L-BFGS-B', bounds=bnds,
                                   options={'maxiter': maxit, 'ftol': 1e-12, 'gtol': 1e-6})
                    p0 = res.x
                except Exception:
                    break
            try:
                Xf = _build_X(p0, K, a, b, q, npol)
                vf = _max_cos(Xf)
            except Exception:
                continue
            if vf < sym_best_v:
                sym_best_v = vf
                sym_best_X = Xf.copy()
            if vf < best_v:
                # quick full polish of symmetric point (short) to update global best early
                try:
                    Y = _lp_polish(Xf.copy(), 1.0)
                    vy = _max_cos(Y)
                    if vy < best_v:
                        best_v = vy; best_X = Y.copy()
                except Exception:
                    pass

    # ---------- break symmetry ----------
    # collect seeds: symmetric best (various noise levels) + record best
    seeds = []
    if sym_best_X is not None:
        for sig in [0.0, 0.002, 0.008, 0.025]:
            if time.time() > deadline - 30.0:
                break
            X0 = sym_best_X.copy()
            if sig > 0:
                X0 = _norm(X0 + sig * rng.standard_normal(X0.shape))
            # repel with random strength
            try:
                a1 = 30.0 + rng.rand()*20.0
                a2 = 90.0 + rng.rand()*60.0
                a3 = 200.0 + rng.rand()*120.0
                X1 = _repel(X0.copy(), [a1,a2,a3], [120,160,200],
                            [0.08+rng.rand()*0.06, 0.03+rng.rand()*0.03, 0.015+rng.rand()*0.015],
                            0.6+rng.rand()*0.25, 0.998)
                X1 = _lp_polish(X1, 1.5)
                v1 = _max_cos(X1)
                if v1 < best_v:
                    best_v = v1; best_X = X1.copy()
                # keep as seed for further mutation
                seeds.append(X1)
            except Exception:
                continue
        # nearby decomposition: drop one orbit (K points) -> void refill + polish
        # this moves one orbit to plane/fixed points topology
        try:
            if time.time() < deadline - 30.0:
                Kguess = sym_best_X.shape[0]  # 173
                # pick worst orbit block implicitly by worst points
                G = sym_best_X @ sym_best_X.T
                np.fill_diagonal(G, -2.0)
                mi = G.max(axis=1)
                order = np.argsort(-mi)
                for kk in [12, 24, 43]:
                    if time.time() > deadline - 30.0:
                        break
                    sel = order[:kk]
                    mask = np.ones(n, dtype=bool); mask[sel] = False
                    rem = best_X[mask] if best_X.shape[0]==n else sym_best_X[mask]
                    # void fill from random candidates
                    cand = _norm(rng.standard_normal((4000, d)))
                    smax = (cand @ rem.T).max(axis=1)
                    chosen = []
                    for _ in range(kk):
                        idx = int(np.argmin(smax))
                        pt = cand[idx].copy(); chosen.append(pt)
                        upd = cand @ pt
                        np.maximum(smax, upd, out=smax); smax[idx]=2.0
                    X2 = sym_best_X.copy() if best_X is None else best_X.copy()
                    # apply to worst indices of X2
                    G2 = X2 @ X2.T; np.fill_diagonal(G2,-2.0)
                    mi2 = G2.max(axis=1); ord2=np.argsort(-mi2)
                    sel2 = ord2[:kk]
                    X2[sel2] = np.stack(chosen,0)
                    X2 = _norm(X2 + 0.005*rng.standard_normal(X2.shape))
                    X2 = _repel(X2, [40.0,120.0,260.0],[100,150,200],[0.1,0.04,0.02],0.7,0.998)
                    X2 = _lp_polish(X2, 1.5)
                    v2 = _max_cos(X2)
                    if v2 < best_v:
                        best_v=v2; best_X=X2.copy()
                    seeds.append(X2)
        except Exception:
            pass

    # short basin-hopping from global best + seeds
    t_basin_end = deadline - 13.0
    pool = [best_X.copy()] + [s.copy() for s in seeds[:4]]
    # add kicked record copies
    for _ in range(2):
        try:
            Xk = _norm(base + (0.15+0.2*rng.rand())*rng.standard_normal(base.shape))
            pool.append(Xk)
        except Exception:
            pass
    it = 0
    while time.time() < t_basin_end:
        it += 1
        # pick parent: best 50% of time
        if rng.rand() < 0.5:
            parent = best_X
        else:
            parent = pool[int(rng.randint(0, len(pool)))]
        Xtry = parent.copy()
        r = rng.rand()
        if r < 0.5:
            # gaussian kick
            sig = 0.03 + 0.15*rng.rand()
            if rng.rand() < 0.1: sig = 0.35
            Xtry = _norm(Xtry + sig*rng.standard_normal(Xtry.shape))
        elif r < 0.8:
            # cap replace kk worst
            G = Xtry @ Xtry.T; np.fill_diagonal(G,-2.0)
            mi = G.max(axis=1); order=np.argsort(-mi)
            kk = int(rng.randint(4, 16))
            sel = order[:kk]; mask=np.ones(n,bool); mask[sel]=False
            rem=Xtry[mask]
            cand=_norm(rng.standard_normal((3000,d)))
            smax=(cand@rem.T).max(axis=1)
            ch=[]
            for _ in range(kk):
                idx=int(np.argmin(smax)); pt=cand[idx].copy(); ch.append(pt)
                np.maximum(smax, cand@pt, out=smax); smax[idx]=2.0
            Xtry[sel]=np.stack(ch,0)
            Xtry=_norm(Xtry+0.004*rng.standard_normal(Xtry.shape))
        else:
            # recombine best with random pool member via reflection/rotation + splice
            if len(pool) >= 2:
                B = pool[int(rng.randint(0,len(pool)))]
                try:
                    M = B.T @ Xtry
                    U,_,Vt = np.linalg.svd(M); Rq = U@Vt
                    Ba = B@Rq
                except Exception:
                    Ba = B
                u = rng.standard_normal(d); u/=np.linalg.norm(u)
                pa = Xtry@u; thr=np.quantile(pa, rng.rand()*0.5+0.25)
                take = pa>thr
                C = Xtry.copy(); C[~take]=Ba[~take]
                Xtry=_norm(C+0.005*rng.standard_normal(C.shape))
            else:
                Xtry=_norm(Xtry+0.05*rng.standard_normal(Xtry.shape))
        try:
            Xtry = _repel(Xtry, [40.0,130.0,260.0],[80,120,150],[0.09,0.04,0.02],0.65,0.998)
            Xtry = _lp_polish(Xtry, 0.8)
            vtry = _max_cos(Xtry)
        except Exception:
            continue
        if vtry < best_v:
            best_v=vtry; best_X=Xtry.copy()
            pool.append(Xtry.copy())
            if len(pool)>8:
                # drop worst (approx by evaluating)
                try:
                    vals=[_max_cos(p) for p in pool]
                    pool=[pool[i] for i in np.argsort(vals)[:8]]
                except Exception:
                    pool=pool[-8:]
        elif rng.rand() < 0.05 and len(pool)<8:
            pool.append(Xtry.copy())

    # final long polish of best
    try:
        remain = deadline - time.time() - 1.0
        if remain > 2.0:
            Y = _lp_polish(best_X.copy(), min(remain, 12.0))
            vy = _max_cos(Y)
            if vy < best_v:
                best_v=vy; best_X=Y.copy()
    except Exception:
        pass
    best_X = _norm(best_X)
    # safety: if somehow best got worse than record, reload record
    try:
        if xb is not None:
            vx = _max_cos(xb)
            if vx < _max_cos(best_X):
                best_X = xb.copy()
    except Exception:
        pass
    return np.asarray(best_X, dtype=float)
