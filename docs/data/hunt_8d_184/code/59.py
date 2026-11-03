import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _reconverge(x, steps=200):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = 0.015 * (0.0015 / 0.015) ** f
        lr = 0.030 * (0.006 / 0.030) ** f
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True) + 1e-300
        D = (E @ x) / rs
        dot = (D * x).sum(axis=1, keepdims=True)
        Dt = D - dot * x
        x = x - lr * Dt
        x = _norm(x)
    return x

def _lp_polish(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline - 2.0:
        if radius < 1e-13:
            break
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 1500:
            idx = np.argpartition(-cs, 1500)[:1500]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 1500
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            tl = max(1.0, deadline - time.time() - 0.5)
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb, best

def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    HOP_END = t0 + 108.0
    rng = np.random.default_rng(7)
    n, d = 184, 8
    try:
        x0 = np.asarray(np.load("best.npy"), dtype=np.float64)
    except Exception:
        x0 = np.asarray(np.load("record.npy"), dtype=np.float64)
    if x0.shape != (n, d):
        x0 = x0.reshape(n, d)
    x0 = _norm(x0)
    gbest = x0.copy()
    gval = _max_cos(x0)

    # --- build contact Jacobian ---
    try:
        G0 = gbest @ gbest.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(np.max(G0))
        iu = np.triu_indices(n, 1)
        cf = G0[iu]
        slack = m0 - cf
        sel = slack < 1e-6
        if int(np.sum(sel)) < 50:
            sel = np.argsort(slack)[:800]
            I = iu[0][sel]; J = iu[1][sel]
        else:
            I = iu[0][sel]; J = iu[1][sel]
            if len(I) > 1800:
                idx = np.argsort(slack[sel])[:1800]
                I, J = I[idx], J[idx]
        p = len(I)
        # tangent bases: null space of each x_i (Vh[1:].T)
        Bs = np.empty((n, d, 7), dtype=np.float64)
        for i in range(n):
            _, _, Vh = np.linalg.svd(gbest[i:i+1, :])
            Bs[i] = Vh[1:, :].T  # 8x7
        Jtan = np.zeros((p, n * 7), dtype=np.float64)
        for k in range(p):
            a = int(I[k]); b = int(J[k])
            xa = gbest[a]; xb = gbest[b]
            Jtan[k, a*7:(a+1)*7] = xb @ Bs[a]
            Jtan[k, b*7:(b+1)*7] = xa @ Bs[b]
        # rigidity Gram + eigh
        C = Jtan.T @ Jtan
        # symmetrize
        C = 0.5 * (C + C.T)
        w, V = np.linalg.eigh(C)  # ascending, V is 1288x1288 columns
        # remove global rotations (28 dims)
        Rmat = np.zeros((n*7, 28), dtype=np.float64)
        c = 0
        for a in range(d):
            for b in range(a+1, d):
                for i in range(n):
                    xi = gbest[i]
                    Ax = np.zeros(d)
                    Ax[a] = xi[b]; Ax[b] = -xi[a]
                    Rmat[i*7:(i+1)*7, c] = Bs[i].T @ Ax
                c += 1
        Q, _ = np.linalg.qr(Rmat)  # 1288x28
        # orthogonalize flexes against rotations
        # take bottom 400 flexes
        Kpool = min(400, n*7)
        Flex = V[:, :Kpool].copy()  # 1288 x Kpool
        # project out Q
        proj = Q @ (Q.T @ Flex)
        Flex = Flex - proj
        # renormalize columns
        nr = np.linalg.norm(Flex, axis=0, keepdims=True) + 1e-300
        Flex = Flex / nr
        flex_ok = True
    except Exception:
        flex_ok = False
        Flex = None
        Bs = None

    if flex_ok:
        # also keep Bs of gbest for moves (fixed reference)
        nflex = Flex.shape[1]
        # cycle flex escapes
        for cyc in range(60):
            if time.time() > HOP_END:
                break
            k = int(rng.integers(5, 21))  # 5-20 directions
            k = min(k, nflex)
            # bias to smallest: sample among bottom 250 with random choice
            pool = min(250, nflex)
            idx = rng.choice(pool, size=k, replace=False)
            # random combination; weight smaller ones slightly more
            coeff = rng.standard_normal(k)
            v = Flex[:, idx] @ coeff
            # remove any residual rotation
            v = v - Q @ (Q.T @ v)
            nv = np.linalg.norm(v)
            if nv < 1e-12 or not np.isfinite(nv):
                continue
            v = v / nv
            s = float(rng.uniform(0.4, 1.2))  # total tangent norm
            dtn = (v * s).reshape(n, 7)
            dx = np.einsum('ijk,ik->ij', Bs, dtn)  # (n,8)
            xtry = _norm(gbest + dx)
            pre = _max_cos(xtry)
            # delete-most-violated is implicit: reconverge focuses on worst
            xtry = _reconverge(xtry, steps=200)
            val = _max_cos(xtry)
            if val < gval and np.all(np.isfinite(xtry)):
                gval = val
                gbest = xtry.copy()
                # refresh Bs? keep old for speed; update every improvement
                # quick update of Bs for subsequent moves is desirable but costly;
                # update incrementally: recompute Bs from new gbest every 2 improvements
                if cyc % 2 == 1 and time.time() < HOP_END - 5:
                    try:
                        for i in range(n):
                            _, _, Vh = np.linalg.svd(gbest[i:i+1, :])
                            Bs[i] = Vh[1:, :].T
                    except Exception:
                        pass
            # also accept sideways rarely to change topology even if slightly worse
            elif val < gval + 5e-06 and rng.random() < 0.05:
                # sideways walk from this point (don't update gbest)
                # do one extra flex step from xtry to diversify next base? keep simple: ignore
                pass
            if gval < 0.4995:
                break
    else:
        # fallback: random tangential shakes + reconverge
        for cyc in range(20):
            if time.time() > HOP_END:
                break
            Z = rng.standard_normal((n, d))
            Z -= (Z * gbest).sum(axis=1, keepdims=True) * gbest
            Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-300
            amp = float(rng.uniform(0.03, 0.10))
            xtry = _norm(gbest + amp * Z * np.sqrt(n) * 0.15)
            xtry = _reconverge(xtry, steps=200)
            val = _max_cos(xtry)
            if val < gval:
                gval = val; gbest = xtry.copy()

    # final LP polish with remaining time
    try:
        if time.time() < HARD - 5:
            gbest, gval = _lp_polish(gbest, HARD)
    except Exception:
        pass
    gbest = _norm(gbest)
    return gbest
