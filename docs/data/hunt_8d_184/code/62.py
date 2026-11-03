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

def _smooth(x, steps, a0, a1, lr0, lr1):
    la0, la1 = np.log(a0), np.log(a1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        alpha = np.exp(la0 + (la1 - la0) * f)
        lr = np.exp(ll0 + (ll1 - ll0) * f)
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-30
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        x = x - lr * gt / S
        x = _norm(x)
    return x

def _lp_polish(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    xb = x.copy()
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        if time.time() > deadline - 2.0:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        cf = G[iu]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cf > thr
        I = iu[0][sel]; J = iu[1][sel]; cs = cf[sel]
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
        try:
            y = _norm(x + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    HOP_END = t0 + 128.0
    rng = np.random.default_rng(7)
    n, d, dt = 184, 8, 7
    try:
        xb0 = np.asarray(np.load("best.npy"), dtype=np.float64).reshape(n, d)
    except Exception:
        xb0 = np.asarray(np.load("record.npy"), dtype=np.float64).reshape(n, d)
    xb0 = _norm(xb0)
    gbest = xb0.copy()
    gval = _max_cos(gbest)
    cur = gbest.copy()
    cur_val = gval

    # --- build contact Jacobian for pairs within 1e-6 of max ---
    try:
        G0 = xb0 @ xb0.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(np.max(G0))
        iu0 = np.triu_indices(n, 1)
        cf0 = G0[iu0]
        sel0 = cf0 > m0 - 1e-6
        I0 = iu0[0][sel0]; J0 = iu0[1][sel0]; cs0 = cf0[sel0]
        # cap to tightest 1600 to bound SVD cost
        if len(I0) > 1600:
            idx = np.argpartition(-cs0, 1600)[:1600]
            I0, J0 = I0[idx], J0[idx]
        p = len(I0)
        # tangent bases T_i (8x7)
        T = np.zeros((n, d, dt))
        for i in range(n):
            xi = xb0[i]
            k = int(np.argmax(np.abs(xi)))
            cols = [j for j in range(d) if j != k]
            V = np.zeros((d, dt))
            for c, j in enumerate(cols):
                e = np.zeros(d); e[j] = 1.0
                V[:, c] = e - xi[j] * xi
            Q, _ = np.linalg.qr(V, mode='reduced')
            # ensure orthogonality to xi (fix sign degeneracy)
            T[i] = Q
        Jp = np.zeros((p, n * dt))
        for r in range(p):
            i, j = int(I0[r]), int(J0[r])
            Jp[r, i * dt:(i + 1) * dt] = T[i].T @ xb0[j]
            Jp[r, j * dt:(j + 1) * dt] = T[j].T @ xb0[i]
        # SVD via Gram eig (1288x1288)
        ATA = Jp.T @ Jp
        # symmetrize
        ATA = 0.5 * (ATA + ATA.T)
        w, Vv = np.linalg.eigh(ATA)  # ascending, columns
        # global rotations (28) in tangent coords
        Rmat = np.zeros((n * dt, 28))
        q = 0
        for a in range(d):
            for b in range(a + 1, d):
                for i in range(n):
                    dx = np.zeros(d)
                    dx[a] = -xb0[i, b]
                    dx[b] = xb0[i, a]
                    Rmat[i * dt:(i + 1) * dt, q] = T[i].T @ dx
                q += 1
        Qr, _ = np.linalg.qr(Rmat, mode='reduced')  # 1288x28
        # project eigenvectors orthogonal to rotations, collect pool
        pool = []
        # consider bottom 250 eigenvectors
        lim = min(250, Vv.shape[1])
        for k in range(lim):
            v = Vv[:, k]
            c = Qr.T @ v
            vo = v - Qr @ c
            nr = float(np.linalg.norm(vo))
            if nr < 0.3:
                continue  # essentially a rotation
            vo = vo / (nr + 1e-300)
            pool.append(vo)
            if len(pool) >= 100:
                # keep scanning? need lowest ones; but take first 100 non-rot
                pass
        if len(pool) > 100:
            pool = pool[:100]
        pool = np.array(pool)  # (K,1288)
        flex_ok = len(pool) >= 5
    except Exception:
        flex_ok = False
        pool = None
        T = None

    # --- flex escape cycles ---
    if flex_ok:
        ncyc = 9
        for cyc in range(ncyc):
            if time.time() > HOP_END:
                break
            if time.time() > DEADLINE - 50:
                break
            K = pool.shape[0]
            k = int(rng.integers(5, 21))
            k = min(k, K)
            sel = rng.choice(K, size=k, replace=False)
            coeff = rng.standard_normal(size=k)
            combo = coeff @ pool[sel, :]  # (1288,)
            nrm = float(np.linalg.norm(combo))
            if nrm < 1e-12 or not np.isfinite(nrm):
                continue
            combo = combo / nrm
            rms = float(rng.uniform(0.03, 0.10))
            S = rms * np.sqrt(n)
            y = combo * S  # tangent coords
            # map to ambient using T
            dx = np.zeros((n, d))
            for i in range(n):
                dx[i] = T[i] @ y[i * dt:(i + 1) * dt]
            base = cur.copy()
            xt = base + dx
            xt = _norm(xt)
            # reconverge: 200 steps power-law-like softmax repulsion
            xt = _smooth(xt, steps=200, a0=40.0, a1=320.0, lr0=0.030, lr1=0.003)
            try:
                v = _max_cos(xt)
            except Exception:
                continue
            if not np.isfinite(v):
                continue
            # keep best
            if v < gval:
                gval = float(v); gbest = xt.copy()
            # sideways walk: advance cur if improving or close
            if v < cur_val:
                cur, cur_val = xt.copy(), float(v)
            elif v < 0.5005 and v < cur_val + 8e-06 and rng.random() < 0.35:
                cur, cur_val = xt.copy(), float(v)
            # if stuck far above, reset cur to gbest occasionally
            if cur_val > 0.5005 and (cyc % 3 == 2):
                cur, cur_val = gbest.copy(), gval
        # also try a couple of pure-random tangential fallbacks from gbest if time remains
        for _ in range(2):
            if time.time() > HOP_END:
                break
            xt = gbest + rng.standard_normal(size=(n, d)) * 0.02
            xt = _norm(xt)
            xt = _smooth(xt, steps=150, a0=45.0, a1=300.0, lr0=0.028, lr1=0.003)
            try:
                v = _max_cos(xt)
            except Exception:
                continue
            if v < gval:
                gval, gbest = float(v), xt.copy()
    else:
        # fallback: small random hops
        for _ in range(6):
            if time.time() > HOP_END:
                break
            xt = gbest + rng.standard_normal(size=(n, d)) * 0.03
            xt = _norm(xt)
            xt = _smooth(xt, steps=200, a0=40.0, a1=320.0, lr0=0.030, lr1=0.003)
            try:
                v = _max_cos(xt)
            except Exception:
                continue
            if v < gval:
                gval, gbest = float(v), xt.copy()

    # --- final LP polish if promising ---
    if time.time() < DEADLINE - 5:
        # only polish best; require pre-LP max below 0.5005 per brief
        try:
            pv = _max_cos(gbest)
        except Exception:
            pv = 1.0
        if pv < 0.5005:
            gbest = _lp_polish(gbest, DEADLINE - 1.0)
    gbest = _norm(gbest)
    if not np.all(np.isfinite(gbest)):
        gbest = _norm(xb0.copy())
    return gbest
