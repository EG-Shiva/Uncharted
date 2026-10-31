import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        cfull = x @ x.T
        c = cfull[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            radius *= 0.5
            continue
        I = iu[0][sel]
        J = iu[1][sel]
        cc = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr() if False else coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _adam(x, steps, k0, k1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lrk = (k1 / k0) ** (1.0 / max(steps - 1, 1)) if steps > 1 else 1.0
    lrr = (lr1 / lr0) ** (1.0 / max(steps - 1, 1)) if steps > 1 else 1.0
    k = float(k0)
    lr = float(lr0)
    with np.errstate(under='ignore', over='ignore'):
        for s in range(steps):
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(G.max())
            E = np.exp(k * (G - cmax))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum() * 0.5)
            if S <= 0 or not np.isfinite(S):
                break
            grad = (E @ x) / S
            dot = (grad * x).sum(axis=1, keepdims=True)
            gt = grad - dot * x
            m = b1 * m + (1 - b1) * gt
            v = b2 * v + (1 - b2) * (gt * gt)
            mhat = m / (1 - b1 ** (s + 1))
            vhat = v / (1 - b2 ** (s + 1))
            step = mhat / (np.sqrt(vhat) + eps)
            x = x - lr * step
            x = x / np.linalg.norm(x, axis=1, keepdims=True)
            k *= lrk
            lr *= lrr
    return x


def _rsphere(rng, n, d):
    a = rng.standard_normal((n, d))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a


def _rand_rot(rng, d):
    a = rng.standard_normal((d, d))
    q, r = np.linalg.qr(a)
    s = np.sign(np.diag(r))
    s[s == 0] = 1.0
    q = q * s[None, :]
    return q


def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    rng = np.random.default_rng(7)
    n, d = 153, 4
    try:
        xb = np.load("best.npy").astype(float)
        if xb.shape != (n, d):
            raise ValueError("shape")
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
            if xb.shape != (n, d):
                raise ValueError("shape")
        except Exception:
            xb = _rsphere(rng, n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        if xr.shape == (n, d):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            if _max_cos(xr) < _max_cos(xb):
                xb = xr
    except Exception:
        pass
    vb = _max_cos(xb)
    pool = [(vb, xb.copy())]
    # quick polish of seed if very fast (helps baseline)
    # main basin-hopping loop
    it = 0
    RESERVE = 22.0
    while time.time() < DEADLINE - RESERVE:
        it += 1
        if time.time() >= DEADLINE - RESERVE:
            break
        pool = sorted(pool, key=lambda t: t[0])
        best_v = pool[0][0]
        # choose parents
        base = pool[0][1]
        second = pool[1][1] if len(pool) > 1 else pool[0][1]
        if len(pool) > 1 and (it % 5 == 0):
            base = second
        mode = it % 8
        xt = base.copy()
        if mode == 0:
            sig = 0.03
            xt = xt + sig * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1 = 600, 15.0, 250.0, 0.02, 0.002
        elif mode == 1:
            sig = 0.12
            xt = xt + sig * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1 = 800, 12.0, 300.0, 0.025, 0.002
        elif mode == 2:
            # random kill + farthest-hole refill
            kkill = int(rng.integers(8, 14))
            idx = rng.choice(n, size=kkill, replace=False)
            keep = np.ones(n, dtype=bool)
            keep[idx] = False
            # refill one by one farthest from kept+filled
            cur = xt[keep]
            new_pts = []
            cand = _rsphere(rng, 4000, d)
            for _ in range(kkill):
                sims = cand @ cur.T
                bc = sims.max(axis=1)
                bi = int(np.argmin(bc))
                pt = cand[bi]
                new_pts.append(pt)
                cur = np.vstack([cur, pt[None, :]])
            xt[idx] = np.array(new_pts)
            xt = xt + 0.01 * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1 = 900, 12.0, 320.0, 0.025, 0.002
        elif mode == 3:
            # cap kill 18-30 + farthest refill
            u = _rsphere(rng, 1, d)[0]
            proj = xt @ u
            kkill = int(rng.integers(18, 31))
            idx = np.argsort(-proj)[:kkill]
            keep = np.ones(n, dtype=bool)
            keep[idx] = False
            cur = xt[keep]
            new_pts = []
            cand = _rsphere(rng, 6000, d)
            for _ in range(kkill):
                sims = cand @ cur.T
                bc = sims.max(axis=1)
                bi = int(np.argmin(bc))
                pt = cand[bi]
                new_pts.append(pt)
                cur = np.vstack([cur, pt[None, :]])
            xt[idx] = np.array(new_pts)
            xt = xt + 0.008 * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1 = 1100, 10.0, 320.0, 0.03, 0.002
        elif mode == 4:
            # hemisphere crossover without matching (robust)
            try:
                A = base
                # random rotation of second parent
                Q = _rand_rot(rng, d)
                B = second @ Q
                vv = _rsphere(rng, 1, d)[0]
                da = A @ vv
                db = B @ vv
                Ia = np.where(da < 0)[0]
                Ib = np.where(db >= 0)[0]
                parts = []
                parts.append(A[Ia])
                parts.append(B[Ib])
                C = np.vstack(parts) if len(parts[0]) + len(parts[1]) > 0 else A.copy()
                # fix count to n
                if C.shape[0] >= n:
                    # keep farthest subset? just random subset + keep balance
                    if C.shape[0] > n:
                        sel = rng.choice(C.shape[0], size=n, replace=False)
                        C = C[sel]
                else:
                    need = n - C.shape[0]
                    extra = _rsphere(rng, need, d)
                    C = np.vstack([C, extra])
                xt = C.copy()
                xt = xt + 0.02 * rng.standard_normal((n, d))
                xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            except Exception:
                xt = xt + 0.15 * rng.standard_normal((n, d))
                xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1 = 1000, 10.0, 300.0, 0.03, 0.002
        elif mode == 5:
            # rattler + overloaded shake
            try:
                G = xt @ xt.T
                np.fill_diagonal(G, -2.0)
                cm = G.max(axis=1)
                ratt = np.argsort(cm)[:3]
                # replace worst rattler by global hole
                cand = _rsphere(rng, 15000, d)
                keepm = np.ones(n, dtype=bool)
                keepm[ratt[0]] = False
                sims = cand @ xt[keepm].T
                bc = sims.max(axis=1)
                bi = int(np.argmin(bc))
                xt[ratt[0]] = cand[bi]
                # shake top-degree nodes
                thr = best_v - 0.008
                mask = G > thr
                deg = mask.sum(axis=1)
                order = np.argsort(-deg)
                top = order[:10]
                xt[top] += 0.12 * rng.standard_normal((10, d))
                xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            except Exception:
                xt = xt + 0.05 * rng.standard_normal((n, d))
                xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1 = 900, 14.0, 300.0, 0.025, 0.002
        elif mode == 6:
            # fresh random restart, long anneal (new basin)
            xt = _rsphere(rng, n, d)
            steps, k0, k1, lr0, lr1 = 1400, 8.0, 350.0, 0.03, 0.002
        else:
            # large global shake
            xt = xt + 0.30 * rng.standard_normal((n, d))
            xt /= np.linalg.norm(xt, axis=1, keepdims=True)
            steps, k0, k1, lr0, lr1 = 1000, 10.0, 280.0, 0.03, 0.002
        try:
            xt = _adam(xt, steps, k0, k1, lr0, lr1)
            vt = _max_cos(xt)
        except Exception:
            continue
        # pool update
        pool = sorted(pool, key=lambda t: t[0])
        cur_best = pool[0][0]
        # keep diverse pool of up to 5 within 0.02
        try:
            dup = False
            for _, p in pool:
                if np.linalg.norm(xt - p) < 1e-3 and np.linalg.norm(xt + p) > 1e-3:
                    # crude duplicate check (sign not relevant, use min)
                    pass
            if len(pool) < 5:
                pool.append((vt, xt.copy()))
                pool = sorted(pool, key=lambda t: t[0])[:5]
            else:
                worst = max(p[0] for p in pool)
                if vt < worst - 2e-6:
                    idxw = int(np.argmax([p[0] for p in pool]))
                    pool[idxw] = (vt, xt.copy())
        except Exception:
            pass
        # LP screening if promising
        try:
            pool = sorted(pool, key=lambda t: t[0])
            cur_best = pool[0][0]
            if vt < cur_best + 0.004:
                sec = min(2.0, (DEADLINE - RESERVE - time.time()) - 0.15)
                if sec > 0.4:
                    xp = _lp_polish(xt, sec)
                    vp = _max_cos(xp)
                    if vp < cur_best - 1e-14:
                        pool.append((vp, xp.copy()))
                        pool = sorted(pool, key=lambda t: t[0])[:5]
                    elif vp < cur_best + 0.02 and len(pool) < 5:
                        pool.append((vp, xp.copy()))
                        pool = sorted(pool, key=lambda t: t[0])[:5]
        except Exception:
            pass
    pool = sorted(pool, key=lambda t: t[0])
    best = pool[0][1].copy()
    rem = DEADLINE - time.time() - 1.5
    if rem > 2.0:
        try:
            best = _lp_polish(best, rem)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # safety: replace non-finite
    if not np.all(np.isfinite(best)):
        best = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    return best
