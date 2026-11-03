import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G


def _lp_polish(X, budget):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < budget and radius > 1e-14:
        if budget - (time.time() - t0) < 0.5 and radius < 5e-4:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            thr = np.partition(c, -3000)[-3000]
            sel = c >= thr
            # trim to 3200 max
            idx = np.where(sel)[0]
            if len(idx) > 3200:
                o = np.argsort(-c[idx])[:3200]
                m2 = np.zeros_like(c, dtype=bool)
                m2[idx[o]] = True
                sel = m2
        I = iu[0][sel]
        J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
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
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth(Y, steps, T0, T1, lr0, lr1, mu=0.6):
    n, d = Y.shape
    V = np.zeros_like(Y)
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1)
        WY = W @ Y
        r = np.sum(W * G, axis=1)
        inv = np.zeros_like(rs)
        mk = rs > 1e-12
        inv[mk] = 1.0 / rs[mk]
        avgWY = WY * inv[:, None]
        avgR = r * inv
        Dir = avgWY - avgR[:, None] * Y
        Dir[~mk] = 0.0
        V *= mu
        V += Dir
        Y -= lr * V
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Y /= nrm
    return Y


def _find_void(Xkeep, rng, n_cand=2500):
    d = Xkeep.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cn = np.linalg.norm(cand, axis=1, keepdims=True)
    cn[cn == 0] = 1.0
    cand /= cn
    dots = cand @ Xkeep.T
    vals = np.max(dots, axis=1)
    bi = int(np.argmin(vals))
    v = cand[bi].copy()
    for _ in range(30):
        dv = Xkeep @ v
        m = float(np.max(dv))
        w = np.exp(np.clip((dv - m) / 0.02, -50, 0))
        g = w @ Xkeep
        g = g - np.dot(g, v) * v
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        g /= ng
        v = v - 0.02 * g
        nv = np.linalg.norm(v)
        if nv == 0:
            break
        v /= nv
    nvv = np.linalg.norm(v)
    if nvv == 0:
        v = cand[bi]
    else:
        v /= nvv
    return v


def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(7)
    n, d = 181, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                nrm = np.linalg.norm(a, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                a = a / nrm
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    best_val = np.inf
    best = None
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    cur = best.copy()
    cur_val = best_val
    # quick diverse pool: one fresh random smoothed
    # exploration without any linprog
    EXPLORE_END = t0 + 138.0
    it = 0
    # temperature for acceptance
    Tacc0 = 3e-04
    while time.time() < EXPLORE_END:
        rem = BUDGET - (time.time() - t0)
        if rem < 35.0:
            break
        it += 1
        # choose perturbation size
        mode = it % 10
        y = cur.copy()
        if mode == 0:
            sig = 0.02
            y = y + rng.standard_normal(y.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 260
        elif mode == 1:
            sig = 0.07
            y = y + rng.standard_normal(y.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 300
        elif mode in (2, 3, 4, 8):
            # jammed-point surgery: replace worst k with voids
            if mode == 2:
                k = int(rng.integers(2, 5))
            elif mode == 3:
                k = int(rng.integers(5, 11))
            elif mode == 4:
                k = int(rng.integers(10, 18))
            else:
                k = int(rng.integers(3, 8))
            try:
                rm, _ = _row_max(y)
                worst = np.argsort(-rm)[:k]
            except Exception:
                worst = rng.choice(n, size=k, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            Xkeep = y[mask]
            # sequential void fill
            newpts = []
            cur_keep = Xkeep
            for _ in range(k):
                v = _find_void(cur_keep, rng, n_cand=2200)
                newpts.append(v)
                cur_keep = np.vstack([cur_keep, v[None, :]])
            y[worst] = np.array(newpts)
            steps = 320
        elif mode == 5:
            # shake neighbourhood of most jammed point
            try:
                rm, G = _row_max(y)
                wi = int(np.argmax(rm))
                row = G[wi].copy()
                row[wi] = -2
                nn = np.argsort(-row)[:8]
                tgt = np.concatenate([[wi], nn])
                y[tgt] += rng.standard_normal((len(tgt), d)) * 0.06
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            except Exception:
                y += rng.standard_normal(y.shape) * 0.03
                y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 280
        elif mode == 6:
            # fresh random restart (new basin)
            y = rng.standard_normal((n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 420
        elif mode == 7:
            # random subset shake
            k = int(rng.integers(8, 25))
            idx = rng.choice(n, size=k, replace=False)
            y[idx] += rng.standard_normal((k, d)) * 0.12
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 300
        else:
            sig = 0.012
            y = y + rng.standard_normal(y.shape) * sig
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            steps = 240
        # anneal params vary slightly
        T0 = 0.015 + 0.01 * rng.random()
        T1 = 0.0015 + 0.001 * rng.random()
        lr0 = 0.08 + 0.05 * rng.random()
        lr1 = 0.015 + 0.01 * rng.random()
        try:
            _smooth(y, steps=steps, T0=T0, T1=T1, lr0=lr0, lr1=lr1)
            v = _max_cos(y)
        except Exception:
            continue
        frac = (time.time() - t0) / 138.0
        Tacc = Tacc0 * (1.0 - 0.8 * min(max(frac, 0.0), 1.0))
        if v < cur_val or (Tacc > 0 and rng.random() < np.exp((cur_val - v) / max(Tacc, 1e-7))):
            cur = y.copy()
            cur_val = v
            if v < best_val:
                best_val = v
                best = y.copy()
        # also keep global best never lost
        if time.time() >= EXPLORE_END:
            break
    # final LP squeeze on best (and cur if different and good)
    rem = BUDGET - (time.time() - t0)
    # reserve 2s for return
    lp_budget = max(rem - 2.0, 1.0)
    # if cur is close to best but different, polish both briefly? polish best only for time
    try:
        if lp_budget > 2.0:
            # give most time to best
            best = _lp_polish(best, budget=lp_budget * 0.85)
            bv = _max_cos(best)
            best_val = bv
            # quick touch on cur if time left and cur different
            rem2 = BUDGET - (time.time() - t0) - 1.0
            if rem2 > 3.0:
                cv = _max_cos(cur)
                if cv < best_val + 5e-04:
                    curp = _lp_polish(cur, budget=rem2)
                    cpv = _max_cos(curp)
                    if cpv < best_val:
                        best = curp
                        best_val = cpv
        else:
            pass
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
