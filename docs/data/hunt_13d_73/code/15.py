import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _norm(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _find_hole(X_keep, rng, n_cand=5000):
    d = X_keep.shape[1]
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ X_keep.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))].copy()

def _smooth(X0, rng_unused, steps, s0, s1, lr0, lr1):
    n, d = X0.shape
    X = X0.copy()
    iu0 = np.triu_indices(n, 1)
    I0, J0 = iu0[0], iu0[1]
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    ls0, ls1 = np.log(s0), np.log(s1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    W = np.zeros((n, n))
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 1.0
        s = np.exp(ls0 + (ls1 - ls0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        c = G[I0, J0]
        cmax = c.max()
        e = np.exp(s * (c - cmax))
        ssum = e.sum()
        if not np.isfinite(ssum) or ssum == 0:
            w = (c >= cmax - 1e-12).astype(float)
            s_ = w.sum()
            if s_ == 0:
                continue
            w /= s_
        else:
            w = e / ssum
        W.fill(0.0)
        W[I0, J0] = w
        W[J0, I0] = w
        grad = W @ X
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        m1 *= b1
        m1 += (1 - b1) * gt
        v1 *= b2
        v1 += (1 - b2) * (gt * gt)
        mhat = m1 / (1 - b1p)
        vhat = v1 / (1 - b2p)
        b1p *= b1
        b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
        X -= lr * step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _lp_polish(x, seconds, t_end=None):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if t_end is not None and now >= t_end:
            break
        if radius < 1e-14:
            break
        cfull = (x @ x.T)[iu]
        m = cfull.max()
        sel = cfull > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cfull[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPLORE_END = t0 + 128.0
    rng = np.random.default_rng(7)
    try:
        Xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            Xb = np.load("record.npy").astype(float)
        except Exception:
            Xb = rng.normal(size=(73, 13))
    Xb = np.asarray(Xb, dtype=float)
    if Xb.shape != (73, 13):
        tmp = rng.normal(size=(73, 13))
        try:
            m0 = min(Xb.shape[0], 73)
            m1 = min(Xb.shape[1], 13)
            tmp[:m0, :m1] = Xb[:m0, :m1]
        except Exception:
            pass
        Xb = tmp
    Xb = _norm(Xb)
    best_val = _max_cos(Xb)
    X_best = Xb.copy()
    X_cur = Xb.copy()
    cur_val = best_val
    n, d = 73, 13
    it = 0
    sigmas = [0.008, 0.02, 0.045, 0.09, 0.17, 0.30]
    while time.time() < EXPLORE_END and time.time() < DEADLINE - 35.0:
        it += 1
        if time.time() > DEADLINE - 35.0:
            break
        base = X_best if rng.random() < 0.70 else X_cur
        r = rng.random()
        X_try = None
        try:
            if r < 0.30:
                sigma = sigmas[int(rng.integers(0, len(sigmas)))]
                if rng.random() < 0.35:
                    sigma *= 0.5
                noise = rng.normal(size=(n, d))
                # tangential component only to stay on sphere manifold
                dn = (noise * base).sum(axis=1, keepdims=True)
                noise -= dn * base
                X_try = base + sigma * noise
                X_try = _norm(X_try)
            elif r < 0.52:
                # coherent cap shake
                u = rng.normal(size=d)
                u /= np.linalg.norm(u)
                dots = base @ u
                thr = float(rng.choice(np.array([0.25, 0.45, 0.65])))
                mask = dots > thr
                if int(mask.sum()) < 3:
                    idx = np.argsort(-dots)[:int(rng.integers(4, 10))]
                    mask = np.zeros(n, dtype=bool)
                    mask[idx] = True
                sigma = float(rng.choice(np.array([0.05, 0.12, 0.22])))
                noise = rng.normal(size=(int(mask.sum()), d))
                dn = (noise * base[mask]).sum(axis=1, keepdims=True)
                noise -= dn * base[mask]
                X_try = base.copy()
                X_try[mask] += sigma * noise
                # small global jitter
                X_try += 0.004 * rng.normal(size=(n, d))
                X_try = _norm(X_try)
            elif r < 0.80:
                # kill overloaded and reinsert into holes
                rm = _row_max(base)
                order = np.argsort(-rm)
                if rng.random() < 0.5:
                    k = int(rng.choice(np.array([1, 1, 2, 2, 3])))
                else:
                    k = int(rng.choice(np.array([1, 2])))
                victims = order[:k].copy()
                if rng.random() < 0.35:
                    victims[0] = int(rng.integers(0, n))
                    victims = np.unique(victims)
                X_try = base.copy()
                for v in victims:
                    others = np.delete(X_try, int(v), axis=0)
                    hole = _find_hole(others, rng, n_cand=6000)
                    # slight jitter to avoid exact symmetry
                    hole = hole + 1e-3 * rng.normal(size=d)
                    hole /= np.linalg.norm(hole)
                    X_try[int(v)] = hole
            else:
                # rattler / random-subset resample
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                rm = G.max(axis=1)
                # rattlers: far below max
                ratt = np.where(rm < best_val - 5e-5)[0]
                X_try = base.copy()
                targets = []
                if len(ratt) > 0 and rng.random() < 0.7:
                    targets = list(rng.choice(ratt, size=min(len(ratt), int(rng.integers(1, 3))), replace=False))
                # add random subset
                nrand = int(rng.integers(1, 4))
                cand = [int(rng.integers(0, n)) for _ in range(nrand)]
                targets = list(targets) + cand
                targets = list(dict.fromkeys(targets))[:4]
                for v in targets:
                    others = np.delete(X_try, int(v), axis=0)
                    hole = _find_hole(others, rng, n_cand=8000)
                    X_try[int(v)] = hole
                # jolt neighbours of targets slightly
                if rng.random() < 0.5:
                    X_try += 0.008 * rng.normal(size=(n, d))
                    X_try = _norm(X_try)
        except Exception:
            continue
        if X_try is None:
            continue
        # fast smooth filter
        try:
            steps = 350 if (it % 3) else 550
            if r >= 0.52 and r < 0.80:
                steps = 500
            s0, s1 = 35.0, 280.0
            lr0, lr1 = 0.025, 0.0025
            # larger kicks get hotter start
            X_sm = _smooth(X_try, rng, steps, s0, s1, lr0, lr1)
        except Exception:
            continue
        try:
            v = _max_cos(X_sm)
        except Exception:
            continue
        if v < cur_val:
            cur_val = v
            X_cur = X_sm.copy()
        elif v < cur_val + 0.004 and rng.random() < 0.12:
            cur_val = v
            X_cur = X_sm.copy()
        improved = v < best_val - 1e-12
        # probe promising basins with short LP
        need_probe = improved or (v < best_val + 0.0015 and (it % 12 == 0))
        if need_probe:
            remain = min(2.5, max(0.6, EXPLORE_END - (time.time() - t0) - 1.0 if False else 2.5))
            # cap probe time, keep global deadline
            tlim = min(time.time() + remain, EXPLORE_END, DEADLINE - 32.0)
            secs = max(0.5, tlim - time.time())
            if secs < 0.4:
                if improved:
                    best_val = v
                    X_best = X_sm.copy()
                continue
            try:
                X_lp = _lp_polish(X_sm, secs, t_end=tlim)
                vlp = _max_cos(X_lp)
            except Exception:
                if improved:
                    best_val = v
                    X_best = X_sm.copy()
                continue
            if vlp < best_val:
                best_val = vlp
                X_best = X_lp.copy()
                cur_val = min(cur_val, vlp)
                X_cur = X_lp.copy()
            elif vlp < cur_val and rng.random() < 0.5:
                X_cur = X_lp.copy()
                cur_val = vlp
            elif improved:
                best_val = v
                X_best = X_sm.copy()
        else:
            if improved:
                best_val = v
                X_best = X_sm.copy()
    # final long polish of best
    try:
        tleft = DEADLINE - time.time() - 1.0
        if tleft > 3.0:
            # a couple of quick smooth anneals before LP to settle
            try:
                X_best = _smooth(X_best, rng, 250, 60.0, 400.0, 0.008, 0.001)
            except Exception:
                pass
            tleft = DEADLINE - time.time() - 0.5
            if tleft > 2.0:
                X_best = _lp_polish(X_best, tleft, t_end=DEADLINE - 0.5)
    except Exception:
        pass
    X_best = _norm(X_best)
    return X_best
