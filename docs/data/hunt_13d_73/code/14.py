import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rowwise_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _random_vec(rng, d):
    v = rng.normal(size=d)
    n = np.linalg.norm(v)
    if n == 0:
        v[0] = 1.0
        n = 1.0
    return v / n

def _find_hole(X_keep, rng, n_cand=4000):
    d = X_keep.shape[1]
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ X_keep.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))].copy()

def _smooth_adam(X0, Iv, Jv, free_mask, steps, s0, s1, lr0, lr1):
    # X0 assumed normalized; Iv,Jv are valid pair indices (at least one free if subset)
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    ls0, ls1 = np.log(s0), np.log(s1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    grad = np.empty_like(X)
    XJ = np.empty((len(Iv), d))
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 1.0
        s = np.exp(ls0 + (ls1 - ls0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        c = G[Iv, Jv]
        cmax = c.max()
        e = np.exp(s * (c - cmax))
        S = e.sum()
        if not np.isfinite(S) or S == 0:
            w = np.zeros_like(e)
            w[c >= cmax - 1e-12] = 1.0
            w /= w.sum()
        else:
            w = e / S
        grad.fill(0.0)
        # grad_I += w*X_J , grad_J += w*X_I
        W = w[:, None]
        np.add.at(grad, Iv, W * X[Jv])
        np.add.at(grad, Jv, W * X[Iv])
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        if free_mask is not None:
            gt[~free_mask] = 0.0
        m1 *= b1
        m1 += (1.0 - b1) * gt
        v1 *= b2
        v1 += (1.0 - b2) * (gt * gt)
        mhat = m1 / (1.0 - b1p)
        vhat = v1 / (1.0 - b2p)
        b1p *= b1
        b2p *= b2
        step = mhat / (np.sqrt(vhat) + eps)
        # clip extreme coordinate to stabilise
        np.clip(step, -3.0, 3.0, out=step)
        X -= lr * step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _lp_polish(x, seconds, t_end=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_end is not None and time.time() > t_end:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    rng = np.random.default_rng(20261102)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.normal(size=(n, d))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (n, d):
        tmp = rng.normal(size=(n, d))
        m0 = min(xb.shape[0], n)
        m1 = min(xb.shape[1], d)
        tmp[:m0, :m1] = xb[:m0, :m1]
        xb = tmp
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    Xbest = xb.copy()
    Xcur = xb.copy()
    cur_val = best_val
    I0_all, J0_all = np.triu_indices(n, 1)
    # reserve final polish
    FINAL_RESERVE = 22.0
    it = 0
    # pre-allocate free mask buffer
    while time.time() < DEADLINE - FINAL_RESERVE:
        it += 1
        remain = DEADLINE - FINAL_RESERVE - time.time()
        if remain < 1.0:
            break
        use_best = rng.random() < 0.70
        base = Xbest if use_best else Xcur
        rmode = rng.random()
        try:
            if rmode < 0.38:
                # global Gaussian kick + global smooth
                sigmas = (0.02, 0.05, 0.09, 0.15, 0.25, 0.35)
                sigma = sigmas[int(rng.integers(0, len(sigmas)))]
                Xt = base + sigma * rng.normal(size=(n, d))
                Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
                steps = int(rng.choice([400, 500, 650]))
                s1 = float(rng.choice([150.0, 220.0, 320.0]))
                lr0 = 0.012
                lr1 = 0.002
                Xs = _smooth_adam(Xt, I0_all, J0_all, None, steps, 30.0, s1, lr0, lr1)
                free_desc = None
            elif rmode < 0.68:
                # kill worst k and reinsert into holes
                k = int(rng.choice([1, 1, 2, 2, 3]))
                rm = _rowwise_max(base)
                order = np.argsort(-rm)
                victims = order[:k].copy()
                if rng.random() < 0.35:
                    victims[0] = int(rng.integers(0, n))
                    victims = np.unique(victims)
                    # top up if duplicates
                    while len(victims) < k:
                        cand = int(rng.integers(0, n))
                        if cand not in victims:
                            victims = np.append(victims, cand)
                Xt = base.copy()
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[victims] = False
                Xkeep = Xt[keep_mask]
                for vi in victims:
                    # recompute hole against current keep + already placed
                    others = np.delete(Xt, vi, axis=0)
                    h = _find_hole(others, rng, n_cand=4000)
                    # small jitter
                    h = h + 0.01 * rng.normal(size=d)
                    h /= np.linalg.norm(h)
                    Xt[vi] = h
                steps = int(rng.choice([400, 550, 700]))
                s1 = float(rng.choice([180.0, 260.0, 350.0]))
                Xs = _smooth_adam(Xt, I0_all, J0_all, None, steps, 40.0, s1, 0.012, 0.002)
            else:
                # cluster re-optimisation with frozen background
                rm = _rowwise_max(base)
                order = np.argsort(-rm)
                # pick centre among top 6 for diversity
                ci = int(rng.integers(0, 6)) if rng.random() < 0.7 else int(rng.integers(0, n))
                centre = int(order[ci % n])
                kcl = int(rng.choice([7, 9, 11, 13]))
                sims = base @ base[centre]
                nearest = np.argsort(-sims)[:kcl]
                free_mask = np.zeros(n, dtype=bool)
                free_mask[nearest] = True
                Xt = base.copy()
                # reinit cluster: random sphere (sometimes around antipode of dense region)
                if rng.random() < 0.5:
                    for idx in nearest:
                        Xt[idx] = _random_vec(rng, d)
                else:
                    # jittered reinit around current positions
                    Xt[nearest] += 0.35 * rng.normal(size=(kcl, d))
                    Xt[nearest] /= np.linalg.norm(Xt[nearest], axis=1, keepdims=True)
                # valid pairs: at least one endpoint free
                valid = free_mask[I0_all] | free_mask[J0_all]
                Iv = I0_all[valid]
                Jv = J0_all[valid]
                steps = int(rng.choice([500, 700, 900]))
                s1 = float(rng.choice([200.0, 300.0, 400.0]))
                Xs = _smooth_adam(Xt, Iv, Jv, free_mask, steps, 40.0, s1, 0.015, 0.002)
        except Exception:
            continue
        try:
            v = _max_cos(Xs)
        except Exception:
            continue
        # track
        if v < best_val:
            # immediate improvement even without LP
            Xbest = Xs.copy()
            best_val = v
            Xcur = Xs.copy()
            cur_val = v
            # polish immediately if promising and time allows
            if time.time() < DEADLINE - FINAL_RESERVE - 3.0:
                try:
                    Xp = _lp_polish(Xs, 2.5, t_end=DEADLINE - FINAL_RESERVE)
                    vp = _max_cos(Xp)
                    if vp < best_val:
                        Xbest = Xp.copy()
                        best_val = vp
                        Xcur = Xp.copy()
                        cur_val = vp
                except Exception:
                    pass
            continue
        # basin-hopping acceptance for cur
        if v < cur_val or (v < cur_val + 0.004 and rng.random() < 0.25):
            Xcur = Xs.copy()
            cur_val = v
        # selective LP polish: only if close to best (potential different basin)
        if v < best_val + 0.002 and (it % 4 == 0 or v < best_val + 0.0004):
            if time.time() > DEADLINE - FINAL_RESERVE - 4.0:
                continue
            try:
                Xp = _lp_polish(Xs, 2.0, t_end=DEADLINE - FINAL_RESERVE)
                vp = _max_cos(Xp)
                if vp < best_val - 1e-12:
                    Xbest = Xp.copy()
                    best_val = vp
                    Xcur = Xp.copy()
                    cur_val = vp
                elif vp < cur_val:
                    # adopt polished cur occasionally
                    if rng.random() < 0.5:
                        Xcur = Xp.copy()
                        cur_val = vp
            except Exception:
                pass
    # final intensive polish of best
    try:
        frem = DEADLINE - time.time() - 1.0
        if frem > 3.0:
            Xf = _lp_polish(Xbest, frem, t_end=DEADLINE - 0.5)
            vf = _max_cos(Xf)
            if vf < best_val:
                Xbest = Xf
                best_val = vf
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
