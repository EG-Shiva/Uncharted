import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, seconds, r_init=1e-3, t_end=1e30):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    try:
        best = _max_cos(X)
    except Exception:
        return X
    radius = r_init
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds or now >= t_end or radius < 1e-14:
            break
        Gfull = (X @ X.T)[iu0]
        m = float(np.max(Gfull))
        thresh = max(20.0 * radius, 1e-9)
        # cap constraints to ~900 tightest for speed
        # select top-K if too many
        sel = Gfull > m - thresh
        cnt = int(np.sum(sel))
        if cnt > 900:
            # keep 900 largest
            # partition threshold
            thr = np.partition(Gfull, -900)[-900]
            sel = Gfull >= thr
            # if still >1100 due to ties, subsample randomly? keep all ties but truncate
            idx = np.where(sel)[0]
            if len(idx) > 1100:
                # keep first 1100 of tightest (sorted)
                order = np.argsort(-Gfull[idx])[:1100]
                mask = np.zeros_like(sel)
                mask[idx[order]] = True
                sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        csub = Gfull[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y = y / nr
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            X = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _smooth(X, steps, T0, T1, s0, s1, rng, noise0=0.0):
    n, d = X.shape
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 1.0
        T = T0 * (1.0 - f) + T1 * f
        step = s0 * (1.0 - f) + s1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # weights
        # avoid overflow: (G-m)<=0
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        # sparsify far pairs (optional, saves nothing but stabilizes)
        # W[G < m - 6*T] = 0.0
        grad = W @ X
        proj = np.sum(grad * X, axis=1, keepdims=True)
        grad = grad - proj * X
        gn = float(np.max(np.sum(grad * grad, axis=1)))
        gn = np.sqrt(gn)
        if not np.isfinite(gn) or gn < 1e-14:
            continue
        grad *= (step / gn)
        X = X - grad
        if noise0 > 0:
            nz = rng.randn(n, d) * (noise0 * (1.0 - f))
            # tangential component only
            nz -= np.sum(nz * X, axis=1, keepdims=True) * X
            X = X + nz
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _hole_best(X, idx, rng, n_cand=2000):
    n, d = X.shape
    C = rng.randn(n_cand, d)
    cn = np.linalg.norm(C, axis=1, keepdims=True)
    cn[cn == 0] = 1.0
    C = C / cn
    others = np.delete(X, idx, axis=0)
    dots = C @ others.T
    md = np.max(dots, axis=1)
    return C[int(np.argmin(md))]

def _worst_indices(X, k):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    mx = np.max(G, axis=1)
    return np.argsort(-mx)[:k], mx

def solve():
    t0 = time.time()
    DEADLINE = t0 + 176.0
    HOP_END = t0 + 145.0
    rng = np.random.RandomState(7)
    n, d = 73, 13
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(np.float64)
            if v.shape != (n, d):
                continue
            nr = np.linalg.norm(v, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            v = v / nr
            if not np.all(np.isfinite(v)):
                continue
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        best = rng.randn(n, d)
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    Xbest = best.copy()
    Xcur = best.copy()
    cur_val = best_val

    it = 0
    # temperature schedule for Metropolis on cur
    while time.time() < HOP_END and time.time() < DEADLINE - 10:
        it += 1
        frac = (time.time() - t0) / (HOP_END - t0)
        frac = min(max(frac, 0.0), 1.0)
        Temp = 0.003 * (1.0 - frac) + 0.0002
        # choose base: mostly cur, sometimes best
        if rng.rand() < 0.25:
            base = Xbest
        else:
            base = Xcur
        X = base.copy()
        mode = it % 5
        try:
            if mode == 0:
                # global small shake
                sigma = 0.02 if (it % 2 == 0) else 0.05
                X = X + sigma * rng.randn(n, d)
                nr = np.linalg.norm(X, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                X = X / nr
                # reinsert single worst
                wi, _ = _worst_indices(X, 1)
                X[wi[0]] = _hole_best(X, int(wi[0]), rng, n_cand=2500)
            elif mode == 1:
                # coherent cap shake
                u = rng.randn(d)
                u /= np.linalg.norm(u)
                dots = X @ u
                k = int(rng.randint(4, 11))
                order = np.argsort(-dots)[:k]
                alpha = 0.06 + 0.08 * rng.rand()
                X[order] += alpha * u + 0.03 * rng.randn(k, d)
                # plus tiny global jitter
                X += 0.008 * rng.randn(n, d)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                wi, _ = _worst_indices(X, 1)
                X[int(wi[0])] = _hole_best(X, int(wi[0]), rng, n_cand=2000)
            elif mode == 2:
                # jam-break: kick most-contacted of base
                G = base @ base.T
                np.fill_diagonal(G, -np.inf)
                mg = float(np.max(G))
                cnt = np.sum(G > mg - 5e-4, axis=1)
                order = np.argsort(-cnt)
                j = int(order[rng.randint(0, 3)])
                X[j] += 0.25 * rng.randn(d)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                # reinsert j into hole computed from perturbed X
                X[j] = _hole_best(X, j, rng, n_cand=3500)
                # small shake rest
                X += 0.01 * rng.randn(n, d)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            elif mode == 3:
                # double hole reinsert
                wi, _ = _worst_indices(X, 2)
                for jj in wi:
                    X[int(jj)] = _hole_best(X, int(jj), rng, n_cand=2500)
                X += 0.015 * rng.randn(n, d)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
            else:
                # medium global kick + cap shift
                X = X + 0.08 * rng.randn(n, d)
                X /= np.linalg.norm(X, axis=1, keepdims=True)
                u = rng.randn(d)
                u /= np.linalg.norm(u)
                dots = X @ u
                k = int(rng.randint(3, 7))
                order = np.argsort(-dots)[:k]
                X[order] += 0.05 * u
                X /= np.linalg.norm(X, axis=1, keepdims=True)
        except Exception:
            continue
        # cheap smooth to change topology (no LP yet)
        try:
            # vary annealing per mode
            if mode == 2:
                X = _smooth(X, steps=350, T0=0.015, T1=0.0015, s0=0.025, s1=0.003, rng=rng, noise0=0.004)
            elif mode == 1:
                X = _smooth(X, steps=300, T0=0.012, T1=0.001, s0=0.02, s1=0.002, rng=rng, noise0=0.002)
            else:
                X = _smooth(X, steps=280, T0=0.01, T1=0.001, s0=0.02, s1=0.002, rng=rng, noise0=0.001)
        except Exception:
            continue
        try:
            v = _max_cos(X)
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        # Metropolis accept to cur
        accept = False
        if v < cur_val:
            accept = True
        else:
            try:
                if rng.rand() < np.exp((cur_val - v) / max(Temp, 1e-9)):
                    accept = True
            except Exception:
                accept = False
        if accept:
            Xcur = X.copy()
            cur_val = v
        # promising? -> short LP polish
        # only polish if close to global best to save time
        if v < best_val + 0.0025 and v < cur_val + 0.005:
            tl = min(1.6, HOP_END - (time.time() - t0), DEADLINE - 10 - (time.time() - t0))
            if tl < 0.4:
                # skip polish, just check improvement
                if v < best_val:
                    Xbest = X.copy()
                    best_val = v
                continue
            try:
                Y = _lp_polish(X, seconds=tl, r_init=3e-3, t_end=DEADLINE - 2)
                vy = _max_cos(Y)
            except Exception:
                continue
            if vy < best_val and np.all(np.isfinite(Y)):
                Xbest = Y.copy()
                best_val = vy
                # also move cur to new best region
                Xcur = Y.copy()
                cur_val = vy
            elif vy < cur_val:
                Xcur = Y.copy()
                cur_val = vy
        else:
            if v < best_val:
                Xbest = X.copy()
                best_val = v
        if time.time() > HOP_END:
            break
    # final long polish
    remain = DEADLINE - time.time() - 1.0
    if remain < 3.0:
        remain = 3.0
    remain = min(remain, 28.0)
    try:
        Xbest = _lp_polish(Xbest, seconds=remain, r_init=1e-3, t_end=DEADLINE - 0.5)
    except Exception:
        pass
    nr = np.linalg.norm(Xbest, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    Xbest = Xbest / nr
    return Xbest
