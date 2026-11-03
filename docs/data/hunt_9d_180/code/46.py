import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def build_full(S):
    # S: (20,9) -> X: (180,9) by 9 cyclic rolls per seed
    assert S.shape == (20, 9), f"seed shape {S.shape}"
    X = np.empty((180, 9), dtype=float)
    for i in range(20):
        s = S[i]
        for k in range(9):
            X[i * 9 + k] = np.roll(s, k)
    assert X.shape == (180, 9), "count failed"
    assert X.shape[0] == 180
    return X

def adam_sym(S_init, alphas_steps_lrs, rng=None):
    S = S_init / np.linalg.norm(S_init, axis=1, keepdims=True)
    m = np.zeros_like(S)
    v = np.zeros_like(S)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = 180
    for alpha, steps, lr in alphas_steps_lrs:
        for _ in range(steps):
            t += 1
            X = build_full(S)
            Xn = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
            X = X / Xn
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            gmax = G.max()
            E = np.exp(alpha * (G - gmax))
            np.fill_diagonal(E, 0.0)
            Ssum = E.sum()
            if not np.isfinite(Ssum) or Ssum == 0:
                break
            gradX = (E @ X) * (n / Ssum)
            ndx = (gradX * X).sum(axis=1, keepdims=True)
            gx = gradX - ndx * X
            # aggregate to seeds: Gs[i,p] = sum_k gx[i*9+k,(p+k)%9]
            Gs = np.zeros_like(S)
            for i in range(20):
                block = gx[i * 9:(i + 1) * 9]  # (9,9): rows k, cols j
                # for each p, sum over k of block[k,(p+k)%9]
                for k in range(9):
                    # contribution of row k to all p: p = (j-k) mod 9
                    row = block[k]
                    # roll row by -k? j->p=j-k => Gs[i] += np.roll(row, -k)? check:
                    # Gs[i,p] += row[(p+k)%9], so vector over p is roll(row, -k)? roll(row,-k)[p]=row[(p+k)%9] yes
                    Gs[i] += np.roll(row, -k)
            nd = (Gs * S).sum(axis=1, keepdims=True)
            gt = Gs - nd * S
            m = b1 * m + (1.0 - b1) * gt
            v = b2 * v + (1.0 - b2) * gt * gt
            bc1 = 1.0 - b1 ** t
            bc2 = 1.0 - b2 ** t
            mhat = m / bc1
            vhat = v / bc2
            delta = lr * mhat / (np.sqrt(vhat) + eps)
            dn = (delta * S).sum(axis=1, keepdims=True)
            delta = delta - dn * S
            rn = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12
            clip = 0.05
            scale = np.minimum(1.0, clip / (rn + 1e-12))
            delta = delta * scale
            S = S - delta
            S /= np.linalg.norm(S, axis=1, keepdims=True)
    X = build_full(S)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return S, X

def adam_free(x, alpha, steps, lr):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        E = np.exp(alpha * (G - gmax))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S == 0:
            break
        grad = (E @ x) * (n / S)
        nd = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - nd * x
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * gt * gt
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mhat = m / bc1
        vhat = v / bc2
        delta = lr * mhat / (np.sqrt(vhat) + eps)
        dn = (delta * x).sum(axis=1, keepdims=True)
        delta = delta - dn * x
        rn = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12
        clip = 0.05
        scale = np.minimum(1.0, clip / (rn + 1e-12))
        delta = delta * scale
        x = x - delta
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def smooth_refine(x):
    x = adam_free(x, 25.0, 250, 0.006)
    x = adam_free(x, 70.0, 300, 0.004)
    x = adam_free(x, 180.0, 350, 0.002)
    x = adam_free(x, 450.0, 200, 0.0009)
    return x

def lp_polish(x, seconds, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c_full = (x @ x.T)[iu]
        m = c_full.max()
        sel = c_full > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 6000:
            idx = np.where(sel)[0]
            order = np.argsort(-c_full[idx])[:6000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        c = c_full[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def solve():
    np.random.seed(12345)
    t0 = time.time()
    deadline = t0 + 176.0
    # fallback best
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (180, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = max_cos(a)
                    if v < best_val:
                        best_val = v
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        r0 = np.random.RandomState(0)
        best = r0.randn(180, 9)
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = max_cos(best)
    # ---- Stage 1: symmetry-reduced search (20 seeds x 9 cyclic shifts) ----
    sym_best_X = None
    sym_best_val = 1e9
    try:
        rng = np.random.RandomState(7)
        sched_short = [(30.0, 200, 0.012), (90.0, 200, 0.007), (220.0, 250, 0.0035), (500.0, 150, 0.0012)]
        n_restart = 3
        for r in range(n_restart):
            if time.time() > t0 + 45.0:
                break
            S0 = rng.randn(20, 9)
            S0 /= np.linalg.norm(S0, axis=1, keepdims=True)
            _, Xsym = adam_sym(S0, sched_short)
            # immediate shape/count assertion
            assert Xsym.shape == (180, 9)
            assert Xsym.shape[0] == 180
            Xsym /= np.linalg.norm(Xsym, axis=1, keepdims=True)
            if not np.all(np.isfinite(Xsym)):
                continue
            v = max_cos(Xsym)
            if v < sym_best_val:
                sym_best_val = v
                sym_best_X = Xsym.copy()
    except Exception:
        sym_best_X = None
    # if construction failed, fallback continues with best
    candidates = []
    if sym_best_X is not None:
        candidates.append(sym_best_X)
    # ---- Stage 2: break symmetry, free optimize ----
    # deterministic perturbations 1e-3 .. 1e-2
    sigmas = [0.002, 0.005, 0.009]
    rng2 = np.random.RandomState(99)
    # also include best as candidate for free hops if sym bad
    free_time_limit = t0 + 145.0
    tried = 0
    # build list of starting points: perturbed sym + perturbed best
    starts = []
    if sym_best_X is not None:
        for s in sigmas:
            starts.append(("sym", sym_best_X, s))
    # add one perturbed best start to hedge
    starts.append(("best", best, 0.006))
    if sym_best_X is not None:
        starts.append(("sym", sym_best_X, 0.003))
    for kind, X0, sig in starts:
        if time.time() > free_time_limit:
            break
        if deadline - time.time() < 12.0:
            break
        try:
            y = X0 + sig * rng2.randn(180, 9)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_refine(y)
            v = max_cos(y)
            # short LP snap if promising
            if v < best_val + 0.002:
                secs = 1.5 if v >= best_val else 2.5
                remain = free_time_limit - time.time()
                secs = min(secs, max(0.5, remain - 0.5))
                if secs > 0.4:
                    y = lp_polish(y, secs, radius_init=0.004)
                    v = max_cos(y)
            if v < best_val:
                best = y
                best_val = v
            # also track sym-derived even if not beating best, keep for final polish diversity:
            # keep best sym-derived free optimum
            tried += 1
        except Exception:
            continue
    # extra free hops from current best with small kicks if time remains
    rng3 = np.random.RandomState(2024)
    while time.time() < free_time_limit - 2.0 and (deadline - time.time()) > 14.0:
        # only a couple iterations to avoid timeout
        if tried > 6:
            break
        try:
            sig = float(rng3.choice([0.004, 0.008, 0.015]))
            y = best + sig * rng3.randn(180, 9)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = smooth_refine(y)
            v = max_cos(y)
            if v < best_val + 0.001:
                secs = min(2.0, max(0.6, free_time_limit - time.time() - 0.5))
                if secs > 0.4 and v < best_val + 0.0005:
                    y = lp_polish(y, secs, radius_init=0.003)
                    v = max_cos(y)
            if v < best_val:
                best = y
                best_val = v
            tried += 1
        except Exception:
            break
        break  # single extra hop to save time for final polish
    # ---- Stage 3: final LP polish ----
    remain = deadline - time.time() - 1.0
    if remain > 2.0:
        try:
            best = lp_polish(best, min(remain, 25.0), radius_init=1e-3)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
