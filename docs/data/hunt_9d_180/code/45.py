import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def expand_seeds(S):
    S = np.asarray(S, dtype=float)
    assert S.shape == (20, 9), f"seed shape {S.shape} != (20,9)"
    X = np.empty((180, 9), dtype=float)
    for k in range(20):
        rk = S[k]
        for s in range(9):
            X[k * 9 + s] = np.roll(rk, s)
    assert X.shape == (180, 9), "construction failed shape"
    assert X.shape[0] == 180, "construction failed count"
    # no NaN, no zero rows
    assert np.all(np.isfinite(X)), "non-finite in construction"
    nrm = np.linalg.norm(X, axis=1)
    assert np.all(nrm > 1e-12), "zero row in construction"
    X = X / nrm[:, None]
    return X


def adam_sym(S, alpha, steps, lr, rng_clip=0.05):
    S = S / np.linalg.norm(S, axis=1, keepdims=True)
    m = np.zeros_like(S)
    v = np.zeros_like(S)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        X = expand_seeds(S)  # (180,9) normalized
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        E = np.exp(alpha * (G - gmax))
        np.fill_diagonal(E, 0.0)
        Sm = E.sum()
        if not np.isfinite(Sm) or Sm == 0:
            break
        # grad w.r.t X (softmax weighted), scale by n/Sm style
        grad_X = (E @ X) * (180.0 / Sm)
        # backprop to seeds via inverse rolls
        grad_S = np.zeros_like(S)
        for k in range(20):
            acc = np.zeros(9)
            base = k * 9
            for s in range(9):
                acc += np.roll(grad_X[base + s], -s)
            grad_S[k] = acc
        # tangent projection on seeds
        nd = (grad_S * S).sum(axis=1, keepdims=True)
        gt = grad_S - nd * S
        m = b1 * m + (1.0 - b1) * gt
        vv = b2 * v + (1.0 - b2) * gt * gt
        # need v variable name not clash
        v = vv
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        mhat = m / bc1
        vhat = v / bc2
        delta = lr * mhat / (np.sqrt(vhat) + eps)
        dn = (delta * S).sum(axis=1, keepdims=True)
        delta = delta - dn * S
        rn = np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12
        scale = np.minimum(1.0, rng_clip / (rn + 1e-12))
        delta = delta * scale
        S = S - delta
        S = S / np.linalg.norm(S, axis=1, keepdims=True)
    return S


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
        vv = max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x


def solve():
    np.random.seed(7)
    t0 = time.time()
    deadline = t0 + 176.0
    # load fallback best
    fallback = None
    fallback_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (180, 9):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                vv = max_cos(a)
                if vv < fallback_val:
                    fallback_val = vv
                    fallback = a.copy()
        except Exception:
            pass
    if fallback is None:
        r0 = np.random.RandomState(7)
        fallback = r0.randn(180, 9)
        fallback /= np.linalg.norm(fallback, axis=1, keepdims=True)
        fallback_val = max_cos(fallback)

    best_sym_X = None
    best_sym_val = 1e9
    best_sym_S = None
    # symmetry-reduced search
    try:
        sched = [(30.0, 350, 0.010), (80.0, 350, 0.007), (200.0, 350, 0.004), (500.0, 250, 0.002)]
        for restart in range(3):
            if time.time() > t0 + 70.0:
                break
            rng = np.random.RandomState(1000 + restart)
            S = rng.randn(20, 9)
            S /= np.linalg.norm(S, axis=1, keepdims=True)
            for (alpha, steps, lr) in sched:
                if time.time() > t0 + 70.0:
                    break
                # chunk steps to allow time check
                chunk = 175
                done = 0
                while done < steps:
                    if time.time() > t0 + 70.0:
                        break
                    cs = min(chunk, steps - done)
                    S = adam_sym(S, alpha, cs, lr)
                    done += cs
            try:
                X = expand_seeds(S)
                assert X.shape == (180, 9) and np.all(np.isfinite(X))
                vv = max_cos(X)
            except Exception:
                continue
            if vv < best_sym_val:
                best_sym_val = vv
                best_sym_X = X.copy()
                best_sym_S = S.copy()
    except Exception:
        pass

    # if construction failed, fallback immediately
    if best_sym_X is None or best_sym_S is None or not np.all(np.isfinite(best_sym_X)):
        # fallback path required by contract
        try:
            x = fallback / np.linalg.norm(fallback, axis=1, keepdims=True)
        except Exception:
            r = np.random.RandomState(7)
            x = r.randn(180, 9)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        remain = deadline - time.time() - 1.0
        if remain > 3.0:
            try:
                x = lp_polish(x, min(remain, 20.0), radius_init=1e-3)
            except Exception:
                pass
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        return x

    # break symmetry: release all 180 independently + tiny deterministic perturbation 1e-3..1e-2
    try:
        rngb = np.random.RandomState(999)
        sigma = 0.005  # within 1e-3..1e-2
        Xfree = best_sym_X + sigma * rngb.randn(180, 9)
        Xfree /= np.linalg.norm(Xfree, axis=1, keepdims=True)
        assert Xfree.shape == (180, 9)
    except Exception:
        Xfree = fallback.copy()

    # free optimisation (annealed LogSumExp Adam)
    try:
        free_sched = [(25.0, 250, 0.006), (70.0, 300, 0.004), (180.0, 350, 0.002), (450.0, 200, 0.0009)]
        for (alpha, steps, lr) in free_sched:
            if time.time() > t0 + 130.0:
                break
            Xfree = adam_free(Xfree, alpha, steps, lr)
    except Exception:
        pass

    # choose better of free vs fallback before polish
    try:
        vfree = max_cos(Xfree)
    except Exception:
        Xfree = fallback.copy()
        vfree = fallback_val

    cur = Xfree if vfree < fallback_val + 0.002 else fallback.copy()
    # if free is much worse, still keep free for diversity? keep best
    if vfree < fallback_val:
        cur = Xfree
    else:
        # keep free anyway if we want to polish a new basin; but ensure we return best at end
        # polish the more promising: polish both briefly if time allows -> polish free first
        cur = Xfree

    remain = deadline - time.time() - 1.0
    if remain > 3.0:
        try:
            cur = lp_polish(cur, min(remain, 28.0), radius_init=0.004)
        except Exception:
            pass

    try:
        vcur = max_cos(cur)
    except Exception:
        cur = fallback.copy()
        vcur = fallback_val

    # final safety: return best among cur and fallback
    if fallback_val < vcur:
        cur = fallback.copy()

    cur = cur / np.linalg.norm(cur, axis=1, keepdims=True)
    # strict shape contract
    assert cur.shape == (180, 9)
    return cur
