import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def pair_to_Y(X, rng=None):
    # greedy near-antipodal pairing: repeatedly take most negative pair
    n = X.shape[0]
    G = X @ X.T
    unpaired = np.ones(n, dtype=bool)
    Y = []
    for _ in range(n // 2):
        idx = np.where(unpaired)[0]
        if len(idx) == 0:
            break
        if len(idx) == 2:
            a, b = idx[0], idx[1]
        else:
            sub = G[np.ix_(idx, idx)]
            # mask diagonal
            np.fill_diagonal(sub, 10.0)
            k = int(np.argmin(sub))
            ii, jj = divmod(k, len(idx))
            a, b = idx[ii], idx[jj]
        # representative along a-b (since b ~ -a)
        v = X[a] - X[b]
        nv = np.linalg.norm(v)
        if nv < 1e-12 or not np.isfinite(nv):
            v = X[a]
            nv = np.linalg.norm(v)
        v = v / nv
        Y.append(v)
        unpaired[a] = False
        unpaired[b] = False
    Y = np.array(Y, dtype=float)
    # safety: if odd leftover, pad random
    if Y.shape[0] != n // 2:
        extra = (n // 2) - Y.shape[0]
        r = rng.standard_normal((extra, X.shape[1])) if rng is not None else np.random.standard_normal((extra, X.shape[1]))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        Y = np.vstack([Y, r]) if len(Y) else r
    return Y


def adam_antipodal(Y0, steps=1500, a0=40.0, a1=250.0, lr=0.004):
    Y = Y0.copy()
    n, d = Y.shape
    m = np.zeros_like(Y)
    vv = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        frac = t / max(1, steps - 1)
        alpha = a0 * ((a1 / a0) ** frac)
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        A = np.abs(G)
        M = float(np.max(A))
        # avoid overflow: exp(alpha*(A-M)) in [0,1]
        E = np.exp(alpha * (A - M))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        Sg = np.sign(G)
        # gradient of LSE(|.|)
        Ew = E * Sg
        Grad = (Ew @ Y) / S
        dot = np.sum(Grad * Y, axis=1, keepdims=True)
        Gt = Grad - dot * Y
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        Y = Y - lr * step
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        Y /= nr
    return Y


def adam_full(X0, steps=1000, a0=40.0, a1=350.0, lr=0.003):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        frac = t / max(1, steps - 1)
        alpha = a0 * ((a1 / a0) ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        lrt = lr * (1.0 - 0.5 * frac)
        X = X - lrt * step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        X /= nr
    return X


def lp_polish_deadline(x, deadline, r_init=0.004, max_pairs=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    iu0 = np.triu_indices(n, 1)
    radius = r_init
    it = 0
    while time.time() < deadline and radius > 1e-13 and it < 80:
        it += 1
        try:
            G = x @ x.T
            c_all = G[iu0]
            m = float(c_all.max())
        except Exception:
            break
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > max_pairs:
            part = np.argpartition(c_all, -max_pairs)[-max_pairs:]
            sel2 = np.zeros_like(sel)
            sel2[part] = True
            sel = sel2
        I = iu0[0][sel]
        J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        cols = np.empty((p, 2 * d + 1), dtype=int)
        ar = np.arange(d)
        cols[:, :d] = I[:, None] * d + ar
        cols[:, d:2 * d] = J[:, None] * d + ar
        cols[:, 2 * d] = n * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
                radius *= 0.5
                continue
            y = y / yn
            v = max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    QUENCH_END = t0 + 140.0
    rng = np.random.default_rng(777)
    n, d, h = 176, 9, 88

    # load incumbents
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    cands.append(a / nr)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])

    # ---- build Y seeds ----
    Y_seeds = []
    # record-derived pairing from best and record
    for X in cands[:2]:
        try:
            Y_seeds.append(pair_to_Y(X, rng))
        except Exception:
            pass
    # lattice seed: random choice among {-1,0,1}^9
    try:
        lat = rng.integers(-1, 2, size=(2000, d)).astype(float)
        nz = np.linalg.norm(lat, axis=1) > 1e-9
        lat = lat[nz]
        lat /= np.linalg.norm(lat, axis=1, keepdims=True)
        # deduplicate roughly by greedy farthest to 88
        perm = rng.permutation(lat.shape[0])[:500]
        pool = lat[perm]
        # farthest point sample 88
        sel = [0]
        dist = np.full(pool.shape[0], 1e9)
        for _ in range(1, h):
            last = pool[sel[-1]]
            dd = 1.0 - np.abs(pool @ last)
            dist = np.minimum(dist, dd)
            sel.append(int(np.argmax(dist)))
        Y_seeds.append(pool[np.array(sel)])
    except Exception:
        pass
    # random seeds
    for _ in range(3):
        R = rng.standard_normal((h, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        Y_seeds.append(R)

    # ---- antipodal Adam: alpha 40->250, lr 0.004 ----
    Y_opts = []
    for Y0 in Y_seeds:
        if time.time() > QUENCH_END - 30:
            break
        try:
            Yo = adam_antipodal(Y0, steps=1500, a0=40.0, a1=250.0, lr=0.004)
        except Exception:
            continue
        Y_opts.append(Yo)
        # keep symmetric candidate immediately
        try:
            Xs = np.vstack([Yo, -Yo])
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            v = max_cos(Xs)
            if v < best_v and np.all(np.isfinite(Xs)):
                best_v = v
                best_X = Xs.copy()
        except Exception:
            pass

    # if antipodal failed, keep at least something
    if not Y_opts:
        Y_opts = Y_seeds[:1]

    # ---- release symmetry gradually + 1000-step LogSumExp quench ----
    amps = [0.0, 0.01, 0.03, 0.08]
    # order Y_opts by symmetric score (best first) to spend time well
    scored = []
    for Yo in Y_opts:
        try:
            Xs = np.vstack([Yo, -Yo])
            v = max_cos(Xs / np.linalg.norm(Xs, axis=1, keepdims=True))
        except Exception:
            v = 1e9
        scored.append(v)
    order = np.argsort(np.array(scored))
    Y_opts = [Y_opts[i] for i in order]

    for Yo in Y_opts:
        for amp in amps:
            if time.time() > QUENCH_END:
                break
            try:
                Yp = Yo.copy()
                # independent tangent perturbations for +/- sides
                if amp > 0:
                    # top half perturbations
                    N1 = rng.standard_normal((h, d))
                    # project to tangent of each y
                    dot1 = np.sum(N1 * Yp, axis=1, keepdims=True)
                    N1 -= dot1 * Yp
                    N1 /= (np.linalg.norm(N1, axis=1, keepdims=True) + 1e-12)
                    N2 = rng.standard_normal((h, d))
                    dot2 = np.sum(N2 * Yp, axis=1, keepdims=True)
                    N2 -= dot2 * Yp
                    N2 /= (np.linalg.norm(N2, axis=1, keepdims=True) + 1e-12)
                    # random tangent magnitudes ~ amp * |N(0,1)|
                    s1 = amp * np.abs(rng.standard_normal((h, 1)))
                    s2 = amp * np.abs(rng.standard_normal((h, 1)))
                    Ypos = Yp + s1 * N1
                    Yneg = -Yp + s2 * N2
                    Ypos /= np.linalg.norm(Ypos, axis=1, keepdims=True)
                    Yneg /= np.linalg.norm(Yneg, axis=1, keepdims=True)
                    X0f = np.vstack([Ypos, Yneg])
                else:
                    X0f = np.vstack([Yp, -Yp])
                    X0f /= np.linalg.norm(X0f, axis=1, keepdims=True)
                Xq = adam_full(X0f, steps=1000, a0=40.0, a1=350.0, lr=0.003)
                vq = max_cos(Xq)
                if vq < best_v and np.all(np.isfinite(Xq)):
                    best_v = vq
                    best_X = Xq.copy()
            except Exception:
                continue
        if time.time() > QUENCH_END:
            break

    # ---- single deadline-aware LP polish ----
    try:
        fd = HARD_END - 1.0
        if time.time() < fd - 0.5:
            best_X = lp_polish_deadline(best_X, fd, r_init=0.004)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
