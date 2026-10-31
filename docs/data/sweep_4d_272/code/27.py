import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _refine(X0, stages, mu=0.85, max_step=0.05):
    X = X0.copy()
    vel = np.zeros_like(X)
    for beta, steps, lr in stages:
        for _ in range(steps):
            G = X @ X.T
            np.fill_diagonal(G, -1.0)
            m = float(np.max(G))
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            rs = np.sum(W, axis=1, keepdims=True) + 1e-12
            grad = W @ X
            d = np.sum(grad * X, axis=1, keepdims=True)
            gt = (grad - d * X) / rs
            vel = mu * vel - lr * gt
            vn = np.sqrt(np.sum(vel * vel, axis=1, keepdims=True)) + 1e-12
            scale = np.minimum(1.0, max_step / vn)
            vel = vel * scale
            X = X + vel
            # normalize inline
            nn = np.linalg.norm(X, axis=1, keepdims=True)
            nn = np.maximum(nn, 1e-12)
            X = X / nn
    return X


def _lp_polish(x, t_end):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        if time.time() + 0.8 > t_end and radius < 5e-4:
            break
        c = (x @ x.T)[iu0]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            break
        # subsample if absurdly many (keep tightest)
        if p > 4000:
            idx = np.argsort(-c[sel])[:4000]
            I = I[idx]; J = J[idx]
            cc = c[sel][idx]
            p = 4000
        else:
            cc = c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _void_for(others, rng, n_cand=3000):
    cand = rng.standard_normal((n_cand, 4))
    cand = _normalize(cand)
    s = cand @ others.T
    scores = np.max(s, axis=1)
    bi = int(np.argmin(scores))
    b = cand[bi].copy()
    bs = float(scores[bi])
    step = 0.05
    for _ in range(200):
        prop = b + step * rng.standard_normal((32, 4))
        prop = _normalize(prop)
        sc = np.max(prop @ others.T, axis=1)
        j = int(np.argmin(sc))
        if sc[j] < bs:
            b, bs = prop[j], float(sc[j])
            step *= 1.05
        else:
            step *= 0.97
        if step < 1e-5:
            break
    return b


def solve():
    t0 = time.time()
    T_HARD = t0 + 175.0
    T_EXP = t0 + 128.0
    rng = np.random.default_rng(12345)
    try:
        best = np.load("best.npy")
    except Exception:
        try:
            best = np.load("record.npy")
        except Exception:
            best = rng.standard_normal((272, 4))
    best = _normalize(np.asarray(best, dtype=np.float64))
    if best.shape != (272, 4):
        best = _normalize(rng.standard_normal((272, 4)))
    try:
        rec = _normalize(np.asarray(np.load("record.npy"), dtype=np.float64))
        if rec.shape == (272, 4):
            vr = _max_cos(rec)
            vb = _max_cos(best)
            if vr < vb:
                best = rec.copy()
    except Exception:
        pass
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    n = 272

    fast_stages = [(80.0, 150, 0.012), (220.0, 200, 0.007), (650.0, 220, 0.003)]
    hop = 0
    while time.time() < T_EXP and hop < 500:
        if time.time() + 0.45 > T_EXP:
            break
        elapsed = time.time() - t0
        prog = elapsed / max(1e-9, (T_EXP - t0))
        temp = 0.0016 * (1.0 - prog) + 0.00008
        mode = hop % 7
        Xp = cur.copy()
        if mode == 0:
            Xp = Xp + 0.012 * rng.standard_normal(Xp.shape)
        elif mode == 1:
            Xp = Xp + 0.035 * rng.standard_normal(Xp.shape)
        elif mode == 2:
            Xp = Xp + 0.080 * rng.standard_normal(Xp.shape)
        elif mode == 3:
            # coherent cluster kick: pick center, move its nearest neighbours together
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            ci = int(rng.integers(0, n))
            order = np.argsort(-G[ci])[:12]
            # random tangent direction at center
            rv = rng.standard_normal(4)
            rv = rv - np.dot(rv, cur[ci]) * cur[ci]
            nn = np.linalg.norm(rv) + 1e-12
            rv = rv / nn
            mag = 0.10 + 0.05 * rng.random()
            Xp[order] = Xp[order] + mag * rv[None, :]
            mask = np.ones(n, dtype=bool)
            mask[order] = False
            Xp[mask] = Xp[mask] + 0.008 * rng.standard_normal((n - len(order), 4))
        elif mode == 4:
            # overloaded / worst personal-best kick
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            cnt = np.sum(G > m - 2e-6, axis=1)
            pw = np.max(G, axis=1)
            score = cnt.astype(float) * 2.0 + (pw - m + 0.01) * 50.0
            idx = np.argsort(-score)[:20]
            # random subset of them gets big kick
            k = 8 + int(rng.integers(0, 8))
            sel = rng.choice(idx, size=min(k, len(idx)), replace=False)
            Xp[sel] = Xp[sel] + 0.13 * rng.standard_normal((len(sel), 4))
            mask = np.ones(n, dtype=bool)
            mask[sel] = False
            Xp[mask] = Xp[mask] + 0.008 * rng.standard_normal((n - len(sel), 4))
        elif mode == 5:
            # worst-point void reinsertion (topology change)
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            pw = np.max(G, axis=1)
            k = 3 + int(rng.integers(0, 3))  # 3-5
            worst = np.argsort(-pw)[:k]
            # find holes sequentially
            tmp = Xp.copy()
            # randomize order
            rng.shuffle(worst)
            for wi in worst:
                others = np.delete(tmp, wi, axis=0)
                b = _void_for(others, rng, n_cand=2500)
                tmp[wi] = b + 0.004 * rng.standard_normal(4)
            Xp = tmp
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            Xp[mask] = Xp[mask] + 0.007 * rng.standard_normal((n - k, 4))
        else:
            k = 5
            idx = rng.choice(n, size=k, replace=False)
            Xp[idx] = rng.standard_normal((k, 4))
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Xp[mask] = Xp[mask] + 0.010 * rng.standard_normal((n - k, 4))
        Xp = _normalize(Xp)
        Xnew = _refine(Xp, fast_stages, mu=0.85, max_step=0.05)
        v = _max_cos(Xnew)
        if v < best_val:
            best = Xnew
            best_val = v
            cur = Xnew
            cur_val = v
        elif v < cur_val:
            cur = Xnew
            cur_val = v
        else:
            # Metropolis acceptance of worsening to change basin
            if rng.random() < np.exp((cur_val - v) / max(temp, 1e-9)):
                cur = Xnew
                cur_val = v
            # occasional restart to best to intensify
            if hop % 25 == 24 and rng.random() < 0.5:
                cur = best.copy()
                cur_val = best_val
        hop += 1

    # medium sharpen of current best before LP if time allows
    if time.time() + 2.0 < T_HARD - 3.0:
        try:
            cand = _refine(best, [(150.0, 250, 0.008), (450.0, 300, 0.004), (950.0, 250, 0.002)], mu=0.85, max_step=0.04)
            vv = _max_cos(cand)
            if vv < best_val:
                best = cand
                best_val = vv
        except Exception:
            pass
    t_left_end = T_HARD - 1.0
    if time.time() < t_left_end - 1.0:
        try:
            best = _lp_polish(best, t_left_end)
        except Exception:
            pass
    best = _normalize(best)
    return best
