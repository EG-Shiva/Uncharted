import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def quench_mixed(x_init, free_idx, steps1, steps2, T0, T1, lr):
    x = x_init.copy()
    n, d = x.shape
    # ensure normalized
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0 = np.log(T0)
    l1 = np.log(T1)
    total = steps1 + steps2
    if total <= 0:
        return x
    free_mask = np.zeros(n, dtype=bool)
    free_mask[np.asarray(free_idx, dtype=int)] = True
    all_idx = np.arange(n)
    for s in range(1, total + 1):
        frac = s / total
        T = float(np.exp(l0 + (l1 - l0) * frac))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = float(G.max())
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x) / T
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        upd = (m / c1) / (np.sqrt(v / c2) + eps)
        if s <= steps1:
            idx = free_mask
        else:
            idx = all_idx  # joint phase: use slice
            # update all
            x = x - lr * upd
            nr = np.linalg.norm(x, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            x /= nr
            continue
        # subset phase
        x[idx] = x[idx] - lr * upd[idx]
        nr = np.linalg.norm(x[idx], axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x[idx] /= nr
    return x


def quench_all(x_init, steps, T0, T1, lr):
    x = x_init.copy()
    n, d = x.shape
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0 = np.log(T0)
    l1 = np.log(T1)
    for s in range(1, steps + 1):
        T = float(np.exp(l0 + (l1 - l0) * (s / steps)))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = float(G.max())
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x) / T
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        upd = (m / c1) / (np.sqrt(v / c2) + eps)
        x = x - lr * upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and (time.time() - (t_end - seconds)) < seconds:
        if radius < 1e-14:
            break
        if time.time() + 0.3 > t_end:
            break
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        # cap constraints for speed
        if p > 2500:
            # keep most tight
            order = np.argsort(-c[sel])[:2500]
            I, J = I[order], J[order]
            p = len(I)
            csv = c[sel][order]
        else:
            csv = c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def far_hole(X_rem, rng, n_samp=8000):
    d = X_rem.shape[1]
    S = rng.standard_normal((n_samp, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    worst = (S @ X_rem.T).max(axis=1)
    return S[int(np.argmin(worst))]


def solve():
    t0 = time.time()
    T_LIMIT = 172.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(42)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None:
            try:
                a = np.array(c, dtype=float)
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                if a.shape == (102, 12) and np.all(np.isfinite(a)):
                    cands.append((max_cos(a), a))
            except Exception:
                pass
    if not cands:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a
        best = max_cos(a)
    else:
        cands.sort(key=lambda t: t[0])
        best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 102, 12
    x_cur = x_best.copy()
    cur = best
    # archive of distinct good configs
    archive = [(best, x_best.copy())]
    T_acc = 0.0018
    it = 0
    # hopping phase: reserve ~22s for final LP
    while time.time() < t_end - 22.0 and it < 5000:
        it += 1
        r = rng.random()
        x_try = x_cur.copy()
        # compute row worst for targeting
        G0 = x_cur @ x_cur.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        order = np.argsort(-rowmax)
        if r < 0.50:
            # block deletion: free k points (worst + neighbours + random)
            k = int(rng.integers(6, 15))  # 6..14
            w0 = int(order[0])
            sim = x_cur @ x_cur[w0]
            sim[w0] = -2.0
            nbr = np.argsort(-sim)[:5]
            free = set([w0] + nbr.tolist())
            # add more worst
            pi = 0
            while len(free) < k and pi < n:
                free.add(int(order[pi]))
                pi += 1
            while len(free) < k:
                free.add(int(rng.integers(n)))
            free = np.array(sorted(free), dtype=int)
            # heavy kick to free block, tiny to core
            x_try[free] += 0.28 * rng.standard_normal((len(free), d))
            mask = np.ones(n, dtype=bool)
            mask[free] = False
            x_try[mask] += 0.002 * rng.standard_normal((n - len(free), d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = quench_mixed(x_try, free, 250, 250, 0.022, 0.003, 0.003)
        elif r < 0.72:
            # hole re-insertion of 1-3 worst
            nk = int(rng.integers(1, 4))
            worst = order[:nk].copy()
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            Xrem = x_cur[mask]
            # sequential greedy holes to avoid clustering new points
            newpts = []
            Xcur = Xrem.copy()
            for _ in range(nk):
                h = far_hole(Xcur, rng, n_samp=8000)
                newpts.append(h)
                Xcur = np.vstack([Xcur, h[None, :]])
            for wi, h in zip(worst, newpts):
                x_try[wi] = h + 0.01 * rng.standard_normal(d)
            x_try[mask] += 0.003 * rng.standard_normal((n - nk, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = quench_mixed(x_try, worst, 200, 300, 0.018, 0.0025, 0.003)
        else:
            # moderate global shake from current
            sig = 0.012 if (it % 2 == 0) else 0.025
            x_try = x_cur + sig * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = quench_all(x_try, 450, 0.020, 0.003, 0.0025)
        vq = max_cos(x_q)
        if not np.isfinite(vq):
            continue
        if vq < best - 1e-12:
            best = vq
            x_best = x_q.copy()
            archive.append((best, x_best.copy()))
            if len(archive) > 4:
                archive.sort(key=lambda t: t[0])
                archive = archive[:4]
            x_cur, cur = x_q.copy(), vq
        else:
            # Metropolis walk on cur to change basin
            if vq < cur or rng.random() < np.exp((cur - vq) / T_acc):
                x_cur, cur = x_q.copy(), vq
            # occasional restart to best to avoid drift
            if it % 60 == 0 and rng.random() < 0.5:
                # cool restart: go back near best with small noise
                x_cur = x_best + 0.006 * rng.standard_normal((n, d))
                x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
                cur = max_cos(x_cur)
        # slow annealing of acceptance temp
        T_acc *= 0.9997
        if T_acc < 0.0006:
            T_acc = 0.0006
    # final snaps: polish archive best-first with remaining time
    archive.sort(key=lambda t: t[0])
    # deduplicate time: polish at most 3
    remain_total = (t_end - time.time()) - 1.0
    if remain_total < 3.0:
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        return x_best
    n_pol = min(len(archive), 3)
    per = remain_total / n_pol
    polished = []
    for k in range(n_pol):
        v0, xk = archive[k]
        budget = per if k < n_pol - 1 else (t_end - time.time() - 0.8)
        if budget < 1.0:
            polished.append((v0, xk))
            continue
        xp = lp_polish(xk, min(budget, 14.0), t_end - 0.5)
        vp = max_cos(xp)
        polished.append((vp, xp))
    polished.sort(key=lambda t: t[0])
    bv, bx = polished[0]
    bx /= np.linalg.norm(bx, axis=1, keepdims=True)
    return bx
