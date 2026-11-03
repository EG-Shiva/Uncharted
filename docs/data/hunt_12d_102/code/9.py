import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def adam_quench(x_init, steps, T0, T1, lr):
    x = x_init.copy()
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    lT0 = np.log(T0); lT1 = np.log(T1)
    for s in range(1, steps + 1):
        lt = lT0 + (lT1 - lT0) * (s / steps)
        T = float(np.exp(lt))
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
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20 * radius, 1e-9)
        # cap constraints for speed
        if int(sel.sum()) > 2500:
            # keep top 2500
            idx = np.argpartition(-c, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
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
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def best_holes(X_rem, rng, k, n_samp=8000):
    d = X_rem.shape[1]
    S = rng.standard_normal((n_samp, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    chosen = []
    cur = X_rem
    for _ in range(k):
        if len(chosen) > 0:
            aug = np.vstack([cur] + [c[None, :] for c in chosen])
        else:
            aug = cur
        dots = S @ aug.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        chosen.append(S[bi].copy())
        # avoid picking same hole again: penalize nearby candidates
        # by removing near-duplicates
        dd = S @ chosen[-1]
        keep = dd < 0.98
        if int(keep.sum()) > 1000:
            S = S[keep]
    return chosen

def solve():
    t0 = time.time()
    T_LIMIT = 172.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (102, 12) and np.all(np.isfinite(a)):
                cands.append((max_cos(a), a))
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a; best = max_cos(a)
    else:
        cands.sort(key=lambda t: t[0])
        best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 102, 12
    # elite pool for diverse restarts
    elites = [(best, x_best.copy())]
    if len(cands) > 1:
        elites.append((cands[1][0], cands[1][1].copy()))

    it = 0
    # hopping phase, reserve ~18s final
    while time.time() < t_end - 20.0 and it < 500:
        it += 1
        # pick base: mostly best, sometimes second elite
        if len(elites) > 1 and (it % 7 == 0):
            # sort elites
            elites.sort(key=lambda t: t[0])
            base = elites[1][1]
        else:
            base = x_best
        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        mval = float(rowmax.max())
        cnt = (G0 > mval - 0.005).sum(axis=1)
        order = np.argsort(-rowmax)
        order_c = np.argsort(-cnt)
        mode = it % 6
        x_try = base.copy()
        if mode == 0:
            sig = float(rng.uniform(0.02, 0.045))
            x_try = base + sig * rng.standard_normal((n, d))
            steps = int(rng.integers(450, 750)); T0 = float(rng.uniform(0.02, 0.05)); T1 = float(rng.uniform(0.0025, 0.0045)); lr = float(rng.uniform(0.0018, 0.0028))
        elif mode == 1:
            sig = float(rng.uniform(0.06, 0.12))
            x_try = base + sig * rng.standard_normal((n, d))
            steps = int(rng.integers(500, 850)); T0 = float(rng.uniform(0.03, 0.06)); T1 = float(rng.uniform(0.002, 0.004)); lr = float(rng.uniform(0.0018, 0.003))
        elif mode == 2:
            k = int(rng.integers(2, 7))
            idx = order[:k]
            x_try[idx] += float(rng.uniform(0.08, 0.18)) * rng.standard_normal((k, d))
            x_try += 0.004 * rng.standard_normal((n, d))
            steps = int(rng.integers(500, 900)); T0 = float(rng.uniform(0.02, 0.045)); T1 = float(rng.uniform(0.0025, 0.004)); lr = float(rng.uniform(0.0018, 0.0026))
        elif mode == 3:
            ci = int(order_c[int(rng.integers(0, 6))])
            Gr = (base @ base[ci]); Gr[ci] = -2.0
            nn = int(rng.integers(4, 9))
            nbr = np.argsort(-Gr)[:nn]
            cl = np.append(nbr, ci)
            x_try[cl] += float(rng.uniform(0.05, 0.11)) * rng.standard_normal((len(cl), d))
            x_try += 0.003 * rng.standard_normal((n, d))
            steps = int(rng.integers(500, 850)); T0 = float(rng.uniform(0.02, 0.05)); T1 = float(rng.uniform(0.0025, 0.004)); lr = float(rng.uniform(0.0018, 0.0028))
        elif mode == 4:
            k = int(rng.integers(2, 5))
            worst = order[:k].copy()
            # order worst shuffled
            rng.shuffle(worst)
            mask = np.ones(n, dtype=bool); mask[worst] = False
            holes = best_holes(base[mask], rng, k, n_samp=8000)
            for wi, h in zip(worst, holes):
                x_try[wi] = h + 0.006 * rng.standard_normal(d)
            x_try[mask] += 0.004 * rng.standard_normal((n - k, d))
            steps = int(rng.integers(550, 950)); T0 = float(rng.uniform(0.018, 0.04)); T1 = float(rng.uniform(0.002, 0.0038)); lr = float(rng.uniform(0.0018, 0.0026))
        else:
            # single-hole + medium global shake
            wi = int(order[0])
            mask = np.ones(n, dtype=bool); mask[wi] = False
            holes = best_holes(base[mask], rng, 1, n_samp=6000)
            x_try[wi] = holes[0] + 0.005 * rng.standard_normal(d)
            x_try += float(rng.uniform(0.008, 0.025)) * rng.standard_normal((n, d))
            steps = int(rng.integers(450, 800)); T0 = float(rng.uniform(0.015, 0.04)); T1 = float(rng.uniform(0.0025, 0.0045)); lr = float(rng.uniform(0.0018, 0.0028))
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        x_q = adam_quench(x_try, steps, T0, T1, lr)
        vq = max_cos(x_q)
        if vq < best - 1e-12:
            budget = min(4.0, max(1.0, (t_end - 20.0 - time.time()) / 4.0))
            if time.time() + budget < t_end - 18.0:
                x_q = lp_polish(x_q, budget, t_end - 18.0)
                vq = max_cos(x_q)
            if vq < best:
                x_best, best = x_q, vq
                elites.append((best, x_best.copy()))
                elites.sort(key=lambda t: t[0])
                elites = elites[:3]
        elif vq < best + 0.004 and (it % 6 == 0):
            budget = min(2.0, max(0.7, (t_end - 20.0 - time.time()) / 10.0))
            if budget > 0.6 and time.time() + budget < t_end - 18.0:
                x_p = lp_polish(x_q, budget, t_end - 18.0)
                vp = max_cos(x_p)
                if vp < best:
                    x_best, best = x_p, vp
                    elites.append((best, x_best.copy()))
                    elites.sort(key=lambda t: t[0])
                    elites = elites[:3]
                elif vp < elites[-1][0] and len(elites) < 3:
                    elites.append((vp, x_p.copy()))
        if time.time() >= t_end - 20.0:
            break
    remain = t_end - time.time() - 2.0
    if remain > 2.0:
        x_best = lp_polish(x_best, min(remain, 16.0), t_end - 1.0)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
