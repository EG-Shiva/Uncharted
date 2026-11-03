import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _adam_lse(y0, steps, T0, T1, lr0, lr1, rng=None):
    y = y0 / np.linalg.norm(y0, axis=1, keepdims=True)
    n, d = y.shape
    m = np.zeros_like(y)
    v = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    best = _max_cos(y)
    best_x = y.copy()
    for s in range(steps):
        f = s / max(steps - 1, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = y @ y.T
        np.fill_diagonal(G, -np.inf)
        mval = float(G.max())
        E = np.exp((G - mval) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            continue
        W = E / S
        Grad = W @ y
        dot = np.sum(Grad * y, axis=1, keepdims=True)
        Gt = Grad - dot * y
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (s + 1))
        vhat = v / (1 - b2 ** (s + 1))
        step = lr * mhat / (np.sqrt(vhat) + eps)
        y = y - step
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        if (s + 1) % 200 == 0 or s == steps - 1:
            val = _max_cos(y)
            if val < best:
                best = val
                best_x = y.copy()
    return best_x, best

def _topk_descent(y0, steps=200, K=300, lr0=0.008, lr1=0.0005):
    y = y0 / np.linalg.norm(y0, axis=1, keepdims=True)
    n, d = y.shape
    best = _max_cos(y)
    best_x = y.copy()
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        lr = lr0 * (lr1 / lr0) ** f
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        flat = G[iu]
        if K < flat.size:
            thr = np.partition(flat, -K)[-K]
        else:
            thr = flat.min() - 1
        # build gradient from pairs >= thr
        sel = flat >= thr
        I = iu[0][sel]; J = iu[1][sel]
        if len(I) == 0:
            break
        w = flat[sel] - thr + 1e-12
        w = w / w.sum() * len(I)
        grad = np.zeros_like(y)
        # accumulate weighted
        for k in range(len(I)):
            i, j = int(I[k]), int(J[k])
            wk = w[k]
            grad[i] += wk * y[j]
            grad[j] += wk * y[i]
        grad /= max(len(I), 1)
        dot = np.sum(grad * y, axis=1, keepdims=True)
        gt = grad - dot * y
        # normalize step per point to avoid blowup
        gn = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        gt = gt / np.maximum(gn, 1.0)
        y = y - lr * gt
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        if (s + 1) % 50 == 0:
            val = _max_cos(y)
            if val < best:
                best = val
                best_x = y.copy()
    val = _max_cos(y)
    if val < best:
        best, best_x = val, y.copy()
    return best_x, best

def _lp_polish_deadline(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        win = max(30.0 * radius, 2e-7)
        sel = c > m - win
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 5000:
            idx = np.where(sel)[0]
            order = np.argsort(c[idx])[::-1]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order[:5000]]] = True
            sel = keep
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def _holes(x, rng, S=15000, k=12):
    n, d = x.shape
    cand = rng.standard_normal((S, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = np.empty(S)
    for a in range(0, S, 3000):
        b = min(a + 3000, S)
        scores[a:b] = (cand[a:b] @ x.T).max(axis=1)
    order = np.argsort(scores)
    chosen = []
    for h in order:
        c = cand[h]
        ok = True
        for q in chosen:
            if abs(float(c @ cand[q])) > 0.65:
                ok = False
                break
        if ok:
            chosen.append(int(h))
            if len(chosen) >= k:
                break
    return cand[np.array(chosen)]

def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(7)
    best_x = None
    best_v = np.inf
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (256, 6):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                vv = _max_cos(a)
                if vv < best_v:
                    best_v = vv
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = rng.standard_normal((256, 6))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_v = _max_cos(best_x)
    x_best = best_x.copy()
    cur_best = float(best_v)
    cur_x = x_best.copy()
    cur_v = cur_best
    temp0 = 0.0015
    it = 0
    ANNEAL_END = t0 + LIMIT - 30.0
    while time.time() < ANNEAL_END:
        it += 1
        remain = ANNEAL_END - time.time()
        if remain < 2.0:
            break
        elapsed = time.time() - t0
        temp = temp0 * (1.0 - elapsed / LIMIT) + 1e-4
        # base: mostly best, sometimes current for diversity
        base = x_best if rng.random() < 0.75 else cur_x
        Gtmp = base @ base.T
        np.fill_diagonal(Gtmp, -2.0)
        mtmp = float(Gtmp.max())
        rmax = Gtmp.max(axis=1)
        order = np.argsort(rmax)  # ascending: rattlers first
        u = rng.random()
        x_try = base.copy()
        if u < 0.35:
            sigmas = [0.02, 0.05, 0.12, 0.25]
            sigma = sigmas[it % len(sigmas)]
            prox = np.clip((rmax - (mtmp - 0.06)) / 0.06, 0, 1)
            scale = (0.4 + 2.0 * prox)[:, None]
            x_try = x_try + sigma * scale * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            T0, T1 = 0.02, 0.0012
            steps, lr0, lr1 = 700, 0.02, 0.002
            if sigma > 0.1:
                T0, steps = 0.03, 900
        elif u < 0.65:
            ks = [8, 16, 32]
            k = int(ks[rng.integers(0, len(ks))])
            worst_pool = order[-64:]
            nw = min(k * 3 // 4, len(worst_pool))
            sel = set(int(v) for v in rng.choice(worst_pool, size=nw, replace=False))
            while len(sel) < k:
                sel.add(int(rng.integers(0, 256)))
            if rng.random() < 0.5:
                sel.add(int(order[0]))
            idx = np.array(sorted(sel), dtype=int)
            rnd = rng.standard_normal((len(idx), 6))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            x_try[idx] = rnd
            x_try = x_try + 0.008 * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            T0, T1 = 0.022, 0.001
            steps, lr0, lr1 = 1000, 0.025, 0.0015
        elif u < 0.85:
            k = int(rng.choice([6, 12, 20]))
            rep_idx = order[:k].copy()
            try:
                H = _holes(base, rng, S=12000, k=k)
            except Exception:
                H = rng.standard_normal((k, 6))
                H /= np.linalg.norm(H, axis=1, keepdims=True)
            for ii, h in zip(rep_idx, H):
                x_try[int(ii)] = h
            x_try = x_try + 0.015 * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            T0, T1 = 0.014, 0.001
            steps, lr0, lr1 = 800, 0.02, 0.0015
        else:
            # pair-break: directly separate top pairs then anneal
            iu = np.triu_indices(256, 1)
            flat = Gtmp[iu]
            top = np.argsort(flat)[-24:]
            Ii, Ji = iu[0][top], iu[1][top]
            for a, b in zip(Ii, Ji):
                a, b = int(a), int(b)
                # push apart along tangent
                pa = x_try[a] - (x_try[a] @ x_try[b]) * x_try[b]
                pb = x_try[b] - (x_try[b] @ x_try[a]) * x_try[a]
                na = np.linalg.norm(pa); nb = np.linalg.norm(pb)
                if na > 1e-12:
                    x_try[a] = x_try[a] - 0.05 * pa / na
                if nb > 1e-12:
                    x_try[b] = x_try[b] - 0.05 * pb / nb
            # plus shake hubs (highest rmax)
            hubs = order[-12:]
            x_try[hubs] += 0.06 * rng.standard_normal((len(hubs), 6))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            T0, T1 = 0.018, 0.0008
            steps, lr0, lr1 = 900, 0.022, 0.0012
        try:
            x_opt, v0 = _adam_lse(x_try, steps=steps, T0=T0, T1=T1, lr0=lr0, lr1=lr1)
        except Exception:
            continue
        try:
            x_opt2, v = _topk_descent(x_opt, steps=150, K=400, lr0=0.006, lr1=0.0006)
        except Exception:
            x_opt2, v = x_opt, v0
        # acceptance for current walk
        if v < cur_v or rng.random() < np.exp((cur_v - v) / max(temp, 1e-6)):
            cur_x, cur_v = x_opt2.copy(), float(v)
        # LP polish if promising
        thresh = cur_best + (0.003 if u < 0.65 else 0.006)
        if v < thresh:
            lp_t = 1.0 if v < cur_best + 0.0004 else 0.5
            lp_t = min(lp_t, max(0.3, ANNEAL_END - time.time() - 0.5))
            if lp_t > 0.3:
                try:
                    x_lp = _lp_polish_deadline(x_opt2, time.time() + lp_t)
                    v2 = _max_cos(x_lp)
                except Exception:
                    x_lp, v2 = x_opt2, v
                if v2 < v:
                    x_opt2, v = x_lp, v2
                    if v < cur_v:
                        cur_x, cur_v = x_opt2.copy(), float(v)
            else:
                v2 = v
            if v < cur_best:
                cur_best = float(v)
                x_best = x_opt2.copy()
    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 3:
        try:
            # short topk + long LP on best
            xb, _ = _topk_descent(x_best, steps=200, K=500, lr0=0.004, lr1=0.0004)
            if _max_cos(xb) < _max_cos(x_best):
                x_best = xb
        except Exception:
            pass
        try:
            x_best = _lp_polish_deadline(x_best, time.time() + min(rem, 26.0))
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
