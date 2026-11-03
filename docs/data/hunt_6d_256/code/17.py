import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_lse(y0, steps, T0, T1, lr0, lr1):
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
        E /= S
        Grad = E @ y
        dot = np.sum(Grad * y, axis=1, keepdims=True)
        Gt = Grad - dot * y
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (s + 1))
        vhat = v / (1 - b2 ** (s + 1))
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # remove radial part
        step = step - np.sum(step * y, axis=1, keepdims=True) * y
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

def _topk_descent(y0, steps=200, K=30, lr0=0.005, lr1=0.0004):
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
        sel = flat >= thr
        I = iu[0][sel]; J = iu[1][sel]
        if len(I) == 0:
            break
        vals = flat[sel]
        w = vals - thr + 1e-12
        w = w / w.sum() * len(I)
        grad = np.zeros_like(y)
        # vectorized accumulate
        np.add.at(grad, I, (w[:, None] * y[J]))
        np.add.at(grad, J, (w[:, None] * y[I]))
        grad /= max(len(I), 1)
        dot = np.sum(grad * y, axis=1, keepdims=True)
        gt = grad - dot * y
        gn = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        # limit per-point step to 1
        scale = np.minimum(1.0, 1.0 / np.maximum(gn, 1e-12).squeeze()[:, None] if False else 1.0)
        gt = gt / np.maximum(gn, 1.0)
        y = y - lr * gt
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        if (s + 1) % 80 == 0:
            val = _max_cos(y)
            if val < best:
                best = val
                best_x = y.copy()
    val = _max_cos(y)
    if val < best:
        best, best_x = val, y.copy()
    return best_x, best

def _squeeze_top(x0, deadline):
    x = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    iu = np.triu_indices(n, 1)
    lr = 3e-4
    it = 0
    while time.time() < deadline:
        it += 1
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        flat = G[iu]
        # true bottleneck: within 3e-9 of max (usually 2 pairs)
        sel = flat >= m - 3e-9
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            # fallback to top 4
            thr = np.partition(flat, -4)[-4]
            sel = flat >= thr
            cnt = int(np.count_nonzero(sel))
        I = iu[0][sel]; J = iu[1][sel]
        grad = np.zeros_like(x)
        np.add.at(grad, I, x[J])
        np.add.at(grad, J, x[I])
        grad /= max(cnt, 1)
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        gn = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-12
        gt = gt / np.maximum(gn, 1.0)
        # only move involved points to avoid raising bulk
        involved = np.zeros(n, dtype=bool)
        involved[I] = True
        involved[J] = True
        step = np.zeros_like(x)
        step[involved] = gt[involved]
        y = x - lr * step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            best = vv
            best_x = y.copy()
            x = y
            lr = min(lr * 1.3, 5e-4)
            # early exit if huge win? continue
        else:
            lr *= 0.5
            if lr < 1e-11:
                # restart with smaller K / different lr
                lr = 5e-05 if (it % 2 == 0) else 3e-4
                # if stuck many times, perturb slightly the involved points orthogonally
                if it % 20 == 0:
                    break
        if it > 3000:
            break
    return best_x, best

def _lp_polish(x, deadline, narrow=True):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 5e-4 if narrow else 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        if narrow:
            win = max(8.0 * radius, 5e-9)
        else:
            win = max(30.0 * radius, 2e-7)
        sel = c > m - win
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(c[idx])[::-1]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order[:3000]]] = True
            sel = keep
        I, J = iu[0][sel], iu[1][sel]
        p = int(np.count_nonzero(sel))
        csel = c[sel]
        nv = n * d + 1
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
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
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
            radius = min(radius * 1.6, 0.02)
        else:
            radius *= 0.5
    return x

def _void_candidates(x, rng, S=10000, k=8):
    n, d = x.shape
    cand = rng.standard_normal((S, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = np.empty(S)
    for a in range(0, S, 2500):
        b = min(a + 2500, S)
        scores[a:b] = (cand[a:b] @ x.T).max(axis=1)
    order = np.argsort(scores)
    chosen = []
    for h in order:
        c = cand[h]
        ok = True
        for q in chosen:
            if abs(float(c @ cand[q])) > 0.6:
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
    rng = np.random.default_rng(12345)
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
    temp0 = 0.0012
    it = 0
    ANNEAL_END = t0 + LIMIT - 28.0
    while time.time() < ANNEAL_END:
        it += 1
        remain = ANNEAL_END - time.time()
        if remain < 1.5:
            break
        elapsed = time.time() - t0
        temp = temp0 * (1.0 - elapsed / LIMIT) + 8e-05
        base = x_best if rng.random() < 0.72 else cur_x
        Gtmp = base @ base.T
        np.fill_diagonal(Gtmp, -2.0)
        mtmp = float(Gtmp.max())
        rmax = Gtmp.max(axis=1)
        order = np.argsort(rmax)
        u = rng.random()
        x_try = base.copy()
        if u < 0.32:
            sigmas = [0.015, 0.04, 0.10, 0.22]
            sigma = sigmas[it % len(sigmas)]
            prox = np.clip((rmax - (mtmp - 0.06)) / 0.06, 0, 1)
            scale = (0.35 + 2.2 * prox)[:, None]
            x_try = x_try + sigma * scale * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            T0, T1 = 0.025, 0.00035
            steps, lr0, lr1 = 650, 0.02, 0.002
            if sigma > 0.08:
                T0, steps = 0.032, 850
        elif u < 0.60:
            ks = [6, 12, 24, 36]
            k = int(ks[rng.integers(0, len(ks))])
            worst_pool = order[-64:]
            nw = min(k * 3 // 4, len(worst_pool))
            sel = set(int(v) for v in rng.choice(worst_pool, size=nw, replace=False))
            while len(sel) < k:
                sel.add(int(rng.integers(0, 256)))
            idx = np.array(sorted(sel), dtype=int)
            rnd = rng.standard_normal((len(idx), 6))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            x_try[idx] = rnd
            x_try = x_try + 0.006 * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            T0, T1 = 0.024, 0.0003
            steps, lr0, lr1 = 900, 0.025, 0.0015
        elif u < 0.82:
            k = int(rng.choice([5, 10, 16]))
            rep_idx = order[:k].copy()
            try:
                H = _void_candidates(base, rng, S=10000, k=k)
            except Exception:
                H = rng.standard_normal((k, 6))
                H /= np.linalg.norm(H, axis=1, keepdims=True)
            for ii, h in zip(rep_idx, H):
                x_try[int(ii)] = h
            x_try = x_try + 0.012 * rng.standard_normal(x_try.shape)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            T0, T1 = 0.016, 0.0003
            steps, lr0, lr1 = 750, 0.02, 0.0015
        else:
            iu = np.triu_indices(256, 1)
            flat = Gtmp[iu]
            top = np.argsort(flat)[-20:]
            Ii, Ji = iu[0][top], iu[1][top]
            for a, b in zip(Ii, Ji):
                a, b = int(a), int(b)
                pa = x_try[a] - (x_try[a] @ x_try[b]) * x_try[b]
                pb = x_try[b] - (x_try[b] @ x_try[a]) * x_try[a]
                na = np.linalg.norm(pa); nb = np.linalg.norm(pb)
                if na > 1e-12:
                    x_try[a] = x_try[a] - 0.06 * pa / na
                if nb > 1e-12:
                    x_try[b] = x_try[b] - 0.06 * pb / nb
            hubs = order[-10:]
            x_try[hubs] += 0.05 * rng.standard_normal((len(hubs), 6))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            T0, T1 = 0.020, 0.00025
            steps, lr0, lr1 = 850, 0.022, 0.0012
        try:
            x_opt, v0 = _adam_lse(x_try, steps=steps, T0=T0, T1=T1, lr0=lr0, lr1=lr1)
        except Exception:
            continue
        try:
            # sharp focus on top 40 then broader 300 mix
            x_opt2, v = _topk_descent(x_opt, steps=140, K=40, lr0=0.004, lr1=0.0004)
        except Exception:
            x_opt2, v = x_opt, v0
        if v < cur_v or rng.random() < np.exp((cur_v - v) / max(temp, 1e-6)):
            cur_x, cur_v = x_opt2.copy(), float(v)
        # squeeze isolated peak immediately (cheap)
        if v < cur_best + 0.002:
            try:
                xs, vs = _squeeze_top(x_opt2, time.time() + 0.25)
                if vs < v:
                    x_opt2, v = xs, vs
                    if v < cur_v:
                        cur_x, cur_v = x_opt2.copy(), float(v)
            except Exception:
                pass
        if v < cur_best + 0.0006:
            lp_t = 0.9 if v < cur_best + 0.00015 else 0.4
            lp_t = min(lp_t, max(0.25, ANNEAL_END - time.time() - 0.4))
            if lp_t > 0.25:
                try:
                    x_lp = _lp_polish(x_opt2, time.time() + lp_t, narrow=True)
                    v2 = _max_cos(x_lp)
                except Exception:
                    x_lp, v2 = x_opt2, v
                if v2 < v:
                    x_opt2, v = x_lp, v2
                    if v < cur_v:
                        cur_x, cur_v = x_opt2.copy(), float(v)
            if v < cur_best:
                cur_best = float(v)
                x_best = x_opt2.copy()
    # final intensive polish
    rem = LIMIT - (time.time() - t0) - 1.0
    if rem > 4:
        try:
            xb, vb = _adam_lse(x_best, steps=700, T0=0.004, T1=5e-05, lr0=0.012, lr1=0.001)
            if vb < _max_cos(x_best):
                x_best = xb
        except Exception:
            pass
    if rem > 3:
        try:
            xb2, _ = _topk_descent(x_best, steps=350, K=25, lr0=0.003, lr1=0.0002)
            if _max_cos(xb2) < _max_cos(x_best):
                x_best = xb2
        except Exception:
            pass
    try:
        xs, _ = _squeeze_top(x_best, time.time() + min(6.0, max(1.0, rem * 0.25)))
        if _max_cos(xs) < _max_cos(x_best):
            x_best = xs
    except Exception:
        pass
    try:
        x_best = _lp_polish(x_best, time.time() + max(1.0, LIMIT - (time.time() - t0) - 1.0), narrow=True)
    except Exception:
        pass
    try:
        xs2, _ = _squeeze_top(x_best, time.time() + max(0.5, LIMIT - (time.time() - t0) - 0.8))
        if _max_cos(xs2) < _max_cos(x_best):
            x_best = xs2
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
