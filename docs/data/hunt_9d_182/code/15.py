import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    return x / n

def adam_run(x_init, steps, T0, T1, lr0, lr1, t0, time_limit, off=0):
    X = _norm(x_init.copy())
    n, d = X.shape
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = X.copy()
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lt0, lt1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for it in range(steps):
        if (it & 63) == 0 and (time.time() - t0) > time_limit:
            break
        f = it / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bv:
            bv = m
            bx = X.copy()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        s = float(E.sum() * 0.5)
        if not np.isfinite(s) or s <= 0:
            continue
        Grad = (E @ X) / s
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        glob = it + 1 + off
        m1 *= b1
        m1 += (1 - b1) * Gt
        m2 *= b2
        m2 += (1 - b2) * (Gt * Gt)
        m1h = m1 / (1 - b1 ** glob)
        m2h = m2 / (1 - b2 ** glob)
        st = m1h / (np.sqrt(m2h) + eps)
        X = X - lr * st
        X = _norm(X)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < bv:
        bv = m
        bx = X.copy()
    return bx, bv

def greedy_jam(x_init, iters, t0, time_limit, step0=6e-4):
    x = _norm(x_init.copy())
    best = _max_cos(x)
    bx = x.copy()
    step = step0
    n = x.shape[0]
    for it in range(iters):
        if (it & 63) == 0 and (time.time() - t0) > time_limit:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        dot = float(G[i, j])
        # tangent directions decreasing dot
        vi = dot * x[i] - x[j]
        vj = dot * x[j] - x[i]
        ni = float(np.linalg.norm(vi))
        nj = float(np.linalg.norm(vj))
        if ni < 1e-14 or nj < 1e-14:
            step *= 0.5
            if step < 1e-13:
                step = step0 * 0.3
            continue
        vi /= ni
        vj /= nj
        improved = False
        # try symmetric then single
        for mode in (0, 1, 2):
            y = x.copy()
            if mode == 0:
                y[i] = x[i] + step * vi
                y[j] = x[j] + step * vj
                y[i] /= np.linalg.norm(y[i])
                y[j] /= np.linalg.norm(y[j])
            elif mode == 1:
                y[i] = x[i] + step * vi
                y[i] /= np.linalg.norm(y[i])
            else:
                y[j] = x[j] + step * vj
                y[j] /= np.linalg.norm(y[j])
            v = _max_cos(y)
            if v < best:
                x = y
                best = v
                bx = y.copy()
                step = min(step * 1.2, 4e-3)
                improved = True
                break
        if not improved:
            step *= 0.5
            if step < 1e-13:
                # random tangent kick to worst point
                step = step0 * 0.5
                if it % 50 == 0:
                    w = i
                    rnd = np.random.randn(x.shape[1])
                    rnd = rnd - float(rnd @ x[w]) * x[w]
                    rn = float(np.linalg.norm(rnd))
                    if rn > 1e-12:
                        y = x.copy()
                        y[w] = y[w] + (1e-5) * rnd / rn
                        y[w] /= np.linalg.norm(y[w])
                        v = _max_cos(y)
                        if v < best + 5e-7:
                            x = y
                            if v < best:
                                best = v
                                bx = y.copy()
    return bx, best

def lp_polish(x_init, t0, time_limit, r0=6e-4):
    x = _norm(x_init.copy())
    best = _max_cos(x)
    bx = x.copy()
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    radius = r0
    while (time.time() - t0) < time_limit and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(25 * radius, 2e-7)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 4500:
            idx = np.argpartition(-c, 4500)[:4500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
            cnt = 4500
        I, J = iu[0][sel], iu[1][sel]
        p = cnt
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, ci)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(n * d + 1)
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
        dx = res.x[:-1].reshape(n, d)
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            y = x + alpha * dx
            y = _norm(y)
            v = _max_cos(y)
            if v < best - 1e-12:
                best = v
                bx = y.copy()
                x = y
                radius = min(radius * 1.5, 0.01)
                improved = True
                break
        if not improved:
            radius *= 0.5
    return bx, best

def void_replace(X, k, rng, n_cand=4000):
    Y = X.copy()
    n, d = Y.shape
    G = Y @ Y.T
    np.fill_diagonal(G, -10.0)
    per = G.max(axis=1)
    worst = np.argsort(-per)[:k]
    for idx in worst:
        others = np.delete(Y, idx, axis=0)
        cand = rng.randn(n_cand, d)
        cand = _norm(cand)
        sc = cand @ others.T
        mx = sc.max(axis=1)
        v = cand[int(np.argmin(mx))].copy()
        for _ in range(120):
            dots = others @ v
            mm = float(dots.max())
            T = 0.003
            w = np.exp((dots - mm) / T)
            w /= w.sum()
            g = w @ others
            g = g - float(g @ v) * v
            v = v - 0.07 * g
            nv = float(np.linalg.norm(v))
            if not np.isfinite(nv) or nv < 1e-12:
                break
            v /= nv
        Y[idx] = v
    return _norm(Y)

def solve():
    t_start = time.time()
    BUDGET = 168.0
    def rem():
        return BUDGET - (time.time() - t_start)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = np.random.RandomState(0).randn(182, 9)
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (182, 9):
        rngf = np.random.RandomState(1)
        y = rngf.randn(182, 9)
        if x0.ndim == 2:
            mh = min(x0.shape[0], 182)
            mw = min(x0.shape[1], 9)
            y[:mh, :mw] = x0[:mh, :mw]
        x0 = y
    x0 = _norm(x0)
    best_val = _max_cos(x0)
    x_best = x0.copy()
    n, d = 182, 9

    # Stage 1: broad anneal from incumbent
    if rem() > 30:
        xb, vb = adam_run(x_best, steps=7000, T0=0.0032, T1=0.00028,
                          lr0=0.0022, lr1=0.0002,
                          t0=time.time(), time_limit=time.time() + min(22.0, rem() - 25), off=0)
        if vb < best_val:
            best_val = vb
            x_best = xb
        # quick greedy to close single-pair gap
        xg, vg = greedy_jam(x_best, iters=1500, t0=time.time(),
                            time_limit=time.time() + min(4.0, max(0.5, rem() - 25)))
        if vg < best_val:
            best_val = vg
            x_best = xg

    # Stage 2: basin hops
    sigs = [0.005, 0.012, 0.020, 0.007, 0.015, 0.009, 0.025, 0.004]
    for trial, sig in enumerate(sigs):
        if rem() < 32:
            break
        rng = np.random.RandomState(2000 + trial * 131)
        if trial % 3 == 2:
            try:
                Xp = void_replace(x_best, 2 if trial < 5 else 3, rng)
                Xp = Xp + (sig * 0.35) * rng.randn(n, d)
            except Exception:
                Xp = x_best + sig * rng.randn(n, d)
        else:
            # contact-weighted noise + push worst pair apart
            G = x_best @ x_best.T
            np.fill_diagonal(G, -10.0)
            flat = G[np.triu_indices(n, 1)]
            iu0, iu1 = np.triu_indices(n, 1)
            k2 = int(np.argmax(flat))
            a, b = int(iu0[k2]), int(iu1[k2])
            thr = np.sort(flat)[-600]
            msk = flat > thr - 1e-12
            cnt = np.zeros(n)
            for aa, bb in zip(iu0[msk], iu1[msk]):
                cnt[aa] += 1
                cnt[bb] += 1
            mx = cnt.max() if cnt.max() > 0 else 1.0
            cnt = cnt / mx
            noise = rng.randn(n, d)
            scale = (sig * (0.35 + cnt))[:, None]
            Xp = x_best + scale * noise
            try:
                v = rng.randn(d)
                v /= np.linalg.norm(v) + 1e-12
                Xp[a] += sig * 1.8 * v
                Xp[b] -= sig * 1.8 * v
            except Exception:
                pass
        Xp = _norm(Xp)
        tlim = time.time() + min(9.0, rem() - 24)
        xb, vb = adam_run(Xp, steps=3200, T0=0.0016, T1=0.00020,
                          lr0=0.0016, lr1=0.00016,
                          t0=time.time(), time_limit=tlim, off=8000 + trial * 3200)
        # greedy jam
        xg, vg = greedy_jam(xb, iters=800, t0=time.time(),
                            time_limit=time.time() + min(2.0, max(0.3, rem() - 24)))
        if vg < vb:
            xb, vb = xg, vg
        if vb < best_val + 0.0015 and rem() > 26:
            try:
                xl, vl = lp_polish(xb, time.time(), time.time() + min(2.5, rem() - 24), r0=4e-4)
            except Exception:
                xl, vl = xb, vb
            if vl < best_val - 1e-12:
                best_val = vl
                x_best = xl
            elif vb < best_val - 1e-12:
                best_val = vb
                x_best = xb
        else:
            if vb < best_val - 1e-12:
                best_val = vb
                x_best = xb

    # Stage 3: sharp equalization
    if rem() > 12:
        xb, vb = adam_run(x_best, steps=4500, T0=0.00045, T1=0.00010,
                          lr0=0.0006, lr1=0.00010,
                          t0=time.time(), time_limit=time.time() + min(12.0, rem() - 9), off=60000)
        if vb < best_val:
            best_val = vb
            x_best = xb
        xg, vg = greedy_jam(x_best, iters=2500, t0=time.time(),
                            time_limit=time.time() + min(5.0, max(0.5, rem() - 8)))
        if vg < best_val:
            best_val = vg
            x_best = xg

    # Stage 4: intensive LP micro-polish
    r = rem() - 1.2
    if r > 3:
        r = min(r, 40.0)
        try:
            xl, vl = lp_polish(x_best, time.time(), time.time() + r * 0.7, r0=3e-4)
            if vl < best_val:
                best_val = vl
                x_best = xl
            # final greedy to squeeze last bit
            if rem() > 2:
                xg, vg = greedy_jam(x_best, iters=3000, t0=time.time(),
                                    time_limit=time.time() + (rem() - 1.0))
                if vg < best_val:
                    best_val = vg
                    x_best = xg
        except Exception:
            pass

    x_best = _norm(x_best)
    x_best[~np.isfinite(x_best)] = 0.1
    x_best = _norm(x_best)
    return x_best
