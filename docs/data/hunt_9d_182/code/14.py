import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    return x / n

def adam_refine(x_init, steps, T0, T1, lr0, lr1, time_limit, it_offset=0):
    t0 = time.time()
    n, d = x_init.shape
    X = _norm(x_init.copy())
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_val = float(G0.max())
    best_x = X.copy()
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lt0, lt1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for it in range(steps):
        if (it & 127) == 0 and time.time() - t0 > time_limit:
            break
        frac = it / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_val:
            best_val = m
            best_x = X.copy()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        s = float(E.sum() * 0.5)
        if not np.isfinite(s) or s <= 0:
            continue
        Grad = (E @ X) / s
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        glob = it + 1 + it_offset
        m1 *= b1
        m1 += (1 - b1) * Gt
        m2 *= b2
        m2 += (1 - b2) * (Gt * Gt)
        m1h = m1 / (1 - b1 ** glob)
        m2h = m2 / (1 - b2 ** glob)
        step = m1h / (np.sqrt(m2h) + eps)
        X = X - lr * step
        X = _norm(X)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < best_val:
        best_val = m
        best_x = X.copy()
    return best_x, best_val

def greedy_pair(x, iters, t_deadline):
    x = _norm(x.copy())
    best = max_cos(x)
    step = 4e-4
    n = x.shape[0]
    iu0 = np.triu_indices(n, 1)
    for it in range(iters):
        if time.time() >= t_deadline:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        i = k // n
        j = k % n
        dot = float(G[i, j])
        vi = dot * x[i] - x[j]
        vj = dot * x[j] - x[i]
        ni = float(np.linalg.norm(vi))
        nj = float(np.linalg.norm(vj))
        if ni < 1e-14 or nj < 1e-14:
            step *= 0.5
            if step < 1e-12:
                step = 2e-4
            continue
        vi /= ni
        vj /= nj
        improved = False
        for s in (step, step * 0.5):
            y = x.copy()
            y[i] = x[i] + s * vi
            y[j] = x[j] + s * vj
            y[i] /= np.linalg.norm(y[i])
            y[j] /= np.linalg.norm(y[j])
            v = max_cos(y)
            if v < best - 1e-13:
                x = y
                best = v
                step = min(step * 1.2, 3e-3)
                improved = True
                break
            # single sided
            y2 = x.copy()
            y2[i] = x[i] + s * vi
            y2[i] /= np.linalg.norm(y2[i])
            v2 = max_cos(y2)
            if v2 < best - 1e-13:
                x = y2
                best = v2
                step = min(step * 1.15, 3e-3)
                improved = True
                break
            y3 = x.copy()
            y3[j] = x[j] + s * vj
            y3[j] /= np.linalg.norm(y3[j])
            v3 = max_cos(y3)
            if v3 < best - 1e-13:
                x = y3
                best = v3
                step = min(step * 1.15, 3e-3)
                improved = True
                break
        if not improved:
            step *= 0.6
            if step < 5e-7:
                step = 5e-7
                # random tangent kick to worst point occasionally
                if (it % 50) == 0:
                    step = 2e-4
    return x, best

def lp_polish_ls(x, seconds):
    x = _norm(x.copy())
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 6e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-14:
        it += 1
        if time.time() - start > seconds:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        # include band wide enough for curvature
        thresh = m - max(30.0 * radius, 4e-7)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 5000:
            idx = np.argpartition(-c, 5000)[:5000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
            cnt = 5000
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
            y = _norm(x + alpha * dx)
            v = max_cos(y)
            if v < best - 1e-12:
                best = v
                best_x = y.copy()
                x = y
                radius = min(radius * 1.5, 0.01)
                improved = True
                break
        if not improved:
            radius *= 0.5
        if it % 20 == 0 and time.time() - start > seconds:
            break
    return best_x

def void_reinsert(X, k, rng, n_cand=2500):
    n, d = X.shape
    Y = X.copy()
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
            v = _norm((v - 0.08 * g)[None, :])[0]
        Y[idx] = v
    return _norm(Y)

def solve():
    np.random.seed(0)
    t_start = time.time()
    TIME_BUDGET = 168.0
    def remaining():
        return TIME_BUDGET - (time.time() - t_start)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            rng0 = np.random.RandomState(0)
            x0 = rng0.randn(182, 9)
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (182, 9):
        rng = np.random.RandomState(1)
        y = rng.randn(182, 9)
        if x0.ndim == 2:
            mh = min(x0.shape[0], 182)
            mw = min(x0.shape[1], 9)
            y[:mh, :mw] = x0[:mh, :mw]
        x0 = y
    x0 = _norm(x0)
    best_val = max_cos(x0)
    x_best = x0.copy()
    n = 182

    # Stage 0: immediate greedy + LP jam of current basin (fast gain from single pair)
    if remaining() > 30:
        try:
            xg, vg = greedy_pair(x_best, 1500, time.time() + min(6.0, remaining() - 25))
            if vg < best_val - 1e-13:
                best_val = vg
                x_best = xg
        except Exception:
            pass
        try:
            xlp = lp_polish_ls(x_best, min(14.0, remaining() - 24))
            vlp = max_cos(xlp)
            if vlp < best_val - 1e-13:
                best_val = vlp
                x_best = xlp
        except Exception:
            pass

    # Stage 1: broad anneal from best
    if remaining() > 40:
        try:
            xb, vb = adam_refine(x_best, steps=7000, T0=0.0030, T1=0.00028,
                                 lr0=0.0022, lr1=0.00022,
                                 time_limit=min(22.0, remaining() - 35))
            if vb < best_val + 0.002 and remaining() > 32:
                try:
                    xg2, vg2 = greedy_pair(xb, 600, time.time() + 2.0)
                    if vg2 < vb:
                        xb, vb = xg2, vg2
                except Exception:
                    pass
                try:
                    xlp = lp_polish_ls(xb, min(3.0, remaining() - 30))
                    vlp = max_cos(xlp)
                    if vlp < best_val - 1e-12:
                        best_val = vlp
                        x_best = xlp
                    elif vb < best_val - 1e-12:
                        best_val = vb
                        x_best = xb
                except Exception:
                    if vb < best_val - 1e-12:
                        best_val = vb
                        x_best = xb
            else:
                if vb < best_val - 1e-12:
                    best_val = vb
                    x_best = xb
        except Exception:
            pass

    # Stage 2: basin hops with contact-weighted noise + void reinsertion
    sigs = [0.004, 0.010, 0.018, 0.007, 0.025]
    for trial in range(len(sigs)):
        if remaining() < 36:
            break
        rng = np.random.RandomState(2000 + trial * 131)
        sig = sigs[trial]
        try:
            if trial == 2 or trial == 4:
                k = 2 if trial == 2 else 3
                Xp = void_reinsert(x_best, k, rng)
                Xp = Xp + sig * 0.35 * rng.randn(n, 9)
                Xp = _norm(Xp)
            else:
                G = x_best @ x_best.T
                np.fill_diagonal(G, -10.0)
                flat = G[np.triu_indices(n, 1)]
                thr = np.sort(flat)[-600]
                iu0, iu1 = np.triu_indices(n, 1)
                msk = flat > thr - 1e-12
                cnt = np.zeros(n)
                for a, b in zip(iu0[msk], iu1[msk]):
                    cnt[a] += 1
                    cnt[b] += 1
                mx = cnt.max() if cnt.max() > 0 else 1.0
                cnt = cnt / mx
                noise = rng.randn(n, 9)
                scale = (sig * (0.4 + cnt))[:, None]
                Xp = x_best + scale * noise
                try:
                    k2 = int(np.argmax(flat))
                    a, b = int(iu0[k2]), int(iu1[k2])
                    v = rng.randn(9)
                    v /= np.linalg.norm(v) + 1e-12
                    Xp[a] += sig * 2.0 * v
                    Xp[b] -= sig * 2.0 * v
                except Exception:
                    pass
                Xp = _norm(Xp)
        except Exception:
            continue
        try:
            xb, vb = adam_refine(Xp, steps=3200, T0=0.0016, T1=0.00020,
                                 lr0=0.0016, lr1=0.00016,
                                 time_limit=min(8.0, remaining() - 28),
                                 it_offset=8000 + trial * 3200)
        except Exception:
            continue
        if vb < best_val + 0.002 and remaining() > 28:
            try:
                xlp = lp_polish_ls(xb, min(3.0, remaining() - 26))
                vlp = max_cos(xlp)
            except Exception:
                xlp, vlp = xb, vb
            if vlp < best_val - 1e-12:
                best_val = vlp
                x_best = xlp
            elif vb < best_val - 1e-12:
                best_val = vb
                x_best = xb
        else:
            if vb < best_val - 1e-12:
                best_val = vb
                x_best = xb

    # Stage 3: sharp low-T equalization
    if remaining() > 14:
        try:
            xb, vb = adam_refine(x_best, steps=4500, T0=0.00045, T1=0.00010,
                                 lr0=0.0006, lr1=0.00010,
                                 time_limit=min(12.0, remaining() - 10),
                                 it_offset=60000)
            if vb < best_val:
                best_val = vb
                x_best = xb
        except Exception:
            pass

    # Stage 4: intensive final LP jam
    rem = remaining() - 1.5
    if rem > 3:
        rem = min(rem, 50.0)
        try:
            xg, vg = greedy_pair(x_best, 800, time.time() + min(3.0, rem * 0.15))
            if vg < best_val - 1e-13:
                best_val = vg
                x_best = xg
                rem = remaining() - 1.5
                rem = min(max(rem, 0), 50.0)
        except Exception:
            pass
        if rem > 3:
            try:
                x_best = lp_polish_ls(x_best, rem)
                best_val = max_cos(x_best)
            except Exception:
                pass

    x_best = _norm(x_best)
    x_best[~np.isfinite(x_best)] = 0.1
    x_best = _norm(x_best)
    return x_best
