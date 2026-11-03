import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def adam_refine(x_init, steps, T0, T1, lr0, lr1, time_limit, it_offset=0):
    t0 = time.time()
    n, d = x_init.shape
    X = x_init / np.linalg.norm(x_init, axis=1, keepdims=True)
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
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[~np.isfinite(nrm)] = 1.0
        nrm[nrm < 1e-12] = 1.0
        X = X / nrm
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    if m < best_val:
        best_val = m
        best_x = X.copy()
    return best_x, best_val

def lp_polish_ls(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-14:
        it += 1
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        thresh = m - max(25 * radius, 5e-7)
        sel = c > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 6000:
            idx = np.argpartition(-c, 6000)[:6000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = sel & mask
            cnt = 6000
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
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
        # line search along dx
        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            y = x + alpha * dx
            nr = np.linalg.norm(y, axis=1, keepdims=True)
            nr[~np.isfinite(nr)] = 1.0
            nr[nr < 1e-12] = 1.0
            y = y / nr
            v = max_cos(y)
            if v < best - 1e-12:
                best = v
                best_x = y.copy()
                x = y
                radius = min(radius * 1.6, 0.02)
                improved = True
                break
        if not improved:
            # check if LP predicted improvement but curvature blocked: shrink
            # if LP target already >= best, we are locally optimal at this radius -> shrink to refine
            radius *= 0.5
        # early exit if time low
        if it % 10 == 0 and time.time() - start > seconds:
            break
    return best_x

def void_reinsert(X, k, rng, n_cand=2000):
    n, d = X.shape
    Y = X.copy()
    G = Y @ Y.T
    np.fill_diagonal(G, -10.0)
    per = G.max(axis=1)
    worst = np.argsort(-per)[:k]
    for idx in worst:
        others = np.delete(Y, idx, axis=0)
        cand = rng.randn(n_cand, d)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True) + 1e-12
        sc = cand @ others.T
        mx = sc.max(axis=1)
        v = cand[int(np.argmin(mx))].copy()
        # local refine void: minimize softmax max
        for _ in range(150):
            dots = others @ v
            mm = float(dots.max())
            T = 0.004
            w = np.exp((dots - mm) / T)
            w /= w.sum()
            g = w @ others
            g = g - float(g @ v) * v
            # adaptive step
            v = v - 0.08 * g
            nv = np.linalg.norm(v)
            if not np.isfinite(nv) or nv < 1e-12:
                break
            v /= nv
        Y[idx] = v
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
    return Y

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
    x0 = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    best_val = max_cos(x0)
    x_best = x0.copy()

    n = 182
    # Stage 1: broad anneal from best
    if remaining() > 25:
        xb, vb = adam_refine(x_best, steps=9000, T0=0.0035, T1=0.00030,
                             lr0=0.0025, lr1=0.00022,
                             time_limit=min(32.0, remaining() - 20))
        if vb < best_val:
            best_val = vb
            x_best = xb

    # Stage 2: basin hops
    trial_sigs = [0.004, 0.009, 0.016, 0.007, 0.022, 0.011]
    for trial in range(len(trial_sigs)):
        if remaining() < 38:
            break
        rng = np.random.RandomState(1000 + trial * 77)
        sig = trial_sigs[trial]
        if trial == 3 or trial == 5:
            # void reinsertion of worst 1-3 points + small noise
            k = 2 if trial == 3 else 3
            try:
                Xp = void_reinsert(x_best, k, rng)
            except Exception:
                Xp = x_best.copy()
            Xp = Xp + sig * 0.4 * rng.randn(n, 9)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        else:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -10.0)
            flat = G[np.triu_indices(n, 1)]
            thr = np.sort(flat)[-500]
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
            # push current worst pair apart
            try:
                k2 = int(np.argmax(flat))
                a, b = int(iu0[k2]), int(iu1[k2])
                v = rng.randn(9)
                v /= np.linalg.norm(v) + 1e-12
                Xp[a] += sig * 2.0 * v
                Xp[b] -= sig * 2.0 * v
            except Exception:
                pass
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        xb, vb = adam_refine(Xp, steps=3500, T0=0.0018, T1=0.00022,
                             lr0=0.0018, lr1=0.00018,
                             time_limit=min(10.0, remaining() - 28),
                             it_offset=9000 + trial * 3500)
        # quick local LP to judge basin
        if vb < best_val + 0.002 and remaining() > 30:
            try:
                xlp = lp_polish_ls(xb, min(3.5, remaining() - 27))
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
        xb, vb = adam_refine(x_best, steps=5000, T0=0.0005, T1=0.00012,
                             lr0=0.0007, lr1=0.00012,
                             time_limit=min(16.0, remaining() - 10),
                             it_offset=60000)
        if vb < best_val:
            best_val = vb
            x_best = xb

    # Stage 4: intensive LP polish
    rem = remaining() - 1.5
    if rem > 4:
        rem = min(rem, 55.0)
        try:
            x_best = lp_polish_ls(x_best, rem)
            best_val = max_cos(x_best)
        except Exception:
            pass

    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best[~np.isfinite(x_best)] = 0.1
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
