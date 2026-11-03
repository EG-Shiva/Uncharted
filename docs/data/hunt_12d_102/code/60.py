import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def adam_quench(x_init, steps, T0, T1, lr):
    x = x_init.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    for s in range(1, steps + 1):
        lt = lT0 + (lT1 - lT0) * (s / steps)
        T = float(np.exp(lt))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = G.max()
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x) / T
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        mh = m / c1
        vh = v / c2
        upd = mh / (np.sqrt(vh) + eps)
        x = x - lr * upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def refine_hole(q0, Xf, steps=250, T0=0.01, T1=0.001, lr0=0.08, lr1=0.005):
    q = q0.copy()
    q /= np.linalg.norm(q)
    m = np.zeros_like(q)
    vv = np.zeros_like(q)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    for s in range(1, steps + 1):
        f = s / steps
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(lr0 + (lr1 - lr0) * f)
        dots = Xf @ q
        M = dots.max()
        w = np.exp((dots - M) / T)
        sw = w.sum()
        if not np.isfinite(sw) or sw == 0:
            break
        grad = (w @ Xf) / sw
        gt = grad - np.dot(grad, q) * q
        m = b1 * m + (1.0 - b1) * gt
        vv = b2 * vv + (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1 ** s)
        vh = vv / (1.0 - b2 ** s)
        upd = mh / (np.sqrt(vh) + eps)
        q = q - lr * upd
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n == 0:
            break
        q /= n
    return q


def global_hole(X_cur, rng, n_samp=50000):
    d = X_cur.shape[1]
    # chunk to bound memory but 50k x 92 is fine in one go
    S = rng.standard_normal((n_samp, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    dots = S @ X_cur.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    q0 = S[bi]
    q = refine_hole(q0, X_cur, steps=250)
    # compare refined vs raw, keep better
    def worst_of(qq):
        return float((X_cur @ qq).max())
    if worst_of(q) > worst_of(q0):
        q = q0
    return q


def spectra_top(x, k=1200):
    iu = np.triu_indices(x.shape[0], 1)
    c = (x @ x.T)[iu]
    c = np.sort(c)[::-1]
    return c[:k]


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
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    T_LIMIT = 172.0
    t_end = t0 + T_LIMIT
    t_polish = t_end - 20.0
    t_dead = min(t_polish, t0 + 165.0)
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (102, 12) and np.all(np.isfinite(a)):
                cands.append((max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append((max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_g, x_best = cands[0][0], cands[0][1].copy()
    xc = x_best.copy()
    cur = best_g
    spec_best = spectra_top(x_best)
    spec_cur = spec_best.copy()
    n, d = 102, 12
    trial = 0
    # simultaneous many-point deletion / global regrowth
    while time.time() < t_dead - 3.0 and trial < 12:
        trial += 1
        # time needed estimate: leave room for at least one regrowth
        if time.time() > t_dead - 3.0:
            break
        k = 10 + (trial * 5) % 9  # 10..18
        # analyse overload on xc
        G = xc @ xc.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        rowmax = G.max(axis=1)
        cnt = (G > mx - 0.005).sum(axis=1)
        order_row = np.argsort(-rowmax)
        order_cnt = np.argsort(-cnt)
        mode = trial % 4
        if mode == 1:
            victims = order_row[:k]
        elif mode == 2:
            victims = order_cnt[:k]
        elif mode == 3:
            ci = int(order_row[0])
            dots = xc @ xc[ci]
            dots[ci] = -2.0
            nbr = np.argsort(-dots)[:k - 1]
            victims = np.append(nbr, ci)
        else:
            pool = order_row[:30]
            # deterministic random choice among top-30
            sel = rng.choice(pool, size=k, replace=False)
            victims = np.array(sel, dtype=int)
        keep = np.ones(n, dtype=bool)
        keep[victims] = False
        X_rem = xc[keep].copy()
        # annealed LSE descent on remainder to gain margin
        X_rem = adam_quench(X_rem, steps=800, T0=0.02, T1=0.0025, lr=0.0025)
        X_cur = X_rem
        ok = True
        for j in range(k):
            if time.time() > t_dead - 2.0:
                ok = False
                break
            q = global_hole(X_cur, rng, n_samp=50000)
            X_cur = np.vstack([X_cur, q])
            # re-optimize all points 300-600 Adam steps after each insertion
            s2 = 300 + (j % 3) * 100  # 300,400,500 cycle
            # slightly hotter early, cooler late
            X_cur = adam_quench(X_cur, steps=s2, T0=0.015, T1=0.003, lr=0.002)
        if not ok:
            break
        # normalize safety
        X_cur /= np.linalg.norm(X_cur, axis=1, keepdims=True)
        v_new = max_cos(X_cur)
        spec_new = spectra_top(X_cur)
        # material adjacency difference: mean abs diff of top spectra
        diff = float(np.mean(np.abs(spec_new - spec_cur)))
        topo_new = diff > 1e-4
        if v_new < best_g - 1e-12:
            best_g = v_new
            x_best = X_cur.copy()
            spec_best = spec_new.copy()
            xc = X_cur.copy()
            cur = v_new
            spec_cur = spec_new.copy()
        elif v_new < 0.29 and (topo_new or v_new < cur):
            # wander to different basin even if worse (allow 0.275-0.29)
            xc = X_cur.copy()
            cur = v_new
            spec_cur = spec_new.copy()
        # else stay in current basin; next trial uses different deletion set
        if time.time() >= t_dead - 1.0:
            break
    # final LP polish of global best, strict last-20s budget
    remain = t_end - time.time() - 1.5
    if remain > 2.0:
        try:
            x_best = lp_polish(x_best, min(remain, 17.0), t_end - 1.0)
        except Exception:
            pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
