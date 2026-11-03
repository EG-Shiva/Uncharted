import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

_TRIU_I = None
_TRIU_J = None

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, topk=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        iu = iu0
        c_all = G[iu]
        m = c_all.max()
        thr = m - max(15.0 * radius, 5e-4)
        sel = c_all > thr
        # cap to topk
        if np.count_nonzero(sel) > topk:
            # keep largest topk
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(vals, len(vals) - topk)[-topk:]
            nsel = np.zeros_like(sel)
            nsel[idx[part]] = True
            sel = nsel
        I = iu[0][sel]
        J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = (I[:, None] * d + np.arange(d)).ravel()
        cj = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2 * d + 1)] = None  # placeholder, will overwrite vectorized below
        # vectorized build
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": max(0.2, seconds - (time.time() - start))})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best - 1e-12:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            # accept tiny improvement due to rounding
            if v < best:
                x, best = y, v
                radius *= 0.7
            else:
                radius *= 0.5
        if time.time() - start >= seconds:
            break
    return x

def smooth_adam(x_init, steps, T0, T1, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    # precompute ratio
    lratio = (T1 / max(T0, 1e-12)) if T0 > 0 else 1.0
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (lratio ** frac)
        if T < 1e-12:
            T = 1e-12
        lrs = lr * (1.0 - 0.85 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mval = G.max()
        E = np.exp((G - mval) / T)
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if S == 0 or not np.isfinite(S):
            break
        P = E / S
        Grad = P @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = beta1 * m + (1 - beta1) * Gt
        v = beta2 * v + (1 - beta2) * (Gt * Gt)
        bc1 = 1.0 - beta1 ** (s + 1)
        bc2 = 1.0 - beta2 ** (s + 1)
        mhat = m / bc1
        vhat = v / bc2
        step = lrs * mhat / (np.sqrt(vhat) + eps)
        sn = np.sum(step * x, axis=1, keepdims=True)
        step = step - sn * x
        x = x - step
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        x = x / nrm
        if not np.all(np.isfinite(x)):
            break
    return x

def refine_hole(q, Xcur, iters=180, T=0.005, lr=0.05):
    for _ in range(iters):
        dots = Xcur @ q
        mm = dots.max()
        w = np.exp((dots - mm) / T)
        s = w.sum()
        if s == 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ Xcur
        g = g - (g @ q) * q
        q = q - lr * g
        nq = np.linalg.norm(q)
        if nq == 0 or not np.isfinite(nq):
            break
        q = q / nq
    return q

def solve():
    t0 = time.time()
    BUDGET = 175.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.normal(size=(102, 12))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 102, 12
    if xb.shape != (102, 12):
        tmp = rng.normal(size=(102, 12))
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        m0 = min(xb.shape[0], 102)
        d0 = min(xb.shape[1], 12)
        tmp[:m0, :d0] = xb[:m0, :d0]
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    best = max_cos(xb)
    x_best = xb.copy()
    HOP_DEADLINE = t0 + 128.0
    trial = 0
    # candidate pool for holes (refresh occasionally)
    while True:
        now = time.time()
        if now >= HOP_DEADLINE or (now - t0) > BUDGET - 45:
            break
        trial += 1
        mode = trial % 5
        x_try = x_best.copy()
        if mode == 2:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            mval = float(G.max())
            cnt = (G > mval - 0.005).sum(axis=1)
            order = np.argsort(-cnt)
            nk = int(rng.integers(1, 4))
            worst = order[:nk].copy()
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            Xcur = x_best[mask]
            cand = rng.normal(size=(9000, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            for wi in worst:
                sc = (cand @ Xcur.T).max(axis=1)
                bi = int(np.argmin(sc))
                q = cand[bi].copy()
                q = refine_hole(q, Xcur, iters=220, T=0.005, lr=0.06)
                Xcur = np.vstack([Xcur, q[None, :]])
                x_try[wi] = q
            x_try = x_try + rng.normal(size=x_try.shape) * 0.004
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif mode in (0,):
            sigma = float(rng.uniform(0.012, 0.05))
            x_try = x_try + rng.normal(size=x_try.shape) * sigma
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif mode == 1:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            mval = float(G.max())
            cnt = (G > mval - 0.004).sum(axis=1)
            order = np.argsort(-cnt)
            k = int(rng.integers(5, 22))
            targets = order[:k]
            s_small = float(rng.uniform(0.003, 0.010))
            s_big = float(rng.uniform(0.05, 0.17))
            noise = rng.normal(size=x_try.shape) * s_small
            noise[targets] += rng.normal(size=(k, d)) * s_big
            x_try = x_try + noise
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif mode == 3:
            # rattler / loose point relocation to largest hole
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            mx = G.max(axis=1)  # worst neighbor per point
            # loose = smallest mx (most isolated) -> keep, actually move worst? move random loose among bottom half
            order_loose = np.argsort(mx)  # ascending: most isolated first
            nk = int(rng.integers(1, 3))
            # pick isolated points to keep, move overloaded? we move overloaded to hole
            # choose to move points with largest mx
            order_over = np.argsort(-mx)
            mov = order_over[:nk].copy()
            mask = np.ones(n, dtype=bool)
            mask[mov] = False
            Xfix = x_best[mask]
            cand = rng.normal(size=(9000, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            Xcur = Xfix
            for wi in mov:
                sc = (cand @ Xcur.T).max(axis=1)
                bi = int(np.argmin(sc))
                q = cand[bi].copy()
                q = refine_hole(q, Xcur, iters=200, T=0.004, lr=0.05)
                Xcur = np.vstack([Xcur, q[None, :]])
                x_try[wi] = q
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            # pair split: push current max pair apart
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            ij = int(np.argmax(G))
            a = ij // n
            b = ij % n
            # directions separating them
            xa = x_best[a].copy()
            xb_ = x_best[b].copy()
            # tangent components
            da = xa - xb_
            da = da - (da @ xa) * xa
            db = xb_ - xa
            db = db - (db @ xb_) * xb_
            na = np.linalg.norm(da)
            nb = np.linalg.norm(db)
            if na > 1e-12:
                da /= na
            if nb > 1e-12:
                db /= nb
            amp = float(rng.uniform(0.01, 0.035))
            x_try[a] = x_try[a] + da * amp + rng.normal(size=d) * 0.006
            x_try[b] = x_try[b] + db * amp + rng.normal(size=d) * 0.006
            # also jitter neighbors of a,b
            neigh_a = np.where(G[a] > float(G.max()) - 0.004)[0]
            neigh_b = np.where(G[b] > float(G.max()) - 0.004)[0]
            for nb_idx in list(neigh_a)[:6]:
                x_try[nb_idx] += rng.normal(size=d) * 0.008
            for nb_idx in list(neigh_b)[:6]:
                x_try[nb_idx] += rng.normal(size=d) * 0.008
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)

        steps = int(rng.integers(500, 1500))
        T0p = float(rng.uniform(0.007, 0.024))
        T1p = float(rng.uniform(0.0007, 0.003))
        lrp = float(rng.uniform(0.003, 0.0085))
        # time guard
        remain_hop = HOP_DEADLINE - time.time()
        if remain_hop <= 1.0:
            break
        # rough cost 0.00035 per step
        max_steps = int(remain_hop / 0.00045) - 20
        if max_steps < 200:
            break
        if steps > max_steps:
            steps = max_steps
        x_pol = smooth_adam(x_try, steps, T0p, T1p, lrp)
        val = max_cos(x_pol)
        # snap if promising (allow small slack since LP can gain ~1e-4)
        if val < best + 0.0012:
            snap_budget = 1.6
            if val < best - 2e-9:
                snap_budget = 2.5
            # don't overspend near deadline
            if HOP_DEADLINE - time.time() < snap_budget + 0.5:
                snap_budget = max(0.6, HOP_DEADLINE - time.time() - 0.3)
            if snap_budget >= 0.5:
                try:
                    x_sn = lp_polish(x_pol, snap_budget)
                    v_sn = max_cos(x_sn)
                except Exception:
                    v_sn = val
                    x_sn = x_pol
                if v_sn < val:
                    x_pol, val = x_sn, v_sn
        if val < best - 1e-12:
            best = val
            x_best = x_pol.copy()
        # occasional time check
        if time.time() - t0 > BUDGET - 42:
            break

    # final tightening
    try:
        x_best = smooth_adam(x_best, 700, 0.005, 0.0007, 0.004)
    except Exception:
        pass
    rem = BUDGET - (time.time() - t0) - 1.0
    if rem > 2.0:
        try:
            x_best = lp_polish(x_best, min(rem, 34.0))
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = np.nan_to_num(x_best, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(x_best, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x_best = x_best / nrm
    return x_best
