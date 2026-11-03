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

def refine_hole(q, Xrem, steps=150, T0=0.008, T1=0.002, lr=0.08):
    q = q.copy()
    l0 = np.log(T0); l1 = np.log(T1)
    for s in range(1, steps + 1):
        T = float(np.exp(l0 + (l1 - l0) * s / steps))
        d = Xrem @ q
        mm = d.max()
        w = np.exp((d - mm) / T)
        ws = w.sum()
        if ws == 0 or not np.isfinite(ws):
            break
        g = (w[:, None] * Xrem).sum(axis=0) / ws
        g = g - (g @ q) * q
        q = q - lr * g
        nq = np.linalg.norm(q)
        if nq == 0 or not np.isfinite(nq):
            break
        q /= nq
        lr *= 0.995
    return q

def best_holes(X_rem, rng, k, n_samp=20000):
    d = X_rem.shape[1]
    S = rng.standard_normal((n_samp, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    dots = S @ X_rem.T
    worst = dots.max(axis=1)
    out = []
    chosen = np.zeros(n_samp, dtype=bool)
    Xcur = X_rem.copy()
    for _ in range(k):
        # penalize closeness to already chosen holes
        if len(out) > 0:
            H = np.stack(out)
            prox = S @ H.T
            pm = prox.max(axis=1)
            # prefer far from chosen holes
            score = worst + 0.5 * np.maximum(0, pm)
        else:
            score = worst
        # mask already chosen neighbourhood? just pick argmin not chosen
        order = np.argsort(score)
        bi = None
        for o in order:
            if not chosen[o]:
                bi = int(o)
                break
        if bi is None:
            bi = int(order[0])
        chosen[bi] = True
        q = S[bi].copy()
        q = refine_hole(q, Xcur)
        out.append(q)
        # append to Xcur so next hole avoids this one
        Xcur = np.vstack([Xcur, q[None, :]])
        # update worst for remaining candidates cheaply: recompute only for S?
        # approximate by max with new point
        nd = S @ q
        worst = np.maximum(worst, nd) if False else worst  # keep original worst vs X_rem; diversity via prox term
    return out

def solve():
    t0 = time.time()
    T_LIMIT = 168.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(7)
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
                cands.append((max_cos(a), a))
            except Exception:
                pass
    if len(cands) == 0:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a
        best = max_cos(a)
    else:
        cands.sort(key=lambda t: t[0])
        best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 102, 12
    # wandering state
    x_cur = x_best.copy()
    cur_val = best
    last_improve_it = 0
    it = 0
    # short initial snap if time allows (already near optimum, keep tiny)
    # skip heavy initial polish to save time for hopping
    while time.time() < t_end - 22.0 and it < 500:
        it += 1
        stuck = it - last_improve_it
        # adaptive amplitude
        boost = 1.0 + min(2.0, stuck / 12.0)
        G0 = x_cur @ x_cur.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        cnt = (G0 > G0.max() - 0.005).sum(axis=1)
        order = np.argsort(-rowmax - 0.1 * cnt)
        mode = it % 5
        x_try = x_cur.copy()
        if mode == 0:
            # single best-hole reinsertion of worst point
            wi = int(order[0])
            mask = np.ones(n, dtype=bool); mask[wi] = False
            holes = best_holes(x_cur[mask], rng, 1, n_samp=16000)
            x_try[wi] = holes[0] + 0.004 * rng.standard_normal(d) * boost
            jm = np.ones(n, dtype=bool); jm[wi] = False
            x_try[jm] += 0.003 * boost * rng.standard_normal((n - 1, d))
            steps, T0_, T1_, lr = 700, 0.020, 0.0030, 0.0022
        elif mode == 1:
            # double-hole reinsertion
            nk = 2 if (it % 2 == 0) else 3
            worst = order[:nk]
            mask = np.ones(n, dtype=bool); mask[worst] = False
            holes = best_holes(x_cur[mask], rng, nk, n_samp=16000)
            for wi, h in zip(worst, holes):
                x_try[wi] = h + 0.004 * rng.standard_normal(d) * boost
            x_try[mask] += 0.003 * boost * rng.standard_normal((n - nk, d))
            steps, T0_, T1_, lr = 800, 0.022, 0.0030, 0.0022
        elif mode == 2:
            # kick overloaded hubs heavily
            k = 3 if stuck < 15 else 5
            idx = order[:k]
            x_try[idx] += (0.07 * boost) * rng.standard_normal((k, d))
            x_try += 0.004 * boost * rng.standard_normal((n, d))
            steps, T0_, T1_, lr = 800, 0.025, 0.0035, 0.0022
        elif mode == 3:
            # cluster shake around random seed + neighbours
            ci = int(rng.integers(n))
            gr = (x_cur @ x_cur[ci]); gr[ci] = -2.0
            nbr = np.argsort(-gr)[:5]
            cl = np.append(nbr, ci)
            x_try[cl] += (0.05 * boost) * rng.standard_normal((len(cl), d))
            x_try += 0.003 * boost * rng.standard_normal((n, d))
            steps, T0_, T1_, lr = 700, 0.020, 0.0030, 0.0020
        else:
            # global jitter
            x_try = x_cur + (0.012 * boost) * rng.standard_normal((n, d))
            steps, T0_, T1_, lr = 600, 0.018, 0.0040, 0.0020
        x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        x_q = adam_quench(x_try, steps, T0_, T1_, lr)
        vq = max_cos(x_q)
        # decide to snap
        need_snap = False
        snap_budget = 0.0
        if vq < best - 1e-12:
            need_snap = True
            snap_budget = min(3.0, max(0.9, (t_end - 22.0 - time.time()) / 6.0))
        elif vq < cur_val + 0.001 and (vq < best + 0.002) and (it % 4 == 0):
            need_snap = True
            snap_budget = min(1.6, max(0.7, (t_end - 22.0 - time.time()) / 12.0))
        if need_snap and snap_budget > 0.6 and time.time() + snap_budget < t_end - 20.0:
            x_q = lp_polish(x_q, snap_budget, t_end - 20.0)
            vq = max_cos(x_q)
        if vq < best - 1e-12:
            x_best, best = x_q.copy(), vq
            x_cur, cur_val = x_q.copy(), vq
            last_improve_it = it
        elif vq < cur_val + 0.0015:
            # wandering acceptance: allow sideways move to new basin
            # prefer moves that lower cur
            if vq < cur_val or rng.random() < 0.35:
                x_cur, cur_val = x_q.copy(), vq
        # occasional restart from best if wandering too far
        if stuck > 25 and rng.random() < 0.4:
            x_cur, cur_val = x_best.copy(), best
        if time.time() >= t_end - 22.0:
            break
    remain = t_end - time.time() - 1.5
    if remain > 1.5:
        x_best = lp_polish(x_best, min(remain, 18.0), t_end - 0.8)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
