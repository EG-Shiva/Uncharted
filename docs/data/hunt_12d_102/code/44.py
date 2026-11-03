import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def adam_quench(x_init, steps, T0, T1, lr, rng=None):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0 = np.log(T0)
    l1 = np.log(T1)
    for s in range(1, steps + 1):
        T = float(np.exp(l0 + (l1 - l0) * (s / steps)))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = G.max()
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        # normalize to keep scale stable
        ssum = W.sum()
        if ssum <= 0 or not np.isfinite(ssum):
            break
        W /= ssum
        Grad = W @ x
        # Riemannian projection
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        upd = (m / c1) / (np.sqrt(v / c2) + eps)
        # remove normal component introduced by adaptivity
        upd = upd - np.sum(upd * x, axis=1, keepdims=True) * x
        x = x - lr * upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if not np.all(np.isfinite(x)):
            return x_init
    return x


def find_hole(X_rem, rng, n_samp=8000):
    d = X_rem.shape[1]
    S = rng.standard_normal((n_samp, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    dots = S @ X_rem.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    return S[bi].copy(), float(worst[bi])


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
        thresh = mcur - max(20.0 * radius, 1e-7)
        sel = c > thresh
        # cap number of constraints for speed
        if int(sel.sum()) > 3000:
            # keep tightest 3000
            idx = np.argpartition(-c, 3000)[:3000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
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
        if vv < best - 1e-15:
            x, best, radius = y, vv, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    # reserve final polish
    HOP_END = t0 + 145.0
    rng = np.random.default_rng(7)
    try:
        xb = np.array(np.load("best.npy"), dtype=float)
    except Exception:
        xb = None
    try:
        xr = np.array(np.load("record.npy"), dtype=float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (102, 12):
            try:
                a = c / np.linalg.norm(c, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
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
    x_cur = x_best.copy()
    cur_val = best
    Temp = 8e-5

    it = 0
    while time.time() < HOP_END and time.time() < T_END - 28.0:
        it += 1
        mode = it % 5
        # diagnostics on current state
        G0 = x_cur @ x_cur.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        rowmax = G0.max(axis=1)
        cnt = (G0 > m0 - 2e-3).sum(axis=1)
        order_cnt = np.argsort(-cnt)
        order_row = np.argsort(-rowmax)
        x_try = x_cur.copy()

        if mode == 0:
            # single worst-point hole re-insertion
            wi = int(order_row[int(rng.integers(0, 3))])
            mask = np.ones(n, dtype=bool)
            mask[wi] = False
            X_rem = x_try[mask]
            q, _ = find_hole(X_rem, rng, n_samp=12000)
            # refine hole a bit: repulsion steps on q alone
            for _ in range(60):
                dots = X_rem @ q
                mm = dots.max()
                w = np.exp((dots - mm) / 0.004)
                ws = w.sum()
                if ws == 0 or not np.isfinite(ws):
                    break
                grad = (w @ X_rem) / ws
                grad = grad - np.dot(grad, q) * q
                q = q - 0.05 * grad
                nq = np.linalg.norm(q)
                if nq == 0:
                    break
                q /= nq
            x_try[wi] = q
            x_try += 0.002 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = adam_quench(x_try, steps=380, T0=0.012, T1=0.0025, lr=0.0035)
        elif mode == 1:
            # double hole re-insertion of overloaded hubs
            nk = 2 if (it % 2 == 0) else 3
            worst = order_cnt[:nk].copy()
            # sometimes use rowmax order
            if rng.random() < 0.4:
                worst = order_row[:nk].copy()
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            X_rem = x_try[mask]
            # sequential best holes
            X_cur_rem = X_rem.copy()
            new_pts = []
            for _ in range(nk):
                q, _ = find_hole(X_cur_rem, rng, n_samp=8000)
                new_pts.append(q)
                X_cur_rem = np.vstack([X_cur_rem, q.reshape(1, -1)])
            for k, wi in enumerate(worst):
                x_try[int(wi)] = new_pts[k]
            x_try += 0.002 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = adam_quench(x_try, steps=420, T0=0.014, T1=0.0022, lr=0.0035)
        elif mode == 2:
            # coordinated cluster translation: hub + nearest neighbours move together
            hub = int(order_cnt[int(rng.integers(0, 6))])
            gh = (x_cur @ x_cur[hub])
            gh[hub] = -2.0
            nbr = np.argsort(-gh)[:6]
            cl = np.append(nbr, hub)
            # random tangent direction at hub
            u = rng.standard_normal(d)
            u = u - np.dot(u, x_cur[hub]) * x_cur[hub]
            nu = np.linalg.norm(u)
            if nu > 0:
                u /= nu
            step = 0.04 + 0.03 * rng.random()
            for ci in cl:
                # project u to tangent at ci
                uu = u - np.dot(u, x_cur[int(ci)]) * x_cur[int(ci)]
                nun = np.linalg.norm(uu)
                if nun > 1e-12:
                    uu /= nun
                    x_try[int(ci)] = x_cur[int(ci)] + step * uu + 0.008 * rng.standard_normal(d)
                else:
                    x_try[int(ci)] = x_cur[int(ci)] + 0.02 * rng.standard_normal(d)
            # small global noise
            x_try += 0.0015 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = adam_quench(x_try, steps=420, T0=0.010, T1=0.002, lr=0.003)
        elif mode == 3:
            # cavity enlargement: find hole centre, push its neighbours away
            Sall = rng.standard_normal((6000, d))
            Sall /= np.linalg.norm(Sall, axis=1, keepdims=True)
            worst_all = (Sall @ x_cur.T).max(axis=1)
            hole = Sall[int(np.argmin(worst_all))]
            dots = x_cur @ hole
            # nearest 8 to hole get pushed away along tangent
            near = np.argsort(-dots)[:8]
            for ci in near:
                # direction away from hole projected
                gdir = x_cur[int(ci)] - dots[int(ci)] * hole
                # actually push away: move point away from hole means increase angle -> move opposite to hole projection
                v = x_cur[int(ci)] - 0.06 * hole
                # add noise
                v += 0.01 * rng.standard_normal(d)
                x_try[int(ci)] = v
            x_try += 0.001 * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = adam_quench(x_try, steps=380, T0=0.012, T1=0.0025, lr=0.0032)
        else:
            # global jitter + quench with hotter start
            amp = 0.008 + 0.006 * rng.random()
            x_try = x_cur + amp * rng.standard_normal((n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            x_q = adam_quench(x_try, steps=350, T0=0.016, T1=0.003, lr=0.004)

        try:
            vq = max_cos(x_q)
        except Exception:
            continue
        if not np.isfinite(vq):
            continue
        if vq < best - 1e-15:
            best = vq
            x_best = x_q.copy()
            x_cur = x_q.copy()
            cur_val = vq
            Temp *= 0.98
        elif vq < cur_val:
            x_cur = x_q.copy()
            cur_val = vq
        else:
            # annealed sideways move to change basin
            # decay Temp slowly
            Temp *= 0.9995
            Temp = float(np.clip(Temp, 1e-6, 2e-4))
            if rng.random() < np.exp(-(vq - cur_val) / max(Temp, 1e-12)):
                x_cur = x_q.copy()
                cur_val = vq
            # periodic reset if wandering too far
            if cur_val > best + 0.004 and (it % 20 == 0):
                x_cur = x_best.copy()
                cur_val = best
        # hard time check
        if time.time() > HOP_END:
            break

    # final snap: polish best (+ maybe cur if better than best after quench w/o LP)
    # choose starting point for LP as best
    try:
        x_fin = lp_polish(x_best, seconds=24.0, t_end=T_END - 2.0)
        vf = max_cos(x_fin)
        if vf < best and np.all(np.isfinite(x_fin)):
            x_best = x_fin
            best = vf
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
