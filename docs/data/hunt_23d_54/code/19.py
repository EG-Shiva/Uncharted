import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _anneal(x_init, steps, k0, k1, lr0, lr1, delta, mom=0.7):
    x = x_init.copy()
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    vel = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        m = float(G[iu].max())
        # soft-max weights inside window
        E = np.zeros_like(G)
        # only upper window to save exp
        sel = G > (m - delta)
        np.fill_diagonal(sel, False)
        if np.any(sel):
            arg = k * (G - m)
            arg = np.clip(arg, -20.0, 0.0)
            E[sel] = np.exp(arg[sel])
            # symmetrize (sel symmetric, E symmetric)
        Gr = E @ x
        dot = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - dot * x
        deg = np.sum(E, axis=1, keepdims=True) + 1.0
        Gt /= deg
        vel = mom * vel + Gt
        x = x - lr * vel
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def _farthest_fill(x_base, idx, rng, trials=4000):
    n, d = x_base.shape
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[np.asarray(idx, dtype=int)] = False
    Xk = x_base[keep_mask]
    cand = rng.randn(trials, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    cur = [Xk[i] for i in range(Xk.shape[0])]
    import numpy as _np
    C = cand
    chosen = []
    for _ in range(len(idx)):
        if len(cur) > 0 and len(C) > 0:
            M = _np.stack(cur, axis=0)
            S = C @ M.T
            worst = S.max(axis=1)
            o = int(_np.argmin(worst))
        else:
            o = 0
        chosen.append(C[o])
        cur.append(C[o])
        C = _np.delete(C, o, axis=0)
        if len(C) == 0:
            break
    x2 = x_base.copy()
    for ii, v in zip(idx, chosen):
        x2[ii] = v
    return x2


def _lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = c_all.max()
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 600:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:600]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            rem = seconds - (time.time() - start)
            tl = max(0.2, min(0.9, rem))
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs", options={"time_limit": tl})
            except TypeError:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(20261102)
    n, d = 54, 23
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best = _max_cos(xb)
    x_best = xb.copy()
    iu = np.triu_indices(n, 1)

    # ---- phase 1: fast exploration, collect candidates ----
    cands = []  # (val, x)
    def push(v, x):
        cands.append((float(v), x.copy()))
        cands.sort(key=lambda t: t[0])
        del cands[6:]

    # seed candidate with base
    push(best, x_best)

    it = 0
    t_phase1_end = t0 + 118.0
    # precompute contacts of x_best periodically
    Gb = x_best @ x_best.T
    np.fill_diagonal(Gb, -2.0)
    cnt = np.sum(Gb > best - 1e-9, axis=1)
    order_lo = np.argsort(cnt)
    order_hi = np.argsort(-cnt)
    while time.time() < t_phase1_end:
        it += 1
        if time.time() > t0 + BUDGET - 45.0:
            break
        # refresh contact order every 10 iters
        if it % 10 == 1:
            Gb = x_best @ x_best.T
            np.fill_diagonal(Gb, -2.0)
            cnt = np.sum(Gb > best - 1e-9, axis=1)
            cntw = np.sum(Gb > best - 0.02, axis=1)
            order_lo = np.argsort(cnt + 0.05 * cntw)
            order_hi = np.argsort(-cnt)
        mode = it % 8
        x = x_best.copy()
        if mode == 0:
            x += rng.randn(n, d) * 0.008
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 260, 150, 2200, 0.020, 0.002, 0.06)
        elif mode == 1:
            x += rng.randn(n, d) * 0.03
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 300, 90, 2600, 0.025, 0.002, 0.08)
        elif mode == 2:
            ksel = 4 if (it % 2 == 0) else 2
            idx = order_lo[:ksel].copy()
            x = _farthest_fill(x_best, idx, rng, trials=4000)
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 380, 60, 1600, 0.030, 0.002, 0.12)
        elif mode == 3:
            ksel = 2
            idx = rng.choice(n, size=ksel, replace=False)
            x[idx] = rng.randn(ksel, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 320, 80, 2000, 0.028, 0.002, 0.10)
        elif mode == 4:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            flat = G[iu]
            top = np.argsort(flat)[::-1][:6]
            victims = set()
            for t in top:
                i, j = int(iu[0][t]), int(iu[1][t])
                victims.add(i if rng.rand() < 0.5 else j)
                if len(victims) >= 3:
                    break
            idx = np.array(sorted(victims), dtype=int)
            x[idx] += rng.randn(len(idx), d) * 0.12
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 340, 90, 2000, 0.028, 0.002, 0.10)
        elif mode == 5:
            ksel = 3
            idx = order_hi[:ksel].copy()
            x[idx] += rng.randn(ksel, d) * 0.09
            x += rng.randn(n, d) * 0.005
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 300, 120, 2200, 0.025, 0.002, 0.08)
        elif mode == 6:
            x += rng.randn(n, d) * 0.055
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 260, 150, 2500, 0.020, 0.003, 0.07)
        else:
            ksel = 5
            idx = order_lo[:ksel].copy()
            # strong kick to loose set + mild global
            x[idx] += rng.randn(ksel, d) * 0.10
            x += rng.randn(n, d) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _anneal(x, 360, 70, 1700, 0.030, 0.002, 0.11)
        v = _max_cos(x)
        # keep if within window of best (LP can still win)
        if v < best + 0.015:
            push(v, x)
        # opportunistic raw win without LP
        if v < best - 1e-12 and v < cands[0][0]:
            pass

    # ---- phase 2: selective LP checks on top candidates ----
    # sort candidates, try LP on best few
    cands = sorted(cands, key=lambda t: t[0])[:6]
    deadline_phase2 = t0 + BUDGET - 22.0
    for (v, xc) in list(cands):
        if time.time() > deadline_phase2:
            break
        # skip if already worse than best by margin and not top-2
        rem = deadline_phase2 - time.time()
        if rem < 0.8:
            break
        sec = min(2.5, rem / (len(cands) + 1) + 1.0)
        # give more time to most promising
        if v > best + 0.008 and v > cands[0][0] + 0.004:
            sec = min(sec, 1.0)
        y = _lp_polish(xc, sec, deadline_phase2)
        vy = _max_cos(y)
        if vy < best - 1e-12:
            x_best, best = y, vy
            # local re-search around new best: quick anneal+polish
            for _ in range(2):
                if time.time() > deadline_phase2:
                    break
                z = x_best + rng.randn(n, d) * 0.006
                z /= np.linalg.norm(z, axis=1, keepdims=True)
                z = _anneal(z, 220, 150, 2500, 0.018, 0.002, 0.06)
                vz = _max_cos(z)
                if vz < best + 0.004:
                    rem2 = deadline_phase2 - time.time()
                    if rem2 < 0.7:
                        break
                    zz = _lp_polish(z, min(1.2, rem2), deadline_phase2)
                    vzz = _max_cos(zz)
                    if vzz < best:
                        x_best, best = zz, vzz

    # ---- final polish ----
    rem = max(1.0, min(25.0, (t0 + 178.0) - time.time()))
    try:
        x_best = _lp_polish(x_best, rem)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
