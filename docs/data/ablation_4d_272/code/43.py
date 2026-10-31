import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())

def _lp_polish(x, seconds, r0=2.5e-3, max_pairs=4500):
    t_start = time.time()
    x = _norm_rows(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = float(r0)
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while (time.time() - t_start) < seconds and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel_idx = np.where(c_all > thresh)[0]
        if sel_idx.size == 0:
            radius *= 0.5
            continue
        if sel_idx.size > max_pairs:
            # keep largest
            part = np.argpartition(-c_all[sel_idx], max_pairs - 1)[:max_pairs]
            sel_idx = sel_idx[part]
        I = iu0[0][sel_idx]
        J = iu0[1][sel_idx]
        c_sel = c_all[sel_idx]
        p = sel_idx.size
        nv = n * d + 1
        cols_b = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_b[:, :d] = I[:, None] * d + ar[None, :]
        cols_b[:, d:2 * d] = J[:, None] * d + ar[None, :]
        cols_b[:, 2 * d] = n * d
        cols = cols_b.ravel()
        vals_b = np.empty((p, 2 * d + 1), dtype=np.float64)
        vals_b[:, :d] = x[J]
        vals_b[:, d:2 * d] = x[I]
        vals_b[:, 2 * d] = -1.0
        vals = vals_b.ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best - 1e-15:
            x = y
            best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 5e-13:
                break
        if (time.time() - t_start) >= seconds:
            break
    return x, best

def _anneal(x, steps, a0, a1, s0, s1, mu=0.82, deadline=None):
    n, d = x.shape
    x = _norm_rows(x.copy())
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    best_X = x.copy()
    for it in range(steps):
        if deadline is not None and it % 50 == 0 and time.time() > deadline:
            break
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < bv:
            bv = m
            best_X = x.copy()
        # softmax weights, clipped
        E = np.clip(alpha * (G - m), -50.0, 0.0)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x = _norm_rows(x)
    Gf = x @ x.T
    np.fill_diagonal(Gf, -2.0)
    fv = float(Gf.max())
    if fv < bv:
        return x, fv
    return best_X, bv

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(20261031)
    n, d = 272, 4

    cand = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = _norm_rows(a)
                if np.all(np.isfinite(a)):
                    cand.append(a)
        except Exception:
            pass
    if not cand:
        cand.append(_norm_rows(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cand]
    bi = int(np.argmin(vals))
    best = cand[bi].copy()
    best_val = float(vals[bi])

    # initial polish of incumbent (short, to lock baseline)
    remain = HARD_END - time.time()
    if remain > 20:
        best_p, bv_p = _lp_polish(best, seconds=min(10.0, remain - 12.0))
        if bv_p < best_val:
            best, best_val = best_p, bv_p
        else:
            best = best_p

    # basin-hopping loop
    sigmas = [0.025, 0.012, 0.006, 0.035, 0.015, 0.008, 0.05]
    cfg_a = [(30.0, 400.0), (60.0, 600.0), (25.0, 250.0)]
    it = 0
    while time.time() < HARD_END - 6.0:
        sig = sigmas[it % len(sigmas)]
        a0, a1 = cfg_a[(it // len(sigmas)) % len(cfg_a)]
        # build start: perturb best, with extra kick to high-contact points sometimes
        start = best + rng.standard_normal((n, d)) * sig
        # every 4th iter: strong kick to a random subset (topology change)
        if it % 4 == 3:
            k = 18 + int(rng.integers(0, 14))
            idx = rng.choice(n, size=k, replace=False)
            start[idx] += rng.standard_normal((k, d)) * (sig * 3.0 + 0.03)
        # every 7th iter: relocate rattlers (lowest row-max) to random spots
        if it % 7 == 6:
            try:
                rm, _ = _row_max(best)
                order = np.argsort(rm)
                ratt = order[:6]
                start[ratt] = rng.standard_normal((len(ratt), d))
            except Exception:
                pass
        start = _norm_rows(start)
        # anneal budget adapts to remaining time
        remain = HARD_END - time.time()
        if remain < 8:
            break
        steps = 550 if remain > 60 else 320
        s0 = 0.05 if sig > 0.02 else 0.03
        s1 = 0.004
        anneal_deadline = HARD_END - 5.0
        mid, mid_val = _anneal(start, steps=steps, a0=a0, a1=a1, s0=s0, s1=s1, mu=0.83, deadline=anneal_deadline)
        # short LP polish
        remain2 = HARD_END - time.time()
        if remain2 < 5:
            if mid_val < best_val:
                best, best_val = mid, mid_val
            break
        pol_sec = min(5.0, remain2 - 1.0)
        if pol_sec < 1.0:
            pol_sec = remain2 - 0.5
        pol, pol_val = _lp_polish(mid, seconds=float(max(1.0, pol_sec)))
        if pol_val < best_val - 1e-15:
            best, best_val = pol, pol_val
        elif mid_val < best_val - 1e-15:
            # polish failed but anneal improved: polish again briefly with smaller radius
            pol2, pol2_val = _lp_polish(mid, seconds=1.5, r0=8e-4)
            if pol2_val < best_val:
                best, best_val = pol2, pol2_val
            else:
                best, best_val = mid, mid_val
        it += 1
        if it > 60:
            break

    # final safety polish if time allows
    if HARD_END - time.time() > 3.5:
        try:
            pf, pv = _lp_polish(best, seconds=min(2.5, HARD_END - time.time() - 0.5), r0=8e-4)
            if pv <= best_val:
                best, best_val = pf, pv
        except Exception:
            pass
    best = _norm_rows(best)
    # strict finite check
    best[~np.isfinite(best)] = 0.25
    best = _norm_rows(best)
    return best
