import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    return rm, float(rm.max())

def _contacts(x, thr):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.count_nonzero(G >= thr, axis=1)

def _lp_polish(x, deadline, r0=2.0e-3, max_pairs=4500):
    x = _norm_rows(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = float(r0)
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel_idx = np.where(c_all > thresh)[0]
        if sel_idx.size == 0:
            radius *= 0.5
            continue
        if sel_idx.size > max_pairs:
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
        vals_b = np.empty((p, 2 * d + 1), dtype=np.float64)
        vals_b[:, :d] = x[J]
        vals_b[:, d:2 * d] = x[I]
        vals_b[:, 2 * d] = -1.0
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals_b.ravel(), (rows, cols_b.ravel())), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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
            x = y; best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 5e-13:
                break
    return x, best

def _anneal(x, steps, a0, a1, s0, s1, mu=0.83, deadline=None, track_best=True):
    n, d = x.shape
    x = _norm_rows(x.copy())
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    best_X = x.copy()
    for it in range(steps):
        if deadline is not None and (it % 40 == 0) and time.time() > deadline:
            break
        frac = it / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if track_best and m < bv:
            bv = m
            best_X = x.copy()
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
    if track_best:
        Gf = x @ x.T
        np.fill_diagonal(Gf, -2.0)
        fv = float(Gf.max())
        if fv < bv:
            return x, fv
        return best_X, bv
    Gf = x @ x.T
    np.fill_diagonal(Gf, -2.0)
    return x, float(Gf.max())

def _largest_holes(Y, rng, k=3, n_samples=10000):
    d = Y.shape[1]
    C = rng.standard_normal((n_samples, d))
    C = _norm_rows(C)
    # batched max-dot
    best_idx = []
    bs = 2500
    sms = np.empty(n_samples, dtype=np.float64)
    argm = np.empty(n_samples, dtype=np.int64)
    for a in range(0, n_samples, bs):
        Cb = C[a:a+bs]
        S = Cb @ Y.T
        sms[a:a+Cb.shape[0]] = S.max(axis=1)
        argm[a:a+Cb.shape[0]] = S.argmax(axis=1)
    order = np.argsort(sms)
    picks = []
    used = np.zeros(Y.shape[0], dtype=bool)
    for oi in order:
        # diversify: prefer holes near different neighbors
        nn = int(argm[oi])
        if len(picks) < k:
            # allow reuse but penalize identical neighbor repeats beyond 1
            picks.append(int(oi))
            if len(picks) >= k * 2:
                break
        if len(picks) >= k * 2:
            break
    # take spread subset: greedy far apart
    sel = []
    for pi in picks:
        c = C[pi]
        ok = True
        for q in sel:
            if float(c @ C[q]) > 0.9:
                ok = False
                break
        if ok:
            sel.append(pi)
        if len(sel) >= k:
            break
    if len(sel) < k:
        sel = picks[:k]
    out = []
    for pi in sel:
        c = C[pi].copy()
        # refine: minimize max dot = move away from nearest neighbors
        step = 0.05
        vel = np.zeros(4)
        for _ in range(90):
            dots = Y @ c
            m = float(dots.max())
            seln = Y[dots > m - 3e-3]
            if seln.shape[0] == 0:
                seln = Y[np.argmax(dots):np.argmax(dots)+1]
            g = seln.sum(axis=0)
            gt = g - float(g @ c) * c
            nn_ = float(np.linalg.norm(gt))
            if nn_ < 1e-14:
                break
            gt /= nn_
            vel = 0.55 * vel + gt
            vn = float(np.linalg.norm(vel))
            if vn > 1e-14:
                c = c - (step / vn) * vel
                c = c / np.linalg.norm(c)
                step *= 0.992
        out.append(c / np.linalg.norm(c))
    while len(out) < k:
        r = rng.standard_normal(4); r /= np.linalg.norm(r)
        out.append(r)
    return np.array(out)

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(977277)
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
    # quick lock baseline
    if HARD_END - time.time() > 25:
        bp, bv = _lp_polish(best, deadline=time.time() + 6.0, r0=1.5e-3)
        if bv < best_val:
            best, best_val = bp, bv
        else:
            best = bp

    sigmas = [0.022, 0.011, 0.006, 0.032, 0.014, 0.045, 0.008, 0.018]
    alpha_cfgs = [(28.0, 420.0), (60.0, 650.0), (18.0, 220.0), (45.0, 900.0)]
    it = 0
    while time.time() < HARD_END - 7.0:
        if it > 80:
            break
        remain = HARD_END - time.time()
        if remain < 9:
            break
        sig = sigmas[it % len(sigmas)]
        a0, a1 = alpha_cfgs[(it // len(sigmas)) % len(alpha_cfgs)]
        mode = it % 6
        start = None
        try:
            rm, mv = _row_max(best)
        except Exception:
            rm = None
        if mode == 0:
            start = best + rng.standard_normal((n, d)) * sig
        elif mode == 1:
            # subset strong kick on high-contact points
            start = best + rng.standard_normal((n, d)) * sig
            if rm is not None:
                order = np.argsort(-rm)
                kk = 14 + int(rng.integers(0, 12))
                idx = order[:kk]
                # push them apart along tangent noise
                start[idx] += rng.standard_normal((kk, d)) * (sig * 4.0 + 0.04)
            else:
                idx = rng.choice(n, size=20, replace=False)
                start[idx] += rng.standard_normal((len(idx), d)) * 0.08
        elif mode == 2:
            # hole transplant: move worst rattlers to largest holes of the rest
            order = np.argsort(rm)  # ascending: rattlers first
            nr = 3 + int(rng.integers(0, 2))  # 3-4
            ratt = order[:nr]
            keep_mask = np.ones(n, dtype=bool); keep_mask[ratt] = False
            Y = best[keep_mask]
            holes = _largest_holes(Y, rng, k=nr, n_samples=9000)
            start = best.copy() + rng.standard_normal((n, d)) * (sig * 0.5)
            start[ratt] = holes + rng.standard_normal((nr, d)) * 0.004
        elif mode == 3:
            # worst-pair split surgery
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            # find top ~8 pairs
            iu = np.triu_indices(n, 1)
            c_all = G[iu]
            top = np.argpartition(-c_all, 8)[:8]
            involved = set()
            for t in top:
                involved.add(int(iu[0][t])); involved.add(int(iu[1][t]))
            involved = np.array(sorted(involved), dtype=int)
            start = best + rng.standard_normal((n, d)) * sig
            # separate each top pair along their difference direction
            for t in top:
                i = int(iu[0][t]); j = int(iu[1][t])
                diff = best[i] - best[j]
                diff = diff - 0.5 * float(diff @ (best[i] + best[j])) * (best[i] + best[j])
                nn_ = float(np.linalg.norm(diff))
                if nn_ < 1e-12:
                    diff = rng.standard_normal(4); nn_ = float(np.linalg.norm(diff))
                diff /= nn_
                amp = 0.05 + rng.random() * 0.05
                start[i] += diff * amp
                start[j] -= diff * amp
            # extra jitter on involved
            start[involved] += rng.standard_normal((len(involved), d)) * (sig * 0.7)
        elif mode == 4:
            # random block regrow (topology change)
            start = best + rng.standard_normal((n, d)) * sig
            k = 8 + int(rng.integers(0, 8))
            idx = rng.choice(n, size=k, replace=False)
            start[idx] = rng.standard_normal((k, d))
        else:
            # global shake + mild
            start = best + rng.standard_normal((n, d)) * (sig + 0.008)
        start = _norm_rows(start)
        steps = 520 if remain > 55 else 340
        s0 = 0.055 if sig > 0.02 else 0.032
        s1 = 0.004
        mid, mid_val = _anneal(start, steps=steps, a0=a0, a1=a1, s0=s0, s1=s1,
                               mu=0.83, deadline=HARD_END - 5.0)
        remain2 = HARD_END - time.time()
        if remain2 < 5.5:
            if mid_val < best_val:
                best, best_val = mid, mid_val
            break
        pol_sec = min(4.5, remain2 - 1.2)
        pol, pol_val = _lp_polish(mid, deadline=time.time() + max(1.0, pol_sec), r0=2.0e-3)
        if pol_val < best_val - 1e-15:
            best, best_val = pol, pol_val
        elif mid_val < best_val - 1e-15:
            # anneal improved but LP slipped: retry tighter radius briefly
            pol2, pol2_val = _lp_polish(mid, deadline=time.time() + 1.2, r0=7e-4, max_pairs=5000)
            if pol2_val < best_val:
                best, best_val = pol2, pol2_val
            else:
                # keep annealed point as new base if clearly better to allow drift
                if mid_val < best_val - 2e-07:
                    best, best_val = mid, mid_val
        it += 1

    if HARD_END - time.time() > 3.0:
        try:
            pf, pv = _lp_polish(best, deadline=min(HARD_END - 0.4, time.time() + 2.5), r0=7e-4)
            if pv <= best_val:
                best, best_val = pf, pv
        except Exception:
            pass
    best = _norm_rows(best)
    best[~np.isfinite(best)] = 0.25
    best = _norm_rows(best)
    return best
