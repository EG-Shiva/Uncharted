import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline - 0.7:
            break
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        m = float(c.max())
        sel_thr = max(20 * radius, 1e-4)
        sel = c > m - sel_thr
        I0, J0 = iu[0][sel], iu[1][sel]
        p0 = len(I0)
        if p0 == 0:
            break
        # cap constraints for speed
        if p0 > 2500:
            # keep largest 2500
            vals_sel = c[sel]
            kth = np.argpartition(vals_sel, -2500)[-2500:]
            I = I0[kth]; J = J0[kth]
            cs = vals_sel[kth]
            p = 2500
        else:
            I, J = I0, J0
            cs = c[sel]
            p = p0
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _refine_single(y, Xf, steps=350, lr=0.05, T0=0.015, T1=0.0012):
    y = y / (np.linalg.norm(y) + 1e-12)
    m_ad = np.zeros_like(y)
    v_ad = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 5e-5:
            T = 5e-5
        dots = Xf @ y
        mv = float(dots.max())
        E = np.exp((dots - mv) / T)
        S = float(E.sum())
        if S <= 0 or not np.isfinite(S):
            break
        grad = (E @ Xf) / S
        gt = grad - float(grad @ y) * y
        b1p *= b1; b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * gt
        v_ad = b2 * v_ad + (1.0 - b2) * (gt * gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        # cap step norm
        sn = float(np.linalg.norm(step)) + 1e-12
        if sn > 1.0:
            step = step / sn
        y = y - lr * step
        nrm = float(np.linalg.norm(y))
        if nrm == 0 or not np.isfinite(nrm):
            break
        y = y / nrm
    return y

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(7)
    n, d = 154, 10
    best0 = None
    best0_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best0_val:
                    best0_val = v
                    best0 = a.copy()
        except Exception:
            pass
    if best0 is None:
        best0 = _norm(rng.normal(size=(n, d)))
        best0_val = _max_cos(best0)
    # --- deflation set selection ---
    Xfull = best0.copy()
    try:
        G = Xfull @ Xfull.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        rowmax = G.max(axis=1)
        deg = (G > mcur - 0.005).sum(axis=1)
        order_low = np.argsort(rowmax)  # rattlers first
        order_high = np.argsort(-deg)   # hubs first
        NDEL = 34  # -> 120 remain (within 114-124)
        NRATT = 11
        ratt = list(order_low[:NRATT])
        s = set(ratt)
        hubs = []
        for idx in order_high:
            if idx not in s:
                hubs.append(int(idx))
            if len(hubs) >= (NDEL - NRATT):
                break
        victims = ratt + hubs
        keep = np.array([i for i in range(n) if i not in set(victims)], dtype=int)
        Xcur = Xfull[keep].copy()
    except Exception:
        Xcur = Xfull[:120].copy()
        victims = []
    # --- polish reduced set substantially lower ---
    # budget 7s or less if time low
    try:
        t_now = time.time()
        red_budget = min(7.0, max(2.0, (DEAD - t_now - 90.0)))
        if red_budget > 1.0:
            Xcur = _lp_polish(Xcur, red_budget, deadline=DEAD)
    except Exception:
        pass
    try:
        prev_max = _max_cos(Xcur)
    except Exception:
        prev_max = best0_val
    # --- regrowth one at a time to exactly 154 ---
    # short polish 1.2-1.6s per insertion
    need = n - Xcur.shape[0]
    # safety: if need<=0, already done
    for k in range(need):
        if time.time() > DEAD - 22.0:
            break  # leave time for greedy fill + final polish
        m_now = Xcur.shape[0]
        # sample best of >=20000
        C = 20000
        try:
            cand = rng.normal(size=(C, d))
            cand = _norm(cand)
            Dots = cand @ Xcur.T
            md = Dots.max(axis=1)
            # get few best holes
            K = 5
            idx = np.argpartition(md, K)[:K]
            # sort them best first
            srt = idx[np.argsort(md[idx])]
            holes = [cand[int(srt[i])] for i in range(K)]
            hole_vals = [float(md[int(srt[i])]) for i in range(K)]
        except Exception:
            holes = [_norm(rng.normal(size=(d,)))]
            hole_vals = [1.0]
        # try best hole
        y0 = holes[0]
        try:
            y0 = _refine_single(y0, Xcur, steps=300, lr=0.05, T0=0.012, T1=0.001)
        except Exception:
            pass
        Xtrial = np.vstack([Xcur, y0[None, :]])
        Xtrial = _norm(Xtrial)
        # short LP polish over all points (2-4s requested; use ~1.4s for budget)
        short_t = 1.4
        # give a bit more time for last few insertions
        if k >= need - 6:
            short_t = 1.6
        # if lots of time left, use 1.4 else shrink
        remain = DEAD - time.time() - 20.0
        left_slots = need - k
        if remain < left_slots * short_t:
            short_t = max(0.6, remain / max(left_slots, 1) - 0.05)
        try:
            Xpol = _lp_polish(Xtrial, short_t, deadline=DEAD)
            vpol = _max_cos(Xpol)
        except Exception:
            Xpol, vpol = Xtrial, _max_cos(Xtrial)
        # jump detection: retry from next-best holes
        jump_thr = 0.012
        # also consider absolute high jump vs prev_max
        if vpol > prev_max + jump_thr and len(holes) > 1:
            best_alt = Xpol
            best_alt_v = vpol
            # try up to 2 alternates with cheaper polish
            for alt_i in range(1, min(3, len(holes))):
                try:
                    ya = holes[alt_i]
                    ya = _refine_single(ya, Xcur, steps=250, lr=0.05, T0=0.012, T1=0.001)
                    Xalt = _norm(np.vstack([Xcur, ya[None, :]]))
                    Xalt_p = _lp_polish(Xalt, max(0.6, short_t * 0.7), deadline=DEAD)
                    valt = _max_cos(Xalt_p)
                    if valt < best_alt_v:
                        best_alt_v = valt
                        best_alt = Xalt_p
                except Exception:
                    continue
            # also try swapped order: remove neighbour of best and reinsert both
            # neighbour = closest point to y0 in Xcur
            try:
                if best_alt_v > prev_max + jump_thr:
                    dots_n = Xcur @ (y0 / (np.linalg.norm(y0) + 1e-12))
                    nb = int(np.argmax(dots_n))
                    # remove neighbour, keep size m_now (drop nb), then add y0 and re-add neighbour at hole
                    mask = np.ones(m_now, dtype=bool)
                    mask[nb] = False
                    Xbase = Xcur[mask]
                    # find hole for neighbour reinsertion: sample small
                    C2 = 8000
                    cand2 = _norm(rng.normal(size=(C2, d)))
                    # evaluate vs Xbase+ y0
                    Xtmp = np.vstack([Xbase, y0[None, :]])
                    Xtmp = _norm(Xtmp)
                    D2 = cand2 @ Xtmp.T
                    md2 = D2.max(axis=1)
                    j2 = int(np.argmin(md2))
                    y_nb = cand2[j2]
                    y_nb = _refine_single(y_nb, Xtmp, steps=200, lr=0.05, T0=0.012, T1=0.001)
                    Xswap = np.vstack([Xtmp, y_nb[None, :]])
                    Xswap = _norm(Xswap)
                    Xswap_p = _lp_polish(Xswap, max(0.6, short_t * 0.8), deadline=DEAD)
                    vswap = _max_cos(Xswap_p)
                    if vswap < best_alt_v:
                        best_alt_v = vswap
                        best_alt = Xswap_p
            except Exception:
                pass
            Xpol, vpol = best_alt, best_alt_v
        Xcur = _norm(Xpol)
        prev_max = float(vpol)
        # if vpol is NaN etc, restore
        if not np.isfinite(prev_max):
            Xcur = _norm(np.vstack([Xcur[:m_now], holes[0][None, :]])) if Xcur.shape[0] == m_now else Xcur
            try:
                prev_max = _max_cos(Xcur)
            except Exception:
                prev_max = best0_val
    # if time cut left us short of 154, greedy fill without LP
    while Xcur.shape[0] < n:
        try:
            C = 12000
            cand = _norm(rng.normal(size=(C, d)))
            md = (cand @ Xcur.T).max(axis=1)
            j = int(np.argmin(md))
            y = cand[j]
            y = _refine_single(y, Xcur, steps=200, lr=0.05, T0=0.010, T1=0.001)
            Xcur = _norm(np.vstack([Xcur, y[None, :]]))
        except Exception:
            break
        if time.time() > DEAD - 16.0:
            # fill rest purely greedy
            while Xcur.shape[0] < n:
                cand = _norm(rng.normal(size=(4000, d)))
                md = (cand @ Xcur.T).max(axis=1)
                j = int(np.argmin(md))
                Xcur = np.vstack([Xcur, cand[j][None, :]])
            break
    # --- finish with remaining time in LP polish ---
    try:
        remain = DEAD - time.time() - 1.5
        if remain > 3 and Xcur.shape[0] == n:
            lp_time = min(max(5.0, remain), 40.0)
            Xcur = _lp_polish(Xcur, lp_time, deadline=DEAD)
    except Exception:
        pass
    # fallback to incumbent if regrowth worse and time allows small polish of incumbent
    try:
        vcur = _max_cos(_norm(Xcur))
    except Exception:
        vcur = 1e9
    if (not np.isfinite(vcur)) or (vcur > best0_val + 1e-12 and Xcur.shape[0] == n):
        # keep whichever is better; if regrowth failed to beat, return polished incumbent
        # but we already spent time; just return best0 (maybe quick snap if time left)
        try:
            if DEAD - time.time() > 4:
                bpol = _lp_polish(best0, min(5.0, DEAD - time.time() - 1.0), deadline=DEAD)
                bv = _max_cos(bpol)
                if bv < best0_val:
                    best0, best0_val = bpol, bv
        except Exception:
            pass
        if vcur > best0_val:
            Xcur = best0.copy()
    # sanitise
    Xcur = np.asarray(Xcur, dtype=float)
    if Xcur.shape != (n, d):
        # pad/truncate deterministically
        tmp = best0.copy()
        if Xcur.shape[0] >= n:
            Xcur = Xcur[:n]
        else:
            extra = tmp[:n - Xcur.shape[0]]
            Xcur = np.vstack([Xcur, extra])
            Xcur = _norm(Xcur)
    Xcur = _norm(Xcur)
    bad = ~np.isfinite(Xcur).all(axis=1)
    if np.any(bad):
        rb = _norm(rng.normal(size=(int(np.sum(bad)), d)))
        Xcur[bad] = rb
    Xcur = _norm(Xcur)
    return Xcur
