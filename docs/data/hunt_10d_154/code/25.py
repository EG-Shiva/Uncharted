import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _mc(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        try:
            v = _mc(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _refine_single(y, X, T=0.004, iters=160, lr=0.08):
    m_ad = np.zeros_like(y)
    v_ad = np.zeros_like(y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    for t in range(1, iters + 1):
        d = X @ y
        mv = float(d.max())
        e = np.exp((d - mv) / max(T, 1e-12))
        s = float(e.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w = e / s
        g = w @ X
        gt = g - float(g @ y) * y
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * gt
        v_ad = b2 * v_ad + (1.0 - b2) * (gt * gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        st = mhat / (np.sqrt(vhat) + eps)
        y = y - lr * st
        nrm = float(np.linalg.norm(y))
        if nrm == 0 or not np.isfinite(nrm):
            break
        y = y / nrm
    return y

def _refine_two_stage(y, X):
    y = _refine_single(y, X, T=0.005, iters=120, lr=0.08)
    y = _refine_single(y, X, T=0.0015, iters=80, lr=0.04)
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
                v = _mc(a)
                if v < best0_val:
                    best0_val = v
                    best0 = a.copy()
        except Exception:
            pass
    if best0 is None:
        best0 = _norm(rng.normal(size=(n, d)))
        best0_val = _mc(best0)

    # --- deflation: contact analysis ---
    X0 = best0.copy()
    G = X0 @ X0.T
    np.fill_diagonal(G, -2.0)
    m0 = float(G.max())
    # dual-ish proxy: row max and contact counts
    cnt = (G > m0 - 1e-7).sum(axis=1)
    rowmax = G.max(axis=1)
    # rattlers = 0 contacts at tight tol; fallback to lowest rowmax
    ratt = np.where(cnt == 0)[0].tolist()
    # want total delete ~30 -> remaining 124 (within 114-124)
    NDEL = 30
    # if rattler count differs, adjust
    order_high = np.argsort(-cnt, kind="stable")
    # also consider dual: rowmax high
    # combine: sort by (-cnt, -rowmax)
    order = sorted(range(n), key=lambda i: (-int(cnt[i]), -float(rowmax[i])))
    dellist = []
    for r in ratt:
        if len(dellist) < NDEL:
            dellist.append(int(r))
    for idx in order:
        if len(dellist) >= NDEL:
            break
        if idx not in dellist:
            dellist.append(int(idx))
    dellist = sorted(dellist)
    keep = [i for i in range(n) if i not in set(dellist)]
    Xcur = X0[keep].copy()  # (124,10)
    # --- LP-polish reduced set until substantially lower ---
    # budget ~10s
    init_budget = 10.0
    if DEAD - time.time() < init_budget + 60:
        init_budget = max(4.0, (DEAD - time.time()) * 0.1)
    Xcur = _lp_polish(Xcur, seconds=init_budget, deadline=DEAD - 1.0)
    try:
        cur_max = _mc(Xcur)
    except Exception:
        cur_max = 1e9

    n_need = n - Xcur.shape[0]
    # --- regrowth one at a time ---
    for slot in range(n_need):
        if time.time() > DEAD - 22.0:
            break
        remaining_slots = n_need - slot
        time_left = (DEAD - 20.0) - time.time()
        if time_left < 0.5:
            break
        per_slot = time_left / remaining_slots - 0.25
        # clamp 0.8..3.5 (spec 2-4s when possible)
        if per_slot > 3.0:
            per_slot = 3.0
        if per_slot < 0.8:
            per_slot = 0.8
        # hole sampling: at least 20000
        C = 20000
        # adapt C slightly if very low time
        try:
            cand = rng.normal(size=(C, d))
            cand = _norm(cand)
            D = cand @ Xcur.T
            md = D.max(axis=1)
            K = 6
            idxp = np.argpartition(md, K)[:K]
            o = idxp[np.argsort(md[idxp])]
            hole_vals = md[o]
        except Exception:
            o = None
            hole_vals = None
        if o is None:
            # fallback random
            y_new = _norm(rng.normal(size=(1, d)))[0]
            Xcur = np.vstack([Xcur, y_new[None, :]])
            Xcur = _lp_polish(Xcur, seconds=per_slot, deadline=DEAD - 1.0)
            continue
        old_max = _mc(Xcur)
        Xpre = Xcur.copy()
        best_try_X = None
        best_try_v = 1e9
        # try best hole first
        n_tries = 2 if remaining_slots > 2 else 1
        # first attempt uses most of budget, retry uses smaller
        for ti in range(min(3, K)):
            if ti >= 2 and best_try_v < old_max + 0.015:
                break
            if ti > 1:
                # only do extra retry if jump sharp
                if best_try_v < old_max + 0.012:
                    break
            y0 = cand[o[ti]].copy()
            try:
                y0 = _refine_two_stage(y0, Xpre)
            except Exception:
                pass
            Xtrial = np.vstack([Xpre, y0[None, :]])
            # short LP polish: first try gets full per_slot, retries get 1.2s
            if ti == 0:
                lp_s = per_slot
            else:
                lp_s = min(1.5, per_slot)
            try:
                Xtrial = _lp_polish(Xtrial, seconds=lp_s, deadline=DEAD - 1.0)
                v = _mc(Xtrial)
            except Exception:
                continue
            if v < best_try_v:
                best_try_v = v
                best_try_X = Xtrial
            # spec: if jump sharply, retry next-best hole -> continue loop
            if ti == 0 and v > old_max + 0.012:
                # will loop to next hole
                continue
            else:
                if ti == 0:
                    break
        # swapped-order neighbour reinsert fallback if still sharp jump
        if best_try_X is not None and best_try_v > old_max + 0.015 and time.time() < DEAD - 25:
            try:
                # find neighbouring high-contact point to the inserted point
                y_ins = best_try_X[-1]
                dots = Xpre @ y_ins
                j = int(np.argmax(dots))
                # contact count of Xpre
                Gp = Xpre @ Xpre.T
                np.fill_diagonal(Gp, -2.0)
                mp = float(Gp.max())
                cntp = (Gp > mp - 1e-7).sum(axis=1)
                # only if neighbour is relatively loaded
                # remove j, keep inserted point, then reinsert j swapped
                Xtmp = np.delete(Xpre, j, axis=0)
                # Xtmp + y_ins
                # search hole for j against Xtmp+y_ins
                Xtmp2 = np.vstack([Xtmp, y_ins[None, :]])
                C2 = 12000
                cand2 = _norm(rng.normal(size=(C2, d)))
                D2 = cand2 @ Xtmp2.T
                md2 = D2.max(axis=1)
                b2i = int(np.argmin(md2))
                yj = cand2[b2i].copy()
                yj = _refine_two_stage(yj, Xtmp2)
                Xt2 = np.vstack([Xtmp2, yj[None, :]])
                # Xt2 has same size as Xpre+1 (we removed one added two? check)
                # Xtmp is (m-1), +y_ins => m, +yj => m+1 = desired size
                Xt2 = _lp_polish(Xt2, seconds=min(1.8, per_slot), deadline=DEAD - 1.0)
                v2 = _mc(Xt2)
                if v2 < best_try_v:
                    best_try_v = v2
                    best_try_X = Xt2
            except Exception:
                pass
        if best_try_X is not None:
            Xcur = best_try_X
        else:
            # emergency: just append refined best hole without polish
            y0 = cand[o[0]].copy()
            try:
                y0 = _refine_two_stage(y0, Xpre)
            except Exception:
                pass
            Xcur = np.vstack([Xpre, y0[None, :]])
        # loop continues; short LP already done inside
    # if regrowth stopped early (time), fill randomly to 154
    while Xcur.shape[0] < n:
        y = _norm(rng.normal(size=(1, d)))[0]
        Xcur = np.vstack([Xcur, y[None, :]])
    if Xcur.shape[0] > n:
        Xcur = Xcur[:n]
    # --- finish with remaining time in LP polish ---
    # keep fallback to incumbent
    try:
        cur_final_v = _mc(Xcur)
    except Exception:
        cur_final_v = 1e9
    remain = DEAD - time.time() - 0.8
    if remain < 3:
        remain = 3
    if remain > 40:
        remain = 40
    try:
        Xcur = _lp_polish(Xcur, seconds=remain, deadline=DEAD - 0.5)
        v_after = _mc(Xcur)
    except Exception:
        v_after = cur_final_v
    # return better of regrown vs original incumbent
    try:
        if best0_val < _mc(Xcur):
            # incumbent still better; return polished incumbent if time allows? quick polish
            # do tiny extra polish on best0 with leftover 1s? no time, return best0
            return _norm(best0)
        else:
            return _norm(Xcur)
    except Exception:
        return _norm(Xcur)
