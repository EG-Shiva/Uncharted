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

def _lp_polish(x, seconds, t_deadline=1e18):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    cur_best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if now >= t_deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
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
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            cur_best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            # still move to y if not worse by much? no, shrink
            radius *= 0.5
            # if stuck, keep x but continue
            if radius < 1e-12:
                break
    return cur_best_x

def _refine_single(x0, Y, steps=400, lr=0.08, T0=0.02, T1=0.002, seed_adam=True):
    d = Y.shape[1]
    x = np.asarray(x0, dtype=float).copy()
    nrm = np.linalg.norm(x)
    if nrm == 0 or not np.isfinite(nrm):
        x = np.random.normal(size=d); nrm = np.linalg.norm(x)
    x = x / nrm
    m_ad = np.zeros(d); v_ad = np.zeros(d)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0; b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 1e-12:
            T = 1e-12
        dots = Y @ x  # (m,)
        mv = float(dots.max())
        E = np.exp((dots - mv) / T)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ Y) / S  # (d,)
        Gt = Grad - float(Grad @ x) * x
        b1p *= beta1; b2p *= beta2
        m_ad = beta1 * m_ad + (1.0 - beta1) * Gt
        v_ad = beta2 * v_ad + (1.0 - beta2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        x = x - lr * step
        nrm = np.linalg.norm(x)
        if nrm == 0 or not np.isfinite(nrm):
            break
        x = x / nrm
    return x

def _best_holes(cand, cur, k=8):
    D = cand @ cur.T
    md = D.max(axis=1)
    idx = np.argpartition(md, k)[:k]
    # sort these k by md
    order = idx[np.argsort(md[idx])]
    return order, md

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    FINAL_RESERVE = 25.0
    rng = np.random.default_rng(42)
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

    # --- contact analysis on incumbent ---
    G0 = best0 @ best0.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(G0.max())
    tol = 1e-9
    cnt = (G0 > m0 - tol).sum(axis=1)
    # ensure 11 rattlers
    ratt_idx = np.where(cnt == 0)[0]
    if len(ratt_idx) != 11:
        order_low = np.argsort(cnt, kind="stable")
        ratt_idx = order_low[:11]
    ratt_set = set(int(v) for v in ratt_idx)
    # highest-contact / dual proxy = contact count (also rowmax tie)
    order_high = np.argsort(-cnt, kind="stable")
    KDEL = 27  # in 19..29 -> remaining 116 (in 114..124)
    high_del = []
    for v in order_high:
        if int(v) in ratt_set:
            continue
        high_del.append(int(v))
        if len(high_del) >= KDEL:
            break
    del_idx = set(ratt_set) | set(high_del)
    keep = np.array([i for i in range(n) if i not in del_idx], dtype=int)
    # safety: ensure 114..124 remain
    # if keep not in range adjust
    if not (114 <= len(keep) <= 124):
        # force to 116 by adjusting
        target = 116
        need_del = n - target
        # take 11 lowest + top (need_del-11) highest
        order_low = np.argsort(cnt, kind="stable")
        ratt2 = list(order_low[:11])
        s2 = set(ratt2)
        high2 = []
        for v in order_high:
            if int(v) in s2:
                continue
            high2.append(int(v))
            if len(high2) >= need_del - 11:
                break
        del_idx = set(ratt2) | set(high2)
        keep = np.array([i for i in range(n) if i not in del_idx], dtype=int)

    cur = best0[keep].copy()
    cur = _norm(cur)

    # --- LP-polish deflated set until substantially lower ---
    # budget ~10s but respect deadline
    now = time.time()
    deflate_budget = 10.0
    # leave enough for regrowth + final: need at least (154-len(cur))*0.9 + FINAL_RESERVE
    need_slots = n - len(cur)
    min_needed = need_slots * 0.9 + FINAL_RESERVE + 2.0
    avail = DEAD - now
    if avail - min_needed < 2.0:
        deflate_budget = 2.0
    else:
        deflate_budget = min(10.0, avail - min_needed)
    if deflate_budget > 1.0:
        try:
            cur = _lp_polish(cur, deflate_budget, t_deadline=DEAD - min_needed + 2.0)
        except Exception:
            pass
    prev_max = _max_cos(cur)

    # --- regrowth one at a time ---
    HOP_DEAD = DEAD - FINAL_RESERVE
    # insertion loop
    slot = 0
    while len(cur) < n and time.time() < HOP_DEAD - 1.0:
        remaining = n - len(cur)
        time_left_hop = HOP_DEAD - time.time()
        # per-slot LP budget adaptive
        # need 0.25 overhead per slot for sampling+refine
        per_slot = (time_left_hop / remaining) - 0.28
        lp_budget = float(np.clip(per_slot, 0.6, 2.2))
        # at least 20000 samples
        C = 20000
        # if plenty time, use 24000
        if time_left_hop / remaining > 2.8:
            C = 24000
        cand = _norm(rng.normal(size=(C, d)))
        order, md = _best_holes(cand, cur, k=10)
        # try best hole first
        best_try_x = None
        best_try_val = 1e9
        best_try_cfg = None
        # we will try up to 1 primary + retries if jump
        # primary
        x0 = cand[order[0]]
        xr = _refine_single(x0, cur, steps=350, lr=0.06, T0=0.015, T1=0.0015)
        aug = np.vstack([cur, xr[None, :]])
        aug_p = aug
        try:
            aug_p = _lp_polish(aug, lp_budget, t_deadline=HOP_DEAD)
        except Exception:
            aug_p = aug
        new_max = _max_cos(aug_p)
        jump = new_max - prev_max
        # sharp jump threshold
        if jump > 0.012 and remaining >= 1 and time.time() < HOP_DEAD - 3.0:
            # retry from next-best holes (2 alternatives, shorter polish)
            alt_budget = max(0.6, min(1.2, lp_budget * 0.6))
            tried = [(new_max, aug_p, order[0])]
            for kk in [1, 2, 4]:
                if kk >= len(order):
                    break
                if time.time() > HOP_DEAD - 2.0:
                    break
                xa0 = cand[order[kk]]
                # enforce diversity: skip if too close to first try
                # (cosine > 0.9 means same hole region)
                if kk > 1 and float(xa0 @ cand[order[0]]) > 0.96:
                    continue
                xar = _refine_single(xa0, cur, steps=300, lr=0.06, T0=0.015, T1=0.0015)
                aaug = np.vstack([cur, xar[None, :]])
                try:
                    aaug_p = _lp_polish(aaug, alt_budget, t_deadline=HOP_DEAD)
                except Exception:
                    aaug_p = aaug
                av = _max_cos(aaug_p)
                tried.append((av, aaug_p, order[kk]))
                if av < new_max - 1e-4:
                    break
            # pick best among tried
            tried.sort(key=lambda t: t[0])
            new_max, aug_p, _ = tried[0]
            jump = new_max - prev_max
            # if still sharp jump, try neighbour-relocation swap
            if jump > 0.012 and time.time() < HOP_DEAD - 4.0:
                try:
                    # neighbour: closest point in cur to new point
                    new_pt = aug_p[-1]
                    dots_n = cur @ new_pt
                    j_close = int(np.argmax(dots_n))
                    # among 5 closest, pick highest-contact one
                    Gc = cur @ cur.T
                    np.fill_diagonal(Gc, -2.0)
                    mc = float(Gc.max())
                    cntc = (Gc > mc - 1e-7).sum(axis=1)
                    closest5 = np.argsort(-dots_n)[:min(5, len(cur))]
                    j_star = max(closest5.tolist(), key=lambda j: cntc[j])
                    # relocate j_star to best hole of set without it
                    mask = np.ones(len(cur), dtype=bool); mask[j_star] = False
                    cur_wo = cur[mask]
                    C2 = 10000
                    cand2 = _norm(rng.normal(size=(C2, d)))
                    D2 = cand2 @ cur_wo.T
                    md2 = D2.max(axis=1)
                    b2 = int(np.argmin(md2))
                    xj0 = cand2[b2]
                    xjr = _refine_single(xj0, cur_wo, steps=300, lr=0.06, T0=0.015, T1=0.0015)
                    cur_reloc = cur.copy(); cur_reloc[j_star] = xjr
                    # short polish of relocated base (1s)
                    try:
                        cur_reloc_p = _lp_polish(cur_reloc, min(1.0, alt_budget), t_deadline=HOP_DEAD)
                    except Exception:
                        cur_reloc_p = cur_reloc
                    # now insert new hole into relocated base
                    C3 = 12000
                    cand3 = _norm(rng.normal(size=(C3, d)))
                    D3 = cand3 @ cur_reloc_p.T
                    md3 = D3.max(axis=1)
                    b3 = int(np.argmin(md3))
                    xn0 = cand3[b3]
                    xnr = _refine_single(xn0, cur_reloc_p, steps=300, lr=0.06, T0=0.015, T1=0.0015)
                    aaug2 = np.vstack([cur_reloc_p, xnr[None, :]])
                    try:
                        aaug2_p = _lp_polish(aaug2, alt_budget, t_deadline=HOP_DEAD)
                    except Exception:
                        aaug2_p = aaug2
                    av2 = _max_cos(aaug2_p)
                    if av2 < new_max:
                        # accept swapped path: update cur base implicitly via aug
                        aug_p = aaug2_p
                        new_max = av2
                except Exception:
                    pass
        cur = _norm(aug_p)
        prev_max = _max_cos(cur)
        slot += 1

    # if time ran out and still <154 (should not), fill greedily without LP
    while len(cur) < n:
        C = 20000
        cand = _norm(rng.normal(size=(C, d)))
        D = cand @ cur.T
        md = D.max(axis=1)
        b = int(np.argmin(md))
        xr = _refine_single(cand[b], cur, steps=250, lr=0.06, T0=0.015, T1=0.0015)
        cur = np.vstack([cur, xr[None, :]])
        cur = _norm(cur)

    # --- finish with remaining time in LP polish ---
    remain = DEAD - time.time()
    # keep at least 1s margin
    lp_final = remain - 1.0
    if lp_final > 3.0:
        try:
            cur = _lp_polish(cur, lp_final, t_deadline=DEAD - 0.5)
        except Exception:
            pass
    cur = _norm(cur)
    cur_val = _max_cos(cur)
    # fallback to incumbent if regrowth failed to beat it
    if cur_val > best0_val and best0.shape == (n, d):
        # try a quick polish of incumbent with leftover? no time; return incumbent
        return _norm(best0)
    return _norm(cur)
