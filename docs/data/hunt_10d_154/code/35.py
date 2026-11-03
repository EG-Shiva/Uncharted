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
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() > deadline:
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
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _refine_single(p0, X, steps=260, lr=0.03, T0=0.02, T1=0.0025):
    # minimize softmax max dot(p, X) over p on sphere, X fixed
    p = p0.copy() / (np.linalg.norm(p0) + 1e-12)
    m_ad = np.zeros_like(p)
    v_ad = np.zeros_like(p)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 8e-4:
            T = 8e-4
        s = X @ p  # (m,)
        mv = s.max()
        e = np.exp((s - mv) / T)
        S = e.sum()
        if not np.isfinite(S) or S <= 0:
            break
        w = e / S
        grad = w @ X  # (d,)
        gt = grad - np.dot(grad, p) * p
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * gt
        v_ad = b2 * v_ad + (1.0 - b2) * (gt * gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        # cap step norm
        sn = float(np.linalg.norm(step)) + 1e-12
        if sn > 1.0:
            step = step / sn
        p = p - lr * step
        nrm = float(np.linalg.norm(p))
        if nrm == 0 or not np.isfinite(nrm):
            p = p0 / (np.linalg.norm(p0) + 1e-12)
        else:
            p = p / nrm
    return p

def _best_holes(cur, rng, C=20000, K=10):
    d = cur.shape[1]
    # batch to limit memory if C large
    # single batch of C=20000 is fine (20k x 120)
    cand = rng.normal(size=(C, d))
    cand = _norm(cand)
    D = cand @ cur.T
    md = D.max(axis=1)
    idx = np.argpartition(md, K)[:K]
    order = idx[np.argsort(md[idx])]
    return cand, md, order

def solve():
    t0 = time.time()
    DEAD = t0 + 170.0
    rng = np.random.default_rng(7)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _max_cos(best)
    base = best.copy()

    # --- deflation: contact counts with wide band ---
    G = base @ base.T
    np.fill_diagonal(G, -2.0)
    mcur = float(G.max())
    delta = 0.004
    cnt = (G > mcur - delta).sum(axis=1)
    order_asc = np.argsort(cnt)  # low first
    order_desc = np.argsort(-cnt)  # high first
    # 11 lowest as rattlers
    n_ratt = 11
    ratt = order_asc[:n_ratt].tolist()
    # 23 highest not overlapping
    n_high = 23
    high = []
    for ii in order_desc:
        if ii not in ratt:
            high.append(int(ii))
        if len(high) >= n_high:
            break
    victims = np.array(ratt + high, dtype=int)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[victims] = False
    cur = base[keep_mask].copy()  # 120 points
    # small jitter to help reorganise core before polish
    cur = _norm(cur + rng.normal(scale=0.015, size=cur.shape))

    # LP-polish the deflated set until substantially lower
    # spend ~8s
    try:
        cur = _lp_polish(cur, 8.0, deadline=DEAD - 100.0)
    except Exception:
        pass
    cur_max = _max_cos(cur)

    # --- regrowth one at a time to exactly 154 ---
    n_target = 154
    n_slots = n_target - cur.shape[0]
    for slot in range(n_slots):
        t_rem = DEAD - time.time()
        # reserve ~12s for final polish
        if t_rem < 14.0:
            # emergency: fill remaining with plain greedy holes, no LP
            n_left = n_target - cur.shape[0]
            for _ in range(n_left):
                cand, md, order = _best_holes(cur, rng, C=20000, K=5)
                p0 = cand[order[0]]
                p_ref = _refine_single(p0, cur, steps=150, lr=0.03, T0=0.015, T1=0.003)
                cur = np.vstack([cur, p_ref[None, :]])
            break
        # choose LP seconds adaptively
        slots_left = n_target - cur.shape[0]
        # keep pace: if much time left use 2s, else shrink
        if t_rem > slots_left * 2.5 + 15:
            lp_sec = 2.0
        elif t_rem > slots_left * 1.2 + 12:
            lp_sec = 1.0
        else:
            lp_sec = 0.7
        # sample holes
        C = 22000
        cand, md, order = _best_holes(cur, rng, C=C, K=10)
        # sorted order already best-first
        prev_max = cur_max
        best_try_val = 1e9
        best_try_X = None
        # try up to 3 holes if jump sharp
        n_tries = 3
        for ti in range(n_tries):
            if time.time() > DEAD - 13.0:
                break
            oi = order[ti] if ti < len(order) else order[0]
            p0 = cand[oi]
            p_ref = _refine_single(p0, cur, steps=260, lr=0.03, T0=0.02, T1=0.0025)
            trial = np.vstack([cur, p_ref[None, :]])
            try:
                polished = _lp_polish(trial, lp_sec, deadline=DEAD - 10.0)
            except Exception:
                polished = trial
            v = _max_cos(polished)
            if v < best_try_val:
                best_try_val = v
                best_try_X = polished
            # accept early if no sharp jump
            if v - prev_max < 0.008:
                break
        if best_try_X is None:
            # fallback single
            p_ref = _refine_single(cand[order[0]], cur, steps=150, lr=0.03, T0=0.015, T1=0.003)
            best_try_X = np.vstack([cur, p_ref[None, :]])
            best_try_val = _max_cos(best_try_X)
        # swapped-order repair if jump still sharp: remove neighbouring high-contact point and reinsert both
        if best_try_val - prev_max > 0.015 and time.time() < DEAD - 20.0:
            try:
                # neighbour = closest existing point to new point
                newp = best_try_X[-1]
                dots = best_try_X[:-1] @ newp
                j = int(np.argmax(dots))
                tmp = np.delete(best_try_X, j, axis=0)  # back to same size as cur
                # reinsert removed point j into tmp via hole search
                cand2, md2, order2 = _best_holes(tmp, rng, C=12000, K=5)
                p0b = cand2[order2[0]]
                p_ref2 = _refine_single(p0b, tmp, steps=200, lr=0.03, T0=0.015, T1=0.003)
                trial2 = np.vstack([tmp, p_ref2[None, :]])
                polished2 = _lp_polish(trial2, 1.0, deadline=DEAD - 10.0)
                v2 = _max_cos(polished2)
                if v2 < best_try_val:
                    best_try_val = v2
                    best_try_X = polished2
            except Exception:
                pass
        cur = best_try_X
        cur_max = best_try_val

    # --- finish with remaining time in LP polish ---
    t_rem = DEAD - time.time() - 1.0
    if t_rem > 3.0:
        lp_time = min(max(5.0, t_rem - 1.0), 40.0)
        try:
            cur = _lp_polish(cur, lp_time, deadline=DEAD - 0.5)
        except Exception:
            pass
    # ensure shape and norm
    if cur.shape[0] != 154:
        # pad or trim deterministically
        if cur.shape[0] > 154:
            cur = cur[:154]
        else:
            need = 154 - cur.shape[0]
            extra = _norm(rng.normal(size=(need, d)))
            cur = np.vstack([cur, extra])
    cur = _norm(cur)
    cur[~np.isfinite(cur)] = 0.0
    nr = np.linalg.norm(cur, axis=1, keepdims=True)
    bad = (nr.squeeze() < 1e-12)
    if np.any(bad):
        rb = rng.normal(size=(int(np.sum(bad)), d))
        cur[bad] = _norm(rb)
    cur = _norm(cur)
    return cur
