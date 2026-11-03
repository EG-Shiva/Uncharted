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
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
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
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _refine_single(y, Xfix, steps=260, lr=0.04, T0=0.015, T1=0.0015):
    y = y / (np.linalg.norm(y) + 1e-18)
    m_ad = np.zeros_like(y)
    v_ad = np.zeros_like(y)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0; b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1,1e-12)/max(T0,1e-12)) / (steps-1)
    else:
        logr = 0.0
    for t in range(1, steps+1):
        T = T0 * np.exp(logr*(t-1))
        if T < 1e-12:
            T = 1e-12
        dots = Xfix @ y
        mv = float(dots.max())
        E = np.exp((dots - mv) / T)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ Xfix) / S
        dot = float(Grad @ y)
        Gt = Grad - dot * y
        b1p *= beta1; b2p *= beta2
        m_ad = beta1*m_ad + (1.0-beta1)*Gt
        v_ad = beta2*v_ad + (1.0-beta2)*(Gt*Gt)
        mhat = m_ad/(1.0-b1p); vhat = v_ad/(1.0-b2p)
        step = mhat/(np.sqrt(vhat)+eps)
        y = y - lr*step
        ny = np.linalg.norm(y)
        if ny == 0 or not np.isfinite(ny):
            break
        y = y/ny
    return y

def _greedy_holes(Xcur, rng, C=20000, K=12):
    d = Xcur.shape[1]
    cand = rng.normal(size=(C, d))
    cand = _norm(cand)
    Dots = cand @ Xcur.T
    md = Dots.max(axis=1)
    idx = np.argpartition(md, K)[:K]
    # sort those K by md
    order = idx[np.argsort(md[idx])]
    return cand, md, order

def solve():
    t0 = time.time()
    DEAD = t0 + 170.0
    rng = np.random.default_rng(0)
    n, d = 154, 10
    best_full = None
    best_full_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best_full_val:
                    best_full_val = v
                    best_full = a.copy()
        except Exception:
            pass
    if best_full is None:
        best_full = _norm(rng.normal(size=(n, d)))
        best_full_val = _max_cos(best_full)
    orig_best = best_full.copy()
    orig_val = best_full_val

    # --- contact analysis for deflation ---
    G = best_full @ best_full.T
    np.fill_diagonal(G, -2.0)
    m0 = float(G.max())
    # tight threshold for contacts (jammed)
    cnt = (G > m0 - 1e-7).sum(axis=1)
    rattlers = np.where(cnt == 0)[0]
    # target removal ~36 -> remaining 118
    total_remove = 36
    R = len(rattlers)
    # number of high-contact to remove
    need_high = total_remove - R
    if need_high < 19:
        need_high = 19
    if need_high > 29 + max(0, 11 - R):
        need_high = 29 + max(0, 11 - R)
    order_high = np.argsort(-cnt)
    high_set = []
    for o in order_high:
        if o in set(rattlers.tolist()):
            continue
        high_set.append(int(o))
        if len(high_set) >= need_high:
            break
    remove_idx = sorted([int(v) for v in rattlers.tolist()] + high_set)
    # if too few rattlers, fill up to total_remove with next highest
    if len(remove_idx) < total_remove:
        for o in order_high:
            if int(o) not in remove_idx:
                remove_idx.append(int(o))
                if len(remove_idx) >= total_remove:
                    break
    remove_set = set(remove_idx)
    keep_mask = np.array([i not in remove_set for i in range(n)])
    Xcur = best_full[keep_mask].copy()
    Xcur = _norm(Xcur)

    # --- LP-polish deflated set substantially lower ---
    # budget ~8s or until deadline reserve
    init_polish = 8.0
    # ensure we leave enough time for regrowth + final
    # 36 insertions * ~2s =72 + final 30 => need ~102s
    remain = DEAD - time.time()
    if remain < 120:
        init_polish = min(init_polish, max(3.0, remain - 90.0))
    try:
        Xcur = _lp_polish(Xcur, init_polish, DEAD - 95.0 if (DEAD-time.time())>100 else DEAD-5)
    except Exception:
        pass
    prev_max = _max_cos(Xcur)

    num_to_add = n - Xcur.shape[0]
    # track full best
    full_best = orig_best.copy()
    full_best_val = orig_val

    SHORT = 1.7  # short LP seconds per insertion (2-4s spec, use ~1.7 to fit time; will use 2.0 when time allows)
    for slot in range(num_to_add):
        if time.time() > DEAD - 32.0:
            break
        # adaptive short time: if plenty time use 2.2 else 1.2
        rem_after = DEAD - time.time()
        slots_left = num_to_add - slot
        # reserve 28s final polish
        budget_per = (rem_after - 28.0) / max(slots_left, 1)
        short_t = min(2.5, max(1.0, budget_per - 0.35))
        # at least 20000 samples
        C = 20000
        try:
            cand, md, order = _greedy_holes(Xcur, rng, C=C, K=16)
        except Exception:
            # fallback single random
            y = _norm(rng.normal(size=(1, d)))[0]
            Xcur = np.vstack([Xcur, y[None, :]])
            continue
        # try best hole first
        best_try_X = None
        best_try_val = 1e9
        # attempt logic: up to 2 attempts unless jump small
        # first attempt uses order[0]
        tried = []
        nattempts = 1
        # we will decide retry after seeing jump
        cur_attempt_holes = [int(order[0])]
        # also precompute a second distinct hole far from first for potential retry
        second = None
        for o in order[1:]:
            if float(cand[int(o)] @ cand[int(order[0])]) < 0.9:
                second = int(o)
                break
        if second is None and len(order) > 1:
            second = int(order[1])

        attempt = 0
        X_before = Xcur.copy()
        m_before = prev_max
        final_X_after = None
        final_val_after = 1e9
        while True:
            hid = cur_attempt_holes[attempt] if attempt < len(cur_attempt_holes) else second
            y0 = cand[hid].copy()
            try:
                y0 = _refine_single(y0, X_before, steps=260, lr=0.04, T0=0.015, T1=0.0015)
            except Exception:
                pass
            Xtrial = np.vstack([X_before, y0[None, :]])
            try:
                Xtrial_p = _lp_polish(Xtrial, short_t, DEAD - 28.0 if slot < num_to_add-1 else DEAD - 2)
            except Exception:
                Xtrial_p = Xtrial
            try:
                v = _max_cos(Xtrial_p)
            except Exception:
                v = 1e9
            if v < final_val_after:
                final_val_after = v
                final_X_after = Xtrial_p.copy()
            tried.append((hid, v))
            # jump detection
            jump = v - m_before
            # need retry if sharp jump and we have time and attempts left
            if jump > 0.012 and attempt < 1 and time.time() < DEAD - 35.0:
                # retry from next-best hole
                # pick next hole distinct from tried
                tried_set = set([h for h, _ in tried])
                nxt = None
                for o in order:
                    o = int(o)
                    if o in tried_set:
                        continue
                    # require some distance from tried
                    ok = True
                    for h, _ in tried:
                        if float(cand[o] @ cand[h]) > 0.92:
                            ok = False
                            break
                    if ok:
                        nxt = o
                        break
                if nxt is None:
                    break
                cur_attempt_holes.append(nxt)
                attempt += 1
                continue
            elif jump > 0.025 and attempt < 2 and time.time() < DEAD - 40.0:
                # remove additional neighbouring high-contact point and reinsert both swapped
                try:
                    # neighbour = closest point to inserted y0 among X_before with high contacts
                    dots_n = X_before @ y0
                    # contact load of X_before
                    Gb = X_before @ X_before.T
                    np.fill_diagonal(Gb, -2.0)
                    mb = float(Gb.max())
                    cntb = (Gb > mb - 1e-7).sum(axis=1)
                    # score neighbour by dot + contact load
                    score = dots_n + 0.02 * cntb
                    k = int(np.argmax(score))
                    Xtmp = np.delete(X_before, k, axis=0)
                    # re-greedy insert for k's replacement + y0? swapped order:
                    # first reinsert neighbour region: pick best hole for Xtmp
                    cand2, md2, order2 = _greedy_holes(Xtmp, rng, C=20000, K=8)
                    yA = _refine_single(cand2[int(order2[0])], Xtmp, steps=200, lr=0.04)
                    Xtmp2 = np.vstack([Xtmp, yA[None, :]])
                    # now insert y0-like hole for the second slot
                    cand3, md3, order3 = _greedy_holes(Xtmp2, rng, C=20000, K=8)
                    # use y0 refined against Xtmp2
                    yB = _refine_single(y0, Xtmp2, steps=200, lr=0.04)
                    Xtrial2 = np.vstack([Xtmp2, yB[None, :]])
                    Xtrial2_p = _lp_polish(Xtrial2, short_t, DEAD-28.0)
                    v2 = _max_cos(Xtrial2_p)
                    if v2 < final_val_after:
                        final_val_after = v2
                        final_X_after = Xtrial2_p.copy()
                except Exception:
                    pass
                break
            else:
                break
        Xcur = final_X_after
        prev_max = final_val_after
        # if we have reached full size intermediate, track
        if Xcur.shape[0] == n:
            if final_val_after < full_best_val:
                full_best_val = final_val_after
                full_best = Xcur.copy()
        # also if intermediate full? only at end
    # if time ran out and Xcur <154, greedily fill without polish
    while Xcur.shape[0] < n:
        try:
            cand, md, order = _greedy_holes(Xcur, rng, C=20000, K=4)
            y0 = cand[int(order[0])]
            try:
                y0 = _refine_single(y0, Xcur, steps=150, lr=0.04)
            except Exception:
                pass
            Xcur = np.vstack([Xcur, y0[None, :]])
        except Exception:
            y = _norm(rng.normal(size=(1, d)))[0]
            Xcur = np.vstack([Xcur, y[None, :]])
        if Xcur.shape[0] == n:
            v = _max_cos(Xcur)
            if v < full_best_val:
                full_best_val = v
                full_best = Xcur.copy()
            break
    # finish with remaining time in LP polish on the regrown set
    remain = DEAD - time.time()
    final_polish_t = min(32.0, max(4.0, remain - 2.0))
    if final_polish_t > 2.0:
        try:
            Xcur = _lp_polish(Xcur, final_polish_t, DEAD - 1.0)
            vcur = _max_cos(Xcur)
            if vcur < full_best_val:
                full_best_val = vcur
                full_best = Xcur.copy()
            else:
                # polish best as well briefly if regrown worse
                r2 = (DEAD - time.time()) * 0.4
                if r2 > 2.0:
                    try:
                        fb = _lp_polish(full_best, r2, DEAD-1.0)
                        fv = _max_cos(fb)
                        if fv < full_best_val:
                            full_best = fb
                            full_best_val = fv
                    except Exception:
                        pass
        except Exception:
            pass
    # return the best full-size seen
    if full_best.shape != (n, d):
        full_best = orig_best.copy()
    # ensure full_best is the best between Xcur and full_best
    try:
        vc = _max_cos(Xcur)
        if vc < full_best_val:
            full_best = Xcur.copy()
    except Exception:
        pass
    full_best = _norm(full_best)
    return full_best
