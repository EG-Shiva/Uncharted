import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _per_point_worst_and_contacts(x, m=None):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    w = np.max(g, axis=1)
    if m is None:
        m = float(np.max(w))
    cnt = np.sum(g > m - 1e-9, axis=1)
    return w, cnt, m

def _anneal(x, steps, lr0, lr1, T0, T1, rng, deadline, noise0=0.0):
    n, d = x.shape
    for it in range(steps):
        if (it & 31) == 0 and time.time() > deadline:
            break
        frac = it / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * frac
        T = T0 + (T1 - T0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        thr = m - 8.0 * T
        # weights
        # avoid overflow: exp((G-m)/T)
        W = np.exp((G - m) / T, out=np.empty_like(G), where=(G > thr))
        # zero out where not selected and diag
        # exp with where leaves uninit where False, so mask:
        W[G <= thr] = 0.0
        np.fill_diagonal(W, 0.0)
        S = W @ x
        dot = np.einsum('ij,ij->i', S, x)
        Tang = S - dot[:, None] * x
        x = x - lr * Tang
        if noise0 > 0:
            ns = noise0 * (1.0 - frac)
            if ns > 1e-13:
                nz = rng.standard_normal(size=(n, d))
                nd = np.einsum('ij,ij->i', nz, x)
                nz -= nd[:, None] * x
                x = x + ns * nz
        # normalize
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        # guard NaN
        nr[nr == 0] = 1.0
        x /= nr
    return x

def _void_pick(existing, rng, n_cand=20000, batch=5000):
    d = existing.shape[1]
    best_v = np.inf
    best_p = None
    # batched to limit memory
    for s in range(0, n_cand, batch):
        b = min(batch, n_cand - s)
        cand = rng.standard_normal(size=(b, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ existing.T
        worst = np.max(dots, axis=1)
        j = int(np.argmin(worst))
        if worst[j] < best_v:
            best_v = float(worst[j])
            best_p = cand[j].copy()
    return best_p, best_v

def _refine_single(p, others, rng, iters=3):
    cur = p.copy()
    cur_w = float(np.max(others @ cur))
    for it in range(iters):
        sig = [0.05, 0.02, 0.008][it] if it < 3 else 0.008
        trial = cur[None, :] + sig * rng.standard_normal(size=(2000, cur.shape[0]))
        trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        w = np.max(trial @ others.T, axis=1)
        j = int(np.argmin(w))
        if w[j] < cur_w:
            cur_w = float(w[j])
            cur = trial[j].copy()
    return cur, cur_w

def solve():
    t0 = time.time()
    HARD = t0 + 155.0
    SOFT = t0 + 132.0
    rng = np.random.default_rng(0)
    best_x = None
    best_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        best_x = a
        best_v = _max_cos(a)
    n_full, d = 323, 7

    # homotopy cycles
    n_cycles = 0
    while time.time() < SOFT and n_cycles < 12:
        n_cycles += 1
        if time.time() > HARD - 25:
            break
        base = best_x.copy()
        try:
            w, cnt, m = _per_point_worst_and_contacts(base)
        except Exception:
            w = np.zeros(n_full); cnt = np.zeros(n_full, dtype=int)
        k_remove = int(rng.integers(25, 61))
        # select removal set: rattlers + hubs + random
        ratt = np.where(cnt == 0)[0].tolist()
        order_cnt = np.argsort(-cnt)
        order_w = np.argsort(-w)
        idx_set = set()
        for j in ratt:
            idx_set.add(int(j))
        # add worst hubs
        for j in order_w[:8]:
            if len(idx_set) >= k_remove:
                break
            if rng.random() < 0.8:
                idx_set.add(int(j))
        # add high-contact hubs
        for j in order_cnt[:20]:
            if len(idx_set) >= k_remove:
                break
            if rng.random() < 0.5:
                idx_set.add(int(j))
        # fill randomly (bias to low-contact and random mixture)
        while len(idx_set) < k_remove:
            j = int(rng.integers(0, n_full))
            idx_set.add(j)
        idx_rm = np.array(sorted(idx_set)[:k_remove], dtype=int)
        keep = np.ones(n_full, dtype=bool)
        keep[idx_rm] = False
        y = base[keep].copy()  # (n_red,7)
        # optimize reduced code to well below 0.665
        # fast anneal
        steps_red = 700 if time.time() < SOFT - 30 else 350
        y = _anneal(y, steps=steps_red, lr0=0.05, lr1=0.004, T0=0.02, T1=0.0015,
                    rng=rng, deadline=HARD - 20, noise0=0.002)
        if time.time() > HARD - 20:
            break
        # regrow one at a time
        # randomize insertion order not needed; just append
        # we maintain array list
        cur_list = [y[i] for i in range(y.shape[0])]
        # pre-stack for fast dots
        import math
        n_to_add = n_full - y.shape[0]
        for a_i in range(n_to_add):
            if time.time() > HARD - 18:
                break
            existing = np.stack(cur_list, axis=0)
            # sample void point
            nc = 20000
            # reduce candidates if low time
            if time.time() > SOFT:
                nc = 8000
            p, _ = _void_pick(existing, rng, n_cand=nc, batch=5000)
            p, _ = _refine_single(p, existing, rng, iters=3)
            cur_list.append(p)
            # short global relaxation every 5 inserts
            if (len(cur_list) % 5 == 0) or (a_i == n_to_add - 1):
                eg = np.stack(cur_list, axis=0)
                st = 220
                if time.time() > SOFT:
                    st = 120
                eg = _anneal(eg, steps=st, lr0=0.02, lr1=0.002, T0=0.008, T1=0.001,
                             rng=rng, deadline=HARD - 15, noise0=0.0)
                # write back
                cur_list = [eg[i] for i in range(eg.shape[0])]
        if len(cur_list) != n_full:
            continue
        yfull = np.stack(cur_list, axis=0)
        # short global relax after restore
        if time.time() < HARD - 12:
            st2 = 400 if time.time() < SOFT else 150
            yfull = _anneal(yfull, steps=st2, lr0=0.015, lr1=0.001, T0=0.006, T1=0.0007,
                            rng=rng, deadline=HARD - 10, noise0=0.0)
        try:
            v = _max_cos(yfull)
        except Exception:
            continue
        if v < best_v:
            best_v = v
            best_x = yfull.copy()
        # else discard; continue to next basin

    # final gradient-only polish of best (no LP, to respect time cap)
    try:
        if time.time() < HARD - 6:
            remain = HARD - time.time()
            stf = 900 if remain > 12 else 300
            y = best_x.copy()
            y = _anneal(y, steps=stf, lr0=0.012, lr1=0.0008, T0=0.005, T1=0.0005,
                        rng=rng, deadline=HARD - 2, noise0=0.0)
            v = _max_cos(y)
            if v < best_v:
                best_v = v
                best_x = y.copy()
    except Exception:
        pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    # final safety: replace NaN
    bad = ~np.isfinite(best_x).all(axis=1)
    if np.any(bad):
        rb = rng.standard_normal(size=(int(np.sum(bad)), d))
        rb /= np.linalg.norm(rb, axis=1, keepdims=True)
        best_x[bad] = rb
    return best_x
