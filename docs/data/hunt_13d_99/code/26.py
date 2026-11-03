import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def rowmaxs(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def smooth_refine(v, alpha0, alpha1, steps, lr, beta=0.9, frozen=None, track_best=True):
    n, d = v.shape
    cur = v.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    mom = np.zeros_like(cur)
    if frozen is None:
        frozen = np.zeros(n, dtype=bool)
    move = ~frozen
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = alpha0 * ((alpha1 / max(alpha0, 1e-12)) ** frac)
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        Gt[frozen] = 0.0
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        step = mom / bc
        cur_lr = lr * (1.0 - 0.5 * frac)
        cur[move] = cur[move] - cur_lr * step[move]
        nr = np.linalg.norm(cur, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        cur /= nr
        if track_best and ((s % 200 == 199) or (s == steps - 1)):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
    if track_best:
        return best, best_s
    return cur, max_cos(cur)

def balanced_greedy(v, steps=300, lr0=0.02, delta=8e-05, topk=7, w_top=1.0, w_bal=0.35):
    n, d = v.shape
    cur = v.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    for s in range(steps):
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        Mask = (G > m - delta).astype(float)
        np.fill_diagonal(Mask, 0.0)
        # per-point top-k balancing mask
        Bal = np.zeros_like(Mask)
        # argsort each row descending
        idx = np.argsort(-G, axis=1)[:, :topk]
        for i in range(n):
            Bal[i, idx[i]] = 1.0
        np.fill_diagonal(Bal, 0.0)
        Bal = np.maximum(Bal, Bal.T)
        # zero diagonal already
        Comb = w_top * Mask + w_bal * Bal
        cnt = Comb.sum(axis=1, keepdims=True) + 1e-12
        Grad = Comb @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = (Grad - proj * cur) / cnt
        cur_lr = lr0 * (1.0 - 0.6 * s / max(steps, 1))
        cur = cur - cur_lr * Gt
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        if (s % 100 == 99) or (s == steps - 1):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
        if s == steps // 2:
            delta = max(delta * 0.25, 1e-06)
    return best, best_s

def lp_wide(x, seconds, kmin=900, wmin=3e-07):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius, start = 5e-4, time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        thresh = m - max(30.0 * radius, wmin, 1e-9)
        sel = c > thresh
        # enforce at least kmin pairs
        if np.sum(sel) < kmin:
            order = np.argsort(-c)
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[order[:kmin]] = True
            sel = sel | sel2
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = max_cos(y)
        if vv < best:
            best = vv
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if best < max_cos(x) - 1e-12:
            x = best_x.copy()
    return best_x

def far_reinsert(base, idx, rng, n_cand=2000):
    x0 = base.copy()
    for k in idx:
        cand = rng.standard_normal(size=(n_cand, base.shape[1]))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(x0, int(k), axis=0)
        dots = cand @ others.T
        md = dots.max(axis=1)
        x0[int(k)] = cand[np.argmin(md)]
    return x0

def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPLORE_UNTIL = 128.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        tmp = rng.standard_normal(size=(99, 13))
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 99, 13
    best_val = max_cos(xb)
    best_x = xb.copy()
    cur_x = xb.copy()
    cur_val = best_val
    sigmas = [0.015, 0.04, 0.09, 0.17, 0.30]
    it = 0
    while time.time() - t0 < EXPLORE_UNTIL and it < 1000:
        it += 1
        progress = (time.time() - t0) / max(EXPLORE_UNTIL, 1e-9)
        Temp = 0.0015 * (1.0 - progress) + 0.00015
        Gc = cur_x @ cur_x.T
        np.fill_diagonal(Gc, -2.0)
        rm = Gc.max(axis=1)
        order_worst = np.argsort(rm)
        mode = rng.random()
        try:
            if mode < 0.30:
                sigma = sigmas[int(rng.integers(0, len(sigmas)))]
                x0 = cur_x + rng.standard_normal(size=(n, d)) * sigma
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                a0 = float(rng.choice([15.0, 25.0, 40.0]))
                a1 = a0 * float(rng.choice([4.0, 6.0, 9.0]))
                steps = int(rng.integers(600, 1300))
                lr = float(rng.uniform(0.02, 0.06))
                x1, _ = smooth_refine(x0, a0, a1, steps, lr)
                x1, _ = balanced_greedy(x1, steps=250, lr0=0.018, delta=8e-05, topk=7, w_top=1.0, w_bal=0.4)
            elif mode < 0.55:
                kk = int(rng.integers(2, 8))
                idx = order_worst[-kk:]
                x0 = cur_x.copy()
                sigma = float(rng.uniform(0.05, 0.30))
                x0[idx] = x0[idx] + rng.standard_normal(size=(kk, d)) * sigma
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                if rng.random() < 0.5:
                    pick = rng.choice(idx, size=min(2, kk), replace=False)
                    x0 = far_reinsert(x0, pick, rng, n_cand=2000)
                a0 = float(rng.choice([20.0, 35.0, 55.0]))
                steps = int(rng.integers(600, 1300))
                lr = float(rng.uniform(0.02, 0.06))
                x1, _ = smooth_refine(x0, a0, a0 * 5.0 + 20.0, steps, lr)
                x1, _ = balanced_greedy(x1, steps=250, lr0=0.015, delta=6e-05, topk=8, w_top=1.0, w_bal=0.5)
            elif mode < 0.78:
                # kill worst few and reinsert into voids
                kk = int(rng.integers(2, 6))
                idx = order_worst[-kk:]
                x0 = far_reinsert(cur_x, idx, rng, n_cand=2500)
                # small shake of neighbours
                x0 = x0 + rng.standard_normal(size=(n, d)) * 0.01
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                a0 = float(rng.choice([25.0, 40.0]))
                x1, _ = smooth_refine(x0, a0, a0 * 6.0, int(rng.integers(700, 1400)), float(rng.uniform(0.025, 0.06)))
                x1, _ = balanced_greedy(x1, steps=300, lr0=0.012, delta=1e-04, topk=6, w_top=1.0, w_bal=0.3)
            else:
                # random subset large shake for topology change
                msz = int(rng.integers(8, 18))
                idx = rng.choice(n, size=msz, replace=False)
                x0 = cur_x.copy()
                if rng.random() < 0.5:
                    x0[idx] = rng.standard_normal(size=(msz, d))
                    x0[idx] /= np.linalg.norm(x0[idx], axis=1, keepdims=True)
                else:
                    x0[idx] = x0[idx] + rng.standard_normal(size=(msz, d)) * 0.4
                    x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
                x1, _ = smooth_refine(x0, 20.0, 200.0, int(rng.integers(600, 1200)), float(rng.uniform(0.03, 0.07)))
                x1, _ = balanced_greedy(x1, steps=250, lr0=0.015, delta=8e-05, topk=7, w_top=1.0, w_bal=0.4)
        except Exception:
            continue
        s1 = max_cos(x1)
        if s1 < best_val - 1e-12:
            best_val = s1
            best_x = x1.copy()
            cur_x = x1.copy()
            cur_val = s1
            if time.time() - t0 < EXPLORE_UNTIL - 6:
                try:
                    snapped = lp_wide(best_x, 1.0, kmin=900, wmin=3e-07)
                    sv = max_cos(snapped)
                    if sv < best_val:
                        best_val = sv
                        best_x = snapped.copy()
                        cur_x = snapped.copy()
                        cur_val = sv
                except Exception:
                    pass
        else:
            if s1 <= cur_val or rng.random() < np.exp(-(s1 - cur_val) / max(Temp, 1e-12)):
                cur_x = x1.copy()
                cur_val = s1
            if rng.random() < 0.07:
                cur_x = best_x.copy()
                cur_val = best_val
    remain = BUDGET - (time.time() - t0) - 1.5
    if remain > 5:
        bx = best_x.copy()
        bv = max_cos(bx)
        # low-alpha balancing then high-alpha squeeze
        try:
            tx, _ = smooth_refine(bx, 25.0, 150.0, 900, 0.03)
            tx, _ = balanced_greedy(tx, steps=400, lr0=0.010, delta=5e-05, topk=8, w_top=1.0, w_bal=0.5)
            if max_cos(tx) < bv:
                bx = tx.copy(); bv = max_cos(bx)
        except Exception:
            pass
        try:
            tx2, _ = smooth_refine(bx, 120.0, 700.0, 900, 0.012)
            if max_cos(tx2) < bv:
                bx = tx2.copy(); bv = max_cos(bx)
        except Exception:
            pass
        try:
            polished = lp_wide(bx, remain * 0.85, kmin=1100, wmin=2e-07)
            pv = max_cos(polished)
            if pv < bv:
                bx = polished.copy(); bv = pv
            # second wider pass if time
            if BUDGET - (time.time() - t0) > 6:
                polished2 = lp_wide(bx, 4.0, kmin=1400, wmin=5e-07)
                pv2 = max_cos(polished2)
                if pv2 < bv:
                    bx = polished2.copy(); bv = pv2
        except Exception:
            pass
        if bv < best_val:
            best_x = bx.copy(); best_val = bv
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
