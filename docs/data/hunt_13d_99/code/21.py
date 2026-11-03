import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


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


def greedy_minimax(v, steps=400, lr0=0.02, delta=0.003):
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
        cnt = Mask.sum(axis=1, keepdims=True) + 1e-12
        Grad = Mask @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = (Grad - proj * cur) / cnt
        cur_lr = lr0 * (1.0 - 0.7 * s / max(steps, 1))
        cur = cur - cur_lr * Gt
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        if (s % 100 == 99) or (s == steps - 1):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
        # anneal delta
        if s == steps // 2:
            delta = delta * 0.3
    return best, best_s


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if best < max_cos(x) - 1e-12:
            x = best_x.copy()
    return best_x


def far_points_for_idx(base, idx, rng, n_cand=2500):
    x0 = base.copy()
    for k in idx:
        cand = rng.standard_normal(size=(n_cand, base.shape[1]))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        others = np.delete(x0, k, axis=0)
        dots = cand @ others.T
        md = dots.max(axis=1)
        x0[k] = cand[np.argmin(md)]
    return x0


def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPLORE_UNTIL = 132.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.array(xb, dtype=float)
    if xb.shape != (99, 13):
        xb = rng.standard_normal(size=(99, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 99, 13
    best_val = max_cos(xb)
    best_x = xb.copy()
    cur_x = xb.copy()
    cur_val = best_val

    sigmas = [0.02, 0.05, 0.10, 0.18, 0.30]
    it = 0
    while time.time() - t0 < EXPLORE_UNTIL and it < 1000:
        it += 1
        progress = (time.time() - t0) / max(EXPLORE_UNTIL, 1e-9)
        Temp = 0.002 * (1.0 - progress) + 0.0002
        mode = rng.random()
        # contact / overload info of current walker
        Gc = cur_x @ cur_x.T
        np.fill_diagonal(Gc, -2.0)
        rowmax = Gc.max(axis=1)
        order_worst = np.argsort(rowmax)

        if mode < 0.32:
            sigma = sigmas[int(rng.integers(0, len(sigmas)))]
            x0 = cur_x + rng.standard_normal(size=(n, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            a0 = float(rng.choice([20.0, 30.0, 50.0]))
            a1 = a0 * float(rng.choice([3.0, 5.0, 8.0]))
            steps = int(rng.integers(700, 1800))
            lr = float(rng.uniform(0.015, 0.055))
            x1, _ = smooth_refine(x0, a0, a1, steps, lr)
            x1, s1 = greedy_minimax(x1, steps=300, lr0=0.015, delta=0.004)
        elif mode < 0.60:
            kk = int(rng.integers(2, 9))
            idx = order_worst[-kk:]
            x0 = cur_x.copy()
            sigma = float(rng.uniform(0.05, 0.35))
            x0[idx] = x0[idx] + rng.standard_normal(size=(kk, d)) * sigma
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            # sometimes far re-insert one of them
            if rng.random() < 0.4:
                x0 = far_points_for_idx(x0, rng.choice(idx, size=min(2, kk), replace=False), rng)
            a0 = float(rng.choice([25.0, 40.0, 70.0]))
            steps = int(rng.integers(800, 2000))
            lr = float(rng.uniform(0.015, 0.05))
            x1, _ = smooth_refine(x0, a0, a0 * 4.0 + 20.0, steps, lr)
            x1, s1 = greedy_minimax(x1, steps=300, lr0=0.02, delta=0.003)
        else:
            # cap surgery around overloaded hub
            hub_pool = order_worst[-6:]
            center = int(hub_pool[rng.integers(0, len(hub_pool))])
            dots_c = cur_x @ cur_x[center]
            kcap = int(rng.integers(8, 16))
            idx = np.argsort(-dots_c)[:kcap]
            frozen = np.ones(n, dtype=bool)
            frozen[idx] = False
            # init cap points far from background
            x0 = cur_x.copy()
            # randomize cap points substantially
            if rng.random() < 0.6:
                x0[idx] = rng.standard_normal(size=(kcap, d))
                x0[idx] /= np.linalg.norm(x0[idx], axis=1, keepdims=True)
            else:
                x0 = far_points_for_idx(x0, idx, rng, n_cand=1500)
            # frozen-background optimisation, few restarts implicitly via one run
            a0, a1 = 30.0, 250.0
            steps_f = int(rng.integers(600, 1100))
            lr_f = float(rng.uniform(0.03, 0.07))
            x0, _ = smooth_refine(x0, a0, a1, steps_f, lr_f, frozen=frozen)
            # unfreeze joint refinement
            steps_j = int(rng.integers(700, 1400))
            a0j = float(rng.choice([30.0, 50.0]))
            x1, _ = smooth_refine(x0, a0j, a0j * 5.0, steps_j, 0.03)
            x1, s1 = greedy_minimax(x1, steps=250, lr0=0.015, delta=0.003)

        s1 = max_cos(x1)
        # update best
        if s1 < best_val - 1e-12:
            best_val = s1
            best_x = x1.copy()
            # snap to local optimum quickly to lock gain
            if time.time() - t0 < EXPLORE_UNTIL - 5:
                snapped = lp_polish(best_x, 1.2)
                sv = max_cos(snapped)
                if sv < best_val:
                    best_val = sv
                    best_x = snapped.copy()
            cur_x = x1.copy()
            cur_val = s1
        else:
            # Metropolis move for walker
            if s1 <= cur_val or rng.random() < np.exp(-(s1 - cur_val) / max(Temp, 1e-9)):
                cur_x = x1.copy()
                cur_val = s1
            # occasional restart walker to best
            if rng.random() < 0.08:
                cur_x = best_x.copy()
                cur_val = best_val

    remain = BUDGET - (time.time() - t0) - 1.0
    if remain > 4:
        # alternate high-alpha smooth + LP for final squeeze
        bx = best_x.copy()
        bv = max_cos(bx)
        # short high-alpha tune
        try:
            tx, _ = smooth_refine(bx, 150.0, 800.0, 1200, 0.012)
            tx, _ = greedy_minimax(tx, steps=400, lr0=0.008, delta=0.001)
            if max_cos(tx) < bv:
                bx = tx.copy()
                bv = max_cos(bx)
        except Exception:
            pass
        bx = lp_polish(bx, remain * 0.9)
        bv2 = max_cos(bx)
        if bv2 < best_val:
            best_x = bx.copy()
            best_val = bv2
        else:
            # keep whichever is better between tuned and original
            if bv < best_val:
                best_x = bx.copy() if bv2 <= bv + 1e-12 else best_x
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
