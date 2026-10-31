import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    x = np.asarray(x, dtype=float)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[~np.isfinite(nrm)] = 1.0
    nrm[nrm < 1e-12] = 1.0
    x = x / nrm
    x[~np.isfinite(x)] = 0.0
    n2 = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (~np.isfinite(n2[:, 0])) | (n2[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _count_above(x, thr):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    iu = np.triu_indices(x.shape[0], 1)
    return int((g[iu] > thr).sum())

def _contacts(x, thr_off=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    cnt = (g >= m - thr_off).sum(axis=1)
    return m, cnt, g

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        cmat = x @ x.T
        c = cmat[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _adam_full(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** frac if tau1 != tau0 else tau0
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        G -= gm
        G /= tau
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = G.sum()
        if not np.isfinite(S) or S <= 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _adam_moving(x, mov, steps, tau0, tau1, lr0, lr1):
    m = np.zeros((len(mov), 4))
    v = np.zeros((len(mov), 4))
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** frac if tau1 != tau0 else tau0
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        G -= gm
        G /= tau
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = G.sum()
        if not np.isfinite(S) or S <= 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        g = gt[mov]
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x[mov] = x[mov] - lr * step
        x[mov] /= np.linalg.norm(x[mov], axis=1, keepdims=True)
    return x

def _greedy_refill(frozen, kdel, rng, pool_size=40000):
    pool = rng.standard_normal((pool_size, 4))
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    m = frozen.shape[0]
    best = np.empty(pool_size, dtype=float)
    # initial max-sim to frozen in chunks
    CH = 8000
    for s in range(0, pool_size, CH):
        e = min(s + CH, pool_size)
        sims = pool[s:e] @ frozen.T
        best[s:e] = sims.max(axis=1)
    placed = []
    used = np.zeros(pool_size, dtype=bool)
    for _ in range(kdel):
        # argmin among unused
        tmp = best.copy()
        tmp[used] = np.inf
        bi = int(tmp.argmin())
        used[bi] = True
        p = pool[bi].copy()
        placed.append(p)
        # update best with new point
        upd = pool @ p
        np.maximum(best, upd, out=best)
    return np.array(placed)

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    surgical_end = t0 + 120.0
    rng = np.random.default_rng(20261031)
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (272, 4):
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        x0 = flat[:need].reshape(272, 4)
    x0 = _norm(x0)
    COUNT_THR = 0.8975
    W_CNT = 2e-6
    best_val = _max_cos(x0)
    best_cnt = _count_above(x0, COUNT_THR)
    x_best = x0.copy()
    x_cur = x0.copy()
    cur_val, cur_cnt = best_val, best_cnt
    # quick initial LP snap not needed; start surgery directly
    k_cycle = [24, 36, 48]
    tau_cycle = [(0.012, 0.0012), (0.020, 0.002), (0.008, 0.0008)]
    lr_cycle = [(0.010, 0.001), (0.014, 0.0012), (0.007, 0.0007)]
    cap_cycle = [0.8975, 0.8965, 0.8980]
    T0, T1 = 2e-4, 2e-6
    it = 0
    POOL = 40000
    while time.time() < surgical_end and time.time() < deadline - 40:
        it += 1
        k = k_cycle[it % len(k_cycle)]
        tau0, tau1 = tau_cycle[it % len(tau_cycle)]
        lr0, lr1 = lr_cycle[it % len(lr_cycle)]
        cap = cap_cycle[it % len(cap_cycle)]
        # pick random cap direction
        u = rng.standard_normal((4,))
        u /= np.linalg.norm(u)
        dots = x_cur @ u
        order = np.argsort(-dots)
        topk = order[:k]
        # find rattlers in current state
        _, cntc, _ = _contacts(x_cur)
        ratt = np.where(cntc == 0)[0]
        # delete set = topk union rattlers (limit extra to 3)
        delset = set(topk.tolist())
        for r in ratt.tolist()[:6]:
            delset.add(int(r))
        del_idx = np.array(sorted(delset), dtype=int)
        kdel = len(del_idx)
        keep_mask = np.ones(272, dtype=bool)
        keep_mask[del_idx] = False
        frozen = x_cur[keep_mask]
        # greedy rebuild from pool
        try:
            placed = _greedy_refill(frozen, kdel, rng, pool_size=POOL)
        except Exception:
            continue
        x_new = x_cur.copy()
        x_new[del_idx] = placed
        # local tangential optimisation of inserted only
        try:
            x_new = _adam_moving(x_new, del_idx, steps=120, tau0=0.010, tau1=0.001, lr0=0.012, lr1=0.0015)
        except Exception:
            pass
        # unfreeze short anneal
        try:
            x_new = _adam_full(x_new, steps=220, tau0=tau0, tau1=tau1, lr0=lr0, lr1=lr1)
        except Exception:
            pass
        # short LP polish
        remain = deadline - time.time()
        if remain < 5:
            break
        try:
            x_new = _lp_polish(x_new, seconds=min(1.6, remain - 2.0), deadline=deadline)
        except Exception:
            pass
        new_val = _max_cos(x_new)
        # count with fixed thr 0.8975 and also cycled cap for diversity (use fixed for energy)
        new_cnt = _count_above(x_new, COUNT_THR)
        # update global best lexicographically then energy
        if (new_val < best_val - 1e-12) or (abs(new_val - best_val) <= 1e-12 and new_cnt < best_cnt):
            x_best, best_val, best_cnt = x_new.copy(), new_val, new_cnt
        # SA acceptance on current chain
        Ecur = cur_val + W_CNT * cur_cnt
        Enew = new_val + W_CNT * new_cnt
        dE = Enew - Ecur
        elapsed = (time.time() - t0) / max(surgical_end - t0, 1.0)
        Temp = T0 * (T1 / T0) ** min(max(elapsed, 0.0), 1.0)
        if dE < 0 or rng.random() < np.exp(-dE / max(Temp, 1e-12)):
            x_cur, cur_val, cur_cnt = x_new.copy(), new_val, new_cnt
        # also occasionally restart from best if stuck
        if it % 12 == 0 and cur_val > best_val + 0.002:
            x_cur, cur_val, cur_cnt = x_best.copy(), best_val, best_cnt
    # final polish of best with remaining budget
    try:
        remain = deadline - time.time() - 1.0
        if remain > 3:
            x_best = _lp_polish(x_best, seconds=min(remain * 0.7, 25.0), deadline=deadline - 0.5)
            bv = _max_cos(x_best)
            bc = _count_above(x_best, COUNT_THR)
            best_val, best_cnt = bv, bc
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
