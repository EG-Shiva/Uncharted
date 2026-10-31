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

def _count_above(x, thr):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return int((g > thr).sum() // 2)

def _rattlers(x, m=None, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    if m is None:
        m = float(g.max())
    cnt = (g > m - tol).sum(axis=1)
    pw = g.max(axis=1)
    order = np.argsort(pw)  # worst small first
    rat = np.where(cnt == 0)[0]
    if len(rat) >= 3:
        # pick 3 with smallest personal worst
        idx = rat[np.argsort(pw[rat])[:3]]
        return list(idx), m, cnt
    else:
        return list(order[:3]), m, cnt

def _greedy_pool(pool, frozen, nd):
    # pool (P,4), frozen (Nf,4) -> pick nd points maximizing min distance
    P = pool.shape[0]
    if frozen.shape[0] > 0:
        mf = pool @ frozen.T
        best = mf.max(axis=1)
    else:
        best = np.full(P, -2.0)
    chosen = []
    used = np.zeros(P, dtype=bool)
    for _ in range(nd):
        # argmin among unused
        tmp = best.copy()
        tmp[used] = np.inf
        bi = int(tmp.argmin())
        used[bi] = True
        chosen.append(bi)
        # update best with new point
        b = pool[bi]
        s = pool @ b
        # best = max(best, s)
        np.maximum(best, s, out=best)
        # keep chosen at inf
        best[bi] = np.inf
    return pool[np.array(chosen)]

def _local_opt(frozen, ins, steps, tau0, tau1, lr0, lr1, rng=None):
    # optimize ins (Ni,4) against frozen (Nf,4)+itself
    if ins.shape[0] == 0:
        return ins
    Ni = ins.shape[0]
    m = np.zeros_like(ins)
    v = np.zeros_like(ins)
    b1, b2, eps = 0.9, 0.999, 1e-8
    y = ins.copy()
    Nf = frozen.shape[0]
    for k in range(steps):
        frac = k / max(1, steps - 1)
        tau = tau0 * (tau1 / tau0) ** frac if tau1 != tau0 else tau0
        lr = lr0 + (lr1 - lr0) * frac
        # similarities: y vs frozen and y vs y
        if Nf > 0:
            Sf = y @ frozen.T  # Ni x Nf
            mf = Sf.max() if Sf.size else -2.0
        else:
            mf = -2.0
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        mg = G.max() if Ni > 1 else -2.0
        gm = mf if mf > mg else mg
        # weights
        if Nf > 0:
            Wf = np.exp((Sf - gm) / tau)
            Wf[Wf < 1e-12] = 0.0
            gf = Wf @ frozen
        else:
            gf = 0.0
        W = np.exp((G - gm) / tau)
        np.fill_diagonal(W, 0.0)
        W[W < 1e-12] = 0.0
        gg = W @ y
        # normalize by sum
        ssum = (Wf.sum() if Nf > 0 else 0.0) + W.sum()
        if not np.isfinite(ssum) or ssum == 0:
            break
        grad = (gf if Nf > 0 else 0.0) + gg
        grad = grad / max(ssum, 1e-12) * (Ni + Nf)
        dot = (grad * y).sum(axis=1, keepdims=True)
        g = grad - dot * y
        m *= b1; m += (1 - b1) * g
        v *= b2; v += (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1); bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1; vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        y = y - lr * step
        y = _norm(y)
    return y

def _anneal_global(x, steps, tau0, tau1, lr0, lr1):
    n = x.shape[0]
    m = np.zeros_like(x); v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / max(1, steps - 1)
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        np.subtract(G, gm, out=G)
        np.divide(G, tau, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = G.sum()
        if not np.isfinite(S) or S == 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1; m += (1 - b1) * g
        v *= b2; v += (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1); bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1; vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        try:
            c = (x @ x.T)[iu]
        except Exception:
            break
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SURG_END = t0 + 120.0
    rng = np.random.default_rng(12345)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((272, 4))
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (272, 4):
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        x0 = flat[:need].reshape(272, 4)
    x0 = np.where(np.isfinite(x0), x0, 0.0)
    x0 = _norm(x0)

    best = x0.copy()
    best_val = _max_cos(best)
    best_cnt = _count_above(best, 0.8975)
    cur = best.copy()
    cur_val = best_val
    cur_cnt = best_cnt

    ks = [24, 36, 48]
    caps = [0.8975, 0.8965, 0.8980, 0.8955]
    POOL = 40000

    it = 0
    # short initial polish to stabilize
    # main surgery loop
    while time.time() < SURG_END and time.time() < DEADLINE - 35:
        it += 1
        k = ks[it % len(ks)]
        cap = caps[it % len(caps)]
        # temperature schedule
        frac = (time.time() - t0) / max(1e-9, (SURG_END - t0))
        Temp = 0.0004 * (1 - frac) + 0.00002 * frac
        try:
            # rattlers of cur
            rat, mcur, _ = _rattlers(cur)
            u = rng.standard_normal((4,))
            u /= np.linalg.norm(u)
            dots = cur @ u
            topk = np.argsort(-dots)[:k]
            delset = set(topk.tolist()) | set([int(r) for r in rat])
            delidx = np.array(sorted(delset), dtype=int)
            keepmask = np.ones(272, dtype=bool)
            keepmask[delidx] = False
            frozen = cur[keepmask]
            # pool
            pool = rng.standard_normal((POOL, 4))
            pool = _norm(pool)
            nd = len(delidx)
            re = _greedy_pool(pool, frozen, nd)
            # local tangential opt of inserted only
            # vary local steps by k
            re = _local_opt(frozen, re, steps=250, tau0=0.012, tau1=0.0012, lr0=0.01, lr1=0.001)
            y = cur.copy()
            y[delidx] = re
            y = _norm(y)
            # short global anneal (unfrozen)
            if it % 2 == 0:
                y = _anneal_global(y, steps=350, tau0=0.008, tau1=0.0009, lr0=0.006, lr1=0.0007)
            else:
                y = _anneal_global(y, steps=250, tau0=0.015, tau1=0.0015, lr0=0.008, lr1=0.0008)
            # LP polish short only if promising
            y_val_pre = _max_cos(y)
            y_cnt_pre = _count_above(y, cap)
            # quick filter: skip LP if much worse
            if y_val_pre < cur_val + 0.004:
                y = _lp_polish(y, seconds=1.6, deadline=DEADLINE - 30)
            y_val = _max_cos(y)
            y_cnt_cap = _count_above(y, cap)
            y_cnt_ref = _count_above(y, 0.8975)
            cur_cnt_cap = _count_above(cur, cap)
            # energies
            w = 1.5e-6
            Ecur = cur_val + w * cur_cnt_cap
            Enew = y_val + w * y_cnt_cap
            # update best (primary max, secondary ref count)
            if (y_val < best_val - 1e-12) or (abs(y_val - best_val) <= 1e-12 and y_cnt_ref < best_cnt):
                best = y.copy(); best_val = y_val; best_cnt = y_cnt_ref
            # SA accept for cur
            if Enew < Ecur:
                cur = y.copy(); cur_val = y_val; cur_cnt = y_cnt_ref
            else:
                dE = Enew - Ecur
                if rng.random() < np.exp(-dE / max(Temp, 1e-9)):
                    cur = y.copy(); cur_val = y_val; cur_cnt = y_cnt_ref
                # else keep cur; occasionally restart from best
                if it % 12 == 0 and rng.random() < 0.35:
                    cur = best.copy(); cur_val = best_val; cur_cnt = best_cnt
        except Exception:
            continue

    # final polish of best
    try:
        remain = DEADLINE - time.time() - 2.0
        if remain > 5:
            best = _lp_polish(best, seconds=min(remain, 35.0), deadline=DEADLINE - 1.0)
    except Exception:
        pass
    best = _norm(best)
    return best
