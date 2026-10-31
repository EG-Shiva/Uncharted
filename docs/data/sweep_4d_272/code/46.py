import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _count_above(x, thr):
    G = x @ x.T
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu]
    return int((c > thr).sum())

def _personal_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _anneal_full(x, steps, tau0, tau1, lr0, lr1):
    n = x.shape[0]
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        G -= gm
        G /= tau
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(G.sum())
        if not np.isfinite(S) or S == 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _anneal_partial(x, is_ins, steps, tau0, tau1, lr0, lr1):
    # optimize only rows with is_ins==True, considering pairs touching them
    n = x.shape[0]
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    frozen = ~is_ins
    # mask for allowed pairs: at least one endpoint inserted
    allow = np.ones((n, n), dtype=bool)
    # frozen-frozen not allowed
    ff = np.ix_(np.where(frozen)[0], np.where(frozen)[0])
    # build mask matrix once
    M = np.ones((n, n), dtype=float)
    fidx = np.where(frozen)[0]
    # set frozen-frozen to 0 via outer
    # use boolean: M[i,j]=0 if frozen[i]&frozen[j]
    if len(fidx) > 0:
        # efficient: start with ones, zero block
        pass
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        # mask frozen-frozen out by -inf for max
        if np.any(frozen):
            # copy to avoid modifying G for weights? we need G values for weights
            # compute relevant max
            Gm = G.copy()
            # set frozen-frozen to -2
            # use outer product
            fr = frozen[:, None] & frozen[None, :]
            Gm[fr] = -2.0
            gm = float(Gm.max())
        else:
            gm = float(G.max())
        # weights
        G -= gm
        G /= tau
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        if np.any(frozen):
            G[fr] = 0.0
        S = float(G.sum())
        if not np.isfinite(S) or S == 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        # only update inserted
        g[~is_ins] = 0.0
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        step[~is_ins] = 0.0
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = float(c.max())
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    deadline = t0 + 172.0
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
    nrm = np.linalg.norm(x0, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x0 = x0 / nrm

    x_best = x0.copy()
    best_val = _max_cos(x_best)
    best_cnt = _count_above(x_best, 0.8975)

    cur = x_best.copy()
    cur_val = best_val
    cur_cnt = best_cnt

    ks = [24, 36, 48]
    # tau/lr schedules to cycle (caps variation via count threshold weight too)
    full_taus = [(0.012, 0.0012), (0.008, 0.0007), (0.02, 0.002)]
    full_lrs = [(0.006, 0.0006), (0.004, 0.0004), (0.009, 0.0009)]
    thr_list = [0.8975, 0.8965, 0.8975, 0.8980]
    w_cnt = 1.0e-6
    T0, T1 = 0.0012, 0.00006

    it = 0
    POOL = 40000
    # surgery loop ~120s
    while time.time() < t0 + 120.0 and time.time() < deadline - 50.0:
        it += 1
        elapsed = time.time() - t0
        fracT = elapsed / 120.0
        Temp = T0 * (T1 / T0) ** min(max(fracT, 0.0), 1.0)
        k = ks[it % len(ks)]
        thr = thr_list[it % len(thr_list)]
        tau0, tau1 = full_taus[it % len(full_taus)]
        lr0, lr1 = full_lrs[it % len(full_lrs)]

        # pick random direction
        u = rng.standard_normal((4,))
        un = float(np.linalg.norm(u))
        if un == 0 or not np.isfinite(un):
            u = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            u = u / un
        dots = cur @ u
        order_cap = np.argsort(-dots)
        del_set = set(order_cap[:k].tolist())
        # 3 rattlers: points with smallest personal max
        pm = _personal_max(cur)
        order_r = np.argsort(pm)
        for idx in order_r[:3]:
            del_set.add(int(idx))
        del_idx = np.array(sorted(del_set), dtype=int)
        mdel = len(del_idx)
        keep_mask = np.ones(272, dtype=bool)
        keep_mask[del_idx] = False
        F = cur[keep_mask]  # Nf x 4

        # greedy farthest-point rebuild from pool
        try:
            pool = rng.standard_normal((POOL, 4))
            pool /= np.linalg.norm(pool, axis=1, keepdims=True)
            pool = np.where(np.isfinite(pool), pool, 0.0)
            # renormalize any bad rows
            pn = np.linalg.norm(pool, axis=1, keepdims=True)
            bad = (pn[:, 0] < 1e-12)
            if np.any(bad):
                pool[bad] = np.array([1.0, 0.0, 0.0, 0.0])
                pool /= np.linalg.norm(pool, axis=1, keepdims=True)
        except Exception:
            continue
        # max dot to frozen
        try:
            maxF = pool @ F.T
            # max over frozen
            best_scores = maxF.max(axis=1)
        except Exception:
            continue
        Y = []
        used = np.zeros(POOL, dtype=bool)
        # forbid pool points too close to frozen? handled by scores
        for s in range(mdel):
            idx = int(np.argmin(np.where(used, np.inf, best_scores)))
            # if all used (should not happen) break
            ynew = pool[idx].copy()
            Y.append(ynew)
            used[idx] = True
            best_scores[idx] = np.inf
            try:
                d = pool @ ynew
                # elementwise max
                np.maximum(best_scores, d, out=best_scores)
                best_scores[used] = np.inf
            except Exception:
                pass
        Y = np.stack(Y, axis=0)
        x_new = cur.copy()
        x_new[del_idx] = Y
        x_new /= np.linalg.norm(x_new, axis=1, keepdims=True)

        # local tangential optimisation of inserted only
        is_ins = np.zeros(272, dtype=bool)
        is_ins[del_idx] = True
        try:
            x_new = _anneal_partial(x_new, is_ins, 220, 0.006, 0.0008, 0.006, 0.0006)
        except Exception:
            pass
        # unfreeze: short global anneal
        try:
            x_new = _anneal_full(x_new, 300, tau0, tau1, lr0, lr1)
        except Exception:
            pass
        try:
            v_new = _max_cos(x_new)
            c_new = _count_above(x_new, 0.8975)
            c_thr = _count_above(x_new, thr)
        except Exception:
            continue
        # energies
        E_cur = cur_val + w_cnt * cur_cnt
        E_new = v_new + w_cnt * c_new
        # also consider thr count slightly? use same w
        accept = False
        if E_new < E_cur:
            accept = True
        else:
            delta = E_new - E_cur
            try:
                prob = float(np.exp(-delta / max(Temp, 1e-9)))
            except Exception:
                prob = 0.0
            if rng.random() < prob:
                accept = True
        if accept:
            cur = x_new.copy()
            cur_val = v_new
            cur_cnt = c_new
        if v_new < best_val - 1e-12 or (abs(v_new - best_val) < 1e-12 and c_new < best_cnt):
            # strict improvement
            best_val = v_new
            best_cnt = c_new
            x_best = x_new.copy()
        # also if thr count improved a lot, keep as cur anyway via SA above

    # refine best with a couple low-temp anneals
    for _ in range(2):
        if time.time() > deadline - 42:
            break
        try:
            xc = _anneal_full(x_best.copy(), 350, 0.005, 0.0006, 0.0035, 0.00035)
            vc = _max_cos(xc)
            cc = _count_above(xc, 0.8975)
            if vc < best_val or (abs(vc - best_val) < 1e-12 and cc < best_cnt):
                x_best = xc
                best_val = vc
                best_cnt = cc
            # also move cur to best for final polish
            cur = x_best.copy()
            cur_val = best_val
            cur_cnt = best_cnt
        except Exception:
            break
    remain = deadline - time.time() - 2.0
    if remain > 6:
        try:
            xp = _lp_polish(x_best, min(remain, 36.0), deadline - 1.0)
            vp = _max_cos(xp)
            if vp <= best_val:
                x_best = xp
                best_val = vp
        except Exception:
            pass
    if not np.all(np.isfinite(x_best)):
        x_best = x0.copy()
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
