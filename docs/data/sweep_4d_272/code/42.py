import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _count_above(x, thr=0.8975):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return int(np.sum(G > thr) // 2)

def _rattlers(x, m=None, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    if m is None:
        m = float(np.max(G))
    cnt = np.sum(G > m - tol, axis=1)
    return np.where(cnt == 0)[0], cnt, m

def _local_refine_ins(Xfull, n_fro, beta, steps, lr):
    # only update rows n_fro:
    n = Xfull.shape[0]
    ins = np.arange(n_fro, n)
    for _ in range(steps):
        G = Xfull @ Xfull.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ Xfull
        g = grad[ins]
        xi = Xfull[ins]
        d = np.sum(g * xi, axis=1, keepdims=True)
        gt = g - d * xi
        # normalize step by mean norm
        sc = float(np.mean(np.linalg.norm(gt, axis=1))) + 1e-12
        xi = xi - (lr / sc) * gt
        xi = xi / np.maximum(np.linalg.norm(xi, axis=1, keepdims=True), 1e-12)
        Xfull[ins] = xi
    return Xfull

def _global_refine(X0, stages):
    X = X0.copy()
    n = X.shape[0]
    vel = np.zeros_like(X)
    mu = 0.85
    max_step = 0.05
    for beta, steps, lr in stages:
        for _ in range(steps):
            G = X @ X.T
            np.fill_diagonal(G, -1.0)
            m = float(np.max(G))
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
            rs = np.sum(W, axis=1, keepdims=True) + 1e-12
            grad = W @ X
            d = np.sum(grad * X, axis=1, keepdims=True)
            gt = (grad - d * X) / rs
            vel = mu * vel - lr * gt
            vn = np.sqrt(np.sum(vel * vel, axis=1, keepdims=True)) + 1e-12
            scale = np.minimum(1.0, max_step / vn)
            vel = vel * scale
            X = X + vel
            X = _normalize(X)
    return X

def _lp_polish(x, t_end):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        if time.time() + 0.8 > t_end and radius < 5e-4:
            break
        c = (x @ x.T)[iu0]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu0[0][sel], iu0[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    T_HARD = t0 + 175.0
    T_SURG = t0 + 120.0
    rng = np.random.default_rng(12345)
    try:
        base = np.load("best.npy")
    except Exception:
        base = np.load("record.npy")
    base = _normalize(np.asarray(base, dtype=np.float64))
    if base.shape != (272, 4):
        base = _normalize(rng.standard_normal((272, 4)))
    n = 272
    best = base.copy()
    best_val = _max_cos(best)
    best_cnt = _count_above(best, 0.8975)
    cur = best.copy()
    cur_val = best_val
    cur_cnt = best_cnt
    cur_E = cur_val + 2e-7 * cur_cnt

    caps = [24, 36, 48]
    # short global stages after rebuild
    short_stages = [(120.0, 70, 0.008), (350.0, 70, 0.004)]

    it = 0
    POOL = 32000
    # for speed, chunk pool matmul if needed
    while time.time() < T_SURG and it < 200:
        if time.time() + 2.0 > T_SURG:
            break
        k = caps[it % len(caps)]
        # temperature schedule
        prog = (time.time() - t0) / max(1e-9, (T_SURG - t0))
        Temp = 3e-4 * (1.0 - prog) + 1e-6 * prog
        # random cap direction
        u = rng.standard_normal(4)
        u = u / max(np.linalg.norm(u), 1e-12)
        dots = cur @ u
        idx_cap = np.argsort(-dots)[:k]
        rat, cntarr, mcur = _rattlers(cur)
        # pick up to 3 rattlers (lowest contacts if fewer)
        if len(rat) >= 3:
            extra = rat[:3]
        elif len(rat) > 0:
            extra = rat
            # pad with lowest-contact points not already in cap
            order = np.argsort(cntarr)
            need = 3 - len(rat)
            add = []
            for o in order:
                if o not in idx_cap and o not in rat:
                    add.append(o)
                    if len(add) >= need:
                        break
            extra = np.concatenate([extra, np.array(add, dtype=int)]) if len(add) > 0 else extra
        else:
            order = np.argsort(cntarr)
            extra = []
            for o in order:
                if o not in idx_cap:
                    extra.append(o)
                    if len(extra) >= 3:
                        break
            extra = np.array(extra, dtype=int)
        delset = np.unique(np.concatenate([idx_cap, extra]))
        dnum = len(delset)
        mask = np.ones(n, dtype=bool)
        mask[delset] = False
        frozen = cur[mask]  # (n-dnum,4)
        n_fro = frozen.shape[0]
        # greedy rebuild from pool
        P = POOL
        # adapt pool size if little time left
        if time.time() + 8.0 > T_SURG:
            P = 16000
        pool = rng.standard_normal((P, 4))
        pool = pool / np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
        # max dot to frozen
        # chunk to avoid huge memory: P x n_fro via matmul is fine (32k x 240)
        try:
            fmax = np.max(pool @ frozen.T, axis=1)
        except Exception:
            fmax = np.max(pool @ frozen.T, axis=1)
        inserted = np.zeros((dnum, 4))
        # greedy loop
        for t in range(dnum):
            bi = int(np.argmin(fmax))
            b = pool[bi]
            inserted[t] = b
            # update fmax
            dn = pool @ b
            # in-place max
            np.maximum(fmax, dn, out=fmax)
        Xcand = np.vstack([frozen, inserted])
        Xcand = _normalize(Xcand)
        # local tangential optimisation of inserted only
        # two beta levels, few steps
        try:
            Xcand = _local_refine_ins(Xcand, n_fro, 80.0, 60, 0.03)
            Xcand = _local_refine_ins(Xcand, n_fro, 220.0, 60, 0.02)
        except Exception:
            pass
        # unfreeze: short global anneal
        try:
            Xcand = _global_refine(Xcand, short_stages)
        except Exception:
            pass
        v = _max_cos(Xcand)
        c = _count_above(Xcand, 0.8975)
        E = v + 2e-7 * c
        # decide LP polish: if promising (better than cur by max or much better count)
        do_lp = False
        if v < best_val + 1e-4 and (v < cur_val or c < cur_cnt):
            # only every other iter to save time, or if very good
            if (it % 2 == 0) or (v < best_val):
                do_lp = True
        if do_lp and time.time() + 2.5 < T_SURG:
            try:
                Xcand = _lp_polish(Xcand, min(time.time() + 1.2, T_SURG))
                v = _max_cos(Xcand)
                c = _count_above(Xcand, 0.8975)
                E = v + 2e-7 * c
            except Exception:
                pass
        # best update
        if v < best_val - 1e-12 or (abs(v - best_val) < 1e-12 and c < best_cnt):
            best = Xcand.copy()
            best_val = v
            best_cnt = c
            # also move cur there
            cur = Xcand.copy()
            cur_val = v
            cur_cnt = c
            cur_E = E
        else:
            # SA acceptance for cur
            if E < cur_E:
                cur = Xcand.copy()
                cur_val = v
                cur_cnt = c
                cur_E = E
            else:
                # probabilistic
                try:
                    p = float(np.exp(-(E - cur_E) / max(Temp, 1e-9)))
                except Exception:
                    p = 0.0
                if rng.random() < p:
                    cur = Xcand.copy()
                    cur_val = v
                    cur_cnt = c
                    cur_E = E
                # occasional restart from best if cur drifts too bad
                if v > best_val + 0.004 and rng.random() < 0.15:
                    cur = best.copy()
                    cur_val = best_val
                    cur_cnt = best_cnt
                    cur_E = best_val + 2e-7 * best_cnt
        it += 1

    # final polish on best with remaining time
    if time.time() < T_HARD - 2.0:
        try:
            best = _lp_polish(best, T_HARD - 1.0)
        except Exception:
            pass
    best = _normalize(best)
    return best
