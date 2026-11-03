import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def lse_polish(X, steps, a0, a1, lr, deadline=None):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        if deadline is not None and (t % 100 == 0) and time.time() > deadline:
            break
        frac = t / max(1, steps - 1)
        alpha = a0 * ((a1 / a0) ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        lrt = lr * (1.0 - 0.5 * frac)
        X = X - lrt * step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        X = X / nr
    return X

def refine_one(v, frozen, steps=200, alpha=200.0, lr=0.01):
    m = np.zeros(9)
    vv = np.zeros(9)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        dots = frozen @ v
        M = float(np.max(dots))
        w = np.exp(alpha * (dots - M))
        s = float(w.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        grad = w @ frozen
        gt = grad - float(np.dot(grad, v)) * v
        m = b1 * m + (1 - b1) * gt
        vv = b2 * vv + (1 - b2) * (gt * gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        v = v - lr * step
        nrm = float(np.linalg.norm(v))
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        v = v / nrm
    return v

def screen_best(cur, rng, total=50000, batch=10000):
    best_v = None
    best_s = 1e9
    nrep = total // batch
    for _ in range(nrep):
        T = rng.standard_normal((batch, 9))
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        D = T @ cur.T
        s = D.max(axis=1)
        i = int(np.argmin(s))
        if float(s[i]) < best_s:
            best_s = float(s[i])
            best_v = T[i].copy()
    return best_v, best_s

def lp_polish_deadline(x, deadline, r_init=0.004, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx[order]] = True
            sel = keep
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols[:, 2 * d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    LOOP_END = t0 + 140.0
    HARD_END = t0 + 174.0
    THRESH = 0.4611825753
    rng_global = np.random.default_rng(0)
    n, d = 176, 9
    # load best
    Xb = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    Xb = (a / nr).astype(float)
                    break
        except Exception:
            continue
    if Xb is None:
        Xb = rng_global.standard_normal((n, d))
        Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    base_v = max_cos(Xb)
    best_X = Xb.copy()
    best_v = base_v
    # contact counts for hub choice
    try:
        G0 = Xb @ Xb.T
        np.fill_diagonal(G0, -2.0)
        mc0 = float(G0.max())
        contacts0 = (G0 > mc0 - 1e-9).sum(axis=1)
        hub0 = int(np.argmax(contacts0))
        hub_order = np.argsort(-G0[hub0])
    except Exception:
        hub0 = 0
        hub_order = np.arange(n)

    # trial plan: (k, hub_flag, seed)
    plan = []
    seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    ks = [12, 8, 16, 12, 8, 16, 12, 8]
    hubflags = [True, False, True, False, True, False, True, False]
    for s, k, h in zip(seeds, ks, hubflags):
        plan.append((k, h, s))

    trial_best_X = None
    trial_best_v = 1e9

    for (k, use_hub, seed) in plan:
        if time.time() > LOOP_END:
            break
        rng = np.random.default_rng(seed)
        # choose delete set
        if use_hub:
            # one hub + part of its neighbourhood
            del_idx = np.concatenate([[hub0], hub_order[:k-1]])
            # add slight randomness: replace last 2 with random if k>8
            if k >= 12:
                extra = rng.choice(n, size=2, replace=False)
                del_idx[-2:] = extra
            del_idx = np.unique(del_idx)
            # pad if unique shrank
            while len(del_idx) < k:
                c = rng.integers(0, n)
                if c not in del_idx:
                    del_idx = np.append(del_idx, c)
            del_idx = del_idx[:k]
        else:
            del_idx = rng.choice(n, size=k, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[del_idx] = False
        cur = Xb[mask].copy()
        # Adam-polish remaining 176-k with LogSumExp 50->300, lr 0.003, 800 steps
        cur = lse_polish(cur, steps=800, a0=50.0, a1=300.0, lr=0.003, deadline=LOOP_END)
        if time.time() > LOOP_END:
            break
        # quick check: if remaining max already too high, continue (should be well below 0.46)
        # reinsert one at a time
        ok = True
        for _ in range(k):
            if time.time() > LOOP_END:
                ok = False
                break
            bv, _ = screen_best(cur, rng, total=50000, batch=10000)
            if bv is None:
                ok = False
                break
            bv = refine_one(bv, cur, steps=200, alpha=200.0, lr=0.01)
            cur = np.vstack([cur, bv[None, :]])
            # only 300 global LogSumExp steps per insertion
            cur = lse_polish(cur, steps=300, a0=80.0, a1=250.0, lr=0.003, deadline=LOOP_END)
        if not ok:
            continue
        # completed insertion sequence
        v = max_cos(cur)
        if v < trial_best_v:
            trial_best_v = float(v)
            trial_best_X = cur.copy()
        if v < best_v:
            best_v = float(v)
            best_X = cur.copy()
        # early exit if time low
        if time.time() > LOOP_END:
            break

    # decision: if no completed sequence beats THRESH, return loaded best immediately
    if trial_best_X is None or trial_best_v >= THRESH:
        # also consider best_X if it beats base but not thresh -> per instruction return loaded best
        # return loaded best to guarantee validity
        if best_v < base_v and best_v < THRESH:
            pass  # fall through to LP? instruction says return loaded best if no seq beats thresh
            # enforce instruction: return loaded best
            return Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
        return Xb / np.linalg.norm(Xb, axis=1, keepdims=True)

    # one 20s deadline-checked LP polish on the best trial result
    lp_deadline = min(HARD_END - 2.0, time.time() + 20.0)
    # only polish the overall best (trial_best_X if it is best)
    to_polish = trial_best_X
    # if best_X is even better, polish it
    if best_X is not trial_best_X and best_v < trial_best_v:
        to_polish = best_X
    try:
        polished = lp_polish_deadline(to_polish, deadline=lp_deadline)
        pv = max_cos(polished)
        if pv < best_v:
            best_X, best_v = polished, pv
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    # safety: if somehow worse than loaded best, return loaded best
    if max_cos(best_X) >= base_v and base_v < THRESH + 1e-12:
        # keep improvement only; otherwise return base (which is record-beating)
        # but if we did beat thresh keep it
        if max_cos(best_X) >= THRESH:
            return Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    return best_X
