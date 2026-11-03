import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _global_lse_adam(X, steps, a0, a1, lr, deadline=None):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lr_ratio = (a1 / a0) if a0 > 0 else 1.0
    for t in range(steps):
        if deadline is not None and (t % 100 == 0) and time.time() > deadline:
            break
        frac = t / max(steps - 1, 1)
        alpha = a0 * (lr_ratio ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        Grad = (E @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        if not np.all(np.isfinite(Gt)):
            break
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        bc1 = 1 - b1 ** (t + 1)
        bc2 = 1 - b2 ** (t + 1)
        mhat = m / bc1
        vhat = v / bc2
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        X = X - lr * step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        X = X / nr
    return X

def _refine_single(x, cur, steps=200, alpha=150.0, lr=0.01):
    # x shape (9,), cur shape (m,9), both normalized
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        d = cur @ x  # (m,)
        M = float(np.max(d))
        E = np.exp(alpha * (d - M))
        S = float(np.sum(E))
        if not np.isfinite(S) or S <= 0:
            break
        grad = (E @ cur) / S  # (9,)
        gt = grad - float(np.dot(grad, x)) * x
        if not np.all(np.isfinite(gt)):
            break
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * (gt * gt)
        mhat = m1 / (1 - b1 ** (t + 1))
        vhat = v1 / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        step = np.clip(step, -3.0, 3.0)
        x = x - lr * step
        n = float(np.linalg.norm(x))
        if not np.isfinite(n) or n < 1e-12:
            break
        x = x / n
    return x

def _lp_polish_deadline(x, deadline, r_init=0.004, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
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
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 175.0
    TRIAL_END = t0 + 140.0
    n, d = 176, 9
    rng = np.random.default_rng(0)
    # load best
    Xbest = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    Xbest = a / nr
                    break
        except Exception:
            continue
    if Xbest is None:
        Xbest = rng.standard_normal((n, d))
        Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    best_v = _max_cos(Xbest)
    init_v = float(best_v)
    THRESH = 0.4611825753
    # contact info for hub selection
    def get_hub(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        cnts = np.sum(G > m - 1e-9, axis=1)
        hub = int(np.argmax(cnts))
        return hub, G, m, cnts
    # trial plan: (k, mode) mode 0=random,1=hub+neighbourhood
    plan = [(8, 0), (8, 1), (12, 0), (12, 1), (16, 1), (16, 0), (8, 1), (12, 1), (8, 0), (16, 1)]
    trial_idx = 0
    best_trial_X = None
    NCAND = 50000
    for (k, mode) in plan:
        if time.time() > TRIAL_END:
            break
        # choose deletion set (deterministic via rng sequence)
        try:
            X0 = Xbest.copy()
            if mode == 1:
                hub, G0, m0, cnts = get_hub(X0)
                # order neighbours by cosine descending
                row = G0[hub].copy()
                row[hub] = -2.0
                order = np.argsort(-row)
                # take hub + top k-1 neighbours; mix a few random to diversify by trial
                if trial_idx % 2 == 1 and k > 8:
                    # replace last 2 with random non-neighbour to diversify
                    delset = np.concatenate([[hub], order[:k-1]])
                    # swap
                    extra = rng.choice(n, size=2, replace=False)
                    delset[-2:] = extra
                    delset = np.unique(delset)
                    while len(delset) < k:
                        e = int(rng.integers(0, n))
                        if e not in delset:
                            delset = np.append(delset, e)
                    delset = delset[:k]
                else:
                    delset = np.concatenate([[hub], order[:k-1]])
            else:
                delset = rng.choice(n, size=k, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[delset] = False
            cur = X0[mask].copy()  # (n-k,9)
            # Adam-polish remainder 800 steps alpha 50->300 lr 0.003
            cur = _global_lse_adam(cur, steps=800, a0=50.0, a1=300.0, lr=0.003, deadline=TRIAL_END)
            if time.time() > TRIAL_END:
                break
            # sequential reinsertion
            ok = True
            for ins in range(k):
                if time.time() > TRIAL_END:
                    ok = False
                    break
                mcur = cur.shape[0]
                # vectorised candidates
                C = rng.standard_normal((NCAND, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                # score against frozen set
                D = C @ cur.T  # (NCAND, mcur)
                mm = D.max(axis=1)
                pick = int(np.argmin(mm))
                xnew = C[pick].copy()
                # local refine alone 200 steps
                xnew = _refine_single(xnew, cur, steps=200, alpha=150.0, lr=0.01)
                cur = np.vstack([cur, xnew[None, :]])
                # only 300 global LogSumExp steps per insertion
                cur = _global_lse_adam(cur, steps=300, a0=50.0, a1=300.0, lr=0.003, deadline=TRIAL_END)
                if not np.all(np.isfinite(cur)):
                    ok = False
                    break
            if not ok:
                trial_idx += 1
                continue
            # completed insertion sequence
            v = _max_cos(cur)
            if v < best_v and cur.shape == (n, d):
                best_v = float(v)
                best_trial_X = cur.copy()
        except Exception:
            pass
        trial_idx += 1
    # if no completed sequence beats threshold, return loaded best immediately
    if best_trial_X is None or best_v >= THRESH and best_v >= init_v:
        # No improvement: return loaded best (ensure normalized)
        Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
        return Xbest
    # we have an improved trial; optionally one 20s LP polish
    Xret = best_trial_X if best_trial_X is not None else Xbest
    now = time.time()
    remain = HARD_END - now - 2.0
    if remain > 1.0:
        lp_time = min(20.0, remain)
        deadline = time.time() + lp_time
        try:
            Xlp = _lp_polish_deadline(Xret, deadline)
            vlp = _max_cos(Xlp)
            if vlp < best_v:
                Xret = Xlp
                best_v = vlp
        except Exception:
            pass
    Xret = Xret / np.linalg.norm(Xret, axis=1, keepdims=True)
    # safety: if somehow worse than init, return init
    if _max_cos(Xret) > init_v and init_v < THRESH + 1e-12:
        return Xbest
    return Xret
