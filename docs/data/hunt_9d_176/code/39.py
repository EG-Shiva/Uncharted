import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def lse_polish(X, steps, a0=50.0, a1=300.0, lr=0.003):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        frac = t / max(1, steps - 1)
        alpha = a0 * ((a1 / a0) ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E) * 0.5)
        if not np.isfinite(S) or S <= 0:
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
        X = X - lr * step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    nr = np.linalg.norm(X, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    return X / nr

def refine_single(v0, F, steps=200, alpha=200.0, lr=0.01):
    v = v0.copy() / np.linalg.norm(v0)
    d = v.shape[0]
    m = np.zeros(d)
    vv = np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        dots = F @ v
        M = float(np.max(dots))
        w = np.exp(alpha * (dots - M))
        S = float(np.sum(w))
        if not np.isfinite(S) or S <= 0:
            break
        grad = (w @ F) / S
        gt = grad - float(np.dot(grad, v)) * v
        m = b1 * m + (1 - b1) * gt
        vv = b2 * vv + (1 - b2) * (gt * gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        step = np.clip(step, -3.0, 3.0)
        v = v - lr * step
        nrm = float(np.linalg.norm(v))
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        v = v / nrm
    nrm = float(np.linalg.norm(v))
    if not np.isfinite(nrm) or nrm < 1e-12:
        return v0 / np.linalg.norm(v0)
    return v / nrm

def lp_polish_deadline(x, deadline, r_init=0.003, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() >= deadline:
            break
        G = x @ x.T
        c_all = G[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx_all = np.where(sel)[0]
            order = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[order]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = cnt
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
        if time.time() >= deadline:
            break
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
    HARD_END = t0 + 176.0
    TRIAL_END = t0 + 140.0
    n, d = 176, 9
    rng = np.random.default_rng(0)
    base = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    base = (a / nr).copy()
                    break
        except Exception:
            continue
    if base is None:
        base = rng.standard_normal((n, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
    base_v = max_cos(base)
    best_X = base.copy()
    best_v = float(base_v)
    THRESH = 0.4611825753

    # precompute contacts for hub deletions
    trial = 0
    ks = [8, 12, 16]
    NCAND = 50000
    while time.time() < TRIAL_END:
        trial += 1
        if time.time() > TRIAL_END - 2.0:
            break
        k = ks[(trial - 1) % len(ks)]
        # alternate random vs hub neighbourhood
        use_hub = (trial % 2 == 0)
        del_idx = None
        try:
            if use_hub:
                G0 = base @ base.T
                np.fill_diagonal(G0, -2.0)
                mv = float(G0.max())
                contacts = np.sum(G0 > mv - 1e-7, axis=1)
                # hub: max contacts (require 26+ else argmax)
                hub = int(np.argmax(contacts))
                order = np.argsort(-G0[hub])
                # order includes self? self is -2 so last; take top k-1
                neigh = [j for j in order if j != hub][:max(0, k - 1)]
                del_idx = np.array([hub] + list(neigh), dtype=int)
                if len(del_idx) < k:
                    extra = rng.choice([i for i in range(n) if i not in set(del_idx.tolist())], size=k - len(del_idx), replace=False)
                    del_idx = np.concatenate([del_idx, extra])
            else:
                del_idx = rng.choice(n, size=k, replace=False)
        except Exception:
            del_idx = rng.choice(n, size=k, replace=False)
        mask = np.ones(n, dtype=bool)
        mask[del_idx] = False
        try:
            cur = base[mask].copy()
        except Exception:
            continue
        # Adam-polish remainder 800 steps LogSumExp 50->300 lr 0.003
        try:
            cur = lse_polish(cur, steps=800, a0=50.0, a1=300.0, lr=0.003)
        except Exception:
            continue
        if time.time() >= TRIAL_END:
            break
        # insertion sequence
        ok = True
        for ins in range(k):
            if time.time() >= TRIAL_END:
                ok = False
                break
            # best of >=50k candidates scored against frozen set
            try:
                # two chunks to bound memory
                best_dot = np.inf
                best_cand = None
                for _ in range(2):
                    R = rng.standard_normal((NCAND // 2, d))
                    R /= np.linalg.norm(R, axis=1, keepdims=True)
                    dots = R @ cur.T
                    worst = dots.max(axis=1)
                    pi = int(np.argmin(worst))
                    if float(worst[pi]) < best_dot:
                        best_dot = float(worst[pi])
                        best_cand = R[pi].copy()
                v0 = best_cand
            except Exception:
                ok = False
                break
            try:
                v0 = refine_single(v0, cur, steps=200, alpha=200.0, lr=0.01)
            except Exception:
                pass
            cur = np.vstack([cur, v0[None, :]])
            try:
                cur = lse_polish(cur, steps=300, a0=50.0, a1=300.0, lr=0.003)
            except Exception:
                pass
        if not ok:
            break
        try:
            v = max_cos(cur)
        except Exception:
            continue
        if v < best_v and np.all(np.isfinite(cur)):
            best_v = float(v)
            best_X = (cur / np.linalg.norm(cur, axis=1, keepdims=True)).copy()
            # keep base as best for further deletions? use best_X as new base
            base = best_X.copy()
        # early stop if time low
        if time.time() >= TRIAL_END:
            break

    # return loaded best immediately if no completed sequence beats threshold
    if best_v >= THRESH - 1e-12:
        # check if we actually improved over loaded value; if not return original
        try:
            orig = np.load("best.npy")
            orig = orig / np.linalg.norm(orig, axis=1, keepdims=True)
            return orig
        except Exception:
            return best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    # at most one 20s deadline-checked LP polish
    try:
        deadline = min(time.time() + 20.0, HARD_END - 1.0)
        if time.time() < deadline - 0.5:
            best_X = lp_polish_deadline(best_X, deadline, r_init=0.002)
            best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    except Exception:
        pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
