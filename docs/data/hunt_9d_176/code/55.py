import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def lse_polish(x, steps, a0=50.0, a1=300.0, lr=0.003):
    n, d = x.shape
    m = np.zeros_like(x)
    vv = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    la0 = np.log(a0); la1 = np.log(a1)
    for t in range(steps):
        frac = t / max(steps - 1, 1)
        alpha = float(np.exp(la0 + (la1 - la0) * frac))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        lrt = lr * (1.0 - 0.5 * frac)
        x = x - lrt * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if not np.all(np.isfinite(x)):
            break
    nr = np.linalg.norm(x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    return x / nr

def refine_single(v, Y, steps=200, alpha=200.0, lr=0.005):
    # tangent Adam on sphere for one point vs frozen set Y (m,9)
    m1 = np.zeros(9); vv = np.zeros(9)
    b1, b2, eps = 0.9, 0.999, 1e-8
    v = v / (np.linalg.norm(v) + 1e-30)
    for t in range(steps):
        d = Y @ v
        M = float(np.max(d))
        w = np.exp(alpha * (d - M))
        S = float(np.sum(w))
        if S <= 0 or not np.isfinite(S):
            break
        g = (w @ Y) / S
        g = g - float(np.dot(g, v)) * v
        m1 = b1 * m1 + (1 - b1) * g
        vv = b2 * vv + (1 - b2) * (g * g)
        mh = m1 / (1 - b1 ** (t + 1))
        vh = vv / (1 - b2 ** (t + 1))
        st = mh / (np.sqrt(vh) + eps)
        st = np.clip(st, -3.0, 3.0)
        v = v - lr * st
        nrm = float(np.linalg.norm(v))
        if nrm == 0 or not np.isfinite(nrm):
            break
        v /= nrm
    return v / (np.linalg.norm(v) + 1e-30)

def best_candidate(cur, rng, total=50000, batch=10000):
    d = cur.shape[1]
    best_v = None
    best_s = 2.0
    nb = total // batch
    for _ in range(nb):
        T = rng.standard_normal((batch, d))
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        # score = max dot vs cur
        D = T @ cur.T
        mm = D.max(axis=1)
        j = int(np.argmin(mm))
        if float(mm[j]) < best_s:
            best_s = float(mm[j])
            best_v = T[j].copy()
    return best_v

def lp_polish_deadline(X, deadline, radius_init=1e-3, cap=3000):
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline - 0.6 and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu]
        mval = float(c_all.max())
        thresh = mval - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx_all = np.where(sel)[0]
            order = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[order]] = True
            sel = keep
            cnt = cap
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((176, 9))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (176, 9):
        xb = rng.standard_normal((176, 9))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    best_x = xb.copy()
    try:
        xr = np.load("record.npy")
        xr = np.asarray(xr, dtype=float)
        if xr.shape == (176, 9):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            vr = max_cos(xr)
            if vr < best_val:
                best_val, best_x = vr, xr.copy()
    except Exception:
        pass

    THRESH = 0.4611825753
    trial_best_val = best_val
    trial_best_x = None

    TRIAL_END = t0 + 140.0
    HARD_END = t0 + 165.0

    ks = [8, 12, 16]
    trial = 0
    # precompute contacts of loaded best for hub selection
    while time.time() < TRIAL_END and time.time() < HARD_END - 25.0:
        k = ks[trial % 3]
        hub_mode = (trial % 2 == 1)
        # choose deletion set
        if not hub_mode:
            del_idx = rng.choice(176, size=k, replace=False)
        else:
            G = best_x @ best_x.T
            np.fill_diagonal(G, -2.0)
            M = float(np.max(G))
            C = (G > M - 1e-9)
            deg = C.sum(axis=1)
            hub = int(np.argmax(deg))
            order = np.argsort(-G[hub])
            # order[0] is hub itself? G[hub,hub]=-2 so not first; ensure hub included
            neigh = [hub]
            for cand in order:
                if cand == hub:
                    continue
                neigh.append(int(cand))
                if len(neigh) >= k:
                    break
            del_idx = np.array(neigh[:k], dtype=int)
        mask = np.ones(176, dtype=bool)
        mask[del_idx] = False
        rem = best_x[mask].copy()  # (176-k,9)
        # Adam-polish remaining with LogSumExp 50->300, lr 0.003, 800 steps
        rem = lse_polish(rem, steps=800, a0=50.0, a1=300.0, lr=0.003)
        if time.time() > TRIAL_END:
            break
        cur = rem
        aborted = False
        for s in range(k):
            if time.time() > TRIAL_END:
                aborted = True
                break
            v0 = best_candidate(cur, rng, total=50000, batch=10000)
            if v0 is None:
                aborted = True
                break
            v0 = refine_single(v0, cur, steps=200, alpha=200.0, lr=0.005)
            cur = np.vstack([cur, v0[None, :]])
            # only 300 global LogSumExp steps per insertion, no LP
            cur = lse_polish(cur, steps=300, a0=50.0, a1=300.0, lr=0.003)
        if aborted:
            trial += 1
            continue
        v = max_cos(cur)
        if v < trial_best_val:
            trial_best_val = v
            trial_best_x = cur.copy()
            # early exit if clearly better? continue to use time
        trial += 1
        if trial > 60:
            break
        if time.time() > TRIAL_END:
            break

    if trial_best_x is not None and trial_best_val < THRESH:
        # one 20s deadline-checked LP polish
        deadline = min(time.time() + 20.0, t0 + 172.0)
        try:
            pol = lp_polish_deadline(trial_best_x, deadline, radius_init=1e-3)
            vp = max_cos(pol)
            if vp < trial_best_val:
                trial_best_val = vp
                trial_best_x = pol
        except Exception:
            pass
        trial_best_x = trial_best_x / np.linalg.norm(trial_best_x, axis=1, keepdims=True)
        return trial_best_x
    else:
        # return loaded best immediately, no LP inside loop was called
        best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
        return best_x
