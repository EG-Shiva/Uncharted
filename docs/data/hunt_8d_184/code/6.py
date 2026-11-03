import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _lp_polish(X, deadline):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = X @ X.T
        c = G[iu0, iu1]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3500:
            # keep top 3500
            part = np.argpartition(c, -3500)[-3500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[part] = True
            cnt = 3500
        I = iu0[sel]
        J = iu1[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = np.arange(d)
        cols = np.concatenate([
            (I[:, None] * d + cI).ravel(),
            (J[:, None] * d + cI).ravel(),
            np.full(p, n * d)], axis=0) if False else np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if np.any(~np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X


def _smooth_anneal(X, steps, T0, T1, lr0):
    X = X.copy()
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    mom = np.zeros_like(X)
    vel = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * np.exp(lT * frac)
        lr = lr0 * (1.0 - 0.7 * frac)
        G = X @ X.T
        gut = G[iu0, iu1]
        mx = np.max(gut)
        W = np.exp((gut - mx) / T)
        S = np.sum(W)
        if not np.isfinite(S) or S <= 0:
            break
        A = np.zeros((n, n), dtype=np.float64)
        A[iu0, iu1] = W / S
        As = A + A.T
        grad = As @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        mom = b1 * mom + (1.0 - b1) * gt
        vel = b2 * vel + (1.0 - b2) * (gt * gt)
        b1t = 1.0 - b1 ** (s + 1)
        b2t = 1.0 - b2 ** (s + 1)
        mh = mom / b1t
        vh = vel / b2t
        step = lr * mh / (np.sqrt(vh) + eps)
        # limit step norm per point
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        cap = 0.05
        scale = np.minimum(1.0, cap / (sn + 1e-12))
        step = step * scale
        X = X - step
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _refine_hole(v0, Xkeep, steps, T0, T1, lr0):
    v = v0.copy()
    v = v / np.linalg.norm(v)
    d = v.shape[0]
    mom = np.zeros(d)
    vel = np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT = np.log(T1 / T0)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * np.exp(lT * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        dots = Xkeep @ v
        mx = np.max(dots)
        w = np.exp((dots - mx) / T)
        S = np.sum(w)
        p = w / S
        grad = p @ Xkeep
        gt = grad - float(grad @ v) * v
        mom = b1 * mom + (1.0 - b1) * gt
        vel = b2 * vel + (1.0 - b2) * (gt * gt)
        mh = mom / (1.0 - b1 ** (s + 1))
        vh = vel / (1.0 - b2 ** (s + 1))
        step = lr * mh / (np.sqrt(vh) + eps)
        nrm = float(np.linalg.norm(step))
        if nrm > 0.1:
            step = step * (0.1 / nrm)
        v = v - step
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n < 1e-12:
            break
        v = v / n
    return v


def _propose_hole(Xkeep, rng, n_try=15000):
    d = Xkeep.shape[1]
    C = rng.randn(n_try, d)
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    scores = C @ Xkeep.T
    mx = np.max(scores, axis=1)
    k = min(8, n_try)
    idx = np.argpartition(mx, k)[:k]
    best_v = None
    best_s = 1e9
    for j in idx:
        v0 = C[j]
        v = _refine_hole(v0, Xkeep, steps=180, T0=0.02, T1=0.003, lr0=0.05)
        s = float(np.max(Xkeep @ v))
        if s < best_s:
            best_s = s
            best_v = v
    return best_v


def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.RandomState(7)
    np.random.seed(7)
    Xbest = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            X = np.load(fn)
            X = np.asarray(X, dtype=np.float64)
            if X.shape != (184, 8):
                continue
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            if not np.all(np.isfinite(X)):
                continue
            v = _max_cos(X)
            if v < best:
                best = v
                Xbest = X.copy()
        except Exception:
            continue
    if Xbest is None:
        Xbest = rng.randn(184, 8)
        Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
        best = _max_cos(Xbest)
    n, d = 184, 8
    iu0, iu1 = np.triu_indices(n, 1)
    trial = 0
    n_noimp = 0
    # main basin hopping, reserve ~28s for final polish
    while time.time() - t0 < BUDGET - 28.0:
        trial += 1
        mode = trial % 5
        Xcur = Xbest.copy()
        try:
            G = Xcur @ Xcur.T
            np.fill_diagonal(G, -2.0)
            worst_pp = np.max(G, axis=1)
            order_iso = np.argsort(worst_pp)  # small first (rattlers)
            order_bad = order_iso[::-1]  # large first
        except Exception:
            order_iso = np.arange(n)
            order_bad = np.arange(n)
        if mode == 0:
            # rattler relocation (1-3 points)
            k = int(rng.choice([1, 2, 3]))
            if n_noimp > 10:
                k = 3
            cand = order_iso[:max(20, 5 * k)]
            idx_rm = rng.choice(cand, size=k, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[idx_rm] = False
            cur_keep = Xcur[mask]
            new_pts = []
            for _ in range(k):
                v = _propose_hole(cur_keep, rng)
                new_pts.append(v)
                cur_keep = np.vstack([cur_keep, v[None, :]])
            for ii, vv in zip(idx_rm, new_pts):
                Xcur[ii] = vv
        elif mode == 1:
            # worst-pair surgery
            G2 = Xcur @ Xcur.T
            np.fill_diagonal(G2, -2.0)
            flat = G2[iu0, iu1]
            topk = 12
            idx = np.argpartition(flat, -topk)[-topk:]
            pick = idx[rng.randint(topk)]
            i, j = int(iu0[pick]), int(iu1[pick])
            victim = i if rng.rand() < 0.5 else j
            mask = np.ones(n, dtype=bool)
            mask[victim] = False
            cur_keep = Xcur[mask]
            v = _propose_hole(cur_keep, rng)
            Xcur[victim] = v
            # small shake of partner
            Xcur[i if victim == j else j] += 0.01 * rng.randn(d)
            Xcur = Xcur / np.linalg.norm(Xcur, axis=1, keepdims=True)
        elif mode == 2:
            # shake overloaded hubs + neighbors
            cc_thr = best - 0.008 if np.isfinite(best) else 0.49
            # contact counts approx by threshold
            G3 = Xcur @ Xcur.T
            np.fill_diagonal(G3, -2.0)
            cmax = float(np.max(G3))
            contacts = np.sum(G3 > cmax - 0.008, axis=1)
            hub_order = np.argsort(-contacts)
            nhub = int(rng.randint(2, 5))
            hubs = hub_order[:nhub]
            # neighbors of hubs
            toshake = set(hubs.tolist())
            for h in hubs:
                nbrs = np.argsort(-G3[h])[:6]
                for q in nbrs:
                    toshake.add(int(q))
            # also add random low-contact points
            for q in order_iso[:4]:
                if rng.rand() < 0.5:
                    toshake.add(int(q))
            sigma = 0.06 + 0.04 * rng.rand()
            if n_noimp > 8:
                sigma = 0.12
            for q in toshake:
                Xcur[q] = Xcur[q] + sigma * rng.randn(d)
            Xcur = Xcur / np.linalg.norm(Xcur, axis=1, keepdims=True)
        elif mode == 3:
            # global small shake
            sigma = 0.015 + 0.015 * rng.rand()
            if n_noimp > 8:
                sigma = 0.04
            Xcur = Xcur + sigma * rng.randn(n, d)
            Xcur = Xcur / np.linalg.norm(Xcur, axis=1, keepdims=True)
        else:
            # random subset reinit to holes
            k = int(rng.randint(4, 9))
            idx_rm = rng.choice(n, size=k, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[idx_rm] = False
            cur_keep = Xcur[mask]
            new_pts = []
            for _ in range(k):
                v = _propose_hole(cur_keep, rng)
                new_pts.append(v)
                cur_keep = np.vstack([cur_keep, v[None, :]])
            for ii, vv in zip(idx_rm, new_pts):
                Xcur[ii] = vv
        # smooth anneal
        try:
            Xsm = _smooth_anneal(Xcur, steps=650, T0=0.025, T1=0.003, lr0=0.015)
        except Exception:
            n_noimp += 1
            continue
        try:
            v_sm = _max_cos(Xsm)
        except Exception:
            n_noimp += 1
            continue
        improved = False
        if v_sm < best:
            # promising: LP polish immediately
            deadline = min(time.time() + 5.0, t0 + BUDGET - 28.0)
            try:
                Xlp = _lp_polish(Xsm, deadline)
                v_lp = _max_cos(Xlp)
            except Exception:
                Xlp, v_lp = Xsm, v_sm
            if v_lp < best:
                Xbest, best = Xlp, v_lp
                improved = True
            else:
                Xbest, best = Xsm, v_sm
                improved = True
        elif v_sm < best + 0.002 and (time.time() - t0 < BUDGET - 40.0):
            # close: give short LP chance every other trial
            if trial % 2 == 0:
                deadline = min(time.time() + 3.0, t0 + BUDGET - 28.0)
                try:
                    Xlp = _lp_polish(Xsm, deadline)
                    v_lp = _max_cos(Xlp)
                except Exception:
                    Xlp, v_lp = Xsm, v_sm
                if v_lp < best:
                    Xbest, best = Xlp, v_lp
                    improved = True
        if improved:
            n_noimp = 0
        else:
            n_noimp += 1
        if time.time() - t0 >= BUDGET - 28.0:
            break
    # final polish with remaining time
    try:
        deadline = t0 + BUDGET - 1.5
        Xbest = _lp_polish(Xbest, deadline)
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
