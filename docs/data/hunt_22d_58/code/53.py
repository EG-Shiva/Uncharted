import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _refine_single(p, Y, steps=120, b0=80.0, b1=500.0, lr0=0.08, lr1=0.01):
    p = p / (np.linalg.norm(p) + 1e-12)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        dots = Y @ p
        m = float(dots.max())
        w = np.exp(beta * (dots - m))
        sw = float(w.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        w /= sw
        grad = w @ Y
        gt = grad - float(grad @ p) * p
        p = p - lr * gt
        nrm = float(np.linalg.norm(p))
        if not np.isfinite(nrm) or nrm == 0:
            break
        p = p / nrm
    return p


def _quench(X, steps=140, b0=60.0, b1=400.0, lr0=0.03, lr1=0.005):
    n, d = X.shape
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        X = X - lr * Gt
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr == 0] = 1.0
        X = X / nr
    return X


def _lp_polish_capped(X, budget_sec, deadline, cap=600, max_it=12, r0=5e-4):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= budget_sec:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        it += 1
        if it > max_it:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < cap:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= cap:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list, dtype=int)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 1.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr == 0] = 1.0
        y = y / nr
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y
            best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X


def solve():
    t0 = time.time()
    TRIAL_DEADLINE = t0 + 130.0
    FINAL_DEADLINE = t0 + 150.0
    HARD_DEADLINE = t0 + 168.0
    rng = np.random.default_rng(20261102)
    n, d = 58, 22

    incumbent = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
            if not np.all(np.isfinite(v)):
                continue
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                incumbent = v.copy()
        except Exception:
            continue
    if incumbent is None:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        incumbent = z
        best_val = _max_cos(incumbent)

    best = incumbent.copy()

    ks = [3, 4, 5, 6, 8]
    trial = 0
    # quick degrees cache updated when best improves
    while time.time() < TRIAL_DEADLINE and time.time() < HARD_DEADLINE - 25:
        trial += 1
        # choose k
        k = ks[rng.integers(0, len(ks))]
        mode = trial % 3  # 0 random, 1 hubs, 2 cluster
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m0 = float(G.max())
        except Exception:
            m0 = best_val
            G = None
        remove = None
        try:
            if mode == 0:
                remove = rng.choice(n, size=k, replace=False)
            elif mode == 1:
                # highest-contact hubs
                gap = 0.003
                if G is not None:
                    deg = (G > m0 - gap).sum(axis=1)
                else:
                    deg = np.zeros(n)
                # add tiny noise to break ties deterministically via rng
                order = np.argsort(-deg, kind="stable")
                # take top k, but randomly swap one sometimes for diversity
                remove = order[:k].copy()
                if rng.random() < 0.3:
                    remove[-1] = int(rng.integers(0, n))
                    remove = np.unique(remove)
                    while len(remove) < k:
                        extra = int(rng.integers(0, n))
                        if extra not in remove:
                            remove = np.append(remove, extra)
            else:
                # mutually close cluster around random seed
                seed = int(rng.integers(0, n))
                dots = best[seed] @ best.T
                dots[seed] = -2.0
                order = np.argsort(-dots)
                remove = np.concatenate([[seed], order[:k - 1]])
            remove = np.array(remove, dtype=int)
        except Exception:
            remove = rng.choice(n, size=k, replace=False)

        mask = np.ones(n, dtype=bool)
        mask[remove] = False
        # retained points
        Y = best[mask].copy()  # (n-k, d)

        # sequential reinsertion
        cur = Y
        ok = True
        for _ in range(k):
            if time.time() >= TRIAL_DEADLINE:
                ok = False
                break
            # sample at least 8192 candidates
            ncand = 8192
            try:
                C = rng.standard_normal((ncand, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
                dots = C @ cur.T
                worst = dots.max(axis=1)
                bi = int(np.argmin(worst))
                p0 = C[bi].copy()
            except Exception:
                ok = False
                break
            try:
                p0 = _refine_single(p0, cur, steps=120)
            except Exception:
                pass
            cur = np.vstack([cur, p0[np.newaxis, :]])
        if not ok:
            continue
        Xt = cur.copy()
        # short all-point soft-max quench
        try:
            Xt = _quench(Xt, steps=130, b0=60.0, b1=400.0, lr0=0.03, lr1=0.005)
        except Exception:
            continue
        try:
            vq = _max_cos(Xt)
        except Exception:
            continue
        # prune: LP cannot fix large gaps
        if not np.isfinite(vq):
            continue
        if vq > best_val + 0.002:
            continue
        # capped-pair LP polish with small budget
        if time.time() >= TRIAL_DEADLINE:
            break
        try:
            Xt2 = _lp_polish_capped(Xt, budget_sec=1.0, deadline=min(TRIAL_DEADLINE, HARD_DEADLINE - 22), cap=550, max_it=10, r0=5e-4)
        except Exception:
            continue
        try:
            v2 = _max_cos(Xt2)
        except Exception:
            continue
        if v2 < best_val - 1e-12:
            best = Xt2 / np.linalg.norm(Xt2, axis=1, keepdims=True)
            best_val = v2

    # reserve final ~20s for LP polishing of best
    try:
        remain = min(FINAL_DEADLINE - time.time(), HARD_DEADLINE - time.time() - 2.0)
        if remain > 1.0:
            best = _lp_polish_capped(best, budget_sec=remain, deadline=FINAL_DEADLINE, cap=700, max_it=30, r0=6e-4)
            best_val = _max_cos(best)
    except Exception:
        pass

    # fallback guarantee
    try:
        if _max_cos(best) > _max_cos(incumbent) + 1e-12:
            best = incumbent
    except Exception:
        best = incumbent
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
