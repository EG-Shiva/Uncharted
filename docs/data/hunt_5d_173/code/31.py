import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contact_counts(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G >= m - tol
    cnt = mask.sum(axis=1)
    return m, cnt, G

def _lp_polish(X, seconds, seed_radius=1e-3):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    if seconds <= 0.05:
        return X
    radius = seed_radius
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        cfull = G[iu0]
        m = float(cfull.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = cfull > thresh
        ns = int(sel.sum())
        if ns == 0:
            radius *= 0.5
            continue
        if ns > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(cfull[idx])[-2500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = cfull[sel]
        p = len(I)
        nv = n * d + 1
        cols = np.concatenate([(I[:, None] * d + np.arange(d)),
                               (J[:, None] * d + np.arange(d)),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def _smooth_repel(X, steps, beta0, beta1, lr0):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    for s in range(steps):
        beta = beta0 + (beta1 - beta0) * (s / max(1, steps - 1))
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        m = float(G.max())
        # weights, avoid overflow
        E = (G - m) * beta
        # clip very negative to zero
        E = np.clip(E, -30.0, 0.0)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dots = (grad * X).sum(axis=1, keepdims=True)
        T = grad - dots * X
        tn = float(np.linalg.norm(T))
        if tn < 1e-12 or not np.isfinite(tn):
            break
        # scale so max displacement ~ lr0
        mx = float(np.sqrt((T * T).sum(axis=1).max()))
        if mx < 1e-12:
            break
        step = lr0 / (mx + 1e-12)
        # decay slightly
        step *= (1.0 - 0.4 * s / steps)
        X = X - step * T
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _find_holes(X, rng, n_sample=12000, n_holes=4):
    n, d = X.shape
    S = rng.normal(size=(n_sample, d))
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    # max dot to code
    Dm = S @ X.T
    cover = Dm.max(axis=1)
    order = np.argsort(cover)
    cands = S[order[:max(20, n_holes * 5)]]
    holes = []
    for u0 in cands:
        u = u0.copy()
        for _ in range(80):
            dots = X @ u
            j = int(np.argmax(dots))
            g = X[j]
            t = g - float(g @ u) * u
            tn = float(np.linalg.norm(t))
            if tn < 1e-12:
                break
            u = u - 0.15 * t
            u /= np.linalg.norm(u)
        holes.append(u)
        if len(holes) >= n_holes + 2:
            break
    holes = np.array(holes)
    # dedup: keep diverse
    kept = []
    for h in holes:
        ok = True
        for k in kept:
            if float(h @ k) > 0.96:
                ok = False
                break
        if ok:
            kept.append(h)
        if len(kept) >= n_holes:
            break
    if len(kept) == 0:
        kept = [holes[0]]
    return np.array(kept)

def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    rng = np.random.default_rng(0)
    n, d = 173, 5
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (n, d):
                an = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(an)):
                    v = _max_cos(an)
                    if v < best_val:
                        best_val = v
                        best = an.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    trial = 0
    # quick initial LP to ensure local opt (very short, already opt)
    while True:
        rem = deadline - time.time()
        if rem < 9.0:
            break
        pt = 1.1
        if rem < 25:
            pt = max(0.5, rem - 8.0)
        pt = min(pt, rem - 8.0)
        if pt < 0.45:
            break
        # contact info of explorer
        try:
            m_c, cnt_c, Gc = _contact_counts(cur, tol=1e-7)
        except Exception:
            cnt_c = np.zeros(n)
            m_c = cur_val
        order_desc = np.argsort(-cnt_c)
        order_asc = np.argsort(cnt_c)
        # find holes from cur (cheap every few trials, else reuse best holes)
        try:
            if trial % 2 == 0:
                holes = _find_holes(cur, rng, n_sample=8000, n_holes=4)
            else:
                holes = _find_holes(best, rng, n_sample=8000, n_holes=4)
        except Exception:
            holes = cur[rng.choice(n, 4, replace=False)]
        mode = trial % 7
        y = cur.copy()
        try:
            if mode == 0:
                # rattler -> deepest hole
                victim = int(order_asc[0])
                h = holes[0] + 1e-3 * rng.normal(size=(d,))
                h /= np.linalg.norm(h)
                y[victim] = h
                # small shake of overloaded neighbours
                for k in order_desc[:2]:
                    y[int(k)] += 0.01 * rng.normal(size=(d,))
            elif mode == 1:
                # overload -> hole
                victim = int(order_desc[0])
                h = holes[int(rng.integers(len(holes)))] + 5e-4 * rng.normal(size=(d,))
                h /= np.linalg.norm(h)
                y[victim] = h
                y[int(order_asc[0])] += 0.01 * rng.normal(size=(d,))
            elif mode == 2:
                # dense patch shake
                k = 6
                victims = order_desc[:k]
                # neighbours of top victim
                Gtmp = cur @ cur.T
                np.fill_diagonal(Gtmp, -2.0)
                j0 = int(order_desc[0])
                neigh = np.argsort(-Gtmp[j0])[:8]
                victims = np.unique(np.concatenate([victims, neigh]))
                y[victims] += 0.08 * rng.normal(size=(len(victims), d))
            elif mode == 3:
                # multi reinsert: 3 lowest -> holes
                nv = 3
                victims = order_asc[:nv]
                for ii, vv in enumerate(victims):
                    h = holes[ii % len(holes)] + 2e-3 * rng.normal(size=(d,))
                    h /= np.linalg.norm(h)
                    y[int(vv)] = h
                # shake top a bit
                y[int(order_desc[0])] += 0.02 * rng.normal(size=(d,))
            elif mode == 4:
                # two overloads -> two holes
                v1 = int(order_desc[0]); v2 = int(order_desc[1])
                h1 = holes[0] + 1e-3 * rng.normal(size=(d,)); h1 /= np.linalg.norm(h1)
                h2 = holes[1 % len(holes)] + 1e-3 * rng.normal(size=(d,)); h2 /= np.linalg.norm(h2)
                y[v1] = h1; y[v2] = h2
            elif mode == 5:
                # global small shake biased to high degree
                y += 0.012 * rng.normal(size=(n, d))
                y[order_desc[:8]] += 0.03 * rng.normal(size=(8, d))
            else:
                # random subset medium kick
                sub = rng.choice(n, 10, replace=False)
                y[sub] += 0.06 * rng.normal(size=(10, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            # equilibration smooth phase
            y = _smooth_repel(y, steps=220, beta0=25.0, beta1=90.0, lr0=0.025)
            # LP squeeze
            y = _lp_polish(y, seconds=pt)
            v = _max_cos(y)
        except Exception:
            trial += 1
            continue
        if not np.isfinite(v):
            trial += 1
            continue
        if v < best_val - 1e-12:
            best_val = v
            best = y.copy()
            cur = y.copy()
            cur_val = v
        else:
            # sideways walk: accept if not much worse
            tol = 0.004 * (1.0 - (time.time() - t0) / BUDGET) + 0.0005
            if v < cur_val + tol:
                # probabilistic accept worse to diversify (deterministic via rng)
                if v < cur_val or rng.random() < 0.5:
                    cur = y.copy()
                    cur_val = v
            else:
                # occasionally restart explorer from best with kick
                if trial % 11 == 10:
                    cur = best.copy()
                    cur_val = best_val
        trial += 1
    # final polish of global best with remaining time
    rem = deadline - time.time()
    if rem > 1.5:
        try:
            best = _lp_polish(best, seconds=rem - 1.0, seed_radius=5e-4)
            best_val = _max_cos(best)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
