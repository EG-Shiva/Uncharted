import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _adam_global(X, steps, a0, a1, lr0, lr1, deadline=1e18):
    n, d = X.shape
    m = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    mask = np.ones((n, n))
    np.fill_diagonal(mask, 0.0)
    la = np.log(a1 / a0) / max(steps, 1)
    ll = np.log(lr1 / lr0) / max(steps, 1)
    for t in range(1, steps + 1):
        if (t % 50) == 0 and time.time() > deadline:
            break
        alpha = a0 * np.exp(la * (t - 1))
        lr = lr0 * np.exp(ll * (t - 1))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(G.max())
        E = np.exp(alpha * (G - cmax)) * mask
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        W = E / S
        Grad = W @ X
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mh = m / (1 - b1 ** t)
        vh = vv / (1 - b2 ** t)
        step = lr * mh / (np.sqrt(vh) + eps)
        sdot = (step * X).sum(axis=1, keepdims=True)
        step = step - sdot * X
        nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 0.05 / nm)
        step = step * clip
        X = X - step
        X = _norm(X)
    return X

def _refine_single(y, Xf, steps=100, a0=30.0, a1=500.0, lr0=0.01, lr1=0.002, deadline=1e18):
    d = y.shape[0]
    mm = np.zeros(d)
    vv = np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    la = np.log(a1 / a0) / max(steps, 1)
    ll = np.log(lr1 / lr0) / max(steps, 1)
    for t in range(1, steps + 1):
        if (t % 50) == 0 and time.time() > deadline:
            break
        alpha = a0 * np.exp(la * (t - 1))
        lr = lr0 * np.exp(ll * (t - 1))
        s = Xf @ y
        smax = float(s.max())
        e = np.exp(alpha * (s - smax))
        es = float(e.sum())
        if not np.isfinite(es) or es <= 0:
            break
        w = e / es
        g = w @ Xf
        gt = g - (g @ y) * y
        mm = b1 * mm + (1 - b1) * gt
        vv = b2 * vv + (1 - b2) * (gt * gt)
        mh = mm / (1 - b1 ** t)
        vh = vv / (1 - b2 ** t)
        st = lr * mh / (np.sqrt(vh) + eps)
        st = st - (st @ y) * y
        nn = float(np.linalg.norm(st)) + 1e-12
        if nn > 0.05:
            st *= 0.05 / nn
        y = y - st
        nrm = float(np.linalg.norm(y)) + 1e-30
        y = y / nrm
    return y

def _pick_hole(Xcur, rng, C=30000):
    d = Xcur.shape[1]
    Z = rng.normal(size=(C, d))
    Z = _norm(Z)
    # chunk to bound memory
    best_idx = -1
    best_val = 1e9
    chunk = 6000
    for s0 in range(0, C, chunk):
        Zc = Z[s0:s0 + chunk]
        md = (Zc @ Xcur.T).max(axis=1)
        i = int(np.argmin(md))
        v = float(md[i])
        if v < best_val:
            best_val = v
            best_idx = s0 + i
    return Z[best_idx].copy()

def _select_delete(X, k, rng):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    M = float(G.max())
    Adj = (G > M - 1e-5)
    deg = Adj.sum(axis=1).astype(int)
    rm = G.max(axis=1)
    n_over = min(5, max(3, k // 5))
    n_core = k - n_over
    # overloaded: highest degree then rowmax
    order = np.argsort(-deg * 1000.0 - rm)
    over = list(order[:n_over])
    # BFS connected cluster in contact graph
    # start from high-degree point not in over if possible
    start = int(order[n_over % n])
    # if start in over pick next
    for c in order:
        if int(c) not in over:
            start = int(c)
            break
    visited = set()
    frontier = [start]
    visited.add(start)
    # expand
    while len(visited) < n_core and len(frontier) > 0:
        # shuffle neighbors via rng order for variety
        cur = frontier.pop(0)
        nbrs = np.where(Adj[cur])[0].tolist()
        if len(nbrs) > 1:
            perm = rng.permutation(len(nbrs))
            nbrs = [nbrs[i] for i in perm]
        for nb in nbrs:
            nb = int(nb)
            if nb not in visited and nb not in over:
                visited.add(nb)
                frontier.append(nb)
                if len(visited) >= n_core:
                    break
    # if graph too sparse, fill randomly
    while len(visited) < n_core:
        c = int(rng.randint(0, n))
        if c not in visited and c not in over:
            visited.add(c)
    core = list(visited)
    dele = core + over
    # ensure distinct and size k
    dele = list(dict.fromkeys([int(x) for x in dele]))
    while len(dele) < k:
        c = int(rng.randint(0, n))
        if c not in dele:
            dele.append(c)
    dele = np.array(dele[:k], dtype=int)
    return dele

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _mc(x)
    except Exception:
        return x
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    itn = 0
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14 or itn >= 40:
            break
        itn += 1
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        if len(I) == 0:
            radius *= 0.5
            continue
        if len(I) > 2500:
            od = np.argsort(-c[sel])[:2500]
            I, J = I[od], J[od]
            cv = c[sel][od]
        else:
            cv = c[sel]
        p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _mc(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 176.0
    LP_RESERVE = 20.0
    DEAD = t0 + 172.0
    explore_end = t0 + BUDGET - LP_RESERVE - 2.0
    rng = np.random.RandomState(0)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _mc(a)
                    if v < best_val:
                        best_val = v
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _mc(best)
    # keep valid best throughout
    cycle = 0
    while time.time() < explore_end and time.time() < DEAD - LP_RESERVE:
        cycle += 1
        if time.time() > explore_end:
            break
        # different cluster sizes 18-30
        k = int(rng.randint(18, 31))
        try:
            dele = _select_delete(best, k, rng)
        except Exception:
            continue
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[dele] = False
        try:
            Xrem = best[keep_mask].copy()
        except Exception:
            continue
        m_rem = Xrem.shape[0]  # 124-136
        # fixed small number of softmax-Adam steps on remainder (no LP)
        a0r = 40.0
        a1r = float(rng.uniform(250.0, 500.0))
        lr0r = float(rng.uniform(0.006, 0.014))
        lr1r = 0.0015
        REM_STEPS = 350
        try:
            Xrem = _adam_global(Xrem, steps=REM_STEPS, a0=a0r, a1=a1r, lr0=lr0r, lr1=lr1r, deadline=explore_end)
        except Exception:
            continue
        if time.time() > explore_end - 1.0:
            continue
        # regrow one at a time
        Xcur = Xrem.copy()
        ok = True
        # randomize insertion order effect via fresh candidate draws
        for ti in range(k):
            if time.time() > explore_end - 0.5:
                ok = False
                break
            try:
                y = _pick_hole(Xcur, rng, C=30000)
            except Exception:
                ok = False
                break
            try:
                y = _refine_single(y, Xcur, steps=100, a0=30.0, a1=500.0, lr0=0.01, lr1=0.002, deadline=explore_end)
            except Exception:
                pass
            Xcur = np.vstack([Xcur, y.reshape(1, -1)])
            Xcur = _norm(Xcur)
            if ((ti + 1) % 5 == 0) and (ti + 1 < k or True):
                # 120 global relaxation steps after every five insertions
                try:
                    Xcur = _adam_global(Xcur, steps=120, a0=40.0, a1=350.0, lr0=0.008, lr1=0.0015, deadline=explore_end)
                except Exception:
                    pass
            # time check at least every 50 iterations is inside optimizers; also check here
            if time.time() > explore_end:
                ok = False
                break
        if not ok:
            continue
        if Xcur.shape != (154, 10):
            continue
        try:
            v = _mc(Xcur)
        except Exception:
            continue
        if v < best_val - 1e-12 and np.all(np.isfinite(Xcur)):
            best = _norm(Xcur)
            best_val = v
        # else discard, keep valid best
        if time.time() >= explore_end:
            break
    # reserve only final 20s for LP polish
    remain = DEAD - time.time() - 1.0
    if remain < 1.0:
        remain = 1.0
    final_sec = min(remain, LP_RESERVE)
    try:
        best = _lp_polish(best, seconds=final_sec, deadline=DEAD)
    except Exception:
        pass
    best = _norm(best)
    best = np.where(np.isfinite(best), best, 0.0)
    best = _norm(best)
    return best
