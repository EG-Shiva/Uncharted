import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _per_point_worst(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def _greedy_fill(y, idx, rng, n_cand=4000):
    n, d = y.shape
    idx_set = set(idx.tolist()) if not isinstance(idx, set) else idx
    for k in idx:
        mask = np.ones(n, dtype=bool)
        # fixed = all except remaining to-fill that are still empty?
        # use current y for others (already placed new pts are used)
        mask[k] = False
        others = y[mask]
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ others.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        y[k] = cand[bi]
    # tiny local hill-climb on refilled points
    for k in idx:
        cur = y[k].copy()
        mask = np.ones(n, dtype=bool); mask[k] = False
        others = y[mask]
        cur_w = float((others @ cur).max())
        trial = cur[None, :] + 0.03 * rng.standard_normal(size=(300, d))
        trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        w = (trial @ others.T).max(axis=1)
        bi = int(np.argmin(w))
        if w[bi] < cur_w:
            y[k] = trial[bi]
    return y

def _anneal(x, steps, lr0, lr1, T0, T1, noise0, rng, deadline):
    n, d = x.shape
    for it in range(steps):
        if (it & 63) == 0 and time.time() > deadline:
            break
        frac = it / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * frac
        T = T0 + (T1 - T0) * frac
        # Gram
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # smooth-max weights, cap to avoid overflow; sparsify small weights
        # W = exp((G-m)/T) for G > m - 8T else 0
        thr = m - 8.0 * T
        # compute only where needed
        W = np.zeros_like(G)
        sel = G > thr
        # diag is -2, excluded automatically
        W[sel] = np.exp((G[sel] - m) / T)
        np.fill_diagonal(W, 0.0)
        S = W @ x
        dot = np.einsum('ij,ij->i', S, x)
        Tang = S - dot[:, None] * x
        # normalize step by typical magnitude
        x = x - lr * Tang
        if noise0 > 0:
            ns = noise0 * (1.0 - frac)
            if ns > 1e-12:
                nz = rng.standard_normal(size=(n, d))
                # make tangential
                nd = np.einsum('ij,ij->i', nz, x)
                nz -= nd[:, None] * x
                x = x + ns * nz
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    HARD = t0 + 165.0
    SOFT = t0 + 150.0
    rng = np.random.default_rng(12345)
    # load incumbent
    best_x = None
    best_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        best_x = a
        best_v = _max_cos(a)
    n, d = best_x.shape
    base = best_x.copy()

    # quick short anneal of incumbent to ensure local snap (cheap)
    try:
        y = base.copy()
        y = _anneal(y, steps=600, lr0=0.02, lr1=0.002, T0=0.01, T1=0.0015, noise0=0.0, rng=rng, deadline=HARD - 5)
        v = _max_cos(y)
        if v < best_v:
            best_v = v
            best_x = y.copy()
    except Exception:
        pass

    it = 0
    # main basin-hopping
    while time.time() < SOFT:
        it += 1
        if time.time() > HARD - 8:
            break
        y = best_x.copy()
        w = _per_point_worst(y)
        order = np.argsort(-w)
        # choose surgery size
        r = rng.random()
        if r < 0.35:
            k = int(rng.integers(4, 10))
        elif r < 0.7:
            k = int(rng.integers(10, 20))
        else:
            k = int(rng.integers(20, 36))
        # build index set: half worst, half random + rattler bias
        nworst = k // 2
        idx = list(order[:nworst])
        # add low-contact / rattler points if any (points with small worst are rattlers but also want to move)
        # include random picks
        while len(idx) < k:
            j = int(rng.integers(0, n))
            if j not in idx:
                idx.append(j)
        idx = np.array(idx, dtype=int)
        # occasionally do global shake instead of reinsert
        if rng.random() < 0.25:
            sig = rng.choice([0.005, 0.01, 0.02, 0.05])
            nz = rng.standard_normal(size=(n, d))
            nd = np.einsum('ij,ij->i', nz, y)
            nz -= nd[:, None] * y
            y = y + sig * nz
            y /= np.linalg.norm(y, axis=1, keepdims=True)
        else:
            # randomize selected points then greedy fill
            rnd = rng.standard_normal(size=(len(idx), d))
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            y[idx] = rnd
            y = _greedy_fill(y, idx, rng, n_cand=4000)
        # anneal with schedule varying by iteration
        T0 = float(rng.choice([0.015, 0.02, 0.03]))
        T1 = 0.001
        lr0 = float(rng.choice([0.03, 0.05, 0.08]))
        lr1 = 0.002
        noise0 = float(rng.choice([0.0, 0.002, 0.005]))
        steps = int(rng.choice([600, 900, 1300]))
        # shrink steps if little time left
        remain = SOFT - time.time()
        if remain < 12:
            steps = min(steps, 400)
        y = _anneal(y, steps=steps, lr0=lr0, lr1=lr1, T0=T0, T1=T1, noise0=noise0, rng=rng, deadline=HARD - 5)
        v = _max_cos(y)
        if v < best_v:
            best_v = v
            best_x = y.copy()
        # also try a pure-relax copy without noise if we are close
        if v < best_v + 0.002 and time.time() < SOFT - 2:
            pass
    # final snap: longer low-temp relax of best
    try:
        if time.time() < HARD - 6:
            y = best_x.copy()
            y = _anneal(y, steps=1500, lr0=0.015, lr1=0.001, T0=0.006, T1=0.0006, noise0=0.0, rng=rng, deadline=HARD - 2)
            v = _max_cos(y)
            if v < best_v:
                best_v = v
                best_x = y.copy()
    except Exception:
        pass
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
