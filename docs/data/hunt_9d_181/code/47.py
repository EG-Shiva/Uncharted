import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(X, budget, init_radius=0.006, cap=2000):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = init_radius
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    while True:
        now = time.time()
        if now - t0 >= budget:
            break
        if budget - (now - t0) < 0.2:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = np.where(c > thr)[0]
        if sel.size == 0:
            radius *= 0.5
            continue
        if sel.size > cap:
            cs = c[sel]
            part = np.argpartition(-cs, cap)[:cap]
            sel = sel[part]
        I = iu0[0][sel]
        J = iu0[1][sel]
        csel = c[sel]
        p = len(sel)
        nv = n * d + 1
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2 * d] = J[:, None] * d + ar
        cols_mat[:, 2 * d] = n * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _exp_smooth(X, steps, alpha, lr, max_step=0.02):
    x = X
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        step = lr * gt
        nm = np.linalg.norm(step, axis=1).max()
        if nm > max_step and nm > 0:
            step *= (max_step / nm)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _build_B():
    d = 9
    B = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in (-1.0, 1.0):
                for s2 in (-1.0, 1.0):
                    v = np.zeros(d)
                    v[i] = s1 / np.sqrt(2.0)
                    v[j] = s2 / np.sqrt(2.0)
                    B.append(v)
    return np.array(B)


def _greedy(pool, k, rng):
    N = pool.shape[0]
    s0 = int(rng.integers(0, N))
    sel = [s0]
    best = pool @ pool[s0]
    best[s0] = 2.0
    # mask selected
    for _ in range(1, k):
        idx = int(np.argmin(best))
        sel.append(idx)
        d = pool @ pool[idx]
        np.maximum(best, d, out=best)
        best[idx] = 2.0
    return pool[np.array(sel)].copy(), sel


def _swap_anneal(X, pool, iters, rng, t0_init=0.004, t0_frac=1.0):
    n = X.shape[0]
    cur = X.copy()
    best_m = _max_cos(cur)
    # temperature schedule
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        T = t0_init * (1.0 - frac) * t0_frac
        out = int(rng.integers(0, n))
        inn = int(rng.integers(0, pool.shape[0]))
        trial = cur.copy()
        trial[out] = pool[inn]
        m = _max_cos(trial)
        if m < best_m or (T > 0 and rng.random() < np.exp(-(m - best_m) / max(T, 1e-12)) and m < best_m + 0.02):
            cur = trial
            best_m = m
    return cur, best_m


def _expand_cyclic(reps):
    R = reps.shape[0]
    d = reps.shape[1]
    n = R * d + 1
    X = np.empty((n, d))
    for r in range(R):
        v = reps[r] / np.linalg.norm(reps[r])
        for s in range(d):
            X[r * d + s] = np.roll(v, s)
    X[-1] = np.ones(d) / np.sqrt(float(d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _sym_opt_cyclic(reps0, steps=400, k0=25.0, k1=220.0, lr0=0.05, lr1=0.005):
    reps = reps0 / np.linalg.norm(reps0, axis=1, keepdims=True)
    R, d = reps.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        k = k0 + (k1 - k0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        X = _expand_cyclic(reps)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(np.clip(k * (G - gmax), -50, 0))
        np.fill_diagonal(W, 0.0)
        Gr = W @ X
        dot = np.einsum('ij,ij->i', Gr, X)[:, None]
        Gt = Gr - dot * X
        # pullback to reps
        Grep = np.zeros_like(reps)
        for r in range(R):
            acc = np.zeros(d)
            for sh in range(d):
                g = Gt[r * d + sh]
                acc += np.roll(g, -sh)
            # project tangent to reps[r]
            v = reps[r] / np.linalg.norm(reps[r])
            acc = acc - np.dot(acc, v) * v
            Grep[r] = acc
        nrm = np.linalg.norm(Grep, axis=1).max()
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        step = (lr / max(nrm, 1e-12)) * Grep
        # cap
        sn = np.linalg.norm(step, axis=1).max()
        if sn > 0.05:
            step *= (0.05 / sn)
        reps = reps - step
        reps /= np.linalg.norm(reps, axis=1, keepdims=True)
    return reps


def solve():
    t0 = time.time()
    LIMIT = 172.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        vr = _max_cos(xr)
    except Exception:
        vr = 1e9
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    vb = _max_cos(xb)
    if vr < vb:
        xb, vb = xr.copy(), vr
    n, d = xb.shape  # 181, 9
    best_X = xb.copy()
    best_v = float(vb)

    # ---- build structured pools ----
    B = _build_B()  # 144
    # uniform pool
    U = rng.standard_normal(size=(2500, d))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    # ternary shell sample: entries in {-1,0,1} with 2-4 nonzeros
    T = []
    rngT = np.random.default_rng(1)
    while len(T) < 1200:
        k = int(rngT.integers(2, 5))
        idx = rngT.choice(d, size=k, replace=False)
        signs = rngT.choice(np.array([-1.0, 1.0]), size=k)
        v = np.zeros(d)
        v[idx] = signs
        v /= np.linalg.norm(v)
        T.append(v)
    T = np.array(T)
    pool = np.vstack([B, U, T])
    # dedup not needed

    cands = []
    # candidate A: greedy + swap
    rngA = np.random.default_rng(10)
    XA, _ = _greedy(pool, n, rngA)
    XA, mA = _swap_anneal(XA, pool, 2500, rngA, t0_init=0.003)
    cands.append((XA, mA))
    if time.time() - t0 > LIMIT - 100:
        pass
    else:
        # candidate B: different greedy start + swap on B-heavy pool
        rngB = np.random.default_rng(20)
        # bias pool: shuffle
        perm = rngB.permutation(pool.shape[0])
        pool2 = pool[perm]
        XB, _ = _greedy(pool2, n, rngB)
        XB, mB = _swap_anneal(XB, pool2, 2500, rngB, t0_init=0.005)
        cands.append((XB, mB))

    # candidate C: cyclic orbit symmetric optimisation (20*9+1=181)
    if time.time() - t0 < LIMIT - 90:
        rngC = np.random.default_rng(30)
        R = 20
        reps0 = rngC.standard_normal(size=(R, d))
        reps0 /= np.linalg.norm(reps0, axis=1, keepdims=True)
        reps_opt = _sym_opt_cyclic(reps0, steps=350, k0=30.0, k1=200.0, lr0=0.05, lr1=0.005)
        XC = _expand_cyclic(reps_opt)
        mC = _max_cos(XC)
        # light swap refinement of cyclic code against pool (breaks symmetry slightly)
        XC2, mC2 = _swap_anneal(XC, pool, 800, rngC, t0_init=0.002)
        # keep better of pure cyclic vs swapped
        if mC2 < mC:
            XC, mC = XC2, mC2
        cands.append((XC, mC))
        # second cyclic seed with different init for diversity if time
        if time.time() - t0 < LIMIT - 80:
            rngC2 = np.random.default_rng(40)
            reps02 = rngC2.standard_normal(size=(R, d))
            reps02 /= np.linalg.norm(reps02, axis=1, keepdims=True)
            reps_opt2 = _sym_opt_cyclic(reps02, steps=250, k0=25.0, k1=180.0, lr0=0.05, lr1=0.005)
            XD = _expand_cyclic(reps_opt2)
            mD = _max_cos(XD)
            cands.append((XD, mD))

    # ---- smooth repulsion on candidates (break symmetry, full coords) ----
    smoothed = []
    for (Xc, mc) in cands:
        if time.time() - t0 > LIMIT - 60:
            smoothed.append((Xc, mc))
            continue
        # two-stage smooth: coarse then finer
        y = Xc.copy()
        # adapt steps to remaining time
        y = _exp_smooth(y, steps=250, alpha=25.0, lr=0.005)
        y = _exp_smooth(y, steps=250, alpha=60.0, lr=0.004)
        v = _max_cos(y)
        smoothed.append((y, v))
    # pick top 2 smoothed
    smoothed.sort(key=lambda t: t[1])
    # ---- LP polish top candidates ----
    # always also consider incumbent best
    pool_final = smoothed[:2] if len(smoothed) >= 2 else smoothed
    for (Xc, mc) in pool_final:
        if time.time() - t0 > LIMIT - 30:
            break
        remain = LIMIT - (time.time() - t0) - 8.0
        if remain < 3.0:
            break
        budget = min(12.0, remain / (len(pool_final)))
        y = _lp_polish(Xc, budget=budget, init_radius=0.006, cap=2000)
        # short extra smoothing + re-polish if time
        v = _max_cos(y)
        if v < best_v:
            best_X, best_v = y.copy(), v
    # also polish incumbent if no improvement and time remains (ensures not worse)
    if time.time() - t0 < LIMIT - 10:
        # short polish of incumbent to try squeeze
        remain = LIMIT - (time.time() - t0) - 1.5
        if remain > 4.0:
            yb = _lp_polish(best_X, budget=min(10.0, remain), init_radius=0.004, cap=2000)
            vb2 = _max_cos(yb)
            if vb2 < best_v:
                best_X, best_v = yb, vb2

    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
