import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _eval_viol(x, T, alpha):
    n = x.shape[0]
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    z = alpha * (c - T)
    # stable softplus(z)/alpha
    sp = (np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)) / alpha
    viol = float(sp.sum())
    cnt = int(np.sum(c > T))
    mx = float(c.max())
    return viol, cnt, mx

def _adam_softplus(X0, T, alpha, steps, lr, t_deadline=None):
    X = X0.copy()
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for t in range(1, steps + 1):
        if t_deadline is not None and (t % 50 == 0) and time.time() > t_deadline:
            break
        G = X @ X.T
        Z = alpha * (G - T)
        # clip for sigmoid
        Zc = np.clip(Z, -30.0, 30.0)
        S = 1.0 / (1.0 + np.exp(-Zc))
        np.fill_diagonal(S, 0.0)
        Grad = S @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1.0 - b1) * Gt
        v *= b2
        v += (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - b1p)
        vhat = v / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # project step to tangent for stability
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        # transport momentum to new tangent
        mdot = np.sum(m * X, axis=1, keepdims=True)
        m = m - mdot * X
    return X

def _gd_refine(X_init, steps, a0, a1, s0, s1):
    X = X_init.copy()
    for k in range(steps):
        frac = k / max(steps - 1, 1)
        alpha = a0 * (a1 / a0) ** frac
        step = s0 * (s1 / s0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = G.max()
        W = np.exp(alpha * (G - mx))
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        d = np.sum(grad * X, axis=1, keepdims=True)
        grad -= d * X
        nm = np.linalg.norm(grad, axis=1).max()
        if nm < 1e-14:
            continue
        grad *= (step / nm)
        X -= grad
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    T_EXPL_END = t0 + 88.0
    T_REF_END = t0 + 122.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            rng0 = np.random.default_rng(0)
            xb = rng0.standard_normal(size=(173, 5))
    if xb.shape != (173, 5):
        rng0 = np.random.default_rng(0)
        tmp = rng0.standard_normal(size=(173, 5))
        try:
            nn = min(xb.shape[0], 173)
            dd = min(xb.shape[1], 5)
            tmp[:nn, :dd] = xb[:nn, :dd]
        except Exception:
            pass
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    base_val = _max_cos(xb)
    overall_best = xb.copy()
    overall_val = base_val

    rng = np.random.default_rng(12345)
    # build starts: record, strongly perturbed copies, random
    starts = [xb.copy()]
    for sigma in [0.06, 0.16, 0.35]:
        Xp = xb + sigma * rng.standard_normal(size=xb.shape)
        Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        starts.append(Xp)
    for _ in range(3):
        Xr = rng.standard_normal(size=xb.shape)
        Xr /= np.linalg.norm(Xr, axis=1, keepdims=True)
        starts.append(Xr)

    targets = [0.757, 0.754, 0.751, 0.748]
    alphas = [200, 700, 1800, 3000]
    lr_map = {200: 0.008, 700: 0.005, 1800: 0.003, 3000: 0.002}
    steps_per = 320

    candidates = [xb.copy()]
    cand_viols = []

    for si, Xinit in enumerate(starts):
        if time.time() > T_EXPL_END:
            break
        X = Xinit.copy()
        prev_cnt = None
        for ti, T in enumerate(targets):
            if time.time() > T_EXPL_END:
                break
            for ai, A in enumerate(alphas):
                if time.time() > T_EXPL_END:
                    break
                lr = lr_map[A]
                # slightly vary lr per start for diversity (deterministic)
                lr = lr * (0.8 + 0.4 * ((si + ti) % 3) / 2.0)
                X = _adam_softplus(X, T, A, steps_per, lr, t_deadline=T_EXPL_END)
                viol, cnt, mx = _eval_viol(X, T, A)
                # LP polish trigger: violation below threshold or count changed substantially
                # we only record here; actual LP deferred to end to save time,
                # but remember trajectories meeting criterion with priority
                if ai == len(alphas) - 1:
                    pass
            # snapshot after each target
            viol0, cnt0, mx0 = _eval_viol(X, T, 2000)
            candidates.append(X.copy())
            # priority = (viol, cnt, mx): retain according to violation & count
            cand_viols.append((viol0, cnt0, mx0, len(candidates) - 1))
            # early stop if violation is zero (achieved below-record target)
            if cnt0 == 0:
                # great basin found, keep going to next lower target
                continue
            prev_cnt = cnt0
        # end targets
    # rank candidates by violation at last evaluated target then by max
    # Phase B: soft-max refinement of candidates
    # deduplicate / limit to most promising: sort by max first + violation priority
    scored = []
    for idx, Xc in enumerate(candidates):
        mx = _max_cos(Xc)
        scored.append((mx, idx))
    scored.sort(key=lambda t: t[0])
    # take top 8 diverse + violation-best 2
    take_idx = set([i for _, i in scored[:8]])
    if cand_viols:
        cand_viols.sort(key=lambda t: (t[0], t[1], t[2]))
        for _, _, _, ci in cand_viols[:3]:
            take_idx.add(ci)
    take_list = list(take_idx)

    refined = []
    rng2 = np.random.default_rng(999)
    for k, ci in enumerate(take_list):
        if time.time() > T_REF_END:
            break
        Xc = candidates[ci]
        # deterministic refine params varying slightly
        a0 = 10.0 ** rng2.uniform(0.9, 1.5)
        a1 = 10.0 ** rng2.uniform(2.0, 2.6)
        s0 = 10.0 ** rng2.uniform(-1.6, -1.0)
        s1 = max(s0 * (10.0 ** rng2.uniform(-1.0, -0.3)), 3e-4)
        steps = int(rng2.integers(300, 550))
        Xo = _gd_refine(Xc, steps, a0, a1, s0, s1)
        v = _max_cos(Xo)
        refined.append((v, Xo))
        if v < overall_val:
            overall_val = v
            overall_best = Xo.copy()
    # also consider unrefined candidates
    for Xc in candidates:
        v = _max_cos(Xc)
        if v < overall_val:
            overall_val = v
            overall_best = Xc.copy()

    # Phase C: LP polish top candidates (only those with changed topology or low violation)
    refined.sort(key=lambda t: t[0])
    top_for_lp = []
    for v, Xc in refined[:4]:
        top_for_lp.append(Xc)
    # ensure overall best included
    top_for_lp.append(overall_best.copy())
    # deduplicate by value? just polish best 3 unique by max
    top_for_lp = sorted(top_for_lp, key=lambda X: _max_cos(X))[:3]

    remaining = BUDGET - (time.time() - t0) - 2.0
    if remaining < 5:
        remaining = 5.0
    per = max(4.0, remaining * 0.22)
    for X in top_for_lp:
        if time.time() - t0 > BUDGET - 12:
            break
        Xp2 = _lp_polish(X, per)
        v = _max_cos(Xp2)
        refined.append((v, Xp2))
        if v < overall_val:
            overall_val = v
            overall_best = Xp2.copy()
    leftover = BUDGET - (time.time() - t0) - 1.0
    if leftover > 3:
        overall_best = _lp_polish(overall_best, leftover)
        overall_val = _max_cos(overall_best)
    overall_best /= np.linalg.norm(overall_best, axis=1, keepdims=True)
    # safety: if somehow worse than record and time allows, return record polished briefly
    if overall_val > base_val + 1e-12:
        # return the better of overall and xb
        overall_best = xb.copy()
    # final normalization, no NaN
    overall_best = np.nan_to_num(overall_best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(overall_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    overall_best /= nr
    return overall_best
