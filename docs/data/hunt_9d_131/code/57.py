import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam_lse(X, steps, a0, a1, lr0, lr1, deadline=1e18):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for k in range(steps):
        f = k / max(1, steps - 1)
        alpha = a0 * ((a1 / a0) ** f)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(np.max(G))
        # avoid overflow: exp(alpha*(G-gm)), G-gm<=0
        W = np.exp(alpha * (G - gm))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W)) + 1e-300
        Gr = (W @ X) / S  # scaled gradient of LSE
        # tangential projection
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        # normalize scale to keep Adam stable
        Gt = Gt * (n)
        t += 1
        m1 = b1 * m1 + (1 - b1) * Gt
        m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
        c1 = 1 - b1 ** t
        c2 = 1 - b2 ** t
        mh = m1 / c1
        vh = m2 / c2
        step = lr * mh / (np.sqrt(vh) + eps)
        # keep tangential (project step)
        dot2 = np.sum(step * X, axis=1, keepdims=True)
        step = step - dot2 * X
        # clip step norm
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        cap = 0.05
        scale = np.minimum(1.0, cap / (sn + 1e-12))
        step = step * scale
        X = X - step
        X = _norm(X)
        if (k % 50) == 49 and time.time() > deadline:
            break
    return X

def _refine_batch(Z, Xk, steps=200, T=0.008, lr=0.05):
    # Z: (B,d), Xk: (k,d)
    B = Z.shape[0]
    V = np.zeros_like(Z)
    mom = 0.8
    for it in range(steps):
        f = it / max(1, steps - 1)
        Ti = T * (1.0 + 0.5 * (1 - f))
        lri = lr * (1 - 0.5 * f)
        D = Z @ Xk.T  # B x k
        dm = np.max(D, axis=1, keepdims=True)
        W = np.exp((D - dm) / Ti)
        W = W / (np.sum(W, axis=1, keepdims=True) + 1e-300)
        G = W @ Xk  # B x d
        dot = np.sum(G * Z, axis=1, keepdims=True)
        Gt = G - dot * Z
        V = mom * V + (1 - mom) * Gt
        Z = Z - lri * V
        Z = _norm(Z)
    return Z

def _global_relax(X, steps, thr=0.42245, lr0=0.015, wlse=0.02, deadline=1e18):
    n = X.shape[0]
    V = np.zeros_like(X)
    mom = 0.85
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * (1 - 0.6 * f)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(np.max(G))
        # hinge gradient
        E = G - thr
        E[E < 0] = 0.0
        np.fill_diagonal(E, 0.0)
        Gh = E @ X
        # weak LSE gradient
        alpha = 60.0
        W = np.exp(alpha * (G - gm))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W)) + 1e-300
        Gl = (W @ X) / S * n
        dot_h = np.sum(Gh * X, axis=1, keepdims=True)
        Gh = Gh - dot_h * X
        dot_l = np.sum(Gl * X, axis=1, keepdims=True)
        Gl = Gl - dot_l * X
        Gt = Gh * 2.0 + wlse * Gl
        # normalize by degree-ish to avoid explosion
        V = mom * V + (1 - mom) * Gt
        step = lr * V
        # project step tangentially already; clip
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        cap = 0.03
        sc = np.minimum(1.0, cap / (sn + 1e-12))
        step = step * sc
        X = X - step
        X = _norm(X)
        if (s % 50) == 49 and time.time() > deadline:
            break
    return X

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c_all = Gfull[iu0]
        m = float(np.max(c_all))
        thresh = m - max(20 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 4000:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:4000]
            mask = np.zeros_like(sel)
            mask[idx[order]] = True
            sel = mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    TIME_BUDGET = 168.0
    LP_RESERVE = 30.0
    grow_deadline = t0 + TIME_BUDGET - LP_RESERVE
    lp_deadline = t0 + TIME_BUDGET - 1.0
    np.random.seed(1)
    rng = np.random.default_rng(1)
    n_full, d = 131, 9
    # load start
    X0 = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n_full, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    X0 = v.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = _norm(rng.standard_normal((n_full, d)))
        best_val = _max_cos(X0)
    # diagnose contacts
    G0 = X0 @ X0.T
    np.fill_diagonal(G0, -2.0)
    gmax = float(np.max(G0))
    contact = G0 >= (gmax - 1e-7)
    deg = np.sum(contact, axis=1).astype(int)
    per_max = np.max(G0, axis=1)
    # 9 rattlers: smallest degree
    order_deg = np.argsort(deg, kind="stable")
    rattlers = set(order_deg[:9].tolist())
    # worst hubs: largest per-point max / degree among remaining
    remaining = [i for i in range(n_full) if i not in rattlers]
    # sort remaining by per_max desc, then deg desc
    remaining_sorted = sorted(remaining, key=lambda i: (per_max[i], deg[i]), reverse=True)
    K = 12  # remove 12 more -> N0=110
    extra = remaining_sorted[:K]
    remove = set(list(rattlers) + extra)
    keep = [i for i in range(n_full) if i not in remove]
    Xk = X0[keep].copy()
    # hard optimize smaller code with annealed LSE-Adam
    # aim clearly below 0.42245
    inner_deadline = min(grow_deadline - 60.0, t0 + 45.0)
    if inner_deadline < t0 + 5:
        inner_deadline = t0 + 5
    Xk = _adam_lse(Xk, steps=5000, a0=20.0, a1=400.0, lr0=0.012, lr1=0.002, deadline=inner_deadline)
    # a second short burst if still above target and time left
    cur = _max_cos(Xk)
    if cur > 0.40 and time.time() < inner_deadline - 2:
        Xk = _adam_lse(Xk, steps=2000, a0=50.0, a1=500.0, lr0=0.006, lr1=0.001, deadline=inner_deadline)
    THR = 0.42245
    # greedy grow back to 131
    while Xk.shape[0] < n_full and time.time() < grow_deadline:
        k = Xk.shape[0]
        # time guard: need at least ~1.5s per insertion + LP reserve
        if grow_deadline - time.time() < 2.0:
            break
        # generate candidates
        NC = 20000
        Z = rng.standard_normal((NC, d))
        Z = _norm(Z)
        D = Z @ Xk.T
        worst = np.max(D, axis=1)
        idx30 = np.argpartition(worst, 30)[:30]
        # sort them best first
        o = np.argsort(worst[idx30])
        idx30 = idx30[o]
        Z30 = Z[idx30].copy()
        # refine each against current code
        Z30 = _refine_batch(Z30, Xk, steps=180, T=0.008, lr=0.05)
        # score refined
        Dr = Z30 @ Xk.T
        wr = np.max(Dr, axis=1)
        order = np.argsort(wr, kind="stable")
        # try insertions in order, with geometric distinctness on backtrack
        inserted = False
        tried = []
        for oi in order:
            z = Z30[oi].copy()
            # distinctness: skip if too close to failed tried
            skip = False
            for tz in tried:
                if float(np.dot(z, tz)) > 0.9:
                    skip = True
                    break
            if skip:
                continue
            Xt = np.vstack([Xk, z[None, :]])
            Xt = _global_relax(Xt, steps=400, thr=THR, lr0=0.015, wlse=0.02, deadline=grow_deadline)
            gv = _max_cos(Xt)
            if gv <= 0.43:
                Xk = Xt
                inserted = True
                break
            else:
                tried.append(z.copy())
                # try next distinct hole
                continue
        if not inserted:
            # fallback: take best refined even if above 0.43 but relax more, else take raw best hole
            # pick least-bad tried or best order[0]
            z = Z30[order[0]].copy()
            Xt = np.vstack([Xk, z[None, :]])
            Xt = _global_relax(Xt, steps=400, thr=THR, lr0=0.015, wlse=0.02, deadline=grow_deadline)
            Xk = Xt
        # optional small extra LSE adam on grown code to keep good pairs undisturbed little
        # light touch: 150 steps low lr
        if time.time() < grow_deadline - 1.0:
            # only if global max is high, nudge with adam
            pass
    # if growth stopped early (time), pad with holes greedily without relax if needed
    while Xk.shape[0] < n_full:
        NC = 10000
        Z = rng.standard_normal((NC, d))
        Z = _norm(Z)
        D = Z @ Xk.T
        worst = np.max(D, axis=1)
        bi = int(np.argmin(worst))
        Xk = np.vstack([Xk, Z[bi][None, :]])
    Xk = _norm(Xk)
    # final LP polish with remaining time (reserve)
    # ensure at least a quick polish
    try:
        Xk = _lp_polish(Xk, lp_deadline)
    except Exception:
        pass
    Xk = _norm(Xk)
    # safety: if result is worse than loaded best and NaN-free, consider returning best polished? Must return 131x9.
    # Keep our grown code (different basin) even if slightly worse; but if it is much worse (>0.43) fall back to LP-polished record
    try:
        gv = _max_cos(Xk)
    except Exception:
        gv = 1e9
    if gv > 0.43:
        try:
            fb = _lp_polish(X0, lp_deadline)
            if _max_cos(fb) < gv:
                return _norm(fb)
        except Exception:
            pass
    return _norm(Xk)
