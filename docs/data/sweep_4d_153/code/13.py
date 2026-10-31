import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng, t_deadline=None):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        if t_deadline is not None and (t % 500 == 0) and time.time() > t_deadline:
            break
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac if k0 > 0 else k1
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = G.max()
        # exp weights, max shifted to 1
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        denom = 1.0 - mu ** (t + 1) if mu > 0 else 1.0
        vc = v / denom
        x = x - lr * vc
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert * x, axis=1, keepdims=True)
            pert -= pd * x
            x = x + amp * pert
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def random_ortho4(rng, eps):
    A = rng.standard_normal((4, 4))
    S = (A - A.T) * (eps / 2.0)
    # first order rotation: I+S orthogonalized
    R = np.eye(4) + S
    Q, _ = np.linalg.qr(R)
    return Q


def solve():
    t0 = time.time()
    DEADLINE = t0 + 172
    SEARCH_END = t0 + 112
    FINE_END = t0 + 138
    rng = np.random.default_rng(0)
    n, d = 153, 4
    try:
        best = np.load("best.npy")
    except Exception:
        best = None
    try:
        rec = np.load("record.npy")
    except Exception:
        rec = None
    cands0 = []
    if best is not None:
        try:
            best = np.asarray(best, dtype=float)
            if best.shape == (n, d):
                cands0.append(best / np.linalg.norm(best, axis=1, keepdims=True))
        except Exception:
            pass
    if rec is not None:
        try:
            rec = np.asarray(rec, dtype=float)
            if rec.shape == (n, d):
                cands0.append(rec / np.linalg.norm(rec, axis=1, keepdims=True))
        except Exception:
            pass
    if not cands0:
        cands0.append(rng.standard_normal((n, d)) / np.linalg.norm(rng.standard_normal((n, d)), axis=1, keepdims=True))
        # fix norm
        cands0[0] = cands0[0] / np.linalg.norm(cands0[0], axis=1, keepdims=True)
    # pick better as incumbent
    vals0 = [max_cos(c) for c in cands0]
    bi = int(np.argmin(vals0))
    incumbent = cands0[bi].copy()
    inc_val = float(vals0[bi])
    elites = [(inc_val, incumbent.copy())]

    trial = 0
    while time.time() < SEARCH_END and time.time() < DEADLINE - 30:
        trial += 1
        # choose base
        rbase = rng.random()
        if rbase < 0.6:
            base = incumbent
        elif len(elites) > 1 and rbase < 0.85:
            base = elites[int(rng.integers(len(elites)))][1]
        else:
            # best among elites
            base = elites[0][1]
        x = base.copy()
        mode = rng.random()
        if mode < 0.30:
            sigma = float(rng.choice(np.array([0.02, 0.05, 0.08, 0.12, 0.20, 0.32])))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode < 0.65:
            # worst-point removal + farthest void reinsertion
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            maxper = G.max(axis=1)
            k = int(rng.integers(3, 10))
            idx = np.argsort(-maxper)[:k]
            keep_mask = np.ones(n, dtype=bool)
            # iterative reinsertion: for each j, find void w.r.t. current set
            # build keep set excluding all idx initially
            keep_mask[idx] = False
            Xkeep = x[keep_mask]
            for j in idx:
                nc = 200
                cand = rng.standard_normal((nc, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ Xkeep.T
                md = dots.max(axis=1)
                pick = int(np.argmin(md))
                x[j] = cand[pick]
                # add to keep for next j
                Xkeep = np.vstack([Xkeep, x[j][None, :]])
            # small jitter to rest
            sigma = float(rng.choice(np.array([0.005, 0.015, 0.04])))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif mode < 0.80:
            # local patch rotation around random center
            ci = int(rng.integers(n))
            Gc = x @ x[ci]
            m = int(rng.integers(8, 22))
            neigh = np.argsort(-Gc)[:m]
            eps = float(rng.choice(np.array([0.05, 0.12, 0.25, 0.4])))
            R = random_ortho4(rng, eps)
            x[neigh] = x[neigh] @ R.T
            # jitter patch more
            sigma = float(rng.choice(np.array([0.01, 0.03, 0.07])))
            x[neigh] = x[neigh] + sigma * rng.standard_normal(x[neigh].shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # fresh random start
            x = rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)

        # hyperparams
        k0 = float(rng.choice(np.array([20.0, 40.0, 60.0, 80.0])))
        k1 = float(rng.choice(np.array([400.0, 800.0, 1200.0, 1800.0])))
        lr0 = float(rng.choice(np.array([0.02, 0.035, 0.05])))
        lr1 = float(rng.choice(np.array([0.002, 0.004, 0.008])))
        mu = float(rng.choice(np.array([0.80, 0.88, 0.92])))
        noise = float(rng.choice(np.array([0.0, 0.0, 0.003, 0.008])))
        iters = int(rng.choice(np.array([1500, 2200, 3000, 4000])))
        x = smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng, t_deadline=SEARCH_END)
        val = max_cos(x)
        if val < inc_val - 1e-12:
            incumbent = x.copy()
            inc_val = val
            elites.append((val, x.copy()))
            elites = sorted(elites, key=lambda z: z[0])[:5]
        elif val < inc_val + 0.008:
            # keep diverse near-elites
            if len(elites) < 5:
                elites.append((val, x.copy()))
                elites = sorted(elites, key=lambda z: z[0])[:5]
            else:
                # replace worst if better
                if val < elites[-1][0]:
                    elites[-1] = (val, x.copy())
                    elites = sorted(elites, key=lambda z: z[0])[:5]
        # simulated-annealing style: occasionally move incumbent sideways to escape
        # temperature decays with time
        elapsed = time.time() - t0
        temp = 0.004 * (1.0 - elapsed / (SEARCH_END - t0 + 1e-9))
        if temp > 0 and val < inc_val + temp and rng.random() < 0.08:
            incumbent = x.copy()
            # inc_val unchanged (keep best value for elites)

    # fine tune elites
    elites = sorted(elites, key=lambda z: z[0])[:4]
    fined = []
    for (v, xe) in elites:
        if time.time() > FINE_END:
            break
        y = smooth_relax(xe.copy(), 5000, 150.0, 2500.0, 0.02, 0.002, 0.90, 0.0, rng, t_deadline=FINE_END)
        fined.append((max_cos(y), y))
    if fined:
        fined = sorted(fined, key=lambda z: z[0])
        # merge
        allc = sorted(elites + fined, key=lambda z: z[0])[:3]
    else:
        allc = elites[:2]

    # final LP polish on top candidates with remaining time
    best_val = allc[0][0]
    best_x = allc[0][1].copy()
    # polish best first, then second if time
    remaining = DEADLINE - time.time()
    # reserve at least 5s
    if remaining > 6:
        # give 70% to best
        t_best = min(remaining - 5, 35.0)
        # quick fine before LP already done; now LP
        polished = lp_polish(best_x, t_best)
        pv = max_cos(polished)
        if pv < best_val:
            best_val, best_x = pv, polished
        # try second candidate if time left and promising
        remaining2 = DEADLINE - time.time()
        if len(allc) > 1 and remaining2 > 10:
            # only if second within 0.002 of best
            if allc[1][0] < best_val + 0.002:
                p2 = lp_polish(allc[1][1], min(remaining2 - 2, 15.0))
                v2 = max_cos(p2)
                if v2 < best_val:
                    best_val, best_x = v2, p2
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
