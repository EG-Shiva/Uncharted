import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, t0, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if time.time() > deadline:
            break
        Gfull = x @ x.T
        c = Gfull[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            if radius < 1e-14:
                break
            continue
        if cnt > 2500:
            thr = np.sort(c)[-2500]
            sel = c >= thr
        I = iu0[0][sel]; J = iu0[1][sel]
        cc = c[sel]
        p = len(I)
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((x.ravel(),
                           (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, n * d + 1)).tocsr()
        obj = np.zeros(n * d + 1); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 1e-14:
                break
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-14:
                break
    return x

def _smooth_anneal(X, steps, k0, k1, lr0):
    n, d = X.shape
    m1 = np.zeros_like(X); v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lk0, lk1 = np.log(k0), np.log(k1)
    b1t, b2t = 1.0, 1.0
    for t in range(1, steps + 1):
        frac = t / steps
        k = float(np.exp(lk0 + (lk1 - lk0) * frac))
        lr = lr0 * (1.0 - 0.5 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        gmax = float(G.max())
        W = np.exp(k * (G - gmax))
        np.fill_diagonal(W, 0.0)
        s = float(W.sum())
        if not np.isfinite(s) or s == 0.0:
            continue
        Grad = (W @ X) / s
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        b1t *= b1; b2t *= b2
        m1 = b1 * m1 + (1.0 - b1) * Gt
        v1 = b2 * v1 + (1.0 - b2) * (Gt * Gt)
        mhat = m1 / (1.0 - b1t); vhat = v1 / (1.0 - b2t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        step = np.clip(step, -0.05, 0.05)
        X = X - step
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    BUDGET = 168.0
    DEADLINE = t0 + BUDGET
    n, d = 121, 8
    rng = np.random.default_rng(7)
    # load references
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals0 = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals0))].copy()
    Vbest = float(min(vals0))
    try:
        Rref = np.load("record.npy").astype(float)
        Rref /= np.linalg.norm(Rref, axis=1, keepdims=True)
    except Exception:
        Rref = Xbest.copy()
    TABU_RAD = 0.05
    TABU_W = 20.0
    iu = np.triu_indices(n, 1)

    def procrustes_Q(Y, R):
        M = Y.T @ R
        try:
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            Q = U @ Vt
            if np.linalg.det(Q) < 0:
                U[:, -1] *= -1
                Q = U @ Vt
            return Q
        except Exception:
            return np.eye(d)

    def hinge_adam(Xinit, t, steps, lr, use_tabu=True, seed_off=0):
        X = Xinit.copy()
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        m = np.zeros_like(X); v = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1t, b2t = 1.0, 1.0
        for it in range(1, steps + 1):
            if time.time() > DEADLINE - 25.0:
                break
            # normalize
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            np.fill_diagonal(G, -1.0)
            D = G - t
            D[D < 0] = 0.0
            np.fill_diagonal(D, 0.0)
            W = 2.0 * D
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            if use_tabu and (it % 5 == 0):
                try:
                    Q = procrustes_Q(Y, Rref)
                    Dm = Y @ Q - Rref
                    fn_ = float(np.linalg.norm(Dm))
                    if fn_ > 1e-12:
                        rmsd = fn_ / np.sqrt(n)
                        if rmsd < TABU_RAD:
                            coeff = -2.0 * TABU_W * (TABU_RAD - rmsd)
                            Gr = (Dm @ Q.T) / (np.sqrt(float(n)) * fn_)
                            dot2 = (Gr * Y).sum(axis=1, keepdims=True)
                            GX = GX + coeff * (Gr - dot2 * Y) / nr
                except Exception:
                    pass
            b1t *= b1; b2t *= b2
            m = b1 * m + (1.0 - b1) * GX
            v = b2 * v + (1.0 - b2) * (GX * GX)
            mhat = m / (1.0 - b1t); vhat = v / (1.0 - b2t)
            step = lr * mhat / (np.sqrt(vhat) + eps)
            # anneal lr slightly
            step = np.clip(step, -0.08, 0.08)
            X = Y - step
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    def hole_insert(base, nsample, refine_steps):
        C = rng.standard_normal((nsample, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ base.T
        cm = S.max(axis=1)
        k = int(np.argmin(cm))
        bc = C[k].copy(); cur = float(cm[k])
        for st in range(refine_steps):
            sig = 0.03 * (1.0 - st / float(max(1, refine_steps))) + 0.001
            prop = bc + sig * rng.standard_normal(d)
            pn = np.linalg.norm(prop)
            if pn == 0:
                continue
            prop /= pn
            md = float((prop @ base.T).max())
            if md < cur:
                bc = prop; cur = md
        return bc

    # ---- build diverse starts ----
    starts = []
    starts.append(Xbest.copy())
    # perturbed far to escape basin
    for sig, sd in [(0.15, 101), (0.25, 102), (0.35, 103)]:
        r = np.random.default_rng(sd)
        Xp = Xbest + sig * r.standard_normal((n, d))
        Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        starts.append(Xp)
    # fresh random
    for sd in [201, 202, 203, 204]:
        r = np.random.default_rng(sd)
        Xf = r.standard_normal((n, d))
        Xf /= np.linalg.norm(Xf, axis=1, keepdims=True)
        starts.append(Xf)
    # hole-grown from optimized subsets: drop worst points then regrow
    G0 = Xbest @ Xbest.T
    np.fill_diagonal(G0, -2.0)
    rm0 = G0.max(axis=1)
    order_worst = np.argsort(-rm0)
    for n_start, sd in [(120, 301), (118, 302), (115, 303)]:
        if time.time() > DEADLINE - 60:
            break
        drop = order_worst[:n - n_start]
        keep = np.ones(n, dtype=bool); keep[drop] = False
        base = Xbest[keep].copy()
        # slight jitter of base to diversify
        rj = np.random.default_rng(sd)
        base = base + 0.02 * rj.standard_normal(base.shape)
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        # grow one by one at holes
        cur_base = base
        for _ in range(n - n_start):
            bc = hole_insert(cur_base, nsample=15000, refine_steps=120)
            cur_base = np.vstack([cur_base, bc[None, :]])
        # random order to break ordering tabu bias
        perm = rj.permutation(n)
        cur_base = cur_base[perm]
        starts.append(cur_base / np.linalg.norm(cur_base, axis=1, keepdims=True))

    # ---- threshold continuation ----
    t_seq = []
    for tt in [Vbest - 0.0002, 0.4600, 0.4595, 0.4590]:
        if tt < Vbest - 1e-12:
            t_seq.append(float(tt))
    # deduplicate sorted descending
    t_seq = sorted(set(t_seq), reverse=True)
    if not t_seq:
        t_seq = [Vbest - 0.0003]
    survivors = []
    # hinge optimization per start with continuation
    for si, s0 in enumerate(starts):
        if time.time() > DEADLINE - 45.0:
            break
        Xcur = s0.copy()
        # fewer steps for many starts, more for promising
        steps_per_t = 900 if si < 4 else 700
        lr_use = 0.015 if si % 2 == 0 else 0.008
        use_tabu = True
        for ti, tt in enumerate(t_seq):
            if time.time() > DEADLINE - 40.0:
                break
            Xcur = hinge_adam(Xcur, tt, steps=steps_per_t, lr=lr_use, use_tabu=use_tabu)
            v = _max_cos(Xcur)
            survivors.append((v, Xcur.copy()))
            # if feasible, continue to lower t with warm start, else keep trying same basin with smaller lr
            if v < tt + 2e-6:
                continue
            else:
                # one extra push with smaller lr for this t
                pass
        # also try without tabu for first two starts to allow record basin if better
        if si < 2 and time.time() < DEADLINE - 50:
            Xnt = hinge_adam(s0, t_seq[0], steps=500, lr=0.01, use_tabu=False)
            survivors.append((_max_cos(Xnt), Xnt))

    # rank survivors
    survivors.sort(key=lambda z: z[0])
    # distinctness filter via Procrustes RMSD + Gram
    kept = []
    for v, X in survivors:
        if v > Vbest + 0.006 and len(kept) >= 4:
            continue
        is_dup = False
        for _, Y in kept:
            try:
                Q = procrustes_Q(X, Y)
                rmsd = float(np.linalg.norm(X @ Q - Y)) / np.sqrt(n)
                if rmsd < 0.02:
                    is_dup = True
                    break
            except Exception:
                pass
        if not is_dup:
            kept.append((v, X))
        if len(kept) >= 6:
            break
    # promote best few to full anneal + polish
    n_promote = min(4, len(kept))
    for pi in range(n_promote):
        if time.time() > DEADLINE - 18.0:
            break
        v0, Xp = kept[pi]
        # LogSumExp annealing
        k1 = 5000.0 if pi % 2 == 0 else 3000.0
        Xa = _smooth_anneal(Xp.copy(), steps=1500, k0=50.0, k1=k1, lr0=0.008)
        va = _max_cos(Xa)
        # LP polish short
        rem = DEADLINE - time.time()
        secs = min(5.0, rem - 10.0)
        if secs > 1.0:
            Xa = _lp_polish(Xa, seconds=secs, t0=t0, deadline=DEADLINE - 6.0)
            va = _max_cos(Xa)
        survivors.append((va, Xa.copy()))
        if va < Vbest - 1e-12:
            Vbest = va
            Xbest = Xa.copy()

    # re-evaluate global best over all survivors (cheap safety)
    for v, X in survivors:
        if v < Vbest - 1e-12:
            # verify
            vv = _max_cos(X)
            if vv < Vbest - 1e-12:
                Vbest = vv
                Xbest = X.copy()

    # final squeeze
    rem = DEADLINE - time.time()
    if rem > 4.0:
        secs = min(rem - 1.5, 9.0)
        if secs > 1.0:
            Xbest = _lp_polish(Xbest, seconds=secs, t0=t0, deadline=DEADLINE - 0.5)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
