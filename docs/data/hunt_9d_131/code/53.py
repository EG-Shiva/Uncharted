import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _per_point(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G


def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n


def _lp_polish(X, budget, deadline=1e18):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    radius = 1e-3
    t0 = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    rep = np.repeat(np.arange(n), d)
    while time.time() - t0 < budget and time.time() < deadline:
        if radius < 1e-14:
            break
        G = X @ X.T
        cf = G[iu0]
        m = float(cf.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = cf > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        I = iu0[0][sel]
        J = iu0[1][sel]
        cc = cf[sel]
        if cnt > 2500:
            idx = np.argpartition(-cc, 2500)[:2500]
            I, J, cc = I[idx], J[idx], cc[idx]
            cnt = 2500
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (rep, np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0 or res.x is None:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            bestX = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return bestX, best


def _smooth_global(X, steps, a0, a1, lr0, deadline=1e18):
    n, d = X.shape
    X = X.copy()
    M = np.zeros_like(X)
    V = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best = float(G0.max())
    bestX = X.copy()
    for t in range(1, steps + 1):
        if (t & 63) == 0 and time.time() > deadline:
            break
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * 0.5 * (1.0 + np.cos(np.pi * frac))
        lr = max(lr, lr0 * 0.05)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        if gmax < best:
            best = gmax
            bestX = X.copy()
        E = np.clip(alpha * (G - gmax), -30.0, 0.0)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        M = b1 * M + (1 - b1) * Gt
        V = b2 * V + (1 - b2) * (Gt * Gt)
        Mh = M / (1 - b1 ** t)
        Vh = V / (1 - b2 ** t)
        X = X - lr * Mh / (np.sqrt(Vh) + eps)
        X = _norm(X)
    v = _max_cos(X)
    if v < best:
        best = v
        bestX = X.copy()
    return bestX, best


def _smooth_subset(Xfull, S, steps, a0, a1, lr0, deadline=1e18):
    X = Xfull.copy()
    n, d = X.shape
    k = len(S)
    if k == 0:
        return X, _max_cos(X)
    S = np.asarray(S, dtype=np.int64)
    M = np.zeros((k, d))
    V = np.zeros((k, d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    best = _max_cos(X)
    bestX = X.copy()
    Xs = X[S]
    for t in range(1, steps + 1):
        if (t & 63) == 0 and time.time() > deadline:
            break
        frac = t / steps
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * 0.5 * (1.0 + np.cos(np.pi * frac))
        lr = max(lr, lr0 * 0.05)
        Xs = X[S]
        D = Xs @ X.T
        # mask self dots
        for pi, gi in enumerate(S):
            D[pi, gi] = -2.0
        gmax = float(D.max())
        E = np.clip(alpha * (D - gmax), -30.0, 0.0)
        W = np.exp(E)
        for pi, gi in enumerate(S):
            W[pi, gi] = 0.0
        Grad = W @ X
        dot = np.sum(Grad * Xs, axis=1, keepdims=True)
        Gt = Grad - dot * Xs
        M = b1 * M + (1 - b1) * Gt
        V = b2 * V + (1 - b2) * (Gt * Gt)
        Mh = M / (1 - b1 ** t)
        Vh = V / (1 - b2 ** t)
        Xs2 = Xs - lr * Mh / (np.sqrt(Vh) + eps)
        Xs2 = _norm(Xs2)
        X[S] = Xs2
        # track global best (cheap every 64)
        if (t & 63) == 0:
            v = _max_cos(X)
            if v < best:
                best = v
                bestX = X.copy()
    v = _max_cos(X)
    if v < best:
        best = v
        bestX = X.copy()
    # return best snapshot
    return bestX, best


def _hole_reinit(X, idx, rng, n_cand=3000):
    n, d = X.shape
    Y = X.copy()
    idx = list(idx)
    for gi in idx:
        mask = np.ones(n, dtype=bool)
        mask[gi] = False
        # also exclude other idx not yet placed? use current Y for placed ones
        # frozen = all except gi
        Xo = Y[mask]
        C = rng.standard_normal((n_cand, d))
        C = _norm(C)
        dots = C @ Xo.T
        md = dots.max(axis=1)
        bi = int(np.argmin(md))
        x = C[bi]
        # quick tangential refine vs Xo
        for _ in range(80):
            dv = Xo @ x
            mx = float(dv.max())
            w = np.exp(np.clip(40.0 * (dv - mx), -30.0, 0.0))
            g = (w[:, None] * Xo).sum(axis=0)
            g = g - float(g @ x) * x
            gn = float(np.linalg.norm(g))
            if gn < 1e-12:
                break
            x = x - 0.03 * g / (gn + 1e-12)
            x = x / np.linalg.norm(x)
        Y[gi] = x
    return Y


def solve():
    t_start = time.time()
    T_END = t_start + 172.0
    T_EXPL = t_start + 146.0
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (131, 9) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng.standard_normal((131, 9))
        cands.append(_norm(X0))
    bestX = None
    best = 1e9
    for a in cands:
        v = _max_cos(a)
        if v < best:
            best = v
            bestX = a.copy()
    archive = [bestX.copy()]
    n, d = bestX.shape
    trial = 0
    while time.time() < T_EXPL:
        rem_expl = T_EXPL - time.time()
        if rem_expl < 0.5:
            break
        # pick base
        if len(archive) > 1 and rng.random() < 0.25:
            base = archive[int(rng.integers(len(archive)))]
        else:
            base = bestX
        mode = trial % 7
        X = base.copy()
        do_subset = False
        S = None
        steps, a0, a1, lr0 = 2500, 30.0, 260.0, 0.003
        if mode == 0:
            sig = 0.012
            X = X + sig * rng.standard_normal(X.shape)
            X = _norm(X)
            steps, a0, a1, lr0 = 2200, 40.0, 300.0, 0.0025
        elif mode == 1:
            sig = 0.05
            X = X + sig * rng.standard_normal(X.shape)
            X = _norm(X)
            steps, a0, a1, lr0 = 3200, 20.0, 240.0, 0.004
        elif mode == 2:
            # jammed kick + rattler to holes
            ppm, G = _per_point(X)
            gmax = float(ppm.max())
            ratt = np.where(ppm < gmax - 1e-3)[0]
            thr = gmax - 5e-3
            cnts = np.sum(G > thr, axis=1)
            order = np.argsort(-cnts)
            nj = min(12, n)
            jam = order[:nj]
            X[jam] += 0.06 * rng.standard_normal((nj, d))
            X = _norm(X)
            if len(ratt) > 0:
                X = _hole_reinit(X, ratt[:12], rng, n_cand=2500)
            steps, a0, a1, lr0 = 3000, 25.0, 260.0, 0.003
        elif mode == 3:
            # subset reinit worst 12 via holes then subset smooth
            ppm, G = _per_point(X)
            order = np.argsort(-ppm)
            k = 12
            S = order[:k]
            X = _hole_reinit(X, S, rng, n_cand=3000)
            do_subset = True
            steps, a0, a1, lr0 = 1800, 25.0, 280.0, 0.005
        elif mode == 4:
            # larger subset randomize
            ppm, G = _per_point(X)
            order = np.argsort(-ppm)
            k = 20
            S = np.concatenate([order[:k], rng.choice(n, size=4, replace=False)])
            S = np.unique(S)
            X[S] = _norm(rng.standard_normal((len(S), d)))
            # small noise elsewhere
            X = X + 0.004 * rng.standard_normal(X.shape)
            X = _norm(X)
            do_subset = True
            steps, a0, a1, lr0 = 2000, 15.0, 220.0, 0.006
        elif mode == 5:
            # from-scratch random start
            X = _norm(rng.standard_normal((n, d)))
            do_subset = False
            steps, a0, a1, lr0 = 3800, 12.0, 200.0, 0.005
        else:
            sig = 0.11
            X = X + sig * rng.standard_normal(X.shape)
            X = _norm(X)
            steps, a0, a1, lr0 = 3400, 18.0, 220.0, 0.004
        # cap steps by remaining time (rough)
        if rem_expl < 8 and steps > 1500:
            steps = 1500
        if do_subset:
            Xs, vs = _smooth_subset(X, S, steps, a0, a1, lr0, deadline=T_EXPL)
            # follow with short global
            if time.time() < T_EXPL - 0.3:
                Xs, vs2 = _smooth_global(Xs, min(900, steps // 2), 60.0, 300.0, 0.002, deadline=T_EXPL)
                vs = min(vs, vs2) if vs2 < vs else vs
                # _smooth_global returns its own best; need consistent Xs
                # recompute: keep returned
                X_after = Xs
                vs = _max_cos(X_after) if vs2 > vs else vs
                # actually Xs is best from second call; use it
                Xs = X_after
                vs = _max_cos(Xs)
            else:
                vs = _max_cos(Xs)
        else:
            Xs, vs = _smooth_global(X, steps, a0, a1, lr0, deadline=T_EXPL)
        # archive handling
        if vs < best - 1e-12:
            best = vs
            bestX = Xs.copy()
            archive.append(Xs.copy())
            if len(archive) > 10:
                # drop a middle element to keep diversity + best
                del archive[1]
            # strong LP on improvement
            b = min(3.0, max(0.4, T_EXPL - time.time() - 0.2))
            if b > 0.35 and time.time() < T_END - 1.0:
                Xp, vp = _lp_polish(Xs, b, T_END)
                if vp < best:
                    best = vp
                    bestX = Xp.copy()
                    archive[-1] = Xp.copy()
        else:
            if vs < best + 0.006 and len(archive) < 10 and (trial % 2 == 0):
                archive.append(Xs.copy())
            # occasional LP for near-best to change topology via exact squeeze
            if vs < best + 0.002 and (trial % 4 == 0):
                b = min(1.2, max(0.35, T_EXPL - time.time() - 0.2))
                if b > 0.35 and time.time() < T_END - 1.0:
                    Xp, vp = _lp_polish(Xs, b, T_END)
                    if vp < best - 1e-12:
                        best = vp
                        bestX = Xp.copy()
                        archive.append(Xp.copy())
                        if len(archive) > 10:
                            del archive[1]
        trial += 1
    # final intensive polish: subset on jammed + global LP
    rem = T_END - time.time()
    if rem > 2.0 and bestX is not None:
        # one round of jammed-subset refinement
        try:
            ppm, G = _per_point(bestX)
            gmax = float(ppm.max())
            order = np.argsort(-ppm)
            S = order[:16]
            b1 = min(4.0, max(1.0, rem - 6.0))
            if b1 > 0.5 and rem > 7.0:
                Xs, vs = _smooth_subset(bestX, S, 1200, 30.0, 300.0, 0.004, deadline=T_END - 5.0)
                if vs < best:
                    best = vs
                    bestX = Xs.copy()
        except Exception:
            pass
        rem2 = T_END - time.time()
        if rem2 > 1.2:
            b = min(rem2 - 0.7, 24.0)
            if b > 0.6:
                try:
                    Xp, vp = _lp_polish(bestX, b, T_END - 0.4)
                    if vp < best:
                        best = vp
                        bestX = Xp.copy()
                except Exception:
                    pass
    bestX = _norm(bestX)
    return bestX
