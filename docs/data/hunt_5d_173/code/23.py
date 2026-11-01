import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return X / n


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _lp_polish(X, deadline, r0=2e-3, cap=3500):
    X = _norm(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    try:
        while time.time() < deadline and radius > 1e-14:
            Gfull = X @ X.T
            c = Gfull[iu0]
            m = float(c.max())
            thr = m - max(20.0 * radius, 2e-9)
            sel = c > thr
            if np.count_nonzero(sel) > cap:
                idx = np.where(sel)[0]
                order = np.argsort(-c[idx])[:cap]
                mask = np.zeros_like(sel, dtype=bool)
                mask[idx[order]] = True
                sel = mask
            I = iu0[0][sel]
            J = iu0[1][sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            b_ub = -c[sel]
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = X + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
            if v < best:
                X, best = y, v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
    except Exception:
        pass
    return X


def _adam_smooth(Xinit, steps, s0, s1, lr0, lr1, deadline=None):
    X = Xinit.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    vv = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    ls0 = float(np.log(s0))
    ls1 = float(np.log(s1))
    ll0 = float(np.log(lr0))
    ll1 = float(np.log(lr1))
    b1p = 1.0
    b2p = 1.0
    denom = steps - 1 if steps > 1 else 1
    for t in range(1, steps + 1):
        if deadline is not None and (t & 127) == 0 and time.time() > deadline:
            break
        frac = (t - 1) / denom
        s = float(np.exp(ls0 + (ls1 - ls0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = float(np.max(G))
        # E = exp(s*(G-cm)), diag 0
        E = np.exp(s * (G - cm))
        np.fill_diagonal(E, 0.0)
        es = float(np.sum(E))
        if not np.isfinite(es) or es <= 0:
            continue
        Grad = (E @ X) / es
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # adam moments, keep tangent
        m1 *= b1
        m1 += (1.0 - b1) * Gt
        md = np.sum(m1 * X, axis=1, keepdims=True)
        m1 -= md * X
        vv *= b2
        vv += (1.0 - b2) * (Gt * Gt)
        b1p *= b1
        b2p *= b2
        mhat = m1 / (1.0 - b1p)
        vhat = vv / (1.0 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sd = np.sum(step * X, axis=1, keepdims=True)
        step -= sd * X
        # clip per-point step
        sn = np.sqrt(np.sum(step * step, axis=1, keepdims=True)) + 1e-12
        sc = np.minimum(1.0, 0.05 / (sn + 1e-12))
        step *= sc
        X -= step
        nr = np.sqrt(np.sum(X * X, axis=1, keepdims=True)) + 1e-12
        X /= nr
    return X


def solve():
    t0 = time.time()
    t_end = t0 + 170.0
    rng = np.random.RandomState(20261101)
    try:
        best = np.load("best.npy").astype(float)
    except Exception:
        try:
            best = np.load("record.npy").astype(float)
        except Exception:
            best = rng.randn(173, 5)
    best = _norm(np.asarray(best, dtype=float))
    n = 173
    if best.shape != (173, 5):
        tmp = rng.randn(173, 5)
        try:
            mm = min(best.shape[0], 173)
            dd = min(best.shape[1], 5)
            tmp[:mm, :dd] = best[:mm, :dd]
        except Exception:
            pass
        best = _norm(tmp)
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    iu = np.triu_indices(n, 1)
    Ttemp = 0.0022
    trial = 0
    while time.time() < t_end - 12.0:
        trial += 1
        X = cur.copy()
        r = rng.rand()
        kind = ""
        if trial % 30 == 0 and rng.rand() < 0.45:
            X = _norm(rng.randn(n, 5))
            kind = "fresh"
        elif r < 0.20:
            sigma = 0.008 + rng.rand() * 0.025
            X = X + rng.randn(n, 5) * sigma
            kind = "small"
        elif r < 0.40:
            k = int(rng.choice([3, 5, 8, 12, 20]))
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.08 + rng.rand() * 0.16
            X[idx] = X[idx] + rng.randn(k, 5) * sigma
            kind = "subset"
        elif r < 0.60:
            # cavity / cap rearrangement around random direction
            u = rng.randn(5)
            u /= (np.linalg.norm(u) + 1e-12)
            dots = X @ u
            msz = int(rng.randint(12, 29))
            idx = np.argsort(-dots)[:msz]
            for j in idx:
                X[j] = u + rng.randn(5) * (0.35 + rng.rand() * 0.35)
            # slight global jiggle
            X += rng.randn(n, 5) * 0.003
            kind = "cap"
        elif r < 0.73:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mm = float(np.max(G))
            cnt = np.sum(G > mm - 1e-3, axis=1)
            # move 1-2 high-contact points to voids
            order_hi = np.argsort(-cnt)
            nmove = 1 if rng.rand() < 0.6 else 2
            # find voids by sampling
            ntry = 12000
            S = _norm(rng.randn(ntry, 5))
            dots = S @ X.T
            md = np.max(dots, axis=1)
            sidx = np.argsort(md)
            chosen = []
            for ci in sidx:
                v = S[ci]
                ok = True
                for c0 in chosen:
                    if float(v @ c0) > 0.9:
                        ok = False
                        break
                if ok:
                    chosen.append(v)
                    if len(chosen) >= nmove:
                        break
            for k2 in range(nmove):
                li = int(order_hi[k2 % len(order_hi)])
                # avoid moving same twice; pick random among top 10
                if k2 == 1:
                    li = int(rng.choice(order_hi[:10]))
                X[li] = chosen[k2] + rng.randn(5) * 0.01
            X += rng.randn(n, 5) * 0.004
            kind = "void"
        elif r < 0.88:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mm = float(np.max(G))
            c = G[iu]
            topk = np.argsort(-c)[:20]
            pick = int(topk[rng.randint(len(topk))])
            a = int(iu[0][pick]); b = int(iu[1][pick])
            eps = 0.05 + rng.rand() * 0.09
            da = X[a] - X[b]
            db = -da
            da = da - float(da @ X[a]) * X[a]
            db = db - float(db @ X[b]) * X[b]
            dna = float(np.linalg.norm(da)) + 1e-12
            dnb = float(np.linalg.norm(db)) + 1e-12
            X[a] = X[a] + (da / dna) * eps + rng.randn(5) * 0.008
            X[b] = X[b] + (db / dnb) * eps + rng.randn(5) * 0.008
            com = np.where((G[a] > mm - 2e-3) | (G[b] > mm - 2e-3))[0]
            if len(com) > 0:
                X[com] = X[com] + rng.randn(len(com), 5) * 0.018
            kind = "break"
        else:
            k = int(rng.randint(1, 4))
            idx = rng.choice(n, size=k, replace=False)
            X[idx] = _norm(rng.randn(k, 5))
            X += rng.randn(n, 5) * 0.004
            kind = "reinit"
        X = _norm(X)
        if kind == "fresh":
            steps, s0, s1, lr0, lr1 = 1500, 40.0, 550.0, 0.008, 0.0008
        elif kind == "cap":
            steps, s0, s1, lr0, lr1 = 1000, 60.0, 550.0, 0.007, 0.0008
        elif kind == "small":
            steps, s0, s1, lr0, lr1 = 600, 80.0, 500.0, 0.005, 0.0007
        elif kind == "void":
            steps, s0, s1, lr0, lr1 = 900, 60.0, 550.0, 0.006, 0.0008
        elif kind == "break":
            steps, s0, s1, lr0, lr1 = 800, 70.0, 550.0, 0.006, 0.0008
        else:
            steps, s0, s1, lr0, lr1 = 800, 60.0, 520.0, 0.006, 0.0008
        adead = min(time.time() + 3.0, t_end - 12.0)
        try:
            X2 = _adam_smooth(X, steps, s0, s1, lr0, lr1, deadline=adead)
        except Exception:
            continue
        X2 = _norm(X2)
        v2 = _max_cos(X2)
        X3 = X2
        v3 = v2
        # polish promising trials with LP
        thresh = min(best_val + 0.003, cur_val + 0.005)
        if v2 < thresh and time.time() < t_end - 12.0:
            try:
                X3 = _lp_polish(X2, deadline=min(time.time() + 1.6, t_end - 12.0), r0=2e-3)
                X3 = _norm(X3)
                v3 = _max_cos(X3)
            except Exception:
                X3, v3 = X2, v2
        if v3 < best_val - 1e-12:
            best, best_val = X3.copy(), float(v3)
            cur, cur_val = X3.copy(), float(v3)
        elif v3 < cur_val:
            cur, cur_val = X3.copy(), float(v3)
        else:
            if v3 < 0.90 and rng.rand() < np.exp((cur_val - v3) / Ttemp):
                cur, cur_val = X3.copy(), float(v3)
        if trial % 25 == 0 and rng.rand() < 0.5:
            cur, cur_val = best.copy(), best_val
        if best_val < 0.70:
            break
    try:
        best = _lp_polish(best, deadline=t_end - 1.0, r0=1.5e-3)
    except Exception:
        pass
    best = _norm(best)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr < 1e-12] = 1.0
    best = best / nr
    return best
