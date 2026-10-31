import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam_quench(X0, steps, b0, b1, lr0, lr1):
    X = _norm(X0.copy())
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1p = 1.0
    b2p = 1.0
    be1, be2 = 0.9, 0.999
    eps = 1e-8
    for it in range(1, steps + 1):
        f = it / float(steps)
        beta = b0 * (b1 / b0) ** f
        lr = lr0 + (lr1 - lr0) * f
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(np.max(G))
        W = np.exp(beta * (G - cmax))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W))
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (W @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= be1
        b2p *= be2
        m *= be1
        m += (1.0 - be1) * Gt
        v *= be2
        v += (1.0 - be2) * Gt * Gt
        mh = m / (1.0 - b1p)
        vh = v / (1.0 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        X = X - step
        X = _norm(X)
    return X

def _top_hubs(X, nhub=2, topk=300):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    iu0 = np.triu_indices(n, 1)
    c = G[iu0]
    k = min(topk, c.size)
    idx = np.argpartition(c, -k)[-k:]
    I = iu0[0][idx]
    J = iu0[1][idx]
    cnt = np.zeros(n, dtype=int)
    for a in I:
        cnt[a] += 1
    for a in J:
        cnt[a] += 1
    order = np.argsort(-cnt)
    return order[:nhub], cnt

def _hole_for_fixed(fixed, rng, ntry=2500, refine=60):
    d = fixed.shape[1]
    C = rng.standard_normal((ntry, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ fixed.T
    w = np.max(dots, axis=1)
    y = C[int(np.argmin(w))].copy()
    y /= np.linalg.norm(y)
    step = 0.06
    for _ in range(refine):
        dd = fixed @ y
        j = int(np.argmax(dd))
        g = fixed[j] - float(fixed[j] @ y) * y
        nn = float(np.linalg.norm(g))
        if nn < 1e-12:
            break
        g /= nn
        y = y - step * g
        ny = float(np.linalg.norm(y))
        if ny < 1e-12:
            break
        y /= ny
        step *= 0.97
    return y / np.linalg.norm(y)

def _hub_regrow(X, rng, nhub=2):
    n, d = X.shape
    hubs, _ = _top_hubs(X, nhub=nhub)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[hubs] = False
    fixed = X[keep_mask]
    Y = X.copy()
    cur_fixed = fixed.copy()
    for h in hubs:
        y = _hole_for_fixed(cur_fixed, rng)
        # small random offset to avoid exact ties
        y = y + rng.standard_normal(d) * 0.002
        y /= np.linalg.norm(y)
        Y[h] = y
        cur_fixed = np.vstack([cur_fixed, y.reshape(1, d)])
    return _norm(Y)

def _subset_rotate(X, k, theta, rng):
    n, d = X.shape
    idx = rng.choice(n, size=k, replace=False)
    a = rng.standard_normal(d)
    a /= np.linalg.norm(a)
    b = rng.standard_normal(d)
    b = b - float(a @ b) * a
    nb = float(np.linalg.norm(b))
    if nb < 1e-12:
        return _norm(X)
    b /= nb
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    R = np.eye(d) + (c - 1.0) * (np.outer(a, a) + np.outer(b, b)) + s * (np.outer(a, b) - np.outer(b, a))
    Y = X.copy()
    Y[idx] = Y[idx] @ R.T
    return _norm(Y)

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.5:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1300:
            o = np.argsort(c[idx])[-1300:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]
        J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            best = vv
            bx = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            if vv <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    rng = np.random.default_rng(7)
    best_val = 1e9
    best_X = None

    def _try_arr(a):
        global_vars = None
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return
        if a.shape != (116, 6):
            return
        if not np.all(np.isfinite(a)):
            return
        nn = np.linalg.norm(a, axis=1)
        if np.any(nn < 1e-12):
            return
        a = a / nn[:, None]
        v = _maxcos(a)
        nonlocal_best = [v, a]
        return nonlocal_best

    # load incumbents
    cands_load = []
    for f in ("best.npy", "record.npy"):
        try:
            a = np.load(f)
            a = np.asarray(a, dtype=float)
            if a.shape != (116, 6):
                continue
            if not np.all(np.isfinite(a)):
                continue
            nn = np.linalg.norm(a, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            a = a / nn
            v = _maxcos(a)
            cands_load.append((v, a))
        except Exception:
            continue
    if cands_load:
        cands_load.sort(key=lambda t: t[0])
        best_val, best_X = cands_load[0][0], cands_load[0][1].copy()
    else:
        best_X = _norm(rng.standard_normal((116, 6)))
        best_val = _maxcos(best_X)

    cur_X = best_X.copy()
    cur_val = best_val

    it = 0
    # reserve ~25s for final LP
    while time.time() < HARD - 25.0:
        it += 1
        mode = it % 6
        try:
            if mode == 0:
                sigma = float(rng.choice(np.array([0.004, 0.010, 0.025])))
                Xp = cur_X + rng.standard_normal((116, 6)) * sigma
                Xp = _norm(Xp)
                steps = 380
                b1 = 1400.0 + float(rng.integers(0, 600))
            elif mode == 1:
                nh = 1 if (rng.random() < 0.5) else 2
                # occasionally 3
                if rng.random() < 0.15:
                    nh = 3
                Xp = _hub_regrow(cur_X, rng, nhub=nh)
                steps = 420
                b1 = 1500.0 + float(rng.integers(0, 600))
            elif mode == 2:
                k = int(rng.integers(8, 30))
                th = float(rng.uniform(0.3, 1.2))
                Xp = _subset_rotate(cur_X, k, th, rng)
                # add tiny jitter to rotated block to break symmetry
                Xp = Xp + rng.standard_normal((116, 6)) * 0.002
                Xp = _norm(Xp)
                steps = 420
                b1 = 1400.0 + float(rng.integers(0, 600))
            elif mode == 3:
                sigma = float(rng.choice(np.array([0.008, 0.020, 0.045])))
                Xp = best_X + rng.standard_normal((116, 6)) * sigma
                Xp = _norm(Xp)
                steps = 400
                b1 = 1600.0 + float(rng.integers(0, 500))
            elif mode == 4:
                # worst-pair surgery: move one endpoint of worst pair to hole
                G = cur_X @ cur_X.T
                np.fill_diagonal(G, -2.0)
                ij = np.unravel_index(int(np.argmax(G)), G.shape)
                keep = np.ones(116, dtype=bool)
                # remove one endpoint (higher degree one)
                # quick degree estimate
                M = float(np.max(G))
                deg = np.sum(G > (M - 0.003), axis=1)
                victim = ij[0] if deg[ij[0]] >= deg[ij[1]] else ij[1]
                keep[victim] = False
                fixed = cur_X[keep]
                y = _hole_for_fixed(fixed, rng)
                Xp = cur_X.copy()
                Xp[victim] = y
                Xp = _norm(Xp + rng.standard_normal((116, 6)) * 0.002)
                steps = 420
                b1 = 1500.0 + float(rng.integers(0, 600))
            else:
                # occasional fresh random restart (rare)
                if it % 30 == 0:
                    Xp = _norm(rng.standard_normal((116, 6)))
                    steps = 550
                    b1 = 1700.0
                else:
                    sigma = 0.015
                    Xp = cur_X + rng.standard_normal((116, 6)) * sigma
                    Xp = _norm(Xp)
                    steps = 350
                    b1 = 1400.0
            # adapt steps if little time
            remain = (HARD - 25.0) - time.time()
            if remain < 10:
                steps = min(steps, 250)
            Xq = _adam_quench(Xp, steps=steps, b0=60.0, b1=b1, lr0=0.04, lr1=0.002)
            vq = _maxcos(Xq)
            # extra refine if promising
            if vq < best_val + 0.002 and remain > 8:
                Xq2 = _adam_quench(Xq, steps=200, b0=200.0, b1=2200.0, lr0=0.015, lr1=0.001)
                vq2 = _maxcos(Xq2)
                if vq2 < vq:
                    Xq = Xq2
                    vq = vq2
            if vq < best_val:
                best_val = vq
                best_X = Xq.copy()
                cur_X = Xq.copy()
                cur_val = vq
            elif vq < cur_val + 0.004:
                # wandering acceptance
                if rng.random() < 0.65:
                    cur_X = Xq.copy()
                    cur_val = vq
            else:
                # if stuck, occasionally reset cur to best with kick
                if it % 25 == 0:
                    cur_X = best_X + rng.standard_normal((116, 6)) * 0.02
                    cur_X = _norm(cur_X)
                    cur_val = _maxcos(cur_X)
            if time.time() > HARD - 25.0:
                break
        except Exception:
            continue

    # final elite LP polish with remaining time
    final_X = best_X.copy()
    final_v = best_val
    # also consider cur if close
    extra = []
    try:
        if cur_val < final_v + 0.003:
            extra.append(cur_X.copy())
    except Exception:
        pass
    # polish best first
    remain = HARD - time.time()
    # leave 3s margin
    budget_best = min(18.0, max(remain - 4.0, 2.0)) if remain > 3 else 0
    if budget_best > 1.0:
        try:
            Xp, vp = _lp_polish(final_X, time.time() + budget_best)
            if vp < final_v:
                final_v = vp
                final_X = Xp.copy()
        except Exception:
            pass
    # polish second candidate if time
    for Xc in extra[:1]:
        rem2 = HARD - time.time()
        if rem2 < 6.0:
            break
        try:
            Xp, vp = _lp_polish(Xc, time.time() + min(8.0, rem2 - 2.0))
            if vp < final_v:
                final_v = vp
                final_X = Xp.copy()
        except Exception:
            pass
    # safety: compare with loaded files in case hopping failed
    for v0, a0 in cands_load:
        if v0 < final_v:
            final_v = v0
            final_X = a0.copy()
    final_X = _norm(final_X)
    # strict finite check
    final_X = np.where(np.isfinite(final_X), final_X, 0.0)
    nn = np.linalg.norm(final_X, axis=1, keepdims=True)
    nn[nn < 1e-12] = 1.0
    final_X = final_X / nn
    return final_X
