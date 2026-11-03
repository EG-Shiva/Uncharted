import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, budget, init_r=1e-3):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = init_r
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < budget and radius > 1e-14:
        if time.time() - t0 + 0.25 > budget and radius < 5e-4:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        thresh = max(20.0 * radius, 1e-9)
        sel = c > m - thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2800:
            kth = np.partition(c, -2800)[-2800]
            sel = c >= kth
            cnt = int(np.sum(sel))
            if cnt > 3200:
                idx = np.where(sel)[0]
                order = np.argsort(-c[idx])[:3200]
                mask = np.zeros_like(sel)
                mask[idx[order]] = True
                sel = mask
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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

def _anneal(X, steps, T0, T1, lr0, lr1, mu, rng):
    n, d = X.shape
    Y = X.copy()
    V = np.zeros_like(Y)
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1)
        WY = W @ Y
        r = np.sum(W * G, axis=1)
        inv = np.zeros_like(rs)
        mk = rs > 1e-12
        inv[mk] = 1.0 / rs[mk]
        Dir = WY * inv[:, None] - (r * inv)[:, None] * Y
        Dir[~mk] = 0.0
        V *= mu
        V += Dir
        Y -= lr * V
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Y /= nrm
    return Y

def _refine_void(v, Xkeep, iters=30):
    for _ in range(iters):
        dv = Xkeep @ v
        m = float(np.max(dv))
        w = np.exp(np.clip((dv - m) / 0.02, -50, 0))
        g = w @ Xkeep
        g = g - np.dot(g, v) * v
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        g /= ng
        v = v - 0.02 * g
        nv = np.linalg.norm(v)
        if nv == 0:
            break
        v /= nv
    return v

def _worst_relocate(X, iters, ncand, rng):
    n, d = X.shape
    Y = X.copy()
    for _ in range(iters):
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        # per-point worst
        Gtmp = G.copy()
        np.fill_diagonal(Gtmp, -1.0)
        # max per row excluding diag(=filled -1, but max>0 so fine)
        s = np.max(Gtmp, axis=1)
        m = float(np.max(s))
        order = np.argsort(-s)[:5]
        # candidate pool
        C = rng.standard_normal((ncand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
        D = C @ Y.T  # ncand x n
        # for fast m_excl: precompute full sorted? just compute per ordered idx on demand
        improved = False
        for idx in order:
            idx = int(idx)
            # sc excluding idx
            D2 = D.copy()
            D2[:, idx] = -10.0
            sc = np.max(D2, axis=1)
            bi = int(np.argmin(sc))
            bval = float(sc[bi])
            # m excluding idx
            # mask row/col idx
            # compute max of Gtmp excluding idx quickly
            # Gtmp includes -1 diag, -2? we set diag -1, off-diag true. Mask:
            col = Gtmp[:, idx].copy()
            row = Gtmp[idx, :].copy()
            # temporarily set to -2
            Gtmp[:, idx] = -2.0
            Gtmp[idx, :] = -2.0
            m_excl = float(np.max(Gtmp))
            # restore
            Gtmp[:, idx] = col
            Gtmp[idx, :] = row
            gnew = bval if bval > m_excl else m_excl
            if gnew < m - 1e-12:
                v = C[bi].copy()
                # refine against keep set
                mask = np.ones(n, dtype=bool); mask[idx] = False
                Xkeep = Y[mask]
                v = _refine_void(v, Xkeep, iters=25)
                # re-evaluate refined
                dnew = Xkeep @ v
                sc2 = float(np.max(dnew))
                gnew2 = sc2 if sc2 > m_excl else m_excl
                if gnew2 < m:
                    Y[idx] = v / np.linalg.norm(v)
                    improved = True
                    break
                else:
                    # still accept unrefined if it improved
                    Y[idx] = C[bi] / np.linalg.norm(C[bi])
                    improved = True
                    break
        # if no improvement, continue with new random pool
    return Y

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(12345)
    n, d = 181, 9
    base = None
    base_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < base_val:
                        base_val = v
                        base = a.copy()
        except Exception:
            pass
    if base is None:
        base = rng.standard_normal((n, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        base_val = _max_cos(base)

    seeds = []
    # symmetric seed: 90 + 90 antipodes + 1
    try:
        R = rng.standard_normal((90, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        pole = rng.standard_normal((1, d))
        pole /= np.linalg.norm(pole, axis=1, keepdims=True)
        sym = np.vstack([R, -R, pole])
        # shuffle deterministically
        perm = rng.permutation(n)
        sym = sym[perm]
        seeds.append(sym)
    except Exception:
        pass
    # random seeds
    for _ in range(4):
        A = rng.standard_normal((n, d))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        seeds.append(A)
    # destroyed-best seed: remove 30 worst, greedy refill
    try:
        Y = base.copy()
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        np.fill_diagonal(G, 1.0)
        contacts = np.sum(G > m - 1e-7, axis=1) - 1
        worst = np.argsort(-contacts)[:30]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[worst] = False
        Xkeep = Y[keep_mask]
        # sequential farthest insertion
        newpts = []
        cur = Xkeep.copy()
        for _ in range(30):
            C = rng.standard_normal((3000, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            D = C @ cur.T
            sc = np.max(D, axis=1)
            bi = int(np.argmin(sc))
            v = C[bi]
            newpts.append(v)
            cur = np.vstack([cur, v[None, :]])
        Y2 = np.vstack([Xkeep, np.array(newpts)])
        # reorder to original size (already 181)
        seeds.append(Y2)
    except Exception:
        pass

    cands = []
    # pipeline each seed with time check
    for si, S in enumerate(seeds):
        rem = BUDGET - (time.time() - t0)
        if rem < 30:
            break
        try:
            Y = _anneal(S, steps=2200, T0=0.03, T1=0.0025, lr0=0.15, lr1=0.02, mu=0.70, rng=rng)
            Y = _worst_relocate(Y, iters=60, ncand=3500, rng=rng)
            Y = _anneal(Y, steps=1200, T0=0.008, T1=0.0012, lr0=0.08, lr1=0.015, mu=0.70, rng=rng)
            Y = _worst_relocate(Y, iters=30, ncand=3500, rng=rng)
            v = _max_cos(Y)
            cands.append((v, Y))
        except Exception:
            continue

    cands.sort(key=lambda t: t[0])
    # keep top 2 + base
    top = []
    for v, Y in cands[:2]:
        top.append((v, Y))
    top.append((base_val, base.copy()))
    # LP polish in order of promise, time-aware
    best_val = base_val
    best_X = base.copy()
    # if pipeline found better raw, update
    for v, Y in cands:
        if v < best_val:
            best_val = v
            best_X = Y.copy()
    for v, Y in sorted(top, key=lambda t: t[0]):
        rem = BUDGET - (time.time() - t0)
        if rem < 4:
            break
        budg = min(rem - 1.0, 22.0 if v > best_val + 1e-6 else 35.0)
        if budg < 2.0:
            continue
        try:
            Z = _lp_polish(Y, budget=float(budg), init_r=0.01)
            vz = _max_cos(Z)
            if vz < best_val:
                best_val = vz
                best_X = Z.copy()
        except Exception:
            continue
        rem2 = BUDGET - (time.time() - t0)
        if rem2 < 3:
            break
    # final short squeeze on best if time left
    remf = BUDGET - (time.time() - t0)
    if remf > 3.0:
        try:
            Z = _lp_polish(best_X, budget=min(remf - 0.8, 12.0), init_r=1e-3)
            vz = _max_cos(Z)
            if vz < best_val:
                best_val = vz
                best_X = Z.copy()
        except Exception:
            pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
