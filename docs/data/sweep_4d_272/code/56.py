import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    return x / n

def _adam_lse(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** f if tau1 > 0 else tau0
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        G -= gm
        G /= tau
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = G.sum()
        if not np.isfinite(S) or S <= 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * (g * g)
        mh = m / (1.0 - b1 ** (k + 1))
        vh = v / (1.0 - b2 ** (k + 1))
        st = mh / (np.sqrt(vh) + eps)
        np.clip(st, -2.0, 2.0, out=st)
        x = x - lr * st
        x = _norm(x)
    return x

def _adam_hinge(x, steps, T, lr0, lr1, rng=None, noise=0.0):
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        f = k / (steps - 1) if steps > 1 else 0.0
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        G -= T
        np.maximum(G, 0.0, out=G)
        np.fill_diagonal(G, 0.0)
        if G.max() <= 0:
            if rng is not None:
                rz = rng.standard_normal((n, d))
                rd = (rz * x).sum(axis=1, keepdims=True)
                rz -= rd * x
                rn = np.linalg.norm(rz, axis=1, keepdims=True) + 1e-12
                x = _norm(x + 0.002 * (rz / rn))
                m *= 0.5
                v *= 0.5
                continue
            else:
                break
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        if noise > 0 and rng is not None:
            nz = rng.standard_normal((n, d))
            nd = (nz * x).sum(axis=1, keepdims=True)
            nz -= nd * x
            g = g + noise * nz
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * (g * g)
        mh = m / (1.0 - b1 ** (k + 1))
        vh = v / (1.0 - b2 ** (k + 1))
        st = mh / (np.sqrt(vh) + eps)
        np.clip(st, -3.0, 3.0, out=st)
        x = x - lr * st
        x = _norm(x)
    return x

def _void_against(fixed, rng, n_cand=6000):
    cand = rng.standard_normal((n_cand, 4))
    cand = _norm(cand)
    s = cand @ fixed.T
    sc = s.max(axis=1)
    bi = int(np.argmin(sc))
    return cand[bi]

def _patch_void_refill(x, patch, rng, n_cand=4000):
    n = x.shape[0]
    y = x.copy()
    mask_fixed = np.ones(n, dtype=bool)
    mask_fixed[patch] = False
    order = np.array(patch, copy=True)
    rng.shuffle(order)
    placed = []
    for p in order:
        if len(placed) == 0:
            others = y[mask_fixed]
        else:
            others = np.concatenate([y[mask_fixed], np.stack(placed)], axis=0)
        cand = rng.standard_normal((n_cand, 4))
        cand = _norm(cand)
        s = cand @ others.T
        sc = s.max(axis=1)
        bi = int(np.argmin(sc))
        y[p] = cand[bi]
        placed.append(cand[bi])
    return y

def _lp_polish(x, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.empty((p, 2 * d + 1))
        cc[:, :d] = x[J]
        cc[:, d:2 * d] = x[I]
        cc[:, 2 * d] = -1.0
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        A_ub = coo_matrix((cc.ravel(), (rows, cols)), shape=(p, nv)).tocsr()
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
        y = _norm(x + res.x[:-1].reshape(n, d))
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    rng = np.random.default_rng(777)
    t0 = time.time()
    deadline = t0 + 172.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((272, 4))
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (272, 4):
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            flat = np.concatenate([flat, rng.standard_normal((need - flat.size,))])
        x0 = flat[:need].reshape(272, 4)
    x0 = np.where(np.isfinite(x0), x0, 0.0)
    x0 = _norm(x0)
    x_best = x0.copy()
    best_val = _max_cos(x_best)
    n = 272
    sigmas = [0.02, 0.05, 0.10, 0.16, 0.03, 0.07, 0.012, 0.22]
    trial = 0
    while time.time() < t0 + 118 and time.time() < deadline - 52:
        trial += 1
        r = rng.random()
        if trial % 9 == 0 and time.time() < t0 + 80:
            x = rng.standard_normal((n, 4))
            x = _norm(x)
            x = _adam_hinge(x, 300, best_val - 0.002, 0.015, 0.002, rng=rng)
        else:
            sigma = sigmas[trial % len(sigmas)]
            x = _norm(x_best + sigma * rng.standard_normal((n, 4)))
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            mx = G.max()
            thr = mx - 1e-9
            cnt = (G >= thr).sum(axis=1)
            pw = G.max(axis=1)
            mode = trial % 5
            if mode == 0 or mode == 1:
                worst = int(np.argmax(pw))
                sims = x[worst] @ x.T
                sims[worst] = -2.0
                k = 5 + int(rng.integers(0, 5))
                nn = np.argsort(-sims)[:k]
                patch = list(nn) + [worst]
                ratt = np.where(cnt == 0)[0]
                for rr in ratt[:3]:
                    if int(rr) not in patch:
                        patch.append(int(rr))
                patch = sorted(set(int(a) for a in patch))
                x = _patch_void_refill(x, patch, rng, n_cand=4000)
            elif mode == 2:
                nk = 2 + (trial % 4)
                for _ in range(nk):
                    gg = x @ x.T
                    np.fill_diagonal(gg, -2.0)
                    idx = int(gg.argmax())
                    ii, jj = divmod(idx, n)
                    victim = ii if rng.random() < 0.5 else jj
                    msk = np.ones(n, dtype=bool)
                    msk[victim] = False
                    x[victim] = _void_against(x[msk], rng, n_cand=6000)
            elif mode == 3:
                hub = int(np.argmax(cnt))
                sims = x[hub] @ x.T
                sims[hub] = -2.0
                nn = np.argsort(-sims)[:4]
                for q in list(nn) + [hub]:
                    x[q] = _norm(x[q] + 0.25 * rng.standard_normal(4))
                zr = np.where(cnt == 0)[0]
                for q in zr:
                    msk = np.ones(n, dtype=bool)
                    msk[int(q)] = False
                    x[int(q)] = _void_against(x[msk], rng, n_cand=6000)
            else:
                Q, _ = np.linalg.qr(rng.standard_normal((4, 4)))
                if np.linalg.det(Q) < 0:
                    Q[:, 0] = -Q[:, 0]
                c0 = x[rng.integers(n)]
                dots = x @ c0
                cl = np.where(dots > 0.5)[0]
                if len(cl) > 4:
                    x[cl] = _norm((x[cl] @ Q.T) * 1.0)
                    x = _norm(x + 0.01 * rng.standard_normal((n, 4)))
        if r < 0.5:
            gap = float(rng.uniform(0.0006, 0.008))
            Tt = best_val - gap
            lr0 = 0.012 if (trial % 2 == 0) else 0.007
            x = _adam_hinge(x, 450, Tt, lr0, 0.001, rng=rng, noise=0.0004 if (trial % 3 == 0) else 0.0)
        else:
            if trial % 3 == 0:
                x = _adam_lse(x, 450, 0.012, 0.001, 0.008, 0.0008)
            elif trial % 3 == 1:
                x = _adam_lse(x, 450, 0.005, 0.0005, 0.005, 0.0004)
            else:
                x = _adam_hinge(x, 250, best_val - 0.0015, 0.010, 0.001, rng=rng)
                x = _adam_lse(x, 250, 0.006, 0.0006, 0.005, 0.0005)
        v = _max_cos(x)
        if v < best_val:
            if v < best_val - 0.00025:
                x2 = _lp_polish(x, 3.5, deadline - 1.0)
                v2 = _max_cos(x2)
                if v2 <= v:
                    x, v = x2, v2
            best_val = v
            x_best = x.copy()
            if time.time() < deadline - 55:
                xr = _adam_lse(x_best.copy(), 280, 0.004, 0.0005, 0.0035, 0.00035)
                vr = _max_cos(xr)
                if vr < best_val:
                    best_val = vr
                    x_best = xr.copy()
    for _ in range(5):
        if time.time() > deadline - 40:
            break
        if rng.random() < 0.6:
            x = _adam_lse(x_best.copy(), 320, 0.0035, 0.00045, 0.0028, 0.0003)
        else:
            x = _adam_hinge(x_best.copy(), 320, best_val - 0.00035, 0.005, 0.0005, rng=rng)
        v = _max_cos(x)
        if v < best_val:
            best_val = v
            x_best = x.copy()
    remain = deadline - time.time() - 2.0
    if remain > 5:
        xp = _lp_polish(x_best, min(remain, 34.0), deadline - 1.0)
        vp = _max_cos(xp)
        if vp <= best_val:
            x_best = xp
            best_val = vp
    if not np.all(np.isfinite(x_best)):
        x_best = x0.copy()
    x_best = _norm(x_best)
    return x_best
