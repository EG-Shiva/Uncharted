import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    x = x / n
    x[~np.isfinite(x)] = 0.0
    r = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (r[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thr=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    c = (g >= m - thr).sum(axis=1)
    return c, m, g

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _adam(x, steps, tau0, tau1, lr0, lr1, mov=None):
    n, d = x.shape
    if mov is None:
        mov = np.arange(n)
        full = True
    else:
        mov = np.asarray(mov, dtype=int)
        full = False
    k = len(mov)
    m = np.zeros((k, d))
    v = np.zeros((k, d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        f = s / max(steps - 1, 1)
        tau = tau0 * (tau1 / tau0) ** f if tau1 != tau0 else tau0
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        W = np.exp((G - gm) / tau)
        np.fill_diagonal(W, 0.0)
        W[W < 1e-13] = 0.0
        grad = W @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        g = gt[mov]
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        st = (m / bc1) / (np.sqrt(v / bc2) + eps)
        np.clip(st, -2.0, 2.0, out=st)
        x[mov] = x[mov] - lr * st
        rn = np.linalg.norm(x[mov], axis=1, keepdims=True)
        rn[rn < 1e-12] = 1.0
        x[mov] = x[mov] / rn
    return x

def _void_greedy(fixed, k, rng, n_cand=3000):
    out = []
    others = fixed
    for _ in range(k):
        cand = rng.standard_normal((n_cand, 4))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        if len(others) > 0:
            s = cand @ others.T
            sc = s.max(axis=1)
        else:
            sc = np.zeros(n_cand)
        if len(out) > 0:
            pl = np.stack(out)
            s2 = cand @ pl.T
            sc = np.maximum(sc, s2.max(axis=1))
        bi = int(np.argmin(sc))
        out.append(cand[bi])
        others = np.vstack([others, cand[bi][None, :]]) if len(others) > 0 else cand[bi][None, :]
    return np.stack(out)

def _hopf_code(n_base, per, rng):
    # Fibonacci base on S2
    nb = n_base
    idx = np.arange(nb) + 0.5
    z = 1.0 - 2.0 * idx / nb
    r = np.sqrt(np.maximum(0.0, 1 - z * z))
    ga = np.pi * (3.0 - np.sqrt(5.0))
    ph = ga * np.arange(nb) + rng.uniform(0, 2 * np.pi)
    bx = r * np.cos(ph); by = r * np.sin(ph); bz = z
    pts = []
    for j in range(nb):
        x, y, zz = bx[j], by[j], bz[j]
        den = np.sqrt(max(2.0 * (1.0 + zz), 1e-12))
        if zz > -0.999999:
            z1r = np.sqrt(max((1.0 + zz) / 2.0, 0.0)); z1i = 0.0
            z2r = x / den; z2i = y / den
        else:
            z1r, z1i, z2r, z2i = 0.0, 0.0, 1.0, 0.0
        shift = rng.uniform(0, 2 * np.pi)
        for a in range(per):
            t = shift + 2 * np.pi * a / per
            ct, st = np.cos(t), np.sin(t)
            # (z1 e^{it}, z2 e^{it})
            r1r = z1r * ct - z1i * st; r1i = z1r * st + z1i * ct
            r2r = z2r * ct - z2i * st; r2i = z2r * st + z2i * ct
            pts.append([r1r, r1i, r2r, r2i])
    pts = np.array(pts, dtype=float)
    # random SO(4) rotation for diversity
    A = rng.standard_normal((4, 4))
    Q, _ = np.linalg.qr(A)
    pts = pts @ Q.T
    return _norm(pts)

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = _norm(np.asarray(xb, dtype=float))
    if xb.shape != (272, 4):
        xb = _norm(rng.standard_normal((272, 4)))
    best_x = xb.copy()
    best_v = _max_cos(best_x)

    cands = []

    # 1) symmetric Hopf restart -> collective basin
    try:
        xh = _hopf_code(34, 8, rng)
        # xh has 272 points
        xh = _adam(xh, 500, 0.02, 0.004, 0.03, 0.008)
        xh = _adam(xh, 500, 0.006, 0.0015, 0.012, 0.004)
        vh = _max_cos(xh)
        cands.append((vh, xh.copy()))
        if vh < best_v:
            best_v, best_x = vh, xh.copy()
    except Exception:
        pass

    # 2) patch surgery trials around hubs + rattlers
    # precompute order each trial from current best
    trial_K = [16, 24, 32, 20, 28, 12]
    for ti, K in enumerate(trial_K):
        if time.time() > HARD - 55:
            break
        try:
            cnt, mval, G = _contacts(best_x, thr=1e-9)
        except Exception:
            cnt = np.zeros(272, dtype=int)
        ratt = np.where(cnt == 0)[0]
        hub = int(np.argmax(cnt))
        # worst pair
        try:
            G2 = best_x @ best_x.T
            np.fill_diagonal(G2, -2.0)
            wi, wj = np.unravel_index(int(G2.argmax()), G2.shape)
            center = wi if (ti % 2 == 0) else hub
            if ti % 3 == 2:
                center = wj
        except Exception:
            center = hub
        sims = best_x @ best_x[center]
        sims[center] = 2.0
        patch = np.argsort(-sims)[:K]
        if len(ratt) > 0:
            # force rattlers in, drop farthest non-center points
            patch_set = set(patch.tolist())
            for r in ratt:
                if r not in patch_set:
                    # replace last
                    patch = np.append(patch[:-1], r)
                    patch_set.add(r)
            patch = np.unique(patch)
        patch = np.asarray(patch, dtype=int)
        # resample patch into voids of frozen core
        mask = np.ones(272, dtype=bool); mask[patch] = False
        fixed = best_x[mask]
        try:
            refill = _void_greedy(fixed, len(patch), rng, n_cand=2500)
        except Exception:
            refill = rng.standard_normal((len(patch), 4))
            refill /= np.linalg.norm(refill, axis=1, keepdims=True)
        x0 = best_x.copy()
        x0[patch] = refill
        # small jitter core to allow escape
        x0 = _norm(x0 + 0.002 * rng.standard_normal(x0.shape) * 0.0)
        x0[patch] = _norm(refill)
        # joint patch-against-core anneal then full anneal
        tau0 = 0.015 if ti % 2 == 0 else 0.025
        x0 = _adam(x0, 350, tau0, 0.004, 0.025, 0.006, mov=patch)
        x0 = _adam(x0, 350, 0.006, 0.0015, 0.012, 0.004, mov=None)
        v0 = _max_cos(x0)
        cands.append((v0, x0.copy()))
        if v0 < best_v + 0.004:
            # snap with short LP to see true basin value
            dl = min(HARD - 30, time.time() + 6.0)
            try:
                xp, vp = _lp_polish(x0, dl)
            except Exception:
                xp, vp = x0, v0
            cands.append((vp, xp.copy()))
            if vp < best_v:
                best_v, best_x = vp, xp.copy()
        elif v0 < best_v:
            best_v, best_x = v0, x0.copy()
    # 3) polish top candidates
    cands.sort(key=lambda kv: kv[0])
    # keep unique-ish top 3
    top = cands[:3] if len(cands) >= 3 else cands
    for _, xc in top:
        if time.time() > HARD - 10:
            break
        try:
            dl = min(HARD - 2, time.time() + max(10.0, (HARD - time.time()) / 2.0))
            xp, vp = _lp_polish(xc, dl)
            if vp < best_v:
                best_v, best_x = vp, xp.copy()
        except Exception:
            continue
    # final LP on global best with remaining time
    if time.time() < HARD - 3:
        try:
            xp, vp = _lp_polish(best_x, HARD - 1.0)
            if vp < best_v:
                best_v, best_x = vp, xp.copy()
        except Exception:
            pass
    out = _norm(best_x)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    rn = np.linalg.norm(out, axis=1, keepdims=True)
    rn[rn < 1e-12] = 1.0
    out = out / rn
    return out
