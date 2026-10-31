import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_relax(x, iters, k0, k1, lr0, lr1, mu):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = G.max()
        # exp weighting; shift by max for stability
        W = np.exp(k * (G - moff))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        if mu > 0:
            v = mu * v + gt
            vc = v / (1.0 - mu ** (t + 1))
            x = x - lr * vc
        else:
            x = x - lr * gt
        # renormalize
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
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

def farthest_candidates(Xkeep, already, nc, rng):
    d = Xkeep.shape[1]
    cand = rng.standard_normal((nc, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ Xkeep.T
    md = dots.max(axis=1)
    if already is not None and len(already) > 0:
        a = np.asarray(already)
        d2 = cand @ a.T
        md = np.maximum(md, d2.max(axis=1))
    return cand, md

def solve():
    t0 = time.time()
    rng = np.random.default_rng(7)
    n, d = 153, 4
    try:
        best = np.load("best.npy")
    except Exception:
        try:
            best = np.load("record.npy")
        except Exception:
            best = rng.standard_normal((n, d))
    best = np.asarray(best, dtype=float)
    if best.shape != (n, d):
        best = rng.standard_normal((n, d))
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best_val = max_cos(best)
    incumbent = best.copy()
    inc_val = best_val
    cands = [(inc_val, incumbent.copy())]

    SEARCH_END = t0 + 118
    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        # choose base
        rb = rng.random()
        if rb < 0.55:
            base = incumbent
        elif rb < 0.78:
            base = best
        else:
            base = cands[int(rng.integers(len(cands)))][1]
        x = base.copy()
        op = rng.random()
        if op < 0.30:
            # gaussian kick
            sigma = float(rng.choice(np.array([0.02, 0.05, 0.09, 0.15, 0.25])))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        elif op < 0.60:
            # kill worst k and reinsert into voids with large samples
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            maxper = G.max(axis=1)
            k = int(rng.integers(2, 9))
            idx = np.argsort(-maxper)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idx] = False
            Xkeep = x[keep_mask]
            reinserted = []
            for j in idx:
                nc = 2500
                cand, md = farthest_candidates(Xkeep, reinserted, nc, rng)
                order = np.argsort(md)
                # pick among top few for diversity
                if rng.random() < 0.8:
                    pick = int(order[0])
                else:
                    pick = int(order[int(rng.integers(1, min(6, nc)))])
                # local hill climb on chosen void point (few steps)
                vpt = cand[pick].copy()
                vbest = float(md[pick])
                for _ in range(12):
                    step = 0.02 * rng.standard_normal(d)
                    step -= np.dot(step, vpt) * vpt
                    vt = vpt + step
                    vt /= np.linalg.norm(vt)
                    sc = float(max(np.dot(vt, Xkeep.T).max() if len(Xkeep) else -2,
                                   max([float(np.dot(vt, q)) for q in reinserted]) if reinserted else -2))
                    if sc < vbest:
                        vbest = sc
                        vpt = vt
                x[j] = vpt
                Xkeep = np.vstack([Xkeep, vpt.reshape(1, -1)])
                reinserted.append(vpt.copy())
        elif op < 0.72:
            # subset relaxation: remove worst k, relax keep alone, then reinsert
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            maxper = G.max(axis=1)
            k = int(rng.integers(3, 8))
            idx = np.argsort(-maxper)[:k]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[idx] = False
            Xkeep = x[keep_mask].copy()
            # relax subset alone briefly (fewer points -> can move more)
            try:
                Xkeep = smooth_relax(Xkeep, 350, 12.0, 200.0, 0.04, 0.003, 0.85)
            except Exception:
                pass
            x[keep_mask] = Xkeep
            # greedy reinsertion
            reinserted = []
            for j in idx:
                cand, md = farthest_candidates(x[keep_mask].copy() if not reinserted else np.vstack([x[keep_mask], np.array(reinserted)]), None, 2000, rng)
                pick = int(np.argmin(md))
                if rng.random() < 0.25:
                    order = np.argsort(md)
                    pick = int(order[int(rng.integers(1, min(5, 2000)))])
                x[j] = cand[pick]
                reinserted.append(cand[pick].copy())
        elif op < 0.86:
            # coherent cap rotation + noise
            c = rng.standard_normal(d); c /= np.linalg.norm(c)
            dots = x @ c
            thr = float(rng.uniform(0.60, 0.92))
            mask = dots > thr
            if int(mask.sum()) < 2:
                mask[int(rng.integers(n))] = True
            # random near-identity rotation
            ang = float(rng.uniform(0.05, 0.35))
            A = rng.standard_normal((d, d)); A = A - A.T
            # approximate rotation via QR of I+ang*A
            M = np.eye(d) + ang * A
            Q, R = np.linalg.qr(M)
            # fix signs to make close to identity
            s = np.sign(np.diag(R)); s[s == 0] = 1
            Q = Q * s[None, :]
            x[mask] = x[mask] @ Q.T
            sigma = float(rng.uniform(0.03, 0.12))
            x[mask] += sigma * rng.standard_normal((int(mask.sum()), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        else:
            # contact-targeted push: perturb points in tightest pairs strongly
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            # collect points involved in near-max pairs
            iu0 = np.triu_indices(n, 1)
            sel = G[iu0] > m - 2e-3
            involved = np.unique(np.concatenate([iu0[0][sel], iu0[1][sel]]))
            if len(involved) == 0:
                involved = np.array([int(np.argmax(G.max(axis=1)))])
            sigma = float(rng.uniform(0.10, 0.30))
            x[involved] += sigma * rng.standard_normal((len(involved), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)

        k0 = float(rng.choice(np.array([8.0, 12.0, 18.0])))
        k1 = float(rng.choice(np.array([120.0, 250.0, 400.0, 600.0])))
        lr0 = float(rng.choice(np.array([0.02, 0.04, 0.06])))
        lr1 = float(rng.choice(np.array([0.001, 0.0025, 0.005])))
        mu = float(rng.choice(np.array([0.0, 0.8, 0.9])))
        iters = int(rng.choice(np.array([500, 800, 1200])))
        if time.time() > SEARCH_END - 2:
            break
        try:
            xo = smooth_relax(x, iters, k0, k1, lr0, lr1, mu)
        except Exception:
            continue
        v = max_cos(xo)
        if v < inc_val - 1e-12:
            inc_val = v
            incumbent = xo.copy()
            cands.append((v, xo.copy()))
            cands = sorted(cands, key=lambda z: z[0])[:5]
        elif v < cands[-1][0] and len(cands) < 5:
            cands.append((v, xo.copy()))
            cands = sorted(cands, key=lambda z: z[0])[:5]
        # occasional pure random basin sample
        if trial % 22 == 0 and time.time() < SEARCH_END - 5:
            xr = rng.standard_normal((n, d))
            xr /= np.linalg.norm(xr, axis=1, keepdims=True)
            try:
                xo2 = smooth_relax(xr, 700, 12.0, 250.0, 0.04, 0.002, 0.85)
            except Exception:
                continue
            v2 = max_cos(xo2)
            if v2 < cands[-1][0] or len(cands) < 5:
                cands.append((v2, xo2.copy()))
                cands = sorted(cands, key=lambda z: z[0])[:5]
            if v2 < inc_val - 1e-12:
                inc_val = v2
                incumbent = xo2.copy()

    cands = sorted(cands, key=lambda z: z[0])
    # ensure original best retained
    found = any(abs(v - best_val) < 1e-15 for v, _ in cands)
    if not found:
        cands.append((best_val, best.copy()))
        cands = sorted(cands, key=lambda z: z[0])[:5]
    time_left = 170 - (time.time() - t0)
    if time_left < 12:
        time_left = 12
    # quench + polish top 3
    polished_best = None
    polished_val = 1e9
    for i, (vv, xc) in enumerate(cands[:3]):
        # quick high-k quench before LP
        try:
            xq = smooth_relax(xc.copy(), 250, 200.0, 800.0, 0.008, 0.001, 0.9)
        except Exception:
            xq = xc
        vq = max_cos(xq)
        # keep better of quenched / original
        start_pt = xq if vq < vv + 1e-6 else xc
        budget = time_left / 3.0
        if i == 0:
            budget *= 1.4
        budget = min(budget, 55)
        try:
            xp = lp_polish(start_pt, budget)
        except Exception:
            xp = start_pt
        vp = max_cos(xp)
        if vp < polished_val:
            polished_val = vp
            polished_best = xp
        if time.time() - t0 > 172:
            break
    if polished_best is None:
        polished_best = incumbent
    polished_best = polished_best / np.linalg.norm(polished_best, axis=1, keepdims=True)
    # return best between polished and loaded best
    if best_val < max_cos(polished_best):
        return best / np.linalg.norm(best, axis=1, keepdims=True)
    return polished_best
