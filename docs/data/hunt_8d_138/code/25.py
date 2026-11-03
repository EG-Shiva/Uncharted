import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _cheap_jam(x, steps, beta0, beta1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        # geometric beta, linear lr
        beta = beta0 * (beta1 / beta0) ** f if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # W in (0,1], max=1
        # reuse G memory for exp
        np.exp(beta * (G - m), out=G)
        np.fill_diagonal(G, 0.0)
        Grad = G @ x
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Grad -= dot * x
        x = x - lr * Grad
        # renormalize
        nr = np.sqrt((x * x).sum(axis=1, keepdims=True)) + 1e-30
        x = x / nr
    return x

def _refine_void(v0, Xrem, beta=150.0, steps=180, lr0=0.08):
    v = v0 / (np.linalg.norm(v0) + 1e-30)
    for t in range(steps):
        dots = Xrem @ v
        mv = dots.max()
        W = np.exp(beta * (dots - mv))
        S = W.sum() + 1e-300
        g = (W @ Xrem) / S
        gt = g - np.dot(g, v) * v
        st = lr0 / (1.0 + t * 0.03)
        v = v - st * gt
        v = v / (np.linalg.norm(v) + 1e-30)
    return v

def _best_void_vec(Xrem, rng, ncand=12000):
    d = Xrem.shape[1]
    cand = rng.standard_normal(size=(ncand, d))
    cand /= (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-30)
    smax = (cand @ Xrem.T).max(axis=1)
    bi = int(np.argmin(smax))
    v0 = cand[bi]
    try:
        v0 = _refine_void(v0, Xrem)
    except Exception:
        pass
    return v0

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iuI, iuJ]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:3000]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iuI[sel]; J = iuJ[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(7)
    n, d = 138, 8
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
                if np.isfinite(v).all():
                    c = _max_cos(v)
                    if c < best_val:
                        best_val = c
                        best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal(size=(n, d))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    # quick tight polish of incumbent to have true value (short)
    # do not spend too much yet
    incumbent = best.copy()
    inc_val = best_val

    it = 0
    lp_calls = 0
    # main loop: leave 18s for final polish
    while time.time() < deadline - 18.0 and it < 5000:
        it += 1
        mode = it % 6
        # --- perturb ---
        if mode == 0 and it > 1:
            # fresh random basin (diversity)
            y = rng.standard_normal(size=(n, d))
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            steps, b0, b1, l0, l1 = 900, 10.0, 350.0, 0.03, 0.0015
        else:
            # compute per-point max for targeting
            G = incumbent @ incumbent.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            order = np.argsort(-per)
            y = incumbent.copy()
            if mode == 1:
                # large core shake
                pool = order[:45]
                k = 12
                idx = rng.choice(pool, size=k, replace=False)
                sigma_core = 0.35
                noise = rng.standard_normal(size=(k, d))
                dot = (noise * y[idx]).sum(axis=1, keepdims=True)
                noise -= dot * y[idx]
                y[idx] += sigma_core * noise
                # tiny global jitter
                gj = rng.standard_normal(size=(n, d)) * 0.005
                dot2 = (gj * y).sum(axis=1, keepdims=True)
                gj -= dot2 * y
                y += gj
                steps, b0, b1, l0, l1 = 650, 20.0, 300.0, 0.02, 0.0015
            elif mode == 2:
                # medium core shake + void rebirths of worst
                pool = order[:40]
                k = 6
                idx = rng.choice(pool, size=k, replace=False)
                noise = rng.standard_normal(size=(k, d))
                dot = (noise * y[idx]).sum(axis=1, keepdims=True)
                noise -= dot * y[idx]
                y[idx] += 0.16 * noise
                y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
                # rebirth 3 worst
                for _ in range(3):
                    Gt = y @ y.T
                    np.fill_diagonal(Gt, -2.0)
                    pert = Gt.max(axis=1)
                    o2 = np.argsort(-pert)
                    # random among top 8 to keep deterministic diversity
                    victim = int(rng.choice(o2[:8]))
                    Xrem = np.delete(y, victim, axis=0)
                    vnew = _best_void_vec(Xrem, rng, ncand=8000)
                    y[victim] = vnew
                steps, b0, b1, l0, l1 = 650, 20.0, 320.0, 0.02, 0.0015
            elif mode == 3:
                # small jitter + 2 rebirths
                gj = rng.standard_normal(size=(n, d))
                dot2 = (gj * y).sum(axis=1, keepdims=True)
                gj -= dot2 * y
                y = y + 0.03 * gj
                y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
                for _ in range(2):
                    Gt = y @ y.T
                    np.fill_diagonal(Gt, -2.0)
                    pert = Gt.max(axis=1)
                    o2 = np.argsort(-pert)
                    victim = int(rng.choice(o2[:10]))
                    Xrem = np.delete(y, victim, axis=0)
                    y[victim] = _best_void_vec(Xrem, rng, ncand=8000)
                steps, b0, b1, l0, l1 = 550, 25.0, 350.0, 0.015, 0.001
            elif mode == 4:
                # neighbor dispersal around most jammed point
                center = int(order[0])
                # nearest neighbors by dot
                dots_c = (y @ y[center])
                dots_c[center] = -2.0
                nn = np.argsort(-dots_c)[:9]
                noise = rng.standard_normal(size=(len(nn), d))
                dot = (noise * y[nn]).sum(axis=1, keepdims=True)
                noise -= dot * y[nn]
                # push away from center
                push = y[nn] - y[center][None, :]
                # tangentize push
                pdot = (push * y[nn]).sum(axis=1, keepdims=True)
                push -= pdot * y[nn]
                y[nn] += 0.22 * push + 0.12 * noise
                # shake center strongly
                nc = rng.standard_normal(size=(d,))
                nc -= np.dot(nc, y[center]) * y[center]
                y[center] += 0.30 * nc
                y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
                steps, b0, b1, l0, l1 = 650, 20.0, 300.0, 0.02, 0.0015
            else:  # mode 5
                gj = rng.standard_normal(size=(n, d))
                dot2 = (gj * y).sum(axis=1, keepdims=True)
                gj -= dot2 * y
                y = y + 0.10 * gj
                y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
                # random re-insert 2 uniform
                idxr = rng.choice(n, size=2, replace=False)
                rp = rng.standard_normal(size=(2, d))
                rp /= (np.linalg.norm(rp, axis=1, keepdims=True) + 1e-30)
                y[idxr] = rp
                steps, b0, b1, l0, l1 = 700, 15.0, 300.0, 0.022, 0.0015
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)

        # cheap jam
        try:
            y = _cheap_jam(y, steps=steps, beta0=b0, beta1=b1, lr0=l0, lr1=l1)
        except Exception:
            continue
        try:
            v = _max_cos(y)
        except Exception:
            continue
        # screen: only LP if promising
        thresh = inc_val + (0.006 if lp_calls < 8 else 0.0025)
        # fresh random needs stricter to save time
        if mode == 0:
            thresh = inc_val + 0.0015
        if v < thresh and time.time() < deadline - 20.0:
            # short LP
            lp_budget = min(2.2, max(0.8, (deadline - 18.0 - time.time()) / 8.0))
            if lp_budget < 0.6:
                continue
            try:
                y2 = _lp_polish(y, time.time() + lp_budget)
                v2 = _max_cos(y2)
            except Exception:
                continue
            lp_calls += 1
            if v2 < inc_val - 1e-12:
                incumbent = y2.copy()
                inc_val = v2
                # keep global best
                if v2 < best_val:
                    best_val = v2
                    best = y2.copy()
        else:
            # even without LP, accept if cheap beats incumbent by margin (rare)
            if v < inc_val - 1e-9:
                # polish it next iteration via LP eventually; store as candidate
                # do immediate LP if very good
                if v < inc_val - 0.0005 and time.time() < deadline - 20.0:
                    try:
                        y2 = _lp_polish(y, time.time() + 2.0)
                        v2 = _max_cos(y2)
                        lp_calls += 1
                        if v2 < inc_val - 1e-12:
                            incumbent = y2.copy()
                            inc_val = v2
                            if v2 < best_val:
                                best_val = v2
                                best = y2.copy()
                    except Exception:
                        pass

    # final tight polish of best incumbent
    try:
        final_deadline = min(deadline, time.time() + 16.0)
        best_out = _lp_polish(incumbent if incumbent is not None else best, final_deadline)
        fv = _max_cos(best_out)
        if fv < best_val:
            best = best_out
    except Exception:
        best = incumbent if incumbent is not None else best
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    # safety: replace non-finite
    if not np.isfinite(best).all():
        best = incumbent / np.linalg.norm(incumbent, axis=1, keepdims=True)
    return best
