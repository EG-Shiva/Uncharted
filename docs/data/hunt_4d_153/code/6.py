import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def solve():
    t0 = time.time()
    DEAD = t0 + 170.0
    np.random.seed(0)
    rng = np.random.default_rng(0)

    # load starting point
    try:
        best = np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            best = np.load("record.npy").astype(np.float64)
        except Exception:
            best = rng.standard_normal((153, 4))
    best = np.asarray(best, dtype=np.float64)
    if best.shape != (153, 4):
        # fallback: tile/truncate deterministically
        tmp = rng.standard_normal((153, 4))
        best = tmp
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    best = best / nr
    best_val = _max_cos(best)

    def lp_polish(x_init, deadline):
        x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
        n, d = x.shape
        cur = _max_cos(x)
        radius = 1e-3
        iu = np.triu_indices(n, 1)
        while time.time() < deadline and radius > 1e-14:
            G = x @ x.T
            c = G[iu]
            m = float(np.max(c))
            thr = m - max(20.0 * radius, 1e-9)
            sel = c > thr
            # cap number of constraints for speed
            if np.count_nonzero(sel) > 2500:
                # keep tightest 2500
                idx = np.where(sel)[0]
                order = np.argsort(-c[idx])[:2500]
                mask = np.zeros_like(sel, dtype=bool)
                mask[idx[order]] = True
                sel = mask
            I = iu[0][sel]
            J = iu[1][sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate(
                [I[:, None] * d + np.arange(d),
                 J[:, None] * d + np.arange(d),
                 np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c[sel],
                              A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < cur:
                x, cur = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return x, cur

    def hinge_smooth(x_init, target, steps, lr0, lr1, noise0, mu=0.9):
        x = x_init.copy()
        n, d = x.shape
        v = np.zeros_like(x)
        for it in range(steps):
            if it % 200 == 0 and time.time() > DEAD:
                break
            frac = it / max(1, steps - 1)
            lr = lr0 * ((lr1 / lr0) ** frac) if lr1 > 0 else lr0
            G = x @ x.T
            np.fill_diagonal(G, -10.0)
            G -= target
            np.maximum(G, 0, out=G)
            feasible = float(np.max(G)) <= 0
            grad = G @ x
            dot = np.einsum('ij,ij->i', grad, x)[:, None]
            grad -= dot * x
            mg = float(np.mean(np.linalg.norm(grad, axis=1)))
            if mg > 1e-12:
                grad /= mg
                v = mu * v + (1.0 - mu) * grad
            else:
                v *= mu
                if feasible and noise0 == 0:
                    break
            if noise0 > 0:
                amp = noise0 * (1.0 - frac)
                if amp > 0:
                    rnd = rng.standard_normal((n, d)) * amp
                    d2 = np.einsum('ij,ij->i', rnd, x)[:, None]
                    rnd -= d2 * x
                    x = x - lr * v + rnd
                else:
                    x = x - lr * v
            else:
                x = x - lr * v
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return x

    def find_hole(others, n_sample=20000, refine=200):
        cand = rng.standard_normal((n_sample, 4))
        cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
        dots = cand @ others.T
        m = dots.max(axis=1)
        order = np.argsort(m)
        best_c = None
        best_v = 1e9
        for k in order[:5]:
            c = cand[k].copy()
            for it in range(refine):
                dd = others @ c
                mx = float(np.max(dd))
                beta = 30.0 + it * 0.3
                w = np.exp(beta * (dd - mx))
                s = float(np.sum(w))
                if s <= 0:
                    break
                w /= s
                gvec = w @ others
                g = gvec - float(gvec @ c) * c
                gn = float(np.linalg.norm(g))
                if gn < 1e-12:
                    break
                g /= gn
                lr = 0.02 / (1.0 + it * 0.02)
                c = c - lr * g
                c /= max(float(np.linalg.norm(c)), 1e-12)
            vv = float(np.max(others @ c))
            if vv < best_v:
                best_v = vv
                best_c = c.copy()
        return best_c

    it = 0
    # reserve ~30s for final polish
    while time.time() < DEAD - 30.0:
        it += 1
        if time.time() > DEAD - 30.0:
            break
        if it % 10 == 0:
            # full random restart exploration
            x_init = rng.standard_normal((153, 4))
            x_init /= np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
            target = best_val - 0.0015
            tmp = hinge_smooth(x_init, target, steps=1200, lr0=0.025, lr1=0.001, noise0=0.003)
            tmp = hinge_smooth(tmp, target, steps=800, lr0=0.007, lr1=0.0003, noise0=0.0)
            vv = _max_cos(tmp)
            if vv < best_val + 0.0005:
                tmp2, vv2 = lp_polish(tmp, deadline=time.time() + 2.0)
                if vv2 < best_val:
                    best, best_val = tmp2, vv2
                elif vv < best_val:
                    best, best_val = tmp, vv
            continue
        if it % 4 == 0:
            # reinsert overloaded point into largest hole
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            cnt = np.sum(G > m - 1e-9, axis=1)
            top = np.argsort(-cnt)[:12]
            victim = int(rng.choice(top))
            others = np.delete(best, victim, axis=0)
            try:
                hole = find_hole(others)
            except Exception:
                continue
            cand = best.copy()
            cand[victim] = hole
            target = best_val - 0.0015
            tmp = hinge_smooth(cand, target, steps=1100, lr0=0.02, lr1=0.0008, noise0=0.003)
            tmp = hinge_smooth(tmp, target, steps=700, lr0=0.007, lr1=0.0003, noise0=0.0)
            vv = _max_cos(tmp)
            if vv < best_val + 0.0004:
                tmp2, vv2 = lp_polish(tmp, deadline=time.time() + 2.5)
                if vv2 < best_val:
                    best, best_val = tmp2, vv2
                elif vv < best_val:
                    best, best_val = tmp, vv
        else:
            sigmas = [0.015, 0.04, 0.09, 0.03]
            sigma = sigmas[it % 4]
            pert = best + rng.standard_normal(best.shape) * sigma
            pert /= np.maximum(np.linalg.norm(pert, axis=1, keepdims=True), 1e-12)
            target = best_val - 0.0012
            tmp = hinge_smooth(pert, target, steps=1000, lr0=0.02, lr1=0.001, noise0=0.002)
            tmp = hinge_smooth(tmp, target, steps=600, lr0=0.006, lr1=0.0003, noise0=0.0)
            vv = _max_cos(tmp)
            if vv < best_val + 0.0004:
                tmp2, vv2 = lp_polish(tmp, deadline=time.time() + 2.0)
                if vv2 < best_val:
                    best, best_val = tmp2, vv2
                elif vv < best_val:
                    best, best_val = tmp, vv

    # final intensive polish
    try:
        best, best_val = lp_polish(best, deadline=DEAD)
    except Exception:
        pass
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    # strict finite guard
    best = np.where(np.isfinite(best), best, 0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    bad = (nr.squeeze() < 1e-12) | (~np.isfinite(nr.squeeze()))
    if np.any(bad):
        rnd = rng.standard_normal(best.shape)
        rnd /= np.maximum(np.linalg.norm(rnd, axis=1, keepdims=True), 1e-12)
        best[bad] = rnd[bad]
        best /= np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    return best
