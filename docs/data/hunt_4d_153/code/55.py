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
    rng = np.random.default_rng(0)

    try:
        best = np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            best = np.load("record.npy").astype(np.float64)
        except Exception:
            best = rng.standard_normal((153, 4))
    best = np.asarray(best, dtype=np.float64)
    if best.shape != (153, 4):
        best = rng.standard_normal((153, 4))
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    best_val = _max_cos(best)

    iu = np.triu_indices(153, 1)

    def lp_polish(x_init, deadline):
        x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
        n, d = x.shape
        cur = _max_cos(x)
        radius = 2e-3
        while time.time() < deadline and radius > 1e-14:
            G = x @ x.T
            c = G[iu]
            m = float(np.max(c))
            thr = m - max(20.0 * radius, 1e-9)
            sel = c > thr
            if np.count_nonzero(sel) > 2500:
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
            if (it % 250 == 0) and time.time() > DEAD - 2.0:
                break
            frac = it / max(1, steps - 1)
            lr = lr0 * ((lr1 / lr0) ** frac) if lr1 > 0 else lr0
            G = x @ x.T
            np.fill_diagonal(G, -10.0)
            G -= target
            np.maximum(G, 0, out=G)
            feas = float(np.max(G)) <= 0
            grad = G @ x
            dot = np.einsum('ij,ij->i', grad, x)[:, None]
            grad -= dot * x
            mg = float(np.mean(np.linalg.norm(grad, axis=1)))
            if mg > 1e-12:
                grad /= mg
                v = mu * v + (1.0 - mu) * grad
            else:
                v *= mu
                if feas and noise0 == 0:
                    break
            if noise0 > 0:
                amp = noise0 * (1.0 - frac)
                if amp > 0:
                    rnd = rng.standard_normal((n, d)) * amp
                    dd = np.einsum('ij,ij->i', rnd, x)[:, None]
                    rnd -= dd * x
                    x = x - lr * v + rnd
                else:
                    x = x - lr * v
            else:
                x = x - lr * v
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return x

    def find_hole(others, n_sample=15000, refine=150):
        cand = rng.standard_normal((n_sample, 4))
        cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
        dots = cand @ others.T
        m = dots.max(axis=1)
        order = np.argsort(m)
        best_c = None
        best_v = 1e9
        for k in order[:4]:
            c = cand[k].copy()
            for it in range(refine):
                dd = others @ c
                mx = float(np.max(dd))
                beta = 30.0 + it * 0.35
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
                nrm = float(np.linalg.norm(c))
                if nrm < 1e-12:
                    break
                c /= nrm
            vv = float(np.max(others @ c))
            if vv < best_v:
                best_v = vv
                best_c = c.copy()
        return best_c

    def multi_hole_insert(others, k):
        pts = []
        cur = others
        for _ in range(k):
            h = find_hole(cur)
            pts.append(h)
            cur = np.vstack([cur, h[None, :]])
        return np.array(pts)

    def farthest_select(pool, m, seed_idx=0):
        # pool (P,4) unit, greedily select m farthest points
        P = pool.shape[0]
        sel = [int(seed_idx % P)]
        best_dot = pool @ pool[sel[0]]
        best_dot[sel[0]] = 2.0
        for _ in range(1, m):
            nxt = int(np.argmin(best_dot))
            sel.append(nxt)
            d = pool @ pool[nxt]
            best_dot = np.minimum(best_dot, d)
            best_dot[sel] = 2.0
        return pool[np.array(sel)]

    def contact_counts(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        cnt = np.sum(G > m - 1e-9, axis=1)
        return G, m, cnt

    # population of 2: best + diverse second
    pop = [best.copy()]
    pop_val = [best_val]
    # quick diverse second from big shake + anneal
    try:
        tmp0 = best + rng.standard_normal(best.shape) * 0.35
        tmp0 /= np.maximum(np.linalg.norm(tmp0, axis=1, keepdims=True), 1e-12)
        tgt = best_val - 0.001
        tmp0 = hinge_smooth(tmp0, tgt, steps=900, lr0=0.02, lr1=0.001, noise0=0.004)
        tmp0 = hinge_smooth(tmp0, tgt, steps=500, lr0=0.006, lr1=0.0003, noise0=0.0)
        v0 = _max_cos(tmp0)
        if v0 < best_val + 0.004:
            tmp0p, v0p = lp_polish(tmp0, deadline=time.time() + 2.0)
            if v0p < v0:
                tmp0, v0 = tmp0p, v0p
            pop.append(tmp0)
            pop_val.append(v0)
        else:
            pop.append(tmp0)
            pop_val.append(v0)
    except Exception:
        pass

    def anneal_two_stage(cand, delta1=0.0015):
        tgt = best_val - delta1
        t1 = hinge_smooth(cand, tgt, steps=900, lr0=0.02, lr1=0.0009, noise0=0.0025)
        t1 = hinge_smooth(t1, tgt, steps=600, lr0=0.007, lr1=0.0003, noise0=0.0)
        return t1

    it = 0
    while time.time() < DEAD - 28.0:
        it += 1
        mode = it % 6
        try:
            if mode == 0:
                # cap deletion: geographically co-located block
                k = int(rng.integers(9, 18))
                center = rng.standard_normal(4)
                center /= max(float(np.linalg.norm(center)), 1e-12)
                dots = best @ center
                victims = np.argsort(-dots)[:k]
                keep = np.delete(np.arange(153), victims)
                others = best[keep]
                newpts = multi_hole_insert(others, k)
                cand = best.copy()
                cand[victims] = newpts
                cand = anneal_two_stage(cand, delta1=0.002)
            elif mode == 1:
                # overloaded multi-reinsert
                _, _, cnt = contact_counts(best)
                k = int(rng.integers(4, 7))
                victims = np.argsort(-cnt)[:k + 4]
                victims = rng.choice(victims, size=k, replace=False)
                keep = np.delete(np.arange(153), victims)
                others = best[keep]
                newpts = multi_hole_insert(others, k)
                cand = best.copy()
                cand[victims] = newpts
                cand = anneal_two_stage(cand, delta1=0.0018)
            elif mode == 2:
                # hemisphere twist: rotate cap by random 2-plane rotation
                u = rng.standard_normal(4)
                u /= max(float(np.linalg.norm(u)), 1e-12)
                s = best @ u
                med = float(np.median(s))
                mask = s > med
                if np.sum(mask) < 10 or np.sum(mask) > 143:
                    mask = s > 0
                a = rng.standard_normal(4); a -= float(a @ u) * u
                # make a orthonormal-ish
                an = float(np.linalg.norm(a))
                if an < 1e-9:
                    continue
                a /= an
                b = rng.standard_normal(4)
                b -= float(b @ a) * a; b -= float(b @ u) * u
                bn = float(np.linalg.norm(b))
                if bn < 1e-9:
                    continue
                b /= bn
                # re-orthogonalize a vs b
                theta = float(rng.uniform(0.12, 0.35)) * (1 if rng.random() < 0.5 else -1)
                ct, st = float(np.cos(theta)), float(np.sin(theta))
                cand = best.copy()
                Xa = cand[mask] @ a
                Xb = cand[mask] @ b
                Ya = ct * Xa + st * Xb
                Yb = -st * Xa + ct * Xb
                cand[mask] = cand[mask] + np.outer(Ya - Xa, a) + np.outer(Yb - Xb, b)
                # add small noise to interface to break symmetry
                cand += rng.standard_normal(cand.shape) * 0.008
                cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
                cand = anneal_two_stage(cand, delta1=0.0015)
            elif mode == 3:
                # global large shake
                sigma = float(rng.choice(np.array([0.12, 0.2, 0.3])))
                cand = best + rng.standard_normal(best.shape) * sigma
                cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
                cand = anneal_two_stage(cand, delta1=0.0015)
            elif mode == 4:
                # splice two parents via farthest selection
                if len(pop) >= 2:
                    # pick best and random other
                    order = np.argsort(np.array(pop_val))
                    p1 = pop[order[0]]
                    p2 = pop[order[int(rng.integers(1, len(pop)))]]
                else:
                    p1 = best
                    p2 = best + rng.standard_normal(best.shape) * 0.3
                    p2 /= np.maximum(np.linalg.norm(p2, axis=1, keepdims=True), 1e-12)
                pool = np.vstack([p1, p2])
                pool /= np.maximum(np.linalg.norm(pool, axis=1, keepdims=True), 1e-12)
                cand = farthest_select(pool, 153, seed_idx=int(rng.integers(0, 306)))
                # small jitter
                cand = cand + rng.standard_normal(cand.shape) * 0.01
                cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
                cand = anneal_two_stage(cand, delta1=0.002)
            else:
                # rattler neighbourhood shake
                G, m, cnt = contact_counts(best)
                low = np.where(cnt <= 2)[0]
                if len(low) == 0:
                    low = np.argsort(cnt)[:3]
                anchor = int(rng.choice(low))
                drow = G[anchor]
                neigh = np.argsort(-drow)[:12]
                cand = best.copy()
                cand[neigh] += rng.standard_normal((len(neigh), 4)) * 0.12
                cand[anchor] += rng.standard_normal(4) * 0.05
                cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
                # optionally reinsert anchor into hole of rest
                if rng.random() < 0.5:
                    keep = np.delete(np.arange(153), anchor)
                    hole = find_hole(cand[keep])
                    cand[anchor] = hole
                cand = anneal_two_stage(cand, delta1=0.0015)

            vv = _max_cos(cand)
            # update population diversity
            if vv < max(pop_val) + 0.001 and len(pop) < 4:
                pop.append(cand.copy()); pop_val.append(vv)
            elif len(pop) >= 2 and vv < max(pop_val):
                j = int(np.argmax(np.array(pop_val)))
                pop[j] = cand.copy(); pop_val[j] = vv

            if vv < best_val + 0.0007:
                cand2, vv2 = lp_polish(cand, deadline=time.time() + 1.8)
                if vv2 < best_val:
                    best, best_val = cand2, vv2
                    # refresh pop best slot
                    j = int(np.argmin(np.array(pop_val)))
                    pop[j] = best.copy(); pop_val[j] = best_val
                elif vv < best_val and vv2 >= best_val:
                    best, best_val = cand, vv
        except Exception:
            continue

    try:
        best, best_val = lp_polish(best, deadline=DEAD - 1.0)
    except Exception:
        pass
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    best = np.where(np.isfinite(best), best, 0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    bad = (nr.squeeze() < 1e-12) | (~np.isfinite(nr.squeeze()))
    if np.any(bad):
        rnd = rng.standard_normal(best.shape)
        rnd /= np.maximum(np.linalg.norm(rnd, axis=1, keepdims=True), 1e-12)
        best[bad] = rnd[bad]
        best /= np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    return best
