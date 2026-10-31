import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1.5e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if deadline is not None and time.time() > deadline - 0.4:
            break
        G = x @ x.T
        iu = iu0
        c_all = G[iu]
        m = float(c_all.max())
        sel_thr = m - max(25.0 * radius, 2e-9)
        sel = c_all > sel_thr
        I = iu[0][sel]; J = iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        # build sparse A_ub : [x_J for d_I, x_I for d_J, -1 for t]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_I = (I[:, None] * d + np.arange(d)).ravel()
        cols_J = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=int)
        cols[0::(2*d+1)] = cols_I[0::d]
        # simpler: concatenate
        cols = np.concatenate([ (I[:, None]*d+np.arange(d)), (J[:,None]*d+np.arange(d)), np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            # accept, grow
            x, best = y, v
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
        # early stop if radius tiny and no progress
    return x

def repel_anneal(x, steps, lr, temp0, temp1, margin, mu):
    n = x.shape[0]
    v = np.zeros_like(x)
    for s in range(steps):
        frac = s / max(steps, 1)
        temp = temp0 * (temp1 / temp0) ** frac if temp0>0 and temp1>0 else temp0
        lrc = lr * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        D = G - cmax
        # mask near-max
        M = D > -margin
        # exp weights
        W = np.zeros_like(G)
        # avoid overflow: D<=0 so exp<=1
        W[M] = np.exp(D[M] / temp)
        scale = float(W.sum() * 0.5) + 1e-12
        G2 = W * G
        rowsum = G2.sum(axis=1, keepdims=True)
        grad = (W @ x) - rowsum * x
        grad /= scale
        v = mu * v + grad
        x = x - lrc * v
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def largest_voids(x_keep, m, rng, S=15000):
    y = rng.standard_normal(size=(S, 4))
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    scores = (y @ x_keep.T).max(axis=1)
    order = np.argsort(scores)
    picked = []
    for idx in order:
        if len(picked) >= m:
            break
        cand = y[idx]
        ok = True
        for q in picked:
            if float(cand @ q) > 0.80:
                ok = False
                break
        if ok:
            picked.append(cand)
    k = 0
    while len(picked) < m:
        picked.append(y[order[k]])
        k += 1
    return np.array(picked)

def single_point_polish(x, rng, budget_sec, deadline):
    # greedy: try to move worst points to lower their own max
    n = x.shape[0]
    t0 = time.time()
    best = max_cos(x)
    while time.time() - t0 < budget_sec and time.time() < deadline - 0.3:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        worst = G.max(axis=1)  # per-point worst
        order = np.argsort(-worst)
        improved = False
        for idx in order[:12]:
            if time.time() > deadline - 0.4:
                break
            cur_w = float(worst[idx])
            if cur_w < best - 1e-12:
                continue
            # find void direction for this point: sample candidates
            S = 8000
            cand = rng.standard_normal(size=(S, 4))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            # score = max dot to others (excluding idx)
            dots = cand @ x.T
            # ignore self column by setting to -2
            dots[:, idx] = -2.0
            sc = dots.max(axis=1)
            bi = int(np.argmin(sc))
            if float(sc[bi]) < cur_w - 2e-6:
                old = x[idx].copy()
                x[idx] = cand[bi]
                # local refine this point alone: gradient away from top-k
                for _ in range(120):
                    d = x[idx] @ x.T
                    d[idx] = -2.0
                    cm = float(d.max())
                    sel = np.where(d > cm - 0.02)[0]
                    g = x[sel].sum(axis=0)
                    # project tangent
                    g = g - (g @ x[idx]) * x[idx]
                    nrm = float(np.linalg.norm(g))
                    if nrm < 1e-12:
                        break
                    g /= nrm
                    step = 0.01
                    trial = x[idx] - step * g
                    trial /= np.linalg.norm(trial)
                    nd = float((trial @ x.T[np.arange(n)!=idx]).max())
                    if nd < cm:
                        x[idx] = trial
                    else:
                        # try smaller
                        step = 0.002
                        trial2 = x[idx] - step * g
                        trial2 /= np.linalg.norm(trial2)
                        nd2 = float((trial2 @ x.T[np.arange(n)!=idx]).max())
                        if nd2 < cm:
                            x[idx] = trial2
                        else:
                            break
                v = max_cos(x)
                if v < best:
                    best = v
                    improved = True
                else:
                    # revert if not global improvement but keep if point improved a lot?
                    # keep change if it doesn't worsen global beyond tiny epsilon, to allow later moves
                    if v > best + 5e-5:
                        x[idx] = old
                    else:
                        # keep, recompute best
                        if v < best:
                            best = v
            # small random tangent tries for fine squeeze
            for _ in range(6):
                tang = rng.standard_normal(size=4)
                tang -= (tang @ x[idx]) * x[idx]
                tn = float(np.linalg.norm(tang))
                if tn < 1e-12:
                    continue
                tang /= tn
                for st in (0.002, 0.0007, 0.0002):
                    tr = x[idx] - st * tang
                    # actually move away from worst neighbor roughly; try both signs
                    tr /= np.linalg.norm(tr)
                    # quick per-point check
                    nd = float((tr @ x.T[np.arange(n)!=idx]).max())
                    if nd < cur_w:
                        old = x[idx].copy()
                        x[idx] = tr
                        v = max_cos(x)
                        if v < best - 1e-13:
                            best = v
                            cur_w = nd
                            improved = True
                            break
                        elif v > best + 2e-6:
                            x[idx] = old
                        else:
                            cur_w = nd
                            break
        if not improved:
            break
    return x

def solve():
    rng = np.random.default_rng(12345)
    np.random.seed(12345)
    t0 = time.time()
    DEADLINE = t0 + 172.0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(np.asarray(xb, dtype=float))
    if xr is not None:
        cands.append(np.asarray(xr, dtype=float))
    if len(cands) == 0:
        cands.append(rng.standard_normal(size=(163, 4)))
    # fix shapes, normalize, pick best
    best = 1e9
    x_best = None
    for c in cands:
        c = np.asarray(c, dtype=float)
        if c.shape != (163, 4):
            tmp = rng.standard_normal(size=(163, 4))
            if c.ndim == 2:
                m0 = min(c.shape[0], 163); d0 = min(c.shape[1], 4)
                tmp[:m0, :d0] = c[:m0, :d0]
            c = tmp
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        # clean nans
        if not np.all(np.isfinite(c)):
            c = rng.standard_normal(size=(163, 4))
            c /= np.linalg.norm(c, axis=1, keepdims=True)
        v = max_cos(c)
        if v < best:
            best = v; x_best = c.copy()
    x_cur = x_best.copy()
    cur = best
    n, d = 163, 4
    it = 0
    while time.time() < DEADLINE - 17.0:
        it += 1
        if it < 4 or rng.random() < 0.68:
            base = x_best; base_val = best
        else:
            base = x_cur; base_val = cur
        r = rng.random()
        x_try = base.copy()
        if r < 0.28:
            sigma = float(10.0 ** rng.uniform(-3.0, -0.7))
            x_try = base + sigma * rng.standard_normal(size=(n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.55:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            thr = base_val - 0.01
            deg = (G > thr).sum(axis=1)
            if rng.random() < 0.6:
                center = int(np.argmax(deg + rng.random(n) * 0.6))
            else:
                center = int(rng.integers(n))
            k = int(rng.integers(5, 30))
            dots = base @ base[center]
            idx = np.argsort(-dots)[:k]
            sigma = float(rng.uniform(0.02, 0.15))
            x_try[idx] = base[idx] + sigma * rng.standard_normal(size=(k, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.78:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            thr = base_val - 0.015
            deg = (G > thr).sum(axis=1)
            m = int(rng.integers(1, 5))
            if rng.random() < 0.5:
                reloc = np.argsort(deg + rng.random(n) * 0.3)[:m]  # rattlers / low deg
            else:
                reloc = np.argsort(-deg + rng.random(n) * 0.3)[:m]  # overloaded
            keep = np.ones(n, dtype=bool); keep[reloc] = False
            voids = largest_voids(base[keep], m, rng, S=15000)
            voids = voids + 0.004 * rng.standard_normal(size=voids.shape)
            voids /= np.linalg.norm(voids, axis=1, keepdims=True)
            x_try[reloc] = voids
            if rng.random() < 0.5:
                x_try[keep] = base[keep] + 0.008 * rng.standard_normal(size=(n-m, d))
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            # cap re-init: pick random direction, replace k nearest
            u = rng.standard_normal(size=4); u /= np.linalg.norm(u)
            dots = base @ u
            k = int(rng.integers(4, 18))
            idx = np.argsort(-dots)[:k]
            # random uniform replacements, biased around -u or uniform
            if rng.random() < 0.5:
                rep = rng.standard_normal(size=(k, 4))
                rep /= np.linalg.norm(rep, axis=1, keepdims=True)
            else:
                # random in cap around u with some spread
                rep = u[None, :] + 0.6 * rng.standard_normal(size=(k, 4))
                rep /= np.linalg.norm(rep, axis=1, keepdims=True)
            x_try[idx] = rep
            # slight shake rest
            if rng.random() < 0.4:
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        if time.time() > DEADLINE - 18.0:
            break
        steps = int(rng.integers(400, 1600))
        lr = float(rng.uniform(0.05, 0.18))
        temp0 = float(10.0 ** rng.uniform(-2.5, -1.9))
        temp1 = temp0 * float(rng.uniform(0.2, 0.6))
        margin = float(rng.uniform(0.02, 0.08))
        mu = float(rng.uniform(0.4, 0.9))
        x_try = repel_anneal(x_try, steps, lr, temp0, temp1, margin, mu)
        # quick check: skip LP if far worse
        vq = max_cos(x_try)
        if vq > cur + 0.008 and vq > best + 0.003:
            # accept as wander with small prob to keep diversity, else skip polish
            if rng.random() < 0.1:
                x_cur = x_try.copy(); cur = vq
            continue
        if time.time() > DEADLINE - 17.5:
            v = vq
        else:
            sec = float(rng.uniform(0.7, 1.5))
            sec = min(sec, max(0.5, DEADLINE - 17.0 - time.time()))
            x_try = lp_polish(x_try, sec, deadline=DEADLINE-16.0)
            v = max_cos(x_try)
        if v < best - 1e-12:
            x_best = x_try.copy(); best = v
            x_cur = x_try.copy(); cur = v
        elif v < cur + 0.002 and rng.random() < 0.5:
            x_cur = x_try.copy(); cur = v
        if it % 15 == 0 and cur > best + 0.002:
            x_cur = x_best.copy(); cur = best
    # final intensive phase
    remain = DEADLINE - time.time() - 0.8
    if remain > 3.0:
        # alternate repel fine + LP + single-point
        # fine repel
        try:
            x_best = repel_anneal(x_best, steps=2500, lr=0.025, temp0=0.004, temp1=0.0009, margin=0.035, mu=0.8)
        except Exception:
            pass
        remain2 = DEADLINE - time.time() - 0.8
        if remain2 > 2.0:
            # single-point greedy with limited budget
            try:
                x_best = single_point_polish(x_best, rng, budget_sec=min(4.0, remain2*0.25), deadline=DEADLINE)
            except Exception:
                pass
        remain3 = DEADLINE - time.time() - 0.5
        if remain3 > 1.0:
            x_best = lp_polish(x_best, min(remain3, 14.0), deadline=DEADLINE)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety: replace non-finite
    if not np.all(np.isfinite(x_best)):
        x_best = x_cur / np.linalg.norm(x_cur, axis=1, keepdims=True)
    return x_best
