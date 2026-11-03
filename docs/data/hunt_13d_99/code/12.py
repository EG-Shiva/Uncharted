import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def exp_refine(cur0, alpha, steps, lr, beta=0.9):
    cur = cur0.copy()
    # best tracking uses pre-step max (m from G)
    G0 = cur @ cur.T
    np.fill_diagonal(G0, -2.0)
    best_s = float(G0.max())
    best = cur.copy()
    mom = np.zeros_like(cur)
    for s in range(steps):
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        if m < best_s:
            best_s = m
            best = cur.copy()
        # weights
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        step = mom / bc
        cur_lr = lr * (1.0 - 0.5 * s / max(steps, 1))
        cur = cur - cur_lr * step
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
    # final check
    fv = max_cos(best)
    # also check last cur
    lv = max_cos(cur)
    if lv < fv:
        return cur, lv
    return best, fv


def hard_refine(cur0, steps, lr, margin=0.01, beta=0.85):
    cur = cur0.copy()
    G0 = cur @ cur.T
    np.fill_diagonal(G0, -2.0)
    best_s = float(G0.max())
    best = cur.copy()
    mom = np.zeros_like(cur)
    n = cur.shape[0]
    for s in range(steps):
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        if m < best_s:
            best_s = m
            best = cur.copy()
        # binary mask of near-max pairs
        W = (G >= m - margin).astype(float)
        np.fill_diagonal(W, 0.0)
        deg = W.sum(axis=1, keepdims=True) + 1e-12
        # tangential repulsion: sum_j (xj - c*xi)
        Grad = W @ cur
        wdot = (W * G).sum(axis=1, keepdims=True)
        Gt = Grad - wdot * cur
        Gt = Gt / deg
        # normalize scale to avoid huge steps when margin large
        # scale by mean norm
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        step = mom / bc
        cur_lr = lr * (1.0 - 0.5 * s / max(steps, 1))
        cur = cur - cur_lr * step
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
    fv = max_cos(best)
    lv = max_cos(cur)
    if lv < fv:
        return cur, lv
    return best, fv


def far_replace(base, idx, rng, n_cand=2500):
    x0 = base.copy()
    n, d = base.shape
    for k in idx:
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # dots to all other current points
        others = np.delete(x0, k, axis=0)
        dots = cand @ others.T
        md = dots.max(axis=1)
        j = int(np.argmin(md))
        x0[k] = cand[j]
    return x0


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
        v = max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if best < max_cos(x) - 1e-12:
            x = best_x.copy()
    return best_x


def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPLORE_UNTIL = 132.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    if (n, d) != (99, 13):
        xb = rng.standard_normal(size=(99, 13))
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
        n, d = 99, 13
    best_val = max_cos(xb)
    best_x = xb.copy()
    pop = [(best_val, best_x.copy())]

    sigmas = [0.02, 0.05, 0.10, 0.18, 0.30, 0.45]
    alphas = [25.0, 40.0, 60.0, 90.0, 130.0]

    it = 0
    while time.time() - t0 < EXPLORE_UNTIL and it < 1000:
        it += 1
        # pick base
        if len(pop) == 1 or rng.random() < 0.75:
            base = best_x
        else:
            k = int(rng.integers(0, len(pop)))
            base = pop[k][1]
        mode = rng.random()
        sigma = sigmas[int(rng.integers(0, len(sigmas)))]
        x0 = None
        if mode < 0.30:
            noise = rng.standard_normal(size=(n, d)) * sigma
            x0 = base + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode < 0.55:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            # bias to high rowmax but sometimes low contacts
            kk = int(rng.integers(2, 10))
            idx = np.argsort(rowmax)[-kk:]
            x0 = base.copy()
            nz = rng.standard_normal(size=(kk, d)) * (sigma * 1.5 + 0.02)
            x0[idx] = x0[idx] + nz
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        elif mode < 0.70:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            # contact count with threshold 1e-3
            contacts = (G >= m - 1e-3).sum(axis=1)
            # score: high rowmax + high contacts -> overloaded
            rowmax = G.max(axis=1)
            # combine: prefer high contacts
            order = np.argsort(rowmax + 0.002 * contacts)
            kk = int(rng.integers(1, 6))
            idx = order[-kk:]
            x0 = far_replace(base, idx, rng, n_cand=2500)
        elif mode < 0.88:
            # coherent cap shift: move a random cap together
            u = rng.standard_normal(size=(d,))
            u /= np.linalg.norm(u)
            proj = base @ u
            kk = int(rng.integers(5, 21))
            thr = np.sort(proj)[-kk]
            mask = proj >= thr
            idxm = np.where(mask)[0]
            sigma_c = float(rng.uniform(0.05, 0.25))
            sigma_i = float(rng.uniform(0.01, 0.05))
            common = rng.standard_normal(size=(d,)) * sigma_c
            x0 = base.copy()
            x0[idxm] = x0[idxm] + common + rng.standard_normal(size=(len(idxm), d)) * sigma_i
            # small global jitter to rest
            x0 += rng.standard_normal(size=(n, d)) * 0.005
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        else:
            # strong global shake + far reinsert of worst
            noise = rng.standard_normal(size=(n, d)) * (sigma + 0.15)
            x0 = base + noise
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
            # reinsert single worst far
            G = x0 @ x0.T
            np.fill_diagonal(G, -2.0)
            w = int(np.argmax(G.max(axis=1)))
            x0 = far_replace(x0, np.array([w]), rng, n_cand=3000)

        alpha = alphas[int(rng.integers(0, len(alphas)))]
        steps1 = int(rng.integers(700, 1600))
        lr1 = float(rng.uniform(0.015, 0.06))
        x1, s1 = exp_refine(x0, alpha, steps1, lr1)
        # second stage: higher alpha focus
        if rng.random() < 0.7:
            alpha2 = alpha * 1.6 + 20.0
            x1, s1 = exp_refine(x1, alpha2, 500, lr1 * 0.5)
        # hard-max greedy finish
        if rng.random() < 0.8:
            marg = float(rng.uniform(0.003, 0.02))
            lr2 = float(rng.uniform(0.005, 0.02))
            st2 = int(rng.integers(200, 600))
            x1, s1 = hard_refine(x1, st2, lr2, margin=marg)
        s1 = max_cos(x1)
        if s1 < best_val - 1e-12:
            best_val = s1
            best_x = x1.copy()
            pop.append((s1, x1.copy()))
            pop = sorted(pop, key=lambda z: z[0])[:5]
        elif len(pop) < 5 and s1 < pop[-1][0]:
            pop.append((s1, x1.copy()))
            pop = sorted(pop, key=lambda z: z[0])[:5]
        elif rng.random() < 0.015 and len(pop) < 5:
            pop.append((s1, x1.copy()))
            pop = sorted(pop, key=lambda z: z[0])[:5]

    remain = BUDGET - (time.time() - t0) - 1.0
    if remain > 5:
        best_x = lp_polish(best_x, remain * 0.85)
        best_val = max_cos(best_x)
        remain2 = BUDGET - (time.time() - t0) - 1.0
        if remain2 > 8 and len(pop) > 1:
            # try runner-up briefly
            try:
                qx = lp_polish(pop[1][1], min(remain2 * 0.4, 10.0))
                qv = max_cos(qx)
                if qv < best_val:
                    best_val = qv
                    best_x = qx.copy()
            except Exception:
                pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
