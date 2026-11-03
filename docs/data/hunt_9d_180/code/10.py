import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish(x, budget, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    bx = x.copy()
    radius = 1e-3
    tstart = time.time()
    iu = np.triu_indices(n, 1)
    iu0, iu1 = iu
    while time.time() - tstart < budget and time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        I = iu0[sel]; J = iu1[sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
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
            best = v; x = y; bx = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            # keep x at improved tangent point? standard: shrink
            radius *= 0.5
            # also track best
            if v < max_cos(bx):
                pass
    # return best found (x tracks last accepted which is best)
    return bx, best

def contact_counts(x, m, tol=1e-9):
    G = x @ x.T
    n = x.shape[0]
    mask = G > m - tol
    np.fill_diagonal(mask, False)
    return np.sum(mask, axis=1), G

def soft_relax(x, steps, T0, T1, lr0, idx=None, rng=None, mu=0.8):
    n, d = x.shape
    x = x.copy()
    vel = np.zeros_like(x)
    if idx is None:
        idx = np.arange(n)
    # precompute mask for subset update
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac if T1 != T0 else T0
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        row = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = E @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gtan = (grad - dot * x) / row
        # clip
        nm = np.linalg.norm(gtan, axis=1, keepdims=True) + 1e-12
        gtan = gtan * np.minimum(1.0, 0.6 / nm)
        vel = mu * vel + gtan
        step = lr * vel
        # only move idx
        x[idx] = x[idx] - step[idx]
        x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
    return x

def best_void_for(others, rng, n_try=4000):
    d = others.shape[1]
    cand = rng.randn(n_try, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    m = np.max(dots, axis=1)
    bi = int(np.argmin(m))
    return cand[bi], float(m[bi])

def solve():
    t0 = time.time()
    deadline = t0 + 174.0
    rng = np.random.RandomState(12345)
    np.random.seed(12345)
    n, d = 180, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (n, d):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                cands.append((max_cos(c), c))
        except Exception:
            pass
    if len(cands) == 0:
        x0 = rng.randn(n, d); x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        b_best = max_cos(x0); x_best = x0
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()
        x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)

    # quick initial squeeze if time allows
    # explore phase until deadline-35
    explore_end = t0 + 132.0
    if explore_end > deadline - 35:
        explore_end = deadline - 35

    it = 0
    sigmas = [0.035, 0.025, 0.015, 0.008, 0.004]
    while time.time() < explore_end and time.time() < deadline - 40:
        it += 1
        # contact info of current best
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        tol = 1e-9
        mask = (x_best @ x_best.T) > m - tol
        np.fill_diagonal(mask, False)
        cnt = np.sum(mask, axis=1)
        order = np.argsort(-cnt)
        # choose centre: mostly overloaded, sometimes random
        r = rng.rand()
        if r < 0.7:
            # pick among top 12
            k = min(12, n)
            ci = int(rng.choice(order[:k]))
        elif r < 0.85:
            ci = int(rng.randint(n))
        else:
            # rattler if exists
            ratt = np.where(cnt == 0)[0]
            if len(ratt) > 0:
                ci = int(ratt[rng.randint(len(ratt))])
            else:
                ci = int(order[0])
        # neighbourhood size
        K = int(rng.choice([8, 12, 16, 22, 30]))
        # nearest to centre by dot
        sims = x_best @ x_best[ci]
        nbr = np.argsort(-sims)[:K]
        # occasionally include rattlers in moved set
        ratt = np.where(cnt == 0)[0]
        move_idx = np.unique(nbr)
        if len(ratt) > 0 and rng.rand() < 0.3:
            move_idx = np.unique(np.concatenate([move_idx, ratt[:2]]))
        sigma = sigmas[rng.randint(len(sigmas))]
        # shake
        x_try = x_best.copy()
        noise = rng.randn(len(move_idx), d) * sigma
        x_try[move_idx] = x_try[move_idx] + noise
        x_try[move_idx] /= np.linalg.norm(x_try[move_idx], axis=1, keepdims=True)
        # void re-seat of rattlers occasionally: move one rattler to void
        if len(ratt) > 0 and rng.rand() < 0.4:
            ri = int(ratt[rng.randint(len(ratt))])
            others = np.delete(x_try, ri, axis=0)
            v, _ = best_void_for(others, rng, n_try=3000)
            # small blend to void to keep diversity
            if rng.rand() < 0.5:
                x_try[ri] = v
            else:
                x_try[ri] = x_try[ri] + 0.5 * (v - x_try[ri])
                x_try[ri] /= np.linalg.norm(x_try[ri])
        # annealed repulsion relax
        T0 = float(rng.choice([0.012, 0.008, 0.005]))
        T1 = T0 * 0.25
        lr0 = float(rng.choice([0.06, 0.04, 0.025]))
        steps = int(rng.choice([250, 350, 450]))
        # update all points but focus step on moved? update all for speed simplicity
        x_rel = soft_relax(x_try, steps=steps, T0=T0, T1=T1, lr0=lr0, idx=None, rng=rng)
        v_rel = max_cos(x_rel)
        # gate LP: only if promising
        if v_rel < b_best + 0.004:
            # short polish 2-3 s
            remaining = min(3.0, deadline - 40 - time.time())
            if remaining < 0.5:
                break
            x_pol, v_pol = lp_polish(x_rel, budget=remaining, deadline=deadline - 35)
            if v_pol < b_best - 1e-12:
                x_best = x_pol; b_best = v_pol
                # tighten explore a bit
                continue
            # also try keeping relaxed if it is better than best (shouldn't happen without polish)
            if v_rel < b_best - 1e-12:
                x_best = x_rel; b_best = v_rel
        # else discard
        # anti-stall: if many iters without improvement, do bigger kick from best
        # (loop naturally does it via sigma choice)

    # final long polish
    trem = deadline - time.time() - 1.0
    if trem > 2:
        x_best, b_best = lp_polish(x_best, budget=trem, deadline=deadline)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
