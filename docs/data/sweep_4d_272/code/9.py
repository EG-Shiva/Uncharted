import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    # fix non-finite
    if not np.all(np.isfinite(x)):
        x = np.where(np.isfinite(x), x, 0.0)
        n = np.linalg.norm(x, axis=1, keepdims=True)
        bad = (n[:, 0] < 1e-12)
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _personal_worst(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _find_void(others, rng, n_cand=8000):
    cand = rng.standard_normal((n_cand, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ others.T
    scores = s.max(axis=1)
    bi = int(np.argmin(scores))
    b = cand[bi].copy()
    bs = float(scores[bi])
    step = 0.06
    # fast hill climb
    for _ in range(250):
        prop = b + step * rng.standard_normal((32, 4))
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        sc = (prop @ others.T).max(axis=1)
        j = int(np.argmin(sc))
        if sc[j] < bs:
            b = prop[j].copy()
            bs = float(sc[j])
            step *= 1.06
        else:
            step *= 0.96
        if step < 1e-5:
            break
    return b

def _anneal(x, steps, beta0, beta1, lr0, lr1, rng=None):
    n, d = x.shape
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    # preallocate Gram
    for k in range(steps):
        frac = k / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** frac if beta0 > 0 else beta1
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        # W = exp(beta*(G-gm))
        np.subtract(G, gm, out=G)
        np.multiply(G, beta, out=G)
        # clip to avoid overflow (exp(>50) huge but we normalize)
        # shift already by max so <=0, fine
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(G.sum())
        if not np.finite(S) or S == 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m1 *= b1
        m1 += (1 - b1) * g
        v1 *= b2
        v1 += (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1)
        bc2 = 1.0 - b2 ** (k + 1)
        mh = m1 / bc1
        vh = v1 / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _lp_polish(x, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = c.max()
        sel = c > mcur - max(20 * radius, 1e-9)
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
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(12345)
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = np.load("record.npy")
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (272, 4):
        flat = x0.ravel()
        need = 272 * 4
        if flat.size < need:
            extra = rng.standard_normal((need - flat.size,))
            flat = np.concatenate([flat, extra])
        x0 = flat[:need].reshape(272, 4)
    x0 = _norm(x0)
    best = x0.copy()
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val

    # exploration budget: leave ~38s for final polish
    explore_end = t0 + 132.0
    if explore_end > deadline - 40:
        explore_end = deadline - 40

    trial = 0
    # schedules
    while time.time() < explore_end:
        trial += 1
        # choose mode
        mode = trial % 5
        y = cur.copy()
        if mode in (0, 1, 3):
            # void reinsertion of worst points (topology change)
            if mode == 0:
                nkill = 2 + (trial % 3)  # 2-4
            elif mode == 1:
                nkill = 1 + (trial % 2)  # 1-2
            else:
                nkill = 3 + (trial % 4)  # 3-6
            # small jitter to diversify void landscape
            y = y + 0.002 * rng.standard_normal(y.shape)
            y = _norm(y)
            pw = _personal_worst(y)
            order = np.argsort(-pw, kind="stable")
            # pick victims: mix of worst + random among top-15
            top = order[:15]
            # shuffle top slightly
            perm = rng.permutation(len(top))
            # always include the absolute worst with high prob
            victims = []
            victims.append(int(order[0]))
            for k in range(1, nkill):
                victims.append(int(top[perm[k % len(top)]]))
            # deduplicate
            victims = list(dict.fromkeys(victims))
            for wi in victims:
                others = np.delete(y, wi, axis=0)
                b = _find_void(others, rng, n_cand=7000)
                y[wi] = b
            # anneal sharp
            beta0, beta1 = (45.0, 280.0) if mode != 3 else (30.0, 180.0)
            y = _anneal(y, steps=420, beta0=beta0, beta1=beta1, lr0=0.012, lr1=0.0012)
        elif mode == 2:
            # Gaussian kick + anneal
            sigma = [0.015, 0.04, 0.008, 0.07][(trial // 5) % 4]
            y = y + sigma * rng.standard_normal(y.shape)
            y = _norm(y)
            y = _anneal(y, steps=380, beta0=50.0, beta1=300.0, lr0=0.010, lr1=0.001)
        else:  # mode 4: rattler-targeted + global shake of overloaded cluster
            pw = _personal_worst(y)
            order = np.argsort(-pw, kind="stable")
            # shake top-30 heavily, rest lightly
            noise = 0.004 * rng.standard_normal(y.shape)
            noise[order[:30]] += 0.03 * rng.standard_normal((30, 4))
            y = _norm(y + noise)
            y = _anneal(y, steps=500, beta0=60.0, beta1=350.0, lr0=0.008, lr1=0.0008)

        val = _max_cos(y)
        # acceptance: always keep global best; SA for cur to diversify
        if val < best_val:
            best, best_val = y.copy(), val
            cur, cur_val = y.copy(), val
        else:
            # temperature decaying
            elapsed = time.time() - t0
            temp = 0.0015 * max(0.0, 1.0 - elapsed / 140.0)
            if val < cur_val + temp or (rng.random() < 0.02 and val < cur_val + 0.01):
                cur, cur_val = y.copy(), val
            # occasionally restart cur from best
            if trial % 25 == 0:
                cur, cur_val = best.copy(), best_val

        # early exit if time low
        if time.time() > explore_end:
            break

    # final tightening of the best basin
    remain = deadline - time.time()
    # give most time to best; if second candidate close, ignore
    if remain > 5:
        best = _lp_polish(best, seconds=remain - 1.0, deadline=deadline - 0.5)
    best = _norm(best)
    return best
