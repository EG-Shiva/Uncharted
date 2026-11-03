import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def smooth_relax(x, steps, T0, T1, lr0, rng):
    n, d = x.shape
    best = x.copy()
    bv = max_cos(x)
    vel = np.zeros_like(x)
    mu = 0.85
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        row = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = E @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gtan = grad - dot * x
        step = gtan / row
        nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = step * np.minimum(1.0, 0.6 / nm)
        vel = mu * vel + step
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if s == steps - 1 or (s & 63) == 63:
            v = max_cos(x)
            if v < bv:
                bv = v
                best = x.copy()
    return best, bv

def lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        cfull = Gfull[iu]
        m = float(np.max(cfull))
        sel = cfull > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        csub = cfull[sel]
        # cap to tightest 3000 to stay fast
        if len(I) > 3000:
            idx = np.argpartition(-csub, 3000)[:3000]
            I, J, csub = I[idx], J[idx], csub[idx]
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
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 178.0
    rng = np.random.RandomState(1)
    try:
        xb = np.asarray(np.load("best.npy"), dtype=float)
    except Exception:
        xb = None
    try:
        xr = np.asarray(np.load("record.npy"), dtype=float)
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None and c.shape == (180, 9):
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            cands.append((max_cos(c), c))
    if not cands:
        x0 = rng.randn(180, 9)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        x_best = x0
        b_best = max_cos(x0)
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 180, 9
    x_cur = x_best.copy()
    cur_val = b_best
    # exploration budget: leave 28s for final LP
    explore_end = deadline - 28.0
    it = 0
    noimp = 0
    while time.time() < explore_end:
        it += 1
        if time.time() > explore_end - 2:
            break
        if noimp > 20:
            x_cur = x_best.copy()
            cur_val = b_best
            noimp = 0
        # analyse current
        G = x_cur @ x_cur.T
        np.fill_diagonal(G, -2.0)
        per_max = np.max(G, axis=1)
        order_worst = np.argsort(-per_max)
        # contacts with loose threshold for guidance
        thr = cur_val - 0.008
        contacts = np.sum(G > thr, axis=1)
        order_low = np.argsort(contacts)
        # choose k
        k = int(rng.choice([3, 4, 4, 5, 6]))
        victims = set()
        victims.add(int(order_worst[0]))
        if rng.rand() < 0.8:
            victims.add(int(order_worst[int(rng.randint(1, 6))]))
        # low contact / rattler
        if rng.rand() < 0.8:
            victims.add(int(order_low[int(rng.randint(0, 8))]))
        while len(victims) < k:
            victims.add(int(rng.randint(0, n)))
        victims = list(victims)
        # void re-insert each victim greedily
        # precompute others quickly per victim with sampling
        for v in victims:
            mask = np.ones(n, dtype=bool); mask[v] = False
            others = x_cur[mask]
            # sample candidates
            n_try = 3500
            cand = rng.randn(n_try, d)
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ others.T
            m = np.max(dots, axis=1)
            bi = int(np.argmin(m))
            y = cand[bi].copy()
            cur = float(m[bi])
            # quick hill refine
            for rr in range(120):
                sig = 0.06 * (0.002 / 0.06) ** (rr / 119.0)
                prop = y + sig * rng.randn(d)
                prop /= np.linalg.norm(prop)
                pm = float(np.max(prop @ others.T))
                if pm < cur:
                    y = prop; cur = pm
            # occasional shake instead of void to diversify
            if rng.rand() < 0.15:
                y = x_cur[v] + 0.25 * rng.randn(d)
                y /= np.linalg.norm(y)
            x_cur[v] = y
        # small extra shake on neighbours of overloaded points
        if rng.rand() < 0.4:
            ov = int(order_worst[0])
            # shake 2 random points slightly
            for _ in range(2):
                j = int(rng.randint(0, n))
                x_cur[j] = x_cur[j] + 0.06 * rng.randn(d)
                x_cur[j] /= np.linalg.norm(x_cur[j])
        # relax: short anneal, parameters vary per iteration
        T0 = float(rng.choice([0.008, 0.012, 0.02]))
        T1 = 0.0012
        lr0 = float(rng.choice([0.25, 0.35, 0.45]))
        steps = int(rng.choice([220, 300, 380]))
        # work on copy
        x_tmp = x_cur.copy()
        x_tmp, v_tmp = smooth_relax(x_tmp, steps, T0, T1, lr0, rng)
        if v_tmp < b_best - 1e-12:
            b_best = v_tmp
            x_best = x_tmp.copy()
            x_cur = x_tmp.copy()
            cur_val = v_tmp
            noimp = 0
        else:
            noimp += 1
            if v_tmp < cur_val:
                x_cur = x_tmp.copy()
                cur_val = v_tmp
            elif v_tmp < b_best + 0.008 and rng.rand() < 0.25:
                x_cur = x_tmp.copy()
                cur_val = v_tmp
            else:
                # revert wanderer to best with small kick to try new region
                if rng.rand() < 0.3:
                    x_cur = x_best.copy()
                    cur_val = b_best
                    # kick few points
                    for _ in range(3):
                        j = int(rng.randint(0, n))
                        x_cur[j] = x_cur[j] + 0.12 * rng.randn(d)
                        x_cur[j] /= np.linalg.norm(x_cur[j])
        # early exit if clearly improved a lot (give LP more time)
        if b_best < 0.4615:
            break
    # final polish of best
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    remain = deadline - time.time() - 1.0
    if remain > 5:
        x_best = lp_polish(x_best, min(remain, 26.0), deadline - 0.5)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
