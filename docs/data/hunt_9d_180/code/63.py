import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def smooth_relax(x, steps, T, lr, mu=0.85):
    n, d = x.shape
    x = x.copy()
    vel = np.zeros_like(x)
    for s in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        # soft-max weights, focus on near-max pairs
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        row = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = E @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gtan = grad - dot * x
        step = gtan / row
        nm = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        # clip large steps
        scl = np.minimum(1.0, 0.6 / nm)
        step = step * scl
        vel = mu * vel + step
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def best_void(others, rng, n_try=4000, n_refine=300):
    d = others.shape[1]
    cand = rng.randn(n_try, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    m = np.max(dots, axis=1)
    bi = int(np.argmin(m))
    y = cand[bi].copy()
    cur = float(m[bi])
    for k in range(n_refine):
        frac = k / max(1, n_refine - 1)
        sig = 0.06 * (0.0005 / 0.06) ** frac
        prop = y + sig * rng.randn(d)
        nrm = np.linalg.norm(prop)
        if nrm < 1e-12:
            continue
        prop /= nrm
        pm = float(np.max(prop @ others.T))
        if pm < cur:
            y = prop
            cur = pm
    return y, cur

def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    xb = x.copy()
    radius = 1e-3
    start = time.time()
    if deadline is None:
        deadline = start + seconds
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        cfull = Gfull[iu]
        m = float(np.max(cfull))
        sel = cfull > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        csub = cfull[sel]
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
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
            # keep global best of polish
            best = v
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            # also allow moving to same level to slide along valley
            if v <= max_cos(x) + 1e-12:
                x = y
            radius *= 0.5
    # return best found
    # re-evaluate xb in case x drifted
    if max_cos(x) < best:
        xb = x.copy()
        best = max_cos(x)
    return xb

def solve():
    t0 = time.time()
    deadline = t0 + 178.0
    rng = np.random.RandomState(1)
    # load starts
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.load(fn)
            c = np.asarray(c, dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                cands.append((max_cos(c), c))
        except Exception:
            pass
    if len(cands) == 0:
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

    explore_end = t0 + 115.0
    # if little time, skip explore
    it = 0
    # temperature for annealing acceptance
    T_accept0 = 2e-4
    T_accept1 = 2e-6
    max_iter = 100000
    while time.time() < explore_end and time.time() < deadline - 55 and it < max_iter:
        it += 1
        frac = (time.time() - t0) / max(1e-9, (explore_end - t0))
        frac = min(max(frac, 0.0), 1.0)
        Tacc = T_accept0 * (T_accept1 / T_accept0) ** frac
        # compute per-point worst
        G = x_cur @ x_cur.T
        np.fill_diagonal(G, -2.0)
        pw = np.max(G, axis=1)
        order_desc = np.argsort(-pw)
        order_asc = np.argsort(pw)
        r = rng.rand()
        x_trial = x_cur.copy()
        if r < 0.45:
            # hub void re-insertion: 1-3 hubs + maybe 1 cold
            nh = int(rng.randint(1, 4))
            hubs = order_desc[:8]
            # random pick among top 8
            perm = rng.permutation(8)[:nh]
            idxs = hubs[perm].tolist()
            # occasionally also move coldest
            if rng.rand() < 0.4:
                idxs.append(int(order_asc[int(rng.randint(0, 6))]))
            for idx in idxs:
                others = np.delete(x_trial, idx, axis=0)
                y, _ = best_void(others, rng, n_try=3500, n_refine=250)
                x_trial[idx] = y
            T = float(np.exp(rng.uniform(np.log(0.0012), np.log(0.012))))
            lr = float(rng.uniform(0.12, 0.35))
            steps = int(rng.randint(250, 550))
            x_trial = smooth_relax(x_trial, steps, T, lr, mu=0.82)
        elif r < 0.85:
            # subset Gaussian shake + relax
            sz = int(rng.randint(8, 32))
            # bias to bad points
            bad_pool = order_desc[:40].tolist()
            rand_pool = rng.permutation(n)[:sz].tolist()
            # mix
            subset = list(set(bad_pool[:sz // 2] + rand_pool))[:sz]
            sigma = float(rng.uniform(0.015, 0.07))
            noise = np.zeros_like(x_trial)
            noise[subset] = rng.randn(len(subset), d) * sigma
            x_trial = x_trial + noise
            x_trial /= np.linalg.norm(x_trial, axis=1, keepdims=True)
            T = float(np.exp(rng.uniform(np.log(0.001), np.log(0.008))))
            lr = float(rng.uniform(0.10, 0.30))
            steps = int(rng.randint(250, 500))
            x_trial = smooth_relax(x_trial, steps, T, lr, mu=0.85)
        else:
            # cold/rattler re-insertion only
            nc = int(rng.randint(1, 3))
            idxs = [int(order_asc[int(k)]) for k in rng.permutation(n)[:nc]]
            # actually prefer coldest among random? just use coldest few
            # replace with explicit coldest + random cold
            idxs = order_asc[:nc].tolist()
            # add small random choice
            for idx in idxs:
                others = np.delete(x_trial, idx, axis=0)
                y, _ = best_void(others, rng, n_try=4000, n_refine=300)
                x_trial[idx] = y
            T = float(np.exp(rng.uniform(np.log(0.0015), np.log(0.010))))
            lr = float(rng.uniform(0.12, 0.32))
            steps = int(rng.randint(300, 550))
            x_trial = smooth_relax(x_trial, steps, T, lr, mu=0.82)
        v = max_cos(x_trial)
        if v < b_best - 1e-12:
            b_best = v
            x_best = x_trial.copy()
            x_cur = x_trial.copy()
            cur_val = v
        elif v < cur_val or rng.rand() < np.exp(-(v - cur_val) / max(Tacc, 1e-12)):
            # annealing move of wanderer (allow slight worsening)
            # but cap worsening to avoid destroying: only if not too bad
            if v < cur_val + 0.02:
                x_cur = x_trial.copy()
                cur_val = v
        # restart wanderer if stuck far above best for long
        if it % 30 == 0 and cur_val > b_best + 0.005:
            # restart from best with small shake
            x_cur = x_best.copy()
            sig = 0.01
            x_cur = x_cur + sig * rng.randn(n, d) * 0.3
            x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
            cur_val = max_cos(x_cur)
            if cur_val < b_best:
                b_best = cur_val
                x_best = x_cur.copy()

    # final polish of best
    remain = deadline - time.time() - 1.0
    if remain > 5:
        # use most of remaining for LP, reserve tiny for return
        try:
            x_best = lp_polish(x_best, seconds=remain - 0.5, deadline=deadline - 0.5)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
