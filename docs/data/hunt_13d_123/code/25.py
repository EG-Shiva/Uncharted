import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def smooth_opt(x, iters=250, beta0=80.0, beta1=350.0, lr0=0.03, lr1=0.002):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    for k in range(iters):
        frac = k / (iters - 1) if iters > 1 else 1.0
        beta = beta0 + (beta1 - beta0) * frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        tang = grad - dot * x
        tn = np.linalg.norm(tang, axis=1, keepdims=True)
        med = float(np.median(tn)) + 1e-12
        step = lr * tang / (tn + 0.5 * med)
        x = x - step
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
        m = float(np.max(c))
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def best_hole(Y, rng, samples=4000):
    m, d = Y.shape
    C = rng.standard_normal((samples, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = Y @ C.T
    worst = dots.max(axis=0)
    return C[int(np.argmin(worst))]


def solve():
    t0 = time.time()
    cand_list = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (123, 13):
                cand_list.append(a.astype(float))
        except Exception:
            pass
    if len(cand_list) == 0:
        rng0 = np.random.default_rng(1)
        X0 = rng0.standard_normal((123, 13))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
    else:
        vals = []
        for a in cand_list:
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            vals.append(max_cos(a))
        X0 = cand_list[int(np.argmin(vals))]
        X0 = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    best = X0.copy()
    best_val = max_cos(best)
    cur = best.copy()
    cur_val = best_val
    n, d = best.shape
    rng = np.random.default_rng(987654321)

    def elapsed():
        return time.time() - t0

    it = 0
    no_improve = 0
    # exploration without heavy LP
    while elapsed() < 135 and it < 5000:
        it += 1
        frac_t = elapsed() / 135.0
        Temp = 0.0012 * (1.0 - frac_t) + 0.00015
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        per_worst = G.max(axis=1)
        order = np.argsort(per_worst)
        # contact count for hub detection
        thr = float(G.max()) - 1e-4
        contacts = (G >= thr).sum(axis=1)
        hub = int(np.argmax(contacts))
        mode = int(rng.integers(0, 7))
        trial = cur.copy()
        if mode == 0:
            sigma = 0.03 + 0.03 * float(rng.random())
            trial = trial + sigma * rng.standard_normal(trial.shape)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode == 1:
            sigma = 0.10 + 0.05 * float(rng.random())
            # shake from global best occasionally to diversify
            base = best if (it % 3 == 0) else cur
            trial = base + sigma * rng.standard_normal(base.shape)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode == 2:
            sigma = 0.25 + 0.15 * float(rng.random())
            base = best if (it % 2 == 0) else cur
            trial = base + sigma * rng.standard_normal(base.shape)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode == 3:
            # delete k worst and greedily refill holes
            k = 2 + int(rng.integers(0, 3))  # 2-4
            idx_rm = order[-k:]
            mask = np.ones(n, dtype=bool)
            mask[idx_rm] = False
            Y = (best if (it % 2 == 0) else cur)[mask]
            for _ in range(k):
                v = best_hole(Y, rng, samples=4000)
                Y = np.vstack([Y, v[None, :]])
            trial = Y + 0.012 * rng.standard_normal(Y.shape)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode == 4:
            # cluster surgery around overloaded hub
            p = 6 + int(rng.integers(0, 7))  # 6-12
            Gw = (best @ best.T) if (it % 2 == 0) else (cur @ cur.T)
            np.fill_diagonal(Gw, -2.0)
            hub2 = int(np.argmax((Gw >= float(Gw.max()) - 1e-4).sum(axis=1)))
            nn = np.argsort(-Gw[hub2])[:p]
            mask = np.ones(n, dtype=bool)
            mask[nn] = False
            Y = (best if (it % 2 == 0) else cur)[mask]
            for _ in range(p):
                v = best_hole(Y, rng, samples=4000)
                Y = np.vstack([Y, v[None, :]])
            trial = Y + 0.02 * rng.standard_normal(Y.shape)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        elif mode == 5:
            # rattler / low-contact reinsertion
            k = 2 + int(rng.integers(0, 2))
            idx_low = order[:k]
            mask = np.ones(n, dtype=bool)
            mask[idx_low] = False
            Y = cur[mask]
            for _ in range(k):
                v = best_hole(Y, rng, samples=4000)
                Y = np.vstack([Y, v[None, :]])
            # light shake of rest
            Y = Y + 0.015 * rng.standard_normal(Y.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            trial = Y
        else:
            # randomize a few points completely
            k = 3 + int(rng.integers(0, 4))  # 3-6
            idx = rng.choice(n, size=k, replace=False)
            base = cur.copy()
            rp = rng.standard_normal((k, d))
            rp /= np.linalg.norm(rp, axis=1, keepdims=True)
            base[idx] = rp
            base = base + 0.02 * rng.standard_normal(base.shape)
            base /= np.linalg.norm(base, axis=1, keepdims=True)
            trial = base

        # cheap smoothing, randomize annealing slightly
        b1 = 320.0 + 80.0 * float(rng.random())
        trial_s = smooth_opt(trial, iters=250, beta0=80.0, beta1=b1, lr0=0.03, lr1=0.002)
        v = max_cos(trial_s)
        # promising -> extend a bit
        if v < best_val + 0.0006 and v >= best_val - 1e-12:
            # only extend if very close, cheap extra 200 iters
            if v < best_val + 0.00025:
                trial_s = smooth_opt(trial_s, iters=200, beta0=300.0, beta1=500.0, lr0=0.008, lr1=0.001)
                v = max_cos(trial_s)
        if v < best_val - 1e-12:
            best = trial_s
            best_val = v
            cur = trial_s.copy()
            cur_val = v
            no_improve = 0
            # squeeze new record lightly, limited frequency
            if elapsed() < 125:
                # only spend LP if meaningful gain
                lx = lp_polish(trial_s, 0.8)
                vlp = max_cos(lx)
                if vlp < best_val - 1e-12:
                    best = lx
                    best_val = vlp
                    cur = lx.copy()
                    cur_val = vlp
            continue
        # walk acceptance
        if v < cur_val or rng.random() < np.exp((cur_val - v) / Temp):
            cur = trial_s
            cur_val = v
            if v < cur_val + 0.002:
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        if no_improve > 70:
            # restart walk from best with noise
            cur = best + 0.05 * rng.standard_normal(best.shape)
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_val = max_cos(cur)
            no_improve = 0

    # longer smoothing on best before final LP
    if elapsed() < 170:
        best = smooth_opt(best, iters=500, beta0=120.0, beta1=500.0, lr0=0.015, lr1=0.001)
        best_val = max_cos(best)
    remain = 178 - elapsed()
    if remain > 1.0:
        remain = min(remain, 28.0)
        best = lp_polish(best, remain)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
