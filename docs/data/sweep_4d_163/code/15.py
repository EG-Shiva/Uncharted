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
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if deadline is not None and time.time() > deadline - 0.5:
            break
        G = x @ x.T
        iu_c = G[iu]
        m = float(iu_c.max())
        sel = iu_c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        # cap huge constraint sets
        if p > 4000:
            # keep most tight
            order = np.argsort(-iu_c[sel])[:4000]
            I, J = I[order], J[order]
            sel_idx = np.where(sel)[0][order]
            c_sel = iu_c[sel_idx]
            p = 4000
        else:
            c_sel = iu_c[sel]
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
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

def quench(x, steps, lr, temp, margin, momentum, rng, noise=0.0):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        # mask to near-max for speed/sparsity, margin adaptive
        M = G > (cmax - margin)
        # softmax weights
        # exp in float64, clip
        W = np.exp((G - cmax) / temp) * M
        # zero diagonal (already -2 gives tiny weight, force 0)
        np.fill_diagonal(W, 0.0)
        ssum = W.sum() * 0.5 + 1e-12
        G2 = W * G
        g = (W @ x - (G2.sum(axis=1, keepdims=True) * x)) / ssum
        # momentum update (heavy ball), gradient points to increase -> subtract
        v = momentum * v + g
        cur_lr = lr * (1.0 - 0.5 * s / max(steps, 1))
        x = x - cur_lr * v
        if noise > 0:
            x = x + noise * np.random.standard_normal(size=x.shape) * 0.0  # placeholder
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

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    DEADLINE = t0 + 172.0
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal(size=(163, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (163, 4):
        tmp = rng.standard_normal(size=(163, 4))
        if xb.ndim == 2:
            mm = min(xb.shape[0], 163)
            dd = min(xb.shape[1], 4)
            tmp[:mm, :dd] = xb[:mm, :dd]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = max_cos(xb)
    x_best = xb.copy()
    x_cur = xb.copy()
    cur = best
    n, d = 163, 4
    it = 0
    while time.time() < DEADLINE - 19.0:
        it += 1
        if rng.random() < 0.72 or it < 3:
            base = x_best
            base_val = best
        else:
            base = x_cur
            base_val = cur
        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        thr = base_val - 0.012
        deg = (G0 > thr).sum(axis=1)
        r = rng.random()
        x_try = base.copy()
        if r < 0.24:
            # global gaussian shake, log-uniform sigma
            sigma = 10.0 ** rng.uniform(-3.4, -1.25)
            x_try = base + sigma * rng.standard_normal(size=(n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.44:
            # patch shake around jammed / random center
            if rng.random() < 0.65:
                center = int(np.argmax(deg + rng.random(n) * 0.6))
            else:
                center = int(rng.integers(n))
            k = int(rng.integers(6, 24))
            dots = base @ base[center]
            idx = np.argsort(-dots)[:k]
            sigma = float(rng.uniform(0.015, 0.085))
            x_try[idx] = base[idx] + sigma * rng.standard_normal(size=(k, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.60:
            # rigid patch rotation + jitter (novel coherent move)
            if rng.random() < 0.7:
                center = int(np.argmax(deg + rng.random(n) * 0.6))
            else:
                center = int(rng.integers(n))
            k = int(rng.integers(8, 30))
            dots = base @ base[center]
            idx = np.argsort(-dots)[:k]
            A = rng.standard_normal(size=(4, 4))
            A = A - A.T
            an = np.linalg.norm(A) + 1e-12
            A = A / an
            eps = float(rng.uniform(0.02, 0.11))
            rot_disp = (base[idx] @ A.T) * eps
            jit = 0.012 * rng.standard_normal(size=(k, d))
            x_try[idx] = base[idx] + rot_disp + jit
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.74:
            # contact-pair splitting: push top pairs apart
            iu0 = np.triu_indices(n, 1)
            c0 = G0[iu0]
            top = np.argsort(-c0)[:24]
            # pick 2-4 pairs
            npairs = int(rng.integers(2, 5))
            chosen = rng.choice(top, size=npairs, replace=False)
            for cc in chosen:
                a = int(iu0[0][cc]); b = int(iu0[1][cc])
                # direction away: tangent component
                da = base[a] - (float(base[a] @ base[b])) * base[b]
                db = base[b] - (float(base[b] @ base[a])) * base[a]
                na = np.linalg.norm(da) + 1e-12; nb = np.linalg.norm(db) + 1e-12
                da /= na; db /= nb
                s = float(rng.uniform(0.02, 0.07))
                x_try[a] = base[a] + s * da + 0.008 * rng.standard_normal(4)
                x_try[b] = base[b] + s * db + 0.008 * rng.standard_normal(4)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            # rattler / low-degree relocation to voids
            m = int(rng.integers(2, 5))
            # lowest degree (with small random tie-break) = rattlers
            reloc = np.argsort(deg + rng.random(n) * 0.4)[:m]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[reloc] = False
            if time.time() > DEADLINE - 30:
                S = 8000
            else:
                S = 15000
            voids = largest_voids(base[keep_mask], m, rng, S=S)
            voids = voids + 0.004 * rng.standard_normal(size=voids.shape)
            voids /= np.linalg.norm(voids, axis=1, keepdims=True)
            x_try[reloc] = voids
            if rng.random() < 0.5:
                sigma = 0.008
                x_try[keep_mask] = base[keep_mask] + sigma * rng.standard_normal(size=(n - m, d))
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        if time.time() > DEADLINE - 21.5:
            break
        steps = int(rng.integers(350, 1300))
        # shorten if low time
        remain = DEADLINE - 19.0 - time.time()
        if remain < 30:
            steps = min(steps, 500)
        lr = float(rng.uniform(0.05, 0.16))
        temp = float(10.0 ** rng.uniform(-2.7, -1.9))
        margin = float(rng.uniform(0.02, 0.07))
        momentum = float(rng.uniform(0.0, 0.88))
        x_try = quench(x_try, steps, lr, temp, margin, momentum, rng)
        if time.time() > DEADLINE - 20.0:
            v = max_cos(x_try)
        else:
            sec = float(rng.uniform(0.6, 1.5))
            sec = min(sec, max(0.5, DEADLINE - 19.0 - time.time()))
            x_try = lp_polish(x_try, sec, deadline=DEADLINE - 18.0)
            v = max_cos(x_try)
        if v < best - 1e-12:
            x_best = x_try.copy()
            best = v
            x_cur = x_try.copy()
            cur = v
        elif v < cur + 0.0012 and rng.random() < 0.5:
            x_cur = x_try.copy()
            cur = v
        if it % 14 == 0 and cur > best + 0.002:
            x_cur = x_best.copy()
            cur = best
    remain = DEADLINE - time.time() - 0.8
    if remain > 2.0:
        x_best = lp_polish(x_best, min(remain, 17.0), deadline=DEADLINE)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
