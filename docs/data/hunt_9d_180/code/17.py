import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def smooth_anneal(x, steps, T0, T1, lr0, mu, rng):
    n, d = x.shape
    x = x.copy()
    best = x.copy()
    # initial value
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(np.max(G0))
    vel = np.zeros_like(x)
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
        step = step * np.minimum(1.0, 0.8 / nm)
        vel = mu * vel + step
        x = x - lr * vel
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if gmax < bv:
            bv = gmax
            best = x.copy()
    return best, bv

def find_void(others, rng, n_try=8000, n_iter=60, batch=16):
    d = others.shape[1]
    cand = rng.randn(n_try, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    m = np.max(dots, axis=1)
    bi = int(np.argmin(m))
    y = cand[bi].copy()
    cur = float(m[bi])
    # batched hill climb with decaying sigma
    for k in range(n_iter):
        frac = k / max(1, n_iter - 1)
        sig = 0.08 * (0.001 / 0.08) ** frac
        prop = y[None, :] + sig * rng.randn(batch, d)
        prop /= np.linalg.norm(prop, axis=1, keepdims=True)
        pm = np.max(prop @ others.T, axis=1)
        j = int(np.argmin(pm))
        if float(pm[j]) < cur:
            y = prop[j].copy()
            cur = float(pm[j])
    return y, cur

def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    if deadline is None:
        deadline = time.time() + seconds
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        iu_c = Gfull[iu]
        m = float(np.max(iu_c))
        sel = iu_c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        csub = iu_c[sel]
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 176.0
    rng = np.random.RandomState(12345)
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for c in (xb, xr):
        if c is not None:
            try:
                c = np.asarray(c, dtype=float)
                if c.shape == (180, 9):
                    c = c / np.linalg.norm(c, axis=1, keepdims=True)
                    cands.append((max_cos(c), c))
            except Exception:
                pass
    if len(cands) == 0:
        x_best = rng.randn(180, 9)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        b_best = max_cos(x_best)
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 180, 9
    x_cur = x_best.copy()
    cur_val = b_best
    it = 0
    noimp = 0
    explore_end = t0 + 112.0
    if explore_end > deadline - 55:
        explore_end = deadline - 55
    sig_cycle = [0.015, 0.035, 0.06]
    while time.time() < explore_end:
        it += 1
        if noimp > 30:
            x_cur = x_best.copy()
            cur_val = b_best
            noimp = 0
            sel = rng.choice(n, size=14, replace=False)
            x_cur[sel] += 0.05 * rng.randn(len(sel), d)
            x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
        op = it % 5
        G = x_cur @ x_cur.T
        np.fill_diagonal(G, -2.0)
        per = np.max(G, axis=1)
        mcur = float(np.max(G))
        cnt = np.sum(G > mcur - 1e-9, axis=1)
        if op == 0 or op == 1:
            nk = 2 if op == 0 else 1
            order = np.argsort(-per)
            pool = order[:12]
            perm = rng.permutation(len(pool))
            chosen = pool[perm[:nk]]
            mask = np.ones(n, dtype=bool)
            mask[chosen] = False
            others_base = x_cur[mask]
            x_try = x_cur.copy()
            placed = []
            for idx in chosen:
                if placed:
                    base = np.vstack([others_base] + placed)
                else:
                    base = others_base
                y, _ = find_void(base, rng, n_try=8000, n_iter=50, batch=16)
                x_try[idx] = y
                placed.append(y.reshape(1, -1))
            nb, vb = smooth_anneal(x_try, steps=650, T0=0.014, T1=0.0007, lr0=0.011, mu=0.85, rng=rng)
        elif op == 2:
            hub = int(np.argmax(cnt))
            k = 8
            neigh = np.argsort(-G[hub])[:k]
            # include hub
            sel = np.unique(np.concatenate([[hub], neigh]))
            x_try = x_cur.copy()
            sigma = 0.05 if (it % 2 == 0) else 0.025
            x_try[sel] += sigma * rng.randn(len(sel), d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            nb, vb = smooth_anneal(x_try, steps=650, T0=0.012, T1=0.0006, lr0=0.012, mu=0.87, rng=rng)
        elif op == 3:
            # rattler / low-contact reinsertion
            order_low = np.argsort(cnt)
            # pick 1 rattler + maybe worst
            cand_low = order_low[0]
            # void for it
            mask = np.ones(n, dtype=bool)
            mask[cand_low] = False
            others = x_cur[mask]
            x_try = x_cur.copy()
            y, _ = find_void(others, rng, n_try=8000, n_iter=50, batch=16)
            x_try[cand_low] = y
            # small shake of 4 random others to accommodate
            sel2 = rng.choice(n, size=4, replace=False)
            x_try[sel2] += 0.02 * rng.randn(4, d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            nb, vb = smooth_anneal(x_try, steps=600, T0=0.010, T1=0.0006, lr0=0.010, mu=0.85, rng=rng)
        else:
            w = cnt.astype(float) + 1.0
            w = w / w.sum()
            k = int(rng.randint(6, 13))
            sel1 = rng.choice(n, size=k, replace=False, p=w)
            sigma = sig_cycle[(it // 5) % len(sig_cycle)]
            x_try = x_cur.copy()
            x_try[sel1] += sigma * rng.randn(k, d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            nb, vb = smooth_anneal(x_try, steps=700, T0=0.013, T1=0.0007, lr0=0.012, mu=0.86, rng=rng)
        if vb < b_best - 1e-12:
            x_best = nb.copy()
            b_best = vb
            x_cur = nb.copy()
            cur_val = vb
            noimp = 0
        elif vb < cur_val:
            x_cur = nb.copy()
            cur_val = vb
            noimp += 1
        else:
            temp = 0.0008
            if rng.rand() < np.exp((cur_val - vb) / temp) * 0.2:
                x_cur = nb.copy()
                cur_val = vb
            noimp += 1
        if time.time() > explore_end - 0.5:
            break
    rem = deadline - time.time() - 1.0
    if rem > 4:
        try:
            if rem > 22:
                nb, vb = smooth_anneal(x_best, steps=500, T0=0.004, T1=0.0004, lr0=0.006, mu=0.9, rng=rng)
                if vb < b_best:
                    x_best = nb
                    b_best = vb
                rem = deadline - time.time() - 1.0
            x_best = lp_polish(x_best, seconds=min(rem, 55.0), deadline=deadline - 0.5)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
