import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 5e-4
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
        # cap constraints to keep HiGHS fast
        if p > 2500:
            idx = np.argpartition(-csub, 2500)[:2500]
            I, J, csub = I[idx], J[idx], csub[idx]
            p = 2500
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

def smooth_relax(x, steps, T0, T1, lr0, mu=0.85):
    n, d = x.shape
    x = x.copy()
    best = x.copy()
    bv = max_cos(x)
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
        if (s & 63) == 63 or s == steps - 1:
            v = max_cos(x)
            if v < bv:
                bv = v
                best = x.copy()
    return best, bv

def best_void(others, rng, n_try=15000, n_refine=500):
    d = others.shape[1]
    cand = rng.randn(n_try, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    m = np.max(dots, axis=1)
    bi = int(np.argmin(m))
    y = cand[bi].copy()
    cur = float(m[bi])
    for k in range(n_refine):
        sig = 0.08 * (0.001 / 0.08) ** (k / max(1, n_refine - 1))
        prop = y + sig * rng.randn(d)
        prop /= np.linalg.norm(prop)
        pm = float(np.max(prop @ others.T))
        if pm < cur:
            y = prop
            cur = pm
    return y, cur

def per_point_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    per = np.max(G, axis=1)
    m = float(np.max(per))
    return G, per, m

def solve():
    t0 = time.time()
    deadline = t0 + 178.0
    rng = np.random.RandomState(7)
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
    explore_end = t0 + 140.0
    it = 0
    noimp = 0
    sig_cycle = [0.012, 0.03, 0.055, 0.02]
    while time.time() < explore_end and time.time() < deadline - 35:
        it += 1
        if noimp > 30:
            x_cur = x_best.copy()
            cur_val = b_best
            noimp = 0
            sel = rng.choice(n, size=14, replace=False)
            x_cur[sel] += 0.05 * rng.randn(len(sel), d)
            x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
            cur_val = max_cos(x_cur)
        mode = it % 4
        if mode == 0 or mode == 1:
            G, per, m = per_point_max(x_cur)
            cnt = np.sum(G > m - 1e-7, axis=1)
            order_worst = np.argsort(-per)
            order_best = np.argsort(cnt)  # low contacts first
            if mode == 0:
                # 1 hub + 2 low-contact (balanced surgery)
                w1 = order_worst[rng.randint(0, 5)]
                # pick 2 low-contact distinct from w1
                chosen = [int(w1)]
                for cand_idx in order_best:
                    cand_idx = int(cand_idx)
                    if cand_idx not in chosen:
                        chosen.append(cand_idx)
                    if len(chosen) >= 3:
                        break
            else:
                # 1-2 worst hubs
                nk = 2 if (it % 8 == 1) else 1
                pool = order_worst[:10]
                perm = rng.permutation(len(pool))
                chosen = [int(pool[perm[k]]) for k in range(nk)]
            mask = np.ones(n, dtype=bool)
            mask[np.array(chosen)] = False
            others_list = [x_cur[mask]]
            x_try = x_cur.copy()
            placed = []
            for idx in chosen:
                base = np.vstack(others_list + placed) if placed else others_list[0]
                y, _ = best_void(base, rng, n_try=12000, n_refine=450)
                x_try[idx] = y
                placed.append(y.reshape(1, -1))
            nb, vb = smooth_relax(x_try, steps=750, T0=0.018, T1=0.0009, lr0=0.011, mu=0.85)
        elif mode == 2:
            G, per, m = per_point_max(x_cur)
            cnt = np.sum(G > m - 1e-7, axis=1)
            w = cnt.astype(float) + 1.0
            w = w / w.sum()
            k = int(rng.randint(8, 20))
            sel1 = rng.choice(n, size=k, replace=False, p=w)
            sigma = sig_cycle[(it // 4) % len(sig_cycle)]
            x_try = x_cur.copy()
            x_try[sel1] += sigma * rng.randn(k, d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            nb, vb = smooth_relax(x_try, steps=850, T0=0.013, T1=0.0007, lr0=0.012, mu=0.86)
        else:
            # mild global jitter to hop nearby basin
            x_try = x_cur.copy()
            x_try += 0.008 * rng.randn(n, d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            nb, vb = smooth_relax(x_try, steps=500, T0=0.008, T1=0.0005, lr0=0.009, mu=0.88)
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
            temp = 0.0006
            if rng.rand() < np.exp((cur_val - vb) / temp) * 0.2:
                x_cur = nb.copy()
                cur_val = vb
            noimp += 1
    rem = deadline - time.time() - 1.0
    if rem > 4:
        try:
            if rem > 18:
                nb, vb = smooth_relax(x_best, steps=600, T0=0.004, T1=0.0004, lr0=0.006, mu=0.9)
                if vb < b_best:
                    x_best = nb
                    b_best = vb
                rem = deadline - time.time() - 1.0
            if rem > 4:
                x_best = lp_polish(x_best, seconds=min(rem, 35.0), deadline=deadline - 0.5)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
