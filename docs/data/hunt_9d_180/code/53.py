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
    radius = 1e-3
    if deadline is None:
        deadline = time.time() + seconds
    start = time.time()
    iu = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14 and it < 60:
        it += 1
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

def anneal(x, steps, T0, T1, lr0, mu, rng, deadline=None):
    n, d = x.shape
    x = x.copy()
    best = x.copy()
    bv = max_cos(x)
    vel = np.zeros_like(x)
    for s in range(steps):
        if deadline is not None and (s & 63) == 0 and time.time() > deadline:
            break
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

def best_void(others, rng, n_try=3000, n_refine=150):
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

def solve():
    t0 = time.time()
    deadline = t0 + 175.0
    rng = np.random.RandomState(7)
    n, d = 180, 9
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
        try:
            if c is not None:
                c = np.asarray(c, dtype=float)
                if c.shape == (n, d):
                    c = c / np.linalg.norm(c, axis=1, keepdims=True)
                    cands.append((max_cos(c), c))
        except Exception:
            pass
    if len(cands) == 0:
        x_best = rng.randn(n, d)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        b_best = max_cos(x_best)
    else:
        cands.sort(key=lambda t: t[0])
        b_best, x_best = cands[0][0], cands[0][1].copy()
    x_cur = x_best.copy()
    cur_val = b_best
    explore_end = min(deadline - 42.0, t0 + 128.0)
    it = 0
    noimp = 0
    while time.time() < explore_end and time.time() < deadline - 40:
        it += 1
        if time.time() > deadline - 42:
            break
        # restart from best if stuck
        if noimp > 25:
            x_cur = x_best.copy()
            cur_val = b_best
            noimp = 0
            # large global jitter to new basin
            x_cur = x_cur + 0.08 * rng.randn(n, d)
            x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
            nb, vb = anneal(x_cur, steps=350, T0=0.02, T1=0.0008, lr0=0.012, mu=0.85, rng=rng, deadline=deadline-40)
            if vb < b_best - 1e-12:
                x_best = nb.copy(); b_best = vb
            x_cur = nb.copy(); cur_val = vb
            continue
        mode = it % 6
        try:
            G = x_cur @ x_cur.T
            np.fill_diagonal(G, -2.0)
            per = np.max(G, axis=1)
            order = np.argsort(-per)
            mcur = float(np.max(G))
            cnt = np.sum(G > mcur - 1e-9, axis=1)
        except Exception:
            order = rng.permutation(n)
            per = np.zeros(n)
            cnt = np.ones(n)
            mcur = cur_val
        if mode == 0:
            # fresh uniform restart: broad repulsion for isotropy
            x_try = rng.randn(n, d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            nb, vb = anneal(x_try, steps=500, T0=0.06, T1=0.0007, lr0=0.014, mu=0.86, rng=rng, deadline=deadline-40)
        elif mode == 1:
            # heavy void reinsertion of overloaded + rattlers
            low = np.argsort(per)  # rattlers first
            hi = order[:12]
            k_hi = int(rng.randint(4, 8))
            k_lo = int(rng.randint(2, 5))
            chosen = list(hi[rng.permutation(len(hi))[:k_hi]]) + list(low[:k_lo])
            # dedup
            chosen = sorted(set(int(c) for c in chosen))
            mask = np.ones(n, dtype=bool); mask[np.array(chosen)] = False
            others = x_cur[mask]
            x_try = x_cur.copy()
            placed = []
            for idx in chosen:
                base = np.vstack([others] + placed) if placed else others
                y, _ = best_void(base, rng, n_try=2500, n_refine=120)
                x_try[idx] = y
                placed.append(y.reshape(1, -1))
            # small global jitter for balance
            x_try = x_try + 0.008 * rng.randn(n, d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            nb, vb = anneal(x_try, steps=450, T0=0.016, T1=0.0006, lr0=0.011, mu=0.87, rng=rng, deadline=deadline-40)
        elif mode == 2:
            # coherent cluster kick
            istar = int(order[rng.randint(0, 3)])
            sims = (x_cur @ x_cur[istar])
            nbr = np.argsort(-sims)[:14]
            x_try = x_cur.copy()
            v = rng.randn(d); v /= np.linalg.norm(v) + 1e-12
            mag = 0.07 + 0.05 * rng.rand()
            for j in nbr:
                xj = x_try[j]
                vt = v - float(v @ xj) * xj
                nm = np.linalg.norm(vt) + 1e-12
                vt = vt / nm
                x_try[j] = x_try[j] + mag * vt + 0.015 * rng.randn(d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            nb, vb = anneal(x_try, steps=400, T0=0.014, T1=0.0006, lr0=0.012, mu=0.85, rng=rng, deadline=deadline-40)
        elif mode == 3:
            # degree-weighted subset perturbation
            w = (cnt.astype(float) + 1.0)
            w = w / w.sum()
            k = int(rng.randint(8, 18))
            sel1 = rng.choice(n, size=k, replace=False, p=w)
            x_try = x_cur.copy()
            sig = 0.03 + 0.03 * rng.rand()
            x_try[sel1] += sig * rng.randn(k, d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            nb, vb = anneal(x_try, steps=380, T0=0.012, T1=0.0006, lr0=0.012, mu=0.85, rng=rng, deadline=deadline-40)
        elif mode == 4:
            # moderate multi-void + jitter
            k = int(rng.randint(2, 5))
            pool = order[:15]
            chosen = sorted(rng.choice(pool, size=k, replace=False))
            mask = np.ones(n, dtype=bool); mask[np.array(chosen)] = False
            others = x_cur[mask]
            x_try = x_cur.copy()
            placed = []
            for idx in chosen:
                base = np.vstack([others] + placed) if placed else others
                y, _ = best_void(base, rng, n_try=2500, n_refine=120)
                x_try[idx] = y
                placed.append(y.reshape(1, -1))
            extra = rng.choice(n, size=8, replace=False)
            x_try[extra] += 0.02 * rng.randn(len(extra), d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            nb, vb = anneal(x_try, steps=400, T0=0.013, T1=0.0006, lr0=0.011, mu=0.86, rng=rng, deadline=deadline-40)
        else:
            # pure jitter wander
            x_try = x_cur.copy()
            k = int(rng.randint(10, 25))
            sel1 = rng.choice(n, size=k, replace=False)
            x_try[sel1] += 0.04 * rng.randn(k, d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            nb, vb = anneal(x_try, steps=350, T0=0.011, T1=0.0007, lr0=0.012, mu=0.85, rng=rng, deadline=deadline-40)
        # promising? short LP jam to test basin depth
        if vb < b_best + 0.002 and time.time() < deadline - 55:
            try:
                rem = 4.0
                if vb < b_best - 1e-9:
                    rem = 6.0
                cand = lp_polish(nb, seconds=rem, deadline=deadline - 38)
                vc = max_cos(cand)
                if vc < vb:
                    nb, vb = cand, vc
            except Exception:
                pass
        if vb < b_best - 1e-12:
            x_best = nb.copy(); b_best = vb
            x_cur = nb.copy(); cur_val = vb
            noimp = 0
        elif vb < cur_val:
            x_cur = nb.copy(); cur_val = vb
            noimp += 1
        else:
            temp = 0.0009
            if rng.rand() < np.exp((cur_val - vb) / temp) * 0.25:
                x_cur = nb.copy(); cur_val = vb
            noimp += 1
    # final exploitation of best
    try:
        rem = deadline - time.time() - 1.0
        if rem > 5:
            if rem > 20:
                nb, vb = anneal(x_best, steps=500, T0=0.005, T1=0.0004, lr0=0.006, mu=0.9, rng=rng, deadline=deadline-30)
                if vb < b_best:
                    x_best = nb; b_best = vb
                rem = deadline - time.time() - 1.0
            if rem > 4:
                x_best = lp_polish(x_best, seconds=min(rem, 45.0), deadline=deadline - 0.5)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
