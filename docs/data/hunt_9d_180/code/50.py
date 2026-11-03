import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

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

def best_void(others, rng, n_try=8000, n_refine=400):
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
        sig = 0.08 * (0.001 / 0.08) ** frac
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

def lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        cfull = Gfull[iu0]
        m = float(np.max(cfull))
        thr = m - max(20.0 * radius, 2e-9)
        sel = cfull > thr
        I = iu0[0][sel]; J = iu0[1][sel]
        csub = cfull[sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            # subsample worst to keep LP fast
            idx = np.argsort(-csub)[:3000]
            I, J, csub = I[idx], J[idx], csub[idx]
            p = len(I)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
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
        try:
            dx = res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = x + dx
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 174.0
    np.random.seed(0)
    rng = np.random.RandomState(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (180, 9):
                c = c / np.linalg.norm(c, axis=1, keepdims=True)
                if np.all(np.isfinite(c)):
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
    explore_end = min(t0 + 100.0, deadline - 60.0)
    it = 0
    noimp = 0
    sigmas = [0.015, 0.035, 0.06]
    while time.time() < explore_end:
        it += 1
        if time.time() > deadline - 60:
            break
        if noimp > 22:
            x_cur = x_best.copy()
            cur_val = b_best
            noimp = 0
            sel = rng.choice(n, size=10, replace=False)
            x_cur[sel] += 0.04 * rng.randn(len(sel), d)
            x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
        if it % 3 != 0:
            G = x_cur @ x_cur.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            per = np.max(G, axis=1)
            cnt = np.sum(G > m - 5e-3, axis=1)
            order = np.argsort(-per)
            nk = 2 if (it % 6 == 1) else 1
            pool = order[:14]
            perm = rng.permutation(len(pool))
            chosen = np.sort(pool[perm[:nk]])
            # also occasionally include a rattler / low-contact point
            if it % 9 == 4:
                cnt_full = np.sum(G > m - 1e-9, axis=1)
                low = np.argsort(cnt_full)[:6]
                # replace one chosen with lowest-contact if distinct
                cand_low = int(low[rng.randint(len(low))])
                if cand_low not in chosen:
                    chosen = np.sort(np.append(chosen[1:] if len(chosen) > 1 else [], cand_low).astype(int))
                    nk = len(chosen)
            mask = np.ones(n, dtype=bool)
            mask[chosen] = False
            others_base = x_cur[mask]
            x_try = x_cur.copy()
            placed = []
            for idx in chosen:
                if time.time() > deadline - 60:
                    break
                base = np.vstack([others_base] + placed) if placed else others_base
                y, _ = best_void(base, rng, n_try=8000, n_refine=350)
                x_try[int(idx)] = y
                placed.append(y.reshape(1, -1))
            nb, vb = smooth_relax(x_try, steps=700, T0=0.014, T1=0.0008, lr0=0.010, mu=0.85)
        else:
            G = x_cur @ x_cur.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            cnt = np.sum(G > m - 5e-3, axis=1)
            w = cnt.astype(float) + 1.0
            w /= w.sum()
            k = int(rng.randint(6, 14))
            sel1 = rng.choice(n, size=k, replace=False, p=w)
            sigma = sigmas[(it // 3) % len(sigmas)]
            x_try = x_cur.copy()
            x_try[sel1] += sigma * rng.randn(k, d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            nb, vb = smooth_relax(x_try, steps=750, T0=0.011, T1=0.0007, lr0=0.012, mu=0.85)
        if vb < b_best - 1e-12:
            x_best = nb.copy(); b_best = vb
            x_cur = nb.copy(); cur_val = vb
            noimp = 0
        elif vb < cur_val:
            x_cur = nb.copy(); cur_val = vb
            noimp += 1
        else:
            temp = 0.0008
            if rng.rand() < np.exp((cur_val - vb) / temp) * 0.2:
                x_cur = nb.copy(); cur_val = vb
            noimp += 1
    # short sharpen then final LP polish
    try:
        rem = deadline - time.time() - 1.0
        if rem > 8:
            nb, vb = smooth_relax(x_best, steps=400, T0=0.004, T1=0.0004, lr0=0.006, mu=0.9)
            if vb < b_best:
                x_best = nb; b_best = vb
        rem = deadline - time.time() - 0.8
        if rem > 5:
            x_best = lp_polish(x_best, seconds=min(rem, 55.0), deadline=deadline - 0.4)
    except Exception:
        pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
