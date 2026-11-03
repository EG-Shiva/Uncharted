import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def fast_relax(x, steps, T0, T1, lr, rng=None):
    n, d = x.shape
    x = x.copy()
    # track best
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(np.max(G0))
    best = x.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac if T1 > 0 else T0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(np.max(G))
        # weights
        # avoid overflow: (G-gmax)<=0
        E = np.exp((G - gmax) / T)
        np.fill_diagonal(E, 0.0)
        row = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = E @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gtan = grad - dot * x
        step = gtan / row
        # clip
        nm = np.sqrt(np.sum(step * step, axis=1, keepdims=True)) + 1e-12
        step = step * np.minimum(1.0, 0.7 / nm)
        x = x - lr * step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if (s & 31) == 31 or s == steps - 1:
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            v = float(np.max(G2))
            if v < bv:
                bv = v
                best = x.copy()
    return best, bv

def void_replace(x, idxs, rng, n_try=3000):
    n, d = x.shape
    x = x.copy()
    for idx in idxs:
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        others = x[mask]
        cand = rng.randn(n_try, d)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ others.T
        m = np.max(dots, axis=1)
        bi = int(np.argmin(m))
        y = cand[bi]
        # quick local refine 200 steps
        cur = float(m[bi])
        sig = 0.05
        for k in range(200):
            sigk = sig * (0.002 / sig) ** (k / 199)
            prop = y + sigk * rng.randn(d)
            prop /= np.linalg.norm(prop)
            pm = float(np.max(prop @ others.T))
            if pm < cur:
                y = prop
                cur = pm
        x[idx] = y
    return x

def lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
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
        if p > 3000:
            # keep tightest 3000 to bound LP size
            order = np.argsort(-csub)
            keep = order[:3000]
            I, J, csub = I[keep], J[keep], csub[keep]
            p = len(I)
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
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 8.0})
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
    deadline = t0 + 178.0
    rng = np.random.RandomState(1)
    # load incumbent
    try:
        xb = np.asarray(np.load("best.npy"), dtype=float)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        bb = max_cos(xb)
    except Exception:
        xb = None; bb = 1e9
    try:
        xr = np.asarray(np.load("record.npy"), dtype=float)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        br = max_cos(xr)
    except Exception:
        xr = None; br = 1e9
    if xb is not None and bb <= br:
        x_best = xb.copy(); b_best = bb
    elif xr is not None:
        x_best = xr.copy(); b_best = br
    else:
        x_best = rng.randn(180, 9)
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        b_best = max_cos(x_best)
    x_cur = x_best.copy(); cur_val = b_best
    n, d = 180, 9
    explore_end = t0 + 128.0
    it = 0
    noimp = 0
    while time.time() < explore_end and time.time() < deadline - 45:
        it += 1
        time_left = explore_end - time.time()
        if time_left < 1:
            break
        # periodic fresh random restart basin (different topology)
        if it % 17 == 0:
            x_try = rng.randn(n, d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            nb, vb = fast_relax(x_try, steps=350, T0=0.02, T1=0.001, lr=0.03)
            # accept to cur to diversify
            if vb < b_best - 1e-12:
                x_best = nb.copy(); b_best = vb
                x_cur = nb.copy(); cur_val = vb; noimp = 0
            elif vb < cur_val or rng.rand() < 0.15:
                x_cur = nb.copy(); cur_val = vb
            noimp += 1
            continue
        if noimp > 30:
            x_cur = x_best.copy(); cur_val = b_best; noimp = 0
        mode = it % 3
        if mode == 0:
            # large subset shake (20-40 points) to leave basin
            G = x_cur @ x_cur.T
            np.fill_diagonal(G, -2.0)
            per = np.max(G, axis=1)
            order = np.argsort(-per)
            k = int(rng.randint(18, 38))
            # half from worst 40, half uniform
            pool_worst = order[:40]
            sel_a = pool_worst[rng.choice(len(pool_worst), size=k // 2, replace=False)]
            sel_b = rng.choice(n, size=k - k // 2, replace=False)
            sel = np.unique(np.concatenate([sel_a, sel_b]))
            sigma = [0.25, 0.4, 0.15][(it // 3) % 3]
            x_try = x_cur.copy()
            x_try[sel] += sigma * rng.randn(len(sel), d)
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
            nb, vb = fast_relax(x_try, steps=260, T0=0.012, T1=0.0008, lr=0.02)
        elif mode == 1:
            # multi void reinsertion of worst points
            G = x_cur @ x_cur.T
            np.fill_diagonal(G, -2.0)
            per = np.max(G, axis=1)
            order = np.argsort(-per)
            nk = int(rng.randint(3, 7))
            pool = order[:15]
            perm = rng.permutation(len(pool))
            chosen = np.sort(pool[perm[:nk]])
            x_try = void_replace(x_cur, chosen, rng, n_try=3000)
            nb, vb = fast_relax(x_try, steps=260, T0=0.012, T1=0.0008, lr=0.02)
        else:
            # random great-circle rotation of a cap (coherent cluster move)
            u = rng.randn(d); u /= np.linalg.norm(u)
            proj = x_cur @ u
            k = int(rng.randint(15, 45))
            # pick side
            if rng.rand() < 0.5:
                sel = np.argsort(-proj)[:k]
            else:
                sel = np.argsort(proj)[:k]
            # random orthogonal plane
            v = rng.randn(d); v = v - (v @ u) * u; v /= np.linalg.norm(v) + 1e-12
            ang = rng.uniform(0.1, 0.5)
            x_try = x_cur.copy()
            # rotate sel points in (u,v) plane by ang + jitter
            for idx in sel:
                a = x_try[idx]
                au = float(a @ u); av = float(a @ v)
                # small rotation
                cu, sv = np.cos(ang), np.sin(ang)
                # rotate (au,av) components
                nau = cu * au - sv * av
                nav = sv * au + cu * av
                a2 = a + (nau - au) * u + (nav - av) * v
                a2 += 0.02 * rng.randn(d)
                a2 /= np.linalg.norm(a2)
                x_try[idx] = a2
            nb, vb = fast_relax(x_try, steps=240, T0=0.010, T1=0.0007, lr=0.02)
        if vb < b_best - 1e-12:
            x_best = nb.copy(); b_best = vb
            x_cur = nb.copy(); cur_val = vb; noimp = 0
        elif vb < cur_val - 1e-12:
            x_cur = nb.copy(); cur_val = vb; noimp += 1
        else:
            # occasional sideways
            if rng.rand() < 0.08:
                x_cur = nb.copy(); cur_val = vb
            noimp += 1
    # final polish
    rem = deadline - time.time() - 1.0
    if rem > 5:
        try:
            # short sharpen with low T
            if rem > 25:
                nb, vb = fast_relax(x_best, steps=250, T0=0.004, T1=0.0004, lr=0.008)
                if vb < b_best:
                    x_best = nb; b_best = vb
                rem = deadline - time.time() - 1.0
            x_best = lp_polish(x_best, seconds=min(rem, 38.0), deadline=deadline - 0.5)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
