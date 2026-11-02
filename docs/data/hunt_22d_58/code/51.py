import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _quench(X, steps, b0, b1, lr0, lr1, mu=0.85, mask=None, rng=None, noise=0.0):
    n, d = X.shape
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    use_mask = mask is not None
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        if use_mask:
            # zero grad for frozen points
            Gt = Gt * mask[:, None]
            V = V * mask[:, None]
        V = mu * V + Gt
        step = lr * V
        # keep tangent
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        # clip
        sn = np.sqrt(np.sum(step * step, axis=1, keepdims=True)) + 1e-12
        over = sn > 0.05
        if np.any(over):
            step = np.where(over, step * (0.05 / sn), step)
        if use_mask:
            step = step * mask[:, None]
        X = X - step
        if noise > 0 and rng is not None:
            X = X + (noise * (1.0 - f)) * (rng.standard_normal(X.shape) * (mask[:, None] if use_mask else 1.0))
        nr = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _hole_for(X, victim, rng, n_cand=4000):
    n, d = X.shape
    mask = np.ones(n, dtype=bool)
    mask[victim] = False
    Y = X[mask]
    C = rng.standard_normal((n_cand, d))
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    dots = C @ Y.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    X2 = X.copy()
    X2[victim] = C[bi]
    return X2

def _lp_polish(X, seconds, deadline):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n, d = X.shape
    best = _max_cos(X)
    radius = 7e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    itlp = 0
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        itlp += 1
        if itlp > 30:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 550:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 550:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 1.5})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y
            best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    TIME_LIMIT = 165.0
    deadline_final = t0 + TIME_LIMIT
    rng = np.random.default_rng(7)
    n, d = 58, 22
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        best = z
        best_val = _max_cos(best)
    # quick initial squeeze
    try:
        b2 = _lp_polish(best, 2.5, deadline_final - 6.0)
        v2 = _max_cos(b2)
        if v2 < best_val:
            best = b2
            best_val = v2
    except Exception:
        pass
    cur = best.copy()
    cur_val = best_val
    T0, T1 = 0.0025, 0.00015
    hop_end = t0 + 132.0
    if hop_end > t0 + TIME_LIMIT - 28:
        hop_end = t0 + TIME_LIMIT - 28
    it = 0
    # estimate total iterations to schedule temperature
    max_it = 70
    while time.time() < hop_end and it < max_it:
        if time.time() > t0 + TIME_LIMIT - 28:
            break
        it += 1
        frac = it / max_it
        Temp = T0 * (T1 / T0) ** frac
        mode = it % 5
        y = cur.copy()
        # degrees of cur for hub targeting
        try:
            Gc = cur @ cur.T
            np.fill_diagonal(Gc, -2.0)
            mc = float(Gc.max())
            deg = (Gc > mc - 0.004).sum(axis=1)
        except Exception:
            deg = np.zeros(n)
            Gc = None
            mc = cur_val
        victims = np.zeros(n, dtype=bool)
        if mode == 0:
            sig = 0.06
            y = y + sig * rng.standard_normal(y.shape)
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
        elif mode == 1:
            # hub hole reinsertion: top-2 hubs
            order = np.argsort(-deg)
            cand = order[:2]
            for vv in cand:
                y = _hole_for(y, int(vv), rng, n_cand=4000)
            # small noise to their top neighbours
            for vv in cand:
                row = (y @ y[int(vv)])
                row[int(vv)] = -2
                top = np.argsort(-row)[:2]
                y[top] += 0.02 * rng.standard_normal((len(top), d))
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
            victims[cand] = True
            # include neighbours in subset mask
            victims[top] = True
        elif mode == 2:
            # worst pair + random -> holes
            if Gc is not None:
                k = int(np.argmax(Gc))
                i0 = k // n
                j0 = k % n
            else:
                i0, j0 = int(rng.integers(0, n)), int(rng.integers(0, n))
            r1 = int(rng.integers(0, n))
            for vv in [int(i0), int(j0), r1][:2]:
                y = _hole_for(y, vv, rng, n_cand=3500)
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
            victims[int(i0)] = True
            victims[int(j0)] = True
            victims[r1] = True
        elif mode == 3:
            k = 6
            idx = rng.choice(n, size=k, replace=False)
            y[idx] += 0.10 * rng.standard_normal((k, d))
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
            victims[idx] = True
        else:
            k = 3
            idx = rng.choice(n, size=k, replace=False)
            for vv in idx:
                y = _hole_for(y, int(vv), rng, n_cand=3000)
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
            victims[idx] = True
        # quench: subset then global
        try:
            if np.any(victims):
                mfloat = victims.astype(float)
                # dilate mask slightly: include nothing else
                y = _quench(y, 320, 25, 140, 0.006, 0.0012, mu=0.80, mask=mfloat)
                y = _quench(y, 320, 90, 550, 0.004, 0.0007, mu=0.85, mask=None)
            else:
                y = _quench(y, 350, 30, 160, 0.005, 0.001, mu=0.80, mask=None)
                y = _quench(y, 300, 150, 700, 0.003, 0.0006, mu=0.85, mask=None)
        except Exception:
            continue
        v = _max_cos(y)
        if v < best_val - 1e-12:
            # squeeze close winners immediately
            try:
                y2 = _lp_polish(y, 1.2, deadline_final - 4.0)
                v2 = _max_cos(y2)
                if v2 < v:
                    y, v = y2, v2
            except Exception:
                pass
            best_val = v
            best = y.copy()
            cur = y.copy()
            cur_val = v
            continue
        # also squeeze near-misses occasionally (different basin may polish below best)
        if v < best_val + 0.0012 and (it % 4 == 0 or v < cur_val):
            try:
                y2 = _lp_polish(y, 0.9, deadline_final - 4.0)
                v2 = _max_cos(y2)
                if v2 < best_val - 1e-12:
                    best_val = v2
                    best = y2.copy()
                    cur = y2.copy()
                    cur_val = v2
                    continue
                # use polished for acceptance
                y, v = y2, v2
            except Exception:
                pass
        # annealing acceptance for cur
        if v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(Temp, 1e-9)):
            cur = y.copy()
            cur_val = v
        # occasional restart from best to avoid drift
        if it % 18 == 0 and cur_val > best_val + 0.002:
            cur = best.copy()
            cur_val = best_val
    remain = deadline_final - time.time() - 1.0
    if remain > 4:
        lp_sec = min(remain - 1.0, 20.0)
        try:
            polished = _lp_polish(best, lp_sec, deadline_final - 1.0)
            vp = _max_cos(polished)
            if vp < best_val:
                best = polished
                best_val = vp
        except Exception:
            pass
    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-12)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best
