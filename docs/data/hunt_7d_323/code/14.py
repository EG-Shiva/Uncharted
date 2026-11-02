import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, m, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.sum(g > m - tol, axis=1)

def _lp_polish(x, seconds, init_radius=4e-3, max_iters=60):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-14 and it < max_iters:
        it += 1
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 2500:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -2500)[-2500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 2500
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2*d] = J[:, None] * d + ar
        Cb[:, 2*d] = n * d
        Vb[:, :d] = x[J]
        Vb[:, d:2*d] = x[I]
        Vb[:, 2*d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def _gd_relax(x, steps, tau0, tau1, lr0, lr1, rng):
    n, d = x.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        tau = tau0 * (tau1 / tau0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # scaled exp for stability
        E = np.exp((G - m) / tau)
        np.fill_diagonal(E, 0.0)
        Gr = E @ x
        # tangential projection
        proj = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - proj * x
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt = Gt / nr
        # scale by row weight to focus jammed points, but bounded
        w = np.sum(E, axis=1, keepdims=True)
        w = np.clip(w / (np.mean(w) + 1e-12), 0.3, 3.0)
        x = x - lr * w * Gt
        # tiny noise to break symmetry early
        if s < steps // 3:
            x = x + (1e-4 * (1 - f)) * rng.standard_normal(size=x.shape)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _hole_reinsert(y, victims, rng, n_cand=6000):
    n, d = y.shape
    vset = set(victims.tolist()) if not isinstance(victims, set) else victims
    # fixed set excludes victims
    mask = np.ones(n, dtype=bool)
    mask[np.array(list(vset), dtype=int)] = False
    Xf = y[mask]
    for k in list(vset):
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ Xf.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        y[k] = cand[bi]
        # extend fixed set with new point for next victim (greedy)
        Xf = np.vstack([Xf, y[k][None, :]])
    return y

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(12345)
    # load start
    best_x = None
    best_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.any(~np.isfinite(a)):
                    continue
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        best_x = a
        best_v = _max_cos(a)
    x = best_x.copy()
    # short initial snap
    remain = BUDGET - (time.time() - t0)
    if remain > 20:
        x = _lp_polish(x, min(6.0, remain - 12.0))
        v = _max_cos(x)
        if v < best_v:
            best_v = v; best_x = x.copy()
        else:
            x = best_x.copy()
    # basin hopping cycles, fixed schedule
    n_cycles = 10
    for cyc in range(n_cycles):
        if time.time() - t0 > BUDGET - 12.0:
            break
        m = _max_cos(x)
        cc = _contacts(x, m, tol=1e-9)
        ratt = np.where(cc == 0)[0]
        order = np.argsort(-cc)
        n_hub = 5
        hubs = order[:n_hub]
        # victims: rattlers + hubs + 2 random
        victims = list(dict.fromkeys(list(ratt.tolist()) + list(hubs.tolist())))
        # add randoms
        perm = rng.permutation(323)
        for p in perm:
            if len(victims) >= 9:
                break
            if p not in victims:
                victims.append(int(p))
        victims = np.array(victims[:9], dtype=int)
        y = x.copy()
        # shake strength schedule
        sig = [0.06, 0.10, 0.04, 0.12, 0.05, 0.08, 0.15, 0.03, 0.07, 0.11][cyc % 10]
        # hole reinsert half of victims (rattlers+top hubs), jitter rest
        n_hole = min(len(victims), 4 + (cyc % 3))
        hole_set = victims[:n_hole]
        jit_set = victims[n_hole:]
        y = _hole_reinsert(y, hole_set, rng, n_cand=6000)
        # jitter
        for k in jit_set:
            noise = rng.standard_normal(size=(7,)) * sig
            # tangential
            noise = noise - np.dot(noise, y[k]) * y[k]
            y[k] = y[k] + noise
            y[k] /= np.linalg.norm(y[k])
        # relax with annealed repulsion
        tau0 = 0.02 if cyc % 2 == 0 else 0.03
        y = _gd_relax(y, steps=400, tau0=tau0, tau1=0.004, lr0=0.02, lr1=0.002, rng=rng)
        # quick polish
        y = _lp_polish(y, 3.0, init_radius=3e-3, max_iters=25)
        v = _max_cos(y)
        if v < best_v - 1e-12:
            best_v = v; best_x = y.copy(); x = y.copy()
        elif v < _max_cos(x) + 0.002 and rng.random() < 0.25:
            # occasional sideways move to new basin
            x = y.copy()
        else:
            # small perturbation of current base for diversity
            x = best_x.copy()
            pj = rng.permutation(323)[:3]
            for k in pj:
                nz = rng.standard_normal(size=(7,)) * 0.02
                nz = nz - np.dot(nz, x[k]) * x[k]
                x[k] = x[k] + nz
                x[k] /= np.linalg.norm(x[k])
        # opportunistic extra GD on best
        if time.time() - t0 > BUDGET - 12.0:
            break
    # final polish of best
    remain = BUDGET - (time.time() - t0)
    if remain > 4:
        best_x = _lp_polish(best_x, min(remain - 1.0, 10.0), init_radius=3e-3, max_iters=60)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
