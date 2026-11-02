import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _worst_per_point(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def _contact_counts(x, m, tol=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return np.sum(g > m - tol, axis=1)


def _lp_polish(x, deadline, init_radius=5e-3, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        remain = deadline - time.time()
        if remain < 0.35:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -cap)[-cap:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = cap
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cm = cols.reshape(p, 2 * d + 1)
        Vm = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cm[:, :d] = I[:, None] * d + ar
        Cm[:, d:2 * d] = J[:, None] * d + ar
        Cm[:, 2 * d] = n * d
        Vm[:, :d] = x[J]
        Vm[:, d:2 * d] = x[I]
        Vm[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        tl = max(0.2, remain - 0.15)
        try:
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs", options={"time_limit": tl})
            except TypeError:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
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
            x = y
            best = v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _tangent_relax(x, steps, eta0, margin, deadline, temp=0.006):
    n, d = x.shape
    eta = float(eta0)
    for s in range(int(steps)):
        if time.time() >= deadline:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        thresh = m - float(margin)
        iuI, iuJ = np.triu_indices(n, 1)
        c = G[iuI, iuJ]
        sel = c > thresh
        if int(sel.sum()) == 0:
            break
        I = iuI[sel]
        J = iuJ[sel]
        cs = c[sel]
        if cs.size > 3000:
            k = 3000
            pi = np.argpartition(cs, -k)[-k:]
            I = I[pi]
            J = J[pi]
            cs = cs[pi]
        w = np.exp((cs - m) / temp) + 0.15 * (cs - thresh) / max(margin, 1e-12)
        w = np.maximum(w, 1e-3)
        Fi = w[:, None] * (x[J] - cs[:, None] * x[I])
        Fj = w[:, None] * (x[I] - cs[:, None] * x[J])
        F = np.zeros_like(x)
        np.add.at(F, I, Fi)
        np.add.at(F, J, Fj)
        deg = np.bincount(np.concatenate([I, J]), minlength=n).astype(np.float64) + 3.0
        F /= deg[:, None]
        fn = np.linalg.norm(F, axis=1, keepdims=True) + 1e-12
        F *= np.minimum(1.0 / (fn + 1e-12), 3.0)
        # per-point clip already tangent; step
        x = x + eta * F
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        eta *= 0.985
        if eta < 0.004:
            eta = 0.004
    return x


def _greedy_reinsert(y, to_replace, rng, n_cand=4000, deadline=None):
    n, d = y.shape
    fixed_mask = np.ones(n, dtype=bool)
    fixed_mask[np.array(to_replace, dtype=int)] = False
    X_fixed = y[fixed_mask]
    new_pts = []
    for k in to_replace:
        if deadline is not None and time.time() >= deadline:
            break
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        if new_pts:
            Xa = np.vstack([X_fixed] + new_pts)
        else:
            Xa = X_fixed
        dots = cand @ Xa.T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        bp = cand[bi]
        new_pts.append(bp[None, :])
        y[k] = bp
    return y


def solve():
    t0 = time.time()
    HARD = t0 + 165.0
    rng = np.random.default_rng(12345)
    x0 = None
    b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < b0:
                        b0 = v
                        x0 = a.copy()
        except Exception:
            pass
    if x0 is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x0 = a
        b0 = _max_cos(a)
    x_best = x0.copy()
    best = float(b0)
    n, d = x_best.shape

    it = 0
    no_improve = 0
    while True:
        now = time.time()
        if now > HARD - 4.0:
            break
        x = x_best.copy()
        s = it % 6
        if s == 0:
            sig = 0.015 * (1.0 + 0.08 * min(no_improve, 12))
            x = x + rng.standard_normal(size=x.shape) * sig
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _tangent_relax(x, steps=60, eta0=0.03, margin=0.025, deadline=HARD - 3.0)
            budget = 1.8
            init_r = 0.006
        elif s == 1:
            w = _worst_per_point(x)
            k = 35
            idx = np.argpartition(w, -k)[-k:]
            x[idx] += rng.standard_normal(size=(k, d)) * 0.06
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            x = _tangent_relax(x, steps=80, eta0=0.035, margin=0.03, deadline=HARD - 3.0)
            budget = 2.0
            init_r = 0.007
        elif s == 2:
            m = best
            cc = _contact_counts(x, m, tol=1e-9)
            ratt = np.where(cc == 0)[0].tolist()
            order = np.argsort(-cc)
            torep = [int(r) for r in ratt[:3]]
            for o in order:
                if len(torep) >= 5:
                    break
                if int(o) not in torep:
                    torep.append(int(o))
            torep = torep[:5]
            x = _greedy_reinsert(x, torep, rng, n_cand=4000, deadline=HARD - 3.0)
            x += rng.standard_normal(size=x.shape) * 0.004
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _tangent_relax(x, steps=40, eta0=0.025, margin=0.02, deadline=HARD - 3.0)
            budget = 2.2
            init_r = 0.007
        elif s == 3:
            m = best
            cc = _contact_counts(x, m, tol=1e-9)
            k = 12
            idx = np.argpartition(-cc, k)[:k] if False else np.argsort(-cc)[:k]
            for kk in idx:
                kk = int(kk)
                t = rng.standard_normal(size=(d,))
                t -= (t @ x[kk]) * x[kk]
                tn = np.linalg.norm(t)
                if tn > 1e-12:
                    t /= tn
                    x[kk] = x[kk] + t * (0.05 + 0.03 * rng.random())
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _tangent_relax(x, steps=80, eta0=0.03, margin=0.03, deadline=HARD - 3.0)
            budget = 2.0
            init_r = 0.007
        elif s == 4:
            k = 25
            idx = rng.choice(n, size=k, replace=False)
            x[idx] += rng.standard_normal(size=(k, d)) * 0.05
            x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
            x = _tangent_relax(x, steps=60, eta0=0.03, margin=0.025, deadline=HARD - 3.0)
            budget = 1.8
            init_r = 0.006
        else:
            w = _worst_per_point(x)
            k = 12
            idx = np.argpartition(w, -k)[-k:].tolist()
            x = _greedy_reinsert(x, [int(v) for v in idx], rng, n_cand=3000, deadline=HARD - 3.0)
            x += rng.standard_normal(size=x.shape) * 0.003
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = _tangent_relax(x, steps=60, eta0=0.03, margin=0.025, deadline=HARD - 3.0)
            budget = 2.0
            init_r = 0.007
        remain = HARD - 3.0 - time.time()
        if remain < 0.6:
            break
        budget = min(budget, remain)
        lp_dead = time.time() + budget
        if lp_dead > HARD - 2.5:
            lp_dead = HARD - 2.5
        try:
            y = _lp_polish(x, deadline=lp_dead, init_radius=init_r)
        except Exception:
            it += 1
            no_improve += 1
            continue
        try:
            v = _max_cos(y)
        except Exception:
            it += 1
            no_improve += 1
            continue
        if v < best - 1e-12 and np.all(np.isfinite(y)):
            best = v
            x_best = y
            no_improve = 0
        else:
            no_improve += 1
        it += 1
        if time.time() > HARD - 3.0:
            break

    remain = HARD - time.time() - 0.5
    if remain > 1.0:
        try:
            x_best = _lp_polish(x_best, deadline=time.time() + min(remain, 10.0), init_radius=2e-3, cap=2500)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety: replace any bad row
    nr = np.linalg.norm(x_best, axis=1)
    bad = ~(np.isfinite(nr)) | (nr < 1e-12)
    if np.any(bad):
        for i in np.where(bad)[0]:
            v = rng.standard_normal(size=(d,))
            v /= np.linalg.norm(v)
            x_best[i] = v
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
