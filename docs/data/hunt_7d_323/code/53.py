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


def _hinge_anneal(x, T, steps, lr0, decay=0.992):
    n, d = x.shape
    lr = float(lr0)
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        np.subtract(G, T, out=G)
        np.maximum(G, 0.0, out=G)
        np.fill_diagonal(G, 0.0)
        # gradient of sum excess^2 /2 : Gr = W @ x
        Gr = G @ x
        dot = np.einsum('ij,ij->i', Gr, x)[:, None]
        Gr -= dot * x
        # clip huge steps
        mx = float(np.linalg.norm(Gr, axis=1).max()) if n else 0.0
        if mx > 0.5 and mx > 0:
            Gr *= (0.5 / mx)
        x = x - lr * Gr
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        lr *= decay
    return x


def _greedy_reinsert(y, to_replace, rng, n_cand=4000):
    n, d = y.shape
    for k in to_replace:
        Xf = np.delete(y, k, axis=0)
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        worst = (cand @ Xf.T).max(axis=1)
        bi = int(np.argmin(worst))
        y[k] = cand[bi]
    return y


def _lp_polish(x, deadline, init_radius=5e-3, max_pairs=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    while True:
        if time.time() >= deadline - 0.3:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > max_pairs:
            idx = np.where(sel)[0]
            cv = c[idx]
            part = np.argpartition(cv, -max_pairs)[-max_pairs:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = max_pairs
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        cr = cols.reshape(p, 2 * d + 1)
        vr = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        cr[:, :d] = I[:, None] * d + ar[None, :]
        cr[:, d:2 * d] = J[:, None] * d + ar[None, :]
        cr[:, 2 * d] = n * d
        vr[:, :d] = x[J]
        vr[:, d:2 * d] = x[I]
        vr[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        tl = deadline - time.time() - 0.08
        if tl < 0.15:
            break
        tl = min(tl, 2.0)
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
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
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x, best


def solve():
    t0 = time.time()
    HOP_END = t0 + 122.0
    FINAL_END = t0 + 174.0
    rng = np.random.default_rng(0)
    x0 = None
    b0 = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
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
        if now >= HOP_END - 0.6:
            break
        if FINAL_END - now < 5.0:
            break
        boost = 1.0 + 0.04 * min(no_improve, 15)
        s = it % 6
        x = x_best.copy()
        try:
            if s == 0:
                sig = 0.024 * boost
                x = x + rng.standard_normal(size=x.shape) * sig
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                T = best - 0.004
                x = _hinge_anneal(x, T, steps=110, lr0=0.8)
            elif s == 1:
                w = _worst_per_point(x)
                k = 20
                idx = np.argpartition(w, -k)[-k:]
                x[idx] += rng.standard_normal(size=(k, d)) * (0.085 * boost)
                x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
                T = best - 0.005
                x = _hinge_anneal(x, T, steps=130, lr0=0.9)
            elif s == 2:
                # coordinated cap shift
                u = rng.standard_normal(size=(d,))
                u /= np.linalg.norm(u)
                dots = x @ u
                k = int(rng.integers(12, 28))
                cap = np.argpartition(dots, -k)[-k:]
                common = rng.standard_normal(size=(d,)) * (0.05 * boost)
                # make common roughly tangential to cap mean
                mvec = x[cap].mean(axis=0)
                mvec /= (np.linalg.norm(mvec) + 1e-12)
                common = common - float(common @ mvec) * mvec
                x[cap] += common[None, :]
                x[cap] += rng.standard_normal(size=(k, d)) * (0.022 * boost)
                x[cap] /= np.linalg.norm(x[cap], axis=1, keepdims=True)
                T = best - 0.004
                x = _hinge_anneal(x, T, steps=130, lr0=0.85)
            elif s == 3:
                # contact-hub neighbourhood kick
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                C = G > (m - 0.01)
                np.fill_diagonal(C, False)
                deg = C.sum(axis=1)
                hub = int(np.argmax(deg))
                nbrs = np.where(C[hub])[0]
                if len(nbrs) > 18:
                    order = np.argsort(-G[hub, nbrs])
                    nbrs = nbrs[order[:18]]
                cl = [hub] + nbrs.tolist()
                # add a few second-hop
                if len(cl) < 22:
                    extra = []
                    for a in nbrs[:6]:
                        nn = np.where(C[a])[0]
                        for b in nn:
                            if b not in cl and b not in extra:
                                extra.append(int(b))
                            if len(cl) + len(extra) >= 24:
                                break
                    cl = cl + extra[:max(0, 24 - len(cl))]
                cl = np.array(cl, dtype=int)
                k2 = len(cl)
                common2 = rng.standard_normal(size=(d,)) * (0.045 * boost)
                mvec2 = x[cl].mean(axis=0)
                mvec2 /= (np.linalg.norm(mvec2) + 1e-12)
                common2 = common2 - float(common2 @ mvec2) * mvec2
                x[cl] += common2[None, :]
                x[cl] += rng.standard_normal(size=(k2, d)) * (0.030 * boost)
                x[cl] /= np.linalg.norm(x[cl], axis=1, keepdims=True)
                T = best - 0.0045
                x = _hinge_anneal(x, T, steps=140, lr0=0.9)
            elif s == 4:
                # rattler / hub void reinsertion
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                cc = np.sum(G > m - 1e-9, axis=1)
                ratt = np.where(cc == 0)[0].tolist()
                w = G.max(axis=1)
                order = np.argsort(-w)
                torep = [int(v) for v in ratt[:4]]
                for o in order:
                    if len(torep) >= 5:
                        break
                    if int(o) not in torep:
                        torep.append(int(o))
                x = _greedy_reinsert(x, torep, rng, n_cand=4000)
                x += rng.standard_normal(size=x.shape) * 0.004
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                T = best - 0.003
                x = _hinge_anneal(x, T, steps=120, lr0=0.8)
            else:
                sig = 0.042 * boost
                x = x + rng.standard_normal(size=x.shape) * sig
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                T = best - 0.005
                x = _hinge_anneal(x, T, steps=130, lr0=1.0)
        except Exception:
            it += 1
            no_improve += 1
            continue
        try:
            v = _max_cos(x)
        except Exception:
            it += 1
            no_improve += 1
            continue
        if v < best - 1e-12:
            best = float(v)
            x_best = x
            no_improve = 0
        else:
            no_improve += 1
        it += 1
    remain = FINAL_END - time.time()
    if remain > 3.0:
        try:
            x_best, best = _lp_polish(x_best, FINAL_END, init_radius=4e-3, max_pairs=2200)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
