import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)


def gd_refine(X, steps, T0, T1, lr0, lr1, mu=0.9, t_deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for s in range(steps):
        if t_deadline is not None and (s % 200 == 0) and time.time() > t_deadline:
            break
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = G.max()
        # softmax over upper pairs (symmetrized)
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # clip extreme grads for stability
        # V update
        V = mu * V + Gt
        X = X - lr * V
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X = X / nrm
    return X


def hole_best_candidate(X_others, rng, n_cand=2000):
    n0, d = X_others.shape
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    dots = C @ X_others.T
    m = dots.max(axis=1)
    b = int(np.argmin(m))
    return C[b], float(m[b])


def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        # keep global best even if drift
        if best < max_cos(best_x):
            best_x = x.copy()
    return best_x


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    search_end = t0 + 138.0
    rng = np.random.default_rng(12345)

    # load start
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (95, 13):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        cands.append(rng.standard_normal((95, 13)))
        cands[0] /= np.linalg.norm(cands[0], axis=1, keepdims=True)
    vals = [max_cos(a) for a in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    # pool of elites
    elites = [best.copy()]
    elite_vals = [best_val]

    n, d = 95, 13
    trial = 0
    # quick LP pre-polish check not needed; go straight to hopping
    while time.time() < search_end and time.time() < deadline - 35:
        trial += 1
        # pick base: mostly best, sometimes 2nd best
        if len(elites) > 1 and (trial % 7 == 0):
            bi = int(rng.integers(len(elites)))
            base = elites[bi]
        else:
            base = best
        Xt = base.copy()
        mode = trial % 8

        if mode == 0:
            sig = 0.02
            Xt = Xt + sig * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 900, 0.010, 0.0007, 0.06, 0.008, 0.9, deadline - 35)
        elif mode == 1:
            sig = 0.06
            Xt = Xt + sig * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 900, 0.014, 0.0008, 0.09, 0.010, 0.9, deadline - 35)
        elif mode == 2:
            sig = 0.15
            Xt = Xt + sig * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 1000, 0.018, 0.0010, 0.12, 0.010, 0.9, deadline - 35)
        elif mode == 3:
            sig = 0.30
            Xt = Xt + sig * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 1000, 0.022, 0.0012, 0.14, 0.010, 0.88, deadline - 35)
        elif mode == 4:
            # subset reinsertion: worst k points
            pm = per_point_max(Xt)
            k = int(rng.choice([2, 3, 4, 6, 8]))
            idx = np.argsort(pm)[-k:]
            # small jitter others
            Xt = Xt + 0.01 * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            for j in idx:
                others = np.delete(Xt, j, axis=0)
                v, _ = hole_best_candidate(others, rng, n_cand=3000)
                Xt[j] = v
            Xt = gd_refine(Xt, 900, 0.012, 0.0007, 0.08, 0.008, 0.9, deadline - 35)
        elif mode == 5:
            # random k points hole reinsertion
            k = int(rng.integers(2, 10))
            idx = rng.choice(n, size=k, replace=False)
            Xt = Xt + 0.01 * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            for j in idx:
                others = np.delete(Xt, j, axis=0)
                v, _ = hole_best_candidate(others, rng, n_cand=3000)
                Xt[j] = v
            Xt = gd_refine(Xt, 900, 0.012, 0.0007, 0.08, 0.008, 0.9, deadline - 35)
        elif mode == 6:
            # rattler / low-contact targeted: compute contacts
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            m = G.max()
            cnt = (G > m - 1e-7).sum(axis=1)
            # pick low-contact + high-contact mix
            order_low = np.argsort(cnt)
            k1 = int(rng.integers(1, 4))
            k2 = int(rng.integers(1, 4))
            idx_low = order_low[:k1]
            pm = per_point_max(Xt)
            idx_high = np.argsort(pm)[-k2:]
            idx = np.unique(np.concatenate([idx_low, idx_high]))
            for j in idx:
                others = np.delete(Xt, j, axis=0)
                v, _ = hole_best_candidate(others, rng, n_cand=4000)
                # mix with old for diversity
                if rng.random() < 0.3:
                    v = v + 0.2 * rng.standard_normal(d)
                    v /= np.linalg.norm(v)
                Xt[j] = v
            Xt = gd_refine(Xt, 1000, 0.015, 0.0008, 0.10, 0.008, 0.9, deadline - 35)
        else:
            # heavy shake of a local patch: pick random direction, shake points near it
            vdir = rng.standard_normal(d)
            vdir /= np.linalg.norm(vdir)
            dots = Xt @ vdir
            # pick cap
            thr = np.quantile(dots, rng.uniform(0.5, 0.9))
            mask = dots >= thr
            if mask.sum() < 2:
                mask = rng.random(n) < 0.1
            Xt[mask] += 0.25 * rng.standard_normal((int(mask.sum()), d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 1000, 0.016, 0.0009, 0.11, 0.009, 0.9, deadline - 35)

        v = max_cos(Xt)
        if v < best_val - 1e-12:
            best_val = v
            best = Xt.copy()
            elites.append(Xt.copy())
            elite_vals.append(v)
            # keep top 4
            if len(elites) > 4:
                o = np.argsort(elite_vals)
                elites = [elites[i] for i in o[:4]]
                elite_vals = [elite_vals[i] for i in o[:4]]
        elif v < min(elite_vals) + 0.002 and len(elites) < 4 and (trial % 10 == 0):
            # diversity reservoir: keep near-best occasionally
            pass

        # opportunistic quick LP if promising (within 1e-3 of best)
        if v < best_val + 0.0005 and (trial % 25 == 0) and time.time() < deadline - 50:
            try:
                Y = lp_polish(Xt, 2.0, deadline - 35)
                vy = max_cos(Y)
                if vy < best_val - 1e-12:
                    best_val = vy
                    best = Y.copy()
                    elites.append(Y.copy())
                    elite_vals.append(vy)
                    if len(elites) > 4:
                        o = np.argsort(elite_vals)
                        elites = [elites[i] for i in o[:4]]
                        elite_vals = [elite_vals[i] for i in o[:4]]
            except Exception:
                pass

    # final polish of best (and runner up, pick winner)
    remain = max(5.0, deadline - time.time() - 2.0)
    try:
        best = lp_polish(best, min(30.0, remain * 0.7), deadline - 1.0)
        best_val = max_cos(best)
    except Exception:
        pass
    # polish second elite briefly if time
    if len(elites) > 1 and time.time() < deadline - 8:
        try:
            o = np.argsort(elite_vals)
            for ii in o[1:2]:
                Y = lp_polish(elites[ii], 4.0, deadline - 1.0)
                if max_cos(Y) < best_val:
                    best = Y
                    best_val = max_cos(Y)
        except Exception:
            pass

    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
