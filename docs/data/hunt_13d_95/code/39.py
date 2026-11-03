import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def gd_refine(X, steps, T0, T1, lr0, lr1, mu=0.9, wc=0.3, wcov=0.3):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    tgt = np.eye(d) / d
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # isotropy regularization, decayed to zero at end
        reg_w = (1.0 - f)
        if reg_w > 0 and (wc > 0 or wcov > 0):
            c = X.mean(axis=0, keepdims=True)  # (1,d)
            if wc > 0:
                # grad of ||mean||^2 projected
                Gc = (c - (X * (X @ c.T)).reshape(n, 1) * np.ones_like(X) * 0 + (np.repeat((X @ c.T).reshape(n, 1), d, axis=1) * 0))  # placeholder
                # simpler: (c - (c.x_i) x_i)/n
                cx = float(np.sum(c * c))  # not needed
                # per point: c_vec - (c_vec.x_i) x_i
                cvec = c.reshape(1, d).repeat(n, axis=0)
                proj = np.sum(cvec * X, axis=1, keepdims=True) * X
                Gc = (cvec - proj) / n
                Gt = Gt + wc * reg_w * Gc
            if wcov > 0:
                S = (X.T @ X) / n
                E = S - tgt
                Gv = (X @ E) * (4.0 / n)
                dotv = np.sum(Gv * X, axis=1, keepdims=True)
                Gv = Gv - dotv * X
                Gt = Gt + wcov * reg_w * Gv
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def random_hole(others, rng, n_cand=2000):
    d = others.shape[1]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    md = (C @ others.T).max(axis=1)
    j = int(np.argmin(md))
    return C[j]


def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 3e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        if t_deadline is not None and time.time() > t_deadline - 2.0:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPL_END = t0 + 140.0
    rng = np.random.default_rng(12345)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (95, 13):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (95, 13):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        z = rng.standard_normal((95, 13))
        cands.append(z / np.linalg.norm(z, axis=1, keepdims=True))
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])
    n, d = best.shape

    cur = best.copy()
    cur_val = best_val

    trial = 0
    # fast screening loop: NO linprog inside
    while time.time() < EXPL_END and time.time() < DEADLINE - 30:
        trial += 1
        mode = trial % 7
        # choose base: mostly best, sometimes cur
        base = best if (rng.random() < 0.75) else cur
        Xt = base.copy()
        if mode in (0, 1, 2):
            sig = [0.06, 0.16, 0.38][mode]
            Xt = Xt + sig * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            if mode == 0:
                Xt = gd_refine(Xt, 500, 0.012, 0.0012, 0.35, 0.03)
            elif mode == 1:
                Xt = gd_refine(Xt, 600, 0.016, 0.0010, 0.45, 0.03)
            else:
                Xt = gd_refine(Xt, 700, 0.020, 0.0010, 0.55, 0.04)
        elif mode in (3, 4, 5):
            # joint block reinsertion of hubs (+rattler)
            ppm = per_point_max(base)
            order = np.argsort(-ppm)  # worst first
            if mode == 3:
                k = int(rng.integers(2, 5))
                # top-k hubs
                idx = order[:k].copy()
            elif mode == 4:
                k = int(rng.integers(4, 9))
                # mix hubs and random
                idx1 = order[:2].tolist()
                idx2 = rng.choice(n, size=k - 2, replace=False).tolist()
                idx = np.array(idx1 + idx2)
                idx = np.unique(idx)
                k = len(idx)
            else:
                k = int(rng.integers(2, 5))
                # lowest-contact style: use row max smallest? include rattlers
                idx = np.concatenate([order[:k - 1], order[-1:]])
            # sequential best-hole filling
            Xtmp = Xt.copy()
            # remove first, then fill one by one
            for j, ii in enumerate(idx):
                others = np.delete(Xtmp, ii, axis=0)
                nh = random_hole(others, rng, n_cand=2500)
                Xtmp[ii] = nh
            Xt = Xtmp / np.linalg.norm(Xtmp, axis=1, keepdims=True)
            # short jitter then anneal
            Xt = Xt + 0.01 * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 600, 0.014, 0.0009, 0.45, 0.03)
        else:
            # cap-localized shake: pick a random center, shake its neighbors strongly
            ci = int(rng.integers(0, n))
            dots = base @ base[ci]
            neigh = np.argsort(-dots)[:int(rng.integers(8, 20))]
            Xt[neigh] += 0.25 * rng.standard_normal((len(neigh), d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_refine(Xt, 550, 0.014, 0.0010, 0.40, 0.03)

        v = max_cos(Xt)
        # update best
        if v < best_val:
            best = Xt.copy()
            best_val = v
            cur = Xt.copy()
            cur_val = v
        else:
            # Metropolis for cur to wander (temperature 0.002)
            if v < cur_val or rng.random() < np.exp(-(v - cur_val) / 0.002):
                cur = Xt.copy()
                cur_val = v
        # early exit if clearly record-beating
        if best_val < 0.2435 and time.time() > t0 + 60:
            pass

    # final LP squeeze on best (time left)
    remain = DEADLINE - time.time()
    if remain > 5:
        try:
            best = lp_polish(best, min(remain - 1.0, 28.0), t_deadline=DEADLINE - 1.0)
        except Exception:
            pass
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
