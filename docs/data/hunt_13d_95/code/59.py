import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max()), G

def _gd_screen(X, steps, T0, T1, lr0, lr1, mu=0.8):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
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
        V = mu * V + Gt
        X = X - lr * V
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _hole_candidate(others, rng, n_samples=4000, refine=120):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_samples, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    md = dots.max(axis=1)
    c = cand[int(np.argmin(md))].copy()
    # quick subgradient refine of max-dot
    for it in range(refine):
        ds = others @ c
        j = int(np.argmax(ds))
        g = others[j] - float(others[j] @ c) * c
        st = 0.03 / (1.0 + it / 50.0)
        c = c - st * g
        nrm = np.linalg.norm(c)
        if nrm < 1e-12:
            c = rng.standard_normal(size=d); c /= np.linalg.norm(c)
        else:
            c /= nrm
    return c

def _lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        if time.time() - start > seconds - 0.15:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) > 1800:
            k = 1800
            idx = np.argpartition(c, -k)[-k:]
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
        obj = np.zeros(nv); obj[-1] = 1.0
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
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPL_END = t0 + 132.0
    rng = np.random.default_rng(0)
    n, d = 95, 13
    try:
        xb = np.load("best.npy").astype(float)
        if xb.shape != (n, d):
            raise ValueError("shape")
    except Exception:
        xb = rng.standard_normal(size=(n, d))
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    try:
        xr = np.load("record.npy").astype(float)
        if xr.shape == (n, d):
            xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
            vr = _max_cos(xr)
            if vr < best_val:
                best_val, xb = vr, xr.copy()
    except Exception:
        pass
    x_best = xb.copy()
    cur = xb.copy()
    cur_val = best_val
    # precompute contact info of best
    trial = 0
    while time.time() < EXPL_END and time.time() < DEADLINE - 35:
        trial += 1
        mode = trial % 8
        Xt = cur.copy()
        # contact stats from cur (cheap)
        try:
            Gtmp = Xt @ Xt.T
            np.fill_diagonal(Gtmp, -2.0)
            rmax = Gtmp.max(axis=1)
            contacts = (Gtmp > cur_val - 1e-7).sum(axis=1)
        except Exception:
            rmax = np.zeros(n); contacts = np.zeros(n)
        if mode == 0:
            sig = 0.03
            nz = rng.standard_normal(size=(n, d))
            nz -= np.sum(nz * Xt, axis=1, keepdims=True) * Xt
            Xt = Xt + sig * nz
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 700, 0.03, 0.004, 0.6, 0.08
        elif mode == 1:
            sig = 0.12
            nz = rng.standard_normal(size=(n, d))
            nz -= np.sum(nz * Xt, axis=1, keepdims=True) * Xt
            Xt = Xt + sig * nz
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 900, 0.03, 0.003, 0.6, 0.06
        elif mode == 2:
            # cap kick around random center
            center = int(rng.integers(n))
            dots = Xt[center] @ Xt.T
            k = int(rng.integers(6, 14))
            order = np.argsort(-dots)[:k]
            nz = rng.standard_normal(size=(n, d))
            nz -= np.sum(nz * Xt, axis=1, keepdims=True) * Xt
            s = np.full(n, 0.01)
            s[order] = 0.28
            Xt = Xt + s[:, None] * nz
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 900, 0.035, 0.003, 0.6, 0.06
        elif mode == 3:
            # worst single hole reinsert
            worst = int(np.argmax(rmax))
            others = np.delete(Xt, worst, axis=0)
            Xt[worst] = _hole_candidate(others, rng, n_samples=4000, refine=120)
            nz = rng.standard_normal(size=(n, d))
            nz -= np.sum(nz * Xt, axis=1, keepdims=True) * Xt
            Xt = Xt + 0.01 * nz
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 800, 0.025, 0.003, 0.5, 0.06
        elif mode == 4:
            # worst two holes
            order = np.argsort(-rmax)[:2]
            for idx in order:
                ii = int(idx)
                others = np.delete(Xt, ii, axis=0)
                Xt[ii] = _hole_candidate(others, rng, n_samples=3000, refine=80)
            nz = rng.standard_normal(size=(n, d))
            nz -= np.sum(nz * Xt, axis=1, keepdims=True) * Xt
            Xt = Xt + 0.012 * nz
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 900, 0.025, 0.003, 0.5, 0.06
        elif mode == 5:
            # rattler / low-contact large kick
            order = np.argsort(contacts)[:6]
            nz = rng.standard_normal(size=(n, d))
            nz -= np.sum(nz * Xt, axis=1, keepdims=True) * Xt
            s = np.full(n, 0.008)
            s[order] = 0.35
            Xt = Xt + s[:, None] * nz
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 900, 0.03, 0.003, 0.6, 0.06
        elif mode == 6:
            # hub + neighbours joint shake (overloaded)
            order = np.argsort(-contacts)[:4]
            # include nearest neighbours of top hub
            hub = int(order[0])
            dots = Xt[hub] @ Xt.T
            nb = np.argsort(-dots)[:10]
            nz = rng.standard_normal(size=(n, d))
            nz -= np.sum(nz * Xt, axis=1, keepdims=True) * Xt
            s = np.full(n, 0.01)
            s[nb] = 0.20
            Xt = Xt + s[:, None] * nz
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 1000, 0.03, 0.0025, 0.6, 0.05
        else:
            sig = 0.22
            nz = rng.standard_normal(size=(n, d))
            nz -= np.sum(nz * Xt, axis=1, keepdims=True) * Xt
            Xt = Xt + sig * nz
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            steps, T0, T1, lr0, lr1 = 1100, 0.04, 0.003, 0.7, 0.06
        Xt = _gd_screen(Xt, steps=steps, T0=T0, T1=T1, lr0=lr0, lr1=lr1, mu=0.8)
        v = _max_cos(Xt)
        if v < best_val - 1e-12:
            best_val = v
            x_best = Xt.copy()
            cur = Xt.copy()
            cur_val = v
        elif v < cur_val or rng.random() < 0.08:
            # annealing acceptance to change basin; allow slight worsening
            # only accept if not too bad
            if v < cur_val + 0.015:
                cur = Xt.copy()
                cur_val = v
        # occasional restart cur from best
        if trial % 25 == 0 and rng.random() < 0.5:
            cur = x_best.copy()
            cur_val = best_val
        if time.time() >= EXPL_END:
            break
    # final polish of best with remaining time
    remain = DEADLINE - time.time() - 0.7
    if remain > 3.0:
        # short re-screen at low T to tighten before LP
        x_best = _gd_screen(x_best, steps=600, T0=0.008, T1=0.0015, lr0=0.25, lr1=0.03, mu=0.8)
        best_val = _max_cos(x_best)
        x_best = _lp_polish(x_best, seconds=min(remain, 38.0), t_deadline=DEADLINE - 0.5)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
