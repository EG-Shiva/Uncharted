import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    x = x / n
    return np.nan_to_num(x)

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lse_grad_X(X, T):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    E = (G - m) / T
    np.clip(E, -20.0, 0.0, out=E)
    W = np.exp(E)
    np.fill_diagonal(W, 0.0)
    s = float(W.sum())
    if s <= 0:
        s = 1.0
    lse = m + T * np.log(s * 0.5 + 1e-300)
    grad = (W @ X) * (2.0 / s)
    return lse, grad, m

def _adam_antipodal(Y0, steps, lr, T0, T1, rng):
    Y = _proj(Y0.copy())
    k, d = Y.shape
    m1 = np.zeros_like(Y)
    v1 = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (1 - frac) + T1 * frac
        X = np.vstack([Y, -Y])
        _, gx, _ = _lse_grad_X(X, T)
        gt = gx[:k] - gx[k:]
        # tangent
        dot = np.einsum('ij,ij->i', gt, Y)
        gt = gt - dot[:, None] * Y
        m1 = b1 * m1 + (1 - b1) * gt
        v1 = b2 * v1 + (1 - b2) * gt * gt
        mh = m1 / (1 - b1 ** (s + 1))
        vh = v1 / (1 - b2 ** (s + 1))
        step = lr * mh / (np.sqrt(vh) + eps)
        sd = np.einsum('ij,ij->i', step, Y)
        step = step - sd[:, None] * Y
        Y = _proj(Y - step)
    X = np.vstack([Y, -Y])
    return Y, X

def _adam_orbit(G0, perms, signs, inv_perms, s_inv, steps, lr, T0, T1):
    G = _proj(G0.copy())
    k, d = G.shape
    nm = len(perms)
    m1 = np.zeros_like(G)
    v1 = np.zeros_like(G)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (1 - frac) + T1 * frac
        Yn = G  # already unit
        # build X
        blocks = []
        for t in range(nm):
            p = perms[t]; sg = signs[t]
            blocks.append(Yn[:, p] * sg[None, :])
        X = np.vstack(blocks)
        _, gx, _ = _lse_grad_X(X, T)
        gradY = np.zeros_like(Yn)
        for t in range(nm):
            gb = gx[t * k:(t + 1) * k]
            ip = inv_perms[t]; si = s_inv[t]
            gradY += gb[:, ip] * si[None, :]
        dot = np.einsum('ij,ij->i', gradY, Yn)
        gradY = gradY - dot[:, None] * Yn
        m1 = b1 * m1 + (1 - b1) * gradY
        v1 = b2 * v1 + (1 - b2) * gradY * gradY
        mh = m1 / (1 - b1 ** (s + 1))
        vh = v1 / (1 - b2 ** (s + 1))
        step = lr * mh / (np.sqrt(vh) + eps)
        sd = np.einsum('ij,ij->i', step, Yn)
        step = step - sd[:, None] * Yn
        G = _proj(Yn - step)
    # final X
    blocks = []
    for t in range(nm):
        p = perms[t]; sg = signs[t]
        blocks.append(G[:, p] * sg[None, :])
    X = np.vstack(blocks)
    return G, X

def _lp_polish(x, seconds):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        csv = c[sel]
        if p > 2500:
            idx = np.argpartition(-csv, 2500)[:2500]
            I, J = I[idx], J[idx]
            csv = csv[idx]
            p = 2500
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs", options={"time_limit": 4.0})
            except Exception:
                res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def _unconstrained_quench(X0, steps, lr0, T0, T1, mu=0.8):
    Y = _proj(X0.copy())
    vel = np.zeros_like(Y)
    for s in range(steps):
        frac = s / max(1, steps)
        T = T0 * (1 - frac) + T1 * frac
        _, g, _ = _lse_grad_X(Y, T)
        dot = np.einsum('ij,ij->i', g, Y)
        g = g - dot[:, None] * Y
        vel = mu * vel + g
        lr = lr0 * (1 - 0.6 * frac)
        Y = Y - lr * vel
        Y = _proj(Y)
    return Y

def solve():
    Tstart = time.time()
    LIMIT = 160.0
    def elapsed():
        return time.time() - Tstart
    def left():
        return LIMIT - elapsed()
    rng = np.random.default_rng(777)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = np.load("record.npy")
    b0 = _proj(np.asarray(b0, dtype=float))
    if b0.shape != (54, 21):
        b0 = _proj(rng.normal(size=(54, 21)))
    try:
        r0 = _proj(np.asarray(np.load("record.npy"), dtype=float))
        if r0.shape != (54, 21):
            r0 = b0
    except Exception:
        r0 = b0
    bv0 = _max_cos(b0)
    rv0 = _max_cos(r0)
    base = b0 if bv0 <= rv0 else r0
    base_val = min(bv0, rv0)

    sym_best = None
    sym_val = 1e9

    # Phase A: 27 antipodal pairs (27 free vectors)
    if left() > 30:
        for trial in range(3):
            if left() < 25:
                break
            Y0 = rng.normal(size=(27, 21))
            Y0 = _proj(Y0)
            steps = 900 if trial < 2 else 600
            _, X = _adam_antipodal(Y0, steps=steps, lr=0.03, T0=0.02, T1=0.002, rng=rng)
            v = _max_cos(X)
            if v < sym_val:
                sym_val = v
                sym_best = X.copy()

    # Phase B: orbits of k generators under random signed perms
    configs = [(3, 18), (3, 18), (6, 9), (6, 9), (2, 27), (6, 9)]
    for (k, m) in configs:
        if left() < 35:
            break
        if k * m != 54:
            continue
        perms = []
        signs = []
        inv_perms = []
        s_inv = []
        for _ in range(m):
            p = rng.permutation(21)
            sg = rng.choice(np.array([-1.0, 1.0]), size=21)
            ip = np.empty_like(p)
            ip[p] = np.arange(21)
            perms.append(p)
            signs.append(sg)
            inv_perms.append(ip)
            s_inv.append(sg[ip])
        G0 = _proj(rng.normal(size=(k, 21)))
        _, X = _adam_orbit(G0, perms, signs, inv_perms, s_inv,
                           steps=700, lr=0.03, T0=0.02, T1=0.002)
        v = _max_cos(X)
        if v < sym_val:
            sym_val = v
            sym_best = X.copy()

    # select best constrained result; fallback to base if none
    if sym_best is None:
        sym_best = base.copy()
        sym_val = base_val

    # break exact symmetry with tiny jitter before release
    if left() > 35:
        jn = rng.normal(size=sym_best.shape) * 2e-4
        jn -= (jn * sym_best).sum(axis=1, keepdims=True) * sym_best
        sym_rel = _proj(sym_best + jn)
        # short unconstrained quench to leave symmetric basin
        qsteps = 400
        sym_rel = _unconstrained_quench(sym_rel, steps=qsteps, lr0=0.05, T0=0.01, T1=0.0015)
        # LP unconstrained at most 30 s
        budget_sym = min(30.0, left() - 14.0)
        if budget_sym > 1.0:
            sym_rel = _lp_polish(sym_rel, budget_sym)
        sym_rel_val = _max_cos(sym_rel)
        try:
            np.save("/tmp/sym_unconstrained.npy", sym_rel)
        except Exception:
            pass
    else:
        sym_rel = sym_best
        sym_rel_val = sym_val

    # compare against starting from best.npy: short polish of base
    base_pol = base.copy()
    budget_base = min(10.0, left() - 2.0)
    if budget_base > 1.0:
        base_pol = _lp_polish(base_pol, budget_base)
    base_pol_val = _max_cos(base_pol)

    if sym_rel_val < base_pol_val:
        best = sym_rel
    else:
        best = base_pol
    best = _proj(best)
    return best
