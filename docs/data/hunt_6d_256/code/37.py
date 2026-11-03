import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_sm(x0, steps, K0, K1, lr, check=200):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(np.clip(K * (G - gmax), -60, 0.0))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        # clip
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        sc = np.minimum(1.0, 0.08 / (sn + 1e-12))
        step = step * sc
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % check == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2.5e-3, cap=4000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    eq_rows = np.repeat(np.arange(n), d)
    eq_cols = np.arange(n * d)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        mm = float(c.max())
        margin = max(25.0 * radius, 8e-5)
        sel = c > (mm - margin)
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            p = cap
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (eq_rows, eq_cols)), shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        val = _max_cos(y)
        if val < best:
            x, best = y, val
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def _build_group(kind):
    if kind == "even_swap":
        P = np.array([1, 0, 3, 2, 5, 4], dtype=int)
        perms = []; signs = []
        for mask in range(64):
            s = np.array([1.0 if ((mask >> i) & 1) == 0 else -1.0 for i in range(6)])
            if np.prod(s) < 0:
                continue
            perms.append(np.arange(6)); signs.append(s.copy())
            perms.append(P.copy()); signs.append(s.copy())
        return np.array(perms), np.array(signs)
    elif kind == "even_swap01":
        P = np.array([1, 0, 2, 3, 4, 5], dtype=int)
        perms = []; signs = []
        for mask in range(64):
            s = np.array([1.0 if ((mask >> i) & 1) == 0 else -1.0 for i in range(6)])
            if np.prod(s) < 0:
                continue
            perms.append(np.arange(6)); signs.append(s.copy())
            perms.append(P.copy()); signs.append(s.copy())
        return np.array(perms), np.array(signs)
    elif kind == "even_cyc":
        # cyclic permutation of three pairs (0,1)->(2,3)->(4,5)
        P1 = np.array([2, 3, 4, 5, 0, 1], dtype=int)
        P2 = np.array([4, 5, 0, 1, 2, 3], dtype=int)
        perms = []; signs = []
        for mask in range(64):
            s = np.array([1.0 if ((mask >> i) & 1) == 0 else -1.0 for i in range(6)])
            if np.prod(s) < 0:
                continue
            # only keep signs constant on each pair to keep group closed under cycling
            # enforce s0==s1, s2==s3, s4==s5 by mapping to pair signs
            # instead use pair-sign subgroup: 2^3=8 signs * 3 cyc * ... need 64
            # fallback: use full even set with 3 cyc would be 96; subsample by fixing s0=+1
            if s[0] < 0:
                continue
            # now 16 signs * 3 = 48
            perms.append(np.arange(6)); signs.append(s.copy())
            perms.append(P1.copy()); signs.append(s.copy())
            perms.append(P2.copy()); signs.append(s.copy())
        return np.array(perms), np.array(signs)
    elif kind == "full_sign":
        perms = []; signs = []
        for mask in range(64):
            s = np.array([1.0 if ((mask >> i) & 1) == 0 else -1.0 for i in range(6)])
            perms.append(np.arange(6)); signs.append(s)
        return np.array(perms), np.array(signs)
    elif kind == "even_only":
        perms = []; signs = []
        for mask in range(64):
            s = np.array([1.0 if ((mask >> i) & 1) == 0 else -1.0 for i in range(6)])
            if np.prod(s) < 0:
                continue
            perms.append(np.arange(6)); signs.append(s)
        return np.array(perms), np.array(signs)
    else:
        raise ValueError

def _expand(S, perms, signs):
    ns = S.shape[0]; g = perms.shape[0]
    Xr = np.empty((ns, g, 6))
    for k in range(g):
        p = perms[k]; s = signs[k]
        Xr[:, k, :] = S[:, p] * s[None, :]
    return Xr.reshape(ns * g, 6)

def _sym_adam(S0, perms, signs, steps, K0, K1, lr):
    S = S0.copy()
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    ns = S.shape[0]
    m = np.zeros_like(S); v = np.zeros_like(S)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = 1e9
    best_S = S.copy()
    g = perms.shape[0]
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        X = _expand(S, perms, signs)
        # reject collapse quickly
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        if gmax < best_val:
            best_val = gmax
            best_S = S.copy()
        W = np.exp(np.clip(K * (G - gmax), -60, 0.0))
        np.fill_diagonal(W, 0.0)
        Ssum = float(W.sum() * 0.5)
        if not np.isfinite(Ssum) or Ssum <= 0:
            continue
        gradX = (W @ X) / Ssum
        GXr = gradX.reshape(ns, g, 6)
        gradS = np.zeros_like(S)
        for k in range(g):
            p = perms[k]; s = signs[k]
            Gk = GXr[:, k, :]
            for l in range(6):
                gradS[:, p[l]] += s[l] * Gk[:, l]
        dot = np.sum(gradS * S, axis=1, keepdims=True)
        gt = gradS - dot * S
        b1p *= b1; b2p *= b2
        m *= b1; m += (1 - b1) * gt
        v *= b2; v += (1 - b2) * (gt * gt)
        mh = m / (1 - b1p); vh = v / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        sdot = np.sum(step * S, axis=1, keepdims=True)
        step = step - sdot * S
        S = S - step
        S /= np.linalg.norm(S, axis=1, keepdims=True)
    Xb = _expand(best_S, perms, signs)
    Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
    return best_S, Xb, float(_max_cos(Xb))

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_END = t0 + 172.0
    HOP_END = t0 + 146.0
    SYM_END = t0 + 52.0
    rng = _np.random.default_rng(20261103)
    n, d = 256, 6
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (n, d) and _np.all(_np.isfinite(a)):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < best:
                    best = v
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(n, d))
        x_best /= _np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    x_inc = x_best.copy()
    v_inc = best

    # ---- symmetry-generated search ----
    sym_best_X = None
    sym_best_val = 1e9
    kinds = ["even_swap", "full_sign", "even_swap01", "even_only", "even_cyc"]
    # quick LP equalize incumbent a bit (cheap)
    try:
        if _time.time() < SYM_END:
            x_best = _lp_polish(x_best, min(_time.time() + 3.0, SYM_END), r0=2e-3, cap=4000)
            best = _max_cos(x_best)
            v_inc = min(v_inc, best)
            if best <= _max_cos(x_inc):
                x_inc = x_best.copy(); v_inc = best
    except Exception:
        pass
    cnt = 0
    for kind in kinds:
        if _time.time() > SYM_END:
            break
        try:
            perms, signs = _build_group(kind)
        except Exception:
            continue
        g = perms.shape[0]
        if 256 % g != 0:
            continue
        ns = 256 // g
        # number of inits depends on remaining time
        n_init = 4 if kind in ("even_swap", "full_sign") else 2
        for ii in range(n_init):
            if _time.time() > SYM_END:
                break
            cnt += 1
            P = rng.normal(size=(ns, d))
            P /= _np.linalg.norm(P, axis=1, keepdims=True)
            P = _np.where(_np.abs(P) < 0.12, 0.12 * _np.sign(P + 1e-9), P)
            P /= _np.linalg.norm(P, axis=1, keepdims=True)
            steps = 650
            # anneal K 100 -> 3000 as directed
            try:
                _, Xb, vb = _sym_adam(P, perms, signs, steps, 100.0, 3000.0, 0.02)
            except Exception:
                continue
            if vb < sym_best_val:
                sym_best_val = vb
                sym_best_X = Xb.copy()
    # release symmetry: unconstrained Adam on best symmetric codes
    released_X = None
    released_val = 1e9
    if sym_best_X is not None:
        # try a couple of annealing strengths
        for (st, k0, k1, lr) in [(700, 80.0, 1800.0, 0.012), (600, 150.0, 2500.0, 0.008)]:
            if _time.time() > HOP_END - 30:
                break
            try:
                y = sym_best_X + 0.015 * rng.normal(size=(n, d))
                y /= _np.linalg.norm(y, axis=1, keepdims=True)
                yb, vb = _adam_sm(y, st, k0, k1, lr)
                if vb < released_val:
                    released_val = vb
                    released_X = yb.copy()
                # also plain without shake
                yb2, vb2 = _adam_sm(sym_best_X, 400, 100.0, 2000.0, 0.010)
                if vb2 < released_val:
                    released_val = vb2
                    released_X = yb2.copy()
                break
            except Exception:
                continue
        if released_X is not None and released_val < best:
            best = released_val
            x_best = released_X.copy()

    # ---- hopping mixing incumbent and symmetric basin ----
    bases = []
    if released_X is not None:
        bases.append(released_X)
    if sym_best_X is not None:
        bases.append(sym_best_X)
    # hopping loop
    cycle = 0
    while _time.time() < HOP_END:
        cycle += 1
        rem = HOP_END - _time.time()
        if rem < 4:
            break
        # pick start
        r = cycle % 10
        if r == 0 and len(bases) > 0:
            x = bases[0].copy()
            x += 0.05 * rng.normal(size=(n, d))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, K0, K1, lr = 650, 60.0, 1500.0, 0.010
        elif r == 5 and sym_best_X is not None:
            x = sym_best_X.copy()
            x += 0.03 * rng.normal(size=(n, d))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, K0, K1, lr = 600, 80.0, 1600.0, 0.010
        elif r == 7:
            x = _np.random.default_rng(1000 + cycle).normal(size=(n, d))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            steps, K0, K1, lr = 700, 80.0, 1600.0, 0.012
        else:
            x = x_best.copy()
            mode = cycle % 5
            if mode == 0:
                pick = rng.choice(n, size=22, replace=False)
                x[pick] += 0.08 * rng.normal(size=(22, d))
                steps, K0, K1, lr = 600, 60.0, 1600.0, 0.010
            elif mode == 1:
                u = rng.normal(size=d); u /= _np.linalg.norm(u)
                proj = x @ u
                thr = _np.quantile(proj, 0.88)
                cap = _np.where(proj > thr)[0]
                if len(cap) < 5:
                    cap = _np.argsort(-proj)[:20]
                tdir = rng.normal(size=d); tdir -= float(tdir @ u) * u
                tn = float(_np.linalg.norm(tdir))
                if tn < 1e-12:
                    tdir = rng.normal(size=d); tn = float(_np.linalg.norm(tdir))
                tdir /= tn
                amp = 0.06 + 0.04 * rng.random()
                x[cap] += amp * tdir[None, :] + 0.015 * rng.normal(size=(len(cap), d))
                steps, K0, K1, lr = 650, 50.0, 1400.0, 0.011
            elif mode == 2:
                G = x @ x.T; _np.fill_diagonal(G, -2.0)
                prow = G.max(axis=1)
                worst = _np.argsort(-prow)[:10]
                x[worst] += 0.06 * rng.normal(size=(10, d))
                pick = rng.choice(n, size=8, replace=False)
                x[pick] += 0.05 * rng.normal(size=(8, d))
                steps, K0, K1, lr = 550, 70.0, 1600.0, 0.009
            elif mode == 3:
                x += 0.035 * rng.normal(size=(n, d))
                steps, K0, K1, lr = 500, 100.0, 2000.0, 0.008
            else:
                pick = rng.choice(n, size=12, replace=False)
                x[pick] += 0.20 * rng.normal(size=(12, d))
                steps, K0, K1, lr = 700, 40.0, 1200.0, 0.012
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            if rem < 25:
                steps = min(steps, 380)
        try:
            xb, vb = _adam_sm(x, steps, K0, K1, lr)
        except Exception:
            continue
        if vb < best - 2e-7:
            best = vb
            x_best = xb.copy()
            # quick LP to equalize if clearly better
            try:
                x_best = _lp_polish(x_best, min(_time.time() + 2.5, T_END - 4), r0=1.5e-3, cap=4000)
                best = _max_cos(x_best)
            except Exception:
                pass
        elif vb < best + 2e-4 and _time.time() < HOP_END - 20:
            # opportunistic LP on near-best to try to push through plateau
            try:
                y = _lp_polish(xb, min(_time.time() + 1.5, HOP_END), r0=1e-3, cap=3000)
                vy = _max_cos(y)
                if vy < best:
                    best = vy
                    x_best = y.copy()
            except Exception:
                pass
    # final polish
    try:
        x_best = _lp_polish(x_best, T_END - 1.0, r0=2e-3, cap=5000)
        best = _max_cos(x_best)
    except Exception:
        pass
    if best > v_inc + 1e-12:
        x_best = x_inc.copy()
    x_best /= _np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
