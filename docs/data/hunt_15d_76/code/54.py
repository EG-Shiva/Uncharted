import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _expand(gens):
    # gens (5,15) -> X (76,15)
    gn = _norm(gens)
    X = np.zeros((76, 15))
    for i in range(5):
        g = gn[i]
        for s in range(15):
            X[i*15+s] = np.roll(g, s)
    X[75] = np.ones(15) / np.sqrt(15.0)
    return X

def _sym_val_grad(flat, T):
    Gmat = flat.reshape(5, 15)
    norms = np.linalg.norm(Gmat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    NG = Gmat / norms
    # expand
    X = np.zeros((76, 15))
    for i in range(5):
        g = NG[i]
        for s in range(15):
            X[i*15+s] = np.roll(g, s)
    X[75] = np.ones(15) / np.sqrt(15.0)
    Gr = X @ X.T
    np.fill_diagonal(Gr, -2.0)
    m = float(Gr.max())
    # stable exp
    E = np.exp((Gr - m) / T)
    np.fill_diagonal(E, 0.0)
    S = float(E.sum())
    if (not np.isfinite(S)) or S <= 0:
        S = 1e-300
        L = m
        W = np.zeros_like(E)
    else:
        L = m + T * np.log(S)
        W = E / S
    # scale pin penalty to remove scale invariance
    pen = float(np.sum((norms[:, 0] - 1.0) ** 2))
    L = L + pen
    # grad w.r.t X
    GX = 2.0 * (W @ X)  # (76,15)
    grad_NG = np.zeros((5, 15))
    for i in range(5):
        acc = np.zeros(15)
        block = GX[i*15:(i+1)*15]
        for s in range(15):
            acc += np.roll(block[s], -s)
        grad_NG[i] = acc
    dot = np.sum(grad_NG * NG, axis=1, keepdims=True)
    grad_G = (grad_NG - dot * NG) / norms
    # penalty grad: 2*(r-1)*G/r
    coef = (2.0 * (norms[:, 0] - 1.0))  # (5,)
    for i in range(5):
        grad_G[i] += coef[i] * (Gmat[i] / float(norms[i, 0]))
    return L, grad_G.ravel()

def _optimize_gens(g0, Ts, maxiters):
    from scipy.optimize import minimize
    x = g0.ravel().copy()
    for T, mi in zip(Ts, maxiters):
        def fun(x_):
            v, g_ = _sym_val_grad(x_, T)
            return v, g_
        try:
            res = minimize(fun, x, method='L-BFGS-B', jac=True,
                           options={'maxiter': int(mi), 'ftol': 1e-14, 'gtol': 1e-9, 'maxfun': int(mi*4+10)})
            x = res.x
        except Exception:
            break
    return x.reshape(5, 15)

def _anneal(X, steps, T0=0.006, T1=0.0008, lr0=0.02, lr1=0.002):
    n, d = X.shape
    Y = X.copy()
    m = np.zeros_like(Y)
    vv = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    ll = np.log(lr1 / lr0)
    lt = np.log(T1 / T0)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 0.0
        lr = lr0 * np.exp(ll * frac)
        T = T0 * np.exp(lt * frac)
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp((G - gm) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if (not np.isfinite(S)) or S <= 0:
            break
        Gr = 2.0 * (E @ Y) / S
        Gt = Gr - np.sum(Gr * Y, axis=1, keepdims=True) * Y
        b1t *= b1; b2t *= b2
        m *= b1; m += (1 - b1) * Gt
        vv *= b2; vv += (1 - b2) * (Gt * Gt)
        mh = m / (1 - b1t)
        vh = vv / (1 - b2t)
        Y = Y - lr * mh / (np.sqrt(vh) + eps)
        Y = _norm(Y)
    return Y

def _lp_polish(x, t_end, max_constraints=1200):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    try:
        iu = np.triu_indices(n, 1)
    except Exception:
        return x
    best = _maxcos(x)
    radius = 1e-3
    while time.time() < t_end - 0.5 and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > mcur - thr
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > max_constraints:
            idx = np.argpartition(-c, max_constraints)[:max_constraints]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 165.0
    # load fallback best
    best_global = None
    bv_global = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (76, 15):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _maxcos(a)
                    if v < bv_global:
                        bv_global = v
                        best_global = a.copy()
        except Exception:
            pass
    if best_global is None:
        rng0 = np.random.default_rng(0)
        best_global = _norm(rng0.normal(size=(76, 15)))
        bv_global = _maxcos(best_global)
    # ---- symmetric C15 stage ----
    Ts = [1.0/50.0, 1.0/150.0, 1.0/450.0]
    maxiters = [250, 250, 350]
    rng = np.random.default_rng(12345)
    sym_cands = []
    n_restart = 6
    for r in range(n_restart):
        if time.time() > deadline - 90:
            break
        g0 = rng.normal(size=(5, 15))
        # avoid near-constant generators (orbit collapse / clash with all-ones)
        for i in range(5):
            # demean slightly to stay away from all-ones fixed point
            mu = g0[i].mean()
            g0[i] -= 0.5 * mu
            nrm = np.linalg.norm(g0[i])
            if nrm < 1e-8 or not np.isfinite(nrm):
                g0[i] = rng.normal(size=15)
            g0[i] /= (np.linalg.norm(g0[i]) + 1e-12)
            g0[i] *= np.sqrt(15.0) * 0.9 + 0.1
        try:
            gopt = _optimize_gens(g0, Ts, maxiters)
        except Exception:
            continue
        # reject orbit-collapsed generators
        try:
            Xo = _expand(gopt)
            if not np.all(np.isfinite(Xo)):
                continue
            G = Xo @ Xo.T
            np.fill_diagonal(G, -2.0)
            mv = float(G.max())
            # intra-orbit check: first 75 rows in 5 blocks
            bad = False
            for i in range(5):
                blk = Xo[i*15:(i+1)*15] @ Xo[i*15:(i+1)*15].T
                np.fill_diagonal(blk, -2.0)
                if float(blk.max()) > 0.95:
                    bad = True
                    break
            if bad:
                continue
            sym_cands.append((mv, Xo, gopt.copy()))
        except Exception:
            continue
    # sort symmetric candidates
    sym_cands.sort(key=lambda t: t[0])
    # release symmetry: short anneal on top candidates
    released = []
    for k, (mv, Xo, gopt) in enumerate(sym_cands[:3]):
        if time.time() > deadline - 70:
            break
        try:
            steps = 1200 if k == 0 else 800
            Yr = _anneal(Xo, steps=steps, T0=0.006, T1=0.0008, lr0=0.02, lr1=0.002)
            v = _maxcos(Yr)
            released.append((v, Yr))
        except Exception:
            continue
    # pick best released, further anneal + compare
    if released:
        released.sort(key=lambda t: t[0])
        cur = released[0][1]
        curv = released[0][0]
        # extra settle anneal for the leader if time allows
        if time.time() < deadline - 60:
            try:
                cur2 = _anneal(cur, steps=1200, T0=0.003, T1=0.0005, lr0=0.008, lr1=0.001)
                v2 = _maxcos(cur2)
                if v2 < curv:
                    cur, curv = cur2, v2
            except Exception:
                pass
        if curv < bv_global and np.all(np.isfinite(cur)):
            best_global = cur.copy()
            bv_global = curv
        # also consider second released if better after extra polish (unlikely)
        # LP polish the leader
        remain = deadline - time.time()
        if remain > 5:
            cap = min(remain - 2.0, 70.0)
            t_end = time.time() + cap
            try:
                pol = _lp_polish(cur, t_end)
                vp = _maxcos(pol)
                if vp < bv_global and np.all(np.isfinite(pol)):
                    best_global = pol.copy()
                    bv_global = vp
            except Exception:
                pass
    # if we still have time and never beat record, at least LP-polish fallback briefly
    # only if symmetric path failed to beat (to avoid wasting); do short polish of best_global
    # best_global is already polished, so skip unless we improved.
    best_global = _norm(best_global)
    if (not np.all(np.isfinite(best_global))) or best_global.shape != (76, 15):
        rngf = np.random.default_rng(1)
        best_global = _norm(rngf.normal(size=(76, 15)))
    return best_global
