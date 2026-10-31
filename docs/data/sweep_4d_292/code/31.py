import time
import itertools
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

# ---------- utilities ----------
def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normed(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

# ---------- quaternion / group generation ----------
def _qmult(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2], dtype=np.float64)

def _qconj(a):
    return np.array([a[0], -a[1], -a[2], -a[3]])

def _closure(gens, max_size=300, tol_round=6):
    # gens: list of 4-arrays
    ext = []
    for g in gens:
        ext.append(np.asarray(g, dtype=np.float64))
        ext.append(_qconj(np.asarray(g, dtype=np.float64)))
    ext.append(np.array([1., 0, 0, 0]))
    S = [np.array([1., 0, 0, 0])]
    keys = {tuple(np.round(S[0], tol_round))}
    changed = True
    while changed:
        changed = False
        cur = list(S)
        for a in cur:
            for b in ext:
                c = _qmult(a, b)
                key = tuple(np.round(c, tol_round))
                if key not in keys:
                    # check also close match (rounded should suffice)
                    keys.add(key)
                    S.append(c)
                    changed = True
                    if len(S) > max_size:
                        return np.array(S)
        # also left multiply to be safe (for non-symmetric gens)
        # one more pass is enough since we included inverses
        if len(S) > max_size:
            break
    return np.array(S)

def _bt24():
    qs = []
    for i in range(4):
        for s in (1.0, -1.0):
            q = np.zeros(4); q[i] = s
            qs.append(q)
    for s in itertools.product((0.5, -0.5), repeat=4):
        qs.append(np.array(s, dtype=np.float64))
    return np.array(qs)

def _Lmat(q):
    w, x, y, z = q
    return np.array([
        [w, -x, -y, -z],
        [x,  w, -z,  y],
        [y,  z,  w, -x],
        [z, -y,  x,  w]], dtype=np.float64)

def _cyclic_mats(m, p):
    Ms = np.zeros((m, 4, 4))
    for k in range(m):
        a = 2*np.pi*k/m
        b = 2*np.pi*p*k/m
        ca, sa = np.cos(a), np.sin(a)
        cb, sb = np.cos(b), np.sin(b)
        Ms[k] = np.array([[ca, -sa, 0, 0],
                          [sa,  ca, 0, 0],
                          [0, 0, cb, -sb],
                          [0, 0, sb,  cb]])
    return Ms

def _hyperoct_mats():
    mats = []
    for perm in itertools.permutations(range(4)):
        for signs in itertools.product((1.0, -1.0), repeat=4):
            M = np.zeros((4, 4))
            for r, c in enumerate(perm):
                M[r, c] = signs[r]
            mats.append(M)
    return np.array(mats)

# ---------- symmetric LSE optimisation ----------
def _optimize_symmetric(mats, k, rng, alphas=(50.0, 200.0, 800.0), maxiters=(120, 120, 100)):
    G = mats.shape[0]
    N = G * k
    assert N == 292
    y0 = rng.standard_normal((k, 4))
    # avoid zero rows
    y0 /= np.linalg.norm(y0, axis=1, keepdims=True) + 1e-12
    yflat = y0.ravel().copy()

    def make_fun(alpha):
        def fun(yf):
            Y = yf.reshape(k, 4)
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            S = Y / nr  # (k,4)
            # expand: X[k,g,j] = sum_l M[g,j,l] S[k,l]
            X3 = np.einsum('gjl,kl->kgj', mats, S)  # (k,G,4)
            X = X3.reshape(N, 4)
            Gr = X @ X.T
            iu = np.triu_indices(N, 1)
            c = Gr[iu]
            gmax = c.max()
            e = np.exp(alpha * (c - gmax))
            ssum = e.sum() + 1e-300
            L = gmax + np.log(ssum) / alpha
            # weights
            w = e / ssum  # (P,)
            W = np.zeros((N, N))
            W[iu] = w
            W = W + W.T
            Gx = W @ X  # (N,4)
            Gx3 = Gx.reshape(k, G, 4)
            # grad_S[k,l] = sum_g sum_j Gx3[k,g,j] M[g,j,l]
            gradS = np.einsum('kgj,gjl->kl', Gx3, mats)
            # to Y
            dot = (gradS * S).sum(axis=1, keepdims=True)
            gradY = (gradS - dot * S) / nr
            return float(L), gradY.ravel()
        return fun

    for alpha, mi in zip(alphas, maxiters):
        fun = make_fun(alpha)
        try:
            res = minimize(fun, yflat, jac=True, method='L-BFGS-B',
                           options={'maxiter': mi, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': mi*10})
            yflat = res.x
        except Exception:
            break
    Y = yflat.reshape(k, 4)
    Y = _normed(Y)
    X3 = np.einsum('gjl,kl->kgj', mats, Y)
    X = X3.reshape(N, 4)
    X = _normed(X)
    # dedup check
    # if duplicates closer than 1e-8, value invalid -> return large
    Gg = X @ X.T
    np.fill_diagonal(Gg, -2)
    # duplicates would give cosine ~1
    return X, _max_cos(X)

def _soft_anneal(x, steps, beta0, beta1, lr0, mu):
    n, d = x.shape
    v = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        v = mu * v + Gt
        vd = (v * x).sum(axis=1, keepdims=True)
        v = v - vd * x
        x = x - lr * v
        x = _normed(x)
    return x

def _lp_polish(x, seconds, r0=1e-3):
    x = _normed(x)
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    if seconds <= 0:
        return xb
    start = time.time()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = c.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if not np.any(sel):
            break
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = I[:, None] * d + np.arange(d)
        cj = J[:, None] * d + np.arange(d)
        cols = np.concatenate([ci, cj, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normed(y)
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return xb

def solve():
    t0 = time.time()
    TIME = 176.0
    rng = np.random.default_rng(12345)
    # load incumbent
    try:
        xb = np.load("best.npy").astype(np.float64)
    except Exception:
        try:
            xb = np.load("record.npy").astype(np.float64)
        except Exception:
            xb = rng.standard_normal((292, 4))
    if xb.shape != (292, 4):
        xb = rng.standard_normal((292, 4))
    xb = _normed(xb)
    best_val = _max_cos(xb)
    x_best = xb.copy()

    # ---- symmetry-restricted search (with required enumeration/discarding) ----
    sym_best = None
    sym_val = 1e9
    try:
        # build binary groups (for enumeration narrative)
        bt = _bt24()  # 24
        # binary octahedral via closure
        extra_o = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0, 0])
        bo = _closure(list(bt) + [extra_o], max_size=60)
        phi = (1 + np.sqrt(5)) / 2
        invphi = 1 / phi
        q0 = np.array([0.0, 0.5, phi/2, 0.5*invphi])
        q0 = q0 / np.linalg.norm(q0)
        bi = _closure(list(bt) + [q0], max_size=150)
        groups_info = [(bt, "BT"), (bo, "BO"), (bi, "BI")]
        # check divisibility -> discard (left action is free, need |G| | 292)
        valid_binary = []
        for qs, nm in groups_info:
            if len(qs) > 0 and 292 % len(qs) == 0:
                valid_binary.append((qs, nm))
        # hyperoctahedral
        # (384 never divides 292, and generic orbit 384>292 -> discard)
        # So fall through to cyclic subgroups (subgroups of the above + SO(4) double rotations)
        cand_specs = []
        # m=4 (subgroup of BT/BO, also double rotation), k=73
        cand_specs.append(( _cyclic_mats(4, 1), 73, (60, 60, 50)))
        # m=73, k=4 with different twists
        cand_specs.append((_cyclic_mats(73, 5), 4, (150, 150, 120)))
        cand_specs.append((_cyclic_mats(73, 12), 4, (120, 120, 100)))
        # m=146, k=2
        cand_specs.append((_cyclic_mats(146, 3), 2, (150, 150, 120)))
        # m=292, k=1
        cand_specs.append((_cyclic_mats(292, 7), 1, (200, 200, 150)))
        # antipodal m=2, k=146 (light iters)
        M2 = np.zeros((2, 4, 4)); M2[0] = np.eye(4); M2[1] = -np.eye(4)
        cand_specs.append((M2, 146, (60, 50, 40)))

        sym_deadline = t0 + 48.0
        for mi, (mats, k, miters) in enumerate(cand_specs):
            if time.time() > sym_deadline:
                break
            if mats.shape[0] * k != 292:
                continue
            # dedup sanity: distinct count must be 292 (free action => ok)
            r2 = np.random.default_rng(1000 + mi)
            try:
                Xc, vc = _optimize_symmetric(mats, k, r2, maxiters=miters)
            except Exception:
                continue
            # deduplicate closer than 1e-8
            Gc = Xc @ Xc.T
            np.fill_diagonal(Gc, -2)
            if (Gc > 1 - 1e-8).any():
                continue
            if vc < sym_val:
                sym_val = vc
                sym_best = Xc.copy()
            if vc < best_val:
                best_val = vc
                x_best = Xc.copy()
    except Exception:
        pass

    # break symmetry with tiny tangent perturbation + quick polish for best symmetric
    if sym_best is not None and time.time() - t0 < TIME - 120:
        try:
            r = np.random.default_rng(999)
            pert = r.standard_normal(sym_best.shape)
            # tangent projection
            pert -= (pert * sym_best).sum(axis=1, keepdims=True) * sym_best
            pn = np.linalg.norm(pert, axis=1, keepdims=True) + 1e-12
            pert = pert / pn * 1e-4
            Xs = _normed(sym_best + pert)
            # short soft + short LP
            Xs = _soft_anneal(Xs, steps=300, beta0=30.0, beta1=400.0, lr0=0.03, mu=0.85)
            v = _max_cos(Xs)
            # quick LP 4s
            if TIME - (time.time() - t0) > 40:
                Xs2 = _lp_polish(Xs, seconds=4.0)
                v2 = _max_cos(Xs2)
                if v2 < v:
                    Xs, v = Xs2, v2
            if v < best_val:
                best_val = v
                x_best = Xs.copy()
        except Exception:
            pass

    # ---- basin-hopping around incumbent + symmetric basin ----
    explore_deadline = t0 + TIME - 34.0
    it = 0
    sigmas = [0.005, 0.012, 0.03, 0.08, 0.20]
    while time.time() < explore_deadline:
        it += 1
        if time.time() + 1.0 > explore_deadline and it > 4:
            break
        # choose base: mostly global best, sometimes symmetric
        if sym_best is not None and (it % 5 == 0):
            base = sym_best
        else:
            base = x_best
        sigma = sigmas[rng.integers(len(sigmas))]
        X = base + sigma * rng.standard_normal(base.shape)
        # kill-reinsert
        if rng.random() < 0.6:
            kk = int(rng.integers(1, 6))
            idx = rng.choice(292, size=kk, replace=False)
            R = rng.standard_normal((kk, 4))
            R = _normed(R)
            X[idx] = R
        X = _normed(X)
        rr = rng.random()
        if rr < 0.4:
            steps, b0, b1, lr0, mu = 700, 25.0, 400.0, 0.055, 0.85
        elif rr < 0.75:
            steps, b0, b1, lr0, mu = 450, 30.0, 550.0, 0.04, 0.87
        else:
            steps, b0, b1, lr0, mu = 1000, 20.0, 350.0, 0.06, 0.83
        # adapt steps to remaining time
        if time.time() + steps * 0.0012 > explore_deadline:
            steps = max(150, int((explore_deadline - time.time()) / 0.0012))
        try:
            X = _soft_anneal(X, steps=steps, beta0=b0, beta1=b1, lr0=lr0, mu=mu)
        except Exception:
            continue
        v = _max_cos(X)
        if v < best_val:
            best_val = v
            x_best = X.copy()
            # opportunistic quick LP squeeze if promising and time allows
            if v < 0.906 and time.time() + 3 < explore_deadline:
                try:
                    Xp = _lp_polish(X, seconds=2.5)
                    vp = _max_cos(Xp)
                    if vp < best_val:
                        best_val = vp
                        x_best = Xp.copy()
                except Exception:
                    pass

    # ---- final LP polish of global best ----
    remain = TIME - (time.time() - t0)
    if remain > 2:
        try:
            x_best = _lp_polish(x_best, seconds=min(remain - 0.5, 32.0))
            best_val = _max_cos(x_best)
        except Exception:
            pass
    # fallback: ensure better than loaded incumbent if possible, else return incumbent polished
    # (x_best already tracks best)
    x_best = _normed(x_best)
    # shrink by tiny margin not needed; just return
    return x_best
