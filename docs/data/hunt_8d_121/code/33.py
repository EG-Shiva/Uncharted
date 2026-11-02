import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    rng = np.random.RandomState(0)
    n, d = 121, 8
    iu_all = np.triu_indices(n, 1)
    Iall, Jall = iu_all[0], iu_all[1]

    def _normed(a):
        a = np.asarray(a, dtype=float)
        nr = np.linalg.norm(a, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        return a / nr

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _normed(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        cands = [_normed(rng.randn(n, d))]
    vals0 = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals0))].copy()
    Vbest = float(min(vals0))
    try:
        Rref = np.load("record.npy").astype(float)
        Rref = _normed(Rref)
    except Exception:
        Rref = Xbest.copy()

    TABU_RAD = 0.05
    TABU_W = 20.0

    def _proc_Q(Y, R):
        try:
            M = Y.T @ R
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            Q = U @ Vt
            if np.linalg.det(Q) < 0:
                U[:, -1] *= -1
                Q = U @ Vt
            return Q
        except Exception:
            return np.eye(Y.shape[1])

    def make_hinge(n_, t_, use_tabu):
        iu = np.triu_indices(n_, 1)
        Ia, Ja = iu[0], iu[1]
        tt = float(t_)
        use_t = bool(use_tabu and n_ == n)
        def fun(flat):
            X = flat.reshape(n_, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = X / nr
            G = Y @ Y.T
            c = G[Ia, Ja]
            diff = c - tt
            pos = diff > 0
            if np.any(pos):
                f = float((diff[pos] ** 2).sum())
                w = np.zeros_like(c)
                w[pos] = 2.0 * diff[pos]
            else:
                f = 0.0
                w = np.zeros_like(c)
            W = np.zeros((n_, n_))
            W[Ia, Ja] = w
            W[Ja, Ia] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            if use_t:
                try:
                    Q = _proc_Q(Y, Rref)
                    D = Y @ Q - Rref
                    fn_ = float(np.linalg.norm(D))
                    if fn_ > 1e-12:
                        rmsd = fn_ / np.sqrt(float(n_))
                        if rmsd < TABU_RAD:
                            f += TABU_W * (TABU_RAD - rmsd) ** 2
                            Gr = (D @ Q.T) / (np.sqrt(float(n_)) * fn_)
                            coeff = -2.0 * TABU_W * (TABU_RAD - rmsd)
                            GY2 = coeff * Gr
                            dot2 = (GY2 * Y).sum(axis=1, keepdims=True)
                            GX = GX + (GY2 - dot2 * Y) / nr
                except Exception:
                    pass
            if not np.isfinite(f):
                return 1e6, np.zeros_like(flat)
            g = np.nan_to_num(GX.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
            return float(f), g
        return fun

    def lbfgs_hinge(Xinit, t_, maxiter=250, use_tabu=True):
        n_ = Xinit.shape[0]
        fun = make_hinge(n_, t_, use_tabu)
        xf = Xinit.ravel().copy()
        try:
            res = minimize(fun, xf, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-12, "maxfun": int(maxiter * 6),
                                    "maxls": 20})
            xf = np.asarray(res.x, dtype=float)
        except Exception:
            pass
        X = xf.reshape(n_, d)
        X = _normed(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) + 1e-16)
        return X

    def lse_refine(x_init, beta, maxiter):
        nn = x_init.shape[0]
        x0f = x_init.ravel().copy()
        def fg(xf):
            X = xf.reshape(nn, d)
            ns = np.sqrt(np.sum(X * X, axis=1))
            ns = np.maximum(ns, 1e-12)
            Y = X / ns[:, None]
            G = Y.dot(Y.T)
            m = np.max(G - 2.0 * np.eye(nn))
            A = beta * (G - m)
            np.clip(A, -700, 0, out=A)
            E = np.exp(A)
            np.fill_diagonal(E, 0.0)
            S = np.sum(np.triu(E, 1))
            if S <= 0:
                S = 1e-300
            F = m + np.log(S) / beta
            EY = E.dot(Y)
            GradY = EY / S
            dot = np.sum(GradY * Y, axis=1)
            GradX = (GradY - dot[:, None] * Y) / ns[:, None]
            return float(F), GradX.ravel()
        try:
            res = minimize(fg, x0f, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-7, "maxfun": int(maxiter * 8),
                                    "maxls": 20})
            Xo = res.x.reshape(nn, d)
        except Exception:
            Xo = x_init.copy()
        return _normed(Xo)

    def lp_polish_timed(x_init, seconds):
        x = _normed(x_init.copy())
        cur = _max_cos(x)
        radius = 1e-3
        start = time.time()
        nv = n * d + 1
        while (time.time() - start < seconds and time.time() < DEADLINE - 1.0
               and radius > 1e-13):
            G = x @ x.T
            c_all = G[Iall, Jall]
            m = c_all.max()
            win = max(20.0 * radius, 2e-9)
            sel = c_all > m - win
            I = Iall[sel]; J = Jall[sel]; c = c_all[sel]
            p = len(I)
            if p == 0:
                break
            if p > 2500:
                idx = np.argpartition(-c, 2500)[:2500]
                I = I[idx]; J = J[idx]; c = c[idx]
                p = 2500
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = _normed(x + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
            if v < cur:
                x, cur = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return x

    def find_hole(base, nsample=20000, refine_steps=120):
        C = rng.randn(nsample, d)
        C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        D = C @ base.T
        cm = D.max(axis=1)
        k = int(np.argmin(cm))
        bc = C[k].copy()
        cur = float(cm[k])
        for st in range(refine_steps):
            sig = 0.03 * (1.0 - st / float(refine_steps)) + 0.001
            prop = bc + sig * rng.randn(d)
            pn = np.linalg.norm(prop)
            if pn == 0:
                continue
            prop /= pn
            md = float((prop @ base.T).max())
            if md < cur:
                bc = prop
                cur = md
        return bc, cur

    def kick_max_pair(x, alpha):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        k = int(np.argmax(G))
        a = k // n; b = k % n
        pa = x[b] - np.dot(x[b], x[a]) * x[a]
        pb = x[a] - np.dot(x[a], x[b]) * x[b]
        na = np.linalg.norm(pa); nb = np.linalg.norm(pb)
        if na > 1e-12:
            pa /= na
        if nb > 1e-12:
            pb /= nb
        y = x.copy()
        y[a] -= alpha * pa
        y[b] -= alpha * pb
        y += rng.randn(n, d) * (alpha * 0.25)
        return _normed(y)

    # ---- build optimized small codes 115..120 ----
    small_pool = []
    for m_t in (115, 118, 120):
        if time.time() > DEADLINE - 140:
            break
        try:
            G = Xbest @ Xbest.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            order = np.argsort(-rm)
            drop = order[:(n - m_t)]
            keep = np.ones(n, dtype=bool)
            keep[drop] = False
            sub = Xbest[keep].copy()
            t_small = Vbest - 0.001
            sub_opt = lbfgs_hinge(sub, t_small, maxiter=120, use_tabu=False)
            small_pool.append(sub_opt)
            vsub = _max_cos(sub_opt)
            _ = vsub
        except Exception:
            pass
    if not small_pool:
        small_pool = [Xbest[:118].copy()]

    def grow_to_121(m_code, t_target, maxiter_each=90):
        cur = m_code.copy()
        while cur.shape[0] < n:
            if time.time() > DEADLINE - 25:
                break
            try:
                bc, _ = find_hole(cur, nsample=15000, refine_steps=80)
            except Exception:
                bc = rng.randn(d); bc /= np.linalg.norm(bc)
            cur = np.vstack([cur, bc[None, :]])
            cur = _normed(cur)
            try:
                cur = lbfgs_hinge(cur, t_target, maxiter=maxiter_each, use_tabu=True)
            except Exception:
                pass
        return _normed(cur)

    survivors = []

    def consider(X):
        nonlocal Xbest, Vbest
        try:
            v = _max_cos(X)
        except Exception:
            return
        if not np.isfinite(v):
            return
        if v < Vbest:
            Vbest = float(v)
            Xbest = _normed(X.copy())
        # keep near-feasible distinct survivors
        if len(survivors) < 12:
            survivors.append((float(v), _normed(X.copy())))
        else:
            # replace worst if better
            survivors.sort(key=lambda z: z[0])
            if float(v) < survivors[-1][0]:
                survivors[-1] = (float(v), _normed(X.copy()))

    targets = [0.4600, 0.4595, 0.4590, 0.4585, 0.4580, 0.4575]
    # quick tabu-deflated hinge trials
    for ti, t_targ in enumerate(targets):
        if time.time() > DEADLINE - 30:
            break
        reps = 6 if ti < 2 else 4
        for rep in range(reps):
            if time.time() > DEADLINE - 28:
                break
            mode = rep % 3
            try:
                if mode == 0:
                    X0 = _normed(rng.randn(n, d))
                elif mode == 1:
                    if rep % 2 == 0:
                        X0 = _normed(Xbest + rng.randn(n, d) * 0.02)
                    else:
                        X0 = kick_max_pair(Xbest, 0.01 + rng.rand() * 0.015)
                else:
                    base = small_pool[(rep + ti) % len(small_pool)]
                    # re-optimize small base a bit at lower target then grow
                    X0 = grow_to_121(base, t_targ, maxiter_each=70)
                    # X0 already optimized inside grow; evaluate directly
                    consider(X0)
                    continue
                # hinge optimize
                mi = 250 if mode == 0 else 200
                Xo = lbfgs_hinge(X0, t_targ, maxiter=mi, use_tabu=True)
                consider(Xo)
                # if feasible, extra LSE snap immediately (cheap)
                v = _max_cos(Xo)
                if v <= t_targ + 1e-7 and time.time() < DEADLINE - 25:
                    Xs = lse_refine(Xo, 300, 120)
                    consider(Xs)
            except Exception:
                continue

    # promote best few distinct survivors to full LSE annealing + LP
    try:
        survivors.sort(key=lambda z: z[0])
    except Exception:
        pass
    # distinct filter by Frobenius distance
    distinct = []
    for v, X in survivors:
        ok = True
        for _, Y in distinct:
            try:
                Q = _proc_Q(X, Y)
                dd = float(np.linalg.norm(X @ Q - Y)) / np.sqrt(float(n))
                if dd < 0.01:
                    ok = False
                    break
            except Exception:
                pass
        if ok:
            distinct.append((v, X))
        if len(distinct) >= 3:
            break
    for v, X in distinct:
        if time.time() > DEADLINE - 12:
            break
        try:
            x = X.copy()
            for beta, mi in [(120, 120), (300, 150), (600, 150)]:
                if time.time() > DEADLINE - 10:
                    break
                x = lse_refine(x, beta, mi)
                consider(x)
            if time.time() < DEADLINE - 6:
                x2 = lp_polish_timed(x, seconds=min(3.0, DEADLINE - 2.0 - time.time()))
                consider(x2)
        except Exception:
            continue

    # final squeeze of global best
    if time.time() < DEADLINE - 4:
        try:
            xb = lp_polish_timed(Xbest, seconds=min(6.0, DEADLINE - 1.5 - time.time()))
            vb = _max_cos(xb)
            if vb < Vbest:
                Vbest = vb
                Xbest = xb.copy()
        except Exception:
            pass

    Xbest = _normed(Xbest)
    return Xbest
