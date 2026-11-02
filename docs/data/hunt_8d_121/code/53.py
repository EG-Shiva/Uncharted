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
    Xref = Xbest.copy()  # frozen tabu reference to force different basin
    TABU_RAD = 0.09
    TABU_W = 8.0

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
            return np.eye(d)

    # E8 roots (240) normalized
    def _e8():
        pts = []
        for i in range(d):
            for j in range(i+1, d):
                for s1 in (-1.0, 1.0):
                    for s2 in (-1.0, 1.0):
                        v = np.zeros(d)
                        v[i] = s1; v[j] = s2
                        pts.append(v)
        # half-integer: 2^8 combos with even minus
        for mask in range(256):
            v = np.ones(d) * 0.5
            nm = 0
            for k in range(d):
                if (mask >> k) & 1:
                    v[k] = -0.5
                    nm += 1
            if nm % 2 == 0:
                pts.append(v.copy())
        E = np.array(pts, dtype=float)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        return E
    try:
        E8 = _e8()
    except Exception:
        E8 = None

    def make_hinge(n_, t_, use_tabu):
        Ia, Ja = np.triu_indices(n_, 1)
        tt = float(t_)
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
            if use_tabu and n_ == n:
                try:
                    Q = _proc_Q(Y, Xref)
                    D = Y @ Q - Xref
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

    def lbfgs_hinge(Xinit, t_, maxiter=180, use_tabu=True):
        n_ = Xinit.shape[0]
        fun = make_hinge(n_, t_, use_tabu)
        xf = Xinit.ravel().copy()
        try:
            res = minimize(fun, xf, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-12, "maxfun": int(maxiter*6), "maxls": 20})
            xf = np.asarray(res.x, dtype=float)
        except Exception:
            pass
        X = xf.reshape(n_, d)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) + 1e-16
        return _normed(X)

    def lse_refine(x_init, beta, maxiter=120):
        nn = x_init.shape[0]
        Ia, Ja = np.triu_indices(nn, 1)
        x0f = x_init.ravel().copy()
        def fg(xf):
            X = xf.reshape(nn, d)
            ns = np.sqrt(np.sum(X * X, axis=1))
            ns = np.maximum(ns, 1e-12)
            Y = X / ns[:, None]
            G = Y.dot(Y.T)
            m = float(np.max(G[Ia, Ja]))
            A = beta * (G - m)
            np.clip(A, -700, 0, out=A)
            E = np.exp(A)
            # zero diag via mask
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E[Ia, Ja]))
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
                                    "gtol": 1e-7, "maxfun": int(maxiter*8), "maxls": 20})
            Xo = res.x.reshape(nn, d)
        except Exception:
            Xo = x_init.copy()
        return _normed(Xo)

    def lp_polish_timed(x_init, seconds):
        x = _normed(x_init.copy())
        cur = _max_cos(x)
        radius = 1.2e-3
        start = time.time()
        nv = n * d + 1
        while (time.time() - start < seconds and time.time() < DEADLINE - 1.0 and radius > 1e-13):
            G = x @ x.T
            c_all = G[Iall, Jall]
            m = c_all.max()
            win = max(25.0 * radius, 3e-9)
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

    def find_hole(base, nsample=15000, refine_steps=80):
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

    def per_point_stats(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rm = G.max(axis=1)
        m = float(G.max())
        cnt = (G > m - 1e-4).sum(axis=1)
        return rm, cnt, m, G

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

    # quick baseline polish
    try:
        if time.time() < DEADLINE - 120:
            Xp = lp_polish_timed(Xbest, 3.0)
            consider(Xp)
    except Exception:
        pass

    # surgery + diverse restarts
    trial = 0
    # deflated targets relative to current best at trial time
    while time.time() < DEADLINE - 12.0 and trial < 400:
        trial += 1
        mode = trial % 7
        t_targ = Vbest - 0.0012 - 0.0004 * ((trial // 7) % 3)
        try:
            if mode == 0:
                # excise overloaded + outliers, hole-regrow
                rm, cnt, m, G = per_point_stats(Xbest)
                # rank by contacts (high) and by rm
                order_c = np.argsort(-cnt)
                order_r = np.argsort(-rm)
                drop_set = set(order_c[:7].tolist() + order_r[:7].tolist())
                # also drop lowest-contact points (outliers) partly
                order_c_lo = np.argsort(cnt)
                for idx in order_c_lo[:4]:
                    drop_set.add(int(idx))
                # add 2 random for diversity
                for _ in range(2):
                    drop_set.add(int(rng.randint(0, n)))
                drop = np.array(sorted(drop_set), dtype=int)
                keep = np.ones(n, dtype=bool); keep[drop] = False
                cur = Xbest[keep].copy()
                # regrow
                while cur.shape[0] < n:
                    bc, _ = find_hole(cur, nsample=12000, refine_steps=60)
                    cur = np.vstack([cur, bc[None, :]])
                cur = _normed(cur)
                Xo = lbfgs_hinge(cur, t_targ, maxiter=170, use_tabu=True)
                v = _max_cos(Xo)
                if v < Vbest + 0.002:
                    Xo2 = lse_refine(Xo, 350.0, maxiter=80)
                    consider(Xo2)
                    v2 = _max_cos(Xo2)
                    if v2 < min(v, Vbest + 0.0005):
                        Xo = Xo2; v = v2
                consider(Xo)
                if v < Vbest + 0.0004 and time.time() < DEADLINE - 25:
                    Xlp = lp_polish_timed(Xo, 1.5)
                    consider(Xlp)
            elif mode == 1:
                # heavy shake of high-contact block only
                rm, cnt, m, G = per_point_stats(Xbest)
                order_c = np.argsort(-cnt)
                k = 18 + int(rng.randint(0, 8))
                blk = order_c[:k]
                X0 = Xbest.copy()
                X0[blk] += rng.randn(k, d) * 0.22
                # small global jitter
                X0 += rng.randn(n, d) * 0.008
                X0 = _normed(X0)
                Xo = lbfgs_hinge(X0, t_targ, maxiter=170, use_tabu=True)
                consider(Xo)
            elif mode == 2:
                # E8-seeded start
                if E8 is not None:
                    idx = rng.choice(E8.shape[0], size=n, replace=False)
                    X0 = E8[idx].copy()
                    # random rotation
                    A = rng.randn(d, d)
                    Q, _ = np.linalg.qr(A)
                    X0 = X0 @ Q
                    X0 = _normed(X0 + rng.randn(n, d) * 0.02)
                else:
                    X0 = _normed(rng.randn(n, d))
                Xo = lbfgs_hinge(X0, t_targ - 0.0005, maxiter=200, use_tabu=False)
                Xo = lse_refine(Xo, 250.0, maxiter=60)
                consider(Xo)
            elif mode == 3:
                # fresh random + hinge (different basin)
                X0 = _normed(rng.randn(n, d))
                Xo = lbfgs_hinge(X0, t_targ, maxiter=220, use_tabu=False)
                v = _max_cos(Xo)
                consider(Xo)
                if v < Vbest + 0.003:
                    Xo2 = lse_refine(Xo, 400.0, maxiter=80)
                    consider(Xo2)
            elif mode == 4:
                # single worst transplant
                rm, cnt, m, G = per_point_stats(Xbest)
                w = int(np.argmax(rm))
                base = np.delete(Xbest, w, axis=0)
                bc, _ = find_hole(base, nsample=15000, refine_steps=100)
                X0 = np.vstack([base, bc[None, :]])
                X0 = _normed(X0)
                Xo = lbfgs_hinge(X0, Vbest - 0.0006, maxiter=150, use_tabu=True)
                consider(Xo)
            elif mode == 5:
                # medium global shake + anneal
                sig = 0.05 + 0.04 * rng.rand()
                X0 = _normed(Xbest + rng.randn(n, d) * sig)
                Xo = lbfgs_hinge(X0, t_targ, maxiter=160, use_tabu=True)
                v = _max_cos(Xo)
                if v < Vbest + 0.002:
                    Xo = lse_refine(Xo, 500.0, maxiter=70)
                consider(Xo)
            else:
                # pair-kick: push top pairs apart + noise
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                # get top 8 pairs
                flat = G[Iall, Jall]
                top = np.argpartition(-flat, 8)[:8]
                X0 = Xbest.copy()
                for tt in top:
                    a = int(Iall[tt]); b = int(Jall[tt])
                    pa = X0[b] - np.dot(X0[b], X0[a]) * X0[a]
                    pb = X0[a] - np.dot(X0[a], X0[b]) * X0[b]
                    na = np.linalg.norm(pa); nb = np.linalg.norm(pb)
                    if na > 1e-12:
                        pa /= na
                    if nb > 1e-12:
                        pb /= nb
                    st = 0.03 + 0.03 * rng.rand()
                    X0[a] -= st * pa
                    X0[b] -= st * pb
                X0 += rng.randn(n, d) * 0.015
                X0 = _normed(X0)
                Xo = lbfgs_hinge(X0, t_targ, maxiter=160, use_tabu=True)
                consider(Xo)
        except Exception:
            continue

    # final promotion of best
    try:
        rem = DEADLINE - time.time() - 0.8
        if rem > 2.0:
            # LSE snap then LP
            Xb = lse_refine(Xbest, 600.0, maxiter=80)
            consider(Xb)
            rem2 = DEADLINE - time.time() - 0.6
            if rem2 > 1.0:
                Xb2 = lp_polish_timed(Xbest, min(rem2, 9.0))
                consider(Xb2)
    except Exception:
        pass
    return _normed(Xbest)
