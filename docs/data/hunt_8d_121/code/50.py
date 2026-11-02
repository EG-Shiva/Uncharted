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

    def contacts(X, thr=1e-7):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        C = G >= m - thr
        np.fill_diagonal(C, False)
        cnt = C.sum(axis=1)
        return m, cnt, G

    def make_hinge(n_, t_):
        iu0 = np.triu_indices(n_, 1)
        Ia, Ja = iu0[0], iu0[1]
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
            if not np.isfinite(f):
                return 1e6, np.zeros_like(flat)
            g = np.nan_to_num(GX.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
            return float(f), g
        return fun

    def lbfgs_hinge(Xinit, t_, maxiter=150):
        n_ = Xinit.shape[0]
        fun = make_hinge(n_, t_)
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

    def lse_refine(x_init, beta, maxiter=120):
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

    def riesz_refine(x_init, s_pow=8.0, maxiter=150):
        nn = x_init.shape[0]
        x0f = x_init.ravel().copy()
        sp = float(s_pow)
        def fg(xf):
            X = xf.reshape(nn, d)
            ns = np.sqrt(np.sum(X * X, axis=1))
            ns = np.maximum(ns, 1e-12)
            Y = X / ns[:, None]
            G = Y.dot(Y.T)
            D2 = np.clip(2.0 - 2.0 * G, 1e-9, 4.0)
            np.fill_diagonal(D2, 1.0)
            P = np.power(D2, -sp / 2.0)
            np.fill_diagonal(P, 0.0)
            F = float(np.sum(np.triu(P, 1)))
            W = sp * np.power(D2, -sp / 2.0 - 1.0)
            np.fill_diagonal(W, 0.0)
            WY = W.dot(Y)
            dot = np.sum(WY * Y, axis=1)
            GradX = (WY - dot[:, None] * Y) / ns[:, None]
            # scale to avoid huge grads
            sc = 1.0 / max(1.0, float(np.max(np.abs(GradX))) + 1e-12)
            # keep true direction but limit step size issues: don't rescale F
            return float(F), (GradX * 1.0).ravel()
        try:
            res = minimize(fg, x0f, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-18,
                                    "gtol": 1e-6, "maxfun": int(maxiter * 6),
                                    "maxls": 15})
            Xo = res.x.reshape(nn, d)
        except Exception:
            Xo = x_init.copy()
        return _normed(Xo)

    Iall, Jall = np.triu_indices(n, 1)

    def lp_polish_timed(x_init, seconds):
        x = _normed(x_init.copy())
        cur = _max_cos(x)
        radius = 1e-3
        start = time.time()
        nv = n * d + 1
        while (time.time() - start < seconds and time.time() < DEADLINE - 2.0
               and radius > 1e-13):
            G = x @ x.T
            c_all = G[Iall, Jall]
            m = c_all.max()
            win = max(20.0 * radius, 3e-9)
            sel = c_all > m - win
            I = Iall[sel]; J = Jall[sel]; c = c_all[sel]
            p = len(I)
            if p == 0:
                break
            if p > 2200:
                idx = np.argpartition(-c, 2200)[:2200]
                I = I[idx]; J = J[idx]; c = c[idx]
                p = 2200
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
            sig = 0.04 * (1.0 - st / float(refine_steps)) + 0.001
            prop = bc + sig * rng.randn(d)
            pn = np.linalg.norm(prop)
            if pn < 1e-12:
                continue
            prop /= pn
            md = float((prop @ base.T).max())
            if md < cur:
                bc = prop
                cur = md
        return bc, cur

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

    it = 0
    # initial quick LP on incumbent to set baseline tight
    if time.time() < DEADLINE - 100:
        try:
            Xlp = lp_polish_timed(Xbest, 3.0)
            consider(Xlp)
        except Exception:
            pass

    while time.time() < DEADLINE - 18.0:
        it += 1
        time_left = DEADLINE - time.time()
        if time_left < 20:
            break
        mode = it % 5
        try:
            if mode in (0, 1, 2, 3):
                # ---- contact-guided surgery ----
                _, cnt, _ = contacts(Xbest, thr=5e-7)
                order_low = np.argsort(cnt)
                order_high = np.argsort(-cnt)
                k = int(rng.choice([3, 4, 5, 6, 7, 8]))
                # bias k by iteration
                if mode == 3:
                    k = int(rng.choice([6, 7, 8, 9, 10]))
                drop_set = set()
                # 1-2 lowest
                nl = 1 + int(rng.randint(0, 2))
                nh = 1 + int(rng.randint(0, 2))
                for ii in range(nl):
                    drop_set.add(int(order_low[(ii + int(rng.randint(0, 6))) % n]))
                for ii in range(nh):
                    drop_set.add(int(order_high[(ii + int(rng.randint(0, 6))) % n]))
                while len(drop_set) < k:
                    drop_set.add(int(rng.randint(0, n)))
                drop = np.array(sorted(drop_set), dtype=int)
                keep = np.ones(n, dtype=bool)
                keep[drop] = False
                rem = Xbest[keep].copy()
                # shake remainder to unjam: larger shake for surgery mode
                sig = 0.015 if mode in (0, 1) else 0.035
                if mode == 3:
                    sig = 0.06
                rem = _normed(rem + rng.randn(*rem.shape) * sig)
                # optionally Riesz-relax the remainder briefly (different basin)
                if mode in (1, 3) and rem.shape[0] > 100:
                    try:
                        rem = riesz_refine(rem, s_pow=float(rng.choice([4.0, 6.0, 8.0, 12.0])), maxiter=60)
                    except Exception:
                        pass
                cur = rem
                # regrow one by one in holes
                for _ in range(n - cur.shape[0]):
                    bc, _ = find_hole(cur, nsample=12000, refine_steps=60)
                    # small random tilt of hole centre to diversify
                    bc = bc + rng.randn(d) * 0.005
                    bc /= max(1e-12, np.linalg.norm(bc))
                    cur = np.vstack([cur, bc[None, :]])
                cur = _normed(cur)
                # aggressive hinge to force new topology
                margin = float(rng.choice([0.001, 0.002, 0.003, 0.005]))
                t_targ = Vbest - margin
                cur = lbfgs_hinge(cur, t_targ, maxiter=140)
                # LSE anneal ramp
                for beta, mi in ((60.0, 60), (180.0, 60), (500.0, 60)):
                    if time.time() > DEADLINE - 20:
                        break
                    cur = lse_refine(cur, beta, maxiter=mi)
                v = _max_cos(cur)
                # LP only if promising to save time
                if v < Vbest + 0.004 and time.time() < DEADLINE - 25:
                    cur2 = lp_polish_timed(cur, 1.2)
                    consider(cur2)
                consider(cur)
            else:
                # ---- fresh Riesz basin hop ----
                X0 = _normed(rng.randn(n, d))
                sp = float(rng.choice([4.0, 6.0, 10.0]))
                X0 = riesz_refine(X0, s_pow=sp, maxiter=120)
                t_targ = Vbest - float(rng.choice([0.001, 0.0025]))
                X0 = lbfgs_hinge(X0, t_targ, maxiter=120)
                X0 = lse_refine(X0, 200.0, maxiter=80)
                consider(X0)
        except Exception:
            continue

    # final promotion of best
    try:
        Xbest = lse_refine(Xbest, 800.0, maxiter=150)
        consider(Xbest)
    except Exception:
        pass
    try:
        if time.time() < DEADLINE - 3.0:
            left = max(1.0, DEADLINE - time.time() - 1.0)
            Xbest = lp_polish_timed(Xbest, min(left, 8.0))
    except Exception:
        pass
    Xbest = _normed(Xbest)
    return Xbest
