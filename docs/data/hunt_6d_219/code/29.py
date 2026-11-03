import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    BUDGET = 168.0
    HOP_END = t0 + 112.0
    rng = np.random.RandomState(777001)
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

    def _norm(X):
        q = np.linalg.norm(X, axis=1, keepdims=True)
        q = np.maximum(q, 1e-12)
        return X / q

    def max_cos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = _norm(v)
                c = max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm(rng.randn(n, d))
        best_val = max_cos(x_best)

    def run_lbfgs(x0, beta, maxiter):
        y0 = x0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(g.max())
            e = np.exp(beta * (g - m))
            S = float(e.sum())
            loss = m + np.log(S + 1e-300) / beta
            w = e / (S + 1e-300)
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
        Y = res.x.reshape(n, d)
        return _norm(Y)

    def best_hole_vec(others, ncand=3000):
        cand = _norm(rng.randn(ncand, d))
        D = cand @ others.T
        md = D.max(axis=1)
        return cand[int(np.argmin(md))]

    # quick initial touch
    try:
        if time.time() - t0 < BUDGET - 110:
            xc = run_lbfgs(x_best, 300.0, maxiter=40)
            vc = max_cos(xc)
            if vc < best_val:
                x_best, best_val = xc, vc
    except Exception:
        pass

    x_cur = x_best.copy()
    cur_val = best_val
    stall = 0
    trial = 0
    b1_pool = [120.0, 220.0, 350.0]
    b2_pool = [700.0, 1200.0, 2200.0]

    def random_cap_shift(X, k_target, delta):
        p = _norm(rng.randn(1, d))[0]
        dots = X @ p
        # k_target largest define cap
        thr = np.sort(dots)[n - k_target]
        idx = np.where(dots >= thr)[0]
        # random orthogonal dir
        v = rng.randn(d)
        v = v - float(v @ p) * p
        vn = float(np.linalg.norm(v))
        if vn < 1e-9:
            return X
        v = v / vn
        cd, sd = float(np.cos(delta)), float(np.sin(delta))
        Xn = X.copy()
        for i in idx:
            x = X[i]
            a = float(x @ p)
            b = float(x @ v)
            rest = x - a * p - b * v
            ap = a * cd - b * sd
            bp = a * sd + b * cd
            # rotate back? Actually rotate whole space in p-v plane
            # this moves cap coherently
            y = rest + ap * p + bp * v
            Xn[i] = y
        return _norm(Xn)

    while time.time() < HOP_END and time.time() - t0 < BUDGET - 55:
        trial += 1
        if stall >= 14:
            x_cur = x_best.copy()
            cur_val = best_val
            stall = 0
        if trial > 220:
            break
        # per-point worst for guidance
        try:
            G0 = x_cur @ x_cur.T
            np.fill_diagonal(G0, -2.0)
            pw = G0.max(axis=1)
            order = np.argsort(-pw)
        except Exception:
            order = np.arange(n)
        mode = int(rng.randint(0, 100))
        x_init = None
        try:
            if mode < 20:
                # block resample uniform: strong topology break
                k = int(rng.choice(np.array([6, 12, 20, 35, 55])))
                idx = rng.choice(n, size=k, replace=False)
                x_init = x_cur.copy()
                x_init[idx] = _norm(rng.randn(k, d))
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                x_init = _norm(x_init + 0.004 * P)
            elif mode < 40:
                # kill worst cluster and best-hole refill sequentially
                k = int(rng.choice(np.array([2, 4, 6, 9])))
                # worst + random among top-30
                cand_pool = order[:30]
                victims = list(order[:2])
                extra = rng.choice(cand_pool, size=max(0, k - len(victims)), replace=False)
                for e in extra:
                    if int(e) not in victims:
                        victims.append(int(e))
                victims = victims[:k]
                tmp = x_cur.copy()
                # delete and refill one by one with far voids
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[victims] = False
                remaining = tmp[keep_mask]
                newpts = []
                for _ in range(k):
                    bh = best_hole_vec(np.vstack([remaining] + newpts) if newpts else remaining,
                                       ncand=3500 if rng.rand() < 0.5 else 1500)
                    # local jitter of hole
                    bh = bh + 0.01 * rng.randn(d)
                    bh = bh / np.linalg.norm(bh)
                    newpts.append(bh.reshape(1, d))
                for ii, vv in enumerate(victims):
                    tmp[vv] = newpts[ii][0]
                x_init = tmp
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                step = np.full((n, 1), 0.003)
                step[victims] = 0.008
                x_init = _norm(x_init + step * P)
            elif mode < 60:
                # coherent cap rotation
                k_target = int(rng.choice(np.array([10, 18, 30, 45, 65])))
                delta = float(rng.choice(np.array([0.05, 0.09, 0.15, 0.22, -0.07, -0.13])))
                # randomize sign
                if rng.rand() < 0.5:
                    delta = -delta
                x_init = random_cap_shift(x_cur, k_target, delta)
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                x_init = _norm(x_init + 0.004 * P)
            elif mode < 80:
                # worst cluster large shake
                c0 = int(order[int(rng.randint(0, 5))])
                Gc = x_cur @ x_cur.T
                np.fill_diagonal(Gc, -2.0)
                nn_size = int(rng.choice(np.array([10, 16, 24, 36])))
                nn = np.argsort(-Gc[c0])[:nn_size]
                sig = float(rng.choice(np.array([0.10, 0.16, 0.25, 0.38])))
                P = rng.randn(n, d)
                P -= (P * x_cur).sum(axis=1, keepdims=True) * x_cur
                Pn = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
                P = P / Pn
                x_init = x_cur.copy()
                step = np.full((n, 1), 0.003)
                step[nn] = sig
                step[c0] = sig * 1.2
                x_init = _norm(x_init + step * P)
            elif mode < 90:
                # from scratch diverse basin
                x_init = _norm(rng.randn(n, d))
                # quick Coulomb-ish pre-relax via low beta short run is done in LBFGS anyway
            else:
                # moderate global jitter + max-pair push
                Gc = x_cur @ x_cur.T
                np.fill_diagonal(Gc, -2.0)
                flat = int(np.argmax(Gc))
                a, b = flat // n, flat % n
                x_init = x_cur.copy()
                s = float(rng.choice(np.array([0.03, 0.06, 0.10])))
                for tgt, oth in ((a, b), (b, a)):
                    dot = float(np.dot(x_init[tgt], x_init[oth]))
                    gd = -(x_init[oth] - dot * x_init[tgt])
                    gd -= float(np.dot(gd, x_init[tgt])) * x_init[tgt]
                    gd /= (np.linalg.norm(gd) + 1e-12)
                    x_init[tgt] = x_init[tgt] + s * gd
                P = rng.randn(n, d)
                P -= (P * x_init).sum(axis=1, keepdims=True) * x_init
                sig = float(rng.choice(np.array([0.01, 0.02, 0.035])))
                x_init = _norm(x_init + sig * P)
        except Exception:
            continue
        if x_init is None:
            continue
        b1 = float(b1_pool[int(rng.randint(0, 3))])
        b2 = float(b2_pool[int(rng.randint(0, 3))])
        try:
            xo = run_lbfgs(x_init, b1, maxiter=40)
            vmid = max_cos(xo)
            # screen very bad
            if vmid > cur_val + 0.008 and vmid > best_val + 0.004:
                # skip second stage most of time
                if rng.rand() < 0.85:
                    stall += 1
                    continue
            xo2 = run_lbfgs(xo, b2, maxiter=35)
        except Exception:
            continue
        v = max_cos(xo2)
        xo = xo2
        if v < best_val - 1e-12:
            x_best, best_val = xo, v
            x_cur, cur_val = xo, v
            stall = 0
        else:
            tol = 0.0008 if stall < 8 else 0.0016
            if v < cur_val + tol and rng.rand() < 0.55:
                x_cur, cur_val = xo, v
                stall += 1
            else:
                if v < cur_val:
                    x_cur, cur_val = xo, v
                stall += 1

    # high-beta sharpening of best
    for hb, it in [(900.0, 60), (1800.0, 60), (3500.0, 55), (7000.0, 45)]:
        if time.time() - t0 > BUDGET - 30:
            break
        try:
            xo = run_lbfgs(x_best, hb, maxiter=it)
            v = max_cos(xo)
            if v < best_val:
                x_best, best_val = xo, v
        except Exception:
            continue

    # trust-region LP squeeze (few steps)
    try:
        remain = BUDGET - (time.time() - t0)
        if remain > 9:
            x = _norm(x_best)
            best = max_cos(x)
            radius = 6e-4
            deadline = time.time() + min(remain - 1.5, 14.0)
            E = np.eye(d)
            while time.time() < deadline and radius > 5e-12:
                G = x @ x.T
                cvec = G[iu0, iu1]
                m = float(cvec.max())
                K = 2000
                if len(cvec) > K:
                    part = np.argpartition(-cvec, K - 1)[:K]
                    Ia = iu0[part]; Ja = iu1[part]; Cab = cvec[part]
                else:
                    Ia, Ja, Cab = iu0, iu1, cvec
                # keep only near-max
                thr = m - max(25 * radius, 1e-7)
                msk = Cab > thr
                if int(msk.sum()) < 200:
                    # take top 600
                    od = np.argsort(-Cab)[:600]
                    Ia, Ja, Cab = Ia[od], Ja[od], Cab[od]
                else:
                    Ia, Ja, Cab = Ia[msk], Ja[msk], Cab[msk]
                p = len(Ia)
                if p == 0:
                    break
                # Build LP: vars [dx (n*d), t]
                Nv = n * d + 1
                rows = np.repeat(np.arange(p), 2 * d + 1)
                c1 = (Ia[:, None] * d + np.arange(d)).ravel()
                c2 = (Ja[:, None] * d + np.arange(d)).ravel()
                c3 = np.full(p, Nv - 1)
                cols = np.concatenate([c1, c2, c3])
                # careful ordering: for each constraint k, coeffs are x[Ja], x[Ia], -1
                vals_list = np.concatenate([x[Ja].ravel(), x[Ia].ravel(), -np.ones(p)])
                # reorder to match rows repetition pattern: we used repeat then concat blocks incorrectly
                # fix: interleave per constraint
                vals = np.empty(p * (2 * d + 1))
                vals[0::(2 * d + 1)] = 0  # placeholder
                # Build correctly per constraint
                # Simpler: build row by row via coo lists
                R = []
                C = []
                V = []
                for k in range(p):
                    i = int(Ia[k]); j = int(Ja[k])
                    for a in range(d):
                        R.append(k); C.append(i * d + a); V.append(float(x[j, a]))
                    for a in range(d):
                        R.append(k); C.append(j * d + a); V.append(float(x[i, a]))
                    R.append(k); C.append(Nv - 1); V.append(-1.0)
                A_ub = coo_matrix((V, (R, C)), shape=(p, Nv)).tocsr()
                b_ub = -Cab
                c_obj = np.zeros(Nv); c_obj[-1] = 1.0
                bounds = [(-radius, radius)] * (n * d) + [(m - 0.02, m)]
                # tangent equalities: x_i . dx_i = 0
                Ae_R = []; Ae_C = []; Ae_V = []
                for i in range(n):
                    for a in range(d):
                        Ae_R.append(i); Ae_C.append(i * d + a); Ae_V.append(float(x[i, a]))
                A_eq = coo_matrix((Ae_V, (Ae_R, Ae_C)), shape=(n, Nv)).tocsr()
                b_eq = np.zeros(n)
                try:
                    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                                  bounds=bounds, method="highs",
                                  options={"time_limit": 4.0})
                except Exception:
                    break
                if not res.success:
                    radius *= 0.4
                    continue
                dx = res.x[:n * d].reshape(n, d)
                tval = float(res.x[-1])
                # project to tangent
                dx -= (dx * x).sum(axis=1, keepdims=True) * x
                # limit
                mx = float(np.max(np.abs(dx)))
                if mx > radius * 1.01:
                    dx *= (radius / max(mx, 1e-16))
                xn = _norm(x + dx)
                nm = max_cos(xn)
                if nm < best - 1e-12 and nm <= m + 1e-12:
                    x, best = xn, nm
                    if best < best_val:
                        x_best, best_val = x.copy(), best
                    radius *= 1.2
                else:
                    radius *= 0.35
                if tval >= m - 1e-12:
                    # no predicted progress
                    radius *= 0.6
    except Exception:
        pass

    # final greedy true-max micro moves
    try:
        remain = BUDGET - (time.time() - t0)
        if remain > 6:
            X = x_best.copy()
            bv = best_val
            endt = time.time() + min(remain - 1.0, 8.0)
            it = 0
            while time.time() < endt and it < 3000:
                it += 1
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                flat = int(np.argmax(G))
                a, b = flat // n, flat % n
                tgt = a if rng.rand() < 0.5 else b
                oth = b if tgt == a else a
                dot = float(np.dot(X[tgt], X[oth]))
                grad = -(X[oth] - dot * X[tgt])
                grad -= float(np.dot(grad, X[tgt])) * X[tgt]
                gn = float(np.linalg.norm(grad))
                if gn < 1e-12:
                    continue
                grad /= gn
                rr = rng.randn(d)
                rr -= float(np.dot(rr, X[tgt])) * X[tgt] + float(np.dot(rr, grad)) * grad
                rn2 = float(np.linalg.norm(rr))
                if rn2 > 1e-12:
                    rr /= rn2
                else:
                    rr = np.zeros(d)
                step = float(rng.choice(np.array([0.0004, 0.0008, 0.0002, 0.0012])))
                tdir = grad * 0.9 + rr * 0.3
                tdir /= (np.linalg.norm(tdir) + 1e-12)
                Xt = X.copy()
                Xt[tgt] = Xt[tgt] + step * tdir
                Xt[tgt] /= (np.linalg.norm(Xt[tgt]) + 1e-12)
                G2 = Xt @ Xt.T
                np.fill_diagonal(G2, -2.0)
                nv = float(np.max(G2))
                if nv < bv:
                    X, bv = Xt, nv
            if bv < best_val:
                x_best, best_val = X, bv
    except Exception:
        pass

    x_best = _norm(x_best)
    return x_best
