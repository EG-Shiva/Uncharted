import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def _peak_shave(X, iters=500):
    X = X.copy()
    n = X.shape[0]
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        m = float(G[i, j])
        G[i, j] = -2.0
        G[j, i] = -2.0
        m2 = float(G.max())
        d = m - m2
        if d < 1e-12:
            break
        eta = min(d * 0.4, 1e-3)
        ui = X[j] - m * X[i]
        vj = X[i] - m * X[j]
        ni = float(np.linalg.norm(ui))
        nj = float(np.linalg.norm(vj))
        if ni < 1e-12 or nj < 1e-12:
            break
        ui /= ni
        vj /= nj
        improved = False
        for scale in (1.0, 0.25, 0.0625):
            e = eta * scale
            Xn = X.copy()
            Xn[i] -= e * ui
            Xn[j] -= e * vj
            Xn[i] /= max(np.linalg.norm(Xn[i]), 1e-12)
            Xn[j] /= max(np.linalg.norm(Xn[j]), 1e-12)
            G2 = Xn @ Xn.T
            np.fill_diagonal(G2, -2.0)
            mn = float(G2.max())
            if mn < m:
                X = Xn
                improved = True
                break
        if not improved:
            break
    return X

def solve():
    np.random.seed(0)
    rng = np.random.RandomState(12345)
    t0 = time.time()
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)

    # load
    x_best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm_rows(v)
                if np.all(np.isfinite(v)):
                    c = _max_cos(v)
                    if c < best_val:
                        best_val = c
                        x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = rng.randn(n, d)
        x_best = _norm_rows(x_best)
        best_val = _max_cos(x_best)

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
            W = np.zeros((n, n), dtype=float)
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            Y = _norm_rows(Y)
            return Y
        except Exception:
            return x0.copy()

    def lp_polish(x, deadline, r_init=1e-3):
        x = _norm_rows(x.copy())
        try:
            best = _max_cos(x)
        except Exception:
            return x
        bx = x.copy()
        radius = r_init
        while time.time() < deadline - 0.5 and time.time() - t0 < 176.0:
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(18.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 4000:
                k = 4000
                part = np.argpartition(-csel, k - 1)[:k]
                I, J, csel = I[part], J[part], csel[part]
                p = k
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                                   shape=(n, n * d + 1)).tocsr()
                obj = np.zeros(n * d + 1); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm_rows(y)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                if v < _max_cos(bx):
                    bx = y.copy()
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return bx

    def void_replace(base, w_idx, nsamp=2500):
        # find largest void ignoring w_idx
        cand = rng.randn(nsamp, d)
        cand = _norm_rows(cand)
        S = cand @ base.T  # nsamp x n
        # exclude w_idx by setting to -inf effect: mask
        S[:, w_idx] = -2.0
        sc = S.max(axis=1)
        k = int(np.argmin(sc))
        return cand[k]

    cur = x_best.copy()
    cur_val = best_val
    # immediate shave
    try:
        Xs = _peak_shave(cur, 2000)
        vs = _max_cos(Xs)
        if vs < best_val:
            x_best, best_val = Xs.copy(), vs
            cur, cur_val = Xs.copy(), vs
    except Exception:
        pass

    # escalating L-BFGS from current
    for beta, mi in [(400, 250), (1200, 250), (3000, 200)]:
        if time.time() - t0 > 60:
            break
        try:
            Y = run_lbfgs(cur, beta, mi)
            Y = _peak_shave(Y, 400)
            v = _max_cos(Y)
            if v < best_val:
                x_best, best_val = Y.copy(), v
            if v < cur_val:
                cur, cur_val = Y.copy(), v
            else:
                # still move cur to Y if close to diversify
                cur = Y.copy(); cur_val = v
        except Exception:
            continue

    temp = 0.0008
    trial = 0
    sigmas = [0.006, 0.015, 0.03, 0.003, 0.01]
    while time.time() - t0 < 132:
        trial += 1
        sigma = sigmas[trial % len(sigmas)]
        mode = trial % 5
        # choose base: mostly cur, sometimes best
        if trial % 4 == 0:
            base = x_best
            base_val = best_val
        else:
            base = cur
            base_val = cur_val
        P = base.copy()
        try:
            if mode == 0:
                P = base + sigma * rng.randn(n, d)
                P = _norm_rows(P)
            elif mode == 1:
                # cluster around worst point
                ppm = _per_point_max(base)
                i = int(np.argmax(ppm))
                dots = base @ base[i]
                # nearest by dot (largest)
                nn = np.argsort(-dots)[:14]
                P[nn] = P[nn] + sigma * 2.5 * rng.randn(len(nn), d)
                # tiny global jitter
                P = P + (sigma * 0.2) * rng.randn(n, d)
                P = _norm_rows(P)
            elif mode == 2:
                # void re-insertion of 1-2 worst points
                ppm = _per_point_max(base)
                order = np.argsort(-ppm)
                nw = 1 if (trial % 2 == 0) else 2
                for k in range(nw):
                    w = int(order[k])
                    P[w] = void_replace(base, w, nsamp=2500)
                # jitter others slightly
                P = P + (sigma * 0.3) * rng.randn(n, d)
                P = _norm_rows(P)
            elif mode == 3:
                # directed pair split + jitter: push top pair apart hard
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                flat = int(np.argmax(G))
                i = flat // n; j = flat % n
                m = float(G[i, j])
                ui = base[j] - m * base[i]
                vj = base[i] - m * base[j]
                ui /= max(np.linalg.norm(ui), 1e-12)
                vj /= max(np.linalg.norm(vj), 1e-12)
                step = 0.02 + 0.01 * rng.rand()
                P[i] = base[i] - step * ui
                P[j] = base[j] - step * vj
                P = P + (sigma * 0.5) * rng.randn(n, d)
                P = _norm_rows(P)
            else:
                # random subset
                k = 12 + int(rng.randint(0, 10))
                idx = rng.choice(n, size=k, replace=False)
                P[idx] = P[idx] + sigma * 2.0 * rng.randn(k, d)
                P = _norm_rows(P)
        except Exception:
            continue
        beta = float(rng.choice([600.0, 900.0, 1500.0, 2500.0]))
        try:
            Q = run_lbfgs(P, beta, 130)
            Q = _peak_shave(Q, 300)
            vq = _max_cos(Q)
        except Exception:
            continue
        if vq < best_val:
            x_best, best_val = Q.copy(), vq
            cur, cur_val = Q.copy(), vq
        elif vq < cur_val:
            cur, cur_val = Q.copy(), vq
        else:
            # annealing acceptance to wander
            if rng.rand() < np.exp(-(vq - cur_val) / max(temp, 1e-9)):
                cur, cur_val = Q.copy(), vq
        temp *= 0.997
        if temp < 1e-5:
            temp = 1e-5
        if time.time() - t0 > 131:
            break

    # final squeeze
    remain_deadline = t0 + 177.0
    try:
        Xp = lp_polish(x_best, min(remain_deadline - 1.0, time.time() + 30.0), r_init=1e-3)
        vp = _max_cos(Xp)
        if vp < best_val:
            x_best, best_val = Xp, vp
        else:
            # try smaller radius polish on cur if cur different
            if cur_val < best_val + 1e-6:
                Xp2 = lp_polish(cur, min(remain_deadline - 0.5, time.time() + 10.0), r_init=5e-4)
                vp2 = _max_cos(Xp2)
                if vp2 < best_val:
                    x_best, best_val = Xp2, vp2
    except Exception:
        pass
    try:
        x_best = _peak_shave(x_best, 800)
    except Exception:
        pass
    x_best = _norm_rows(x_best)
    return x_best
