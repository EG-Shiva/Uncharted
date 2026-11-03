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
    return float(np.max(G))

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def peak_shave(X, iters=800):
    X = X.copy()
    n = X.shape[0]
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        m = float(G.flat[flat])
        G.flat[flat] = -2.0
        # symmetric entry
        G[j, i] = -2.0
        m2 = float(np.max(G))
        d = m - m2
        if d < 1e-12:
            break
        eta = min(d * 0.4, 8e-4)
        ui = X[j] - m * X[i]
        vj = X[i] - m * X[j]
        ni = float(np.linalg.norm(ui))
        nj = float(np.linalg.norm(vj))
        if ni < 1e-12 or nj < 1e-12:
            break
        ui /= ni
        vj /= nj
        improved = False
        for scale in (1.0, 0.3, 0.08):
            e = eta * scale
            Xn = X.copy()
            Xn[i] -= e * ui
            Xn[j] -= e * vj
            Xn[i] /= max(np.linalg.norm(Xn[i]), 1e-12)
            Xn[j] /= max(np.linalg.norm(Xn[j]), 1e-12)
            G2 = Xn @ Xn.T
            np.fill_diagonal(G2, -2.0)
            mn = float(np.max(G2))
            if mn < m:
                X = Xn
                improved = True
                break
        if not improved:
            break
    return X

def adam_smooth(X0, schedule):
    X = _norm_rows(X0.copy())
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    t = 0
    b1, b2 = 0.9, 0.999
    eps = 1e-8
    for alpha, steps, lr in schedule:
        for _ in range(steps):
            t += 1
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mmax = float(np.max(G))
            W = np.exp(alpha * (G - mmax))
            np.fill_diagonal(W, 0.0)
            S = float(W.sum() * 0.5)
            if not np.isfinite(S) or S <= 0:
                break
            grad = (W @ X) / S
            dot = (grad * X).sum(axis=1, keepdims=True)
            gt = grad - dot * X
            m1 = b1 * m1 + (1 - b1) * gt
            m2 = b2 * m2 + (1 - b2) * (gt * gt)
            c1 = 1.0 - b1 ** t
            c2 = 1.0 - b2 ** t
            step = lr * (m1 / c1) / (np.sqrt(m2 / c2) + eps)
            sn = np.linalg.norm(step, axis=1, keepdims=True)
            cap = 0.05
            fac = np.minimum(1.0, cap / np.maximum(sn, 1e-12))
            X = _norm_rows(X - step * fac)
    return X

def void_refine(v, others, iters=150, lr0=0.08):
    v = v / max(np.linalg.norm(v), 1e-12)
    lr = lr0
    best = float(np.max(others @ v))
    bestv = v.copy()
    for _ in range(iters):
        dots = others @ v
        m = float(np.max(dots))
        tau = 0.0025
        w = np.exp((dots - m) / tau)
        w[dots < m - 0.015] = 0.0
        g = w @ others
        gt = g - float(np.dot(g, v)) * v
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-14 or not np.isfinite(nrm):
            break
        vn = v - (lr / nrm) * gt
        vn /= max(np.linalg.norm(vn), 1e-12)
        mn = float(np.max(others @ vn))
        if mn < m:
            v = vn
            if mn < best:
                best = mn
                bestv = vn.copy()
        else:
            lr *= 0.7
            if lr < 1e-5:
                break
    return bestv, best

def find_void(others, rng, nsamp=15000):
    C = rng.randn(nsamp, 6)
    C = _norm_rows(C)
    D = C @ others.T
    vals = D.max(axis=1)
    k = int(np.argmin(vals))
    v0 = C[k]
    return void_refine(v0, others, iters=120)

def lp_polish(x, seconds, t0, budget=172.0):
    x = _norm_rows(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    iu0 = np.triu_indices(n, 1)
    radius = 1.2e-3
    start = time.time()
    while time.time() - start < seconds and (time.time() - t0) < budget:
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu0]
        m = float(c.max())
        thr = m - max(20 * radius, 4e-7)
        sel = c > thr
        if int(sel.sum()) < 1:
            break
        I = iu0[0][sel]; J = iu0[1][sel]; cs = c[sel]
        p = len(I)
        if p > 4000:
            part = np.argpartition(-cs, 4000 - 1)[:4000]
            I, J, cs = I[part], J[part], cs[part]
            p = 4000
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate(
                [I[:, None] * d + np.arange(d),
                 J[:, None] * d + np.arange(d),
                 np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm_rows(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def run_lbfgs(X0, beta, maxiter, iu0, iu1):
    n, d = X0.shape
    y0 = X0.ravel().copy()
    def fun(y):
        Y = y.reshape(n, d)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X = Y / rn
        G = X @ X.T
        g = G[iu0, iu1]
        m = float(np.max(g))
        e = np.exp(beta * (g - m))
        S = float(np.sum(e))
        loss = m + np.log(S) / beta
        w = e / S
        W = np.zeros((n, n))
        W[iu0, iu1] = w
        W = W + W.T
        GX = W @ X
        dot = (GX * X).sum(axis=1, keepdims=True)
        GY = (GX - dot * X) / rn
        return float(loss), GY.ravel()
    try:
        res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8, "maxfun": 20000})
        Y = res.x.reshape(n, d)
        Y = _norm_rows(Y)
        return Y
    except Exception:
        return X0.copy()

def solve():
    np.random.seed(0)
    t0 = time.time()
    BUDGET = 172.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    rng = np.random.RandomState(7)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = _norm_rows(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm_rows(rng.randn(n, d)))
    best = None; bestv = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < bestv:
            bestv = v; best = c.copy()
    # initial shave + smooth
    try:
        best = peak_shave(best, 2000)
        bestv = _max_cos(best)
    except Exception:
        pass
    try:
        Xa = adam_smooth(best, [(120, 400, 0.02), (350, 400, 0.008), (700, 400, 0.004)])
        Xa = peak_shave(Xa, 400)
        va = _max_cos(Xa)
        if va < bestv:
            best, bestv = Xa, va
    except Exception:
        pass

    trial = 0
    sigmas = [0.006, 0.015, 0.03, 0.05, 0.01, 0.08]
    HOP_END = t0 + 128.0
    while time.time() < HOP_END:
        trial += 1
        mode = trial % 4
        Xp = None
        try:
            if mode in (0, 3):
                sigma = sigmas[(trial // 4) % len(sigmas)]
                Xp = _norm_rows(best + sigma * rng.randn(n, d))
            elif mode == 1:
                # kill worst 1-2 points, reinsert at voids
                rm, G = _row_max(best)
                order = np.argsort(-rm)
                nk = 1 if (trial % 8 < 6) else 2
                kill = order[:nk]
                keep = np.delete(np.arange(n), kill)
                others = best[keep]
                Xp = best.copy()
                for k in kill:
                    # exclude k from others for void search
                    mask = np.ones(n, dtype=bool); mask[k] = False
                    oth = Xp[mask]
                    vnew, _ = find_void(oth, rng, nsamp=12000)
                    Xp[k] = vnew
            else:
                # split top pairs: push top-3 pairs apart a bit then shake
                _, G = _row_max(best)
                G2 = G.copy()
                Xp = best.copy()
                for _ in range(3):
                    flat = int(np.argmax(G2))
                    i = flat // n; j = flat % n
                    if i == j:
                        break
                    G2[i, j] = -2.0; G2[j, i] = -2.0
                    m = float(best[i] @ best[j])
                    ui = best[j] - m * best[i]
                    vj = best[i] - m * best[j]
                    ni = float(np.linalg.norm(ui)); nj = float(np.linalg.norm(vj))
                    if ni < 1e-12 or nj < 1e-12:
                        continue
                    ui /= ni; vj /= nj
                    e = 0.004 + 0.004 * rng.rand()
                    Xp[i] = Xp[i] - e * ui
                    Xp[j] = Xp[j] - e * vj
                Xp = _norm_rows(Xp + 0.006 * rng.randn(n, d))
            # relax
            if Xp is None:
                continue
            Q = adam_smooth(Xp, [(200, 300, 0.012), (550, 300, 0.005)])
            Q = peak_shave(Q, 400)
            vq = _max_cos(Q)
            # sharpen promising ones with L-BFGS
            if vq < bestv + 2e-4 and (time.time() - t0) < 110:
                Q2 = run_lbfgs(Q, 600.0, 60, iu0, iu1)
                Q2 = peak_shave(Q2, 200)
                v2 = _max_cos(Q2)
                if v2 < vq:
                    Q, vq = Q2, v2
            if vq < bestv:
                best, bestv = Q, vq
        except Exception:
            continue
        if time.time() - t0 > 126:
            break

    # final sharpen
    try:
        if time.time() - t0 < BUDGET - 30:
            Qb = run_lbfgs(best, 400.0, 80, iu0, iu1)
            Qb = peak_shave(Qb, 300)
            vb = _max_cos(Qb)
            if vb < bestv:
                best, bestv = Qb, vb
            Qc = run_lbfgs(best, 900.0, 80, iu0, iu1)
            Qc = peak_shave(Qc, 300)
            vc = _max_cos(Qc)
            if vc < bestv:
                best, bestv = Qc, vc
    except Exception:
        pass
    remain = BUDGET - (time.time() - t0)
    if remain > 8:
        try:
            Xp = lp_polish(best, min(remain - 2.0, 22.0), t0, BUDGET)
            vp = _max_cos(Xp)
            if vp < bestv:
                best, bestv = Xp, vp
        except Exception:
            pass
    try:
        best = peak_shave(best, 800)
    except Exception:
        pass
    best = _norm_rows(best)
    return best
