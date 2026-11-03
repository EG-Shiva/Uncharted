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

def _contact_deg(X, m, tol=2e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.sum(G > m - tol, axis=1), G

def peak_shave(X, iters=400):
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
        ui /= ni; vj /= nj
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

def run_lbfgs_generic(X0, beta, maxiter):
    X0 = _norm_rows(X0.copy())
    n, d = X0.shape
    iu0, iu1 = np.triu_indices(n, 1)
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

def lp_polish_generic(x, seconds, t0, budget):
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

def void_refine(v, others, iters=120, lr0=0.08):
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

def find_void(others, rng, nsamp=12000):
    C = _norm_rows(rng.randn(nsamp, 6))
    D = C @ others.T
    vals = D.max(axis=1)
    k = int(np.argmin(vals))
    v0 = C[k]
    return void_refine(v0, others, iters=120)

def solve():
    np.random.seed(0)
    t0 = time.time()
    BUDGET = 172.0
    n, d = 219, 6
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
    base219 = best.copy()
    basev = bestv

    def down_up(start, Ntarget, rng):
        # Ntarget e.g. 200: delete worst down to Ntarget
        cur_n = start.shape[0]
        rm, G = _row_max(start)
        m = float(np.max(G))
        deg, _ = _contact_deg(start, m, tol=1e-9)
        order = np.argsort(-rm)
        K = cur_n - Ntarget
        # ensure rattlers included
        ratt = np.where(deg == 0)[0]
        kill_set = set(order[:K].tolist())
        for r in ratt:
            if len(kill_set) >= K:
                break
            kill_set.add(int(r))
        # if still short (no rattler), just take top K
        while len(kill_set) < K:
            for idx in order:
                if int(idx) not in kill_set:
                    kill_set.add(int(idx))
                    break
        kill = np.array(sorted(kill_set), dtype=int)
        keep = np.delete(np.arange(cur_n), kill)
        Xc = start[keep].copy()
        # freely optimise reduced code until jammed
        try:
            Xc = adam_smooth(Xc, [(150, 250, 0.02), (400, 250, 0.01), (800, 150, 0.005)])
        except Exception:
            pass
        try:
            Xc = run_lbfgs_generic(Xc, 500.0, 60)
            Xc = run_lbfgs_generic(Xc, 1200.0, 60)
            Xc = peak_shave(Xc, 200)
        except Exception:
            pass
        # sequential regrow to 219
        need = 219 - Xc.shape[0]
        for k in range(need):
            if time.time() - t0 > BUDGET - 25:
                break
            try:
                vnew, _ = find_void(Xc, rng, nsamp=10000)
            except Exception:
                vnew = _norm_rows(rng.randn(1, 6))[0]
            Xc = np.vstack([Xc, vnew.reshape(1, -1)])
            Xc = _norm_rows(Xc)
            try:
                # annealed beta soft-max global re-opt after every addition
                beta = 250.0 + 30.0 * k
                Xc = adam_smooth(Xc, [(beta, 120, 0.012)])
            except Exception:
                pass
            try:
                Xc = run_lbfgs_generic(Xc, beta, 35)
                Xc = peak_shave(Xc, 120)
            except Exception:
                pass
            if (k % 5 == 4 or k == need - 1):
                try:
                    Xc = run_lbfgs_generic(Xc, 800.0, 45)
                    Xc = peak_shave(Xc, 150)
                except Exception:
                    pass
        return Xc

    def up_down(start, Nover, rng):
        Xc = start.copy()
        # add points at voids to overfill
        for _ in range(Nover - 219):
            try:
                vnew, _ = find_void(Xc, rng, nsamp=10000)
            except Exception:
                vnew = _norm_rows(rng.randn(1, 6))[0]
            Xc = np.vstack([Xc, vnew.reshape(1, -1)])
            Xc = _norm_rows(Xc)
        # optimise overfilled (harder) code
        try:
            Xc = adam_smooth(Xc, [(200, 250, 0.015), (550, 250, 0.007)])
        except Exception:
            pass
        try:
            Xc = run_lbfgs_generic(Xc, 500.0, 60)
            Xc = run_lbfgs_generic(Xc, 1000.0, 60)
            Xc = peak_shave(Xc, 200)
        except Exception:
            pass
        # greedy prune back to 219: remove points whose removal minimises remaining max
        while Xc.shape[0] > 219:
            if time.time() - t0 > BUDGET - 25:
                break
            try:
                G = Xc @ Xc.T
                np.fill_diagonal(G, -2.0)
                rm = np.max(G, axis=1)
                cand = np.argsort(-rm)[:10]
                best_rem = -1
                best_rem_val = 1e9
                for ci in cand:
                    mask = np.ones(Xc.shape[0], dtype=bool)
                    mask[int(ci)] = False
                    Gs = G[np.ix_(mask, mask)]
                    rv = float(np.max(Gs))
                    if rv < best_rem_val:
                        best_rem_val = rv
                        best_rem = int(ci)
                keep = np.delete(np.arange(Xc.shape[0]), best_rem)
                Xc = Xc[keep].copy()
            except Exception:
                break
        # re-optimise pruned 219 with annealed beta + polish
        try:
            Xc = adam_smooth(Xc, [(300, 150, 0.01), (700, 150, 0.005)])
        except Exception:
            pass
        try:
            Xc = run_lbfgs_generic(Xc, 700.0, 60)
            Xc = peak_shave(Xc, 250)
        except Exception:
            pass
        return Xc

    HOP_END = t0 + 128.0
    trial = 0
    Nunders = [200, 203, 206]
    Novers = [227, 225, 230, 226]
    while time.time() < HOP_END:
        trial += 1
        if time.time() - t0 > BUDGET - 30:
            break
        try:
            if trial % 2 == 1:
                Nt = Nunders[(trial // 2) % len(Nunders)]
                Xnew = down_up(best, Nt, rng)
            else:
                No = Novers[(trial // 2) % len(Novers)]
                # small shake of start to diversify overfill basin
                Xs = _norm_rows(best + 0.004 * rng.randn(n, d))
                Xnew = up_down(Xs, No, rng)
            if Xnew.shape != (219, 6):
                continue
            Xnew = _norm_rows(Xnew)
            vn = _max_cos(Xnew)
            if vn < bestv:
                best, bestv = Xnew.copy(), vn
            elif vn < bestv + 5e-4 and (time.time() - t0) < 100:
                # sharpen near-misses with extra L-BFGS
                try:
                    X2 = run_lbfgs_generic(Xnew, 900.0, 50)
                    X2 = peak_shave(X2, 200)
                    v2 = _max_cos(X2)
                    if v2 < bestv:
                        best, bestv = X2.copy(), v2
                except Exception:
                    pass
        except Exception:
            continue

    # final sharpen of global best
    try:
        if time.time() - t0 < BUDGET - 28:
            Qb = run_lbfgs_generic(best, 500.0, 70)
            Qb = peak_shave(Qb, 300)
            vb = _max_cos(Qb)
            if vb < bestv:
                best, bestv = Qb, vb
            Qc = run_lbfgs_generic(best, 1000.0, 70)
            Qc = peak_shave(Qc, 300)
            vc = _max_cos(Qc)
            if vc < bestv:
                best, bestv = Qc, vc
    except Exception:
        pass
    remain = BUDGET - (time.time() - t0)
    if remain > 8:
        try:
            Xp = lp_polish_generic(best, min(remain - 2.0, 20.0), t0, BUDGET)
            vp = _max_cos(Xp)
            if vp < bestv:
                best, bestv = Xp, vp
        except Exception:
            pass
    try:
        best = peak_shave(best, 600)
    except Exception:
        pass
    best = _norm_rows(best)
    return best
