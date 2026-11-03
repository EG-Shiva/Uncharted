import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def norm_rows(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n

def rowmaxs(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def lse_full_grad(X, alpha):
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = G.max()
    E = np.exp(alpha * (G - m))
    np.fill_diagonal(E, 0.0)
    S = E.sum() / 2.0 + 1e-300
    L = m + np.log(S) / alpha
    gradX = (E @ X) / S
    return L, gradX, m

def make_full_obj(alpha):
    def fun(vflat):
        V = vflat.reshape(99, 13)
        nr = np.linalg.norm(V, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        Y = V / nr
        L, gradX, _ = lse_full_grad(Y, alpha)
        dot = (gradX * Y).sum(axis=1, keepdims=True)
        gV = (gradX - dot * Y) / nr
        return float(L), gV.ravel().astype(float)
    return fun

def run_lbfgs(fun, x0, maxiter):
    try:
        res = minimize(fun, x0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-20, 'gtol': 1e-10, 'maxfun': maxiter*5})
        return res.x, float(res.fun)
    except Exception:
        return x0, float('inf')

def lp_polish_wide(x, seconds, floor=1e-6):
    x = norm_rows(np.asarray(x, dtype=float))
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 6e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        thr = max(30.0 * radius, floor)
        sel = c > m - thr
        # ensure at least 800 pairs
        if np.sum(sel) < 800:
            order = np.argsort(-c)
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[order[:800]] = True
            sel = sel | sel2
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 4000:
            order = np.argsort(-c[sel])[:4000]
            I, J = I[order], J[order]
            p = 4000
            csel = c[sel][order]
        else:
            csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d+ar, J[:, None]*d+ar, np.full((p,1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = norm_rows(y)
        v = max_cos(y)
        if v < best:
            best = v; best_x = y.copy(); x = y
            radius = min(radius*1.5, 0.02)
        else:
            radius *= 0.5
    return best_x, best

def find_voids(X, k, rng, n_cand=15000):
    n, d = X.shape
    C = rng.standard_normal(size=(n_cand, d))
    C = norm_rows(C)
    # max dot to X (closest point)
    Md = C @ X.T
    best_dots = Md.max(axis=1)
    order = np.argsort(best_dots)  # smallest first = most isolated
    voids = C[order[:k]].copy()
    # greedy spread among voids: sequentially pick far from chosen
    chosen = []
    pool = C[order[:max(k*20, 500)]]
    taken = list(X)
    for _ in range(k):
        if len(pool) == 0:
            break
        dd = pool @ np.array(taken).T
        md = dd.max(axis=1)
        idx = int(np.argmin(md))
        chosen.append(pool[idx])
        taken.append(pool[idx])
        # remove nearby to diversify
        mask = (pool @ pool[idx] < 0.98)
        pool = pool[mask]
        if len(pool) == 0:
            break
    if len(chosen) == 0:
        return voids
    return norm_rows(np.array(chosen))

def anneal_lbfgs(X0, alphas_iters):
    x0 = X0.ravel()
    for alpha, mi in alphas_iters:
        fun = make_full_obj(alpha)
        x0, _ = run_lbfgs(fun, x0, mi)
    return norm_rows(x0.reshape(99, 13))

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        xb = norm_rows(rng.standard_normal(size=(99, 13)))
    else:
        xb = norm_rows(xb)
    best_val = max_cos(xb)
    best_x = xb.copy()

    # quick baseline anneal of incumbent at moderate alpha to get reference
    # Phase 1: basin hopping with shakes + void surgery
    # time limit for this phase
    PHASE1 = 132.0
    it = 0
    sigmas = [0.25, 0.12, 0.06, 0.03, 0.18, 0.09, 0.04, 0.15, 0.07, 0.02, 0.30, 0.10]
    while time.time() - t0 < PHASE1 and it < 12:
        sig = sigmas[it % len(sigmas)]
        it += 1
        cur = best_x.copy()
        if it % 3 == 0:
            # void surgery: replace worst k points with voids
            k = 4 + (it % 4)  # 4..7
            rm = rowmaxs(cur)
            worst = np.argsort(-rm)[:k]
            voids = find_voids(cur, k, rng)
            if voids.shape[0] >= k:
                cur[worst] = voids[:k] + rng.standard_normal(size=(k, 13)) * 0.01
                cur = norm_rows(cur)
            else:
                cur = cur + rng.standard_normal(size=cur.shape) * sig
                cur = norm_rows(cur)
        elif it % 3 == 1:
            # global shake
            cur = cur + rng.standard_normal(size=cur.shape) * sig
            cur = norm_rows(cur)
        else:
            # subset shake: move random 12 points strongly
            idx = rng.choice(99, size=12, replace=False)
            cur[idx] = cur[idx] + rng.standard_normal(size=(12, 13)) * (sig*1.5)
            cur = norm_rows(cur)
            # tiny jitter rest
            cur = cur + rng.standard_normal(size=cur.shape) * 0.005
            cur = norm_rows(cur)
        # anneal: low alpha to balance shell, then sharpen
        remain = PHASE1 - (time.time() - t0)
        if remain < 8:
            break
        # choose iters based on remaining time
        if remain > 60:
            sched = [(60.0, 120), (200.0, 120), (600.0, 120)]
        elif remain > 30:
            sched = [(60.0, 80), (250.0, 80), (600.0, 80)]
        else:
            sched = [(80.0, 60), (400.0, 60)]
        try:
            Xc = anneal_lbfgs(cur, sched)
        except Exception:
            continue
        v = max_cos(Xc)
        if v < best_val:
            best_val = v; best_x = Xc.copy()
        # also try to keep near-miss diverse restart: if within 2e-4, perturb from Xc next time
        # (implicitly via best_x only on improvement; add stochastic acceptance for diversity)
        elif v < best_val + 5e-4 and (it % 4 == 0):
            # accept as restart base once without overwriting best
            # do one extra short sharpen from Xc
            try:
                Xc2 = anneal_lbfgs(Xc + rng.standard_normal(size=Xc.shape)*0.01, [(400.0, 60)])
                Xc2 = norm_rows(Xc2)
                v2 = max_cos(Xc2)
                if v2 < best_val:
                    best_val = v2; best_x = Xc2.copy()
            except Exception:
                pass

    # Phase 2: polish global best with wide LP + final LBFGS touch
    try:
        remain = BUDGET - (time.time() - t0) - 0.5
        if remain > 4:
            # brief high-alpha LBFGS to tighten before LP
            t_lp = remain
            # first LBFGS sharpen
            if remain > 12:
                try:
                    best_x = anneal_lbfgs(best_x, [(800.0, 80)])
                    best_val = max_cos(best_x)
                except Exception:
                    pass
                remain = BUDGET - (time.time() - t0) - 0.5
            Xp, vp = lp_polish_wide(best_x, min(remain, 28.0), floor=1e-6)
            if vp < best_val:
                best_val = vp; best_x = Xp.copy()
            # final tight floor
            remain = BUDGET - (time.time() - t0) - 0.3
            if remain > 2:
                Xp2, vp2 = lp_polish_wide(best_x, remain, floor=1e-9)
                if vp2 < best_val:
                    best_val = vp2; best_x = Xp2.copy()
    except Exception:
        pass

    best_x = norm_rows(np.asarray(best_x, dtype=float))
    best_x = np.nan_to_num(best_x, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best_x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best_x = best_x / nr
    return best_x
