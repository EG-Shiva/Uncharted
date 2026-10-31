import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _smooth(x, iters, T0, T1, lr0=0.07, lr1=0.015):
    n, d = x.shape
    for k in range(iters):
        frac = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        rs = W.sum(axis=1, keepdims=True)
        step = gt / np.maximum(rs, 1e-12)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, seconds, r0=2e-3, cap=1500):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    iu0, iu1 = np.triu_indices(n, 1)
    cd = np.arange(d)
    radius = r0
    start = time.time()
    while time.time() - start < seconds and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thr)[0]
        if sel.size == 0:
            break
        if sel.size > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = np.where(c_all >= kth)[0]
            if sel.size > cap + 400:
                sel = sel[::2][:cap]
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.02)
        else:
            radius *= 0.5
    return x

# ---- symmetric two-plane cyclic machinery ----
def _make_sym_opt(m, ks, pole, T_sched):
    nrep = len(ks)
    N = 1 + nrep * m
    th = 2.0 * np.pi / m
    s = np.arange(m)
    c1 = np.cos(s * th); s1 = np.sin(s * th)
    c2l = []; s2l = []
    for k in ks:
        c2l.append(np.cos(s * th * k)); s2l.append(np.sin(s * th * k))
    c2l = np.array(c2l); s2l = np.array(s2l)
    iu0, iu1 = np.triu_indices(N, 1)

    def expand(R):
        X = np.empty((N, 4))
        X[0] = pole
        for j in range(nrep):
            r0, r1, r2, r3 = R[j]
            c2 = c2l[j]; s2 = s2l[j]
            seg = np.empty((m, 4))
            seg[:, 0] = r0 * c1 - r1 * s1
            seg[:, 1] = r0 * s1 + r1 * c1
            seg[:, 2] = r2 * c2 - r3 * s2
            seg[:, 3] = r2 * s2 + r3 * c2
            X[1 + j * m:1 + (j + 1) * m] = seg
        return X

    def fun_grad(uflat, T):
        U = uflat.reshape(nrep, 4)
        nr = np.linalg.norm(U, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        R = U / nr
        X = expand(R)
        G = X @ X.T
        c_all = G[iu0, iu1]
        m0 = float(c_all.max())
        ex = np.exp((c_all - m0) / T)
        S = float(ex.sum())
        sval = m0 + T * np.log(S + 1e-300)
        w = ex / (S + 1e-300)
        W = np.zeros((N, N))
        W[iu0, iu1] = w
        W = W + W.T
        GX = W @ X
        # backprop to R
        GR = np.zeros((nrep, 4))
        for j in range(nrep):
            Gs = GX[1 + j * m:1 + (j + 1) * m]
            g0 = Gs[:, 0]; g1 = Gs[:, 1]; g2 = Gs[:, 2]; g3 = Gs[:, 3]
            c2 = c2l[j]; s2 = s2l[j]
            GR[j, 0] = float(np.sum(g0 * c1 + g1 * s1))
            GR[j, 1] = float(np.sum(-g0 * s1 + g1 * c1))
            GR[j, 2] = float(np.sum(g2 * c2 + g3 * s2))
            GR[j, 3] = float(np.sum(-g2 * s2 + g3 * c2))
        dot = np.sum(GR * R, axis=1, keepdims=True)
        GU = (GR - dot * R) / nr
        # pin norm to 1
        pen = 0.0
        for j in range(nrep):
            nn = float(nr[j, 0])
            pen += 1e-3 * (nn - 1.0) ** 2
            GU[j] += 2e-3 * (nn - 1.0) * (U[j] / nn)
        return sval + pen, GU.ravel(), X, m0

    return expand, fun_grad, N

def _opt_one(m, ks, pole, seed, T_sched=(0.02, 0.008, 0.003), maxiter=100):
    expand, fun_grad, N = _make_sym_opt(m, ks, pole, T_sched)
    nrep = len(ks)
    rng = np.random.default_rng(seed)
    U0 = rng.normal(size=(nrep, 4))
    U0 = U0 / np.linalg.norm(U0, axis=1, keepdims=True)
    x0 = U0.ravel()
    for T in T_sched:
        def fg(x):
            v, g, _, _ = fun_grad(x, T)
            return v, g
        try:
            res = minimize(fg, x0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7})
            x0 = res.x
        except Exception:
            break
    _, _, Xf, _ = fun_grad(x0, T_sched[-1])
    Xf = _norm(Xf)
    return Xf, _max_cos(Xf)

def solve():
    t0 = time.time()
    DEAD = 172.0
    SYM_BUDGET = 62.0
    rng = np.random.default_rng(12345)
    # incumbent
    inc = None; inc_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            c = np.load(f)
            c = np.asarray(c, dtype=float)
            if c.shape == (163, 4):
                c = _norm(np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0))
                if np.all(np.isfinite(c)):
                    v = _max_cos(c)
                    if v < inc_val:
                        inc_val = v; inc = c.copy()
        except Exception:
            pass
    if inc is None:
        inc = _norm(rng.normal(size=(163, 4)))
        inc_val = _max_cos(inc)
    pole = np.array([0.0, 0.0, 0.0, 1.0])

    sym_results = []
    # main family 1+6*27
    configs = []
    for k in [1, 2, 4, 5, 7, 8]:
        configs.append((27, [k] * 6))
    # mixed twists
    configs.append((27, [1, 1, 1, 2, 2, 2]))
    configs.append((27, [1, 2, 4, 5, 7, 8]))
    # nearby decompositions if orbit degenerates
    configs.append((54, [1, 5, 7]))
    configs.append((54, [1, 1, 5]))
    configs.append((81, [1, 2]))
    configs.append((81, [1, 4]))
    seed_base = 2000
    for ci, (m, ks) in enumerate(configs):
        for rep in range(2):
            if time.time() - t0 > SYM_BUDGET:
                break
            seed = seed_base + ci * 10 + rep
            try:
                Xf, v = _opt_one(m, ks, pole, seed,
                                 T_sched=(0.02, 0.008, 0.003), maxiter=90)
            except Exception:
                continue
            if not np.all(np.isfinite(Xf)):
                continue
            if v > 0.99:  # degenerated orbit (coincident points)
                continue
            sym_results.append((v, Xf))
        if time.time() - t0 > SYM_BUDGET:
            break
    sym_results.sort(key=lambda t: t[0])
    top_sym = [x for _, x in sym_results[:3]]

    # candidate pool: incumbent + symmetric expansions
    best = inc.copy(); best_val = inc_val
    for X in top_sym:
        v = _max_cos(X)
        if v < best_val:
            best = X.copy(); best_val = v

    # break symmetry: perturb + short smooth descent
    cands = list(top_sym)
    # also include incumbent lightly shaken as fallback diversity
    if len(cands) < 3:
        cands.append(inc)
    perturbed = []
    for X in cands:
        for amp in (0.006, 0.025):
            if time.time() - t0 > DEAD - 45:
                break
            Y = X + rng.normal(scale=amp, size=X.shape)
            Y = _norm(Y)
            Y = _smooth(Y, 220, 0.02, 0.002, 0.07, 0.012)
            v = _max_cos(Y)
            perturbed.append((v, Y))
            if v < best_val:
                best = Y.copy(); best_val = v
        if time.time() - t0 > DEAD - 45:
            break
    # a couple of larger shakes of current best for basin change
    for amp in (0.05, 0.012):
        if time.time() - t0 > DEAD - 40:
            break
        Y = best + rng.normal(scale=amp, size=best.shape)
        Y = _norm(Y)
        Y = _smooth(Y, 200, 0.015, 0.0015, 0.06, 0.01)
        v = _max_cos(Y)
        if v < best_val:
            best = Y.copy(); best_val = v

    # LP polish best with remaining time
    remain = DEAD - (time.time() - t0)
    lp_time = float(min(50.0, max(8.0, remain - 3.0)))
    if lp_time > 3 and time.time() - t0 < DEAD - 4:
        best = _lp_polish(best, lp_time, r0=2e-3, cap=1500)
        best_val = _max_cos(best)
    # final micro shake + repolish if time left
    if DEAD - (time.time() - t0) > 14:
        for _ in range(2):
            Y = best + rng.normal(scale=0.003, size=best.shape)
            Y = _norm(Y)
            Y = _smooth(Y, 100, 0.008, 0.0015, 0.03, 0.008)
            v = _max_cos(Y)
            if v < best_val:
                best = Y; best_val = v
            if DEAD - (time.time() - t0) < 11:
                break
        r2 = float(min(9.0, DEAD - (time.time() - t0) - 1.0))
        if r2 > 2:
            best = _lp_polish(best, r2, r0=8e-4, cap=1500)
    # safety: return incumbent if something worse
    if _max_cos(best) > inc_val and inc is not None:
        best = inc
    best = _norm(best)
    return best
