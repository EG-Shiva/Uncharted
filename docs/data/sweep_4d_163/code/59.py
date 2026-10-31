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
        if time.time() - start > seconds - 0.05:
            break
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

class SymProblem:
    def __init__(self, m, ks):
        self.m = int(m)
        self.ks = list(ks)
        self.nrep = len(ks)
        self.N = 1 + self.nrep * self.m
        th = 2.0 * np.pi / self.m
        s = np.arange(self.m)
        self.c1 = np.cos(s * th)
        self.s1 = np.sin(s * th)
        c2l = []
        s2l = []
        for k in self.ks:
            c2l.append(np.cos(s * th * k))
            s2l.append(np.sin(s * th * k))
        self.c2l = np.array(c2l)
        self.s2l = np.array(s2l)
        self.iu0, self.iu1 = np.triu_indices(self.N, 1)

    def expand_from_R(self, Rpole, Rreps):
        N = self.N; m = self.m; nrep = self.nrep
        X = np.empty((N, 4))
        X[0] = Rpole
        for j in range(nrep):
            r0, r1, r2, r3 = Rreps[j]
            c2 = self.c2l[j]; s2 = self.s2l[j]
            seg = np.empty((m, 4))
            seg[:, 0] = r0 * self.c1 - r1 * self.s1
            seg[:, 1] = r0 * self.s1 + r1 * self.c1
            seg[:, 2] = r2 * c2 - r3 * s2
            seg[:, 3] = r2 * s2 + r3 * c2
            X[1 + j * m:1 + (j + 1) * m] = seg
        return X

    def fun_grad(self, uflat, T):
        nrep = self.nrep
        U = uflat.reshape(nrep + 1, 4)
        nr = np.linalg.norm(U, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        R = U / nr
        Rpole = R[0]
        Rreps = R[1:]
        X = self.expand_from_R(Rpole, Rreps)
        iu0, iu1 = self.iu0, self.iu1
        G = X @ X.T
        c_all = G[iu0, iu1]
        m0 = float(c_all.max())
        ex = np.exp((c_all - m0) / T)
        S = float(ex.sum())
        sval = m0 + T * np.log(S + 1e-300)
        w = ex / (S + 1e-300)
        N = self.N
        W = np.zeros((N, N))
        W[iu0, iu1] = w
        W = W + W.T
        GX = W @ X
        GR = np.zeros((nrep + 1, 4))
        GR[0] = GX[0]
        for j in range(nrep):
            Gs = GX[1 + j * self.m:1 + (j + 1) * self.m]
            g0 = Gs[:, 0]; g1 = Gs[:, 1]; g2 = Gs[:, 2]; g3 = Gs[:, 3]
            c2 = self.c2l[j]; s2 = self.s2l[j]
            c1 = self.c1; s1 = self.s1
            GR[1 + j, 0] = float(np.sum(g0 * c1 + g1 * s1))
            GR[1 + j, 1] = float(np.sum(-g0 * s1 + g1 * c1))
            GR[1 + j, 2] = float(np.sum(g2 * c2 + g3 * s2))
            GR[1 + j, 3] = float(np.sum(-g2 * s2 + g3 * c2))
        dot = np.sum(GR * R, axis=1, keepdims=True)
        GU = (GR - dot * R) / nr
        return sval, GU.ravel(), X, m0

def _sym_opt_once(prob, u0, T_sched, maxiter_per_stage):
    u = u0.copy()
    for T in T_sched:
        def f(v):
            val, g, _, _ = prob.fun_grad(v, T)
            return val, g
        try:
            res = minimize(f, u, jac=True, method='L-BFGS-B',
                           options={'maxiter': maxiter_per_stage, 'ftol': 1e-12, 'gtol': 1e-7})
            u = res.x
        except Exception:
            break
    _, _, X, m0 = prob.fun_grad(u, T_sched[-1])
    return _norm(X), float(m0), u

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    # load incumbent
    cands0 = []
    for fn in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(fn), dtype=float)
            if c.shape == (163, 4):
                c = _norm(c)
                if np.all(np.isfinite(c)):
                    cands0.append(c)
        except Exception:
            pass
    if len(cands0) == 0:
        rng0 = np.random.default_rng(0)
        best = _norm(rng0.standard_normal((163, 4)))
    else:
        vals0 = [_max_cos(c) for c in cands0]
        best = cands0[int(np.argmin(vals0))]
    best_val = _max_cos(best)
    # symmetric configs: main 1+6x27 plus nearby 1+3x54, 1+2x81
    configs = [
        (27, [1, 2, 4, 5, 7, 11]),
        (27, [1, 2, 3, 5, 8, 13]),
        (27, [0, 1, 2, 5, 8, 11]),
        (27, [1, 4, 7, 10, 13, 16]),
        (27, [1, 2, 6, 9, 12, 18]),
        (54, [1, 5, 11]),
        (54, [1, 7, 13]),
        (81, [1, 2]),
        (81, [1, 5]),
    ]
    sym_cands = []
    # deterministic seeds for multi-start
    seed_base = 1000
    T_sched = [0.02, 0.008, 0.003]
    # time box for sym phase: 60s
    sym_deadline = t0 + 60.0
    ci = 0
    for (m, ks) in configs:
        if time.time() > sym_deadline or time.time() > deadline - 60.0:
            break
        # check size
        if 1 + len(ks) * m != 163:
            continue
        prob = SymProblem(m, ks)
        for rep in range(2):
            if time.time() > sym_deadline or time.time() > deadline - 60.0:
                break
            seed = seed_base + ci * 10 + rep
            rng = np.random.default_rng(seed)
            nrep = len(ks)
            u0 = rng.standard_normal(((nrep + 1) * 4,))
            # annealed L-BFGS, short
            Xsym, m0, _ = _sym_opt_once(prob, u0, T_sched, maxiter_per_stage=120)
            # degeneracy check: min off-diag cosine too close to 1
            G = Xsym @ Xsym.T
            np.fill_diagonal(G, -2.0)
            mx = float(G.max())
            if mx > 0.999999:
                # degenerate orbit, skip (nearby decomposition already in list)
                continue
            sym_cands.append((mx, Xsym, (m, tuple(ks), seed)))
        ci += 1
    # sort sym cands by max cos
    sym_cands.sort(key=lambda t: t[0])
    # keep top few
    keep = sym_cands[:4]
    # symmetry breaking: perturb + smooth + LP
    # build pool: symmetric expansions + incumbent
    pool = []
    for _, Xsym, _ in keep:
        pool.append(Xsym)
    # also include incumbent once (to ensure we don't lose it)
    # perturb pool at several amplitudes
    rng = np.random.default_rng(7)
    # time box: leave 8s for final polish
    # quick evaluation: first smooth sym cands lightly and LP polish briefly to rank
    ranked = []
    for X in pool:
        if time.time() > deadline - 45.0:
            break
        # light smooth to remove exact symmetry degeneracy
        y = X + 1e-4 * rng.standard_normal(X.shape)
        y = _norm(y)
        y = _smooth(y, 80, 0.01, 0.002, 0.07, 0.015)
        v = _max_cos(y)
        ranked.append((v, y))
    # if no sym cands, ranked empty -> use incumbent perturbations
    # now main break-symmetry loop over ranked + incumbent
    # create working list
    work_seeds = []
    for _, y in ranked:
        work_seeds.append(y)
    # add incumbent
    work_seeds.append(best.copy())
    sigmas = [0.002, 0.008, 0.025, 0.06]
    # iterate round-robin until time runs out
    it = 0
    while time.time() < deadline - 9.0:
        it += 1
        # pick base: cycle through work_seeds, plus best
        if len(work_seeds) == 0:
            base = best.copy()
        else:
            base = work_seeds[it % len(work_seeds)]
        sigma = sigmas[it % len(sigmas)]
        # alternate small perturb of best vs work seed
        if it % 3 == 0:
            base = best
        trial = base + sigma * rng.standard_normal(base.shape)
        trial = _norm(trial)
        # short smooth descent with varying schedule
        if it % 2 == 0:
            trial = _smooth(trial, 120, 0.008, 0.0015, 0.07, 0.012)
        else:
            trial = _smooth(trial, 180, 0.015, 0.001, 0.06, 0.01)
        v = _max_cos(trial)
        if v > best_val + 0.02:
            continue
        secs = min(2.2, deadline - 9.0 - time.time())
        if secs < 0.4:
            if v < best_val:
                best, best_val = trial, v
            continue
        # LP polish with small radius if close, larger if far
        r0 = 0.002 if v < best_val + 0.002 else 0.004
        trial2 = _lp_polish(trial, secs, r0=r0)
        v2 = _max_cos(trial2)
        if v2 < best_val - 1e-12:
            best, best_val = trial2, v2
            # add improved to work set
            work_seeds.append(trial2.copy())
            if len(work_seeds) > 8:
                work_seeds.pop(0)
        elif v < best_val - 1e-12:
            best, best_val = trial, v
        if time.time() > deadline - 9.5:
            break
    # final polish of best
    rem = deadline - time.time() - 1.0
    if rem > 1.0:
        best = _lp_polish(best, min(rem, 8.0), r0=0.002)
    best = _norm(best)
    return best
