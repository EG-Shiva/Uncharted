import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, r0=0.004, cap=1400):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                sel[int(np.argmax(c))] = True
                cnt = 1
            if cnt > cap:
                idx = np.argsort(-c)[:cap]
                sel2 = np.zeros_like(sel, dtype=bool)
                sel2[idx] = True
                sel = sel2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            cols[:, :d] = I[:, None] * d + ar
            cols[:, d:2 * d] = J[:, None] * d + ar
            cols[:, 2 * d] = n * d
            vals = np.empty((p, 2 * d + 1), dtype=float)
            vals[:, :d] = x[J]
            vals[:, d:2 * d] = x[I]
            vals[:, 2 * d] = -1.0
            A_ub = coo_matrix((vals.ravel(), (rows, cols.ravel())), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x

def _smooth_lbfgs(x_init, deadline, taus=(0.003, 0.001, 0.0004), iters=(70, 70, 80)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.3:
            break
        def fg(yf, _tau=tau):
            Y = yf.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yf)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            GY = (GX - (GX * X).sum(axis=1, keepdims=True) * X) / rn
            return float(L), GY.ravel()
        try:
            r = minimize(fg, y, method="L-BFGS-B", jac=True,
                         options={"maxiter": int(maxiter), "maxfun": int(maxiter * 5), "ftol": 1e-14, "gtol": 1e-8})
            y = r.x
        except Exception:
            break
    Yf = y.reshape(n, d)
    Yf /= np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
    return Yf

def _full_gd(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    bv = float(G.max())
    bx = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.6 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.standard_normal((n, d)) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _tangent_noise(X, amp, rng):
    nz = rng.standard_normal(X.shape) * amp
    nz -= (nz * X).sum(axis=1, keepdims=True) * X
    Y = X + nz
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return Y

# ---- symmetric machinery ----
class SymModel:
    def __init__(self, K, a, L):
        self.K = int(K); self.a = int(a); self.L = int(L)
        self.M = self.L + 1
        self.n = self.L * self.K + 1
        th1 = 2 * np.pi / self.K
        th2 = 2 * np.pi * self.a / self.K
        kk = np.arange(self.K)
        self.c1 = np.cos(th1 * kk); self.s1 = np.sin(th1 * kk)
        self.c2 = np.cos(th2 * kk); self.s2 = np.sin(th2 * kk)
        self.iu0, self.iu1 = np.triu_indices(self.n, 1)
        self.Np = len(self.iu0)

    def expand(self, P):
        # P: Mx4 normalized reps (last is free point)
        K, L = self.K, self.L
        n = self.n
        X = np.empty((n, 4), dtype=float)
        for j in range(L):
            p0, p1, p2, p3 = float(P[j, 0]), float(P[j, 1]), float(P[j, 2]), float(P[j, 3])
            # Q_k p
            X[j * K:(j + 1) * K, 0] = self.c1 * p0 - self.s1 * p1
            X[j * K:(j + 1) * K, 1] = self.s1 * p0 + self.c1 * p1
            X[j * K:(j + 1) * K, 2] = self.c2 * p2 - self.s2 * p3
            X[j * K:(j + 1) * K, 3] = self.s2 * p2 + self.c2 * p3
        X[n - 1] = P[L]
        nr = np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        X /= nr
        return X

    def fg_sym(self, yflat, tau):
        M = self.M; K, L = self.K, self.L; n = self.n
        Y = yflat.reshape(M, 4)
        rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        P = Y / rn
        X = self.expand(P)
        iu0, iu1 = self.iu0, self.iu1
        g = (X @ X.T)[iu0, iu1]
        m = float(g.max())
        e = np.exp((g - m) / tau)
        s = float(e.sum())
        if s <= 0 or not np.isfinite(s):
            return float(m), np.zeros_like(yflat)
        Loss = m + tau * np.log(s / self.Np)
        w = e / s
        W = np.zeros((n, n))
        W[iu0, iu1] = w
        W = W + W.T
        GX = W @ X  # n x 4 grad w.r.t X
        # pullback to P
        GP = np.zeros((M, 4))
        # free point
        GP[L] = GX[n - 1]
        for j in range(L):
            Gj = GX[j * K:(j + 1) * K]  # Kx4
            g0 = Gj[:, 0]; g1 = Gj[:, 1]; g2 = Gj[:, 2]; g3 = Gj[:, 3]
            d0 = float((self.c1 * g0 + self.s1 * g1).sum())
            d1 = float((-self.s1 * g0 + self.c1 * g1).sum())
            d2 = float((self.c2 * g2 + self.s2 * g3).sum())
            d3 = float((-self.s2 * g2 + self.c2 * g3).sum())
            GP[j, 0] = d0; GP[j, 1] = d1; GP[j, 2] = d2; GP[j, 3] = d3
        # P->Y
        dot = (GP * P).sum(axis=1, keepdims=True)
        GY = (GP - dot * P) / rn
        return float(Loss), GY.ravel()

def _sym_optimize(K, a, L, rng, deadline, taus=(0.003, 0.001, 0.0003), iters=(90, 90, 110)):
    model = SymModel(K, a, L)
    M = model.M
    Y0 = rng.standard_normal((M, 4))
    Y0 /= np.maximum(np.linalg.norm(Y0, axis=1, keepdims=True), 1e-12)
    # randomize a bit to avoid symmetric degeneracy: scale
    Y0 = Y0 * (0.8 + 0.4 * rng.random((M, 1)))
    y = Y0.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.4:
            break
        def fg(yf, _tau=tau):
            return model.fg_sym(yf, _tau)
        try:
            r = minimize(fg, y, method="L-BFGS-B", jac=True,
                         options={"maxiter": int(maxiter), "maxfun": int(maxiter * 6), "ftol": 1e-14, "gtol": 1e-7})
            y = r.x
        except Exception:
            break
    Yf = y.reshape(M, 4)
    rn = np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
    Pf = Yf / rn
    X = model.expand(Pf)
    try:
        v = _max_cos(X)
    except Exception:
        v = 1e9
    return X, float(v)

def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    SYM_END = t0 + 42.0
    SEARCH_END = t0 + 152.0
    rng = np.random.default_rng(16327)
    best_x = None; best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (163, 4):
                continue
            a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            if not np.all(np.isfinite(a)):
                continue
            v = _max_cos(a)
            if v < best_v:
                best_v = v; best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        z = rng.standard_normal((163, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best_x = z; best_v = _max_cos(z)
    base_best_x = best_x.copy(); base_best_v = float(best_v)

    # ---- symmetry-constrained search ----
    configs = []
    for a in (5, 8, 11):
        configs.append((27, a, 6))
    for a in (2, 5, 8):
        configs.append((9, a, 18))
    for a in (1, 2):
        configs.append((3, a, 54))
    sym_best_x = None; sym_best_v = 1e9
    sym_results = []
    per_cfg = (SYM_END - time.time()) / max(1, len(configs))
    for (K, a, L) in configs:
        if time.time() > SYM_END - 0.5:
            break
        cfg_dead = min(SYM_END, time.time() + per_cfg)
        # two restarts for small models, one for large
        nrep = 2 if (K in (27, 9)) else 1
        for _ in range(nrep):
            if time.time() > cfg_dead - 0.3:
                break
            try:
                X, v = _sym_optimize(K, a, L, rng, cfg_dead)
            except Exception:
                continue
            sym_results.append((float(v), K, a, L))
            if v < sym_best_v:
                sym_best_v = float(v); sym_best_x = X.copy()
            # tiny time check
            if time.time() > SYM_END - 0.5:
                break
    # if symmetric search failed, fabricate from base
    if sym_best_x is None:
        sym_best_x = base_best_x.copy(); sym_best_v = base_best_v

    # light polish of symmetric best to evaluate fairly (short LP)
    try:
        if time.time() < SYM_END + 6:
            tmp = _smooth_lbfgs(sym_best_x, min(time.time() + 4.0, DEADLINE - 100))
            # keep symmetric topology value but use smoothed for breakout start
            sym_start = tmp
        else:
            sym_start = sym_best_x
    except Exception:
        sym_start = sym_best_x
    try:
        vs = _max_cos(sym_start)
        if vs < sym_best_v:
            sym_best_v = vs
    except Exception:
        pass

    cur = sym_start.copy()
    cur_v = _max_cos(cur)
    # track global best including symmetric start
    if cur_v < best_v:
        best_x = cur.copy(); best_v = float(cur_v)

    # ---- gradual symmetry breaking: noise 0.005, 0.02, 0.05 ----
    amps = [0.005, 0.02, 0.05]
    # stage budgets: split remaining search time
    for si, amp in enumerate(amps):
        if time.time() > SEARCH_END:
            break
        stage_end = SEARCH_END - (len(amps) - si - 1) * ((SEARCH_END - time.time()) / (len(amps) - si)) * 0.9
        # add tangent noise
        try:
            cur = _tangent_noise(cur, amp, rng)
        except Exception:
            pass
        # full anneal params per stage: larger noise stages get hotter/longer
        if si == 0:
            steps, T0, T1, lr0, mu, nz = 1400, 0.004, 0.0004, 0.25, 0.85, 0.002
        elif si == 1:
            steps, T0, T1, lr0, mu, nz = 1800, 0.006, 0.0003, 0.30, 0.85, 0.004
        else:
            steps, T0, T1, lr0, mu, nz = 2200, 0.008, 0.00025, 0.32, 0.88, 0.005
        anneal_dead = min(stage_end - 6.0, time.time() + (stage_end - time.time()) * 0.55)
        if anneal_dead > time.time() + 0.5:
            try:
                cur, _ = _full_gd(cur, steps, T0, T1, lr0, mu, rng, noise0=nz, deadline=anneal_dead)
            except Exception:
                pass
        smooth_dead = min(stage_end - 2.5, time.time() + (stage_end - time.time()) * 0.5)
        if smooth_dead > time.time() + 0.5:
            try:
                cur = _smooth_lbfgs(cur, smooth_dead)
            except Exception:
                pass
        lp_dead = min(stage_end, time.time() + 5.0)
        if lp_dead > time.time() + 0.3:
            try:
                cur = _lp_polish(cur, lp_dead, r0=0.004)
            except Exception:
                pass
        try:
            cv = _max_cos(cur)
        except Exception:
            cv = 1e9
        if cv < best_v - 1e-15:
            best_x = cur.copy(); best_v = float(cv)
        # Metropolis-ish: if stage worsened a lot, revert to best with small kick
        # but keep exploring from current unless far worse
        if cv > best_v + 0.004 and time.time() < SEARCH_END - 5:
            try:
                cur = _tangent_noise(best_x, 0.01, rng)
            except Exception:
                cur = best_x.copy()

    # ---- final exploitation from overall best ----
    # short extra anneal + smooth + LP on best to squeeze
    try:
        if time.time() < DEADLINE - 12:
            cand = best_x.copy()
            # small shake to leave exact jam then re-polish
            cand = _tangent_noise(cand, 0.003, rng)
            cand, _ = _full_gd(cand, 900, 0.003, 0.0003, 0.22, 0.85, rng, noise0=0.001, deadline=min(DEADLINE - 8, time.time() + 6))
            cand = _smooth_lbfgs(cand, min(DEADLINE - 5, time.time() + 4))
            cand = _lp_polish(cand, min(DEADLINE - 1.5, time.time() + 5), r0=0.003)
            cv = _max_cos(cand)
            if cv < best_v:
                best_x = cand; best_v = cv
    except Exception:
        pass
    try:
        if time.time() < DEADLINE - 2:
            best_x = _lp_polish(best_x, DEADLINE - 0.5, r0=0.0015)
            best_v = _max_cos(best_x)
    except Exception:
        pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    # strict feasibility: no zero/nan
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    nr = np.linalg.norm(best_x, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    best_x = best_x / nr
    return best_x
