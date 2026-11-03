import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

# ---------- symmetric log-sum-exp problem ----------
class SymProb:
    def __init__(self, n_orb, double, n_extra, d=13):
        self.n_orb = n_orb
        self.double = double
        self.n_extra = n_extra
        self.d = d
        if double:
            self.n_sym = n_orb * 26
        else:
            self.n_sym = n_orb * 13
        self.n = self.n_sym + n_extra
        self.np_seed = n_orb * d
        self.np_extra = n_extra * d

    def unpack(self, p):
        d = self.d
        S = p[:self.np_seed].reshape(self.n_orb, d) if self.np_seed > 0 else np.zeros((0, d))
        E = p[self.np_seed:].reshape(self.n_extra, d) if self.np_extra > 0 else np.zeros((0, d))
        return S, E

    def build_Y(self, S, E):
        d = self.d
        rows = []
        if self.n_orb > 0:
            for s in range(self.n_orb):
                for k in range(13):
                    rows.append(np.roll(S[s], k))
                if self.double:
                    for k in range(13):
                        rows.append(-np.roll(S[s], k))
        Ys = np.vstack(rows) if len(rows) > 0 else np.zeros((0, d))
        if self.n_extra > 0:
            Y = np.vstack([Ys, E]) if Ys.shape[0] > 0 else E.copy()
        else:
            Y = Ys
        return Y

    def build_X(self, p):
        S, E = self.unpack(p)
        Y = self.build_Y(S, E)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return Y / nrm, Y, nrm

    def fun_jac(self, p, beta):
        S, E = self.unpack(p)
        Y = self.build_Y(S, E)
        n = Y.shape[0]
        d = self.d
        nrm = np.linalg.norm(Y, axis=1)
        nrm[nrm == 0] = 1.0
        X = Y / nrm[:, None]
        G = X @ X.T
        iu0, iu1 = np.triu_indices(n, 1)
        c = G[iu0, iu1]
        m = float(c.max())
        e = np.exp(beta * (c - m))
        se = float(e.sum()) + 1e-300
        L = m + np.log(se) / beta
        w = e / se
        # W matrix
        W = np.zeros((n, n))
        W[iu0, iu1] = w
        W = W + W.T
        grad_X = W @ X  # n x d
        # to grad_Y
        dot = np.sum(grad_X * X, axis=1)
        grad_Y = (grad_X - dot[:, None] * X) / nrm[:, None]
        # back to seeds
        grad_S = np.zeros_like(S)
        # split grad_Y
        idx = 0
        for s in range(self.n_orb):
            # positive block 13 rows
            Gb = grad_Y[idx:idx + 13, :]
            idx += 13
            acc = np.zeros(d)
            for k in range(13):
                acc += np.roll(Gb[k], -k)
            if self.double:
                Gn = grad_Y[idx:idx + 13, :]
                idx += 13
                accn = np.zeros(d)
                for k in range(13):
                    accn += np.roll(Gn[k], -k)
                acc = acc - accn  # sign flip
            grad_S[s] = acc
        grad_E = grad_Y[idx:, :].reshape(-1) if self.n_extra > 0 else np.zeros(0)
        g = np.concatenate([grad_S.reshape(-1), grad_E])
        return L, g

def sym_optimize(prob, p0, betas, maxiter_each=150):
    p = p0.copy()
    for b in betas:
        def f(p_):
            L, g = prob.fun_jac(p_, b)
            return L, g
        try:
            res = minimize(f, p, jac=True, method='L-BFGS-B',
                           options={'maxiter': maxiter_each, 'maxfun': maxiter_each * 4,
                                    'ftol': 1e-14, 'gtol': 1e-7})
            p = res.x
        except Exception:
            break
    return p

def _descend(v0, steps, b0, b1, l0, l1, mu=0.7):
    v = v0.copy()
    vel = np.zeros_like(v)
    n, d = v.shape
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = b0 * (b1 / b0) ** f
        lr = l0 * (l1 / l0) ** f
        G = v @ v.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ v
        dot = np.sum(grad * v, axis=1, keepdims=True)
        gt = (grad - dot * v) / rs
        vel = mu * vel + gt
        # normalize scale guard
        v = v - lr * vel
        v = _norm(v)
    return v

def _lp_polish(x, seconds, p_cap=1200):
    t_end = time.time() + seconds
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    bx = x.copy()
    radius = 1e-3
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while time.time() < t_end and radius > 1e-14 and it < 60:
        it += 1
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        win = max(20.0 * radius, 1e-9)
        sel = c > m - win
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.flatnonzero(sel)
            part = np.argpartition(-c[idx], p_cap - 1)[:p_cap]
            ns = np.zeros_like(sel)
            ns[idx[part]] = True
            sel = ns
        I = iu0[sel]; J = iu1[sel]; cs = c[sel]
        p_ = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p_), 2 * d + 1)
        cI = (I[:, None] * d + np.arange(d)).reshape(p_, d)
        cJ = (J[:, None] * d + np.arange(d)).reshape(p_, d)
        cols = np.concatenate([cI, cJ, np.full((p_, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p_, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p_, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            bx = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = bx.copy()
    return bx

def solve():
    t0 = time.time()
    HARD = 172.0
    n, d = 123, 13
    best_val = 1e9
    best = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    rng = np.random.RandomState(0)
    if best is None:
        best = _norm(rng.randn(n, d))
        best_val = _max_cos(best)
    x_best = best.copy()

    # symmetry search configs: (n_orb, double, n_extra, seed_id, init_mode)
    configs = [
        (9, False, 6, 1, 'rand'),
        (9, False, 6, 2, 'rand'),
        (9, False, 6, 3, 'rand'),
        (8, False, 19, 4, 'rand'),
        (4, True, 19, 5, 'rand'),
        (9, False, 6, 6, 'from_best'),
    ]
    betas = [30.0, 80.0, 220.0, 600.0]
    for (n_orb, double, n_extra, sid, mode) in configs:
        if time.time() - t0 > 105:
            break
        prob = SymProb(n_orb, double, n_extra, d)
        rs = np.random.RandomState(1000 + sid)
        dim = n_orb * d + n_extra * d
        if mode == 'rand':
            p0 = rs.randn(dim) * 1.0
        else:
            # init seeds from best points + noise
            p0 = rs.randn(dim) * 1.0
            try:
                # pick n_orb seeds from best
                idx = rs.choice(n, n_orb, replace=False)
                S0 = x_best[idx] + 0.1 * rs.randn(n_orb, d)
                p0[:n_orb * d] = S0.reshape(-1)
                if n_extra <= n:
                    idx2 = rs.choice(n, n_extra, replace=False)
                    E0 = x_best[idx2] + 0.1 * rs.randn(n_extra, d)
                    p0[n_orb * d:] = E0.reshape(-1)
            except Exception:
                pass
        # quick budget check
        if HARD - (time.time() - t0) < 45:
            break
        p_opt = sym_optimize(prob, p0, betas, maxiter_each=120)
        Xsym, _, _ = prob.build_X(p_opt)
        Xsym = _norm(Xsym)
        if Xsym.shape != (n, d):
            continue
        sv = _max_cos(Xsym)
        # release symmetry: small break + annealed descend
        # choose descend budget adaptively
        try:
            Xt = Xsym + 0.002 * rs.randn(n, d)
            Xt = _norm(Xt)
            # short release
            Xr = _descend(Xt, 400, 40.0, 500.0, 0.007, 0.0007)
            rv = _max_cos(Xr)
        except Exception:
            Xr = Xsym
            rv = sv
        # consider both
        for cand, cv in ((Xsym, sv), (Xr, rv)):
            if cv < best_val and np.isfinite(cv):
                best_val = cv
                x_best = cand.copy()
                # snap with tiny LP if time allows
                if HARD - (time.time() - t0) > 35:
                    try:
                        vp = _lp_polish(cand, 2.0)
                        pv = _max_cos(vp)
                        if pv < best_val:
                            best_val = pv
                            x_best = vp.copy()
                    except Exception:
                        pass
        # also try longer refine of Xr if promising within 0.01 of best
        if rv < best_val + 0.015 and (HARD - (time.time() - t0) > 35):
            try:
                Xr2 = _descend(Xr, 400, 50.0, 700.0, 0.005, 0.0005)
                r2v = _max_cos(Xr2)
                if r2v < best_val:
                    best_val = r2v
                    x_best = Xr2.copy()
                    vp = _lp_polish(Xr2, 2.0)
                    pv = _max_cos(vp)
                    if pv < best_val:
                        best_val = pv
                        x_best = vp.copy()
            except Exception:
                pass

    remain = HARD - (time.time() - t0)
    if remain > 8:
        try:
            x_best = _lp_polish(x_best, min(remain - 2, 26.0))
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
