import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _fingerprint(X, k=200):
    n = X.shape[0]
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    g = G[iu]
    idx = np.argsort(-g)[:k]
    return np.sort(g[idx])[::-1].copy()

def _make_fg_M(n, d, M, beta):
    def fg(yf):
        U = yf.reshape(n, d)
        r = np.linalg.norm(U, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        X = U / r
        MX = X @ M  # (n,d), M symmetric
        m = np.sum(X * MX, axis=1)  # (n,)
        m = np.maximum(m, 1e-12)
        rn = np.sqrt(m)
        Nmat = X @ MX.T
        denom = rn[:, None] * rn[None, :]
        G = Nmat / denom
        iu = np.triu_indices(n, 1)
        # need full G for gradient row sums, but LSE over upper
        g = G[iu]
        mx = float(g.max())
        e = np.exp(beta * (g - mx))
        S = float(e.sum()) + 1e-300
        F = mx + np.log(S) / beta
        w = e / S
        P = np.zeros((n, n))
        P[iu] = w
        P = P + P.T
        # GradX
        # Qmat = MX / rn
        Qmat = MX / rn[:, None]
        PQ = P @ Qmat  # (n,d)
        s = np.sum(P * G, axis=1)  # (n,)
        GradX = PQ / rn[:, None] - s[:, None] * MX / m[:, None]
        dot = np.sum(GradX * X, axis=1, keepdims=True)
        GradU = (GradX - dot * X) / r
        return F, GradU.ravel()
    return fg

def _run_lbfgs_M(X0, M, beta, maxiter):
    n, d = X0.shape
    fg = _make_fg_M(n, d, M, beta)
    y0 = (X0 / np.linalg.norm(X0, axis=1, keepdims=True)).ravel().copy()
    try:
        res = minimize(fg, y0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-7, 'maxfun': maxiter * 2})
        Y = res.x.reshape(n, d)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return X0 / np.linalg.norm(X0, axis=1, keepdims=True)

def _lp_polish_M(X, M, deadline, r0=2e-4, cap_p=1200, max_solves=200):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = r0
    iu = np.triu_indices(n, 1)
    solves = 0
    while time.time() < deadline and radius > 1e-14 and solves < max_solves:
        if time.time() > deadline - 0.35:
            break
        MX = X @ M
        m = np.maximum(np.sum(X * MX, axis=1), 1e-12)
        rn = np.sqrt(m)
        Nmat = X @ MX.T
        G = Nmat / (rn[:, None] * rn[None, :])
        c = G[iu]
        mx = float(c.max())
        sel = c > mx - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        csel = c[sel]
        if p > cap_p:
            od = np.argsort(-csel)[:cap_p]
            I, J = I[od], J[od]
            csel = csel[od]
            p = cap_p
        # coefficients a_I, a_J
        # g = csel
        # a_I = MX_J/(rn_I rn_J) - g*MX_I/m_I
        rnI = rn[I]; rnJ = rn[J]
        den = rnI * rnJ
        AI = MX[J] / den[:, None] - (csel / m[I])[:, None] * MX[I]
        AJ = MX[I] / den[:, None] - (csel / m[J])[:, None] * MX[J]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([AI, AJ, -np.ones((p, 1))], 1).ravel()
        # equality: X_k^T dx_k = 0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        solves += 1
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        y = X + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            improved = False
            for sh in (0.3, 0.1):
                yy = X + step * sh
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    X, best = yy, vv
                    improved = True
                    break
            if not improved:
                radius *= 0.5
    return X

def solve():
    t0 = time.time()
    TLIM = 172.0
    deadline_final = t0 + TLIM
    n, d = 68, 12
    # load incumbent
    x_inc = None
    bv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                G = a @ a.T
                np.fill_diagonal(G, -2.0)
                s = float(G.max())
                if s < bv:
                    bv = s
                    x_inc = a.copy()
        except Exception:
            pass
    rng0 = np.random.default_rng(20261031)
    if x_inc is None:
        x_inc = rng0.standard_normal((n, d))
        x_inc /= np.linalg.norm(x_inc, axis=1, keepdims=True)
        bv = _max_cos(x_inc)
    x_best = x_inc.copy()
    v_best = float(bv)
    f_inc = _fingerprint(x_inc, k=200)
    # homotopy paths
    npaths = 8
    schedule = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    rng = np.random.default_rng(777123)
    # pre-generate Q, eps, start type for determinism
    path_specs = []
    for pi in range(npaths):
        A = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(A)
        eps = rng.uniform(-0.35, 0.35, size=d)
        from_inc = (pi < 4)  # 4 from incumbent, 4 from random
        path_specs.append((Q, eps, from_inc))
    # random starts
    rand_starts = []
    for pi in range(npaths):
        if not path_specs[pi][2]:
            R = rng.standard_normal((n, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            rand_starts.append(R)
        else:
            rand_starts.append(None)

    for pi in range(npaths):
        # reserve time for final polish
        if time.time() > deadline_final - 22.0:
            break
        Q, epsv, from_inc = path_specs[pi]
        if from_inc:
            # small jitter to avoid exact same trajectory
            jr = np.random.default_rng(1000 + pi)
            X = x_inc.copy()
            # tiny shake so L-BFGS under deformed metric diverges
            X = X + 0.005 * jr.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            X = rand_starts[pi].copy()
            # quick standard pre-opt for random starts to make them competitive
            # one short LSE run with identity
            MI = np.eye(d)
            X = _run_lbfgs_M(X, MI, 400.0, 50)
        # six-step homotopy
        for si, alpha in enumerate(schedule):
            if time.time() > deadline_final - 20.0:
                break
            vals = np.exp(epsv * alpha)
            M = (Q * vals[None, :]) @ Q.T
            # symmetrize
            M = 0.5 * (M + M.T)
            if alpha > 1e-12:
                beta = 600.0
                X = _run_lbfgs_M(X, M, beta, 60)
                # short LP in deformed metric
                try:
                    dl = min(time.time() + 1.2, deadline_final - 18.0)
                    if dl > time.time() + 0.3:
                        X = _lp_polish_M(X, M, dl, r0=2e-4, cap_p=1000, max_solves=4)
                except Exception:
                    pass
            else:
                # final standard metric: two sharpness stages
                MI = np.eye(d)
                X = _run_lbfgs_M(X, MI, 1000.0, 80)
                X = _run_lbfgs_M(X, MI, 3000.0, 60)
        v_end = _max_cos(X)
        f_end = _fingerprint(X, k=200)
        dist = float(np.linalg.norm(f_end - f_inc))
        if dist < 1e-4:
            # same basin: accept only if strictly better (fallback), no LP polish
            if v_end < v_best - 1e-15:
                x_best = X.copy()
                v_best = v_end
            continue
        # fingerprint-distinct: LP-polish
        try:
            dl2 = min(time.time() + 3.0, deadline_final - 15.0)
            if dl2 > time.time() + 0.4:
                MI = np.eye(d)
                Xp = _lp_polish_M(X, MI, dl2, r0=4e-4, cap_p=2000, max_solves=60)
            else:
                Xp = X
        except Exception:
            Xp = X
        vp = _max_cos(Xp)
        if vp < v_best:
            x_best = Xp.copy()
            v_best = vp
        elif v_end < v_best:
            x_best = X.copy()
            v_best = v_end
    # final squeeze of global best with remaining time
    remain = deadline_final - time.time()
    if remain > 12:
        try:
            MI = np.eye(d)
            Y = _run_lbfgs_M(x_best, MI, 3000.0, 100)
            if _max_cos(Y) < v_best:
                x_best = Y
                v_best = _max_cos(Y)
        except Exception:
            pass
    remain = deadline_final - time.time()
    if remain > 9:
        try:
            MI = np.eye(d)
            Y = _run_lbfgs_M(x_best, MI, 9000.0, 80)
            if _max_cos(Y) < v_best:
                x_best = Y
                v_best = _max_cos(Y)
        except Exception:
            pass
    remain = deadline_final - time.time()
    if remain > 3.5:
        try:
            MI = np.eye(d)
            x_best = _lp_polish_M(x_best, MI, deadline_final - 0.8, r0=5e-4, cap_p=2200, max_solves=400)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
