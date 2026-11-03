import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-300
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

class _Abort(Exception):
    pass

_ZCACHE = None
def _get_Z():
    global _ZCACHE
    if _ZCACHE is None:
        rng = np.random.default_rng(12345)
        _ZCACHE = rng.standard_normal(size=(8, 7))
    return _ZCACHE

def _build_B(x0, Z):
    n, d = x0.shape
    k = Z.shape[1]
    A = x0 @ Z  # n,k
    W = Z[None, :, :] - x0[:, :, None] * A[:, None, :]
    B = np.empty_like(W)
    for i in range(n):
        Q, _ = np.linalg.qr(W[i])
        B[i] = Q
    return B

def _equimax_stage(x0, B, alpha, lam, bound, t_start, t_budget, t_global_deadline):
    n, d = x0.shape
    k = B.shape[2]
    Nv = n * k
    bnds = [(-bound, bound)] * Nv
    # pre-allocate not needed
    def fun_grad(uflat):
        # time cutoffs -> abort L-BFGS-B
        now = time.time()
        if now > t_global_deadline or (now - t_start) > t_budget:
            raise _Abort()
        U = uflat.reshape(n, k)
        Delta = np.einsum('ndk,nk->nd', B, U)
        V = x0 + Delta
        nrm = np.linalg.norm(V, axis=1)
        # guard degenerate
        nrm = np.maximum(nrm, 1e-12)
        X = V / nrm[:, None]
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        # global LSE
        m = float(np.max(G))
        # G-m <=0
        Eg = np.exp(alpha * (G - m))
        np.fill_diagonal(Eg, 0.0)
        Ssum = float(np.sum(Eg))
        if not np.isfinite(Ssum) or Ssum <= 0:
            Ssum = 1e-300
        Su = Ssum * 0.5
        L = m + np.log(Su) / alpha
        Wmat = Eg / Su
        gradL_X = Wmat @ X
        # per-point smooth maxima (row-stable)
        row_m = np.max(G, axis=1)  # n,
        # row-centered exp
        Er = np.exp(alpha * (G - row_m[:, None]))
        np.fill_diagonal(Er, 0.0)
        Sr = np.sum(Er, axis=1) + 1e-300  # n,
        M = row_m + np.log(Sr) / alpha
        mbar = float(np.mean(M))
        diff = M - mbar
        var = float(np.mean(diff * diff))
        a = 2.0 * diff / n  # n,
        P = Er / Sr[:, None]  # row stochastic
        Amat = a[:, None] * P
        Vsym = Amat + Amat.T
        gradV_X = Vsym @ X
        J = L + lam * var
        grad_X = gradL_X + lam * gradV_X
        dot = np.sum(grad_X * X, axis=1)
        Gt = grad_X - dot[:, None] * X
        dV = Gt / nrm[:, None]
        grad_U = np.einsum('ndk,nd->nk', B, dV)
        if not np.all(np.isfinite(grad_U)):
            grad_U = np.nan_to_num(grad_U, nan=0.0, posinf=0.0, neginf=0.0)
            J = float(np.nan_to_num(J, nan=1e6, posinf=1e6, neginf=1e6))
        return float(J), grad_U.ravel().astype(np.float64)
    u0 = np.zeros(Nv, dtype=np.float64)
    try:
        res = minimize(fun_grad, u0, method="L-BFGS-B", jac=True,
                       bounds=bnds,
                       options={"maxiter": 150, "maxfun": 220, "ftol": 1e-12, "gtol": 1e-7})
        uopt = res.x
    except _Abort:
        # return current best guess from last evaluation? use zeros-> fallback
        # try to extract from exception path: just return x0
        return x0, _max_cos(x0), True
    U = uopt.reshape(n, k)
    Delta = np.einsum('ndk,nk->nd', B, U)
    V = x0 + Delta
    nrm = np.linalg.norm(V, axis=1, keepdims=True) + 1e-300
    Xnew = V / nrm
    return Xnew, _max_cos(Xnew), False

def _equimax_opt(x_init, alphas, lam, bound, t_start, t_budget, t_global_deadline, Z):
    x0 = _norm(x_init.copy())
    aborted = False
    for si, alpha in enumerate(alphas):
        B = _build_B(x0, Z)
        # tighter bound for large alpha to stay stable
        b = bound if si == 0 else bound * 0.8
        x0, mc, ab = _equimax_stage(x0, B, alpha, lam, b, t_start, t_budget, t_global_deadline)
        if ab:
            aborted = True
            break
        if si == 0 and mc > 0.502:
            return None, mc, False
    return x0, _max_cos(x0), aborted

def _lp_polish(x, seconds, t_hard=None):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    pcap = 1800
    while time.time() - start < seconds:
        if t_hard is not None and time.time() > t_hard:
            break
        if radius < 1e-14:
            break
        if time.time() - start > seconds - 0.4:
            break
        G = x @ x.T
        cf = G[iu0]
        m = float(cf.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cf > thr
        I = iu0[0][sel]; J = iu0[1][sel]; cs = cf[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > pcap:
            idx = np.argpartition(-cs, pcap)[:pcap]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = pcap
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    HARD = t0 + 178.0
    SEARCH_DEADLINE = t0 + 152.0  # ~ global 155 cutoff
    n, d = 184, 8
    rng = np.random.default_rng(0)
    Z = _get_Z()
    # load bases
    cands0 = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64)
            if c.shape == (n, d):
                c = _norm(c)
                cands0.append(c)
        except Exception:
            pass
    if len(cands0) == 0:
        cands0.append(_norm(rng.standard_normal(size=(n, d))))
    base_best = min(cands0, key=lambda c: _max_cos(c))
    gval = _max_cos(base_best)
    gbest = base_best.copy()
    # try Sobol if available for diverse starts
    sobol_pts = None
    try:
        from scipy.stats import qmc, norm
        sampler = qmc.Sobol(d=8, scramble=True, seed=7)
        # generate e.g. 256 points in [0,1]^8 -> normal
        u = sampler.random_base2(m=8)  # 256 x 8
        u = np.clip(u, 1e-9, 1 - 1e-9)
        sobol_pts = norm.ppf(u)  # 256 x 8 gaussian-like
    except Exception:
        sobol_pts = None

    sigmas = [0.02, 0.05, 0.08, 0.12, 0.03, 0.06]
    alphas_list = [80, 200, 500]
    lam = 8.0
    trial = 0
    # budget per start ~ up to 8s, first stage early exit
    while time.time() < SEARCH_DEADLINE - 1.0:
        if time.time() > HARD - 30:
            break
        trial += 1
        mode = trial % 10
        if sobol_pts is not None and mode in (8, 9):
            # Sobol start: take random block of 184 rows with random rotation/shift
            idx0 = int(rng.integers(0, len(sobol_pts) - 1))
            rows = []
            for k in range(n):
                rows.append(sobol_pts[(idx0 + k * 37) % len(sobol_pts)])
            x_init = np.array(rows, dtype=np.float64)
            # random orthogonal mix to diversify (fixed random matrix)
            x_init = x_init + 0.3 * rng.standard_normal(size=(n, d))
            x_init = _norm(x_init)
        elif mode in (0, 1, 2, 5, 6):
            # record/best perturbed
            base = base_best if (trial % 2 == 0) else cands0[trial % len(cands0)]
            sig = sigmas[(trial // 2) % len(sigmas)]
            x_init = base + sig * rng.standard_normal(size=(n, d))
            x_init = _norm(x_init)
        else:
            # pure random
            x_init = _norm(rng.standard_normal(size=(n, d)))
        t_start = time.time()
        t_budget = 8.0
        # adjust budget if near deadline
        if SEARCH_DEADLINE - t_start < t_budget:
            t_budget = max(1.0, SEARCH_DEADLINE - t_start - 0.2)
        try:
            xopt, mc, aborted = _equimax_opt(x_init, alphas_list, lam, 0.35, t_start, t_budget, SEARCH_DEADLINE, Z)
        except _Abort:
            continue
        except Exception:
            continue
        if xopt is None:
            continue
        if aborted and mc >= gval:
            continue
        # keep best
        if mc < gval - 1e-12:
            # distinctness not critical; accept
            gval = mc
            gbest = _norm(xopt.copy())
            base_best = gbest.copy()
            # early exit if clearly record-beating by margin? continue to improve
        if time.time() > SEARCH_DEADLINE:
            break
    # single final LP polish 20-25s on best
    remain = HARD - time.time() - 1.0
    lp_sec = min(24.0, max(5.0, remain))
    # only polish if time allows
    try:
        if lp_sec >= 4.0:
            gpol = _lp_polish(gbest, lp_sec, t_hard=HARD - 0.5)
            vpol = _max_cos(gpol)
            if vpol < gval:
                gbest = gpol
                gval = vpol
    except Exception:
        pass
    gbest = _norm(gbest)
    # tiny shrink guard (no-op, keep unit rows)
    return gbest
