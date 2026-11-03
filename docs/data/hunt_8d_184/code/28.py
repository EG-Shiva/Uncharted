import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-300)

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_B(X0):
    n, d = X0.shape
    e1 = np.zeros(d); e1[0] = 1.0
    W = X0 - e1[None, :]
    nw = np.linalg.norm(W, axis=1)
    tiny = nw < 1e-12
    w = np.empty_like(W)
    nz = ~tiny
    w[nz] = W[nz] / nw[nz, None]
    w[tiny] = 0.0
    # E template (8,7): rows 1..7 = eye
    E = np.zeros((d, d-1))
    E[1:, :] = np.eye(d-1)
    w1 = w[:, 1:]  # n,7
    B = E[None, :, :] - 2.0 * w[:, :, None] * w1[:, None, :]
    if np.any(tiny):
        # for x==e1, basis = e2..e8
        B[tiny] = E[None, :, :]
    return B

def _lp_polish(x, budget, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1.5e-3
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    t_start = time.time()
    while time.time() - t_start < budget:
        if time.time() > deadline - 1.5:
            break
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 2e-9)
        I, J = iu[0][sel], iu[1][sel]
        cs = c[sel]
        p = len(I)
        if p == 0:
            break
        if p > 2200:
            idx = np.argpartition(-cs, 2200)[:2200]
            I, J, cs = I[idx], J[idx], cs[idx]
            p = 2200
        nv = n * d + 1
        rows = np.concatenate([np.repeat(np.arange(p), d), np.repeat(np.arange(p), d), np.arange(p)])
        cols = np.concatenate([np.repeat(I * d, d) + np.tile(ar, p),
                               np.repeat(J * d, d) + np.tile(ar, p),
                               np.full(p, n * d)])
        vals = np.concatenate([x[J].ravel(), x[I].ravel(), -np.ones(p)])
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
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
        y = _norm(x + res.x[:-1].reshape(n, d))
        v = _max_cos(y)
        if v < best:
            best = float(v)
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 3e-7:
                break
    return xb

class _TimeUp(Exception):
    pass

def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    CUTOFF = t0 + 155.0
    SEARCH_END = t0 + 150.0
    n, d = 184, 8
    rng = np.random.default_rng(12345)

    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.asarray(np.load(f), dtype=np.float64).reshape(n, d)
            c = _norm(c)
            cands.append((_max_cos(c), c))
        except Exception:
            pass
    if cands:
        cands.sort(key=lambda t: t[0])
        gval, gbest = cands[0][0], cands[0][1].copy()
        x_best0 = cands[0][1].copy()
        x_rec0 = None
        for v, c in cands:
            pass
        # keep record separate if exists
        try:
            x_rec0 = _norm(np.asarray(np.load("record.npy"), dtype=np.float64).reshape(n, d))
        except Exception:
            x_rec0 = gbest.copy()
    else:
        gbest = _norm(rng.standard_normal(size=(n, d)))
        gval = _max_cos(gbest)
        x_best0 = gbest.copy()
        x_rec0 = gbest.copy()

    # build diverse starts
    starts = []
    starts.append(("best", x_best0.copy()))
    starts.append(("record", x_rec0.copy()))
    # record-perturbed / best-perturbed
    sigmas = [0.02, 0.05, 0.12, 0.22]
    for k, sg in enumerate(sigmas):
        starts.append((f"best_p{k}", _norm(x_best0 + sg * rng.standard_normal(size=(n, d)))))
    for k, sg in enumerate([0.05, 0.12, 0.25]):
        starts.append((f"rec_p{k}", _norm(x_rec0 + sg * rng.standard_normal(size=(n, d)))))
    # random starts
    for k in range(4):
        starts.append((f"rand{k}", _norm(rng.standard_normal(size=(n, d)))))
    # Sobol starts (low-discrepancy on sphere via ndtri)
    try:
        from scipy.stats import qmc
        from scipy.special import ndtri
        for sk in [777, 1234]:
            try:
                sampler = qmc.Sobol(d=d, scramble=True, seed=sk)
                pts = sampler.random_base2(m=8)  # 256 x 8 in [0,1]
                pts = pts[:n]
                pts = np.clip(pts, 1e-6, 1 - 1e-6)
                z = ndtri(pts)
                # center? just normalize
                z = _norm(z)
                starts.append((f"sobol{sk}", z))
            except Exception:
                pass
    except Exception:
        pass
    # extra random with different seed stream
    rng2 = np.random.default_rng(999)
    for k in range(2):
        starts.append((f"randB{k}", _norm(rng2.standard_normal(size=(n, d)))))

    alphas = [80.0, 200.0, 500.0]
    betas = [3.0, 3.0, 3.0]
    maxiters = [90, 80, 70]

    NB = n * (d - 1)
    bnds = [(-0.5, 0.5)] * NB

    for sname, Xinit in starts:
        if time.time() > SEARCH_END:
            break
        if time.time() > CUTOFF - 2.0:
            break
        st = time.time()
        Xcur = Xinit.copy()
        try:
            for stage, (alpha, beta, mit) in enumerate(zip(alphas, betas, maxiters)):
                if time.time() > SEARCH_END:
                    raise _TimeUp()
                if time.time() - st > 8.0:
                    break
                X0 = Xcur.copy()
                B = _build_B(X0)  # n,8,7
                # closure
                def fun(Uflat, _B=B, _X0=X0, _al=alpha, _be=beta):
                    if time.time() > CUTOFF:
                        raise _TimeUp()
                    U = Uflat.reshape(n, d - 1)
                    # s = X0 + B u
                    s = _X0 + np.einsum('ija,ia->ij', _B, U)
                    r = np.sqrt(1.0 + np.sum(U * U, axis=1))
                    X = s / r[:, None]
                    C = X @ X.T
                    np.fill_diagonal(C, -2.0)
                    m = float(np.max(C))
                    E = np.exp(_al * (C - m))
                    np.fill_diagonal(E, 0.0)
                    Shalf = float(np.sum(E) * 0.5) + 1e-300
                    L = m + np.log(Shalf) / _al
                    mi = np.max(C, axis=1)
                    Ei = np.exp(_al * (C - mi[:, None]))
                    np.fill_diagonal(Ei, 0.0)
                    Si = np.sum(Ei, axis=1) + 1e-300
                    Mi = mi + np.log(Si) / _al
                    Mbar = float(np.mean(Mi))
                    dev = Mi - Mbar
                    Var = float(np.mean(dev * dev))
                    F = L + _be * Var
                    # grad
                    GW = E / Shalf
                    P = Ei / Si[:, None]
                    dd = 2.0 * dev / n
                    A = dd[:, None] * P
                    G = GW + _be * (A + A.T)
                    dFdx = G @ X
                    dot = np.sum(dFdx * X, axis=1)
                    tang = (dFdx - dot[:, None] * X) / r[:, None]
                    gU = np.einsum('ija,ij->ia', _B, tang)
                    return F, gU.ravel()
                U0 = np.zeros(NB, dtype=np.float64)
                try:
                    res = minimize(fun, U0, method="L-BFGS-B", jac=True, bounds=bnds,
                                   options={"maxiter": mit, "maxfun": mit * 3 + 20,
                                            "ftol": 1e-14, "gtol": 1e-9, "maxcor": 10})
                    Uopt = res.x.reshape(n, d - 1)
                except _TimeUp:
                    raise
                except Exception:
                    # on failure keep current
                    continue
                # rebuild Xcur
                s = X0 + np.einsum('ija,ia->ij', B, Uopt)
                r = np.sqrt(1.0 + np.sum(Uopt * Uopt, axis=1))
                Xcur = s / r[:, None]
                Xcur = _norm(Xcur)
                v = _max_cos(Xcur)
                if stage == 0 and v > 0.502:
                    break
                if v < gval:
                    gval = float(v)
                    gbest = Xcur.copy()
                # rebuild bases next loop (Xcur becomes new X0)
            # end stages
        except _TimeUp:
            break
        # continue to next start
    # final single LP polish on best
    remain = DEADLINE - time.time() - 1.0
    if remain > 6.0:
        budget = min(remain, 23.0)
        try:
            gbest = _lp_polish(gbest, budget, DEADLINE)
        except Exception:
            pass
    gbest = _norm(gbest)
    if not np.all(np.isfinite(gbest)):
        gbest = _norm(x_best0)
    return gbest
