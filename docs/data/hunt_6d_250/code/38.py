import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_B(X0):
    n, d = X0.shape
    k = d - 1
    B = np.empty((n, d, k), dtype=np.float64)
    for i in range(n):
        # SVD of 1x6 row: first right-singular vector = x_i, rest = tangent basis
        _, _, Vh = np.linalg.svd(X0[i:i+1, :], full_matrices=True)
        B[i] = Vh[1:, :].T
    return B

def _select_active(X, slack, cap=4000):
    n = X.shape[0]
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    gv = G[iu]
    M = float(np.max(gv))
    # pairs with M - g <= slack  <=> g >= M - slack
    thr = M - slack
    mask = gv >= thr
    idx = np.where(mask)[0]
    if idx.size == 0:
        # at least top cap
        idx = np.argsort(-gv)[:min(cap, gv.size)]
        I = iu[0][idx]; J = iu[1][idx]
        return I, J, M
    if idx.size > cap:
        # keep tightest cap
        gsel = gv[idx]
        order = np.argsort(-gsel)[:cap]
        idx = idx[order]
    I = iu[0][idx].astype(np.int64)
    J = iu[1][idx].astype(np.int64)
    return I, J, M

def _equalize_once(X0, B, I, J, M0, radius, mu, maxiter, lam=1e-7):
    n, d = X0.shape
    k5 = d - 1
    p = I.shape[0]
    if p == 0:
        return X0, False
    # closure for L-BFGS-B on v=[u(flattened), t]
    Nvars = n * k5
    def fg(v):
        u = v[:-1].reshape(n, k5)
        t = float(v[-1])
        # retraction: Y = X0 + B*u, Xn = Y/|Y|
        # einsum ndk,nk -> nd
        Y = X0 + np.einsum('ndk,nk->nd', B, u, optimize=False)
        nrm = np.sqrt(np.sum(Y * Y, axis=1))
        # guard (should stay ~1)
        nrm = np.maximum(nrm, 1e-12)
        Xn = Y / nrm[:, None]
        XI = Xn[I]
        XJ = Xn[J]
        f = np.sum(XI * XJ, axis=1)
        diff = f - t
        # relu squared sum
        pos = diff > 0.0
        relu = np.where(pos, diff, 0.0)
        S = float(np.dot(relu, relu))
        usq = float(np.dot(u.ravel(), u.ravel()))
        L = t + mu * S + lam * usq
        # grads
        sum_relu = float(np.sum(relu))
        grad_t = 1.0 - 2.0 * mu * sum_relu
        grad_u = (2.0 * lam) * u
        if np.any(pos):
            R = 2.0 * mu * relu  # (p,)
            nrmI = nrm[I]
            nrmJ = nrm[J]
            # (xj - xi*f)/nrmI
            AI = (XJ - XI * f[:, None]) / nrmI[:, None]
            AJ = (XI - XJ * f[:, None]) / nrmJ[:, None]
            BI = B[I]  # p,6,5
            BJ = B[J]
            gI = np.einsum('pdk,pd->pk', BI, AI)
            gJ = np.einsum('pdk,pd->pk', BJ, AJ)
            gI *= R[:, None]
            gJ *= R[:, None]
            np.add.at(grad_u, I, gI)
            np.add.at(grad_u, J, gJ)
        grad = np.empty_like(v)
        grad[:-1] = grad_u.ravel()
        grad[-1] = grad_t
        return L, grad
    v0 = np.zeros(Nvars + 1, dtype=np.float64)
    v0[-1] = M0
    bounds = [(-radius, radius)] * Nvars + [(M0 - 0.002, M0)]
    try:
        res = minimize(fg, v0, jac=True, method="L-BFGS-B",
                       bounds=bounds,
                       options={"maxiter": maxiter, "ftol": 1e-20,
                                "gtol": 1e-9, "maxfun": maxiter * 4,
                                "maxcor": 12})
        v = res.x
    except Exception:
        return X0, False
    u = v[:-1].reshape(n, k5)
    Y = X0 + np.einsum('ndk,nk->nd', B, u, optimize=False)
    nrm = np.linalg.norm(Y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    Xn = Y / nrm
    return Xn, True

def solve():
    t_start = time.time()
    BUDGET = 168.0
    def remaining():
        return BUDGET - (time.time() - t_start)
    n, d = 250, 6
    best = None
    best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                zn = np.linalg.norm(z, axis=1, keepdims=True)
                zn = np.maximum(zn, 1e-12)
                z = z / zn
                if np.all(np.isfinite(z)):
                    v = _max_cos(z)
                    if v < best_val:
                        best_val = v
                        best = z.copy()
        except Exception:
            continue
    if best is None:
        rng = np.random.default_rng(0)
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    # threshold ladder 1e-4 -> 1e-6 with shrinking trust radius
    ladder = [
        (1e-4, 4.0e-3, 250),
        (3e-5, 3.0e-3, 300),
        (1e-5, 2.0e-3, 300),
        (3e-6, 1.5e-3, 350),
        (1e-6, 1.0e-3, 350),
    ]
    mus = [2.0e6, 2.0e7]
    for stage, (slack, radius, maxiter) in enumerate(ladder):
        if remaining() < 18.0:
            break
        # re-extract active set on current cur
        try:
            I, J, M0 = _select_active(cur, slack, cap=4000)
        except Exception:
            continue
        # report size implicitly by adapting maxiter if huge
        if I.size < 50:
            continue
        # cap maxiter by remaining time
        if remaining() < 30:
            maxiter = min(maxiter, 150)
        # build tangent bases (null space of each x_i)
        try:
            B = _build_B(cur)
        except Exception:
            continue
        Xw = cur.copy()
        ok_any = False
        for mi, mu in enumerate(mus):
            if remaining() < 12.0:
                break
            mit = maxiter if mi == 0 else max(150, maxiter // 2)
            try:
                Xn, ok = _equalize_once(Xw, B, I, J, float(np.max((cur @ cur.T)[np.triu_indices(n,1)])) if False else M0, radius, mu, mit)
            except Exception:
                break
            if not ok:
                break
            # evaluate global max
            try:
                v = _max_cos(Xn)
            except Exception:
                break
            # accept warm start if it does not explode (allow small rise for higher mu tightening)
            # for intermediate mu allow rise up to 2e-6, final must beat cur
            if mi < len(mus) - 1:
                # move warm start forward if t-like max on active improved or stable
                # check active max
                Gact = np.sum(Xn[I] * Xn[J], axis=1)
                Ma = float(np.max(Gact))
                Gold = np.sum(Xw[I] * Xw[J], axis=1)
                Mo = float(np.max(Gold))
                if Ma < Mo + 2e-6 and v < cur_val + 5e-5:
                    Xw = Xn
                    ok_any = True
                    # rebuild B? keep fixed B for this stage (fixed tangent coords per spec)
                    # re-extract? keep same I,J for penalty consistency
                else:
                    # keep old warm start
                    pass
            else:
                # final mu: keep only if global improves
                if v < cur_val - 1e-12 and np.all(np.isfinite(Xn)):
                    cur = Xn.copy()
                    cur_val = v
                    if v < best_val:
                        best = Xn.copy()
                        best_val = v
                    ok_any = True
                elif v < cur_val + 1e-7:
                    # no improvement but very close: try to keep tighter tying if active spread shrank
                    Gact = np.sum(Xn[I] * Xn[J], axis=1)
                    Gold = np.sum(cur[I] * cur[J], axis=1)
                    # spread = max - min? or max - median
                    sp_n = float(np.max(Gact) - np.min(Gact))
                    sp_o = float(np.max(Gold) - np.min(Gold))
                    if sp_n < sp_o * 0.5 and v < cur_val + 5e-7:
                        # accept as new cur for next stage even without strict max gain (equalisation)
                        cur = Xn.copy()
                        cur_val = v
                        ok_any = True
                break
        # after each stage, if we improved, re-extract in next stage loop (rebuild B)
        # if time remains and we tied many pairs, continue ladder
        if remaining() < 10.0:
            break
    # keep result only if it beats initial best.npy value; best already tracks that
    out = best.copy()
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out
