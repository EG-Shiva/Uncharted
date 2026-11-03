import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X):
    n, d = X.shape
    B = np.empty((n, d, d-1), dtype=np.float64)
    for i in range(n):
        a = X[i]
        j = int(np.argmin(np.abs(a)))
        idx = [k for k in range(d) if k != j]
        E = np.zeros((d, d-1), dtype=np.float64)
        for col, k in enumerate(idx):
            E[k, col] = 1.0
        Brow = E - a[:, None] * a[idx][None, :]
        Q, _ = np.linalg.qr(Brow)
        # Q is 6x5 orthonormal, spans tangent
        B[i] = Q[:, :d-1]
    return B

def _smooth_opt(X0, beta, maxiter, maxcor=10):
    n, d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.empty_like(D)
        np.exp(D, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if S <= 0 or not np.isfinite(S):
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-12,
                                "maxfun": maxiter * 5, "maxcor": maxcor})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return Y

def _floppy_vectors(X, slack=5e-6, k=32):
    n, d = X.shape
    N = n * (d-1)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    iu = np.triu_indices(n, 1)
    vals = G[iu]
    sel = (m - vals) < slack
    cnt = int(np.count_nonzero(sel))
    if cnt < 50:
        # widen to at least 500 nearest
        order = np.argsort(vals)[-500:]
        I = iu[0][order]; J = iu[1][order]
    else:
        if cnt > 3000:
            # keep tightest 3000
            idx = np.where(sel)[0]
            vv = vals[idx]
            order = np.argsort(vv)[-3000:]
            ii = idx[order]
            I = iu[0][ii]; J = iu[1][ii]
        else:
            I = iu[0][sel]; J = iu[1][sel]
    p = len(I)
    B = _tangent_bases(X)
    A = np.zeros((p, N), dtype=np.float64)
    # fill rows: loop (p~1300, cheap)
    for t in range(p):
        i = int(I[t]); j = int(J[t])
        # B_i^T x_j  (5,)
        vi = B[i].T @ X[j]
        vj = B[j].T @ X[i]
        A[t, i*5:(i+1)*5] = vi
        A[t, j*5:(j+1)*5] = vj
    # smallest singular vectors = smallest eigenvectors of A^T A
    try:
        ATA = A.T @ A
        # symmetrize
        ATA = 0.5 * (ATA + ATA.T)
        # add tiny diagonal for stability? no, keep exact
        w, V = np.linalg.eigh(ATA)
        # w ascending, V columns
        kk = min(k, N)
        Vs = V[:, :kk]  # N x kk smallest
        return Vs, B, float(m), int(p)
    except Exception:
        return None, B, float(m), int(p)

def solve():
    t_start = time.time()
    BUDGET = 168.0
    def rem():
        return BUDGET - (time.time() - t_start)
    rng = np.random.default_rng(0)
    n, d = 250, 6
    best = None
    best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = z / np.linalg.norm(z, axis=1, keepdims=True)
                v = _max_cos(z)
                if v < best_val:
                    best_val = v
                    best = z.copy()
        except Exception:
            continue
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    incumbent = best.copy()
    inc_val = best_val

    Vs = None
    Bcur = None
    # time for floppy loop: ~120s, leave ~15s for final polish
    flop_deadline = t_start + 128.0
    # ensure we have Vs computed once
    nfail = 0
    step_sizes = [0.02, 0.05, 0.1]
    kvec = 32

    while time.time() < flop_deadline and rem() > 18:
        if Vs is None:
            try:
                Vs, Bcur, _, _ = _floppy_vectors(incumbent, slack=5e-6, k=kvec)
            except Exception:
                Vs = None
                Bcur = None
            if Vs is None:
                # fallback: random tangent step
                Bcur = _tangent_bases(incumbent)
                # use identity (random directions)
                Vs = None
                nfail += 1
                if nfail > 3:
                    break
                continue
        # draw random combination of floppy vectors
        kk = Vs.shape[1]
        # several draws per Vs before recompute (Vs reused)
        draws = 8
        improved = False
        for _ in range(draws):
            if time.time() >= flop_deadline or rem() <= 18:
                break
            if Vs is not None:
                c = rng.standard_normal(kk)
                cn = np.linalg.norm(c)
                if cn < 1e-12:
                    continue
                c /= cn
                u = Vs @ c  # N,
            else:
                u = rng.standard_normal(n*5)
                u /= max(np.linalg.norm(u), 1e-12)
            U = u.reshape(n, 5)
            # DX = B_i U_i
            # einsum: DX[i,k] = sum_l B[i,k,l] U[i,l]
            DX = np.einsum('ikl,il->ik', Bcur, U)
            # DX has total Fro norm 1 (since B orthonormal and u norm 1)
            for s in step_sizes:
                if time.time() >= flop_deadline or rem() <= 18:
                    break
                Xt = incumbent + s * DX
                Xt /= np.maximum(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-12)
                v0 = _max_cos(Xt)
                if v0 > inc_val + 2e-4:
                    continue
                # re-polish short beta=4000 then 15000
                try:
                    Xp = _smooth_opt(Xt, 4000.0, 80, maxcor=8)
                except Exception:
                    continue
                if rem() < 16:
                    v = _max_cos(Xp)
                else:
                    try:
                        Xp2 = _smooth_opt(Xp, 15000.0, 80, maxcor=8)
                    except Exception:
                        Xp2 = Xp
                    v = _max_cos(Xp2)
                    Xp = Xp2
                if v < inc_val - 1e-12:
                    incumbent = Xp.copy()
                    inc_val = v
                    if v < best_val:
                        best = Xp.copy()
                        best_val = v
                    Vs = None  # recompute modes at new topology
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # try fresh random combos with same Vs for a while, then refresh Vs with different k/slack to get variety
            # to avoid stagnation, occasionally recompute with slightly different slack or jitter incumbent basis
            # count failures; every 4 outer loops force recompute with jittered slack
            nfail += 1
            if nfail % 5 == 0:
                Vs = None
                # vary k slightly
                kvec = int(rng.choice([25, 32, 40]))
            # also if raw max stuck, add small random tangent jitter to Bcur reuse (keep Vs)
            # continue with new random combos (Vs kept) unless reset above
            if nfail > 200:
                break
        else:
            nfail = 0

    # final quick polish of best if time remains (short high-beta + tiny LP-free)
    if rem() > 8:
        try:
            Xp = _smooth_opt(best, 15000.0, 120, maxcor=10)
            v = _max_cos(Xp)
            if v < best_val:
                best = Xp
                best_val = v
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
