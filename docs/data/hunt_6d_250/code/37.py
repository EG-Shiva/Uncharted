import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam_lse(X0, steps, b0, b1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1m = 0.9
    b2m = 0.999
    eps = 1e-8
    t = 0
    lb0 = np.log(b0)
    lb1 = np.log(b1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = float(np.max(G))
        G -= cm
        G *= beta
        np.clip(G, -30.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G))
        if S < 1e-100 or not np.isfinite(S):
            continue
        GX = G @ X
        GX *= (2.0 / S)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GX -= dot * X
        t += 1
        m *= b1m
        m += (1.0 - b1m) * GX
        v *= b2m
        v += (1.0 - b2m) * (GX * GX)
        c1 = 1.0 - (b1m ** t)
        c2 = 1.0 - (b2m ** t)
        denom = np.sqrt(v / c2) + eps
        step = (lr / c1) * m / denom
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 5.0 * lr + 1e-12
        sc = np.minimum(1.0, cap / sn)
        step *= sc
        X -= step
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
        md = np.sum(m * X, axis=1, keepdims=True)
        m -= md * X
    return X

def _short_polish(Xtry):
    # beta=4000 then 15000 as assigned
    try:
        Y = _adam_lse(Xtry, 260, 4000.0, 4000.0, 0.0025, 0.0007)
        Y = _adam_lse(Y, 220, 15000.0, 15000.0, 0.0012, 0.00035)
    except Exception:
        Y = Xtry
    return Y

def _tangent_bases(X):
    n, d = X.shape
    dd = d - 1
    B = np.empty((n, d, dd), dtype=float)
    for i in range(n):
        xi = X[i]
        k = int(np.argmax(np.abs(xi)))
        M = np.zeros((d, dd), dtype=float)
        c = 0
        for j in range(d):
            if j == k:
                continue
            e = np.zeros(d, dtype=float)
            e[j] = 1.0
            e = e - float(e @ xi) * xi
            M[:, c] = e
            c += 1
        Q, _ = np.linalg.qr(M)
        B[i] = Q
    return B

def _floppy_vectors(X, B, slack=5e-6, k=32):
    n, d = X.shape
    dd = d - 1
    N = n * dd
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    Mmax = float(np.max(G))
    iu0 = np.triu_indices(n, 1)
    vals = G[iu0]
    sel = vals > Mmax - slack
    I = iu0[0][sel]
    J = iu0[1][sel]
    m = len(I)
    if m < 10:
        # fallback larger slack
        order = np.argsort(-vals)
        take = min(800, len(vals))
        I = iu0[0][order[:take]]
        J = iu0[1][order[:take]]
        m = take
    # cap size for eig cost
    if m > 2500:
        # keep tightest 2500
        sv = vals[sel]
        idx = np.argsort(-sv)[:2500]
        I = I[idx]
        J = J[idx]
        m = 2500
    kk = int(min(k, N - 1, m))
    if kk < 8:
        kk = min(N - 1, m)
    A = np.zeros((m, N), dtype=float)
    for r in range(m):
        i = int(I[r]); j = int(J[r])
        # B_i^T x_j , B_j^T x_i
        ai = B[i].T @ X[j]
        aj = B[j].T @ X[i]
        A[r, i * dd:(i + 1) * dd] = ai
        A[r, j * dd:(j + 1) * dd] = aj
    # smallest right singular vectors = smallest eig of A^T A
    try:
        C = A.T @ A
        # tiny ridge for stability
        # eigh ascending
        w, V = np.linalg.eigh(C)
        # V columns; take kk smallest
        Vf = V[:, :kk]
    except Exception:
        rng = np.random.default_rng(0)
        Vf = rng.standard_normal((N, kk))
        # orthonormalize
        Vf, _ = np.linalg.qr(Vf)
    return Vf, m, Mmax

def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    HOP_END = t0 + 128.0
    rng = np.random.default_rng(12345)
    n, d = 250, 6
    dd = d - 1
    # load incumbent
    best = None
    best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                if np.all(np.isfinite(a)):
                    vv = _max_cos(a)
                    if vv < best_val:
                        best_val = vv
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    incumbent = best.copy()
    inc_val = float(best_val)

    # initial floppy basis
    try:
        B = _tangent_bases(incumbent)
        Vf, mnum, _ = _floppy_vectors(incumbent, B, slack=5e-6, k=32)
        K = Vf.shape[1]
    except Exception:
        B = _tangent_bases(incumbent)
        Vf = None
        K = 0

    step_sizes = [0.02, 0.05, 0.1]
    trial = 0
    # if svds path wanted, our eigh vectors are equivalent; keep loop as assigned
    while time.time() < HOP_END and trial < 500:
        trial += 1
        if time.time() > HARD - 25.0:
            break
        # refresh floppy basis periodically or after improvement
        if Vf is None or (trial % 12 == 1 and trial > 1):
            try:
                B = _tangent_bases(incumbent)
                Vf, mnum, _ = _floppy_vectors(incumbent, B, slack=5e-6, k=32)
                K = Vf.shape[1]
            except Exception:
                pass
            if Vf is None:
                # random kick fallback
                continue
        # random combination of smallest singular vectors
        try:
            coeff = rng.standard_normal(K)
            cn = float(np.linalg.norm(coeff))
            if cn < 1e-12:
                continue
            coeff /= cn
            vdir = Vf @ coeff  # (1250,)
            Vt = vdir.reshape(n, dd)
            # randomize step order
            order = [0, 1, 2]
            # shuffle deterministically via rng
            rp = rng.permutation(3)
            for si in rp:
                s = step_sizes[int(si)]
                if time.time() > HOP_END:
                    break
                # large retracted step: X + B*Vt*s
                Delta = np.einsum('nde,ne->nd', B, Vt) * s
                Xt = incumbent + Delta
                nr = np.linalg.norm(Xt, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                Xt /= nr
                vt = _max_cos(Xt)
                if vt <= inc_val + 2e-4:
                    Y = _short_polish(Xt)
                    # renormalize
                    nr2 = np.linalg.norm(Y, axis=1, keepdims=True)
                    nr2[nr2 == 0] = 1.0
                    Y /= nr2
                    vp = _max_cos(Y)
                    if vp < best_val - 1e-12:
                        best = Y.copy()
                        best_val = float(vp)
                        incumbent = Y.copy()
                        inc_val = float(vp)
                        # refresh basis immediately to travel along new plateau
                        try:
                            B = _tangent_bases(incumbent)
                            Vf, mnum, _ = _floppy_vectors(incumbent, B, slack=5e-6, k=32)
                            K = Vf.shape[1]
                        except Exception:
                            pass
                        break  # new incumbent, next direction
                    # else return to incumbent (do nothing), try next step size
                # if not accepted, try next step size with same direction
        except Exception:
            continue

    # final squeeze of best with a bit longer polish if time remains
    try:
        if time.time() < HARD - 12.0:
            Y = _adam_lse(best, 300, 6000.0, 18000.0, 0.0012, 0.0004)
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            Y /= nr
            vy = _max_cos(Y)
            if vy < best_val:
                best = Y
                best_val = vy
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
