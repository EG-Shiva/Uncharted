import time
import numpy as np
from scipy.optimize import minimize, linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _radical_inverse(i, base):
    f = 1.0 / base
    r = 0.0
    while i > 0:
        r += f * (i % base)
        i //= base
        f /= base
    return r

_PRIMES = [2,3,5,7,11,13,17,19,23,29,31,37]

def _halton_gaussian(n, d, offset, rng_shift=0.0):
    # halton in [0,1]^d then Box-Muller to gaussian-like
    U = np.empty((n, d))
    for k in range(n):
        idx = k + 1 + offset
        for j, b in enumerate(_PRIMES[:d]):
            U[k, j] = _radical_inverse(idx, b)
    # deterministic scramble shift
    U = (U + rng_shift) % 1.0
    U = np.clip(U, 1e-9, 1.0 - 1e-9)
    Z = np.empty((n, d))
    for j in range(0, d, 2):
        u1 = U[:, j]
        u2 = U[:, j+1] if j+1 < d else 0.5
        r = np.sqrt(-2.0 * np.log(u1))
        th = 2.0 * np.pi * u2
        Z[:, j] = r * np.cos(th)
        if j+1 < d:
            Z[:, j+1] = r * np.sin(th)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-300
    return Z

def _riesz_fg_factory(IU, p, M):
    def fg(yf):
        n12 = 68
        d12 = 12
        Y = yf.reshape(n12, d12)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        pos = np.where(g > 0, g, 0.0)
        m = float(np.max(pos))
        if m <= 1e-15:
            return 0.0, np.zeros_like(yf)
        q = pos / m
        # q^p stable
        qp = np.power(q, p, out=np.empty_like(q))
        S = float(np.mean(qp)) + 1e-300
        F = m * (S ** (1.0 / p))
        # weights dF/dg
        coeff = (S ** (1.0 / p - 1.0)) / M
        qpm1 = np.power(q, p - 1, out=np.empty_like(q))
        # zero where g<=0
        qpm1[pos <= 0] = 0.0
        w = coeff * qpm1
        P = np.zeros((n12, n12))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return float(F), GY.ravel()
    return fg

def _lse_fg_factory(IU, beta):
    def fg(yf):
        Y = yf.reshape(68, 12)
        nr = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(np.max(g))
        e = np.exp(beta * (g - m))
        S = float(np.sum(e)) + 1e-300
        F = m + np.log(S) / beta
        w = e / S
        P = np.zeros((68, 68))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return float(F), GY.ravel()
    return fg

def _run_stages(X0, makers, iters):
    y = (X0 / (np.linalg.norm(X0, axis=1, keepdims=True) + 1e-300)).ravel().copy()
    for fgmk, mi in zip(makers, iters):
        fg = fgmk
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': mi, 'ftol': 1e-14, 'gtol': 1e-6,
                                  'maxfun': mi * 2, 'maxcor': 12})
            y = r.x.copy()
        except Exception:
            break
    Y = y.reshape(68, 12)
    Y /= (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-300)
    return Y

def _fingerprint(X, K=300):
    G = X @ X.T
    iu = np.triu_indices(68, 1)
    c = G[iu]
    c = np.sort(c)[::-1]
    return c[:K].copy(), float(c[0])

def _hist_dist(f1, f2):
    return float(np.linalg.norm(f1 - f2))

def _lp_polish(x, deadline, r0=8e-4, cap_p=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.4:
            break
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
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
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        step = res.x[:-1].reshape(n, d)
        y = x + step
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            ok = False
            for sh in (0.3, 0.1):
                yy = x + step * sh
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    ok = True
                    break
            if not ok:
                radius *= 0.5
    return x

def _align_average(A, B):
    n, d = A.shape
    # initial assignment
    S = A @ B.T
    try:
        ra, ca = linear_sum_assignment(-S)
        perm = np.empty(n, dtype=int)
        perm[ra] = ca
    except Exception:
        # greedy
        perm = np.argmax(S, axis=1)
    Q = np.eye(d)
    for _ in range(3):
        Bp = B[perm]
        M = Bp.T @ A
        try:
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            Q = U @ Vt
        except Exception:
            break
        S2 = A @ (B @ Q).T
        try:
            ra, ca = linear_sum_assignment(-S2)
            perm2 = np.empty(n, dtype=int)
            perm2[ra] = ca
            perm = perm2
        except Exception:
            break
    Bm = (B[perm]) @ Q
    C = A + Bm
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-300)
    return C

def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    n, d = 68, 12
    IU = np.triu_indices(n, 1)
    M = n * (n - 1) // 2
    rng = np.random.default_rng(987654321)

    # fallback record comparison only (not used as seed)
    fallback = None
    fallback_score = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    s = _max_cos(a)
                    if s < fallback_score:
                        fallback_score = s
                        fallback = a.copy()
        except Exception:
            pass

    # ---- generate incumbent-free population ----
    NPOP = 26
    inits = []
    for i in range(NPOP):
        typ = i % 4
        if typ == 0:
            X = rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif typ == 1:
            X = _halton_gaussian(n, d, offset=i * 977, rng_shift=((i * 0.137) % 1.0))
            # random rotation to diversify
            Qr, _ = np.linalg.qr(rng.standard_normal((d, d)))
            X = X @ Qr.T
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        elif typ == 2:
            # perturbed lattice directions: random integer vectors
            X = np.empty((n, d))
            cnt = 0
            guard = 0
            seen = set()
            while cnt < n and guard < 5000:
                guard += 1
                v = rng.integers(-2, 3, size=d).astype(float)
                if np.all(v == 0):
                    continue
                key = tuple(np.sign(v).astype(int).tolist() + [int(np.sum(v * v))])
                # allow duplicates but jitter anyway
                v = v + 0.35 * rng.standard_normal(d)
                nv = np.linalg.norm(v)
                if nv < 1e-9:
                    continue
                X[cnt] = v / nv
                cnt += 1
            # random rotation
            Qr, _ = np.linalg.qr(rng.standard_normal((d, d)))
            X = X @ Qr.T
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        else:
            # farthest-point sampling from 2000 candidates
            cand = rng.standard_normal((2000, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            sel = [int(rng.integers(0, 2000))]
            # greedy max-min
            best_min = None
            for _ in range(1, n):
                # compute max dot to selected
                S = cand @ cand[sel].T
                md = np.max(S, axis=1)
                md[sel] = 2.0
                nxt = int(np.argmin(md))
                sel.append(nxt)
            X = cand[sel].copy()
        inits.append(X)

    # ---- Phase 1: quick screen Riesz p=8,32 + lse ----
    screened = []
    phase1_end = min(t0 + 75.0, T_END - 85.0)
    for idx, X0 in enumerate(inits):
        if time.time() > phase1_end:
            break
        # vary exponents slightly per idx to diversify: base set includes 8,16,32,64,128
        if idx % 2 == 0:
            makers = [_riesz_fg_factory(IU, 8, M), _riesz_fg_factory(IU, 32, M), _lse_fg_factory(IU, 600.0)]
            iters = [60, 60, 80]
        else:
            makers = [_riesz_fg_factory(IU, 16, M), _riesz_fg_factory(IU, 64, M), _lse_fg_factory(IU, 600.0)]
            iters = [60, 60, 80]
        try:
            X1 = _run_stages(X0, makers, iters)
        except Exception:
            continue
        s = _max_cos(X1)
        fp, _ = _fingerprint(X1)
        screened.append((s, X1, fp))
    if len(screened) == 0:
        # extreme fallback
        return fallback

    screened.sort(key=lambda z: z[0])
    # cluster by sorted-cosine histogram to discard near-duplicates
    kept = []
    TOL = 4e-3
    for s, X, fp in screened:
        dup = False
        for _, _, fp2 in kept:
            if _hist_dist(fp, fp2) < TOL:
                dup = True
                break
        # keep even duplicates if very good? No: discard per direction
        if not dup:
            kept.append((s, X, fp))
        if len(kept) >= 7:
            # allow more if scores very close? stop
            pass
        if len(kept) >= 7 and s > kept[0][0] + 0.02:
            break
    kept = kept[:7]
    # if too few due to tight tol, relax
    if len(kept) < 3:
        kept = screened[:3]

    # ---- Phase 2: longer anneals p=32,128 + high-beta lse ----
    annealed = []
    for s, X, fp in kept:
        if time.time() > T_END - 55.0:
            annealed.append((s, X))
            continue
        makers = [_riesz_fg_factory(IU, 32, M), _riesz_fg_factory(IU, 128, M),
                  _lse_fg_factory(IU, 2000.0), _lse_fg_factory(IU, 6000.0)]
        iters = [80, 80, 100, 80]
        try:
            Xa = _run_stages(X, makers, iters)
        except Exception:
            annealed.append((s, X))
            continue
        sa = _max_cos(Xa)
        annealed.append((sa, Xa))
        if time.time() > T_END - 50.0:
            break
    annealed.sort(key=lambda z: z[0])
    # re-cluster after anneal with tighter tol
    diverse = []
    for s, X in annealed:
        fp, _ = _fingerprint(X)
        dup = False
        for _, X2, fp2 in diverse:
            if _hist_dist(fp, fp2) < 1e-3:
                dup = True
                break
        if not dup:
            diverse.append((s, X, fp))
    diverse = diverse[:5]
    if len(diverse) == 0:
        diverse = [(annealed[0][0], annealed[0][1], screened[0][2])]

    # ---- recombination after optimal rotation + matching ----
    children = []
    pairs = [(0, 1), (0, 2), (1, 2), (0, 3)] if len(diverse) >= 4 else [(0, 1)]
    for (a, b) in pairs:
        if time.time() > T_END - 35.0:
            break
        if a >= len(diverse) or b >= len(diverse):
            continue
        A = diverse[a][1]
        B = diverse[b][1]
        try:
            C0 = _align_average(A, B)
            # small tangent jitter for children variety
            jitter = 0.008 * rng.standard_normal((n, d))
            # project jitter to tangent
            dot = np.sum(jitter * C0, axis=1, keepdims=True)
            jitter -= dot * C0
            C0 = C0 + jitter
            C0 /= np.linalg.norm(C0, axis=1, keepdims=True)
            makers = [_riesz_fg_factory(IU, 64, M), _lse_fg_factory(IU, 2500.0)]
            Cc = _run_stages(C0, makers, [70, 90])
            sc = _max_cos(Cc)
            children.append((sc, Cc))
        except Exception:
            continue
    # merge diverse + children
    pool = [(s, X) for (s, X, _) in diverse] + children
    pool.sort(key=lambda z: z[0])

    # ---- final LP squeeze on top few ----
    bestX = pool[0][1].copy()
    bestS = pool[0][0]
    # also consider second basin even if slightly worse to preserve diversity shot
    for k in range(min(3, len(pool))):
        if time.time() > T_END - 12.0:
            break
        Xc = pool[k][1].copy()
        # extra high-beta polish before LP
        if time.time() < T_END - 20.0:
            try:
                Xc = _run_stages(Xc, [_lse_fg_factory(IU, 8000.0)], [90])
            except Exception:
                pass
        dl = min(time.time() + 6.0, T_END - 2.0)
        try:
            Xp = _lp_polish(Xc, dl, r0=6e-4)
            sp = _max_cos(Xp)
        except Exception:
            sp = _max_cos(Xc)
            Xp = Xc
        if sp < bestS:
            bestS = sp
            bestX = Xp.copy()

    # final comparison to loaded record (sole use as comparison)
    if fallback is not None and fallback_score < bestS:
        bestX = fallback.copy()
        bestS = fallback_score

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
