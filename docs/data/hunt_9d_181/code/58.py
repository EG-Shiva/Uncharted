import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _adam_lse(X, steps, k0=80.0, k1=400.0, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1p = 1.0
    b2p = 1.0
    for s in range(steps):
        frac = s / max(1, steps - 1)
        k = k0 + (k1 - k0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # shifted exp
        E = np.exp(k * (G - gmax))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        E /= S
        Gr = 2.0 * (E @ X)
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        b1p *= b1
        b2p *= b2
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        mh = m / (1 - b1p)
        vh = v / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr = np.where(nr < 1e-12, 1.0, nr)
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    return X

def _lp_polish(X, budget, t_dead=None):
    t0 = time.time()
    x = _norm(X.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while True:
        if time.time() - t0 >= budget:
            break
        if t_dead is not None and time.time() >= t_dead:
            break
        if radius < 1e-14 or it > 40:
            break
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2000:
            csel = c[sel]
            idx = np.argpartition(-csel, 2000)[:2000]
            I = I[idx]; J = J[idx]
            csel2 = csel[idx]
            p = 2000
        else:
            csel2 = c[sel]
        nv = n * d + 1
        cols_mat = np.empty((p, 2 * d + 1), dtype=np.int64)
        cols_mat[:, :d] = I[:, None] * d + ar
        cols_mat[:, d:2 * d] = J[:, None] * d + ar
        cols_mat[:, 2 * d] = n * d
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = cols_mat.ravel()
        vals_mat = np.empty((p, 2 * d + 1))
        vals_mat[:, :d] = x[J]
        vals_mat[:, d:2 * d] = x[I]
        vals_mat[:, 2 * d] = -1.0
        vals = vals_mat.ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel2, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _rand_orth(rng, d=9):
    A = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1
    Q = Q * s[None, :]
    return Q

def _seed_orth_frames(rng):
    n, d = 181, 9
    X = np.zeros((n, d))
    idx = 0
    for _ in range(10):
        Q = _rand_orth(rng, d)
        vecs = Q  # 9 orthonormal rows? Q rows orthonormal too (orthogonal matrix)
        for k in range(9):
            if idx >= 180:
                break
            X[idx] = vecs[k]; idx += 1
            if idx >= 180:
                break
            X[idx] = -vecs[k]; idx += 1
    # last point
    r = rng.standard_normal(d); r /= np.linalg.norm(r)
    X[180] = r
    # tiny jitter
    X = X + rng.standard_normal(X.shape) * 1e-3
    return _norm(X)

def _seed_sparse(rng, rotate=False):
    d = 9
    pool = []
    # k=2
    for a in range(d):
        for b in range(a + 1, d):
            for sa in (-1.0, 1.0):
                for sb in (-1.0, 1.0):
                    v = np.zeros(d)
                    v[a] = sa / np.sqrt(2.0); v[b] = sb / np.sqrt(2.0)
                    pool.append(v)
    # k=3 subset (random subset to keep pool manageable, full 672)
    # add all k=3
    import itertools
    for comb in itertools.combinations(range(d), 3):
        for mask in range(8):
            v = np.zeros(d)
            for j, pos in enumerate(comb):
                s = 1.0 if ((mask >> j) & 1) else -1.0
                v[pos] = s / np.sqrt(3.0)
            pool.append(v)
    pool = np.array(pool)  # (816,9)
    m = pool.shape[0]
    sel = rng.choice(m, size=181, replace=False)
    X = pool[sel].copy()
    X = X + rng.standard_normal(X.shape) * 0.005
    X = _norm(X)
    if rotate:
        Q = _rand_orth(rng, d)
        X = X @ Q
    return _norm(X)

def _e8_pool():
    d8 = 8
    lst = []
    for a in range(d8):
        for b in range(a + 1, d8):
            for sa in (-1.0, 1.0):
                for sb in (-1.0, 1.0):
                    v = np.zeros(d8)
                    v[a] = sa / np.sqrt(2.0); v[b] = sb / np.sqrt(2.0)
                    lst.append(v)
    for mask in range(256):
        bits = [(mask >> j) & 1 for j in range(8)]
        nneg = sum(1 for b in bits if b == 0)  # map 0-> -1, 1-> +1
        if nneg % 2 != 0:
            continue
        v = np.zeros(d8)
        for j in range(8):
            s = 1.0 if bits[j] == 1 else -1.0
            v[j] = s * 0.5 / np.sqrt(2.0) * 2.0  # 0.5/sqrt2*? careful
            # want entries +-0.5/sqrt2? No: h entries +-0.5, unit = h/sqrt2
            v[j] = s * 0.5 / np.sqrt(2.0) * np.sqrt(2.0) / 1.0  # placeholder
        # redo correctly: unit = h / sqrt2, h=+-0.5
        for j in range(8):
            s = 1.0 if bits[j] == 1 else -1.0
            v[j] = s * 0.5 / np.sqrt(2.0) * 1.0  # = s*0.3535? Wait 0.5/sqrt2=0.3535 correct
            # Actually h/sqrt2 = 0.5/sqrt2 = 0.3535 yes
        # fix double scaling above: v[j] already correct
        lst.append(v.copy())
    # NOTE: first loop 112, second 128 => 240
    return np.array(lst)

_E8 = None
def _seed_e8(rng):
    global _E8
    if _E8 is None:
        _E8 = _e8_pool()
    pool = _E8
    sel = rng.choice(pool.shape[0], size=181, replace=False)
    V = pool[sel]  # (181,8) already unit in 8D? check norm: D8 type norm 1, half-int norm? 8*(0.3535^2)=8*0.125=1 yes
    h = rng.standard_normal(181) * 0.45
    X = np.zeros((181, 9))
    X[:, :8] = V
    X[:, 8] = h
    X = _norm(X)
    Q = _rand_orth(rng, 9)
    X = X @ Q
    return _norm(X)

def _seed_antipodal(rng, npair=90):
    d = 9
    A = rng.standard_normal((n pair if False else 90, d)) if False else rng.standard_normal((90, d))
    A = _norm(A)
    X = np.zeros((181, d))
    X[:90] = A
    X[90:180] = -A
    r = rng.standard_normal(d); r /= np.linalg.norm(r)
    X[180] = r
    # small jitter to break exact antipodality slightly
    X = X + rng.standard_normal(X.shape) * 0.002
    return _norm(X)

def _align_procrustes(A, B, iters=3):
    # invariant feat matching first
    n = A.shape[0]
    GA = A @ A.T
    GB = B @ B.T
    np.fill_diagonal(GA, -2.0); np.fill_diagonal(GB, -2.0)
    # sorted rows descending
    FA = np.sort(GA, axis=1)[:, ::-1][:, :60]  # top-60 enough, faster
    FB = np.sort(GB, axis=1)[:, ::-1][:, :60]
    A2 = np.sum(FA * FA, axis=1)[:, None]
    B2 = np.sum(FB * FB, axis=1)[None, :]
    C = A2 + B2 - 2.0 * (FA @ FB.T)
    try:
        _, perm = linear_sum_assignment(C)
    except Exception:
        perm = np.arange(n)
    Bp = B[perm].copy()
    # procrustes
    for _ in range(iters):
        M = Bp.T @ A
        try:
            U, S, Vt = np.linalg.svd(M)
            R = U @ Vt
        except Exception:
            R = np.eye(A.shape[1])
        Ba = Bp @ R
        # hungarian on inner products
        Smat = A @ Ba.T
        try:
            _, perm2 = linear_sum_assignment(-Smat)
        except Exception:
            break
        # reorder Ba to match A
        Ba = Ba[perm2] if False else None
        # perm2 maps A_i -> Ba_{perm2[i]}; so new Bp ordering:
        # we have Ba aligned version of Bp; reorder it:
        # Use Ba_perm = (Bp@R)[perm2]? Actually need to reorder rows of Ba
        # linear_sum_assignment on -Smat: row i matched to col perm2[i]
        # So reordered aligned B: Bn[i] = Ba[perm2[i]]? No, Ba rows correspond to A rows currently (Bp already ordered).
        # Wait after first step Bp is ordered to A. Second matching between A and Ba (both ordered) gives perm2.
        # If perm2 is not identity, reorder Ba.
        cur = (Bp @ R)
        # build reordered
        Bn = cur[perm2] if False else cur
        # Correct mapping: Smat[i,j] = A_i . Ba_j, assignment i->perm2[i], so Bn[i]=Ba[perm2[i]]
        Bn = cur[perm2]
        # update Bp for next iter (un-rotate? Bp should be in original B coords? simpler set Bp = Bn @ R.T? No, keep aligned)
        # For next procrustes we want Bp (ordered, aligned) -> recompute M = Bn.T @ A
        Bp = Bn
    # final Bp is aligned and ordered to A
    # recompute final rotation once more for cleanliness
    M = Bp.T @ A
    try:
        U, S, Vt = np.linalg.svd(M)
        R = U @ Vt
        Bp = Bp @ R  # tiny extra? Actually Bp already aligned; this would double-rotate. Skip.
    except Exception:
        pass
    return Bp

def _slerp_child(A, Ba, t):
    n, d = A.shape
    C = np.empty_like(A)
    for i in range(n):
        a = A[i]; b = Ba[i]
        dt = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if dt > 0.999999:
            c = (a + b) * 0.5
            nrm = np.linalg.norm(c)
            if nrm < 1e-12:
                c = a
                nrm = 1.0
            C[i] = c / nrm
        elif dt < -0.999999:
            C[i] = a
        else:
            om = float(np.arccos(dt))
            s = float(np.sin(om))
            ca = float(np.sin((1 - t) * om)) / s
            cb = float(np.sin(t * om)) / s
            C[i] = ca * a + cb * b
    return _norm(C)

def solve():
    t0 = time.time()
    BUDGET = 168.0
    DEAD = t0 + BUDGET
    rng = np.random.default_rng(0)
    # baseline
    best = None; best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (181, 9):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val:
                        best_val = v; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((181, 9)))
        best_val = _max_cos(best)
    # build 12 independent seeds
    seeds = []
    try:
        seeds.append(_seed_orth_frames(np.random.default_rng(1)))
        seeds.append(_seed_orth_frames(np.random.default_rng(2)))
        seeds.append(_seed_orth_frames(np.random.default_rng(3)))
        seeds.append(_seed_sparse(np.random.default_rng(4), rotate=False))
        seeds.append(_seed_sparse(np.random.default_rng(5), rotate=True))
        seeds.append(_seed_sparse(np.random.default_rng(6), rotate=False))
        seeds.append(_seed_e8(np.random.default_rng(7)))
        seeds.append(_seed_e8(np.random.default_rng(8)))
        seeds.append(_seed_e8(np.random.default_rng(9)))
        seeds.append(_seed_antipodal(np.random.default_rng(10)))
        seeds.append(_seed_antipodal(np.random.default_rng(11)))
        tmp = np.random.default_rng(12).standard_normal((181, 9))
        seeds.append(_norm(tmp))
    except Exception:
        pass
    # optimise each with Adam LSE, ~ adapt steps to time
    results = []
    # reserve time: phase1 max 95s
    nseeds = len(seeds)
    for si, s0 in enumerate(seeds):
        rem = DEAD - time.time()
        # need at least ~25s left for recombination+polish
        if rem < 28.0:
            break
        # steps: 550 normally, reduce if late
        steps = 550
        if rem < 60:
            steps = 350
        if rem < 40:
            steps = 250
        try:
            X = _adam_lse(s0.copy(), steps=steps, k0=80.0, k1=400.0, lr=0.02)
            v = _max_cos(X)
        except Exception:
            continue
        results.append((v, X))
        if v < best_val - 1e-12:
            best_val = v; best = X.copy()
    if len(results) == 0:
        return _norm(best)
    results.sort(key=lambda t: t[0])
    # LP-polish top up to 2 distinct quickly if time allows
    polished = []
    for k in range(min(2, len(results))):
        rem = DEAD - time.time()
        if rem < 22.0:
            break
        try:
            Xp = _lp_polish(results[k][1], budget=min(2.5, max(0.8, rem - 18.0)))
            vp = _max_cos(Xp)
        except Exception:
            continue
        polished.append((vp, Xp))
        if vp < best_val:
            best_val = vp; best = Xp.copy()
    # choose two best distinct outcomes for crossover
    cand_pool = polished + results
    cand_pool.sort(key=lambda t: t[0])
    if len(cand_pool) >= 2:
        A = cand_pool[0][1].copy()
        # pick second with some distance (not identical)
        B = None
        for vp, Xp in cand_pool[1:]:
            # quick distance: mean best-match inner product
            try:
                S = np.abs(Xp @ A.T).max(axis=1).mean()
            except Exception:
                S = 0
            if S < 0.9995 or True:
                B = Xp.copy()
                break
        if B is None:
            B = cand_pool[1][1].copy()
    elif len(cand_pool) == 1:
        A = cand_pool[0][1].copy()
        B = best.copy()
    else:
        return _norm(best)
    # align
    try:
        Ba = _align_procrustes(A, B, iters=3)
    except Exception:
        Ba = B.copy()
    # children at multiple t
    for t in [0.5, 0.3, 0.7]:
        rem = DEAD - time.time()
        if rem < 8.0:
            break
        try:
            C = _slerp_child(A, Ba, t)
        except Exception:
            continue
        # smooth child
        steps_c = 350 if rem > 18 else 200
        try:
            C = _adam_lse(C, steps=steps_c, k0=80.0, k1=400.0, lr=0.015)
        except Exception:
            pass
        vc = _max_cos(C)
        if vc < best_val:
            best_val = vc; best = C.copy()
        rem2 = DEAD - time.time()
        if rem2 < 5.0:
            continue
        try:
            Cp = _lp_polish(C, budget=min(4.0, max(1.0, rem2 - 4.0)))
            vp = _max_cos(Cp)
            if vp < best_val:
                best_val = vp; best = Cp.copy()
            # also try extra smoothing after LP? one short anneal
        except Exception:
            pass
    # final polish of best if time left
    rem = DEAD - time.time()
    if rem > 3.0:
        try:
            Bp2 = _lp_polish(best, budget=min(rem - 1.0, 6.0))
            vp = _max_cos(Bp2)
            if vp < best_val:
                best = Bp2; best_val = vp
        except Exception:
            pass
    best = _norm(best)
    return best
