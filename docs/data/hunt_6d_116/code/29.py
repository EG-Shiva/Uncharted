import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contact_set(X, tol=1e-6):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    m = float(c.max())
    idx = np.where(c > m - tol)[0]
    I = iu[0][idx]; J = iu[1][idx]
    s = set((int(a)*1000+int(b)) for a, b in zip(I, J))
    return s, m, I, J, c

def _jaccard(a, b):
    if not a and not b:
        return 1.0
    u = len(a | b)
    if u == 0:
        return 1.0
    return len(a & b) / u

def _tangent_bases(X, rng):
    n, d = X.shape
    k = d - 1
    A = rng.standard_normal((n, d, k))
    # project out normal: A - x (x^T A)
    proj = np.einsum('ij,ijk->ik', X, A)  # n x k
    A = A - X[:, :, None] * proj[:, None, :]
    B = np.empty_like(A)
    for i in range(n):
        q, _ = np.linalg.qr(A[i])
        B[i] = q
    return B

def _soft_kick(X, rng, rms_target, nvec, sep_scale):
    n, d = X.shape
    s, m, I, J, c = _contact_set(X, tol=1e-6)
    # top pairs for separation
    iu = np.triu_indices(n, 1)
    G = X @ X.T
    flat = G[iu]
    order = np.argsort(-flat)
    K = int(rng.integers(5, 11))
    topk = order[:K]
    ta = iu[0][topk]; tb = iu[1][topk]
    if len(s) < 10:
        # fallback random kick
        DX = rng.standard_normal(X.shape)
        DX = DX - (np.sum(DX*X, axis=1, keepdims=True))*X
        rms = np.sqrt(np.mean(np.sum(DX*DX, axis=1))) + 1e-15
        DX *= (rms_target / rms)
        return _norm(X + DX), s
    B = _tangent_bases(X, rng)
    p = len(I)
    N5 = n * 5
    R = np.empty((p, N5), dtype=float)
    # rows: X[J] @ B[I], X[I] @ B[J]
    # vectorize with loop (p~560 fine)
    for r in range(p):
        i = int(I[r]); j = int(J[r])
        R[r, i*5:(i+1)*5] = B[i].T @ X[j]
        R[r, j*5:(j+1)*5] = B[j].T @ X[i]
    try:
        _, Sg, Vt = np.linalg.svd(R, full_matrices=False)
    except Exception:
        DX = rng.standard_normal(X.shape)
        DX = DX - (np.sum(DX*X, axis=1, keepdims=True))*X
        rms = np.sqrt(np.mean(np.sum(DX*DX, axis=1))) + 1e-15
        DX *= (rms_target / rms)
        return _norm(X + DX), s
    nv = min(nvec, Vt.shape[0])
    Vtail = Vt[-nv:, :]  # nv x N5 smallest
    coeff = rng.standard_normal(nv)
    coeff /= (np.linalg.norm(coeff) + 1e-15)
    # optional weight by 1/(s+eps) to favor softer? keep uniform for stability
    v = coeff @ Vtail  # N5
    Vmat = v.reshape(n, 5)
    DX = np.einsum('ijk,ik->ij', B, Vmat)  # n x 6 tangent
    rms = np.sqrt(np.mean(np.sum(DX*DX, axis=1))) + 1e-15
    DX *= (rms_target / rms)
    # separation component for largest dual-force proxy (largest cosines)
    Sep = np.zeros_like(X)
    for a, b in zip(ta, tb):
        a = int(a); b = int(b)
        # tangent at a away from b
        dab = X[a] @ X[b]
        ga = X[a] - dab * X[b]  # direction at b? we want move a away from b:
        # move a along -proj_b(a): -(X[b] - dab X[a])
        ta_vec = -(X[b] - dab * X[a])
        tb_vec = -(X[a] - dab * X[b])
        na = np.linalg.norm(ta_vec) + 1e-15
        nb = np.linalg.norm(tb_vec) + 1e-15
        Sep[a] += ta_vec / na
        Sep[b] += tb_vec / nb
    # project Sep to tangent (already) and scale
    Sep = Sep - (np.sum(Sep*X, axis=1, keepdims=True))*X
    srms = np.sqrt(np.mean(np.sum(Sep*Sep, axis=1))) + 1e-15
    if srms > 1e-12:
        Sep *= (sep_scale / srms)
        DX = DX + Sep
    Y = _norm(X + DX)
    return Y, s

def _feas_adam(X, t_target, steps, lr, rng=None):
    X = _norm(X.copy())
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(1, steps+1):
        G = X @ X.T
        H = G - t_target
        np.fill_diagonal(H, 0.0)
        np.maximum(H, 0.0, out=H)
        # gradient 2 H @ X
        Gr = 2.0 * (H @ X)
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        m1 = b1 * m1 + (1 - b1) * Gt
        m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
        c1 = 1 - b1 ** s
        c2 = 1 - b2 ** s
        mh = m1 / c1
        vh = m2 / c2
        step = mh / (np.sqrt(vh) + eps)
        # re-project step
        dot2 = np.sum(step * X, axis=1, keepdims=True)
        stept = step - dot2 * X
        # optional clip
        # decay lr slightly
        lrc = lr * (1.0 - 0.5 * (s-1) / max(steps-1, 1))
        X = X - lrc * stept
        X = _norm(X)
    return X

def _smooth(X, steps, beta0, beta1, s0, s1):
    X = _norm(X.copy())
    vel = np.zeros_like(X)
    mu = 0.72
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * frac
        step = s0 + (s1 - s0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max()
        W = np.exp(beta * (G - mx))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if not np.isfinite(S) or S == 0:
            break
        W /= S
        grad = W @ X
        proj = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - proj * X
        nrm = np.sqrt((gt ** 2).sum(axis=1, keepdims=True)) + 1e-15
        direc = gt / nrm
        vel = mu * vel + (1.0 - mu) * direc
        vn = np.sqrt((vel ** 2).sum(axis=1, keepdims=True)) + 1e-15
        vdir = vel / vn
        X = X - step * vdir
        X = _norm(X)
    return X

def _lp_polish(X, seconds):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    best_x = X.copy()
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        c_all = G[iu]
        m = c_all.max()
        thresh = m - max(20 * radius, 1e-9)
        idx = np.where(c_all > thresh)[0]
        if len(idx) > 1500:
            order = np.argsort(c_all[idx])[-1500:]
            idx = idx[order]
        if len(idx) == 0:
            break
        sel = np.zeros_like(c_all, dtype=bool)
        sel[idx] = True
        I, J = iu[0][sel], iu[1][sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            best = v
            best_x = y.copy()
            X = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    HARD_END = t0 + BUDGET - 1.0
    EXPL_END = t0 + BUDGET - 26.0
    rng = np.random.default_rng(777123)
    def clean(a):
        a = np.asarray(a, dtype=float)
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        return _norm(a)
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = clean(np.load(f))
            if c is not None:
                cands.append(c)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((116, 6))))
    vals = [_maxcos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    cur_x = x_best.copy()
    cur_val = best_val
    cur_set, _, _, _, _ = _contact_set(cur_x, tol=1e-6)
    best_set = set(cur_set)
    it = 0
    T_TARGET = 0.5958
    while time.time() < EXPL_END:
        it += 1
        if time.time() > HARD_END - 20:
            break
        frac = min(1.0, (time.time() - t0) / max(EXPL_END - t0, 1e-9))
        temp = 0.0025 * (1 - frac) + 0.00008 * frac
        # soft-mode params
        rms_target = float(rng.uniform(0.05, 0.25))
        nvec = int(rng.integers(20, 41))
        sep_scale = float(rng.uniform(0.0, 0.06))
        # alternate occasional larger kick
        if rng.random() < 0.15:
            rms_target = float(rng.uniform(0.20, 0.35))
        base = cur_x.copy()
        base_set = cur_set
        try:
            kicked, _ = _soft_kick(base, rng, rms_target, nvec, sep_scale)
        except Exception:
            continue
        # feasibility phase 500-1000 steps
        fsteps = int(rng.integers(500, 1001))
        # shrink steps if low on time
        if HARD_END - time.time() < 40:
            fsteps = min(fsteps, 500)
        try:
            feas = _feas_adam(kicked, T_TARGET, steps=fsteps, lr=0.02)
        except Exception:
            continue
        # annealed softmax
        try:
            trial = _smooth(feas, steps=int(rng.integers(220, 380)),
                            beta0=30, beta1=220, s0=0.025, s1=0.003)
        except Exception:
            continue
        v = _maxcos(trial)
        # short LP polish if promising
        do_lp = (v < best_val + 0.002) and (it % 4 == 0 or v < best_val)
        if do_lp and time.time() < EXPL_END - 2:
            try:
                trial_p = _lp_polish(trial, 0.6)
                v2 = _maxcos(trial_p)
                trial = trial_p
                v = v2
            except Exception:
                pass
        # topology gate
        try:
            trial_set, _, _, _, _ = _contact_set(trial, tol=1e-6)
        except Exception:
            continue
        jac = _jaccard(base_set, trial_set)
        # global best always kept
        if v < best_val - 1e-12:
            best_val = v
            x_best = trial.copy()
            best_set = set(trial_set)
            # move cur to new best (new basin by definition if improved a lot? accept)
            cur_x = trial.copy()
            cur_val = v
            cur_set = set(trial_set)
            continue
        # otherwise require material change
        if jac >= 0.9:
            # slid back: discard unless SA wants it and it's close? discard
            continue
        # different basin: Metropolis on cur
        if v < cur_val or rng.random() < np.exp(-(v - cur_val) / max(temp, 1e-12)):
            if v < cur_val + 0.03:
                cur_x = trial.copy()
                cur_val = v
                cur_set = set(trial_set)
        # periodic restart to best with forced diversification (soft kick handles it)
        if it % 60 == 0:
            cur_x = x_best.copy()
            cur_val = best_val
            cur_set = set(best_set)
            # force a big random soft kick next iter by perturbing cur slightly
            # to avoid exact repeat, add tiny noise
            cur_x = _norm(cur_x + 0.005 * rng.standard_normal(cur_x.shape))
    # final polish
    remain = HARD_END - time.time() - 0.5
    if remain > 3:
        try:
            x_best = _lp_polish(x_best, min(remain, 20.0))
        except Exception:
            pass
    return _norm(x_best)
