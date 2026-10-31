import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _build_Rs(K, a):
    th1 = 2.0 * np.pi / float(K)
    th2 = 2.0 * np.pi * float(a) / float(K)
    Rs = np.zeros((K, 4, 4))
    for k in range(K):
        c1 = np.cos(k * th1); s1 = np.sin(k * th1)
        c2 = np.cos(k * th2); s2 = np.sin(k * th2)
        Rs[k] = np.array([[c1, -s1, 0., 0.],
                          [s1, c1, 0., 0.],
                          [0., 0., c2, -s2],
                          [0., 0., s2, c2]])
    return Rs


def _expand(V, Rs):
    # V: (M,4), Rs: (K,4,4) with X[r,k]=Rs[k]@V[r]
    M = V.shape[0]
    K = Rs.shape[0]
    # einsum: X[r,k,i] = sum_j Rs[k,i,j]*V[r,j]
    X = np.einsum('kij,rj->rki', Rs, V).reshape(M * K, 4)
    return X


def _run_sym(K, a, steps, seed, lr=0.015, a0=20.0, a1=240.0, deadline=None):
    M = 153 // K
    assert M * K == 153
    rng = np.random.default_rng(seed)
    Rs = _build_Rs(K, a)
    Y = rng.standard_normal((M, 4))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    iu = np.triu_indices(M * K, 1)
    best_val = 1e9
    best_X = None
    log_ratio = np.log(a1 / a0)
    for t in range(1, steps + 1):
        if deadline is not None and t % 50 == 0 and time.time() > deadline:
            break
        frac = t / float(steps)
        alpha = a0 * np.exp(log_ratio * frac)
        # forward
        # Y already unit
        X = _expand(Y, Rs)
        G = X @ X.T
        cij = G[iu]
        m0 = float(np.max(cij))
        e = np.exp(alpha * (cij - m0))
        S = float(np.sum(e))
        w = e / S
        # gradient w.r.t X: Wsym @ X
        N = X.shape[0]
        W = np.zeros((N, N))
        W[iu] = w
        W = W + W.T
        dEdX = W @ X  # (N,4)
        # pullback to reps: grad_V[r,j]=sum_k sum_i dEdX[r,k,i]*Rs[k,i,j]
        g3 = dEdX.reshape(M, K, 4)
        grad_V = np.einsum('rki,kij->rj', g3, Rs)
        # tangent projection
        dot = np.sum(grad_V * Y, axis=1, keepdims=True)
        gt = grad_V - dot * Y
        # Adam
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        mhat = m / (1.0 - b1 ** t)
        vhat = v / (1.0 - b2 ** t)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        # clip per-row norm for safety
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        cap = 0.05
        scale = np.ones_like(sn)
        big = sn[:, 0] > cap
        scale[big] = cap / sn[big]
        step = step * scale
        Y = Y - step
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm < 1e-12] = 1e-12
        Y = Y / nrm
        if t % 200 == 0 or t == steps:
            val = _max_cos(X)
            if val < best_val:
                best_val = val
                best_X = X.copy()
    Xf = _expand(Y, Rs)
    vf = _max_cos(Xf)
    if best_X is None or vf < best_val:
        best_val = vf
        best_X = Xf
    return best_X, best_val


def _lp_polish(x, seconds, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while True:
        if time.time() - start > seconds:
            break
        if time.time() > deadline:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu0]
        mv = float(c.max())
        sel = c > mv - max(20.0 * radius, 1e-9)
        if np.count_nonzero(sel) == 0:
            break
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn < 1e-12] = 1e-12
        y = y / yn
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def solve():
    t0 = time.time()
    DEADLINE = t0 + 168.0
    rng = np.random.default_rng(12345)
    # load fallback record
    fallback = None
    fallback_score = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.array(a, dtype=float)
            if a.shape == (153, 4) and np.all(np.isfinite(a)):
                nrm = np.linalg.norm(a, axis=1, keepdims=True)
                if np.min(nrm) > 1e-9:
                    a = a / nrm
                    s = _max_cos(a)
                    if s < fallback_score:
                        fallback_score = s
                        fallback = a.copy()
        except Exception:
            continue
    if fallback is None:
        a = rng.standard_normal((153, 4))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        fallback = a
        fallback_score = _max_cos(a)
    best_x = fallback.copy()
    best_score = float(fallback_score)
    try:
        # ---- symmetry phase: minimal crash-safe ----
        # (K,a): C17 with a=1..8 (9 reps), C9 with a=1,2 (17 reps)
        configs = [(17, 1), (17, 2), (17, 3), (17, 4), (17, 5), (17, 6), (17, 7), (17, 8),
                   (9, 1), (9, 2)]
        sym_cands = []
        # time budget for symmetry: at most ~90s, leave >=40s for polishing
        sym_deadline = min(DEADLINE - 45.0, t0 + 95.0)
        base_seed = 1000
        for idx, (K, a) in enumerate(configs):
            if time.time() > sym_deadline:
                break
            # steps adapted to remaining time
            remain = sym_deadline - time.time()
            if remain < 3.0:
                break
            # ~2200 steps each is cheap (<2s); use fewer for C9 (larger M)
            if K == 17:
                steps = 2200
            else:
                steps = 1600
            try:
                Xsym, vsym = _run_sym(K, a, steps, seed=base_seed + idx * 17 + a,
                                      deadline=sym_deadline)
            except Exception:
                continue
            try:
                if Xsym is None or Xsym.shape != (153, 4) or not np.all(np.isfinite(Xsym)):
                    continue
                Xsym = Xsym / np.linalg.norm(Xsym, axis=1, keepdims=True)
                vsym = _max_cos(Xsym)
                sym_cands.append((vsym, Xsym))
            except Exception:
                continue
        sym_cands.sort(key=lambda t: t[0])
        # ---- break symmetry + LP polish ----
        # pick top few expanded codes
        to_polish = []
        for vs, Xs in sym_cands[:3]:
            to_polish.append(Xs)
        # always include fallback as safety (will be polished too if time)
        # add tangential noise to symmetric candidates
        noisy = []
        for Xs in to_polish:
            if time.time() > DEADLINE - 30.0:
                break
            try:
                pert = Xs.copy()
                nz = rng.standard_normal(pert.shape)
                # tangential component, scale 1e-3
                d = np.sum(nz * pert, axis=1, keepdims=True)
                tang = nz - d * pert
                tn = np.linalg.norm(tang, axis=1, keepdims=True)
                tn[tn < 1e-12] = 1e-12
                tang = tang / tn * 1e-3
                pert = pert + tang
                pert /= np.linalg.norm(pert, axis=1, keepdims=True)
                noisy.append(pert)
            except Exception:
                continue
        # polish noisy codes + fallback, keep best
        # allocate LP budget
        cands_lp = []
        for p in noisy:
            cands_lp.append(p)
        # ensure we also consider raw symmetric (no noise) best
        if len(sym_cands) > 0:
            cands_lp.append(sym_cands[0][1])
        # limit to 4 to fit time
        cands_lp = cands_lp[:4]
        lp_per = 8.0
        # if many candidates, reduce per-candidate
        if len(cands_lp) > 0:
            remain = DEADLINE - time.time() - 4.0
            if remain < 2.0:
                cands_lp = cands_lp[:1]
                remain = DEADLINE - time.time() - 2.0
            lp_per = max(2.0, min(9.0, remain / max(1, len(cands_lp) + 1)))
        for p in cands_lp:
            if time.time() > DEADLINE - 3.0:
                break
            try:
                xp, vp = _lp_polish(p, lp_per, DEADLINE - 1.5)
                if vp < best_score and np.all(np.isfinite(xp)):
                    best_score = vp
                    best_x = xp.copy()
            except Exception:
                continue
        # final polish of current best briefly if time remains
        if time.time() < DEADLINE - 6.0:
            try:
                xp, vp = _lp_polish(best_x, min(6.0, DEADLINE - time.time() - 2.0), DEADLINE - 1.0)
                if vp < best_score and np.all(np.isfinite(xp)):
                    best_score = vp
                    best_x = xp.copy()
            except Exception:
                pass
    except Exception:
        # any failure: fall back to record
        pass
    # ---- validate and return ----
    try:
        best_x = np.array(best_x, dtype=float)
        if best_x.shape != (153, 4) or not np.all(np.isfinite(best_x)):
            best_x = fallback
        nrm = np.linalg.norm(best_x, axis=1, keepdims=True)
        nrm[~np.isfinite(nrm)] = 1.0
        nrm[nrm < 1e-12] = 1.0
        best_x = best_x / nrm
        if not np.all(np.isfinite(best_x)):
            best_x = fallback
        # if our search is worse than fallback, return fallback
        try:
            s = _max_cos(best_x)
            if s > fallback_score:
                best_x = fallback
        except Exception:
            best_x = fallback
    except Exception:
        try:
            best_x = fallback
        except Exception:
            rr = np.random.default_rng(0).standard_normal((153, 4))
            rr /= np.linalg.norm(rr, axis=1, keepdims=True)
            best_x = rr
    return best_x
