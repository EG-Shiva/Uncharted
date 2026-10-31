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
    return float(np.max(G))

def _tangent_bases(X):
    n, d = X.shape
    m = d - 1
    T = np.empty((n, d, m))
    for i in range(n):
        x = X[i]
        k = int(np.argmin(np.abs(x)))
        cols = []
        for j in range(d):
            if j == k:
                continue
            v = np.zeros(d)
            v[j] = 1.0
            v = v - float(np.dot(v, x)) * x
            cols.append(v)
        M = np.stack(cols, axis=1)  # 6x5
        Q, _ = np.linalg.qr(M)
        T[i] = Q
    return T

def _build_R(X, T, I, J):
    p = len(I)
    n = X.shape[0]
    m = T.shape[2]
    R = np.zeros((p, n * m))
    for t in range(p):
        i = int(I[t]); j = int(J[t])
        xi = X[i]; xj = X[j]
        Ti = T[i]; Tj = T[j]
        ai = Ti.T @ xj  # 5
        aj = Tj.T @ xi
        R[t, i * m:(i + 1) * m] = ai
        R[t, j * m:(j + 1) * m] = aj
    return R

def _lp_polish(x, deadline):
    x = _norm(x.copy())
    n, d = x.shape
    best = _maxcos(x)
    bx = x.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1400:
            o = np.argsort(c[idx])[-1400:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
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
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _maxcos(y)
        if v < best:
            best = v; bx = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            if v <= best + 1e-15:
                x = y
            radius *= 0.5
    return bx, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 171.0
    rng = np.random.default_rng(7)

    def clean(a):
        try:
            a = np.asarray(a, dtype=float)
        except Exception:
            return None
        if a.shape != (116, 6) or not np.all(np.isfinite(a)):
            return None
        nn = np.linalg.norm(a, axis=1, keepdims=True)
        if np.any(nn < 1e-12):
            return None
        return a / nn

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
    best_x = cands[bi].copy()
    best_v = float(vals[bi])
    cur_x = best_x.copy()
    cur_v = best_v
    n, d = 116, 6
    m = d - 1

    STEPS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    thresh_cycle = [3e-7, 1e-6, 1e-5, 5e-7]
    outer = 0

    while time.time() < HARD_END - 6.0:
        outer += 1
        # use best as base (attack flat valley of incumbent)
        base_x = best_x.copy()
        base_v = best_v
        # active set
        athr = thresh_cycle[(outer - 1) % len(thresh_cycle)]
        try:
            G = base_x @ base_x.T
            iu = np.triu_indices(n, 1)
            c = G[iu]
            Mx = float(np.max(c))
            slack = Mx - c
            idx = np.where(slack < athr)[0]
            if idx.size < 50:
                idx = np.argsort(slack)[:400]
            if idx.size > 700:
                idx = idx[np.argsort(slack[idx])[:700]]
            I = iu[0][idx]; J = iu[1][idx]
            p = len(I)
        except Exception:
            continue
        # tangent + rigidity
        try:
            T = _tangent_bases(base_x)
            R = _build_R(base_x, T, I, J)
        except Exception:
            continue
        # SVD nullspace
        try:
            # full nullspace needed
            _, S, Vt = np.linalg.svd(R, full_matrices=True)
            smax = float(S[0]) if S.size else 1.0
            tol = max(1e-9, smax * 1e-10)
            rank = int(np.sum(S > tol))
            Ndim = n * m
            if rank >= Ndim:
                continue
            Z = Vt[rank:Ndim, :]  # k x 580
            k = Z.shape[0]
            if k <= 0:
                continue
            # cap k for sampling stability
            if k > 120:
                # keep smallest singular vectors? Vt already ordered, tail is null
                # keep all, sampling still ok
                pass
        except Exception:
            continue

        # evolutionary search in nullspace
        best_step_v = base_v
        best_step_x = None
        # population
        NDIR = 160
        try:
            W = rng.standard_normal((NDIR, k))
            Wn = np.linalg.norm(W, axis=1, keepdims=True) + 1e-15
            W = W / Wn
        except Exception:
            continue
        # allow a few generations
        for gen in range(4):
            if time.time() > HARD_END - 6.0:
                break
            # coeffs: k->580
            # Acoef = W @ Z  (NDIR x 580)
            try:
                Acoef = W @ Z
                A3 = Acoef.reshape(NDIR, n, m)
                # D = T @ a per point: einsum
                # T: n x d x m, A3: NDIR x n x m -> D: NDIR x n x d
                D = np.einsum('n d m, q n m -> q n d', T, A3)
            except Exception:
                break
            # evaluate steps
            improved_this_gen = False
            for qi in range(NDIR):
                Di = D[qi]
                for s in STEPS:
                    Xt = _norm(base_x + s * Di)
                    v = _maxcos(Xt)
                    if v < best_step_v - 1e-13:
                        best_step_v = v
                        best_step_x = Xt.copy()
                        # keep direction for breeding? mark
            # breed next generation from top dirs
            # score dirs by their best over steps (recompute quickly for ranking)
            # to rank, use s=3e-4 and 1e-3 approximate: evaluate fast
            scores = np.empty(NDIR)
            for qi in range(NDIR):
                # quick score: min over two probe steps
                s1 = STEPS[3]
                Xt1 = _norm(base_x + s1 * D[qi])
                scores[qi] = _maxcos(Xt1)
                # also try 1e-3
                Xt2 = _norm(base_x + 1e-3 * D[qi])
                v2 = _maxcos(Xt2)
                if v2 < scores[qi]:
                    scores[qi] = v2
            order = np.argsort(scores)
            elite = W[order[:16]]
            if best_step_x is not None:
                break  # found improvement, go polish it
            # build next W: elite + mutations + fresh
            Wnew = np.empty_like(W)
            Wnew[:16] = elite
            for ii in range(16, NDIR):
                if ii < 96:
                    a = elite[rng.integers(0, 16)]
                    b = elite[rng.integers(0, 16)]
                    alpha = rng.random()
                    w = alpha * a + (1 - alpha) * b
                    w = w + 0.35 * rng.standard_normal(k)
                    w /= (np.linalg.norm(w) + 1e-15)
                    Wnew[ii] = w
                else:
                    w = rng.standard_normal(k)
                    w /= (np.linalg.norm(w) + 1e-15)
                    Wnew[ii] = w
            W = Wnew
        if best_step_x is not None and best_step_v < base_v - 5e-14:
            # backtracking refinement around that step: try interpolate smaller/larger
            cur_x_try = best_step_x
            cur_v_try = best_step_v
            # short LP polish
            dl = min(time.time() + 1.2, HARD_END - 5.0)
            if dl > time.time() + 0.15:
                try:
                    Xp, vp = _lp_polish(cur_x_try, dl)
                except Exception:
                    Xp, vp = cur_x_try, cur_v_try
            else:
                Xp, vp = cur_x_try, cur_v_try
            if vp < best_v - 5e-14:
                best_x, best_v = Xp.copy(), float(vp)
                continue
            elif cur_v_try < best_v - 5e-14:
                best_x, best_v = cur_x_try.copy(), float(cur_v_try)
                # also try polish from there next outer
                continue
        # no improvement: try a tiny random rotation within nullspace + polish to move along valley?
        # take the best-ranked direction even if not improving and do very small step + polish to shift basin
        # only occasionally to avoid wasting time
        if outer % 6 == 0 and time.time() < HARD_END - 20.0:
            try:
                # pick top elite direction, small step 3e-4, polish
                w0 = W[0]
                a0 = (w0 @ Z).reshape(n, m)
                D0 = np.einsum('ndm,nm->nd', T, a0)
                Xt = _norm(base_x + 3e-4 * D0)
                Xp, vp = _lp_polish(Xt, min(time.time() + 0.8, HARD_END - 5.0))
                if vp < best_v - 5e-14:
                    best_x, best_v = Xp.copy(), float(vp)
            except Exception:
                pass
        # loop with updated active set (same best, different threshold gives different null)
        continue

    rem = HARD_END - time.time()
    if rem > 4.0:
        try:
            Xf, vf = _lp_polish(best_x, time.time() + min(rem - 1.0, 12.0))
            if vf <= best_v:
                best_x, best_v = Xf, vf
        except Exception:
            pass
    return _norm(best_x)
