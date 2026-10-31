import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    HARD = 172.0
    # deadlines
    SYM_END = t0 + 55.0
    BREAK_END = t0 + 145.0
    rng = np.random.default_rng(0)

    def _normr(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n = np.maximum(n, 1e-12)
        return x / n

    def _max_cos(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())

    # ---------- load incumbent ----------
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal((272, 4))
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = xb
    xb = _normr(np.asarray(xb, dtype=np.float64))
    xr = _normr(np.asarray(xr, dtype=np.float64))
    # ensure 272x4
    if xb.shape != (272, 4):
        # fallback resize deterministically
        tmp = rng.standard_normal((272, 4))
        tmp = _normr(tmp)
        m = min(len(xb), 272)
        tmp[:m] = _normr(xb[:m].reshape(m, -1)[:, :4] if xb.shape[1] != 4 else xb[:m])
        xb = tmp
    if xr.shape != (272, 4):
        xr = xb
    vb = _max_cos(xb)
    vr = _max_cos(xr)
    if vr < vb:
        best = xr.copy(); best_val = vr
    else:
        best = xb.copy(); best_val = vb

    # ---------- Q8 matrices ----------
    Li = np.array([[0., -1., 0., 0.],
                   [1., 0., 0., 0.],
                   [0., 0., 0., -1.],
                   [0., 0., 1., 0.]], dtype=np.float64)
    Lj = np.array([[0., 0., -1., 0.],
                   [0., 0., 0., 1.],
                   [1., 0., 0., 0.],
                   [0., -1., 0., 0.]], dtype=np.float64)
    Lk = np.array([[0., 0., 0., -1.],
                   [0., 0., -1., 0.],
                   [0., 1., 0., 0.],
                   [1., 0., 0., 0.]], dtype=np.float64)
    I4 = np.eye(4, dtype=np.float64)
    Ls_anti = [I4, -I4]
    Ls_q8 = [I4, -I4, Li, -Li, Lj, -Lj, Lk, -Lk]

    def expand_Y(Y, Ls):
        # Y (m,d) -> X (L*m, d)
        blocks = [Y @ L.T for L in Ls]
        return np.vstack(blocks)

    def sym_gd(Y0, Ls, steps, a0, a1, s0, s1, mu=0.85):
        m, d = Y0.shape
        Lnum = len(Ls)
        # stack Ls for fast fold: Ls_arr (L,d,d)
        Ls_arr = np.stack(Ls, axis=0)
        Y = Y0.copy()
        vel = np.zeros_like(Y)
        N = m * Lnum
        for it in range(steps):
            frac = it / max(steps - 1, 1)
            alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
            smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
            # expand
            # X blocks
            X = np.vstack([Y @ L.T for L in Ls])  # (N,d)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gm = float(G.max())
            # weights
            W = np.exp(np.clip(alpha * (G - gm), -50, 0))
            np.fill_diagonal(W, 0.0)
            gradX = W @ X
            # project on X tangent
            dot = np.sum(gradX * X, axis=1, keepdims=True)
            gtX = gradX - dot * X
            # fold to Y: gradY[k] = sum_t gtX[t*m+k] @ L_t
            gradY = np.zeros_like(Y)
            for t, L in enumerate(Ls):
                block = gtX[t * m:(t + 1) * m]
                gradY += block @ L
            # momentum
            vel = mu * vel + gradY
            nrm = float(np.linalg.norm(vel, axis=1).max())
            if nrm < 1e-14:
                continue
            Y = Y - (smax / nrm) * vel
            Y = _normr(Y)
        return Y

    def full_gd(X0, steps, a0, a1, s0, s1, mu=0.85):
        X = X0.copy()
        vel = np.zeros_like(X)
        for it in range(steps):
            frac = it / max(steps - 1, 1)
            alpha = a0 * (a1 / a0) ** frac if a0 != a1 else a0
            smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gm = float(G.max())
            W = np.exp(np.clip(alpha * (G - gm), -50, 0))
            np.fill_diagonal(W, 0.0)
            grad = W @ X
            dot = np.sum(grad * X, axis=1, keepdims=True)
            gt = grad - dot * X
            vel = mu * vel + gt
            nrm = float(np.linalg.norm(vel, axis=1).max())
            if nrm < 1e-14:
                continue
            X = X - (smax / nrm) * vel
            X = _normr(X)
        return X

    def lp_polish(X, deadline, maxpairs=4000):
        X = _normr(X.copy())
        n, d = X.shape
        try:
            cur = _max_cos(X)
        except Exception:
            return X, 2.0
        radius = 1e-3
        iu0 = np.triu_indices(n, 1)
        ar = np.arange(d)
        while time.time() < deadline and radius > 1e-14:
            G = X @ X.T
            c_all = G[iu0]
            m = float(c_all.max())
            thresh = m - max(20.0 * radius, 1e-9)
            sel = c_all > thresh
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > maxpairs:
                idx = np.where(sel)[0]
                order = np.argpartition(-c_all[idx], maxpairs - 1)[:maxpairs]
                keep = np.zeros_like(sel, dtype=bool)
                keep[idx[order]] = True
                sel = keep
            c = c_all[sel]
            I = iu0[0][sel]; J = iu0[1][sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = X + res.x[:-1].reshape(n, d)
            y = _normr(y)
            v = _max_cos(y)
            if v < cur:
                X, cur = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return X, cur

    def halton_sphere(n, dim=4, seed=0):
        # deterministic low-discrepancy via Halton if available else rng
        try:
            from scipy.stats import qmc
            # different scramble per seed via randomization with fixed seed
            eng = qmc.Halton(d=dim, scramble=True, seed=seed)
            u = eng.random(n=n)
            u = np.clip(u, 1e-9, 1 - 1e-9)
            try:
                from scipy.special import ndtri
                z = ndtri(u)
            except Exception:
                # logit fallback
                z = np.log(u / (1 - u))
            z = np.asarray(z, dtype=np.float64)
            # normalize
            return _normr(z)
        except Exception:
            r = np.random.default_rng(seed)
            z = r.standard_normal((n, dim))
            return _normr(z)

    def symmetrize_antipodal(X):
        # greedy pair near-antipodal to get 136 reps
        n = X.shape[0]
        used = np.zeros(n, dtype=bool)
        reps = []
        G = X @ X.T
        for i in range(n):
            if used[i]:
                continue
            # find unused j minimizing dot (most antipodal)
            best_j = -1; best_v = 2.0
            for j in range(n):
                if used[j] or j == i:
                    continue
                v = float(G[i, j])
                if v < best_v:
                    best_v = v; best_j = j
            if best_j >= 0:
                y = X[i] - X[best_j]
                nn = float(np.linalg.norm(y))
                if nn < 1e-9:
                    y = X[i]
                    nn = float(np.linalg.norm(y))
                y = y / nn
                reps.append(y)
                used[i] = True; used[best_j] = True
            else:
                reps.append(X[i] / np.linalg.norm(X[i]))
                used[i] = True
        reps = np.array(reps, dtype=np.float64)
        if len(reps) < 136:
            extra = rng.standard_normal((136 - len(reps), 4))
            reps = np.vstack([reps, _normr(extra)])
        return _normr(reps[:136])

    # ---------- symmetry-constrained global search ----------
    sym_candidates = []  # list of (val, X272)

    # one symmetrized incumbent start
    try:
        Y_sym0 = symmetrize_antipodal(best)
        X_sym0 = expand_Y(Y_sym0, Ls_anti)
        sym_candidates.append((_max_cos(X_sym0), X_sym0, Y_sym0, 'anti-sym-inc'))
    except Exception:
        pass

    # antipodal random + low-discrepancy starts
    anti_starts = []
    # low-discrepancy starts
    for s in range(3):
        try:
            Yh = halton_sphere(136, dim=4, seed=100 + s)
            anti_starts.append(Yh)
        except Exception:
            pass
    # random starts (deterministic seeds)
    for s in range(9):
        r = np.random.default_rng(1000 + s)
        anti_starts.append(_normr(r.standard_normal((136, 4))))
    # torus-structured starts
    try:
        for s in range(2):
            r = np.random.default_rng(2000 + s)
            # Hopf-like torus layers: param grid + jitter
            Yt = []
            # 136 = 8 layers x 17? approximate
            for k in range(136):
                a = (k * 2.399963 + s) % (np.pi / 2) + 0.1
                b = (k * 3.14159 * 2 / 17.0) + 0.3 * s
                c = (k * 2.399963 * 1.7) % (2 * np.pi)
                ca, sa = np.cos(a), np.sin(a)
                Yt.append([ca * np.cos(b), ca * np.sin(b), sa * np.cos(c), sa * np.sin(c)])
            Yt = np.array(Yt, dtype=np.float64)
            Yt = Yt + 0.05 * r.standard_normal((136, 4))
            anti_starts.append(_normr(Yt))
    except Exception:
        pass

    # run antipodal anneals
    for si, Y0 in enumerate(anti_starts):
        if time.time() > SYM_END:
            break
        # vary schedule slightly by index
        a0 = 25 + (si % 3) * 10
        a1 = 140 + (si % 4) * 20
        steps = 700 if si < 6 else 500
        try:
            Yopt = sym_gd(Y0, Ls_anti, steps=steps, a0=a0, a1=a1, s0=0.06, s1=0.004, mu=0.85)
            Xopt = expand_Y(Yopt, Ls_anti)
            v = _max_cos(Xopt)
            sym_candidates.append((v, Xopt, Yopt, f'anti-{si}'))
        except Exception:
            continue

    # Q8 starts: 34 seeds
    q8_starts = []
    for s in range(3):
        try:
            q8_starts.append(halton_sphere(34, dim=4, seed=300 + s))
        except Exception:
            pass
    for s in range(5):
        r = np.random.default_rng(3000 + s)
        q8_starts.append(_normr(r.standard_normal((34, 4))))
    for si, S0 in enumerate(q8_starts):
        if time.time() > SYM_END - 5:
            break
        try:
            steps = 600 if si < 4 else 400
            Sopt = sym_gd(S0, Ls_q8, steps=steps, a0=30, a1=160, s0=0.06, s1=0.004, mu=0.85)
            Xopt = expand_Y(Sopt, Ls_q8)
            v = _max_cos(Xopt)
            sym_candidates.append((v, Xopt, Sopt, f'q8-{si}'))
        except Exception:
            continue

    # sort symmetric candidates
    try:
        sym_candidates.sort(key=lambda t: t[0])
    except Exception:
        pass
    # keep top few distinct
    top_sym = sym_candidates[:3] if len(sym_candidates) >= 3 else sym_candidates

    # ---------- graduated symmetry breaking ----------
    epsilons = [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    # also include incumbent as candidate for breaking comparison
    # track global best
    for _, Xsym, _, _tag in list(top_sym):
        # start from symmetric code
        Xcur = Xsym.copy()
        # eps=0 : tiny jitter to allow leaving manifold + anneal + polish
        for ei, eps in enumerate(epsilons):
            if time.time() > BREAK_END:
                break
            try:
                r = np.random.default_rng(5000 + ei * 100 + int(abs(float(Xsym.sum())) * 1e6) % 10000)
            except Exception:
                r = np.random.default_rng(5000 + ei * 100)
            if eps == 0.0:
                Xn = Xcur + 1e-12 * r.standard_normal(Xcur.shape)
            else:
                Xn = Xcur + eps * r.standard_normal(Xcur.shape)
            Xn = _normr(Xn)
            # re-anneal asymmetric
            try:
                # shorter for small eps, longer for large
                steps = 350 if eps < 1e-3 else 500
                Xa = full_gd(Xn, steps=steps, a0=40, a1=220, s0=0.03, s1=0.003, mu=0.8)
            except Exception:
                Xa = Xn
            va = _max_cos(Xa)
            # LP polish with budget depending on promise
            budget = 1.5
            if va < best_val + 0.002:
                budget = 2.5
            else:
                budget = 0.8
            # cap by remaining
            rem = BREAK_END - (time.time())
            if rem < 0.5:
                break
            budget = min(budget, max(0.4, rem - 0.2))
            try:
                Xp, vp = lp_polish(Xa, time.time() + budget)
            except Exception:
                Xp, vp = Xa, va
            if vp < best_val - 1e-12 and np.all(np.isfinite(Xp)):
                best = Xp.copy(); best_val = vp
                Xcur = Xp.copy()
            else:
                # even if not beating global, continue chain from improved Xcur if better than Xcur
                try:
                    vcur = _max_cos(Xcur)
                    if vp < vcur:
                        Xcur = Xp.copy()
                except Exception:
                    pass
            # also opportunistic: if Xcur already beats, keep
        if time.time() > BREAK_END:
            break

    # ---------- final full-asymmetric refinement of incumbent ----------
    # a few direct anneal+polish cycles on best to squeeze
    while time.time() < t0 + HARD - 12.0:
        rem = (t0 + HARD - 2.0) - time.time()
        if rem < 3.0:
            break
        try:
            # small shake to escape tiny basin? use very small noise
            r = np.random.default_rng(int(time.time() * 1000) % (2 ** 31 - 1))
            # alternate between pure polish and small anneal
            Xa = full_gd(best, steps=250, a0=60, a1=250, s0=0.015, s1=0.002, mu=0.8)
            va = _max_cos(Xa)
            if va < best_val + 0.005:
                Xp, vp = lp_polish(Xa, time.time() + min(3.0, rem - 1.0))
                if vp < best_val - 1e-12:
                    best = Xp; best_val = vp
                    continue
            # direct polish of best
            Xp2, vp2 = lp_polish(best, time.time() + min(2.0, rem - 1.0))
            if vp2 < best_val - 1e-12:
                best = Xp2; best_val = vp2
            else:
                break
        except Exception:
            break

    # final safety polish with remaining time
    remain = (t0 + HARD) - time.time()
    if remain > 3.0:
        try:
            best, best_val = lp_polish(best, time.time() + min(remain - 1.0, 8.0))
        except Exception:
            pass
    best = _normr(best)
    bad = ~np.all(np.isfinite(best), axis=1)
    if np.any(bad):
        rr = np.random.default_rng(999)
        rnd = rr.standard_normal((int(np.sum(bad)), 4))
        best[bad] = _normr(rnd)
    return best
