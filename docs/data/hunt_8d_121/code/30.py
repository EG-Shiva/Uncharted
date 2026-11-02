import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    np.random.seed(0)
    n, d = 121, 8
    dm = d - 1
    N = n * dm
    iu = np.triu_indices(n, 1)
    Iall = iu[0]; Jall = iu[1]

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        rng0 = np.random.default_rng(0)
        X0 = rng0.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))

    def lp_polish(x, seconds, radius_init=4e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            it += 1
            if it > 200 or radius < 1e-14:
                break
            G = x @ x.T
            cf = G[iu]
            m = float(cf.max())
            thr = max(20.0 * radius, 2e-9)
            sel = cf > m - thr
            cnt = int(sel.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 2500:
                kth = np.sort(cf)[-2500]
                sel = cf >= kth
                cnt = int(sel.sum())
                if cnt > 3000:
                    idxp = np.argpartition(-cf, 3000)[:3000]
                    mk = np.zeros_like(sel)
                    mk[idxp] = True
                    sel = mk
                    cnt = 3000
            c = cf[sel]
            I = iu[0][sel]; J = iu[1][sel]
            p = cnt
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            valsA = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(),
                                   (np.repeat(np.arange(n), d), np.arange(n * d))),
                                  shape=(n, nv)).tocsr()
            except Exception:
                radius *= 0.5
                continue
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            yn = np.linalg.norm(y, axis=1, keepdims=True)
            yn[yn == 0] = 1.0
            y = y / yn
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.6, 0.02)
            else:
                radius *= 0.5
        return x, best

    def make_fun(beta):
        b = float(beta)
        def fun(flat):
            Xr = flat.reshape(n, d)
            nr = np.linalg.norm(Xr, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = Xr / nr
            G = Y @ Y.T
            c = G[iu]
            m = float(c.max())
            e = np.exp(b * (c - m))
            s = float(e.sum())
            if not np.isfinite(s) or s <= 0:
                return float(m), np.zeros_like(flat)
            f = m + np.log(s) / b
            w = e / s
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(f), GX.ravel()
        return fun

    def lbfgs_anneal(X, betas, maxiters):
        Xf = X.ravel().copy()
        for beta, mi in zip(betas, maxiters):
            if time.time() > DEADLINE - 8.0:
                break
            fun = make_fun(float(beta))
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-14, "maxfun": int(mi * 6)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        Xn = Xf.reshape(n, d)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        return Xn

    def tangent_bases(X):
        T = np.empty((n, d, dm))
        for i in range(n):
            x = X[i]
            k = int(np.argmax(np.abs(x)))
            V = np.zeros((d, dm))
            col = 0
            for j in range(d):
                if j == k:
                    continue
                v = -x[j] * x
                v[j] += 1.0
                V[:, col] = v
                col += 1
            Q, _ = np.linalg.qr(V)
            T[i] = Q
        return T

    def build_A(X, T, I, J):
        p = len(I)
        A = np.zeros((p, N))
        for r in range(p):
            i = int(I[r]); j = int(J[r])
            ai = T[i].T @ X[j]
            aj = T[j].T @ X[i]
            A[r, i * dm:(i + 1) * dm] = ai
            A[r, j * dm:(j + 1) * dm] = aj
        return A

    def equilibrium_mask(A, slack):
        p = A.shape[0]
        # minimise slack^T lam s.t. A^T lam=0, sum=1, lam>=0  (dual of polish LP)
        obj = slack * 1e6 + 1e-9
        try:
            Aeq = np.zeros((N + 1, p))
            Aeq[:N, :] = A.T
            Aeq[N, :] = 1.0
            beq = np.zeros(N + 1); beq[N] = 1.0
            res = linprog(obj, A_eq=Aeq, b_eq=beq, bounds=(0, None), method="highs",
                          options={"time_limit": 4.0, "primal_feasibility_tolerance": 1e-9})
            if res.status == 0 and np.all(np.isfinite(res.x)):
                lam = np.asarray(res.x, dtype=float)
                lam[lam < 0] = 0
                s = lam.sum()
                if s > 0:
                    lam /= s
                    mask = lam > 1e-7
                    if int(mask.sum()) >= 10:
                        return mask, lam
        except Exception:
            pass
        return np.ones(p, dtype=bool), None

    if DEADLINE - time.time() > 6:
        Xp, Vp = lp_polish(Xbest, seconds=2.5, radius_init=1e-4)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp

    Xcur = Xbest.copy()
    Vcur = Vbest
    # contact set of current walk point for topology comparison
    def contact_set(X, thr=1e-4):
        G = X @ X.T
        cf = G[iu]
        m = float(cf.max())
        return set(np.where(cf > m - thr)[0].tolist()), m

    cur_set, cur_m = contact_set(Xcur)
    STEPS = [0.01, 0.02, 0.04, 0.07, 0.10, 0.15, 0.20]
    outer = 0
    rng = np.random.default_rng(12345)

    while time.time() < DEADLINE - 14.0 and outer < 30:
        outer += 1
        if time.time() > DEADLINE - 14.0:
            break
        X = Xcur.copy()
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        Gall = G[iu]
        m0 = float(Gall.max())
        slack_all = m0 - Gall
        # assemble Jacobian of pairs with slack < 1e-5, cap to K closest
        K = 950
        order = np.argpartition(slack_all, min(K, len(slack_all) - 1))[:K]
        # sort those K by slack
        order = order[np.argsort(slack_all[order])]
        IK = iu[0][order]; JK = iu[1][order]
        slackK = slack_all[order]
        cK = Gall[order]
        T = tangent_bases(X)
        A = build_A(X, T, IK, JK)
        mask, lam = equilibrium_mask(A, slackK)
        As = A[mask]
        # SVD of stressed Jacobian to get near-nullspace
        try:
            # full_matrices to expose exact null when p<N
            U, S, Vh = np.linalg.svd(As, full_matrices=True)
        except Exception:
            # fallback: random tangent kick + anneal
            Xn = X + 0.01 * rng.standard_normal((n, d))
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            Xn = lbfgs_anneal(Xn, [4000.0, 30000.0], [80, 80])
            v = _max_cos(Xn)
            Xn, v = lp_polish(Xn, seconds=0.6, radius_init=4e-4)
            if v < Vbest:
                Xbest, Vbest = Xn, v
                Xcur, Vcur = Xn.copy(), v
                cur_set, cur_m = contact_set(Xcur)
            continue
        p_tot = As.shape[0]
        # S length = min(p_tot,N)
        ns = len(S)
        tol = 1e-6
        small_idx = np.where(S < tol)[0]
        if p_tot < N:
            # extra exact null rows: Vh[p_tot:] (Vh is N x N when full)
            # Vh shape is (N,N) if full
            if Vh.shape[0] == N:
                null_extra = Vh[p_tot:, :]
                if len(small_idx) > 0:
                    null_small = Vh[small_idx, :]
                    null_mat = np.vstack([null_small, null_extra])
                else:
                    null_mat = null_extra
            else:
                null_mat = Vh[small_idx, :] if len(small_idx) > 0 else np.zeros((0, N))
        else:
            null_mat = Vh[small_idx, :] if len(small_idx) > 0 else np.zeros((0, N))
        n_null = null_mat.shape[0]
        if n_null == 0:
            # take 6 softest modes
            k = min(6, Vh.shape[0])
            null_mat = Vh[ns - k:ns, :] if ns >= k else Vh[-k:, :]
            n_null = null_mat.shape[0]
        # cap huge null for speed (random subspace)
        if n_null > 250:
            sel = rng.choice(n_null, size=250, replace=False)
            null_mat = null_mat[sel, :]
            n_null = 250
        n_dir = 4 if outer < 20 else 3
        # precompute per-direction unit displacements
        dir_DX = []
        for dd in range(n_dir):
            coeff = rng.standard_normal(n_null)
            y = coeff @ null_mat  # (N,)
            ynorm = float(np.linalg.norm(y))
            if ynorm < 1e-12:
                coeff = rng.standard_normal(n_null)
                y = coeff @ null_mat
                ynorm = float(np.linalg.norm(y)) + 1e-12
            y = y * (np.sqrt(n) / ynorm)  # RMS=1
            # map to DX (n x d)
            DX = np.zeros((n, d))
            for i in range(n):
                DX[i] = T[i] @ y[i * dm:(i + 1) * dm]
            # normalise RMS exactly
            rms = float(np.sqrt((DX ** 2).sum() / n))
            if rms > 1e-12:
                DX /= rms
            dir_DX.append(DX)
        # line search on sphere, monitor true max + contacts crossing 1e-4
        cand_steps = []  # (m_new, count_new, dd, s, Xnew)
        rnorms = [np.linalg.norm(DX, axis=1) for DX in dir_DX]
        for dd, DX in enumerate(dir_DX):
            rn = rnorms[dd]
            inv = np.zeros_like(rn)
            nz = rn > 1e-14
            inv[nz] = 1.0 / rn[nz]
            Ux = DX * inv[:, None]
            for s in STEPS:
                if time.time() > DEADLINE - 13.0:
                    break
                ang = s * rn
                ca = np.cos(ang)[:, None]
                sa = np.sin(ang)[:, None]
                Xn = ca * X + sa * Ux
                # rows with rn~0 stay
                Xn[~nz] = X[~nz]
                Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
                Gn = Xn @ Xn.T
                np.fill_diagonal(Gn, -2.0)
                mn = float(Gn.max())
                cnt4 = int((m0 - Gn[iu] < 1e-4).sum()) if False else int(((mn - Gn[iu]) < 1e-4).sum())
                cand_steps.append((mn, cnt4, dd, s, Xn))
        # choose steps to polish: prefer low mn but include topology diversity
        cand_steps.sort(key=lambda t: t[0])
        # keep those not exploding
        filt = [t for t in cand_steps if t[0] < m0 + 0.015]
        if not filt:
            filt = cand_steps[:3]
        # pick up to 6: best 2 + spaced larger steps
        picks = []
        picks.append(filt[0])
        # add a medium and a large step from different direction if possible
        for t in filt[1:]:
            if len(picks) >= 6:
                break
            # enforce some diversity in (dd,s)
            if any(abs(t[3] - q[3]) < 1e-12 and t[2] == q[2] for q in picks):
                continue
            # require contact count change or step>=0.07 for topology swap
            if t[3] >= 0.07 or abs(t[1] - int((slack_all < 1e-4).sum())) > 5 or len(picks) < 2:
                picks.append(t)
        best_pol = None
        best_pol_v = 1e9
        for (mn, cnt4, dd, s, Xn) in picks:
            if time.time() > DEADLINE - 12.0:
                break
            Xq, vq = lp_polish(Xn, seconds=0.65, radius_init=5e-4)
            # light LSE touch if close to best
            if vq < Vbest + 0.002 and time.time() < DEADLINE - 11.0:
                Xq2 = lbfgs_anneal(Xq, [12000.0], [60])
                v2 = _max_cos(Xq2)
                Xq2, v2 = lp_polish(Xq2, seconds=0.4, radius_init=3e-4)
                if v2 < vq:
                    Xq, vq = Xq2, v2
            if vq < best_pol_v:
                best_pol_v = vq
                best_pol = (Xq, vq, dd, s, mn, cnt4)
            if vq < Vbest - 1e-12:
                Xbest, Vbest = Xq.copy(), float(vq)
        if best_pol is not None:
            Xq, vq, dd, s, mn, cnt4 = best_pol
            # accept sideways move if within tolerance to keep walking new basin
            qset, _ = contact_set(Xq)
            inter = len(qset & cur_set)
            union = len(qset | cur_set) + 1e-9
            jacc = inter / union
            if vq < Vbest + 1e-12:
                # already updated best; move walk point
                Xcur, Vcur = Xbest.copy(), Vbest
                cur_set, cur_m = contact_set(Xcur)
            elif vq < Vcur + 5e-4 and jacc < 0.92:
                Xcur, Vcur = Xq.copy(), float(vq)
                cur_set, cur_m = qset, float(vq)
            elif vq < Vcur - 1e-12:
                Xcur, Vcur = Xq.copy(), float(vq)
                cur_set, cur_m = qset, float(vq)
            else:
                # no progress: randomise walk slightly along floppy + small noise
                if outer % 3 == 0:
                    DX = dir_DX[0]
                    Xn = Xcur + 0.005 * rng.standard_normal((n, d))
                    Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
                    Xcur = Xn
                    Vcur = _max_cos(Xcur)
        else:
            # fallback perturbation
            Xn = Xcur + 0.008 * rng.standard_normal((n, d))
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            Xcur = Xn
            Vcur = _max_cos(Xcur)

    # final intensive polish of best
    if time.time() < DEADLINE - 9.0:
        Xbest, Vbest = lp_polish(Xbest, seconds=min(4.0, max(1.0, DEADLINE - time.time() - 7.0)), radius_init=3e-4)
    if time.time() < DEADLINE - 5.0:
        Xt = lbfgs_anneal(Xbest, [3000.0, 20000.0, 80000.0], [120, 120, 120])
        vt = _max_cos(Xt)
        Xt, vt = lp_polish(Xt, seconds=max(0.5, DEADLINE - time.time() - 2.0), radius_init=3e-4)
        if vt < Vbest:
            Xbest, Vbest = Xt, vt
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
