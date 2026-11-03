import time
import numpy as np

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    n, d = 93, 13
    rng = np.random.default_rng(20261102)

    def _norm(X):
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return X / nr

    def _maxcos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    iu0_all, iu1_all = np.triu_indices(n, 1)

    # load baseline
    base = None
    base_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < base_val:
                base_val = v
                base = a.copy()
        except Exception:
            pass
    if base is None:
        base = _norm(rng.standard_normal((n, d)))
        base_val = _maxcos(base)
    gbest = base.copy()
    gval = float(base_val)

    # ---------- C93 frequency search with exact LP over weights ----------
    # param: s_j>=0 sum=1, f_t(s)=sum_{j<6} s_j cos(2pi a_j t/93)+s_6*1
    best_freqs = []
    try:
        from scipy.optimize import linprog
        nfreq_target = 320
        c_obj = np.zeros(8); c_obj[-1] = 1.0
        A_eq = np.zeros((1, 8)); A_eq[0, :7] = 1.0
        b_eq = np.array([1.0])
        bnds = [(0.0, 1.0)] * 7 + [(-1.0, 1.0)]
        t_vals = np.arange(1, 93)
        results = []
        n_tried = 0
        # fixed seed sampling
        frng = np.random.default_rng(777)
        # include a few structured sets
        preset = []
        while n_tried < nfreq_target:
            if time.time() - t0 > 28.0:
                break
            if n_tried < 8:
                # evenly spaced / random structured
                a = np.sort(frng.choice(np.arange(1, 47), size=6, replace=False))
            else:
                a = np.sort(frng.choice(np.arange(1, 47), size=6, replace=False))
            # avoid duplicate sets (cheap check)
            n_tried += 1
            # build A_mat 92x7
            # cos table
            A_mat = np.empty((92, 7), dtype=np.float64)
            for j, aj in enumerate(a):
                A_mat[:, j] = np.cos(2.0 * np.pi * float(aj) * t_vals / 93.0)
            A_mat[:, 6] = 1.0
            A_ub = np.hstack([A_mat, -np.ones((92, 1))])
            b_ub = np.zeros(92)
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                              bounds=bnds, method='highs',
                              options={'presolve': True})
            except Exception:
                continue
            if res is None or res.x is None:
                continue
            if not np.all(np.isfinite(res.x)):
                continue
            T = float(res.x[-1])
            s = np.asarray(res.x[:7], dtype=np.float64)
            # clip
            s = np.clip(s, 0, 1)
            if s.sum() > 0:
                s = s / s.sum()
            results.append((T, tuple(int(x) for x in a), s.copy()))
        results.sort(key=lambda x: x[0])
        best_freqs = results[:6]
    except Exception:
        best_freqs = []

    def build_symmetric(a_tuple, s_vec, phases=None):
        a = np.asarray(a_tuple, dtype=np.float64)
        r = np.sqrt(np.clip(s_vec[:6], 0, 1))
        c = float(np.sqrt(np.clip(s_vec[6], 0, 1)))
        # allow negative c? keep positive
        if phases is None:
            ph = np.zeros(6)
        else:
            ph = np.asarray(phases, dtype=np.float64)
        ks = np.arange(93)[:, None]  # 93x1
        # angles 93x6
        ANG = 2.0 * np.pi * ks * a[None, :] / 93.0 + ph[None, :]
        X = np.empty((93, 13), dtype=np.float64)
        for j in range(6):
            X[:, 2 * j] = r[j] * np.cos(ANG[:, j])
            X[:, 2 * j + 1] = r[j] * np.sin(ANG[:, j])
        X[:, 12] = c
        return _norm(X)

    # ---------- full-coordinate annealed Riemannian Adam ----------
    def adam_refine(Xinit, stages, t0, deadline):
        X = _norm(Xinit.copy())
        N = X.shape[0]
        iu0, iu1 = np.triu_indices(N, 1)
        m_ad = np.zeros_like(X)
        v_ad = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        for (tau, steps, lr) in stages:
            for s in range(steps):
                if time.time() - t0 > deadline:
                    return X
                G = X @ X.T
                cc = G[iu0, iu1]
                mx = float(cc.max())
                E = np.exp((cc - mx) / tau)
                S = float(E.sum()) + 1e-300
                W = np.zeros((N, N), dtype=np.float64)
                W[iu0, iu1] = E
                W[iu1, iu0] = E
                grad = (W @ X) / S
                dot = np.einsum('ij,ij->i', grad, X)[:, None]
                gt = grad - dot * X
                tt += 1
                m_ad = b1 * m_ad + (1 - b1) * gt
                v_ad = b2 * v_ad + (1 - b2) * gt * gt
                mh = m_ad / (1.0 - b1 ** tt)
                vh = v_ad / (1.0 - b2 ** tt)
                upd = mh / (np.sqrt(vh) + eps)
                dd = np.einsum('ij,ij->i', upd, X)[:, None]
                upd_t = upd - dd * X
                X = _norm(X - lr * upd_t)
        return X

    STAGES_FAST = [(0.015, 200, 0.02), (0.007, 200, 0.012), (0.003, 200, 0.008), (0.0012, 150, 0.005)]
    STAGES_FINE = [(0.006, 150, 0.01), (0.0025, 150, 0.007), (0.001, 150, 0.005), (0.0004, 120, 0.003)]

    # evaluate symmetric candidates with symmetry-breaking
    cand_inits = []
    for idx, (T, a_tup, s_vec) in enumerate(best_freqs[:4]):
        try:
            Xsym = build_symmetric(a_tup, s_vec, phases=np.zeros(6))
            cand_inits.append(Xsym)
        except Exception:
            continue

    # perturb + refine candidates
    prng = np.random.default_rng(4242)
    for ci, Xsym in enumerate(cand_inits):
        if time.time() - t0 > DEADLINE - 90:
            break
        for rep in range(2):
            if time.time() - t0 > DEADLINE - 90:
                break
            sig = 0.015 if rep == 0 else 0.035
            # random phases also break symmetry equivalently; use noise
            Xn = _norm(Xsym + sig * prng.standard_normal((n, d)))
            # fast anneal
            Xr = adam_refine(Xn, STAGES_FAST, t0, DEADLINE - 85)
            v = _maxcos(Xr)
            if v < gval:
                gval = v
                gbest = Xr.copy()
            # extra fine on promising
            if v < gval + 0.004:
                Xr2 = adam_refine(Xr, STAGES_FINE, t0, DEADLINE - 85)
                v2 = _maxcos(Xr2)
                if v2 < gval:
                    gval = v2
                    gbest = Xr2.copy()

    # ---------- also refine baseline basin (safety + improvement) ----------
    if time.time() - t0 < DEADLINE - 60:
        try:
            # small shake of best to escape 2-pair plateau but stay in basin
            for rep in range(3):
                if time.time() - t0 > DEADLINE - 45:
                    break
                sig = [0.004, 0.010, 0.020][rep]
                Xs = _norm(base + sig * prng.standard_normal((n, d)))
                Xr = adam_refine(Xs, STAGES_FAST, t0, DEADLINE - 40)
                Xr = adam_refine(Xr, STAGES_FINE, t0, DEADLINE - 40)
                v = _maxcos(Xr)
                if v < gval:
                    gval = v
                    gbest = Xr.copy()
            # pure polish of base without shake
            if time.time() - t0 < DEADLINE - 45:
                Xr0 = adam_refine(base, STAGES_FINE, t0, DEADLINE - 40)
                v0 = _maxcos(Xr0)
                if v0 < gval:
                    gval = v0
                    gbest = Xr0.copy()
        except Exception:
            pass

    # ---------- tangent LP polish on current best ----------
    def lp_polish(Xinit, budget_s, t0, deadline):
        try:
            from scipy.optimize import linprog
            from scipy.sparse import coo_matrix
        except Exception:
            return Xinit
        X = _norm(Xinit.copy())
        cur = _maxcos(X)
        N = X.shape[0]
        iu0_, iu1_ = np.triu_indices(N, 1)
        M = len(iu0_)
        dm = d - 1
        Nvar = N * dm + 1
        ar = np.arange(dm)
        radius = 3e-3
        c_obj = np.zeros(Nvar); c_obj[-1] = 1.0
        t_start = time.time()
        it = 0
        while True:
            if time.time() - t0 > deadline:
                break
            if time.time() - t_start > budget_s:
                break
            it += 1
            if radius < 1e-12 or it > 18:
                break
            G = X @ X.T
            c_all = G[iu0_, iu1_]
            cmax = float(c_all.max())
            K = min(700, M)
            order = np.argpartition(-c_all, K - 1)[:K]
            thr = cmax - 2e-3
            mask = np.zeros(M, dtype=bool)
            mask[order] = True
            mask |= (c_all >= thr)
            I = iu0_[mask]; J = iu1_[mask]; csub = c_all[mask]
            p = len(I)
            if p == 0:
                break
            T = np.empty((N, d, dm))
            for ii in range(N):
                xi = X[ii]
                k = int(np.argmin(np.abs(xi)))
                B = np.empty((d, dm))
                col = 0
                for jj in range(d):
                    if jj == k:
                        continue
                    vv = -xi * xi[jj]; vv[jj] += 1.0
                    B[:, col] = vv; col += 1
                Qq, _ = np.linalg.qr(B)
                T[ii] = Qq
            TI = T[I]; TJ = T[J]
            XI = X[I]; XJ = X[J]
            CI = np.einsum('pdm,pd->pm', TI, XJ)
            CJ = np.einsum('pdm,pd->pm', TJ, XI)
            nnz = p * (2 * dm + 1)
            rows = np.empty(nnz, dtype=np.int64)
            cols = np.empty(nnz, dtype=np.int64)
            data = np.empty(nnz, dtype=np.float64)
            r1 = np.repeat(np.arange(p), dm)
            c1 = (I[:, None] * dm + ar[None, :]).ravel()
            c2 = (J[:, None] * dm + ar[None, :]).ravel()
            rows[0:p*dm] = r1; cols[0:p*dm] = c1; data[0:p*dm] = CI.ravel()
            rows[p*dm:2*p*dm] = r1; cols[p*dm:2*p*dm] = c2; data[p*dm:2*p*dm] = CJ.ravel()
            rows[2*p*dm:2*p*dm+p] = np.arange(p); cols[2*p*dm:2*p*dm+p] = Nvar-1; data[2*p*dm:2*p*dm+p] = -1.0
            A = coo_matrix((data, (rows, cols)), shape=(p, Nvar)).tocsr()
            b = -csub
            bnds = [(-radius, radius)] * (N * dm) + [(cmax - 0.05, cmax + 0.005)]
            try:
                res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bnds, method='highs',
                              options={'presolve': True})
            except Exception:
                break
            if not res.success or res.x is None:
                radius *= 0.5
                continue
            y = res.x[:N*dm].reshape(N, dm)
            dX = np.einsum('idm,im->id', T, y)
            Xn = _norm(X + dX)
            vn = _maxcos(Xn)
            if vn < cur - 1e-12:
                cur = vn
                X = Xn
                radius = min(radius * 1.5, 1e-2)
            else:
                radius *= 0.5
        return X

    if time.time() - t0 < DEADLINE - 12:
        try:
            Xp = lp_polish(gbest, 16.0, t0, DEADLINE - 6)
            vp = _maxcos(Xp)
            if vp < gval and np.all(np.isfinite(Xp)):
                gval = vp
                gbest = Xp.copy()
            # final tiny Adam after LP
            if time.time() - t0 < DEADLINE - 8:
                Xf = adam_refine(gbest, [(0.001, 80, 0.004), (0.0004, 60, 0.002)], t0, DEADLINE - 2)
                vf = _maxcos(Xf)
                if vf < gval:
                    gval = vf
                    gbest = Xf.copy()
        except Exception:
            pass

    # safety: if somehow worse than base, return base
    try:
        if _maxcos(gbest) > base_val and np.all(np.isfinite(base)):
            gbest = base.copy()
    except Exception:
        pass
    gbest = _norm(np.asarray(gbest, dtype=np.float64))
    # shrink tiny margin to enforce strict feasibility (no-op for cosine but avoid nan)
    return gbest
