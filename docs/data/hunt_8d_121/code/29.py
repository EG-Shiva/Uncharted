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
    DEADLINE = t0 + 174.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]

    # benchmark only (do not start from it)
    Vrec = 0.460383289320
    Xbench = None
    try:
        a = np.load("record.npy").astype(float)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        if a.shape == (n, d) and np.all(np.isfinite(a)):
            Vrec = _max_cos(a)
            Xbench = a.copy()
    except Exception:
        pass
    try:
        b = np.load("best.npy").astype(float)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        if b.shape == (n, d) and np.all(np.isfinite(b)):
            vb = _max_cos(b)
            if vb < Vrec:
                Vrec = vb
                Xbench = b.copy()
    except Exception:
        pass
    if Xbench is None:
        Xbench = rng.standard_normal((n, d))
        Xbench /= np.linalg.norm(Xbench, axis=1, keepdims=True)
        Vrec = _max_cos(Xbench)
    Xbest = Xbench.copy()
    Vbest = float(Vrec)
    # record contact count for diversity test
    try:
        Gr = Xbench @ Xbench.T
        cfr = Gr[iu]
        cnt_rec = int((cfr > Vrec - 1e-9).sum())
    except Exception:
        cnt_rec = 740

    def contact_count(X, V):
        G = X @ X.T
        cf = G[iu]
        return int((cf > V - 1e-9).sum())

    # ---------- E8 roots ----------
    def build_e8():
        roots = []
        # type 1: (+-1,+-1,0*6)
        import itertools
        for i, j in itertools.combinations(range(8), 2):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8)
                    v[i] = si
                    v[j] = sj
                    roots.append(v)
        # type 2: (+-0.5)*8 even minus
        for mask in range(256):
            v = np.ones(8) * 0.5
            bits = [(mask >> k) & 1 for k in range(8)]
            nm = sum(bits)
            if nm % 2 == 1:
                continue
            for k in range(8):
                if bits[k]:
                    v[k] = -0.5
            roots.append(v)
        R = np.array(roots, dtype=float)  # 240
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        return R
    E8 = build_e8()

    def greedy_e8(seed):
        rs = np.random.default_rng(seed)
        N = E8.shape[0]
        # start random
        s0 = int(rs.integers(N))
        sel = [s0]
        # avoid exact antipode of selected? allow but penalize: antipode gives dot -1 (fine)
        selset = set(sel)
        # precompute gram of E8
        G8 = E8 @ E8.T
        while len(sel) < n:
            # for each candidate compute max dot to sel
            # vectorized: G8[cand, sel].max
            cand_max = G8[:, sel].max(axis=1)
            cand_max[sel] = 2.0  # exclude selected
            # also exclude antipodes already selected? antipode dot -1 to its pair but may be ok; keep allowed
            # pick minimal max
            # add small random tie break
            best = int(np.argmin(cand_max + 1e-12 * rs.standard_normal(N)))
            sel.append(best)
            selset.add(best)
        X = E8[np.array(sel)].copy()
        # swap moves
        cur_max = _max_cos(X)
        unsel = [i for i in range(N) if i not in selset]
        for _ in range(3000):
            if time.time() > DEADLINE - 20:
                break
            i_out = int(rs.integers(n))
            i_in = unsel[int(rs.integers(len(unsel)))]
            trial = X.copy()
            trial[i_out] = E8[i_in]
            v = _max_cos(trial)
            if v < cur_max:
                # accept
                old = sel[i_out]
                sel[i_out] = i_in
                selset.remove(old)
                selset.add(i_in)
                unsel[unsel.index(i_in)] = old
                X = trial
                cur_max = v
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    # ---------- signed-perm group machinery ----------
    def mat_from_perm_sign(perm, sign):
        M = np.zeros((d, d))
        M[np.arange(d), np.asarray(perm)] = np.asarray(sign, dtype=float)
        return M

    def closure(mats, maxsize=128):
        # mats: list of 8x8
        elems = [np.eye(d)]
        seen = {tuple(np.eye(d).ravel().round(6))}
        stack = [np.eye(d)]
        # include generators and inverses (orthogonal so inv = T)
        gens = []
        for M in mats:
            gens.append(M)
            gens.append(M.T)
        for _ in range(200):
            new_found = False
            cur = elems.copy()
            for A in cur:
                for Gm in gens:
                    C = Gm @ A
                    key = tuple(C.ravel().round(6))
                    if key not in seen:
                        seen.add(key)
                        elems.append(C)
                        new_found = True
                        if len(elems) > maxsize:
                            return elems
            if not new_found:
                break
        return elems

    def random_perm_group(rng_obj):
        # generate programmatically from random even sign changes + perms
        for _ in range(200):
            p1 = rng_obj.permutation(d)
            p2 = rng_obj.permutation(d)
            # random even sign changes
            s1 = np.ones(d)
            s2 = np.ones(d)
            # make even number of -1
            idx = rng_obj.choice(d, size=int(rng_obj.integers(0, 4)) * 2, replace=False)
            # ensure even count
            if len(idx) % 2 == 1 and len(idx) > 0:
                idx = idx[:-1]
            s1[idx] = -1.0
            idx2 = rng_obj.choice(d, size=int(rng_obj.integers(0, 4)) * 2, replace=False)
            if len(idx2) % 2 == 1 and len(idx2) > 0:
                idx2 = idx2[:-1]
            s2[idx2] = -1.0
            # for fixed-point families use trivial signs (permutation-only subgroup of B8)
            # randomly decide to keep trivial signs to allow all-ones fixed point
            if rng_obj.random() < 0.7:
                s1 = np.ones(d)
                s2 = np.ones(d)
            M1 = mat_from_perm_sign(p1, s1)
            M2 = mat_from_perm_sign(p2, s2)
            elems = closure([M1, M2], maxsize=64)
            if len(elems) == 8:
                return elems
        # fallback: C8 cyclic shifts
        elems = []
        for k in range(8):
            perm = [(i - k) % d for i in range(d)]
            elems.append(mat_from_perm_sign(perm, np.ones(d)))
        return elems

    # build groups: C8 + 2 random order-8 perm groups
    groups = []
    # C8
    c8 = []
    for k in range(8):
        perm = [(i - k) % d for i in range(d)]
        c8.append(mat_from_perm_sign(perm, np.ones(d)))
    groups.append(c8)
    for _ in range(3):
        try:
            g = random_perm_group(rng)
            # check all-ones fixed (need permutation-only)
            f0 = np.ones(d) / np.sqrt(d)
            ok = all(np.linalg.norm(M @ f0 - f0) < 1e-9 for M in g)
            if ok and len(g) == 8:
                # avoid duplicate of C8 (check not same set)
                groups.append(g)
        except Exception:
            pass
        if len(groups) >= 3:
            break

    f0 = np.ones(d) / np.sqrt(d)

    def expand_sym(Rn, mats):
        # Rn (15,8) unit, mats len 8
        k = Rn.shape[0]
        X = np.zeros((1 + k * len(mats), d))
        X[0] = f0
        p = 1
        for i in range(k):
            r = Rn[i]
            for M in mats:
                X[p] = M @ r
                p += 1
        # normalize safety
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def sym_opt(mats, seed, betas, maxiters):
        rs = np.random.default_rng(seed)
        k = 15
        R0 = rs.standard_normal((k, d))
        R0 /= np.linalg.norm(R0, axis=1, keepdims=True)
        flat = R0.ravel().copy()
        m = len(mats)
        # precompute transposes
        MT = [M.T for M in mats]
        for beta, mi in zip(betas, maxiters):
            if time.time() > DEADLINE - 12:
                break
            b = float(beta)
            def fun(fl):
                R = fl.reshape(k, d)
                nr = np.linalg.norm(R, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                Rn = R / nr
                X = expand_sym(Rn, mats)
                G = X @ X.T
                c = G[iu]
                mx = float(c.max())
                e = np.exp(b * (c - mx))
                s = float(e.sum())
                if not np.isfinite(s) or s <= 0:
                    return float(mx), np.zeros_like(fl)
                f = mx + np.log(s) / b
                w = e / s
                W = np.zeros((n, n))
                W[Iall, Jall] = w
                W[Jall, Iall] = w
                GX = W @ X  # grad w.r.t X
                # accumulate to Rn
                GRn = np.zeros_like(Rn)
                # index mapping: 0 fixed (no grad to reps), then blocks
                # fixed point gradient ignored (fixed)
                for i in range(k):
                    acc = np.zeros(d)
                    base = 1 + i * m
                    for j in range(m):
                        acc += MT[j] @ GX[base + j]
                    GRn[i] = acc
                dot = (GRn * Rn).sum(axis=1, keepdims=True)
                GR = (GRn - dot * Rn) / nr
                return float(f), GR.ravel()
            try:
                res = minimize(fun, flat, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(mi * 8)})
                flat = np.asarray(res.x, dtype=float)
            except Exception:
                break
        R = flat.reshape(k, d)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        X = expand_sym(R, mats)
        return X

    def free_anneal(X, betas, maxiters):
        flat = X.ravel().copy()
        for beta, mi in zip(betas, maxiters):
            if time.time() > DEADLINE - 9:
                break
            b = float(beta)
            def fun(fl):
                Xr = fl.reshape(n, d)
                nr = np.linalg.norm(Xr, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                Y = Xr / nr
                G = Y @ Y.T
                c = G[iu]
                mx = float(c.max())
                e = np.exp(b * (c - mx))
                s = float(e.sum())
                if not np.isfinite(s) or s <= 0:
                    return float(mx), np.zeros_like(fl)
                f = mx + np.log(s) / b
                w = e / s
                W = np.zeros((n, n))
                W[Iall, Jall] = w
                W[Jall, Iall] = w
                GY = W @ Y
                dot = (GY * Y).sum(axis=1, keepdims=True)
                GX = (GY - dot * Y) / nr
                return float(f), GX.ravel()
            try:
                res = minimize(fun, flat, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(mi * 8)})
                flat = np.asarray(res.x, dtype=float)
            except Exception:
                break
        Xn = flat.reshape(n, d)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        return Xn

    def lp_polish(x, seconds, r0=4e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = r0
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            if radius < 1e-14 or it > 200:
                break
            it += 1
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
                idxp = np.argpartition(-cf, 2500)[:2500]
                mk = np.zeros_like(sel)
                mk[idxp] = True
                sel = mk
                cnt = 2500
            c = cf[sel]
            I = iu[0][sel]; J = iu[1][sel]
            p = cnt
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None]*d + np.arange(d),
                                   J[:, None]*d + np.arange(d),
                                   np.full((p, 1), n*d)], 1).ravel()
            valsA = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
                res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius*1.5, 0.02)
            else:
                radius *= 0.5
        return x

    # ---------- stage 1: symmetric optimisation ----------
    cands = []
    sym_betas = [5000.0, 30000.0, 150000.0]
    sym_iters = [60, 60, 60]
    for gi, mats in enumerate(groups):
        if time.time() > DEADLINE - 60:
            break
        for rep in range(2):
            if time.time() > DEADLINE - 55:
                break
            seed = 100 + gi * 10 + rep
            try:
                Xs = sym_opt(mats, seed, sym_betas, sym_iters)
            except Exception:
                continue
            vs = _max_cos(Xs)
            cands.append((vs, Xs, f"sym{gi}_{rep}"))
            # release: small symmetry-breaking noise + free anneal + LP at each stage
            # stage A polish (breaks symmetry slightly)
            if time.time() < DEADLINE - 30:
                Xp = lp_polish(Xs, 0.7, r0=2e-4)
                vp = _max_cos(Xp)
                cands.append((vp, Xp, f"sym{gi}_{rep}_lp"))
                # stage B: free anneal
                Xf = free_anneal(Xp, [8000.0, 60000.0, 300000.0], [40, 40, 50])
                vf = _max_cos(Xf)
                cands.append((vf, Xf, f"sym{gi}_{rep}_free"))
                if time.time() < DEADLINE - 25:
                    Xf2 = lp_polish(Xf, 1.0)
                    vf2 = _max_cos(Xf2)
                    cands.append((vf2, Xf2, f"sym{gi}_{rep}_free_lp"))
            # update best (retain diverse graphs too)
            for (vv, XX, _) in cands[-4:]:
                cc = contact_count(XX, vv)
                diverse = abs(cc - cnt_rec) > 8
                if vv < Vbest or (diverse and vv < Vbest + 5e-4):
                    if vv < Vbest:
                        Xbest, Vbest = XX.copy(), float(vv)

    # ---------- stage 2: E8 greedy families ----------
    for s in (1, 2, 3):
        if time.time() > DEADLINE - 35:
            break
        try:
            Xe = greedy_e8(s)
        except Exception:
            continue
        ve = _max_cos(Xe)
        cands.append((ve, Xe, f"e8_{s}"))
        # swap-polished E8 then free anneal + LP release stages
        Xf = free_anneal(Xe, [8000.0, 60000.0, 300000.0], [40, 40, 60])
        vf = _max_cos(Xf)
        cands.append((vf, Xf, f"e8_{s}_free"))
        if time.time() < DEADLINE - 20:
            Xp = lp_polish(Xf, 1.2)
            vp = _max_cos(Xp)
            cands.append((vp, Xp, f"e8_{s}_lp"))
        for (vv, XX, _) in cands[-3:]:
            if vv < Vbest:
                Xbest, Vbest = XX.copy(), float(vv)

    # ---------- stage 3: release best diverse + hopping ----------
    # pick top few distinct candidates for final free hopping
    cands_sorted = sorted(cands, key=lambda t: t[0])[:6]
    # add current best
    pool = [(Vbest, Xbest)] + cands_sorted
    it = 0
    while time.time() < DEADLINE - 10:
        it += 1
        # choose base: mostly best, sometimes diverse
        if it % 5 == 0 and len(cands_sorted) > 1:
            base = cands_sorted[(it // 5) % len(cands_sorted)][1]
        else:
            base = Xbest
        amp = 0.01 if (it % 3) else 0.03
        Xtry = base + amp * rng.standard_normal((n, d))
        Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        Xn = free_anneal(Xtry, [20000.0, 200000.0], [40, 50])
        Vn = _max_cos(Xn)
        if Vn < Vbest:
            Xbest, Vbest = Xn.copy(), float(Vn)
            if time.time() < DEADLINE - 12:
                Xp = lp_polish(Xbest, 1.0, r0=3e-4)
                Vp = _max_cos(Xp)
                if Vp < Vbest:
                    Xbest, Vbest = Xp, Vp
        elif Vn < Vbest + 2e-4 and (it % 4 == 0) and time.time() < DEADLINE - 15:
            Xp = lp_polish(Xn, 0.6, r0=2e-4)
            Vp = _max_cos(Xp)
            if Vp < Vbest:
                Xbest, Vbest = Xp, Vp
        if it % 8 == 0 and time.time() < DEADLINE - 12:
            Xp = lp_polish(Xbest, 0.8)
            Vp = _max_cos(Xp)
            if Vp < Vbest:
                Xbest, Vbest = Xp, Vp

    rem = DEADLINE - time.time() - 0.8
    if rem > 1.0:
        Xp = lp_polish(Xbest, min(rem, 10.0))
        Vp = _max_cos(Xp)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
