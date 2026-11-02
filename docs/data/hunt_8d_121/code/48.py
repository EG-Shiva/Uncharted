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
    BUDGET = 168.0
    DEADLINE = t0 + BUDGET
    T_DISC_END = t0 + 38.0
    T_CONT_END = t0 + 128.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 121, 8
    iu = np.triu_indices(n, 1)
    Iall, Jall = iu[0], iu[1]

    # incumbent
    Xinc = None; Vinc = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < Vinc:
                        Vinc = v; Xinc = a.copy()
        except Exception:
            pass
    if Xinc is None:
        Xinc = rng.standard_normal((n, d))
        Xinc /= np.linalg.norm(Xinc, axis=1, keepdims=True)
        Vinc = _max_cos(Xinc)
    Xbest = Xinc.copy(); Vbest = float(Vinc)

    # ---------- pools ----------
    def e8_roots():
        pts = []
        for i in range(8):
            for j in range(i+1, 8):
                for s1 in (-1.0, 1.0):
                    for s2 in (-1.0, 1.0):
                        v = np.zeros(8)
                        v[i] = s1; v[j] = s2
                        pts.append(v / np.sqrt(2.0))
        for mask in range(256):
            bits = [(mask >> k) & 1 for k in range(8)]
            if sum(bits) % 2 == 1:
                continue
            v = np.ones(8) * 0.5
            for k in range(8):
                if bits[k]:
                    v[k] = -0.5
            pts.append(v)
        P = np.stack(pts, axis=0)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        return P  # 240

    def hypercube():
        N = 256
        P = np.empty((N, 8))
        for m in range(N):
            v = np.ones(8)
            for k in range(8):
                if (m >> k) & 1:
                    v[k] = -1.0
            P[m] = v / np.sqrt(8.0)
        return P

    def cross():
        P = np.zeros((16, 8))
        for i in range(8):
            P[2*i, i] = 1.0; P[2*i+1, i] = -1.0
        return P

    def rand_Q(rs):
        A = rs.standard_normal((8, 8))
        Q, R = np.linalg.qr(A)
        s = np.sign(np.diag(R))
        s[s == 0] = 1.0
        Q = Q * s[None, :]
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        return Q

    try:
        E8 = e8_roots()
        HC = hypercube()
        CR = cross()
        pools = [E8, HC, CR]
        # random rotations/unions of pools
        for rr in range(3):
            Q = rand_Q(np.random.default_rng(100 + rr))
            pools.append(E8 @ Q.T)
        # one rotated hypercube
        Qh = rand_Q(np.random.default_rng(999))
        pools.append(HC @ Qh.T)
        P = np.concatenate(pools, axis=0)
        # dedup roughly (round)
        # keep as is; size ~ 240*4+256*2+16 = 1488
        # normalize
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        mP = P.shape[0]
        Gpool = P @ P.T
    except Exception:
        P = None; Gpool = None; mP = 0

    # ---------- discrete selection ----------
    disc_cands = []  # list of (val, sel array)
    if P is not None and mP > n:
        # greedy function
        def greedy(seed):
            sel = [int(seed % mP)]
            cur = Gpool[sel[0], :].copy()
            cur[sel[0]] = 2.0
            selected_mask = np.zeros(mP, dtype=bool)
            selected_mask[sel[0]] = True
            for _ in range(n - 1):
                j = int(np.argmin(np.where(selected_mask, 2.0, cur)))
                sel.append(j)
                selected_mask[j] = True
                gj = Gpool[j, :]
                # cur = min(cur, gj)
                np.minimum(cur, gj, out=cur)
                cur[j] = 2.0
                # keep selected masked
                # (cur for selected already 2.0)
            return np.array(sel, dtype=int)

        def max_of_sel(sel):
            Gs = Gpool[np.ix_(sel, sel)]
            np.fill_diagonal(Gs, -2.0)
            return float(Gs.max())

        seeds = [0, 37, 101, 311, 557, 809, 1021, 1233]
        for si, s in enumerate(seeds):
            if time.time() > T_DISC_END:
                break
            try:
                sel = greedy(s)
                cur = max_of_sel(sel)
                # simulated-annealing swaps, incremental eval via full submatrix
                # budget per seed
                steps = 3500
                T0, T1 = 0.004, 0.00015
                sel_list = sel.tolist()
                sel_arr = sel.copy()
                # for fast rest-max we just compute full each time
                for step in range(steps):
                    if time.time() > T_DISC_END:
                        break
                    frac = step / max(1, steps - 1)
                    T = T0 * (1 - frac) + T1 * frac
                    i_pos = int(rng.integers(0, n))
                    u = int(rng.integers(0, mP))
                    # skip if u already selected
                    # quick check via set? use boolean mask
                    # maintain set
                    # for speed, check if u in sel via linear scan on small array
                    if np.any(sel_arr == u):
                        continue
                    old = int(sel_arr[i_pos])
                    sel_arr[i_pos] = u
                    # compute new max: need dots of u to others + rest max
                    # compute dmax
                    dmax = float(Gpool[u, sel_arr[np.arange(n) != i_pos]].max())
                    # rest max: max among pairs not involving i_pos
                    # compute via submatrix of remaining n-1 points
                    rem = sel_arr[np.arange(n) != i_pos]
                    Gr = Gpool[np.ix_(rem, rem)]
                    np.fill_diagonal(Gr, -2.0)
                    rmax = float(Gr.max())
                    newv = dmax if dmax > rmax else rmax
                    if newv < cur or rng.random() < np.exp((cur - newv) / max(T, 1e-12)):
                        cur = newv
                        sel_list[i_pos] = u
                        # accept (sel_arr already updated)
                        if newv < cur:
                            pass
                    else:
                        sel_arr[i_pos] = old
                    # occasional greedy improvement check: keep best
                # final greedy hill-climbing passes (only improving)
                for _ in range(800):
                    if time.time() > T_DISC_END:
                        break
                    i_pos = int(rng.integers(0, n))
                    u = int(rng.integers(0, mP))
                    if np.any(sel_arr == u):
                        continue
                    old = int(sel_arr[i_pos])
                    sel_arr[i_pos] = u
                    dmax = float(Gpool[u, sel_arr[np.arange(n) != i_pos]].max())
                    rem = sel_arr[np.arange(n) != i_pos]
                    Gr = Gpool[np.ix_(rem, rem)]
                    np.fill_diagonal(Gr, -2.0)
                    rmax = float(Gr.max())
                    newv = dmax if dmax > rmax else rmax
                    if newv < cur - 1e-12:
                        cur = newv
                    else:
                        sel_arr[i_pos] = old
                disc_cands.append((float(cur), sel_arr.copy()))
            except Exception:
                continue
        # sort
        try:
            disc_cands.sort(key=lambda t: t[0])
        except Exception:
            pass

    # ---------- continuous helpers ----------
    def lbfgs_run(Xinit, beta, maxiter):
        if time.time() > DEADLINE - 8.0:
            Xn = Xinit / np.linalg.norm(Xinit, axis=1, keepdims=True)
            return Xn
        b = float(beta)
        Xf = (Xinit / np.linalg.norm(Xinit, axis=1, keepdims=True)).ravel().copy()
        def fun(flat):
            Xr = flat.reshape(n, d)
            nr = np.linalg.norm(Xr, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = Xr / nr
            Gm = Y @ Y.T
            c = Gm[iu]
            mm = float(c.max())
            e = np.exp(b * (c - mm))
            s = float(e.sum())
            if not np.isfinite(s) or s <= 0:
                return float(mm), np.zeros_like(flat)
            f = mm + np.log(s) / b
            w = e / s
            W = np.zeros((n, n))
            W[Iall, Jall] = w
            W[Jall, Iall] = w
            GY = W @ Y
            dot = (GY * Y).sum(axis=1, keepdims=True)
            GX = (GY - dot * Y) / nr
            return float(f), GX.ravel()
        try:
            res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20, "gtol": 1e-12, "maxfun": int(maxiter * 5)})
            Xf = np.asarray(res.x, dtype=float)
        except Exception:
            pass
        Xn = Xf.reshape(n, d)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
        nn = np.linalg.norm(Xn, axis=1, keepdims=True); nn[nn == 0] = 1.0
        Xn /= nn
        return Xn

    def lp_polish(x, seconds, radius_init=4e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = radius_init
        while time.time() - start < seconds and time.time() < DEADLINE - 1.5:
            if radius < 1e-14:
                break
            G = x @ x.T
            cf = G[iu]
            m = float(cf.max())
            thr = max(20.0 * radius, 2e-9)
            selm = cf > m - thr
            cnt = int(selm.sum())
            if cnt == 0:
                radius *= 0.5
                continue
            if cnt > 2200:
                kth = np.sort(cf)[-2200]
                selm = cf >= kth
                cnt = int(selm.sum())
                if cnt > 2600:
                    idxp = np.argpartition(-cf, 2600)[:2600]
                    mk = np.zeros_like(selm); mk[idxp] = True
                    selm = mk; cnt = 2600
            c = cf[selm]
            I = iu[0][selm]; J = iu[1][selm]
            p_ = cnt
            nv = n * d + 1
            try:
                rows = np.repeat(np.arange(p_), 2 * d + 1)
                cols = np.concatenate([I[:, None] * d + np.arange(d),
                                       J[:, None] * d + np.arange(d),
                                       np.full((p_, 1), n * d)], 1).ravel()
                valsA = np.concatenate([x[J], x[I], -np.ones((p_, 1))], 1).ravel()
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p_, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
            y = x + res.x[:-1].reshape(n, d)
            yn = np.linalg.norm(y, axis=1, keepdims=True); yn[yn == 0] = 1.0
            y = y / yn
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.6, 0.02)
            else:
                radius *= 0.5
        return x, best

    # ---------- continuous release (<= ~90s) ----------
    cont_best_X = None; cont_best_V = 1e9
    if disc_cands and time.time() < T_CONT_END:
        # take best few discrete selections
        K = min(3, len(disc_cands))
        top = disc_cands[:K]
        sigmas = [1e-3, 1e-2, 3e-2]
        for (dv, selarr) in top:
            if time.time() > T_CONT_END - 5.0:
                break
            try:
                Xd = P[selarr].copy()
                Xd /= np.linalg.norm(Xd, axis=1, keepdims=True)
            except Exception:
                continue
            # small-beta trials with different symmetry-breaking sizes
            best_local_X = None; best_local_V = 1e9
            for sg in sigmas:
                if time.time() > T_CONT_END - 5.0:
                    break
                rs = np.random.default_rng(int(1000 + int(sg * 1e5) + int(dv * 1e6) % 10000))
                Xp = Xd + sg * rs.standard_normal((n, d))
                Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
                Xo = lbfgs_run(Xp, 180.0, 50)
                vo = _max_cos(Xo)
                if vo < best_local_V:
                    best_local_V = vo; best_local_X = Xo.copy()
                if time.time() > T_CONT_END - 5.0:
                    break
            if best_local_X is None:
                continue
            # continue ordinary annealing with larger betas
            try:
                Xc = lbfgs_run(best_local_X, 500.0, 70)
                if time.time() > T_CONT_END - 3.0:
                    vc = _max_cos(Xc)
                    if vc < best_local_V:
                        best_local_V, best_local_X = vc, Xc
                else:
                    Xc2 = lbfgs_run(Xc, 1200.0, 70)
                    vc2 = _max_cos(Xc2)
                    if vc2 < best_local_V:
                        best_local_V, best_local_X = vc2, Xc2
                    else:
                        vc = _max_cos(Xc)
                        if vc < best_local_V:
                            best_local_V, best_local_X = vc, Xc
            except Exception:
                pass
            if best_local_V < cont_best_V:
                cont_best_V = best_local_V; cont_best_X = best_local_X.copy()
            if cont_best_V < Vbest:
                Vbest = cont_best_V; Xbest = cont_best_X.copy()

    # if continuous failed, try quick anneal of incumbent briefly (basin hop via perturbation)
    if (cont_best_X is None or cont_best_V >= Vbest - 1e-9) and time.time() < T_CONT_END - 10.0:
        try:
            rs = np.random.default_rng(7)
            Xp = Xinc + 0.02 * rs.standard_normal((n, d))
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            Xo = lbfgs_run(Xp, 500.0, 60)
            vo = _max_cos(Xo)
            if vo < Vbest:
                Vbest, Xbest = vo, Xo.copy()
                cont_best_X, cont_best_V = Xo.copy(), vo
        except Exception:
            pass

    # ---------- LP polish remainder ----------
    rem = DEADLINE - time.time() - 1.0
    if rem > 3.0:
        # polish best few: Xbest and cont best and incumbent
        cands_pol = [(Xbest, Vbest)]
        if cont_best_X is not None:
            cands_pol.append((cont_best_X, cont_best_V))
        cands_pol.append((Xinc, Vinc))
        per = rem / len(cands_pol)
        per = min(per, 25.0)
        for (Xc0, vc0) in cands_pol:
            if time.time() > DEADLINE - 2.0:
                break
            try:
                Xp, vp = lp_polish(Xc0, seconds=per)
                if vp < Vbest:
                    Vbest, Xbest = vp, Xp.copy()
            except Exception:
                continue

    # final safety
    Xbest = np.asarray(Xbest, dtype=float)
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    Xbest = np.nan_to_num(Xbest, nan=0.0, posinf=0.0, neginf=0.0)
    nn = np.linalg.norm(Xbest, axis=1, keepdims=True); nn[nn == 0] = 1.0
    Xbest /= nn
    # tiny shrink guard not needed; return
    # ensure incumbent if something went wrong
    try:
        vf = _max_cos(Xbest)
        if not np.isfinite(vf) or vf > Vbest + 1e-12:
            Xbest = Xinc.copy()
    except Exception:
        Xbest = Xinc.copy()
    return Xbest
