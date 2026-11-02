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
    rng0 = np.random.default_rng(0)
    if not cands:
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

    def make_full_fun(beta):
        b = float(beta)
        def fun(flat):
            Xr = flat.reshape(n, d)
            nr = np.linalg.norm(Xr, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = Xr / nr
            G = Y @ Y.T
            cc = G[iu]
            m = float(cc.max())
            e = np.exp(b * (cc - m))
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
            fun = make_full_fun(float(beta))
            try:
                res = minimize(fun, Xf, jac=True, method="L-BFGS-B",
                               options={"maxiter": int(mi), "ftol": 1e-20, "gtol": 1e-12,
                                        "maxfun": int(mi * 8)})
                Xf = np.asarray(res.x, dtype=float)
            except Exception:
                break
        Xn = Xf.reshape(n, d)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        return Xn

    def block_joint_refine(Binit, Xrest, beta=5000.0, maxiter=200):
        m = Xrest.shape[0]
        B = Binit.shape[0]
        iuB = np.triu_indices(B, 1)
        b = float(beta)
        def fun(flat):
            Xb = flat.reshape(B, d)
            nr = np.linalg.norm(Xb, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = Xb / nr
            S = Y @ Xrest.T
            Gbb = Y @ Y.T
            cbb = Gbb[iuB] if B > 1 else np.array([-2.0])
            m1 = float(S.max()) if S.size else -2.0
            m2 = float(cbb.max()) if cbb.size else -2.0
            mm = m1 if m1 > m2 else m2
            eS = np.exp(b * (S - mm))
            eB = np.exp(b * (cbb - mm))
            s = float(eS.sum() + eB.sum())
            if not np.isfinite(s) or s <= 0:
                return float(mm), np.zeros_like(flat)
            f = mm + np.log(s) / b
            wS = eS / s
            wB = eB / s
            grad = wS @ Xrest
            if B > 1:
                W = np.zeros((B, B))
                W[iuB] = wB
                W = W + W.T
                grad = grad + W @ Y
            dot = (grad * Y).sum(axis=1, keepdims=True)
            GX = (grad - dot * Y) / nr
            return float(f), GX.ravel()
        try:
            r = minimize(fun, Binit.ravel().copy(), jac=True, method="L-BFGS-B",
                         options={"maxiter": int(maxiter), "ftol": 1e-20, "gtol": 1e-12})
            Xb = r.x.reshape(B, d)
            Xb /= np.linalg.norm(Xb, axis=1, keepdims=True)
            return Xb
        except Exception:
            return Binit / np.linalg.norm(Binit, axis=1, keepdims=True)

    def farthest_block_init(Xrest, B, rng, n_cand=30000):
        C = rng.standard_normal((n_cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        S = Xrest @ C.T  # m x cand
        rest_score = S.max(axis=0)  # cand
        placed = []
        placed_scores = []  # for candidates: max dot to placed
        # iterative greedy
        cur_max_to_placed = np.full(n_cand, -2.0)
        order_pick = []
        avail = np.ones(n_cand, dtype=bool)
        for k in range(B):
            total = np.maximum(rest_score, cur_max_to_placed)
            total[~avail] = 2.0
            j = int(np.argmin(total))
            order_pick.append(j)
            avail[j] = False
            placed.append(C[j])
            # update cur_max_to_placed
            dj = C @ C[j]
            dj[~avail] = -2.0
            # don't reuse picked
            cur_max_to_placed = np.maximum(cur_max_to_placed, dj)
            cur_max_to_placed[j] = 2.0
        P = np.stack(placed, axis=0)
        return P

    # initial squeeze
    if DEADLINE - time.time() > 6:
        Xp, Vp = lp_polish(Xbest, seconds=2.0, radius_init=1e-4)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp

    trial = 0
    while time.time() < DEADLINE - 12.0 and trial < 60:
        rem = DEADLINE - 12.0 - time.time()
        if rem < 2.0:
            break
        trial += 1
        rng = np.random.default_rng(1000 + trial * 7919)
        mode = trial % 7

        Xtry = None
        # ---- propose ----
        if mode == 0:
            # small jitter exploit
            sigma = 0.008 if (trial % 2 == 0) else 0.015
            Xtry = Xbest + sigma * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            Xtry = lbfgs_anneal(Xtry, [1500.0, 12000.0, 80000.0], [120, 150, 150])
        elif mode == 1:
            # low-beta exploration from medium jitter (topology change)
            sigma = 0.06 + 0.04 * rng.random()
            Xtry = Xbest + sigma * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            Xtry = lbfgs_anneal(Xtry, [60.0, 400.0, 3000.0, 30000.0], [120, 120, 150, 150])
        elif mode == 2 or mode == 3 or mode == 5:
            # block surgery: eject jammed cluster, joint re-insert
            if mode == 2:
                B = int(4 + rng.integers(0, 3))  # 4-6
            elif mode == 3:
                B = int(7 + rng.integers(0, 5))  # 7-11
            else:
                B = int(3 + rng.integers(0, 3))  # 3-5 random seed region
            G0 = Xbest @ Xbest.T
            np.fill_diagonal(G0, -2.0)
            rowmax0 = G0.max(axis=1)
            thr = float(rowmax0.max()) - 1e-7
            deg = (G0 >= thr).sum(axis=1)
            if mode == 5:
                seed = int(rng.integers(0, n))
            else:
                # prefer overloaded or worst rowmax
                if rng.random() < 0.6:
                    seed = int(np.argmax(deg + 0.1 * (rowmax0 - rowmax0.min())))
                else:
                    seed = int(np.argmax(rowmax0))
            dots = G0[seed].copy()
            dots[seed] = -2.0
            nbrs = np.argsort(-dots)[:B - 1]
            block_idx = np.concatenate([[seed], nbrs])
            mask = np.ones(n, dtype=bool)
            mask[block_idx] = False
            Xrest = Xbest[mask]
            # farthest init + joint refine (two restarts)
            P0 = farthest_block_init(Xrest, B, rng, n_cand=30000)
            P0 = block_joint_refine(P0, Xrest, beta=6000.0, maxiter=250)
            # second restart with noise
            P1 = P0 + 0.05 * rng.standard_normal((B, d))
            P1 /= np.linalg.norm(P1, axis=1, keepdims=True)
            P1 = block_joint_refine(P1, Xrest, beta=6000.0, maxiter=150)
            # evaluate both
            def block_val(P):
                S = P @ Xrest.T
                Gbb = P @ P.T
                np.fill_diagonal(Gbb, -2.0)
                Gr = float((Xrest @ Xrest.T).max()) if m_rest_ok(Xrest) else -2.0
                return max(float(S.max()), float(Gbb.max()), Gr)
            def m_rest_ok(Xr):
                return True
            v0 = block_val(P0); v1 = block_val(P1)
            P = P0 if v0 <= v1 else P1
            Xtry = np.empty_like(Xbest)
            Xtry[mask] = Xrest
            Xtry[block_idx] = P
            # full anneal to reconcile
            if trial % 2 == 0:
                Xtry = lbfgs_anneal(Xtry, [500.0, 6000.0, 40000.0], [100, 130, 150])
            else:
                Xtry = lbfgs_anneal(Xtry, [2000.0, 20000.0], [150, 180])
        elif mode == 4:
            # heavy resample: reinit worst K points randomly then joint refine
            G0 = Xbest @ Xbest.T
            np.fill_diagonal(G0, -2.0)
            rowmax0 = G0.max(axis=1)
            order0 = np.argsort(-rowmax0)
            K = 14 + int(rng.integers(0, 8))
            worst = order0[:K]
            mask = np.ones(n, dtype=bool)
            mask[worst] = False
            Xrest = Xbest[mask]
            B = K
            P0 = farthest_block_init(Xrest, B, rng, n_cand=40000)
            P0 = block_joint_refine(P0, Xrest, beta=4000.0, maxiter=200)
            Xtry = np.empty_like(Xbest)
            Xtry[mask] = Xrest
            Xtry[worst] = P0
            Xtry = lbfgs_anneal(Xtry, [300.0, 4000.0, 30000.0], [100, 130, 150])
        else:  # mode 6
            # pure random + low-beta funnel (new basin attempt)
            if rng.random() < 0.5:
                Xtry = rng.standard_normal((n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            else:
                Xtry = Xbest + 0.35 * rng.standard_normal((n, d))
                Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            Xtry = lbfgs_anneal(Xtry, [80.0, 800.0, 8000.0, 60000.0], [120, 130, 150, 150])

        if Xtry is None:
            continue
        v = _max_cos(Xtry)
        # polish if promising or periodic
        promising = (v < Vbest + 0.004) or (trial % 5 == 0 and v < Vbest + 0.012)
        if promising:
            secs = min(5.0, DEADLINE - 12.0 - time.time() - 0.5)
            if secs > 0.7:
                Xp, Vp = lp_polish(Xtry, seconds=secs, radius_init=3e-4)
                Xtry, v = Xp, Vp
        if v < Vbest - 1e-12:
            Vbest = float(v)
            Xbest = Xtry.copy()

    rem = DEADLINE - time.time() - 1.0
    if rem > 1.5:
        Xbest, Vbest = lp_polish(Xbest, seconds=min(rem, 9.0), radius_init=2e-4)
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
