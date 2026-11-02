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
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands = [X0]
    vals = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals))].copy()
    Vbest = float(min(vals))

    def anneal(X, betas, maxiters):
        flat = X.ravel().copy()
        for beta, mi in zip(betas, maxiters):
            if time.time() > DEADLINE - 9.0:
                break
            b = float(beta)
            def fun(fl):
                Xr = fl.reshape(n, d)
                nr = np.linalg.norm(Xr, axis=1, keepdims=True)
                nr = np.maximum(nr, 1e-12)
                Y = Xr / nr
                G = Y @ Y.T
                c = G[iu]
                m = float(c.max())
                e = np.exp(b * (c - m))
                s = float(e.sum())
                if not np.isfinite(s) or s <= 0:
                    return float(m), np.zeros_like(fl)
                f = m + np.log(s) / b
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
                               options={"maxiter": int(mi), "ftol": 1e-20,
                                        "gtol": 1e-12, "maxfun": int(mi*10)})
                flat = np.asarray(res.x, dtype=float)
            except Exception:
                break
        Xn = flat.reshape(n, d)
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        return Xn

    def lp_polish(x, seconds, r0=5e-4):
        start = time.time()
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = r0
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.0:
            if radius < 1e-14 or it > 300:
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

    def find_holes(X, k, nsamp=12000, refine=60):
        R = rng.standard_normal((nsamp, d))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        S = R @ X.T
        m = S.max(axis=1)
        # pick k diverse holes greedily
        order = np.argpartition(m, min(nsamp-1, k*20))[:k*20]
        cand = R[order]
        cm = m[order]
        o = np.argsort(cm)
        cand = cand[o]; cm = cm[o]
        holes = []
        for h in cand:
            if len(holes) >= k:
                break
            ok = True
            for hh in holes:
                if float(h @ hh) > 0.9:
                    ok = False; break
            if ok:
                holes.append(h.copy())
        holes = np.array(holes)
        # refine each hole: move away from nearest point
        for i in range(len(holes)):
            h = holes[i]
            step = 0.05
            for _ in range(refine):
                dots = X @ h
                j = int(np.argmax(dots))
                g = X[j] - dots[j]*h
                gn = np.linalg.norm(g)
                if gn < 1e-12:
                    break
                h = h - step*g/gn
                h /= np.linalg.norm(h)
                step *= 0.97
            holes[i] = h
        return holes

    # schedules
    B1 = [3000.0, 20000.0, 120000.0]
    M1 = [50, 50, 50]
    B2 = [8000.0, 60000.0, 400000.0]
    M2 = [40, 40, 60]

    it = 0
    # quick initial squeeze already near optimum; go straight to hopping
    while time.time() < DEADLINE - 10.0:
        it += 1
        if it == 1:
            Xtry = Xbest.copy()
            Xn = anneal(Xtry, B2, M2)
            Vn = _max_cos(Xn)
            if Vn < Vbest:
                Xbest, Vbest = Xn, Vn
                tl = min(3.0, max(0.5, DEADLINE-10.0-time.time()))
                if tl > 0.5:
                    Xp = lp_polish(Xbest, tl)
                    Vp = _max_cos(Xp)
                    if Vp < Vbest:
                        Xbest, Vbest = Xp, Vp
            continue
        G = Xbest @ Xbest.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        order = np.argsort(-rowmax)
        mode = it % 6
        if mode == 1:
            Xtry = Xbest + 0.008*rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            Xn = anneal(Xtry, B2, M2)
        elif mode == 2:
            Xtry = Xbest + 0.028*rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            Xn = anneal(Xtry, B1, M1)
        elif mode == 3:
            k = int(rng.integers(6, 13))
            worst = order[:k]
            Xtry = Xbest.copy()
            Xtry[worst] += 0.14*rng.standard_normal((k, d))
            Xtry += 0.004*rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            Xn = anneal(Xtry, B1, M1)
        elif mode == 4:
            # void re-insertion: replace worst few by holes
            k = int(rng.integers(2, 6))
            worst = order[:k]
            try:
                holes = find_holes(Xbest, k)
            except Exception:
                holes = None
            Xtry = Xbest.copy()
            if holes is not None and len(holes)==k:
                alpha = 0.95
                Xtry[worst] = alpha*holes + 0.05*rng.standard_normal((k, d))
            else:
                R = rng.standard_normal((k, d)); R/=np.linalg.norm(R,axis=1,keepdims=True)
                Xtry[worst] = 0.5*Xtry[worst]+0.5*R
            Xtry += 0.003*rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            Xn = anneal(Xtry, B1, [60, 60, 60])
        elif mode == 5:
            # jammed-cluster surgery: coherent shift of worst + neighbours
            w = int(order[0])
            neigh = np.argsort(-G[w])[:int(rng.integers(8, 14))]
            Xtry = Xbest.copy()
            shift = rng.standard_normal(d); shift/=np.linalg.norm(shift)+1e-12
            amp = float(rng.uniform(0.03, 0.09))
            Xtry[neigh] += amp*shift + 0.01*rng.standard_normal((len(neigh), d))
            # small global jitter
            Xtry += 0.002*rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            Xn = anneal(Xtry, B1, M1)
        else:
            # weighted noise by badness
            wgt = rowmax - rowmax.min()
            wgt = wgt/(wgt.max()+1e-12)
            wgt = 0.4 + 1.6*wgt
            Xtry = Xbest + 0.016*wgt[:,None]*rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            Xn = anneal(Xtry, B2, M2)
        if not np.all(np.isfinite(Xn)):
            continue
        Vn = _max_cos(Xn)
        if Vn < Vbest:
            Xbest, Vbest = Xn, Vn
            tl = min(3.5, max(0.5, DEADLINE-10.0-time.time()))
            if tl > 0.5:
                Xp = lp_polish(Xbest, tl)
                Vp = _max_cos(Xp)
                if Vp < Vbest:
                    Xbest, Vbest = Xp, Vp
        elif Vn < Vbest + 2e-4 and (it % 4 == 0) and time.time() < DEADLINE - 20:
            # explore nearby basin: squeeze the challenger, keep if it becomes best
            Xp = lp_polish(Xn, 0.8, r0=2e-4)
            Vp = _max_cos(Xp)
            if Vp < Vbest:
                Xbest, Vbest = Xp, Vp
        if it % 10 == 0 and time.time() < DEADLINE - 14:
            Xp = lp_polish(Xbest, 1.2)
            Vp = _max_cos(Xp)
            if Vp < Vbest:
                Xbest, Vbest = Xp, Vp

    rem = DEADLINE - time.time() - 0.8
    if rem > 1.0:
        Xp = lp_polish(Xbest, min(rem, 22.0))
        Vp = _max_cos(Xp)
        if Vp < Vbest:
            Xbest, Vbest = Xp, Vp
        # final high-beta snap
        if time.time() < DEADLINE - 2.0:
            Xn = anneal(Xbest, [100000.0, 600000.0], [40, 60])
            Vn = _max_cos(Xn)
            if Vn < Vbest:
                Xbest, Vbest = Xn, Vn
                r2 = DEADLINE - time.time() - 0.5
                if r2 > 0.5:
                    Xp2 = lp_polish(Xbest, min(r2, 8.0), r0=2e-4)
                    Vp2 = _max_cos(Xp2)
                    if Vp2 < Vbest:
                        Xbest, Vbest = Xp2, Vp2
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
