import time
import numpy as np
from scipy.optimize import minimize, differential_evolution, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, best, t_deadline, max_iters=10, r_init=8e-4, cap=1100):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    radius = r_init
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while it < max_iters and time.time() < t_deadline and radius > 1e-14:
        it += 1
        Cfull = X @ X.T
        c = Cfull[iu0, iu1]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu0[sel], iu1[sel]
        p = len(I)
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        ct = np.full((p, 1), n * d)
        cols = np.concatenate([cI, cJ, ct], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        vv = _max_cos(y)
        if vv < best - 1e-13:
            X, best = y, vv
            radius = min(radius * 1.5, 0.01)
        else:
            radius *= 0.5
    return X, best

def _adam_lse(X0, steps, s0, s1, lr0, lr1, t_lim=None):
    n, d = X0.shape
    X = X0.copy()
    m = np.zeros((n, d)); v = np.zeros((n, d))
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = b1, b2
    iu0, iu1 = np.triu_indices(n, 1)
    l0, l1 = np.log(lr0), np.log(lr1)
    ls0, ls1 = np.log(s0), np.log(s1)
    WX = np.empty((n, d))
    for k in range(steps):
        if t_lim is not None and (k & 63) == 0 and time.time() > t_lim:
            break
        f = k / (steps - 1) if steps > 1 else 1.0
        s = float(np.exp(ls0 + (ls1 - ls0) * f))
        lr = float(np.exp(l0 + (l1 - l0) * f))
        G = X @ X.T
        c = G[iu0, iu1]
        mx = float(c.max())
        e = np.exp(s * (c - mx))
        se = float(e.sum())
        if not np.isfinite(se) or se <= 0:
            continue
        e /= se
        WX.fill(0.0)
        np.add.at(WX, iu0, e[:, None] * X[iu1])
        np.add.at(WX, iu1, e[:, None] * X[iu0])
        sp = np.sum(WX * X, axis=1, keepdims=True)
        Gt = WX - sp * X
        m *= b1; m += (1 - b1) * Gt
        v *= b2; v += (1 - b2) * (Gt * Gt)
        mh = m / (1 - b1p); vh = v / (1 - b2p)
        b1p *= b1; b2p *= b2
        step = mh / (np.sqrt(vh) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        over = sn > 3.0
        if np.any(over):
            step = np.where(over, step * (3.0 / np.maximum(sn, 1e-12)), step)
        X = X - lr * step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X /= nr
    return X

def _hole_dir(X, rng, n_try=2000):
    n, d = X.shape
    C = rng.standard_normal(size=(n_try, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    S = C @ X.T
    mm = S.max(axis=1)
    return C[int(np.argmin(mm))]

def solve():
    t0 = time.time()
    T_END = t0 + 165.0
    SEARCH_END = t0 + 138.0
    LP_END = t0 + 163.0
    rng = np.random.default_rng(0)
    n, d = 73, 13
    Xrec = None; best_file = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn)
            vv = np.asarray(vv, dtype=np.float64)
            if vv.shape == (n, d):
                vv = vv / np.linalg.norm(vv, axis=1, keepdims=True)
                if np.all(np.isfinite(vv)):
                    cc = _max_cos(vv)
                    if cc < best_file:
                        best_file = cc; Xrec = vv.copy()
        except Exception:
            pass
    if Xrec is None:
        Xrec = rng.standard_normal(size=(n, d))
        Xrec /= np.linalg.norm(Xrec, axis=1, keepdims=True)
        best_file = _max_cos(Xrec)
    Xbest = Xrec.copy(); best = float(best_file)

    # ---- Z73 cyclic family (cheap) ----
    try:
        N, NM, NLAG = 73, 6, 36
        L = np.arange(1, NLAG + 1)[:, None]
        M = np.arange(1, NM + 1)[None, :]
        CTAB = np.cos(2.0 * np.pi * L * M / 73.0)
        def lagmax(w):
            w = np.asarray(w, dtype=float)
            s = float(w.sum())
            if not np.isfinite(s) or s <= 1e-15:
                return 1.0
            wn = w / s
            lags = wn[6] + CTAB @ wn[:6]
            return float(lags.max())
        res = differential_evolution(lagmax, [(0, 1)] * 7, maxiter=120, popsize=4,
                                     seed=0, polish=False, workers=1, updating='deferred', tol=1e-8)
        x_de = np.asarray(res.x, dtype=float)
        try:
            r2 = minimize(lagmax, x_de, method="Nelder-Mead",
                          options={"maxiter": 600, "xatol": 1e-8, "fatol": 1e-12})
            if float(r2.fun) < float(res.fun):
                x_de = np.asarray(r2.x, dtype=float)
        except Exception:
            pass
        wn = x_de / float(x_de.sum())
        r_m = np.sqrt(np.maximum(wn[:6], 0)); h = float(np.sqrt(max(wn[6], 0)))
        if h < 1e-6:
            h = 1e-6
            tot = float((r_m ** 2).sum() + h * h); r_m /= np.sqrt(tot); h /= np.sqrt(tot)
        phi = np.random.default_rng(1).uniform(0, 2 * np.pi, size=NM)
        Xcyc = np.zeros((N, d))
        for k in range(N):
            for mm in range(NM):
                ang = 2 * np.pi * (mm + 1) * k / 73.0 + float(phi[mm])
                Xcyc[k, 2 * mm] = r_m[mm] * np.cos(ang)
                Xcyc[k, 2 * mm + 1] = r_m[mm] * np.sin(ang)
            Xcyc[k, 12] = h
        Xcyc /= np.linalg.norm(Xcyc, axis=1, keepdims=True)
        for eps in (1e-4, 1e-3, 1e-2):
            if time.time() > t0 + 30:
                break
            Xp = Xcyc + np.random.default_rng(100 + int(-np.log10(eps))).standard_normal(size=(n, d)) * eps
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            Xp = _adam_lse(Xp, 200, 100, 800, 0.02, 0.003, t_lim=t0 + 32)
            vv = _max_cos(Xp)
            if vv < best:
                Xbest = Xp.copy(); best = vv
        # also try cyclic raw polished briefly
        if time.time() < t0 + 32:
            Xc2 = _adam_lse(Xcyc, 200, 100, 800, 0.02, 0.003, t_lim=t0 + 34)
            vv = _max_cos(Xc2)
            if vv < best:
                Xbest = Xc2.copy(); best = vv
    except Exception:
        pass

    # ---- topology-changing basin hopping on incumbent ----
    iu0, iu1 = np.triu_indices(n, 1)
    trial = 0
    rng2 = np.random.default_rng(7)
    # per-point worst for targeting
    while time.time() < SEARCH_END:
        trial += 1
        G = Xbest @ Xbest.T
        np.fill_diagonal(G, -2.0)
        flat = int(np.argmax(G))
        wi, wj = flat // n, flat % n
        pw = G.max(axis=1)
        order = np.argsort(-pw)
        mode = trial % 5
        Xp = Xbest.copy()
        if mode == 0:
            # kick one endpoint of worst pair
            tgt = wi if rng2.random() < 0.5 else wj
            sig = 0.03 + 0.05 * rng2.random()
            Xp[tgt] += rng2.standard_normal(d) * sig
            Xp[tgt] /= np.linalg.norm(Xp[tgt])
        elif mode == 1:
            # hole reinsert worst point
            tgt = int(order[0])
            hd = _hole_dir(Xbest, rng2, n_try=1500)
            # blend: replace with hole + small noise
            Xp[tgt] = hd + rng2.standard_normal(d) * 0.005
            Xp[tgt] /= np.linalg.norm(Xp[tgt])
        elif mode == 2:
            # kick top-3 worst points moderately
            sig = 0.02 + 0.03 * rng2.random()
            for tgt in order[:3]:
                Xp[int(tgt)] += rng2.standard_normal(d) * sig
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 3:
            # global small shake
            sig = 0.008 + 0.012 * rng2.random()
            Xp = Xbest + rng2.standard_normal(size=(n, d)) * sig
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        else:
            # cap shake: pick random dir, push cap points
            vdir = rng2.standard_normal(d); vdir /= np.linalg.norm(vdir)
            dots = Xbest @ vdir
            thr = np.quantile(dots, 0.85)
            sel = dots >= thr
            sig = 0.04
            Xp[sel] += rng2.standard_normal(size=(int(sel.sum()), d)) * sig
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        # short anneal screening
        s0 = 150.0; s1 = 1200.0
        steps = 220 if mode != 4 else 180
        Xq = _adam_lse(Xp, steps, s0, s1, 0.025, 0.003, t_lim=SEARCH_END)
        vq = _max_cos(Xq)
        if vq < best - 1e-12:
            Xbest = Xq; best = vq
            # immediate squeeze on improvement if time allows
            if time.time() < LP_END - 12:
                try:
                    Xs, vs = _lp_polish(Xbest, best, min(LP_END, time.time() + 4.0), max_iters=6, r_init=6e-4)
                    if vs < best:
                        Xbest, best = Xs, vs
                except Exception:
                    pass
        # occasional escape: accept slightly worse as restart base? no, keep elitist
        if trial > 400:
            break

    # ---- final equalize + squeeze ----
    try:
        if time.time() < LP_END - 20:
            Xbest = _adam_lse(Xbest, 500, 120, 2000, 0.015, 0.002, t_lim=LP_END - 14)
            best = min(best, _max_cos(Xbest))
        if time.time() < LP_END:
            Xbest, best = _lp_polish(Xbest, best, LP_END, max_iters=14, r_init=8e-4)
    except Exception:
        pass
    # safety: retain incumbent if somehow worse than file
    if best > best_file + 1e-12:
        # check file again
        try:
            Xf = Xrec.copy(); Xf /= np.linalg.norm(Xf, axis=1, keepdims=True)
            if _max_cos(Xf) <= best:
                return Xf
        except Exception:
            pass
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
