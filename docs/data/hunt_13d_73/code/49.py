import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    for z in (xb, xr):
        if z is not None and np.shape(z) == (n, d):
            try:
                zn = z / np.linalg.norm(z, axis=1, keepdims=True)
                if np.all(np.isfinite(zn)):
                    cands.append((_max_cos(zn), zn.copy()))
            except Exception:
                pass
    if not cands:
        r = np.random.RandomState(0).randn(n, d)
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        cands.append((_max_cos(r), r))
    cands.sort(key=lambda t: t[0])
    best_val, Xbest = cands[0][0], cands[0][1].copy()
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)

    iu0, iu1 = np.triu_indices(n, 1)

    def make_fun(alpha):
        def fun(Yflat):
            Y = Yflat.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            c = G[iu0, iu1]
            M = float(c.max())
            a = alpha * (c - M)
            # a <= 0
            e = np.exp(a)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                # fallback to max subgradient
                k = int(np.argmax(c))
                W = np.zeros((n, n))
                i, j = int(iu0[k]), int(iu1[k])
                W[i, j] = 1.0; W[j, i] = 1.0
                loss = M
            else:
                loss = M + np.log(S) / alpha
                p = e / S
                W = np.zeros((n, n))
                W[iu0, iu1] = p
                W[iu1, iu0] = p
            gX = W @ X
            dot = np.sum(gX * X, axis=1, keepdims=True)
            gY = (gX - dot * X) / rn
            return float(loss), gY.ravel().astype(float)
        return fun

    def run_lbfgs(Y0, alpha, maxiter):
        fun = make_fun(alpha)
        try:
            res = minimize(fun, Y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12,
                                    "gtol": 1e-5, "maxcor": 10})
            return res.x
        except Exception:
            return Y0

    def eval_Y(Yflat):
        Y = Yflat.reshape(n, d)
        Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
        return _max_cos(Y), Y

    rng = np.random.RandomState(20261102)

    # quick polish of incumbent at moderate sharpness
    try:
        Y = Xbest.ravel().copy()
        Y = run_lbfgs(Y, 1200.0, 60)
        v, Yn = eval_Y(Y)
        if v < best_val:
            best_val, Xbest = v, Yn.copy()
        if time.time() < HARD_END - 30:
            Y = run_lbfgs(Y, 4000.0, 60)
            v, Yn = eval_Y(Y)
            if v < best_val:
                best_val, Xbest = v, Yn.copy()
    except Exception:
        pass

    # helper: best hole for a removed index
    def best_hole(Xref, keep_mask, nsamp=2500):
        # Xref normalized; keep_mask bool (n,) True to keep
        Xk = Xref[keep_mask]
        H = rng.standard_normal(size=(nsamp, d))
        H /= np.linalg.norm(H, axis=1, keepdims=True)
        # max dot to kept set
        D = H @ Xk.T
        m = D.max(axis=1)
        bi = int(np.argmin(m))
        h = H[bi].copy()
        # local refine h: few projected gradient steps on max-dot
        for _ in range(60):
            dots = Xk @ h
            k2 = int(np.argmax(dots))
            # move away from worst keeper(s): average of top few
            order = np.argpartition(-dots, 4)[:4]
            g = Xk[order].mean(axis=0)
            # tangent component
            g = g - np.dot(g, h) * h
            gn = float(np.linalg.norm(g))
            if gn < 1e-12:
                break
            step = 0.02 / (1.0 + _ * 0.1)
            hn = h - step * g / gn
            hn /= np.linalg.norm(hn)
            # accept if improves
            md_new = float(np.max(Xk @ hn))
            md_old = float(np.max(Xk @ h))
            if md_new < md_old:
                h = hn
            else:
                # try smaller step once
                hn2 = h - 0.3 * step * g / gn
                hn2 /= np.linalg.norm(hn2)
                if float(np.max(Xk @ hn2)) < md_old:
                    h = hn2
                else:
                    break
        return h

    sigmas = [1e-3, 3e-3, 8e-3, 2e-2, 5e-3]
    trial = 0
    # budget trials until ~HARD_END-8
    while time.time() < HARD_END - 8.0 and trial < 40:
        trial += 1
        if time.time() > HARD_END - 10:
            break
        mode = trial % 6
        Xinit = None
        if mode == 5:
            # hole reinsertion on worst point
            try:
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -1.0)
                rowmax = G.max(axis=1)
                iw = int(np.argmax(rowmax))
                # also consider second worst occasionally
                if trial % 12 == 11:
                    order = np.argsort(-rowmax)
                    iw = int(order[1])
                mask = np.ones(n, dtype=bool); mask[iw] = False
                h = best_hole(Xbest, mask, nsamp=2500)
                Xinit = Xbest.copy()
                Xinit[iw] = h
                # small shake of neighbours of iw to open space
                G2 = Xbest @ Xbest.T
                np.fill_diagonal(G2, -2.0)
                neigh = np.argsort(-G2[iw])[:6]
                for jj in neigh:
                    Xinit[jj] = Xinit[jj] + 1e-3 * rng.standard_normal(d)
                Xinit = Xinit / np.linalg.norm(Xinit, axis=1, keepdims=True)
            except Exception:
                Xinit = None
            sig_alphas = [(800.0, 50), (3500.0, 60)]
        elif mode == 4:
            # targeted worst-pair kick + global shake
            try:
                G = Xbest @ Xbest.T
                np.fill_diagonal(G, -2.0)
                k = int(np.argmax(G[iu0, iu1]))
                a, b = int(iu0[k]), int(iu1[k])
                Xinit = Xbest + sigmas[trial % len(sigmas)] * rng.standard_normal((n, d))
                # extra kick to the pair apart along tangent
                pa = Xbest[a]; pb = Xbest[b]
                # direction to separate: -pb tangent at pa, -pa tangent at pb
                da = -(pb - np.dot(pb, pa) * pa)
                db = -(pa - np.dot(pa, pb) * pb)
                Xinit[a] += 3.0 * sigmas[trial % len(sigmas)] * da / (np.linalg.norm(da) + 1e-12)
                Xinit[b] += 3.0 * sigmas[trial % len(sigmas)] * db / (np.linalg.norm(db) + 1e-12)
                Xinit = Xinit / np.linalg.norm(Xinit, axis=1, keepdims=True)
            except Exception:
                Xinit = None
            sig_alphas = [(600.0, 50), (3000.0, 60)]
        else:
            s = sigmas[trial % len(sigmas)]
            # alternate small global shake and cap shake
            Xinit = Xbest + s * rng.standard_normal((n, d))
            Xinit = Xinit / np.linalg.norm(Xinit, axis=1, keepdims=True)
            if trial % 2 == 0:
                sig_alphas = [(500.0, 45), (2500.0, 55)]
            else:
                sig_alphas = [(1200.0, 50), (5000.0, 55)]

        if Xinit is None:
            continue
        Y = Xinit.ravel().copy()
        for (al, mi) in sig_alphas:
            if time.time() > HARD_END - 8.0:
                break
            Y = run_lbfgs(Y, al, mi)
        try:
            v, Yn = eval_Y(Y)
        except Exception:
            continue
        if v < best_val and np.all(np.isfinite(Yn)):
            best_val, Xbest = v, Yn.copy()

    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
