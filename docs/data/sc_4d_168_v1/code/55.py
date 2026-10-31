import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.linalg import expm


def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def _nm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def _skew(v):
    A = np.zeros((4, 4), dtype=float)
    A[0, 1] = v[0]; A[0, 2] = v[1]; A[0, 3] = v[2]
    A[1, 2] = v[3]; A[1, 3] = v[4]; A[2, 3] = v[5]
    return A - A.T


def _Rvec(v):
    return expm(_skew(np.asarray(v, dtype=float)))


def _lse(D, beta):
    cm = float(np.max(D))
    s = np.sum(np.exp(np.clip(beta * (D - cm), -30.0, 0.0)))
    return cm + np.log(s + 1e-300) / beta, cm


def _sph_kmeans(X, K, rng, iters=25):
    n = X.shape[0]
    # farthest point seeding (angular)
    first = int(rng.integers(0, n))
    centers = [X[first].copy()]
    for _ in range(1, K):
        C = np.stack(centers, axis=0)
        d = C @ X.T
        md = d.max(axis=0)
        nxt = int(np.argmin(md))
        centers.append(X[nxt].copy())
    C = np.stack(centers, axis=0)
    lab = np.zeros(n, dtype=int)
    for _ in range(iters):
        D = X @ C.T
        lab = np.argmax(D, axis=1)
        Cn = np.zeros_like(C)
        for k in range(K):
            m = lab == k
            if np.any(m):
                v = X[m].sum(axis=0)
                nv = np.linalg.norm(v)
                if nv < 1e-12:
                    Cn[k] = C[k]
                else:
                    Cn[k] = v / nv
            else:
                Cn[k] = _nm(rng.normal(size=(1, 4)))[0]
        if np.allclose(Cn, C):
            C = Cn
            break
        C = Cn
    # fix empty clusters
    for k in range(K):
        if not np.any(lab == k):
            lab[int(rng.integers(0, n))] = k
    return lab


def _degrees(X, topk=600):
    n = X.shape[0]
    G = X @ X.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    m = float(np.max(c))
    # adaptive: threshold to get ~topk
    if len(c) > topk:
        thr = np.sort(c)[-topk]
        thr = min(thr, m - 1e-12)
    else:
        thr = m - 1e-9
    sel = c > thr
    I = iu[0][sel]; J = iu[1][sel]
    deg = np.zeros(n, dtype=int)
    for i in I:
        deg[i] += 1
    for j in J:
        deg[j] += 1
    return deg, m, int(np.sum(sel)), (I, J)


def _chash(X):
    deg, m, nc, _ = _degrees(X)
    return (nc, tuple(sorted(deg.tolist())))


def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    SEARCH = t0 + 148.0
    rng = np.random.default_rng(12345)

    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.load(f).astype(float)
            if c.shape == (168, 4):
                c = _nm(c)
                if np.all(np.isfinite(c)):
                    cands.append(c)
        except Exception:
            pass
    if not cands:
        xb = _nm(rng.normal(size=(168, 4)))
        bv = _mc(xb)
    else:
        vs = [_mc(c) for c in cands]
        bi = int(np.argmin(vs))
        xb = cands[bi].copy()
        bv = float(vs[bi])

    x_best = xb.copy(); best = bv
    x_cur = xb.copy(); cur = bv
    h_best = _chash(x_best); h_cur = h_best

    def adam_full(X, steps, b0=30.0, b1=420.0, lr0=0.02, lr1=0.0018):
        X = X.copy()
        m = np.zeros_like(X); v = np.zeros_like(X)
        b1a, b2a, eps = 0.9, 0.999, 1e-8
        lb0, lb1 = np.log(b0), np.log(b1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        for t in range(1, steps + 1):
            if time.time() > SEARCH:
                break
            f = t / steps
            beta = float(np.exp(lb0 + (lb1 - lb0) * f))
            lr = float(np.exp(ll0 + (ll1 - ll0) * f))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gm = float(np.max(G))
            W = np.exp(np.clip(beta * (G - gm), -20, 0))
            np.fill_diagonal(W, 0.0)
            Gr = (W @ X) * beta
            Gt = Gr - np.sum(Gr * X, axis=1, keepdims=True) * X
            m = b1a * m + (1 - b1a) * Gt
            v = b2a * v + (1 - b2a) * (Gt * Gt)
            mh = m / (1 - b1a ** t); vh = v / (1 - b2a ** t)
            X = _nm(X - lr * mh / (np.sqrt(vh) + eps))
        return X

    def lp_polish(x, deadline):
        x = _nm(x.copy())
        n, d = x.shape
        b = _mc(x)
        rad = 1e-3
        iu = np.triu_indices(n, 1)
        while time.time() < deadline and rad > 1e-14:
            c = (x @ x.T)[iu]
            mm = float(np.max(c))
            sel = c > mm - max(20 * rad, 1e-9)
            I, J = iu[0][sel], iu[1][sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            Aeq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                r = linprog(obj, A_ub=Aub, b_ub=-c[sel], A_eq=Aeq,
                            b_eq=np.zeros(n),
                            bounds=[(-rad, rad)] * (n * d) + [(None, None)],
                            method="highs")
            except Exception:
                break
            if r.status != 0:
                rad *= 0.5
                continue
            y = _nm(x + r.x[:-1].reshape(n, d))
            vv = _mc(y)
            if vv < b:
                x, b, rad = y, vv, min(rad * 1.5, 0.05)
            else:
                rad *= 0.5
        return x, b

    def opt_rotation(Xc, Xf, beta, rho0, n_iter, seed):
        rs = np.random.default_rng(seed)
        # initial direction
        u = rs.normal(size=6); u /= (np.linalg.norm(u) + 1e-12)
        c = u * rho0
        try:
            R = _Rvec(c)
        except Exception:
            R = np.eye(4)
        D = (Xc @ R.T) @ Xf.T
        blse, bc = _lse(D, beta)
        best_c = c.copy(); best_R = R.copy()
        step = max(0.06, rho0 * 0.5)
        for it in range(n_iter):
            prop = best_c + step * rs.normal(size=6)
            try:
                Rp = _Rvec(prop)
            except Exception:
                step *= 0.9
                continue
            Dp = (Xc @ Rp.T) @ Xf.T
            vlse, _ = _lse(Dp, beta)
            if vlse < blse:
                blse = vlse; best_c = prop; best_R = Rp
                step *= 1.06
            else:
                step *= 0.985
                if step < 1e-4:
                    step = 1e-4
        # true max of best
        Df = (Xc @ best_R.T) @ Xf.T
        tm = float(np.max(Df))
        return best_R, blse, tm

    K = 8
    cyc = 0
    # main rigid-cluster cycles
    while time.time() < SEARCH - 2.0:
        cyc += 1
        # recluster current walker
        try:
            lab = _sph_kmeans(x_cur, K, rng, iters=25)
        except Exception:
            lab = np.asarray([i % K for i in range(168)])
        deg, cmax, nc, _ = _degrees(x_cur)
        # order clusters: defect (min degree) first, then max-degree desc
        cids = list(range(K))
        info = []
        for k in cids:
            idx = np.where(lab == k)[0]
            if len(idx) == 0:
                continue
            dmax = int(np.max(deg[idx])) if len(idx) else 0
            dmin = int(np.min(deg[idx])) if len(idx) else 99
            info.append((k, dmin, dmax, len(idx)))
        # priority: contains global min-degree point, then high max degree
        gmin = int(np.min(deg))
        def keyf(t):
            k, dmin, dmax, sz = t
            has_def = 1 if dmin == gmin else 0
            return (-has_def, -dmax)
        info.sort(key=keyf)
        if not info:
            break
        for (k, dmin, dmax, sz) in info:
            if time.time() > SEARCH - 2.0:
                break
            idx = np.where(lab == k)[0]
            if len(idx) < 4 or len(idx) > 60:
                continue
            Xc = x_cur[idx].copy()
            mask = np.ones(168, dtype=bool); mask[idx] = False
            Xf = x_cur[mask].copy()
            # 3 restarts over magnitudes 0.05-0.5
            rhos = [0.08, 0.22, 0.42]
            # vary with cycle to cover range
            if cyc % 2 == 0:
                rhos = [0.05, 0.15, 0.32]
            bestR = None; best_tm = 1e9; best_lse = 1e9
            for ri, rh in enumerate(rhos):
                if time.time() > SEARCH - 1.5:
                    break
                try:
                    Rr, vlse, tm = opt_rotation(Xc, Xf, 220.0, rh, 90, 1000 + cyc * 100 + k * 10 + ri)
                except Exception:
                    continue
                if vlse < best_lse:
                    best_lse = vlse; best_tm = tm; bestR = Rr
            if bestR is None:
                continue
            # build proposal
            Xp = x_cur.copy()
            Xp[idx] = _nm(Xc @ bestR.T)
            inter = best_tm
            # skip hopeless huge collisions? still anneal a bit if moderately bad
            if inter > cur + 0.06:
                continue
            # short annealed LSE from proposal (unfrozen)
            try:
                Xo = adam_full(Xp, steps=380, b0=28., b1=420., lr0=0.02, lr1=0.0018)
            except Exception:
                continue
            vo = _mc(Xo)
            # decide to LP: if promising vs cur/best or topology changed potential
            do_lp = False
            if vo < best:
                do_lp = True
            elif vo < cur + 0.002:
                do_lp = True
            elif vo < best + 0.003 and (cyc % 2 == 0):
                do_lp = True
            if not do_lp:
                continue
            bud = min(1.2, max(0.4, SEARCH - time.time() - 0.5))
            if bud <= 0.15:
                # accept without LP
                if vo < best:
                    best = vo; x_best = Xo.copy(); h_best = _chash(x_best)
                    x_cur = Xo.copy(); cur = vo; h_cur = h_best
                continue
            try:
                Xl, vl = lp_polish(Xo, min(SEARCH + 2.0, time.time() + bud, HARD - 5.0))
            except Exception:
                continue
            hl = _chash(Xl)
            if vl < best - 1e-12:
                best = vl; x_best = Xl.copy(); h_best = hl
                x_cur = Xl.copy(); cur = vl; h_cur = hl
                # refresh labels after improvement
                break
            elif hl != h_cur and vl < cur + 0.0025 and vl < best + 0.002:
                # topology changed: walk to new basin even if slightly worse
                x_cur = Xl.copy(); cur = vl; h_cur = hl
            elif vl < cur - 5e-7:
                x_cur = Xl.copy(); cur = vl; h_cur = hl
        # alternate K between 6-10 to vary partition scale
        K = 6 + (cyc % 5)
        if K < 6:
            K = 6
        if K > 10:
            K = 10
        # occasional plain kick to avoid stall (small noise + anneal)
        if time.time() < SEARCH - 6.0 and cyc % 3 == 0:
            try:
                Xn = _nm(x_cur + 0.02 * rng.normal(size=(168, 4)))
                Xo = adam_full(Xn, steps=350, b0=28., b1=400., lr0=0.018, lr1=0.0015)
                vo = _mc(Xo)
                if vo < cur:
                    Xl, vl = lp_polish(Xo, min(time.time() + 0.8, HARD - 5.0))
                    if vl < best:
                        best = vl; x_best = Xl.copy(); h_best = _chash(x_best)
                    if vl < cur:
                        x_cur = Xl.copy(); cur = vl; h_cur = _chash(x_cur)
            except Exception:
                pass

    # final polish of best
    try:
        if time.time() < HARD - 2.0:
            x_best, best = lp_polish(x_best, HARD - 1.0)
    except Exception:
        pass
    return _nm(x_best)
