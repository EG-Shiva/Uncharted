import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    return rm, G

def _smooth_opt(X0, beta, maxiter, maxcor=10):
    n, d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.empty_like(D)
        np.exp(D, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if S <= 0 or not np.isfinite(S):
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20,
                                "gtol": 1e-12, "maxfun": maxiter * 5,
                                "maxcor": maxcor})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return Y

def _lp_polish(x, seconds, t0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 8e-4
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(iu_c.max())
        sel = iu_c > m - max(8.0 * radius, 3e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 3200:
            thr = np.sort(iu_c[sel])[-3200]
            sel = iu_c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-iu_c[sel], A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.008)
        else:
            radius *= 0.5
    return x

def solve():
    t_start = time.time()
    BUDGET = 168.0
    def remaining():
        return BUDGET - (time.time() - t_start)
    rng = np.random.default_rng(12345)
    n, d = 250, 6
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = z / np.linalg.norm(z, axis=1, keepdims=True)
                cands.append(z)
        except Exception:
            pass
    if not cands:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    elite = min(cands, key=_max_cos).copy()
    elite_val = _max_cos(elite)
    cur = elite.copy()
    cur_val = elite_val

    # Stage A: progressive collective anneal of elite
    if remaining() > 50:
        for beta, it in [(4000.0, 100), (12000.0, 120), (28000.0, 120)]:
            if remaining() < 35:
                break
            try:
                nxt = _smooth_opt(cur, beta, it)
            except Exception:
                break
            v = _max_cos(nxt)
            if v < elite_val:
                elite = nxt.copy(); elite_val = v
            # move wandering front if not far worse
            if v < cur_val + 5e-4:
                cur = nxt.copy(); cur_val = v
            else:
                cur = elite.copy(); cur_val = elite_val

    # Stage B: basin hopping with topology-changing moves
    trial = 0
    while remaining() > 24 and trial < 60:
        trial += 1
        # choose base: mostly wanderer, sometimes elite
        if rng.random() < 0.75:
            base = cur
            base_val = cur_val
        else:
            base = elite
            base_val = elite_val
        try:
            rm, G = _rowmax(base)
        except Exception:
            rm = np.zeros(n); G = base @ base.T
        iu = np.triu_indices(n, 1)
        try:
            gv = G[iu]
            k = int(np.argmax(gv))
            a, b = int(iu[0][k]), int(iu[1][k])
            order = np.argsort(-rm)
        except Exception:
            a, b = 0, 1
            order = np.arange(n)
            gv = None
        mode = trial % 3
        Y = base.copy()
        try:
            if mode == 0:
                # block void reinsertion: replace k worst by greedy holes
                kblk = 6 + (trial % 4)  # 6..9
                victims = [int(v) for v in order[:kblk]]
                # mix one random rattler to diversify
                if rng.random() < 0.5:
                    victims[-1] = int(rng.integers(0, n))
                for w in victims:
                    Nc = 8000
                    cand = rng.standard_normal((Nc, d))
                    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                    sc = cand @ Y.T
                    sc[:, w] = -2.0
                    mx = sc.max(axis=1)
                    bi = int(np.argmin(mx))
                    Y[w] = cand[bi]
                Y += rng.standard_normal(Y.shape) * 0.0006
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta0, it0 = 7000.0, 90
            elif mode == 1:
                # large cluster shake around worst pair
                dots_a = Y @ Y[a]
                dots_b = Y @ Y[b]
                neigh = np.where((dots_a > base_val - 0.03) | (dots_b > base_val - 0.03))[0]
                neigh = [int(v) for v in neigh]
                rng.shuffle(neigh)
                sig = 0.05 + 0.09 * rng.random()
                for v in neigh[:26]:
                    if v == a or v == b:
                        continue
                    Y[v] += rng.standard_normal(d) * sig * 0.7
                # split the bottleneck pair apart
                eps = 0.01 + 0.02 * rng.random()
                na = Y[a] - eps * Y[b]
                nb = Y[b] - eps * Y[a]
                na /= np.linalg.norm(na); nb /= np.linalg.norm(nb)
                Y[a] = na; Y[b] = nb
                # a few global jolts
                extra = rng.choice(n, 5, replace=False)
                Y[extra] += rng.standard_normal((5, d)) * 0.03
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta0, it0 = 8000.0, 100
            else:
                # random subset reset for new basin
                krs = 10 + (trial % 5)
                idx = rng.choice(n, krs, replace=False)
                R = rng.standard_normal((krs, d))
                R /= np.linalg.norm(R, axis=1, keepdims=True)
                Y[idx] = R
                # slight shake of worst neighbourhood to accommodate
                Y += rng.standard_normal(Y.shape) * 0.0004
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                beta0, it0 = 5000.0, 100
            Y2 = _smooth_opt(Y, beta0, it0)
            v2 = _max_cos(Y2)
            # escalate if promising (within 3e-4 of elite)
            if v2 < elite_val + 3e-4 and remaining() > 28:
                try:
                    Y3 = _smooth_opt(Y2, 20000.0, 70)
                    v3 = _max_cos(Y3)
                    if v3 < v2:
                        Y2, v2 = Y3, v3
                except Exception:
                    pass
            if v2 < elite_val - 1e-12:
                elite = Y2.copy(); elite_val = v2
                cur = Y2.copy(); cur_val = v2
                if remaining() > 26:
                    try:
                        sq = _lp_polish(elite, min(1.5, remaining() - 24.0), time.time())
                        vs = _max_cos(sq)
                        if vs < elite_val:
                            elite = sq; elite_val = vs
                            cur = sq.copy(); cur_val = vs
                    except Exception:
                        pass
            elif v2 < cur_val:
                cur = Y2.copy(); cur_val = v2
            elif v2 < cur_val + 2e-4 and rng.random() < 0.25:
                # sideways drift to new basin
                cur = Y2.copy(); cur_val = v2
            # occasional pull back if wanderer too bad
            if cur_val > elite_val + 5e-4 and rng.random() < 0.3:
                cur = elite.copy(); cur_val = elite_val
        except Exception:
            continue

    # Stage C: final squeeze of elite
    if remaining() > 6:
        try:
            nxt = _smooth_opt(elite, 30000.0, 100)
            v = _max_cos(nxt)
            if v < elite_val:
                elite = nxt; elite_val = v
        except Exception:
            pass
    if remaining() > 4:
        try:
            polished = _lp_polish(elite, remaining() - 1.2, time.time())
            vp = _max_cos(polished)
            if vp < elite_val:
                elite = polished; elite_val = vp
        except Exception:
            pass
    elite = elite / np.linalg.norm(elite, axis=1, keepdims=True)
    elite = np.where(np.isfinite(elite), elite, 0.0)
    elite = elite / np.maximum(np.linalg.norm(elite, axis=1, keepdims=True), 1e-12)
    return elite
