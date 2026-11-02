import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def solve():
    t0 = time.time()
    DEADLINE = t0 + 170.0
    n0, d = 121, 8
    rng_global = np.random.default_rng(0)

    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n0, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X0 = rng_global.standard_normal((n0, d))
        X0 = _norm(X0)
        cands = [X0]
    vals0 = [_max_cos(a) for a in cands]
    Xbest = cands[int(np.argmin(vals0))].copy()
    Vbest = float(min(vals0))

    # ---------- Adam helpers ----------
    def grad_hinge(X, t):
        N = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        D = G - float(t)
        pos = D > 0
        f = float((D[pos] ** 2).sum() * 0.5) if np.any(pos) else 0.0
        # f = sum_{i<j} diff^2 ; full matrix double counts, so halve
        # gradient: sum_j h'(Gkj) Y_j with h'=2*diff for i<j
        # using full W= 2*diff*pos with diag 0 then GY double counts? No:
        # full sum = 2* upper sum, so halve f but gradient from full W
        # equals 2x upper gradient? Let's use upper-consistent scaling:
        # We want grad_k = sum_{j} 2*diff_kj+ Y_j. Full W gives exactly that.
        W = np.zeros_like(G)
        W[pos] = 2.0 * D[pos]
        np.fill_diagonal(W, 0.0)
        GY = W @ X
        dot = (GY * X).sum(axis=1, keepdims=True)
        Gp = GY - dot * X
        return f, Gp

    def grad_lse(X, beta):
        N = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        A = beta * (G - m)
        np.clip(A, -80, 0, out=A)
        E = np.exp(A)
        np.fill_diagonal(E, 0.0)
        # S over upper only
        S = float(np.triu(E, 1).sum())
        if not np.isfinite(S) or S <= 0:
            return m, np.zeros_like(X)
        # weights over full matrix: E/S gives 2x? Upper S, full E sum =2S,
        # W=E/S full gives grad double of upper definition; rescale by 0.5
        W = E / S * 0.5
        np.fill_diagonal(W, 0.0)
        GY = W @ X
        dot = (GY * X).sum(axis=1, keepdims=True)
        Gp = GY - dot * X
        f = m + np.log(S) / beta
        return float(f), Gp

    def adam_run(X, gfn, steps, lr, b1=0.9, b2=0.999, eps=1e-8):
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        for t in range(1, int(steps) + 1):
            if time.time() > DEADLINE - 35.0:
                break
            _, Gp = gfn(X)
            # clip extreme grads
            np.nan_to_num(Gp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            gn = float(np.linalg.norm(Gp) / max(1, Gp.shape[0]))
            if gn > 5.0:
                Gp *= (5.0 / gn)
            m = b1 * m + (1 - b1) * Gp
            v = b2 * v + (1 - b2) * (Gp * Gp)
            mh = m / (1 - b1 ** t)
            vh = v / (1 - b2 ** t)
            X = X - lr * mh / (np.sqrt(vh) + eps)
            X = _norm(X)
        return X

    def find_hole(base, rng, nsample=20000):
        C = rng.standard_normal((nsample, d))
        C = _norm(C)
        D = C @ base.T
        cm = D.max(axis=1)
        k = int(np.argmin(cm))
        return C[k].copy()

    endpoints = []

    extras_list = [5, 6, 7]
    for si in range(3):
        if time.time() > DEADLINE - 45.0:
            break
        rng = np.random.default_rng(1000 + si * 77)
        n_extra = extras_list[si % 3]
        base = Xbest.copy()
        # slight jitter to diversify basins (tiny, keeps holes similar but not identical)
        if si > 0:
            base = _norm(base + 1e-3 * rng.standard_normal(base.shape))
        pts = [base[i] for i in range(n0)]
        cur_base = base
        for _ in range(n_extra):
            h = find_hole(cur_base, rng, nsample=20000)
            pts.append(h)
            cur_base = np.stack(pts, axis=0)
        X = np.stack(pts, axis=0)  # N x d
        N = X.shape[0]

        # hinge at t=0.460
        X = adam_run(X, lambda Y: grad_hinge(Y, 0.460), steps=400, lr=0.02)
        if time.time() > DEADLINE - 40.0:
            # still try deflation quickly
            pass
        # LogSumExp annealing
        X = adam_run(X, lambda Y: grad_lse(Y, 100.0), steps=250, lr=0.01)
        X = adam_run(X, lambda Y: grad_lse(Y, 350.0), steps=250, lr=0.005)

        # deflate one at a time until 121
        while X.shape[0] > n0:
            if time.time() > DEADLINE - 38.0:
                break
            N = X.shape[0]
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            D = G - 0.44
            D[D < 0] = 0.0
            scores = (D ** 2).sum(axis=1)
            kill = int(np.argmax(scores))
            mask = np.ones(N, dtype=bool)
            mask[kill] = False
            X = X[mask].copy()
            # 150 cheap Adam steps after each deletion (hinge t=0.460)
            X = adam_run(X, lambda Y: grad_hinge(Y, 0.460), steps=150, lr=0.008)

        if X.shape[0] != n0:
            continue
        # final short LSE repair for endpoint quality
        X = adam_run(X, lambda Y: grad_lse(Y, 250.0), steps=200, lr=0.005)
        v = _max_cos(X)
        endpoints.append((v, X.copy()))
        # early exit if time low
        if time.time() > DEADLINE - 38.0:
            break

    # keep only endpoints below 0.462, else keep best anyway for LP attempt
    good = [(v, X) for v, X in endpoints if v < 0.462]
    if good:
        good.sort(key=lambda z: z[0])
        Xcand, Vcand = good[0][1], good[0][0]
    elif endpoints:
        endpoints.sort(key=lambda z: z[0])
        Xcand, Vcand = endpoints[0][1], endpoints[0][0]
    else:
        Xcand, Vcand = Xbest.copy(), Vbest

    # ---------- LP polish single best endpoint for at most 30s ----------
    def lp_polish(x, seconds, radius_init=3e-4):
        start = time.time()
        x = _norm(x)
        best = _max_cos(x)
        radius = radius_init
        n = x.shape[0]
        iu = np.triu_indices(n, 1)
        it = 0
        while time.time() - start < seconds and time.time() < DEADLINE - 1.5:
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
            y = _norm(y)
            vv = _max_cos(y)
            if vv < best:
                x, best = y, vv
                radius = min(radius * 1.6, 0.02)
            else:
                radius *= 0.5
        return x, best

    # only LP if promising or time allows; cap 30s
    if Vcand < 0.462 or Vcand < Vbest + 0.005:
        Xp, Vp = lp_polish(Xcand, seconds=30.0)
    else:
        Xp, Vp = Xcand, Vcand

    if Vp < Vbest and np.all(np.isfinite(Xp)):
        Xbest, Vbest = _norm(Xp), Vp

    # tiny shrink for strictness (avoid boundary issues) - renormalize after
    Xbest = _norm(Xbest)
    return Xbest
