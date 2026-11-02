import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X):
    n, d = X.shape
    B = np.empty((n, d, 4), dtype=np.float64)
    for i in range(n):
        x = X[i]
        k = int(np.argmin(np.abs(x)))
        T = np.empty((d, 4), dtype=np.float64)
        c = 0
        for j in range(d):
            if j == k:
                continue
            v = -x[j] * x
            v[j] += 1.0
            T[:, c] = v
            c += 1
        Q, _ = np.linalg.qr(T)
        B[i] = Q[:, :4]
    return B

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    EXP_END = t0 + 118.0
    LP_CUTOFF = t0 + 166.0
    rng = np.random.default_rng(0)
    n, d = 453, 5
    # load
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                continue
            nn = np.linalg.norm(cc, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            cc = cc / nn
            if not np.all(np.isfinite(cc)):
                continue
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)

    iu_r, iu_c = np.triu_indices(n, 1)

    def find_holes(Xkeep, k, n_cand=12000):
        C = rng.standard_normal((n_cand, d))
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-30)
        D = C @ Xkeep.T
        md = np.max(D, axis=1)
        holes = []
        # diversity: penalize closeness to already chosen holes
        for _ in range(k):
            bi = int(np.argmin(md))
            h = C[bi].copy()
            # local refine: push away from closest keeper
            for __ in range(40):
                dots = Xkeep @ h
                ci = int(np.argmax(dots))
                xc = Xkeep[ci]
                c_ = float(dots[ci])
                t = xc - c_ * h
                tn = float(np.linalg.norm(t))
                if tn < 1e-12:
                    break
                t /= tn
                h = h - 0.008 * t
                nh = float(np.linalg.norm(h))
                if not np.isfinite(nh) or nh == 0:
                    break
                h /= nh
            holes.append(h)
            # suppress candidates near h
            dh = C @ h
            md = np.maximum(md, dh)
            # avoid re-pick same: set picked to large
            md[bi] = 10.0
        return np.array(holes)

    def anneal(X, steps, tau0, tau1, lr0, lr1, mu=0.85):
        nonlocal_best = [bestV]
        # use closure via lists
        v = np.zeros_like(X)
        ltau = np.log(max(tau1,1e-12) / max(tau0,1e-12))
        llr = np.log(max(lr1,1e-12) / max(lr0,1e-12))
        for it in range(steps):
            if it % 40 == 0 and time.time() > EXP_END:
                break
            frac = it / max(steps - 1, 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            # elitist track is done outside; just dynamics here
            # softmax weights
            # clip for overflow
            E = (G - m) / tau
            # E<=0, exp safe
            P = np.exp(E)
            # gradient
            Grad = P @ X
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            # normalize scale to avoid blowup: limit norm
            # scale by 1/(mean weight) approx: divide by max row sum
            rs = np.sum(P, axis=1, keepdims=True) + 1e-12
            # use average: Gt / rs keeps step bounded
            Gt = Gt / rs
            v = mu * v + (1.0 - mu) * Gt
            X = X - lr * v * 8.0
            X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
            if not np.all(np.isfinite(X)):
                break
        return X

    # ---- exploration: hub ejection + anneal ----
    n_cycles = 200
    for cyc in range(n_cycles):
        if time.time() > EXP_END - 1.0:
            break
        # compute row max / degrees on bestX (reference) + jitter choice
        try:
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            rm = np.max(G, axis=1)
        except Exception:
            rm = np.zeros(n)
        # number to eject 1..3
        if cyc % 7 == 3:
            k = 3
        elif cyc % 3 == 1:
            k = 2
        else:
            k = 1
        # candidate pool: top constrained
        order = np.argsort(-rm)
        pool = order[:24]
        # deterministic random pick among pool
        perm = rng.permutation(pool)[:k]
        victims = [int(v) for v in perm]
        Xcur = bestX.copy()
        mask = np.ones(n, dtype=bool)
        mask[victims] = False
        Xkeep = Xcur[mask]
        try:
            holes = find_holes(Xkeep, k, n_cand=10000 if k > 1 else 14000)
        except Exception:
            continue
        for vi, h in zip(victims, holes):
            Xcur[vi] = h
        # small collective tangent shake (larger on neighbours of victims)
        sigma = 0.006 if (cyc % 4 != 2) else 0.016
        N = rng.standard_normal((n, d))
        N -= np.sum(N * Xcur, axis=1, keepdims=True) * Xcur
        Xcur = Xcur + sigma * N
        Xcur /= (np.linalg.norm(Xcur, axis=1, keepdims=True) + 1e-30)
        # anneal params vary per cycle for diversity
        if cyc % 4 == 0:
            Xcur = anneal(Xcur, 320, 0.004, 0.0004, 0.005, 0.0006, 0.85)
        elif cyc % 4 == 1:
            Xcur = anneal(Xcur, 380, 0.002, 0.0002, 0.004, 0.0004, 0.88)
        elif cyc % 4 == 2:
            Xcur = anneal(Xcur, 260, 0.006, 0.0006, 0.006, 0.0008, 0.80)
        else:
            Xcur = anneal(Xcur, 340, 0.003, 0.00025, 0.0045, 0.0005, 0.86)
        try:
            vv = _max_cos(Xcur)
        except Exception:
            continue
        if vv < bestV and np.isfinite(vv):
            bestV = vv
            bestX = Xcur.copy()
            # intensify around new best: quick micro greedy shave
            # do a few worst-pair steps inline
            for _ in range(30):
                if time.time() > EXP_END - 0.5:
                    break
                G2 = bestX @ bestX.T
                np.fill_diagonal(G2, -2.0)
                flat = int(np.argmax(G2))
                i = flat // n; j = flat % n
                xi = bestX[i].copy(); xj = bestX[j].copy()
                cij = float(xi @ xj)
                ti = xj - cij * xi
                ni = float(np.linalg.norm(ti))
                if ni < 1e-12:
                    break
                ti /= ni
                tj = xi - cij * xj
                nj = float(np.linalg.norm(tj))
                if nj < 1e-12:
                    break
                tj /= nj
                step = 2e-4
                vi2 = xi - step * ti; vi2 /= np.linalg.norm(vi2)
                vj2 = xj - step * tj; vj2 /= np.linalg.norm(vj2)
                o1 = bestX[i].copy(); o2 = bestX[j].copy()
                bestX[i] = vi2; bestX[j] = vj2
                m2 = _max_cos(bestX)
                if m2 < bestV:
                    bestV = m2
                else:
                    bestX[i] = o1; bestX[j] = o2
                    break

    # ---- final tiny LP polish (few solves) ----
    try:
        if time.time() < LP_CUTOFF - 3.0:
            Nvar = n * 4
            c_obj = np.zeros(Nvar + 1)
            c_obj[-1] = 1.0
            ar4 = np.arange(4)
            X = bestX.copy()
            radius = 6e-5
            for _ in range(4):
                if time.time() > LP_CUTOFF or time.time() > T_END - 5.0:
                    break
                if radius < 1e-12:
                    break
                G = X @ X.T
                c_all = G[iu_r, iu_c]
                m = float(np.max(c_all))
                if m < bestV and np.isfinite(m):
                    bestV = m
                    bestX = X.copy()
                slack_all = m - c_all
                tight = np.where(slack_all < 1.5e-5)[0]
                if tight.size == 0:
                    tight = np.argsort(slack_all)[:500]
                if tight.size > 2000:
                    part = np.argpartition(slack_all[tight], 2000)[:2000]
                    tight = tight[part]
                I = iu_r[tight]; J = iu_c[tight]
                p = int(I.size)
                b_ub = slack_all[tight].astype(np.float64)
                B = _tangent_bases(X)
                aI = np.einsum('pdk,pd->pk', B[I], X[J])
                aJ = np.einsum('pdk,pd->pk', B[J], X[I])
                cols2 = np.empty((p, 9), dtype=np.int64)
                cols2[:, 0:4] = I[:, None] * 4 + ar4[None, :]
                cols2[:, 4:8] = J[:, None] * 4 + ar4[None, :]
                cols2[:, 8] = Nvar
                vals2 = np.empty((p, 9), dtype=np.float64)
                vals2[:, 0:4] = aI
                vals2[:, 4:8] = aJ
                vals2[:, 8] = -1.0
                rows = np.repeat(np.arange(p), 9)
                A_ub = coo_matrix((vals2.ravel(), (rows, cols2.ravel())), shape=(p, Nvar + 1)).tocsr()
                rem = min(LP_CUTOFF - time.time(), T_END - 5.0 - time.time(), 8.0)
                if rem < 0.5:
                    break
                bounds = [(-radius, radius)] * Nvar + [(None, None)]
                try:
                    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs", options={"time_limit": max(0.5, rem)})
                except Exception:
                    radius *= 0.5
                    continue
                if res.status != 0 or res.x is None:
                    radius *= 0.5
                    continue
                U = np.asarray(res.x[:Nvar], dtype=np.float64).reshape(n, 4)
                if not np.all(np.isfinite(U)):
                    radius *= 0.5
                    continue
                DX = np.einsum('ndk,nk->nd', B, U)
                Xt = X + DX
                Xt /= (np.linalg.norm(Xt, axis=1, keepdims=True) + 1e-30)
                mt = _max_cos(Xt)
                if mt < m - 1e-13 and np.isfinite(mt):
                    X = Xt
                    if mt < bestV:
                        bestV = mt
                        bestX = Xt.copy()
                    radius = min(radius * 1.4, 3e-4)
                else:
                    radius *= 0.5
    except Exception:
        pass

    bestX /= (np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-30)
    return bestX
