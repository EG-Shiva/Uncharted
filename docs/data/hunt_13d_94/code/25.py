import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def lbfgs_smooth(X0, s_list, iter_list, deadline=None):
    n, d = X0.shape
    iu0, iu1 = np.triu_indices(n, 1)
    X = X0.copy()
    for s, maxiter in zip(s_list, iter_list):
        if deadline is not None and time.time() > deadline:
            break
        def fg(Yflat):
            Y = Yflat.reshape(n, d)
            r = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True)) + 1e-30
            Xn = Y / r
            G = Xn @ Xn.T
            g = G[iu0, iu1]
            M = np.max(g)
            # stable exp
            e = np.exp(s * (g - M))
            S = np.sum(e)
            if S <= 0 or not np.isfinite(S):
                S = 1.0
                e = np.ones_like(e) / len(e)
            else:
                e = e / S
            f = M + np.log(max(S, 1e-300)) / s if True else M
            # build symmetric weight matrix action without full matrix
            # grad_X = W @ Xn
            # use accumulation
            W = np.zeros((n, n), dtype=float)
            W[iu0, iu1] = e
            # symmetrize
            W = W + W.T
            grad_X = W @ Xn
            dot = np.sum(grad_X * Xn, axis=1, keepdims=True)
            Gt = grad_X - dot * Xn
            grad_Y = Gt / r
            return float(f), grad_Y.ravel()
        try:
            res = minimize(fg, X.ravel(), jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-5, "maxfun": 10000, "maxcor": 10})
            Xn = res.x.reshape(n, d)
            nr = np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-30
            X = Xn / nr
        except Exception:
            break
        if not np.all(np.isfinite(X)):
            X = X0.copy()
            break
    return X

def greedy_best(others, rng, n_try=2000):
    d = others.shape[1]
    C = rng.normal(size=(n_try, d))
    C /= np.sqrt(np.sum(C * C, axis=1, keepdims=True)) + 1e-30
    D = C @ others.T
    M = np.max(D, axis=1)
    b = int(np.argmin(M))
    return C[b]

def lp_polish(x, seconds, deadline=1e18):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    BUDGET = 175.0
    PHASE1_END = t0 + 135.0
    rng = np.random.default_rng(7)
    n, d = 94, 13
    bestX = None
    best = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                v = max_cos(a)
                if v < best:
                    best = v
                    bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = rng.normal(size=(n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        best = max_cos(bestX)
    cands = [(best, bestX.copy())]
    def add_cand(val, mat):
        cands.append((val, mat.copy()))
        cands.sort(key=lambda z: z[0])
        del cands[6:]
    sched_choices = [
        ([40, 150, 450], [80, 100, 120]),
        ([30, 100, 300], [80, 80, 100]),
        ([60, 200, 600], [60, 80, 100]),
        ([50, 150, 350], [100, 100, 150]),
        ([25, 80, 250], [60, 80, 80]),
    ]
    sigmas = [0.005, 0.012, 0.03, 0.07, 0.15, 0.30]
    trial = 0
    while time.time() < PHASE1_END and time.time() - t0 < BUDGET - 40:
        trial += 1
        # pick base
        if rng.random() < 0.60:
            base = bestX
        else:
            base = cands[int(rng.integers(len(cands)))][1]
        r = rng.random()
        Y = base.copy()
        if r < 0.10:
            # from scratch partial
            Y = rng.normal(size=(n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        elif r < 0.50:
            sigma = sigmas[int(rng.integers(len(sigmas)))]
            # occasionally subset shake targeting jammed cluster
            if rng.random() < 0.40:
                try:
                    G = base @ base.T
                    np.fill_diagonal(G, -2.0)
                    M = float(np.max(G))
                    cnt = np.sum(G > M - 2e-4, axis=1)
                    # jammed first
                    order = np.argsort(-cnt)
                    ksub = int(rng.integers(6, 20))
                    idx = order[:ksub]
                    # bigger noise on jammed subset
                    Y = base.copy()
                    Y[idx] += sigma * 1.5 * rng.normal(size=(len(idx), d))
                    # small noise elsewhere
                    mask = np.ones(n, bool); mask[idx] = False
                    Y[mask] += (sigma * 0.25) * rng.normal(size=(np.sum(mask), d))
                except Exception:
                    Y = base + sigma * rng.normal(size=(n, d))
            else:
                Y = base + sigma * rng.normal(size=(n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        else:
            # kill and greedy reinsert worst / jammed points
            Y = base.copy()
            try:
                G = Y @ Y.T
                np.fill_diagonal(G, -2.0)
                M = float(np.max(G))
                per_point_max = np.max(G, axis=1)
                cnt = np.sum(G > M - 3e-4, axis=1)
                # score = per_point_max + 0.002*cnt normalized
                score = per_point_max + 0.0015 * cnt
                order = np.argsort(-score)
            except Exception:
                order = rng.permutation(n)
            k = int(rng.integers(1, 6))
            # with prob mix random
            kill = []
            for ii in range(k):
                if rng.random() < 0.75:
                    # take next worst not already taken
                    for cand_idx in order:
                        if int(cand_idx) not in kill:
                            kill.append(int(cand_idx))
                            break
                else:
                    ri = int(rng.integers(n))
                    if ri not in kill:
                        kill.append(ri)
                    else:
                        for cand_idx in rng.permutation(n):
                            if int(cand_idx) not in kill:
                                kill.append(int(cand_idx))
                                break
            for idx in kill:
                others = np.delete(Y, idx, axis=0)
                nt = 2500 if k <= 2 else 1200
                Y[idx] = greedy_best(others, rng, n_try=nt)
            # small shake after reinsert to break exact ties
            Y += 0.003 * rng.normal(size=(n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        si = int(rng.integers(len(sched_choices)))
        s_list, it_list = sched_choices[si]
        # randomize iters slightly
        it_list = [int(max(40, x * rng.uniform(0.7, 1.2))) for x in it_list]
        try:
            Yopt = lbfgs_smooth(Y, s_list, it_list, deadline=PHASE1_END)
            val = max_cos(Yopt)
        except Exception:
            continue
        if not np.isfinite(val):
            continue
        if val < cands[-1][0] + 0.005 or val < best:
            add_cand(val, Yopt)
        if val < best - 1e-12:
            best = val
            bestX = Yopt.copy()
        if time.time() >= PHASE1_END:
            break
    # Phase 2 polish
    cands.sort(key=lambda z: z[0])
    plist = []
    plist.append(bestX.copy())
    for v, m in cands:
        if len(plist) >= 4:
            break
        dup = False
        for q in plist:
            try:
                if np.linalg.norm(m - q) < 1e-9:
                    dup = True
                    break
            except Exception:
                pass
        if not dup:
            plist.append(m.copy())
    vals = [max_cos(m) for m in plist]
    o = np.argsort(np.array(vals))
    plist = [plist[i] for i in o]
    remaining = BUDGET - (time.time() - t0) - 2.0
    if remaining < 4:
        remaining = 4.0
    per = min(remaining / max(1, len(plist)), 16.0)
    for m in plist:
        if time.time() - t0 > BUDGET - 3.0:
            break
        try:
            mp = lp_polish(m, seconds=per, deadline=t0 + BUDGET - 2.0)
            v = max_cos(mp)
            if v < best:
                best = v
                bestX = mp.copy()
        except Exception:
            continue
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    bestX = bestX / nr
    return bestX
