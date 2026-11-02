import time
import numpy as np
from scipy.optimize import minimize

def solve():
    t0 = time.time()
    BUDGET = 155.0  # leave margin for verifier 180s
    rng = np.random.default_rng(7)
    n, d = 76, 15
    iu0 = np.triu_indices(n, 1)
    iu0r = iu0[0].astype(np.int64)
    iu0c = iu0[1].astype(np.int64)
    M = len(iu0r)

    def max_cos(Y):
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())

    # load base
    base = None
    best_score = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                s = max_cos(a)
                if s < best_score:
                    best_score = s
                    base = a.copy()
        except Exception:
            pass
    if base is None:
        base = rng.standard_normal((n, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        best_score = max_cos(base)
    global_best = base.copy()
    global_score = best_score

    # factory for smooth LSE value+grad
    def make_fg(T):
        def fg(xflat):
            X = xflat.reshape(n, d)
            nr = np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
            Y = X / nr
            # pairwise dots upper
            # compute G via matmul
            G = Y @ Y.T
            g = G[iu0r, iu0c]
            gm = float(g.max())
            # stable exp
            e = np.exp((g - gm) / T)
            S = float(e.sum()) + 1e-300
            F = gm + T * np.log(S)
            w = e / S
            # build symmetric weight matrix
            W = np.zeros((n, n), dtype=np.float64)
            W[iu0r, iu0c] = w
            # symmetrize (diag stays 0)
            W = W + W.T
            gradY = W @ Y
            dot = np.sum(gradY * Y, axis=1, keepdims=True)
            gradX = (gradY - dot * Y) / nr
            return F, gradX.ravel()
        return fg

    def run_anneal(X0, Ts, iters):
        x = X0.ravel().copy()
        for T, mi in zip(Ts, iters):
            if time.time() - t0 > BUDGET:
                break
            fg = make_fg(T)
            try:
                res = minimize(fg, x, method="L-BFGS-B", jac=True,
                               options={"maxiter": mi, "ftol": 1e-12,
                                        "gtol": 1e-7, "maxfun": mi * 20,
                                        "maxcor": 10})
                x = res.x
            except Exception:
                break
        X = x.reshape(n, d)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def contacts_of(Y, thr):
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        mask = G > (m - thr)
        cnt = mask.sum(axis=1)
        return m, cnt, G

    def largest_holes(Y, k, nprobe=4000):
        P = rng.standard_normal((nprobe, d))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        S = P @ Y.T
        smax = S.max(axis=1)
        idx = np.argsort(smax)[:k]
        return P[idx]

    # quick initial squeeze of base with sharp T to ensure local opt
    try:
        if time.time() - t0 < BUDGET - 20:
            Xp = run_anneal(global_best, [0.006, 0.003, 0.0018], [120, 150, 150])
            sp = max_cos(Xp)
            if sp < global_score:
                global_score = sp
                global_best = Xp.copy()
    except Exception:
        pass

    trial = 0
    # temperature schedules
    sched_fast = ([0.02, 0.01, 0.005, 0.0025], [60, 60, 90, 120])
    sched_sharp = ([0.01, 0.005, 0.0025, 0.0015], [60, 80, 120, 150])

    while time.time() - t0 < BUDGET - 6:
        trial += 1
        # choose base for diversification/intensification
        if trial % 7 == 0:
            # from original base to diversify
            cur = base.copy()
        else:
            cur = global_best.copy()

        m, cnt, G = contacts_of(cur, 0.004)
        order = np.argsort(cnt)  # loose first
        mode = trial % 4

        X0 = cur.copy()
        if mode == 0:
            # full shake, varying sigma
            sigmas = [0.04, 0.10, 0.20, 0.35]
            sigma = sigmas[(trial // 4) % len(sigmas)]
            X0 = cur + sigma * rng.standard_normal((n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            Ts, Its = sched_fast
        elif mode == 1:
            # teleport loosest points into voids
            k = 8 + (trial // 4) % 5  # 8..12
            loose = order[:k]
            holes = largest_holes(cur, k, nprobe=4000)
            jitter = 0.02
            X0[loose] = holes + jitter * rng.standard_normal((k, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            # small shake of rest
            rest_mask = np.ones(n, dtype=bool)
            rest_mask[loose] = False
            X0[rest_mask] += 0.03 * rng.standard_normal((rest_mask.sum(), d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            Ts, Its = sched_sharp
        elif mode == 2:
            # subset hard shake (jammed cluster)
            # pick random subset biased to high-contact points
            k = 10 + (trial % 6)
            # high contact indices
            hc = np.where(cnt >= np.median(cnt))[0]
            if len(hc) < k:
                hc = np.arange(n)
            chosen = rng.choice(hc, size=min(k, len(hc)), replace=False)
            X0[chosen] += 0.30 * rng.standard_normal((len(chosen), d))
            # others tiny shake
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            Ts, Its = sched_fast
        else:
            # medium shake + random rotation of a subspace for topology change
            sigma = 0.12
            X0 = cur + sigma * rng.standard_normal((n, d))
            X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
            Ts, Its = sched_sharp

        try:
            Xt = run_anneal(X0, Ts, Its)
        except Exception:
            continue
        st = max_cos(Xt)
        if st < global_score:
            global_score = st
            global_best = Xt.copy()
            # extra squeeze immediately
            if time.time() - t0 < BUDGET - 10:
                try:
                    Xs = run_anneal(global_best, [0.0015], [200])
                    ss = max_cos(Xs)
                    if ss < global_score:
                        global_score = ss
                        global_best = Xs.copy()
                except Exception:
                    pass
        # opportunistic: if close, try one more sharp polish
        elif st < global_score + 0.004 and (trial % 5 == 0):
            if time.time() - t0 < BUDGET - 10:
                try:
                    Xs = run_anneal(Xt, [0.0018], [150])
                    ss = max_cos(Xs)
                    if ss < global_score:
                        global_score = ss
                        global_best = Xs.copy()
                except Exception:
                    pass

    # final polish
    try:
        if time.time() - t0 < 172:
            Xs = run_anneal(global_best, [0.002, 0.0012], [120, 200])
            ss = max_cos(Xs)
            if ss < global_score:
                global_score = ss
                global_best = Xs.copy()
    except Exception:
        pass

    global_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    # guard NaN
    if not np.all(np.isfinite(global_best)):
        global_best = base / np.linalg.norm(base, axis=1, keepdims=True)
    return global_best
