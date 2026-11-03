import time
import numpy as np

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def solve():
    import time
    import numpy as np
    t0 = time.time()
    deadline = t0 + 170.0
    rng = np.random.RandomState(0)

    n_d, d_d = 249, 6
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = None
    try:
        rec = np.load("record.npy")
    except Exception:
        rec = None

    cands = []
    if x0 is not None:
        try:
            x0 = np.asarray(x0, dtype=np.float64)
            if x0.shape == (n_d, d_d):
                cands.append(x0)
        except Exception:
            pass
    if rec is not None:
        try:
            rec = np.asarray(rec, dtype=np.float64)
            if rec.shape == (n_d, d_d):
                cands.append(rec)
        except Exception:
            pass
    if len(cands) == 0:
        cands.append(rng.standard_normal(size=(n_d, d_d)))

    def normed(a):
        a = np.asarray(a, dtype=np.float64)
        nr = np.linalg.norm(a, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return a / nr

    best = None
    best_val = 1e100
    for c in cands:
        cn = normed(c)
        v = _max_cos(cn)
        if v < best_val:
            best_val = v
            best = cn.copy()

    # ---------- LSE heavy-ball polish ----------
    def lse_run(X_init, betas, steps_list, lrs, mu, t_dead):
        X = X_init.copy()
        nn, dd = X.shape
        V = np.zeros_like(X)
        # current best for this run
        G0 = X @ X.T
        np.fill_diagonal(G0, -2.0)
        cur0 = float(G0.max())
        run_best = X.copy()
        run_val = cur0
        # also consider global best outside; caller merges
        for beta, steps, lr in zip(betas, steps_list, lrs):
            for s in range(steps):
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                m0 = float(G.max())
                # G -> beta*(G-m0) in place
                G -= m0
                G *= beta
                # exp; max is 0 so no overflow, underflow -> 0 fine
                W = np.exp(G)
                np.fill_diagonal(W, 0.0)
                S = float(W.sum())
                if not np.isfinite(S) or S < 1e-300:
                    S = 1e-300
                Grad = (W @ X) * (2.0 / S)
                dot = np.sum(Grad * X, axis=1, keepdims=True)
                Gt = Grad - dot * X
                V *= mu
                V -= lr * Gt
                vdot = np.sum(V * X, axis=1, keepdims=True)
                V -= vdot * X
                X += V
                nr = np.sqrt(np.sum(X * X, axis=1, keepdims=True))
                nr[nr == 0] = 1.0
                X /= nr
                if (s & 63) == 0:
                    if time.time() > t_dead:
                        # evaluate current X once
                        Ge = X @ X.T
                        np.fill_diagonal(Ge, -2.0)
                        ve = float(Ge.max())
                        if ve < run_val:
                            run_val = ve
                            run_best = X.copy()
                        return run_best, run_val
                    Ge = X @ X.T
                    np.fill_diagonal(Ge, -2.0)
                    ve = float(Ge.max())
                    if ve < run_val:
                        run_val = ve
                        run_best = X.copy()
            if time.time() > t_dead:
                break
        return run_best, run_val

    # stage 1: strong anneal from current best
    try:
        betas = [150.0, 500.0, 1500.0, 4000.0]
        stepss = [2500, 2500, 2500, 2500]
        lrss = [0.006, 0.003, 0.0012, 0.0005]
        nb, nv = lse_run(best, betas, stepss, lrss, 0.85, deadline - 60.0)
        if nv < best_val - 1e-12:
            best_val = nv
            best = nb.copy()
    except Exception:
        pass

    # basin hops: perturb + light re-polish
    sigmas = [0.02, 0.05, 0.012, 0.03]
    for hop, sigma in enumerate(sigmas):
        if time.time() > deadline - 45.0:
            break
        try:
            Xp = best + sigma * rng.standard_normal(size=best.shape)
            Xp = normed(Xp)
            betas2 = [500.0, 2000.0]
            steps2 = [1200, 1200]
            lrs2 = [0.002, 0.0006]
            # shorter for later hops if time low
            if time.time() > deadline - 70.0:
                steps2 = [800, 800]
            nb2, nv2 = lse_run(Xp, betas2, steps2, lrs2, 0.85, deadline - 30.0)
            if nv2 < best_val - 1e-12:
                best_val = nv2
                best = nb2.copy()
        except Exception:
            continue

    # ---------- greedy worst-pair micro polish ----------
    try:
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        # ensure best_val consistent
        best_val = float(G.max())
        n, d = best.shape
        per_max = G.max(axis=1)
        order = np.argsort(per_max)[-25:]
        flat = int(np.argmax(G))
        a, b = divmod(flat, n)
        radius = 1e-4
        # reserve a little time to return
        g_dead = deadline - 1.0
        while time.time() < g_dead and radius > 1e-13:
            improved = False
            # refresh worst info
            flat = int(np.argmax(G))
            a, b = divmod(flat, n)
            per_max = G.max(axis=1)
            order = np.argsort(per_max)[-25:]
            for _ in range(4000):
                if time.time() >= g_dead:
                    break
                r = rng.rand()
                if r < 0.7:
                    ii = a if rng.rand() < 0.5 else b
                else:
                    ii = int(order[rng.randint(len(order))])
                t = rng.standard_normal(d)
                dt = float(t @ best[ii])
                t = t - dt * best[ii]
                nm = float(np.linalg.norm(t))
                if nm < 1e-14:
                    continue
                t /= nm
                # 50% directed away from worst neighbour
                if rng.rand() < 0.5:
                    rival = b if ii == a else (a if ii == b else int(np.argmax(G[ii])))
                    dv = -best[rival].copy()
                    dv = dv - float(dv @ best[ii]) * best[ii]
                    n2 = float(np.linalg.norm(dv))
                    if n2 > 1e-12:
                        dv /= n2
                        t = 0.7 * dv + 0.3 * t
                        t = t - float(t @ best[ii]) * best[ii]
                        nn = float(np.linalg.norm(t))
                        if nn < 1e-14:
                            continue
                        t /= nn
                step_len = radius * (0.2 + 0.8 * rng.rand())
                trial_row = best[ii] + step_len * t
                nr = float(np.linalg.norm(trial_row))
                if nr == 0 or not np.isfinite(nr):
                    continue
                trial_row /= nr
                dots = best @ trial_row
                dots[ii] = -2.0
                if float(dots.max()) >= best_val:
                    continue
                # only possible improvement if ii touches current max
                # full check
                Xt = best.copy()
                Xt[ii] = trial_row
                Gt = Xt @ Xt.T
                np.fill_diagonal(Gt, -2.0)
                v = float(Gt.max())
                if v < best_val - 1e-12:
                    best = Xt
                    G = Gt
                    best_val = v
                    improved = True
                    flat = int(np.argmax(G))
                    a, b = divmod(flat, n)
            if not improved:
                radius *= 0.5
            # if improved keep radius to keep squeezing
        # final renormalize
        best = normed(best)
    except Exception:
        pass

    best = normed(best)
    # guard NaN
    best[~np.isfinite(best)] = 0.0
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best
