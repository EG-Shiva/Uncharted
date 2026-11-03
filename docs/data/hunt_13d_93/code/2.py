import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEADLINE = 172.0  # leave margin for 180s limit
    rng = np.random.default_rng(0)
    np.random.seed(0)

    # load starts
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape != (93, 13):
                continue
            a = _norm(np.asarray(a, dtype=np.float64))
            v = _maxcos(a)
            if v < best_val:
                best_val = v
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((93, 13)))
        best_val = _maxcos(best)

    n, d = 93, 13

    def anneal(X, steps, tau0, tau1, s0, s1, mu=0.9):
        V = np.zeros_like(X)
        ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
        ls = np.log(s1 / s0) if s0 != s1 else 0.0
        for k in range(steps):
            if time.time() - t0 > DEADLINE:
                break
            frac = k / max(1, steps - 1)
            tau = tau0 * np.exp(ltau * frac)
            step = s0 * np.exp(ls * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            thresh = cmax - max(5.0 * tau, 0.005)
            # weights
            E = np.exp((G - cmax) / tau)
            E[G <= thresh] = 0.0
            np.fill_diagonal(E, 0.0)
            S = E.sum(axis=1, keepdims=True) + 1e-12
            Gr = (E @ X) / S
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Ft = Gr - dot * X
            V = mu * V + (1.0 - mu) * Ft
            X = X - step * V
            X = _norm(X)
        return X

    def greedy(curX, cur_val, step_init, max_iters, tau=0.0012):
        curX = curX.copy()
        bestX = curX.copy()
        bv = cur_val
        step = step_init
        for _ in range(max_iters):
            if time.time() - t0 > DEADLINE:
                break
            G = curX @ curX.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            thresh = cmax - 0.006
            E = np.exp((G - cmax) / tau)
            E[G <= thresh] = 0.0
            np.fill_diagonal(E, 0.0)
            # if isolated (should not happen) jitter
            if E.sum() == 0:
                break
            S = E.sum(axis=1, keepdims=True) + 1e-12
            Gr = (E @ curX) / S
            dot = (Gr * curX).sum(axis=1, keepdims=True)
            Ft = Gr - dot * curX
            Xp = _norm(curX - step * Ft)
            Gp = Xp @ Xp.T
            np.fill_diagonal(Gp, -2.0)
            vp = float(np.max(Gp))
            if vp < cur_val:
                curX = Xp
                cur_val = vp
                if vp < bv:
                    bv = vp
                    bestX = Xp.copy()
                step = min(step * 1.2, 0.005)
                # early exit if step tiny and no progress possible
            else:
                step *= 0.5
                if step < 1e-13:
                    break
        return bestX, bv, curX, cur_val

    # quick initial squeeze of starting point
    try:
        _, best_val, best_c, best_c_val = greedy(best, best_val, 1e-3, 800)
        if best_c_val < best_val:
            best, best_val = best_c, best_c_val
        # keep best among refined
        # greedy returns bestX already, but ensure
        # re-evaluate
    except Exception:
        pass

    def hole_reinsert(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        rowmax = G.max(axis=1)
        k = int(np.argmax(rowmax))
        mask = np.ones(n, dtype=bool)
        mask[k] = False
        Xr = X[mask]  # 92x13
        M = 10000
        C = rng.standard_normal((M, d))
        C = _norm(C)
        D = C @ Xr.T  # Mx92
        worst = D.max(axis=1)
        idx = int(np.argmin(worst))
        Xn = X.copy()
        Xn[k] = C[idx]
        # small jitter others to break symmetry
        Xn += 0.0015 * rng.standard_normal(Xn.shape)
        return _norm(Xn)

    sigmas = [0.02, 0.01, 0.005, 0.03, 0.008, 0.015]
    max_trials = 28
    for trial in range(max_trials):
        if time.time() - t0 > 125.0:
            break
        sigma = sigmas[trial % len(sigmas)]
        mode = trial % 4
        if mode == 0:
            Xt = best + sigma * rng.standard_normal(best.shape)
            Xt = _norm(Xt)
        elif mode == 1:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            worst = np.argsort(rm)[-10:]
            Xt = best.copy() + 0.003 * rng.standard_normal(best.shape)
            Xt[worst] += sigma * rng.standard_normal((len(worst), d))
            Xt = _norm(Xt)
        elif mode == 2:
            try:
                Xt = hole_reinsert(best)
            except Exception:
                Xt = _norm(best + sigma * rng.standard_normal(best.shape))
        else:
            Xt = best + sigma * rng.standard_normal(best.shape)
            Xt = _norm(Xt)

        # explore
        Xt = anneal(Xt, steps=1300, tau0=0.015, tau1=0.0012, s0=0.015, s1=0.0008, mu=0.9)
        if time.time() - t0 > DEADLINE:
            v = _maxcos(Xt)
            if v < best_val:
                best, best_val = Xt, v
            break
        # refine
        try:
            bX, bv, _, _ = greedy(Xt, _maxcos(Xt), 8e-4, 700, tau=0.0012)
        except Exception:
            continue
        if bv < best_val - 1e-12:
            best, best_val = bX, bv

    # final intensive micro-polish on best with remaining time
    try:
        # coarse then fine
        for sinit, taut, iters in [(5e-4, 0.0012, 1500), (2e-4, 0.0008, 1500), (1e-4, 0.0006, 2000)]:
            if time.time() - t0 > DEADLINE - 2:
                break
            bX, bv, cX, cV = greedy(best, best_val, sinit, iters, tau=taut)
            if bv < best_val:
                best, best_val = bX, bv
            else:
                best = cX  # continue from moved point even if not better? keep best
                # revert to best for next round to stay monotonic
                best = bX if bv <= best_val else best
                # actually keep global best
                pass
            # also try tiny random kicks around best and re-polish
            if time.time() - t0 > DEADLINE - 2:
                break
    except Exception:
        pass

    best = _norm(best)
    return best
