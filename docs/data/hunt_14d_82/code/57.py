import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def solve():
    t0 = time.time()
    HARD = t0 + 165.0
    n, d = 82, 14
    rng = np.random.default_rng(0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                cands.append(_norm(a))
        except Exception:
            pass
    if len(cands) == 0:
        X_best = _norm(rng.standard_normal((n, d)))
    else:
        cands.sort(key=_max_cos)
        X_best = cands[0].copy()
    best_max = _max_cos(X_best)

    def consider(X):
        nonlocal X_best, best_max
        if X is None:
            return False
        X = np.asarray(X, dtype=float)
        if X.shape != (n, d) or not np.all(np.isfinite(X)):
            return False
        X = _norm(X)
        m = _max_cos(X)
        if m < best_max - 1e-12:
            best_max = m
            X_best = X.copy()
            return True
        return False

    iu = np.triu_indices(n, 1)
    iu0_r, iu0_c = iu

    def quench(X_init, steps, tau0, tau1, lr0, lam0, Ktop=500):
        Y = X_init.copy()
        m1 = np.zeros_like(Y)
        m2 = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        local_best_X = _norm(Y)
        local_best_m = _max_cos(local_best_X)
        for st in range(steps):
            if time.time() > HARD - 6:
                break
            frac = st / max(1, steps - 1)
            tau = tau0 * (1 - frac) + tau1 * frac
            lr = lr0 * (1.0 - 0.6 * frac)
            lam = lam0 * (1.0 - frac)  # shell weight decays to pure minimax
            r = np.linalg.norm(Y, axis=1, keepdims=True)
            r[r == 0] = 1.0
            X = Y / r
            G = X @ X.T
            np.fill_diagonal(G, -np.inf)
            mx = float(np.max(G))
            # softmax weights (upper)
            S = (G - mx) / tau
            np.clip(S, -50, 0, out=S)
            E = np.exp(S)
            np.fill_diagonal(E, 0.0)
            Eu = E[iu0_r, iu0_c]
            ss = float(np.sum(Eu))
            if not np.isfinite(ss) or ss <= 0:
                Ws = np.ones_like(Eu) / len(Eu)
            else:
                Ws = Eu / ss
            W = np.zeros((n, n))
            W[iu0_r, iu0_c] = (1.0 - lam) * Ws
            # top-K uniform shell term
            if lam > 1e-12:
                c = G[iu0_r, iu0_c]
                K = Ktop if len(c) >= Ktop else len(c)
                part = np.argpartition(-c, K - 1)[:K]
                Wk = np.zeros_like(Eu)
                Wk[part] = 1.0 / K
                W[iu0_r, iu0_c] += lam * Wk
            W = W + W.T
            GradX = W @ X
            dot = np.sum(GradX * X, axis=1, keepdims=True)
            GradY = (GradX - dot * X) / r
            m1 = b1 * m1 + (1 - b1) * GradY
            m2 = b2 * m2 + (1 - b2) * (GradY ** 2)
            bc1 = 1 - b1 ** (st + 1)
            bc2 = 1 - b2 ** (st + 1)
            upd = (m1 / bc1) / (np.sqrt(m2 / bc2) + eps)
            Y = Y - lr * upd
            if (st % 50 == 49) or st == steps - 1:
                Xt = _norm(Y)
                mt = _max_cos(Xt)
                if mt < local_best_m:
                    local_best_m = mt
                    local_best_X = Xt.copy()
        return local_best_X

    def refine_hole(v, Others, iters=80, tau=0.01, lr=0.05):
        v = v / (np.linalg.norm(v) + 1e-30)
        m1 = np.zeros_like(v)
        m2 = np.zeros_like(v)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for st in range(iters):
            dots = Others @ v
            mx = float(np.max(dots))
            e = np.exp(np.clip((dots - mx) / tau, -50, 0))
            s = float(np.sum(e))
            if not np.isfinite(s) or s <= 0:
                break
            w = e / s
            g = Others.T @ w
            g = g - float(np.dot(g, v)) * v
            m1 = b1 * m1 + (1 - b1) * g
            m2 = b2 * m2 + (1 - b2) * (g * g)
            upd = (m1 / (1 - b1 ** (st + 1))) / (np.sqrt(m2 / (1 - b2 ** (st + 1))) + eps)
            v = v - lr * upd
            nv = np.linalg.norm(v)
            if nv == 0 or not np.isfinite(nv):
                break
            v = v / nv
        return v

    def hole_repair(X, passes=8, Ncand=4000):
        X = _norm(X)
        cur_m = _max_cos(X)
        for _ps in range(passes):
            if time.time() > HARD - 6:
                break
            pm = _point_max(X)
            order = np.argsort(-pm)
            improved = False
            for idx in order[:6]:
                i = int(idx)
                Others = np.delete(X, i, axis=0)
                C = _norm(rng.standard_normal((Ncand, d)))
                dots = C @ Others.T
                md = np.max(dots, axis=1)
                bi = int(np.argmin(md))
                if md[bi] < pm[i] - 1e-9:
                    v = refine_hole(C[bi], Others)
                    Xt = X.copy()
                    Xt[i] = v / np.linalg.norm(v)
                    mt = _max_cos(Xt)
                    if mt < cur_m - 1e-12:
                        X = Xt
                        cur_m = mt
                        improved = True
                        break
            if not improved:
                break
        return X

    def pair_polish(X, iters=1500, step=2e-4):
        X = _norm(X)
        cur = _max_cos(X)
        st = step
        for _ in range(iters):
            if time.time() > HARD - 5:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G.max(axis=1)))
            # full max pair
            flat = int(np.argmax(G))
            i = flat // n
            j = flat % n
            # push apart along tangent
            gi = X[j] - float(X[i] @ X[j]) * X[i]
            gj = X[i] - float(X[i] @ X[j]) * X[j]
            ni = np.linalg.norm(gi); nj = np.linalg.norm(gj)
            if ni == 0 or nj == 0 or not np.isfinite(ni + nj):
                break
            Xt = X.copy()
            Xt[i] = Xt[i] - st * gi / ni
            Xt[j] = Xt[j] - st * gj / nj
            Xt = _norm(Xt)
            mt = _max_cos(Xt)
            if mt < cur:
                X = Xt
                cur = mt
                st = min(st * 1.05, 1e-3)
            else:
                st *= 0.7
                if st < 1e-8:
                    # random tangent kick to escape
                    st = step * 0.5
                    break
        return X

    # ---- multi-restart campaign ----
    # restart 0: quench incumbent directly (shell compression)
    try:
        if time.time() < HARD - 10:
            Xq = quench(X_best, steps=600, tau0=0.012, tau1=0.002, lr0=0.03, lam0=0.6)
            consider(Xq)
            Xq2 = hole_repair(Xq if _max_cos(Xq) < best_max + 1e-6 else X_best)
            consider(pair_polish(Xq2, iters=800))
    except Exception:
        pass

    noise_levels = [0.02, 0.06, 0.12, 0.20, 0.30, 0.45, 0.08, 0.15]
    for ri, nl in enumerate(noise_levels):
        if time.time() > HARD - 12:
            break
        try:
            pert = rng.standard_normal((n, d)) * nl
            pert = pert - (np.sum(pert * X_best, axis=1, keepdims=True)) * X_best
            X0 = _norm(X_best + pert)
            # extra scrambling for large noise: reinsert 2 worst points randomly
            if nl >= 0.2:
                pm = _point_max(X0)
                worst = np.argsort(-pm)[:2]
                for w in worst:
                    v = rng.standard_normal(d)
                    v = v / np.linalg.norm(v)
                    X0[int(w)] = v
                X0 = _norm(X0)
            steps = 350 if nl < 0.25 else 300
            Xq = quench(X0, steps=steps, tau0=0.010, tau1=0.0015, lr0=0.03, lam0=0.7)
            consider(Xq)
            # hole repair on the quenched point (even if not best, may lead to best)
            Xh = hole_repair(Xq, passes=6, Ncand=3000)
            consider(Xh)
            if _max_cos(Xh) < best_max + 2e-4:
                Xp = pair_polish(Xh, iters=400, step=2e-4)
                consider(Xp)
        except Exception:
            continue

    # final intensive polish of the best
    try:
        if time.time() < HARD - 8:
            Xf = quench(X_best, steps=250, tau0=0.004, tau1=0.0008, lr0=0.015, lam0=0.3)
            consider(Xf)
            Xf = pair_polish(X_best, iters=1200, step=1.5e-4)
            consider(Xf)
    except Exception:
        pass

    return _norm(X_best)
