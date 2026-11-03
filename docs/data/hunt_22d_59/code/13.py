import time
import numpy as np

def _max_cos(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _smooth_opt(X, steps, beta0, beta1, lr0, lr1, mu):
    X = X.copy()
    vel = np.zeros_like(X)
    best_val = _max_cos(X)
    best_X = X.copy()
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if S < 1e-30:
            continue
        Grad = (W @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        vel = mu * vel + (1.0 - mu) * Gt
        X = X - lr * vel
        X = _normed(X)
        if (s % 50 == 49) or (s == steps - 1):
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X, best_val

def _contact_opt(X, steps, lr0, lr1, mu, marg0, marg1):
    X = X.copy()
    vel = np.zeros_like(X)
    best_val = _max_cos(X)
    best_X = X.copy()
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    lm0 = np.log(marg0); lm1 = np.log(marg1)
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        marg = float(np.exp(lm0 + (lm1 - lm0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = (G >= (m - marg)).astype(np.float64)
        deg = W.sum(axis=1, keepdims=True)
        Grad = (W @ X) / np.maximum(deg, 1.0)
        zm = (deg[:, 0] == 0)
        if np.any(zm):
            Grad[zm, :] = 0.0
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        vel = mu * vel + (1.0 - mu) * Gt
        X = X - lr * vel
        X = _normed(X)
        if (s % 100 == 99) or (s == steps - 1):
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X, best_val

def _hole_for_others(O, rng, n_starts=224, steps=320):
    S = n_starts
    d = O.shape[1]
    V = rng.standard_normal((S, d))
    V = _normed(V)
    vel = np.zeros_like(V)
    mu = 0.85
    best_val = np.inf
    best_v = V[0].copy()
    for s in range(steps):
        frac = s / float(max(1, steps - 1))
        beta = 20.0 * ((300.0 / 20.0) ** frac)
        lr = 0.08 * ((0.005 / 0.08) ** frac)
        G = V @ O.T
        mrow = G.max(axis=1, keepdims=True)
        W = np.exp(beta * (G - mrow))
        Ssum = W.sum(axis=1, keepdims=True) + 1e-30
        Grad = (W @ O) / Ssum
        dot = np.sum(Grad * V, axis=1, keepdims=True)
        Gt = Grad - dot * V
        vel = mu * vel + (1.0 - mu) * Gt
        V = V - lr * vel
        V = _normed(V)
        if s % 80 == 79 or s == steps - 1:
            G2 = V @ O.T
            rm = G2.max(axis=1)
            i = int(np.argmin(rm))
            if float(rm[i]) < best_val:
                best_val = float(rm[i])
                best_v = V[i].copy()
    return best_v, best_val

def _lp_polish(X0, iters=12, delta0=0.006, marg=0.012):
    try:
        from scipy.optimize import linprog
    except Exception:
        return X0.copy(), _max_cos(X0)
    X = X0.copy()
    n, d = X.shape
    Nvars = n * d + 1
    best_val = _max_cos(X)
    best_X = X.copy()
    delta = float(delta0)
    for _ in range(iters):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        thr = m - marg
        iu_all, ju_all = np.where(np.triu(G >= thr, 1))
        if iu_all.size == 0:
            break
        if iu_all.size > 1000:
            vals = G[iu_all, ju_all]
            idx = np.argsort(-vals)[:1000]
            iu = iu_all[idx]; ju = ju_all[idx]
        else:
            iu = iu_all; ju = ju_all
        nA = iu.size
        A_ub = np.zeros((nA, Nvars), dtype=np.float64)
        b_ub = np.zeros(nA, dtype=np.float64)
        for k in range(nA):
            i = int(iu[k]); j = int(ju[k])
            A_ub[k, i * d:(i + 1) * d] = X[j]
            A_ub[k, j * d:(j + 1) * d] = X[i]
            A_ub[k, Nvars - 1] = -1.0
            b_ub[k] = -float(G[i, j])
        A_eq = np.zeros((n, Nvars), dtype=np.float64)
        for i in range(n):
            A_eq[i, i * d:(i + 1) * d] = X[i]
        b_eq = np.zeros(n, dtype=np.float64)
        c = np.zeros(Nvars, dtype=np.float64)
        c[-1] = 1.0
        bounds = [(-delta, delta)] * (n * d) + [(-1.0, 2.0)]
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                          bounds=bounds, method='highs')
        except Exception:
            break
        if not getattr(res, 'success', False):
            delta *= 0.5
            if delta < 1e-8:
                break
            continue
        dvec = np.asarray(res.x[:n * d]).reshape(n, d)
        Xn = _normed(X + dvec)
        m2 = _max_cos(Xn)
        if m2 < m - 1e-12:
            X = Xn
            if m2 < best_val:
                best_val = m2
                best_X = Xn.copy()
            delta = min(delta * 1.4, 0.02)
        else:
            delta *= 0.5
            if delta < 1e-9:
                break
    return best_X, best_val

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    deadline = t0 + 168.0

    def tl(need=5.0):
        return (time.time() - t0) < (168.0 - need)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (59, 22):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _normed(a)
            cands.append(a)
        except Exception:
            continue
    if len(cands) == 0:
        cur = _normed(rng.standard_normal((59, 22)))
    else:
        vals = [_max_cos(c) for c in cands]
        cur = cands[int(np.argmin(vals))].copy()
    cur_val = _max_cos(cur)
    best_X = cur.copy()
    best_val = cur_val

    # Stage 0: polish start basin
    if tl(140):
        Y, v = _smooth_opt(best_X, steps=3500, beta0=30.0, beta1=320.0,
                           lr0=0.02, lr1=0.001, mu=0.9)
        if v < best_val:
            best_val = v; best_X = Y.copy()
        Y, v = _lp_polish(best_X, iters=10, delta0=0.006, marg=0.012)
        if v < best_val:
            best_val = v; best_X = Y.copy()
        Y, v = _contact_opt(best_X, steps=1200, lr0=0.005, lr1=0.0004,
                            mu=0.85, marg0=0.005, marg1=0.0008)
        if v < best_val:
            best_val = v; best_X = Y.copy()
        cur = best_X.copy(); cur_val = best_val

    sigmas = [0.008, 0.015, 0.025, 0.04, 0.07, 0.12]
    trial = 0
    TEMP = 0.0005
    while time.time() < deadline - 38.0:
        trial += 1
        r = float(rng.random())
        Xt = None
        if r < 0.45:
            sigma = sigmas[trial % len(sigmas)]
            # occasionally shake best instead of cur
            base = cur if (trial % 3 != 0) else best_X
            Xt = _normed(base + sigma * rng.standard_normal((59, 22)))
            Y, v = _smooth_opt(Xt, steps=2000, beta0=40.0, beta1=420.0,
                               lr0=0.015, lr1=0.001, mu=0.88)
        elif r < 0.75:
            k = int(rng.integers(2, 9))
            idx = rng.choice(59, size=k, replace=False)
            base = cur.copy()
            noise = rng.standard_normal((59, 22))
            base[idx] += 0.16 * noise[idx]
            rest = np.ones(59, dtype=bool); rest[idx] = False
            base[rest] += 0.005 * noise[rest]
            Xt = _normed(base)
            Y, v = _smooth_opt(Xt, steps=2200, beta0=50.0, beta1=500.0,
                               lr0=0.014, lr1=0.0008, mu=0.88)
        else:
            # hole reinsertion: victim = overloaded point of cur
            try:
                Gc = cur @ cur.T
                np.fill_diagonal(Gc, -2.0)
                mc = float(Gc.max())
                deg = (Gc >= (mc - 0.004)).sum(axis=1)
                order = np.argsort(-deg)
                top = order[:5]
                victim = int(top[int(rng.integers(0, len(top)))])
                O = np.delete(cur, victim, axis=0)
                hv, _ = _hole_for_others(O, rng, n_starts=224, steps=300)
                base = cur.copy()
                base[victim] = hv + 0.004 * rng.standard_normal(22)
                Xt = _normed(base)
                Y, v = _smooth_opt(Xt, steps=2200, beta0=50.0, beta1=500.0,
                                   lr0=0.012, lr1=0.0008, mu=0.88)
            except Exception:
                Xt = _normed(cur + 0.02 * rng.standard_normal((59, 22)))
                Y, v = _smooth_opt(Xt, steps=1800, beta0=40.0, beta1=400.0,
                                   lr0=0.014, lr1=0.001, mu=0.88)
        # light hard polish
        Y2, v2 = _contact_opt(Y, steps=600, lr0=0.004, lr1=0.0004,
                              mu=0.85, marg0=0.005, marg1=0.0008)
        if v2 < best_val - 1e-12:
            best_val = v2; best_X = Y2.copy()
            cur = Y2.copy(); cur_val = v2
        elif v < best_val - 1e-12:
            best_val = v; best_X = Y.copy()
            cur = Y.copy(); cur_val = v
        else:
            # Metropolis walk on cur
            cand_val = v2 if v2 < v else v
            cand_X = Y2 if v2 < v else Y
            if cand_val < cur_val:
                cur = cand_X.copy(); cur_val = cand_val
            else:
                try:
                    p = float(np.exp((cur_val - cand_val) / TEMP))
                except Exception:
                    p = 0.0
                if float(rng.random()) < p:
                    cur = cand_X.copy(); cur_val = cand_val
                # periodic restart to best
                if trial % 18 == 0:
                    cur = best_X.copy(); cur_val = best_val
        if trial > 200:
            break

    # Final intensive polish of best
    if time.time() < deadline - 30.0:
        Y, v = _smooth_opt(best_X, steps=6000, beta0=80.0, beta1=800.0,
                           lr0=0.008, lr1=0.0005, mu=0.9)
        if v < best_val:
            best_val = v; best_X = Y.copy()
    if time.time() < deadline - 20.0:
        Y, v = _lp_polish(best_X, iters=22, delta0=0.005, marg=0.010)
        if v < best_val:
            best_val = v; best_X = Y.copy()
    if time.time() < deadline - 10.0:
        Y, v = _contact_opt(best_X, steps=2200, lr0=0.003, lr1=0.00015,
                            mu=0.85, marg0=0.003, marg1=0.0004)
        if v < best_val:
            best_val = v; best_X = Y.copy()
    if time.time() < deadline - 5.0:
        Y, v = _lp_polish(best_X, iters=10, delta0=0.0015, marg=0.006)
        if v < best_val:
            best_val = v; best_X = Y.copy()

    best_X = _normed(best_X)
    return best_X
