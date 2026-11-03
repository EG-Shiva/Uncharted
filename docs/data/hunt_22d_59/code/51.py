import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    BUDGET = 168.0
    np.random.seed(0)
    rng = np.random.RandomState(0)

    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.randn(59, 22)))
    vals = [_max_cos(c) for c in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(np.min(vals))

    n_, d_ = 59, 22

    def adam_polish(X, steps, b0, b1, lr0, lr1):
        X = X.copy()
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        b_1 = 0.9; b_2 = 0.999; eps = 1e-8
        la0 = np.log(b0); la1 = np.log(b1)
        ll0 = np.log(lr0); ll1 = np.log(lr1)
        cur_best = X.copy()
        cur_val = _max_cos(X)
        # fast loop
        for t in range(1, steps + 1):
            if (t & 127) == 0 and (time.time() - t0) > BUDGET:
                break
            frac = t / steps
            alpha = float(np.exp(la0 + (la1 - la0) * frac))
            lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            Mx = float(np.max(G))
            G -= Mx
            G *= alpha
            np.exp(G, out=G)
            np.fill_diagonal(G, 0.0)
            S = float(np.sum(G))
            if not np.isfinite(S) or S <= 0:
                continue
            Grad = (G @ X) / S
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Grad -= dot * X
            m *= b_1; m += 0.1 * Grad
            v *= b_2; v += 0.001 * (Grad * Grad)
            c1 = 1.0 - (b_1 ** t); c2 = 1.0 - (b_2 ** t)
            mc = m / c1; vc = v / c2
            X -= lr * mc / (np.sqrt(vc) + eps)
            nn = np.linalg.norm(X, axis=1, keepdims=True)
            nn[nn == 0] = 1.0
            X /= nn
            if (t % 500 == 0) or (t == steps):
                cv = _max_cos(X)
                if cv < cur_val:
                    cur_val = cv
                    cur_best = X.copy()
        fv = _max_cos(X)
        if fv < cur_val:
            cur_val = fv; cur_best = X.copy()
        return cur_best, cur_val

    def contacts_count(X, tol=2e-4):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = float(np.max(G))
        A = G > (mx - tol)
        np.fill_diagonal(A, False)
        return np.sum(A, axis=1), mx, G

    # ---- deletion route: top-10 high-contact points ----
    try:
        cnt, mx0, _ = contacts_count(best, tol=1e-6)
    except Exception:
        cnt = np.zeros(59); mx0 = best_val
    order = np.argsort(-cnt)
    top10 = order[:10].tolist()

    best58 = None; best58_val = 1e9
    for idx in top10:
        if (time.time() - t0) > BUDGET - 120:
            break
        Y = np.delete(best, idx, axis=0)  # 58x22
        Yp, Yv = adam_polish(Y, steps=1500, b0=50.0, b1=500.0, lr0=0.005, lr1=0.0005)
        if Yv < best58_val:
            best58_val = Yv; best58 = Yp.copy()
    # fallback if none
    if best58 is None:
        Y = np.delete(best, int(order[0]), axis=0)
        best58 = Y; best58_val = _max_cos(Y)

    # thorough polish of easier 58-point problem
    if (time.time() - t0) < BUDGET - 110:
        best58, best58_val = adam_polish(best58, steps=6000, b0=40.0, b1=800.0, lr0=0.006, lr1=0.0003)

    def largest_hole(X, ns=2500):
        H = rng.randn(ns, X.shape[1])
        H = _norm(H)
        D = H @ X.T
        rm = np.max(D, axis=1)
        i = int(np.argmin(rm))
        return H[i].copy(), float(rm[i])

    # ---- restore 59 by splitting ----
    # candidate ks: hole-nearest + high contacts + random
    hole_dir, _ = largest_hole(best58)
    dots = best58 @ hole_dir
    hole_nearest = int(np.argmin(dots))
    cnt58, _, _ = contacts_count(best58, tol=1e-6)
    ord58 = np.argsort(-cnt58).tolist()
    cand_ks = [hole_nearest] + ord58[:4]
    # add random ks deterministically
    cand_ks += [int(rng.randint(58)) for _ in range(4)]
    # dedup preserve order, then extend to 12
    seen = []
    for k in cand_ks:
        if k not in seen:
            seen.append(k)
    while len(seen) < 12:
        k = int(rng.randint(58))
        if k not in seen:
            seen.append(k)
    cand_ks = seen[:12]
    eps_list = [0.03, 0.06, 0.10, 0.15, 0.20, 0.08, 0.12, 0.05, 0.18, 0.04, 0.09, 0.14]

    overall_best = best.copy(); overall_val = best_val

    for vi in range(len(cand_ks)):
        if (time.time() - t0) > BUDGET - 75:
            break
        k = int(cand_ks[vi]) % 58
        eps = float(eps_list[vi % len(eps_list)])
        xk = best58[k].copy()
        # tangent direction: mix of hole-directed and random
        if vi % 2 == 0:
            t = hole_dir - float(hole_dir @ xk) * xk
            nn = float(np.linalg.norm(t))
            if nn < 1e-8:
                r = rng.randn(22); t = r - float(r @ xk) * xk; nn = float(np.linalg.norm(t))
            t /= nn
        else:
            r = rng.randn(22)
            t = r - float(r @ xk) * xk
            nn = float(np.linalg.norm(t))
            if nn < 1e-12:
                continue
            t /= nn
        xp = _norm((xk + eps * t).reshape(1, -1))[0]
        xm = _norm((xk - eps * t).reshape(1, -1))[0]
        Z = np.empty((59, 22), dtype=np.float64)
        Z[:58] = best58
        Z[k] = xp
        Z[58] = xm
        Z = _norm(Z + 0.002 * rng.randn(59, 22))
        # staged LSE: low beta to move split pair apart, then high beta
        Z1, v1 = adam_polish(Z, steps=1800, b0=25.0, b1=200.0, lr0=0.007, lr1=0.001)
        if (time.time() - t0) > BUDGET - 70:
            if v1 < overall_val:
                overall_best = Z1; overall_val = v1
            continue
        Z2, v2 = adam_polish(Z1, steps=2800, b0=150.0, b1=900.0, lr0=0.004, lr1=0.0003)
        if v2 < overall_val - 1e-12:
            overall_val = v2; overall_best = Z2.copy()
        elif v1 < overall_val - 1e-12:
            overall_val = v1; overall_best = Z1.copy()

    # ---- dimensional route: lift to R23, optimise, project ----
    for trial in range(2):
        if (time.time() - t0) > BUDGET - 45:
            break
        sig = 0.06 if trial == 0 else 0.12
        base = overall_best if overall_val < best_val else best
        Aug = np.hstack([base * 1.0, sig * rng.randn(59, 1)])
        Aug = _norm(Aug)  # 59x23 on S22
        # polish in 23D
        W, _ = adam_polish(Aug, steps=3500, b0=50.0, b1=600.0, lr0=0.005, lr1=0.0004)
        # penalise last coordinate gradually: scale + repolish short
        for shrink in [0.5, 0.25, 0.0]:
            if (time.time() - t0) > BUDGET - 40:
                break
            W[:, 22] *= shrink
            nn = np.linalg.norm(W, axis=1, keepdims=True); nn[nn == 0] = 1.0; W /= nn
            W, _ = adam_polish(W, steps=600, b0=80.0, b1=600.0, lr0=0.003, lr1=0.0005)
        # project back
        P = _norm(W[:, :22].copy())
        P, Pv = adam_polish(P, steps=3000, b0=60.0, b1=1000.0, lr0=0.004, lr1=0.0003)
        if Pv < overall_val - 1e-12:
            overall_val = Pv; overall_best = P.copy()

    # ---- final greedy basin-hopping on 59 with remaining time ----
    sigmas = [0.004, 0.010, 0.025, 0.05]
    it = 0
    while (time.time() - t0) < BUDGET - 8:
        it += 1
        remain = BUDGET - (time.time() - t0)
        if remain < 6:
            break
        steps = 2500 if remain > 40 else 1500
        sigma = sigmas[it % len(sigmas)]
        Y = overall_best + sigma * rng.randn(59, 22)
        if it % 6 == 0:
            Y[int(rng.randint(59))] = rng.randn(22)
        Y = _norm(Y)
        Yp, Yv = adam_polish(Y, steps=steps, b0=80.0, b1=900.0, lr0=0.004, lr1=0.0003)
        if Yv < overall_val - 1e-12:
            overall_val = Yv; overall_best = Yp.copy()

    # final squeeze
    if (time.time() - t0) < BUDGET - 2:
        try:
            overall_best, overall_val = adam_polish(overall_best, steps=2000, b0=600.0, b1=2000.0, lr0=0.0015, lr1=0.00015)
        except Exception:
            pass
    overall_best = _norm(overall_best)
    return overall_best
