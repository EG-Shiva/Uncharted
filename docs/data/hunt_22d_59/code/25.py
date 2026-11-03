import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _smooth_opt(X0, steps, beta0, beta1, lr0, lr1, mu):
    X = X0.copy()
    n = X.shape[0]
    vel = np.zeros_like(X)
    best_val = _max_cos(X)
    best_X = X.copy()
    if steps <= 1:
        return best_X
    lbb = np.linspace(np.log(beta0), np.log(beta1), steps)
    betas = np.exp(lbb)
    lll = np.linspace(np.log(lr0), np.log(lr1), steps)
    lrs = np.exp(lll)
    om = 1.0 - mu
    for s in range(steps):
        beta = float(betas[s])
        lr = float(lrs[s])
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        G -= m
        G *= beta
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(G.sum() * 0.5)
        if S < 1e-30 or not np.isfinite(S):
            continue
        Gr = (G @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gr -= X * dot
        vel *= mu
        vel += om * Gr
        X -= lr * vel
        rn = np.linalg.norm(X, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X /= rn
        if (s % 100 == 99) or (s == steps - 1):
            G2 = X @ X.T
            np.fill_diagonal(G2, -2.0)
            v = float(np.max(G2))
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X

def _hard_opt(X0, steps, lr0, lr1, mu, marg0, marg1):
    X = X0.copy()
    n = X.shape[0]
    vel = np.zeros_like(X)
    best_val = _max_cos(X)
    best_X = X.copy()
    if steps <= 1:
        return best_X
    lll = np.linspace(np.log(lr0), np.log(lr1), steps)
    lrs = np.exp(lll)
    llm = np.linspace(np.log(marg0), np.log(marg1), steps)
    margs = np.exp(llm)
    om = 1.0 - mu
    for s in range(steps):
        lr = float(lrs[s])
        marg = float(margs[s])
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = (G >= (m - marg)).astype(np.float64)
        np.fill_diagonal(W, 0.0)
        deg = W.sum(axis=1, keepdims=True)
        Gr = (W @ X) / np.maximum(deg, 1.0)
        zm = (deg[:, 0] == 0)
        if np.any(zm):
            Gr[zm, :] = 0.0
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gr -= X * dot
        vel *= mu
        vel += om * Gr
        X -= lr * vel
        rn = np.linalg.norm(X, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X /= rn
        if (s % 150 == 149) or (s == steps - 1):
            G2 = X @ X.T
            np.fill_diagonal(G2, -2.0)
            v = float(np.max(G2))
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X

def _refine_candidate(y0, Xf, steps, lr0, lr1, marg0, marg1):
    y = y0.copy()
    nrm = np.linalg.norm(y)
    if nrm < 1e-12:
        y = np.ones_like(y) / np.sqrt(y.size)
    else:
        y = y / nrm
    lrs = np.exp(np.linspace(np.log(lr0), np.log(lr1), steps))
    margs = np.exp(np.linspace(np.log(marg0), np.log(marg1), steps))
    for s in range(steps):
        d = Xf @ y
        m = float(np.max(d))
        marg = float(margs[s])
        lr = float(lrs[s])
        act = d >= (m - marg)
        if not np.any(act):
            act[np.argmax(d)] = True
        g = np.mean(Xf[act], axis=0)
        dot = float(np.dot(g, y))
        g = g - dot * y
        y = y - lr * g
        nrm = np.linalg.norm(y)
        if nrm < 1e-12:
            break
        y /= nrm
    d = Xf @ y
    return y, float(np.max(d))

def _find_hole(Xf, rng, n_sample=3000, refine=220):
    d = Xf.shape[1]
    C = rng.standard_normal((n_sample, d))
    C = _normed(C)
    D = C @ Xf.T
    rm = D.max(axis=1)
    idx = np.argsort(rm)[:3]
    best_y = C[idx[0]].copy()
    best_m = float(rm[idx[0]])
    for j in idx:
        y0 = C[j].copy()
        y1, m1 = _refine_candidate(y0, Xf, refine, 0.03, 0.001, 0.006, 0.0004)
        if m1 < best_m:
            best_m = m1
            best_y = y1.copy()
    return best_y

def _kill_fill(Xb, k, rng):
    n = Xb.shape[0]
    G = Xb @ Xb.T
    np.fill_diagonal(G, -2.0)
    rowmax = G.max(axis=1)
    worst = np.argsort(rowmax)[-k:]
    worst_set = set(int(w) for w in worst)
    Xw = Xb.copy()
    filled = []
    for slot in worst:
        slot = int(slot)
        fixed_idx = [i for i in range(n) if (i not in worst_set) or (i in filled)]
        # fixed_idx excludes not-yet-filled worst slots
        # build fixed matrix
        Xf = Xw[fixed_idx]
        y = _find_hole(Xf, rng, n_sample=2500, refine=200)
        Xw[slot] = y
        filled.append(slot)
        worst_set.discard(slot)
        # add worst_set update: remaining worst still excluded, filled included
        # need to keep track: for next iter, excluded = worst_set (remaining)
        # filled list already included via filled check above
        # Trick: rebuild worst_set logic: use original worst list
        # Simpler: maintain filled set
        worst_set = set(int(w) for w in worst) - set(filled)
    return _normed(Xw)

def solve():
    t0 = time.time()
    deadline = t0 + 165.0
    rng = np.random.default_rng(0)
    Xb = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (59, 22):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _normed(a)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                Xb = a.copy()
        except Exception:
            continue
    if Xb is None:
        Xb = _normed(rng.standard_normal((59, 22)))
        best_val = _max_cos(Xb)
    x_best = Xb.copy()

    def remain():
        return deadline - time.time()

    # Stage 0: light polish of incumbent
    try:
        if remain() > 30:
            y = _smooth_opt(x_best, 2000, 60.0, 600.0, 0.015, 0.001, 0.88)
            v = _max_cos(y)
            if v < best_val:
                best_val = v
                x_best = y.copy()
            y2 = _hard_opt(y, 800, 0.004, 0.0004, 0.85, 0.004, 0.0005)
            v2 = _max_cos(y2)
            if v2 < best_val:
                best_val = v2
                x_best = y2.copy()
            elif v < best_val:
                best_val = v
                x_best = y.copy()
    except Exception:
        pass

    sigmas = [0.02, 0.05, 0.10, 0.015, 0.03, 0.07, 0.12, 0.008]
    trial = 0
    while remain() > 22.0:
        trial += 1
        mode = trial % 10
        try:
            if mode <= 2:
                sigma = sigmas[trial % len(sigmas)]
                Xt = x_best + sigma * rng.standard_normal((59, 22))
                Xt = _normed(Xt)
                s_steps, h_steps = 1300, 700
            elif mode == 3:
                Xt = _kill_fill(x_best, 3, rng)
                Xt = Xt + 0.006 * rng.standard_normal((59, 22))
                Xt = _normed(Xt)
                s_steps, h_steps = 1700, 900
            elif mode == 4:
                Xt = _kill_fill(x_best, 5, rng)
                Xt = Xt + 0.006 * rng.standard_normal((59, 22))
                Xt = _normed(Xt)
                s_steps, h_steps = 1800, 900
            elif mode == 5:
                k = 6
                c = int(rng.integers(0, 59))
                dots = x_best @ x_best[c]
                blk = np.argsort(-dots)[:k]
                Xt = x_best.copy()
                noise = rng.standard_normal((59, 22))
                mask = np.zeros(59)
                mask[blk] = 1.0
                Xt = Xt + (0.008 + 0.24 * mask[:, None]) * noise
                Xt = _normed(Xt)
                s_steps, h_steps = 1500, 800
            elif mode == 6:
                Xt = _kill_fill(x_best, 1, rng)
                Xt = Xt + 0.004 * rng.standard_normal((59, 22))
                Xt = _normed(Xt)
                s_steps, h_steps = 1500, 800
            elif mode == 7:
                Xt = _kill_fill(x_best, 8, rng)
                Xt = Xt + 0.008 * rng.standard_normal((59, 22))
                Xt = _normed(Xt)
                s_steps, h_steps = 2000, 900
            elif mode == 8:
                k = 4
                c = int(rng.integers(0, 59))
                dots = x_best @ x_best[c]
                blk = np.argsort(-dots)[:k]
                Xt = x_best.copy()
                noise = rng.standard_normal((59, 22))
                mask = np.zeros(59)
                mask[blk] = 1.0
                Xt = Xt + (0.006 + 0.18 * mask[:, None]) * noise
                Xt = _normed(Xt)
                s_steps, h_steps = 1400, 700
            else:
                sigma = 0.16 if (trial % 20 == 9) else 0.025
                Xt = x_best + sigma * rng.standard_normal((59, 22))
                Xt = _normed(Xt)
                s_steps, h_steps = 1300, 700
            # adapt steps to remaining time
            if remain() < 35:
                s_steps = min(s_steps, 900)
                h_steps = min(h_steps, 500)
            y = _smooth_opt(Xt, s_steps, 60.0, 650.0, 0.014, 0.001, 0.88)
            y2 = _hard_opt(y, h_steps, 0.004, 0.0004, 0.85, 0.004, 0.0005)
            v2 = _max_cos(y2)
            v = _max_cos(y)
            cand_v = v2 if v2 < v else v
            cand = y2 if v2 < v else y
            if cand_v < best_val - 1e-12:
                best_val = cand_v
                x_best = cand.copy()
            if remain() < 0:
                break
        except Exception:
            continue

    # Final polish
    try:
        r = remain()
        if r > 8:
            ss = 3500 if r > 20 else 1500
            hs = 1500 if r > 20 else 700
            y = _smooth_opt(x_best, ss, 150.0, 1500.0, 0.006, 0.0003, 0.9)
            v = _max_cos(y)
            if v < best_val:
                best_val = v
                x_best = y.copy()
            if remain() > 5:
                y2 = _hard_opt(x_best, hs, 0.003, 0.0002, 0.85, 0.002, 0.0002)
                v2 = _max_cos(y2)
                if v2 < best_val:
                    best_val = v2
                    x_best = y2.copy()
    except Exception:
        pass

    x_best = _normed(x_best)
    return x_best
