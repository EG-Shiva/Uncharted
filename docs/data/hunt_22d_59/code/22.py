import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _smooth(X0, steps, beta0, beta1, lr0, lr1, mu=0.9):
    X = X0.copy()
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
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        vel = mu * vel + (1.0 - mu) * Gt
        X = X - lr * vel
        X = _normed(X)
        if (s % 200 == 199) or (s == steps - 1):
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X, best_val

def _hard(X0, steps, lr0, lr1, marg0, marg1, mu=0.85):
    X = X0.copy()
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
        W = (G >= (m - marg)).astype(float)
        deg = W.sum(axis=1, keepdims=True)
        Grad = (W @ X) / np.maximum(deg, 1.0)
        zm = (deg[:, 0] == 0)
        if np.any(zm):
            Grad[zm, :] = 0.0
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        vel = mu * vel + (1.0 - mu) * Gt
        X = X - lr * vel
        X = _normed(X)
        if (s % 200 == 199) or (s == steps - 1):
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X, best_val

def _polish(X, t_end, big=True):
    # short combined polish; checks time via t_end callable
    if big:
        X, v = _smooth(X, steps=1800, beta0=30.0, beta1=300.0, lr0=0.012, lr1=0.0008, mu=0.9)
        X, v = _hard(X, steps=800, lr0=0.003, lr1=0.0002, mu=0.85, marg0=0.004, marg1=0.0005)
    else:
        X, v = _smooth(X, steps=1200, beta0=30.0, beta1=250.0, lr0=0.012, lr1=0.0008, mu=0.9)
        X, v = _hard(X, steps=500, lr0=0.003, lr1=0.0002, mu=0.85, marg0=0.004, marg1=0.0005)
    return X, v

def _contacts(G, m, tol=0.002):
    A = (G >= (m - tol)).astype(int)
    np.fill_diagonal(A, 0)
    return A.sum(axis=1)

def _largest_cap(X, rng, n_rand=20000, topk=4, refine=300):
    # batched random sampling
    R = rng.standard_normal((n_rand, X.shape[1]))
    R = _normed(R)
    S = R @ X.T
    rm = S.max(axis=1)
    idx = np.argpartition(rm, topk)[:topk]
    best_c = R[idx[0]].copy()
    best_v = float(rm[idx[0]])
    # local maximization of min angle = minimize max dot
    for j in idx:
        c = R[j].copy()
        cv = float(rm[j])
        bc = c.copy(); bv = cv
        lr = 0.02
        for s in range(refine):
            d = X @ c
            m = float(d.max())
            # active set
            act = d >= (m - 0.01)
            if np.any(act):
                g = X[act].mean(axis=0)
            else:
                g = X[np.argmax(d)]
            # tangent component, move away
            gt = g - float(g @ c) * c
            c = c - lr * gt
            c = c / max(np.linalg.norm(c), 1e-12)
            lr *= (1.0 - 0.002)
            if (s % 25 == 24) or (s == refine - 1):
                m2 = float((X @ c).max())
                if m2 < bv:
                    bv = m2; bc = c.copy()
        if bv < best_v:
            best_v = bv; best_c = bc.copy()
    return best_c, best_v

def solve():
    t0 = time.time()
    def tl(limit=150.0):
        return (time.time() - t0) < limit
    rng = np.random.default_rng(0)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a = _normed(a)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        best_X = _normed(rng.standard_normal((59, 22)))
    else:
        vals = [_max_cos(c) for c in cands]
        best_X = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(best_X)

    # Stage 0: light direct polish of incumbent (keep fallback)
    try:
        Xp, vp = _polish(best_X, tl, big=True)
        if vp < best_val:
            best_val = vp; best_X = Xp.copy()
    except Exception:
        pass

    # ---- Direction 1: 58-point deletion + cap insertion ----
    # try a few deletion choices
    try:
        G0 = best_X @ best_X.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        cnt0 = _contacts(G0, m0, tol=0.002)
        rowmax0 = G0.max(axis=1)
        order = np.argsort(-(cnt0 * 10.0 + rowmax0))  # highest load first
    except Exception:
        order = np.arange(59)
    n58_tries = 3
    for ti in range(n58_tries):
        if not tl(120.0):
            break
        try:
            del_idx = int(order[ti % len(order)])
            X58 = np.delete(best_X, del_idx, axis=0)  # 58x22
            # small shake to unlock
            X58 = _normed(X58 + 0.004 * rng.standard_normal(X58.shape))
            X58p, v58 = _polish(X58, tl, big=False)
            # cap search against polished 58
            cap, _ = _largest_cap(X58p, rng, n_rand=16000, topk=4, refine=250)
            X59 = np.vstack([X58p, cap.reshape(1, -1)])
            X59 = _normed(X59 + 0.002 * rng.standard_normal(X59.shape))
            X59p, v59 = _polish(X59, tl, big=True)
            if v59 < best_val:
                best_val = v59; best_X = X59p.copy()
            # second insertion variant: different cap seed with more noise
            if not tl(120.0):
                break
            cap2, _ = _largest_cap(X58p, rng, n_rand=8000, topk=3, refine=150)
            X59b = np.vstack([X58p, cap2.reshape(1, -1)])
            X59b = _normed(X59b + 0.006 * rng.standard_normal(X59b.shape))
            X59b, v59b = _polish(X59b, tl, big=False)
            if v59b < best_val:
                best_val = v59b; best_X = X59b.copy()
        except Exception:
            continue

    # ---- Direction 2: overcomplete 60/61 -> greedy deletion ----
    for extra in (1, 2):
        if not tl(135.0):
            break
        try:
            n0 = 59 + extra
            # perturbed copy + random extras
            base = best_X + 0.01 * rng.standard_normal(best_X.shape)
            add = rng.standard_normal((extra, 22))
            Y = np.vstack([base, add])
            Y = _normed(Y)
            Yp, _ = _polish(Y, tl, big=False)
            # iteratively delete worst point
            while Yp.shape[0] > 59:
                if not tl(140.0):
                    break
                G = Yp @ Yp.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                rowmax = G.max(axis=1)
                cnt = _contacts(G, m, tol=0.002)
                # load score
                score = rowmax + 0.002 * (cnt / max(1, cnt.max()))
                kill = int(np.argmax(score))
                Yp = np.delete(Yp, kill, axis=0)
                Yp = _normed(Yp + 0.002 * rng.standard_normal(Yp.shape))
                Yp, _ = _polish(Yp, tl, big=False)
            v = _max_cos(Yp)
            # final joint re-polish
            Yp, v = _polish(Yp, tl, big=True)
            if v < best_val:
                best_val = v; best_X = Yp.copy()
        except Exception:
            continue

    # final fine hard polish if time remains
    if tl(148.0):
        try:
            Xf, vf = _hard(best_X, steps=1200, lr0=0.001, lr1=0.00006, mu=0.8, marg0=0.0015, marg1=0.0003)
            if vf < best_val:
                best_val = vf; best_X = Xf.copy()
        except Exception:
            pass

    best_X = _normed(best_X)
    return best_X
