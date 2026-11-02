import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, t_end=1e30):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        Cfull = X @ X.T
        c = Cfull[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            X, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 175.0
    SEARCH_END = t0 + 138.0
    rng = np.random.RandomState(20261102)
    n, d = 73, 13
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.randn(n, d)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        m0 = min(xb.shape[0], n); m1 = min(xb.shape[1], d)
        tmp[:m0, :m1] = xb[:m0, :m1]
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    Xbest = xb.copy()
    Xcur = xb.copy()
    cur = best

    I0 = np.triu_indices(n, 1)[0]
    J0 = np.triu_indices(n, 1)[1]

    def refine(X0, steps, s0, s1, lr0, lr1):
        X = X0.copy()
        m1 = np.zeros_like(X)
        v1 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1p, b2p = b1, b2
        ls0, ls1 = np.log(s0), np.log(s1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        for k in range(steps):
            frac = k / (steps - 1) if steps > 1 else 1.0
            s = np.exp(ls0 + (ls1 - ls0) * frac)
            lr = np.exp(ll0 + (ll1 - ll0) * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            cmax = G.max()
            E = np.exp(s * (G - cmax))
            np.fill_diagonal(E, 0.0)
            sumE = E.sum() * 0.5
            if sumE == 0 or not np.isfinite(sumE):
                Mask = (G >= cmax - 1e-12)
                cnt = Mask.sum() * 0.5
                if cnt <= 0:
                    continue
                W = Mask.astype(float) / float(cnt)
                np.fill_diagonal(W, 0.0)
            else:
                W = E / sumE
            Gt = W @ X
            dot = (Gt * X).sum(axis=1, keepdims=True)
            gt = Gt - dot * X
            m1 *= b1
            m1 += (1 - b1) * gt
            v1 *= b2
            v1 += (1 - b2) * (gt * gt)
            mhat = m1 / (1 - b1p)
            vhat = v1 / (1 - b2p)
            b1p *= b1
            b2p *= b2
            step = mhat / (np.sqrt(vhat) + eps)
            X -= lr * step
            nr = np.sqrt((X * X).sum(axis=1, keepdims=True))
            nr[nr == 0] = 1.0
            X /= nr
            if time.time() > DEADLINE - 2:
                break
        return X

    def greedy_reinsert(Y, idxs, ntry):
        Z = Y.copy()
        for idx in idxs:
            C = rng.randn(ntry, d)
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            others = np.delete(Z, idx, axis=0)
            dots = C @ others.T
            md = dots.max(axis=1)
            bi = int(np.argmin(md))
            Z[idx] = C[bi]
        return Z

    def worst_indices(X, k):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max(axis=1)
        return np.argsort(-mx)[:k]

    def contact_counts(X, tol=1e-3):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = G.max()
        return (G >= cm - tol).sum(axis=1), cm, G

    trial = 0
    Temp0, Temp1 = 0.002, 0.00015
    while time.time() < SEARCH_END and time.time() < DEADLINE - 8:
        trial += 1
        frac_global = (time.time() - t0) / (SEARCH_END - t0)
        frac_global = min(max(frac_global, 0.0), 1.0)
        Temp = Temp0 * (Temp1 / Temp0) ** frac_global
        mode = int(rng.randint(10))
        # base for perturbation: mostly Xbest, sometimes Xcur
        use_cur = rng.rand() < 0.25
        base = Xcur if use_cur else Xbest
        base_val = cur if use_cur else best

        Xp = None
        sigma_class = 1  # 0 small,1 med,2 large
        if mode == 0:  # small noise
            sigma = 0.008 + 0.008 * rng.rand()
            Xp = base + sigma * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            sigma_class = 0
        elif mode == 1:  # medium noise
            sigma = 0.025 + 0.02 * rng.rand()
            Xp = base + sigma * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            sigma_class = 1
        elif mode == 2:  # larger shake
            sigma = 0.06 + 0.05 * rng.rand()
            Xp = base + sigma * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            sigma_class = 2
        elif mode == 3:  # greedy reinsert worst 1-2
            k = 1 if rng.rand() < 0.5 else 2
            victims = worst_indices(base, k).tolist()
            # add a rattler if exists
            cnt, cm, G = contact_counts(base, tol=1e-9)
            ratt = np.where(cnt == 0)[0]
            if len(ratt) > 0 and rng.rand() < 0.7:
                v = int(ratt[int(rng.randint(len(ratt)))])
                if v not in victims:
                    victims.append(v)
            Xp = greedy_reinsert(base, victims, ntry=4000)
            Xp = Xp + 0.008 * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            sigma_class = 0
        elif mode == 4:  # reinit 2-3 random points
            Xp = base.copy()
            kk = 2 if rng.rand() < 0.7 else 3
            for _ in range(kk):
                ii = int(rng.randint(n))
                vv = rng.randn(d); vv /= np.linalg.norm(vv)
                Xp[ii] = vv
            Xp = Xp + 0.005 * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            sigma_class = 2
        elif mode == 5:  # targeted max-pair kick
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            # find top pair
            ij = np.argmax(G)
            a = ij // n; b = ij % n
            Xp = base + 0.02 * rng.randn(n, d)
            Xp[a] += 0.10 * rng.randn(d)
            Xp[b] += 0.10 * rng.randn(d)
            # also kick neighbours of a,b
            neigh = np.argsort(-G[a])[:3]
            for vv in neigh:
                Xp[int(vv)] += 0.04 * rng.randn(d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            sigma_class = 1
        elif mode == 6:  # cluster shake around random center
            ci = int(rng.randint(n))
            dots = base @ base[ci]
            order = np.argsort(-dots)[:10]
            Xp = base.copy()
            s = 0.05 + 0.03 * rng.rand()
            for vv in order:
                Xp[int(vv)] += s * rng.randn(d)
            Xp += 0.005 * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            sigma_class = 1
        elif mode == 7:  # heavy global
            sigma = 0.12 + 0.06 * rng.rand()
            Xp = base + sigma * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            sigma_class = 2
        elif mode == 8:  # single rattler global search
            cnt, cm, G = contact_counts(base, tol=1e-9)
            ratt = np.where(cnt == 0)[0]
            if len(ratt) > 0:
                victims = [int(ratt[int(rng.randint(len(ratt)))])]
            else:
                victims = [int(np.argmin(cnt))]
            Xp = greedy_reinsert(base, victims, ntry=8000)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            sigma_class = 0
        else:  # small + overloaded kick
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            cm = G.max()
            cnt = (G >= cm - 2e-3).sum(axis=1)
            order = np.argsort(-cnt)
            Xp = base + 0.015 * rng.randn(n, d)
            for vv in order[:4]:
                Xp[int(vv)] += 0.05 * rng.randn(d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
            sigma_class = 1

        if sigma_class == 2:
            steps, s0, s1, lr0, lr1 = 950, 60.0, 2200.0, 0.02, 0.0007
        elif sigma_class == 1:
            steps, s0, s1, lr0, lr1 = 750, 120.0, 3200.0, 0.012, 0.0005
        else:
            steps, s0, s1, lr0, lr1 = 550, 200.0, 4500.0, 0.008, 0.0004
        remain = SEARCH_END - (time.time())
        if remain < 10:
            steps = min(steps, 350)
        Xs = refine(Xp, steps, s0, s1, lr0, lr1)
        v = _max_cos(Xs)
        # tighten promising basins
        if v < best + 5e-4:
            try:
                Xs2 = refine(Xs, 220, 1000.0, 8000.0, 0.003, 0.0003)
                v2 = _max_cos(Xs2)
                if v2 < v:
                    Xs, v = Xs2, v2
            except Exception:
                pass
        if v < best - 1e-12:
            best = v
            Xbest = Xs.copy()
            Xcur = Xs.copy()
            cur = v
        else:
            # Metropolis acceptance for walk
            if v < cur + 0.01:
                prob = np.exp((cur - v) / max(Temp, 1e-9)) if v > cur else 1.0
                if rng.rand() < prob:
                    Xcur = Xs.copy()
                    cur = v
            # periodic restart of walk to best if stuck
            if trial % 30 == 0 and cur > best + 0.002:
                Xcur = Xbest.copy()
                cur = best

    # final intensify + polish
    try:
        if time.time() < DEADLINE - 6:
            Xbest = refine(Xbest, 600, 800.0, 9000.0, 0.003, 0.0002)
            best = _max_cos(Xbest)
    except Exception:
        pass
    try:
        rem = DEADLINE - time.time() - 1.0
        if rem > 3:
            Xbest = _lp_polish(Xbest, min(rem, 26.0), t_end=DEADLINE - 0.5)
    except Exception:
        pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
