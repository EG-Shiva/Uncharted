import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius, start = 1e-3, time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    rng = np.random.RandomState(20261102)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.randn(73, 13)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 73, 13
    if xb.shape != (n, d):
        # fix shape by resampling / truncating deterministically
        tmp = rng.randn(n, d)
        m = min(xb.shape[0], n)
        tmp[:m, :min(xb.shape[1], d)] = xb[:m, :min(xb.shape[1], d)]
        xb = tmp / np.linalg.norm(tmp, axis=1, keepdims=True)
    best = _max_cos(xb)
    Xbest = xb.copy()
    iu0 = np.triu_indices(n, 1)
    I0, J0 = iu0[0], iu0[1]

    def smooth_run(X0, steps, s0, s1, lr0, lr1):
        X = X0.copy()
        m1 = np.zeros_like(X)
        v1 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        # precompute powers for bias correction incrementally to avoid pow each step
        b1p, b2p = b1, b2
        # geometric schedules via log interpolation
        ls0, ls1 = np.log(s0), np.log(s1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        grad = np.empty_like(X)
        for k in range(steps):
            frac = k / (steps - 1) if steps > 1 else 1.0
            s = np.exp(ls0 + (ls1 - ls0) * frac)
            lr = np.exp(ll0 + (ll1 - ll0) * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            c = G[I0, J0]
            cmax = c.max()
            # softmax weights over pairs
            e = np.exp(s * (c - cmax))
            ssum = e.sum()
            if ssum == 0 or not np.isfinite(ssum):
                # fallback to subgradient on max pairs
                w = (c >= cmax - 1e-12).astype(float)
                w /= w.sum()
            else:
                w = e / ssum
            grad.fill(0.0)
            # distribute: grad_i += w_ij * X_j
            WXj = w[:, None] * X[J0]
            WXi = w[:, None] * X[I0]
            np.add.at(grad, I0, WXj)
            np.add.at(grad, J0, WXi)
            dot = (grad * X).sum(axis=1, keepdims=True)
            gt = grad - dot * X
            m1 *= b1
            m1 += (1 - b1) * gt
            v1 *= b2
            v1 += (1 - b2) * (gt * gt)
            # bias correction
            mhat = m1 / (1 - b1p)
            vhat = v1 / (1 - b2p)
            b1p *= b1
            b2p *= b2
            step = mhat / (np.sqrt(vhat) + eps)
            # clip extreme steps for stability
            # limit per-row norm
            X -= lr * step
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
        return X

    def random_sphere_vec():
        v = rng.randn(d)
        v /= np.linalg.norm(v)
        return v

    def greedy_reinsert(X, idxs, ntry=4000):
        # for each idx, sample candidates and keep best (min max-dot)
        Y = X.copy()
        for idx in idxs:
            G = Y @ Y.T
            # dots of idx to others not needed; we search new vector
            C = rng.randn(ntry, d)
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            others = np.delete(Y, idx, axis=0)  # (n-1,d)
            dots = C @ others.T  # (ntry,n-1)
            md = dots.max(axis=1)
            bi = int(np.argmin(md))
            Y[idx] = C[bi]
        return Y

    # initial contact info for targeted kicks
    SEARCH_BUDGET = 132.0
    trial = 0
    # quick first smooth on loaded best to ensure local tightness with fast method
    # diverse schedule list
    while time.time() - t0 < SEARCH_BUDGET:
        trial += 1
        remain = SEARCH_BUDGET - (time.time() - t0)
        if remain < 2:
            break
        # choose perturbation mode deterministically cycling + random
        mode = trial % 10
        # contact overload of current best (recompute every 5 trials)
        if trial % 5 == 1:
            Gc = Xbest @ Xbest.T
            np.fill_diagonal(Gc, -2.0)
            cm = Gc.max()
            # contacts within 2e-3
            cnt = (Gc >= cm - 2e-3).sum(axis=1)
            order = np.argsort(-cnt)
        sigma = 0.02
        Xp = Xbest.copy()
        if mode == 0:
            # heavy global shake
            sigma = 0.12
            Xp = Xbest + sigma * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 1:
            # reinsert 1-2 worst + rattler-like
            Gc = Xbest @ Xbest.T
            np.fill_diagonal(Gc, -2.0)
            cm = Gc.max()
            cnt = (Gc >= cm - 1e-9).sum(axis=1)
            # rattlers = zero contacts
            ratt = np.where(cnt == 0)[0]
            k = 2
            victims = list(ratt[:2])
            # add most overloaded
            ov = np.argsort(-cnt)[:k]
            for v in ov:
                if v not in victims:
                    victims.append(int(v))
            victims = victims[:3]
            if len(victims) == 0:
                victims = [int(rng.randint(n))]
            Xp = greedy_reinsert(Xbest, victims, ntry=3000)
            Xp = Xp + 0.01 * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 2:
            # re-randomize 2 random points completely
            Xp = Xbest.copy()
            for _ in range(2):
                ii = int(rng.randint(n))
                Xp[ii] = random_sphere_vec()
            Xp = Xp + 0.005 * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 3:
            sigma = 0.06
            Xp = Xbest + sigma * rng.randn(n, d)
            # extra kick to top overloaded
            for v in order[:5]:
                Xp[int(v)] += 0.08 * rng.randn(d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 4:
            sigma = 0.008
            Xp = Xbest + sigma * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 5:
            sigma = 0.03
            Xp = Xbest + sigma * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 6:
            # single point reinsert (rattler move)
            Gc = Xbest @ Xbest.T
            np.fill_diagonal(Gc, -2.0)
            cm = Gc.max()
            cnt = (Gc >= cm - 1e-9).sum(axis=1)
            ratt = np.where(cnt == 0)[0]
            if len(ratt) > 0:
                victims = [int(ratt[0])]
            else:
                victims = [int(np.argmin(cnt))]
            Xp = greedy_reinsert(Xbest, victims, ntry=5000)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 7:
            sigma = 0.05
            # random subspace rotation of a cluster: pick 8 points, random orthogonal jitter
            Xp = Xbest + sigma * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        elif mode == 8:
            sigma = 0.015
            Xp = Xbest + sigma * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        else:
            sigma = 0.09
            Xp = Xbest + sigma * rng.randn(n, d)
            Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
        # choose smooth hyperparams depending on sigma (large kick -> coarse then fine)
        if sigma >= 0.08:
            steps, s0, s1, lr0, lr1 = 1100, 80.0, 1800.0, 0.02, 0.0008
        elif sigma >= 0.03:
            steps, s0, s1, lr0, lr1 = 900, 120.0, 2500.0, 0.012, 0.0006
        else:
            steps, s0, s1, lr0, lr1 = 700, 200.0, 3500.0, 0.008, 0.0004
        # shrink steps if little time left
        if remain < 15:
            steps = min(steps, 400)
        Xs = smooth_run(Xp, steps, s0, s1, lr0, lr1)
        v = _max_cos(Xs)
        if v < best - 1e-12:
            best = v
            Xbest = Xs
            # intensify around new best with short fine run if time allows
            if time.time() - t0 < SEARCH_BUDGET - 5:
                Xf = smooth_run(Xbest, 350, 500.0, 5000.0, 0.004, 0.0003)
                vf = _max_cos(Xf)
                if vf < best:
                    best = vf
                    Xbest = Xf
        # occasional acceptance of near-equal with different topology to diversify:
        # if within 5e-4 and random, keep as base for next perturbation? we keep elitist
        # but every 20 trials, if stuck, do a random walk step from last Xs
        if trial % 25 == 0 and (time.time() - t0) < SEARCH_BUDGET - 10:
            # try to escape: perturb current Xs even if worse and smooth again
            Xp2 = Xs + 0.04 * rng.randn(n, d)
            Xp2 /= np.linalg.norm(Xp2, axis=1, keepdims=True)
            Xs2 = smooth_run(Xp2, 600, 120.0, 2200.0, 0.01, 0.0006)
            v2 = _max_cos(Xs2)
            if v2 < best - 1e-12:
                best = v2
                Xbest = Xs2
    # final polish with remaining time
    rem = 178.0 - (time.time() - t0)
    if rem > 4:
        # first a final sharp smooth to tighten
        try:
            Xbest = smooth_run(Xbest, 500, 800.0, 6000.0, 0.003, 0.0002)
        except Exception:
            pass
        rem2 = 178.0 - (time.time() - t0)
        if rem2 > 3:
            try:
                Xbest = _lp_polish(Xbest, min(rem2 - 0.5, 28.0))
            except Exception:
                pass
    Xbest = Xbest / np.linalg.norm(Xbest, axis=1, keepdims=True)
    return Xbest
