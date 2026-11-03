import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def gd_all(X, steps, T0, T1, lr0, lr1, mu=0.85):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        sw = W.sum()
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        # clip extreme
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def joint_place(C, k, rng, steps=600, T0=0.012, T1=0.0008, lr0=0.3, lr1=0.03):
    m, d = C.shape
    Y = rng.normal(size=(k, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    V = np.zeros_like(Y)
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    Cn = C  # assumed normalized
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        D1 = Y @ Cn.T  # k x m
        D2 = Y @ Y.T   # k x k
        m0 = float(max(D1.max(), D2[np.triu_indices(k, 1)].max() if k > 1 else D1.max()))
        W1 = np.exp((D1 - m0) / T)
        W2 = np.exp((D2 - m0) / T) if k > 1 else None
        if W2 is not None:
            np.fill_diagonal(W2, 0.0)
        ssum = W1.sum() + (W2.sum() if W2 is not None else 0.0)
        if not np.isfinite(ssum) or ssum <= 0:
            break
        W1 /= ssum
        if W2 is not None:
            W2 /= ssum
        Grad = W1 @ Cn
        if W2 is not None:
            Grad = Grad + W2 @ Y
        dot = np.sum(Grad * Y, axis=1, keepdims=True)
        Gt = Grad - dot * Y
        V = 0.85 * V + Gt
        Y = Y - lr * V
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y


def joint_best_hole(C, k, rng):
    # multiple restarts, pick best true max
    bestY = None
    bestv = 1e9
    # one restart from best-of-random sampling
    # fast sampling to seed
    cand = rng.normal(size=(4000, C.shape[1]))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    # greedy sequential seeding for k>1: pick sequentially far from C and each other
    seeds = []
    # first: farthest from C
    md = (cand @ C.T).max(axis=1)
    seeds.append(cand[int(np.argmin(md))])
    for a in range(1, k):
        S = np.stack(seeds, axis=0)
        d1 = (cand @ C.T).max(axis=1)
        d2 = (cand @ S.T).max(axis=1)
        dd = np.maximum(d1, d2)
        seeds.append(cand[int(np.argmin(dd))])
    Y0 = np.stack(seeds, axis=0)
    for r in range(3):
        if r == 0:
            Yinit = Y0
        else:
            Yinit = rng.normal(size=(k, C.shape[1]))
            Yinit /= np.linalg.norm(Yinit, axis=1, keepdims=True)
        # run joint_place starting from Yinit: emulate by calling with init override
        # inline optimization starting from Yinit
        Y = Yinit.copy()
        V = np.zeros_like(Y)
        steps = 500
        T0, T1 = 0.010, 0.0007
        lr0, lr1 = 0.30, 0.03
        lT0, lT1 = np.log(T0), np.log(T1)
        ll0, ll1 = np.log(lr0), np.log(lr1)
        for s in range(steps):
            f = s / max(1, steps - 1)
            T = float(np.exp(lT0 + (lT1 - lT0) * f))
            lr = float(np.exp(ll0 + (ll1 - ll0) * f))
            D1 = Y @ C.T
            D2 = Y @ Y.T if k > 1 else None
            if k > 1:
                off = D2[np.triu_indices(k, 1)].max()
                m0 = float(max(D1.max(), off))
            else:
                m0 = float(D1.max())
            W1 = np.exp((D1 - m0) / T)
            if k > 1:
                W2 = np.exp((D2 - m0) / T)
                np.fill_diagonal(W2, 0.0)
                ssum = W1.sum() + W2.sum()
                W1 /= ssum
                W2 /= ssum
            else:
                W2 = None
                W1 /= W1.sum()
            Grad = W1 @ C
            if W2 is not None:
                Grad = Grad + W2 @ Y
            dot = np.sum(Grad * Y, axis=1, keepdims=True)
            Gt = Grad - dot * Y
            V = 0.85 * V + Gt
            Y = Y - lr * V
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        # evaluate
        D1 = Y @ C.T
        v1 = float(D1.max())
        v2 = float((Y @ Y.T)[np.triu_indices(k, 1)].max()) if k > 1 else -2.0
        v = max(v1, v2)
        if v < bestv:
            bestv = v
            bestY = Y.copy()
    return bestY


def lp_polish(x, seconds, deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        if seconds - (time.time() - start) < 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) > 1600:
            idx = np.argpartition(c, -1600)[-1600:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    EXPL_END = t0 + 138.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (95, 13):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (95, 13):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        x0 = rng.normal(size=(95, 13))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        cands.append(x0)
    vals = [max_cos(c) for c in cands]
    best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))
    n, d = 95, 13

    it = 0
    while time.time() < EXPL_END and time.time() < DEADLINE - 35:
        it += 1
        # compute load stats on best
        G = best @ best.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        pm = G.max(axis=1)
        tol = 0.01
        counts = (G > m - tol).sum(axis=1)
        order_low = np.argsort(pm)          # rattlers first
        order_high = np.argsort(-counts)    # overloaded first

        u = rng.random()
        if u < 0.62:
            # joint hole reinsertion targeting imbalance
            if rng.random() < 0.7:
                k = int(rng.choice([2, 3, 3, 4, 4, 5, 6]))
            else:
                k = int(rng.integers(2, 6))
            victims = set()
            victims.add(int(order_low[0]))
            # add overloaded
            hi_ptr = 0
            while len(victims) < k and hi_ptr < n:
                cidx = int(order_high[hi_ptr])
                hi_ptr += 1
                if cidx not in victims:
                    victims.add(cidx)
            # occasional random swap for diversity
            if rng.random() < 0.35:
                victims = list(victims)
                victims[rng.integers(0, len(victims))] = int(rng.integers(0, n))
                victims = set(victims)
            victims = sorted(victims)
            keep = [i for i in range(n) if i not in victims]
            C = best[keep]
            Y = joint_best_hole(C, len(victims), rng)
            Xt = np.empty_like(best)
            Xt[keep] = C
            Xt[victims] = Y
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            # fast all-point annealed polish
            T0 = float(rng.choice([0.008, 0.012, 0.016]))
            Xt = gd_all(Xt, steps=450, T0=T0, T1=0.0006, lr0=0.12, lr1=0.01)
            v = max_cos(Xt)
            if v < best_val:
                best, best_val = Xt, v
            elif v < best_val + 0.004 and rng.random() < 0.02:
                # rare sideways drift to new basin
                best, best_val = Xt, v
        elif u < 0.82:
            # coordinated subset shake + polish
            ksub = int(rng.integers(6, 16))
            # bias to high-contact region
            probs = (counts.astype(float) + 1.0)
            probs /= probs.sum()
            try:
                idx = rng.choice(n, size=ksub, replace=False, p=probs)
            except Exception:
                idx = rng.choice(n, size=ksub, replace=False)
            sigma = float(rng.choice([0.05, 0.10, 0.18, 0.28, 0.40]))
            Xt = best.copy()
            Xt[idx] += sigma * rng.normal(size=(ksub, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_all(Xt, steps=500, T0=0.010, T1=0.0005, lr0=0.12, lr1=0.008)
            v = max_cos(Xt)
            if v < best_val:
                best, best_val = Xt, v
        else:
            # global shake + polish
            sigma = float(rng.choice([0.03, 0.07, 0.15, 0.25]))
            Xt = best + sigma * rng.normal(size=(n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = gd_all(Xt, steps=550, T0=0.012, T1=0.0005, lr0=0.12, lr1=0.008)
            v = max_cos(Xt)
            if v < best_val:
                best, best_val = Xt, v
        # safety time check
        if time.time() > EXPL_END:
            break

    # final squeeze: short gd cool then LP
    try:
        best = gd_all(best, steps=600, T0=0.004, T1=0.0002, lr0=0.06, lr1=0.005)
        bv = max_cos(best)
        if bv < best_val:
            best_val = bv
    except Exception:
        pass
    remain = DEADLINE - time.time() - 2.0
    if remain > 5:
        try:
            best = lp_polish(best, seconds=min(remain, 32.0), deadline=DEADLINE - 1.0)
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
