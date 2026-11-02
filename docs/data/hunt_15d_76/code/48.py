import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts(X, thr=5e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    cnt = np.sum(G > m - thr, axis=1)
    return cnt, m, G

def _refine_hole(h, Y, iters=160):
    # minimize max(Y @ h) by soft-max descent, tangent projected
    h = h / np.linalg.norm(h)
    for it in range(iters):
        s = Y @ h
        m = float(s.max())
        # anneal temperature
        T = 0.008 * (0.25 ** (it / max(1, iters - 1)))
        e = np.exp((s - m) / T)
        se = float(e.sum())
        if se == 0 or not np.isfinite(se):
            break
        w = e / se
        g = w @ Y
        gh = float(g @ h)
        gt = g - gh * h
        lr = 0.08 * (0.2 ** (it / max(1, iters - 1)))
        h = h - lr * gt
        nh = np.linalg.norm(h)
        if nh == 0 or not np.isfinite(nh):
            break
        h /= nh
    return h

def solve():
    import numpy as np
    import time
    t0 = time.time()
    deadline = t0 + 165.0
    rng = np.random.default_rng(7)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None

    def normed(a):
        a = np.asarray(a, dtype=float)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        return a

    cands = []
    if xb is not None and xb.shape == (76, 15):
        try:
            cands.append(normed(xb))
        except Exception:
            pass
    if xr is not None and xr.shape == (76, 15):
        try:
            cands.append(normed(xr))
        except Exception:
            pass
    if len(cands) == 0:
        r = rng.normal(size=(76, 15))
        return _norm(r)

    best_val = 1e9
    Xbest = cands[0].copy()
    for c in cands:
        v = _maxcos(c)
        if v < best_val:
            best_val = v
            Xbest = c.copy()

    n, d = 76, 15
    cur = Xbest.copy()
    cur_val = best_val

    sigmas = [0.06, 0.15, 0.30, 0.02, 0.10, 0.22, 0.045, 0.35]
    trial = 0

    # main walker loop
    while time.time() < deadline - 9.0:
        trial += 1
        # time-adaptive steps
        remain = deadline - time.time()
        if remain < 10:
            break
        # contacts of walker
        cnt, mcur, _ = _contacts(cur)
        order = np.argsort(-cnt)

        sigma = sigmas[(trial - 1) % len(sigmas)]
        # number to eject: 2..5, larger when sigma large
        if sigma > 0.2:
            k_re = 4 + (trial % 2)  # 4-5
        elif sigma > 0.08:
            k_re = 3
        else:
            k_re = 2

        Xt = cur.copy()
        # choose victims: half overloaded, half random
        victims = []
        # overloaded picks
        no = min(k_re, 3)
        for j in range(no):
            idx = int(order[(trial * 2 + j * 7) % n])
            victims.append(idx)
        while len(victims) < k_re:
            idx = int(rng.integers(0, n))
            if idx not in victims:
                victims.append(idx)

        # sequential hole-optimized reinsertion
        for idx in victims:
            if time.time() > deadline - 9.0:
                break
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            Y = Xt[mask]
            NS = 3000
            if remain < 30:
                NS = 1500
            C = rng.normal(size=(NS, d))
            C = _norm(C)
            scores = (C @ Y.T).max(axis=1)
            bi = int(np.argmin(scores))
            h = _refine_hole(C[bi], Y, iters=140)
            Xt[idx] = h

        # gaussian shake (not on just-reinserted if sigma small? shake all to change topology)
        Xt = Xt + sigma * rng.normal(size=(n, d))
        Xt = _norm(Xt)

        # annealed soft-max Adam descent
        # steps adapt to time
        if remain > 80:
            steps = 1600
        elif remain > 40:
            steps = 1200
        else:
            steps = 800
        beta0, beta1 = 45.0, 400.0
        lr0, lr1 = 0.008, 0.0007
        m = np.zeros_like(Xt)
        vv = np.zeros_like(Xt)
        b1, b2, eps = 0.9, 0.999, 1e-8
        b1t, b2t = 1.0, 1.0
        lb = np.log(beta1 / beta0)
        ll = np.log(lr1 / lr0)
        Y = Xt
        # local fast loop with periodic time check
        for s_idx in range(steps):
            frac = s_idx / (steps - 1) if steps > 1 else 0.0
            beta = beta0 * np.exp(lb * frac)
            lr = lr0 * np.exp(ll * frac)
            G = Y @ Y.T
            np.fill_diagonal(G, -10.0)
            gm = float(G.max())
            E = np.exp(beta * (G - gm))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if S == 0 or not np.isfinite(S):
                break
            E /= S
            Gr = 2.0 * (E @ Y)
            # tangent projection
            Gt = Gr - (np.sum(Gr * Y, axis=1, keepdims=True)) * Y
            b1t *= b1
            b2t *= b2
            m *= b1
            m += (1 - b1) * Gt
            vv *= b2
            vv += (1 - b2) * (Gt * Gt)
            mhat = m / (1 - b1t)
            vhat = vv / (1 - b2t)
            Y = Y - lr * mhat / (np.sqrt(vhat) + eps)
            # renorm
            Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
            if (s_idx & 511) == 511 and time.time() > deadline - 9.0:
                break
        Xt = Y
        prop = _maxcos(Xt)

        # update best ever
        if prop < best_val:
            best_val = prop
            Xbest = Xt.copy()
            cur = Xt.copy()
            cur_val = prop
            continue

        # walker Metropolis acceptance (allow sideways to change basin)
        # temperature for acceptance
        Tmeta = 0.0025
        if prop < cur_val:
            cur = Xt.copy()
            cur_val = prop
        else:
            # accept worse with prob exp(-(prop-cur)/Tmeta), but only if not too bad
            if prop - cur_val < 0.02:
                if float(rng.random()) < np.exp(-(prop - cur_val) / Tmeta):
                    cur = Xt.copy()
                    cur_val = prop
            # occasional restart to best if walker drifts too high
            if cur_val > best_val + 0.015:
                # restart from best with big kick next trial
                cur = Xbest.copy()
                cur_val = best_val

        # opportunistic: if walker equals best region, keep going
        trial += 0

    Xbest = _norm(Xbest)
    # safety: finite check, fallback to loaded best
    if not np.all(np.isfinite(Xbest)):
        Xbest = normed(cands[0])
    # ensure no degradation vs loaded best
    try:
        vfinal = _maxcos(Xbest)
        v0 = _maxcos(normed(cands[0]))
        if vfinal > v0 and np.all(np.isfinite(Xbest)):
            # keep Xbest if it is better than nothing? actually revert if worse
            # but we want best found; if our search never improved, Xbest==best so equal
            # this guards numerical drift
            if vfinal > v0 + 1e-12:
                # check if we actually improved at all; if not, return original
                if best_val >= v0 - 1e-15:
                    Xbest = normed(cands[0])
    except Exception:
        pass
    return Xbest
