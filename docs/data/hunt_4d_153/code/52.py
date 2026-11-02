import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def random_sphere(n, d, rng):
    z = rng.standard_normal((n, d))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z


def smooth_adam(X, steps, a0, a1, lr0, lr1, t_deadline=1e30):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    lrat = lr1 / lr0 if lr0 != 0 else 1.0
    arat = a1 / a0 if a0 != 0 else 1.0
    for t in range(1, steps + 1):
        if t % 400 == 0 and time.time() > t_deadline:
            break
        frac = t / steps
        alpha = a0 * (arat ** frac)
        lr = lr0 * (lrat ** frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m0 = np.max(G)
        W = np.exp(alpha * (G - m0))
        np.fill_diagonal(W, 0.0)
        grad_raw = W @ X
        dot = np.sum(grad_raw * X, axis=1, keepdims=True)
        grad_t = grad_raw - dot * X
        m *= b1
        m += (1.0 - b1) * grad_t
        v *= b2
        v += (1.0 - b2) * (grad_t * grad_t)
        b1t = 1.0 - b1 ** t
        b2t = 1.0 - b2 ** t
        mhat = m / b1t
        vhat = v / b2t
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step -= sdot * X
        nrm = np.linalg.norm(step, axis=1, keepdims=True)
        scale = np.where(nrm > 0.02, 0.02 / (nrm + 1e-12), 1.0)
        step *= scale
        X = X - step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    I0 = iu0[0]
    J0 = iu0[1]
    while time.time() - start < seconds and radius > 1e-14:
        cfull = (x @ x.T)[iu0]
        m = float(np.max(cfull))
        thr = m - max(20.0 * radius, 1e-9)
        sel = cfull > thr
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3000:
            idx = np.where(sel)[0]
            order = np.argsort(cfull[idx])[-3000:]
            keep = np.zeros_like(sel)
            keep[idx[order]] = True
            sel = keep
            cnt = 3000
        I = I0[sel]
        J = J0[sel]
        csel = cfull[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if np.any(~np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        vv = max_cos(y)
        if vv < best:
            best = vv
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() - start >= seconds:
            break
    return best_x


def contact_info(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - tol)
    cc = B.sum(axis=1)
    return cc, m, G


def solve():
    t0 = time.time()
    DEADLINE = 178.0
    EXPLORE_END = t0 + 145.0
    rng = np.random.default_rng(7)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (153, 4):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (153, 4):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if len(cands) == 0:
        cands.append(random_sphere(153, 4, rng))
    bestX = None
    bestV = 1e30
    for c in cands:
        v = max_cos(c)
        if v < bestV:
            bestV = v
            bestX = c.copy()
    n, d = bestX.shape
    curX = bestX.copy()
    curV = bestV
    # quick contact order
    trial = 0
    iu_all0 = np.triu_indices(n, 1)
    while time.time() < EXPLORE_END and (time.time() - t0) < DEADLINE - 30:
        trial += 1
        elapsed = time.time() - t0
        prog = elapsed / 145.0
        if prog < 0:
            prog = 0.0
        if prog > 1:
            prog = 1.0
        # temperature: allow ~1e-3 early, ~1e-7 late
        T = 0.0012 * (1.0 - prog) ** 1.5 + 2e-7
        mode = trial % 6
        Xs = None
        # base is curX (walker), occasionally bestX
        base = curX if (trial % 4 != 0) else bestX
        if mode == 0:
            sigma = float(rng.uniform(0.015, 0.05))
            Xs = base + sigma * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xs = smooth_adam(Xs, steps=1400, a0=60, a1=350, lr0=0.004, lr1=0.0004, t_deadline=EXPLORE_END)
        elif mode == 1:
            sigma = float(rng.uniform(0.10, 0.24))
            Xs = base + sigma * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xs = smooth_adam(Xs, steps=2000, a0=30, a1=260, lr0=0.006, lr1=0.0005, t_deadline=EXPLORE_END)
        elif mode == 2:
            # cooperative patch shift
            c = int(rng.integers(0, n))
            dots = base @ base[c]
            k = int(rng.integers(10, 28))
            idx = np.argsort(-dots)[:k]
            r = rng.standard_normal(4)
            r -= (r @ base[c]) * base[c]
            rn = np.linalg.norm(r)
            if rn < 1e-12:
                r = np.array([1.0, 0.0, 0.0, 0.0])
                r -= (r @ base[c]) * base[c]
                rn = np.linalg.norm(r)
            r /= (rn + 1e-12)
            mag = float(rng.uniform(0.05, 0.18))
            s = mag * r
            Xs = base.copy()
            Xs[idx] += s
            Xs += 0.006 * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xs = smooth_adam(Xs, steps=1800, a0=40, a1=300, lr0=0.005, lr1=0.0004, t_deadline=EXPLORE_END)
        elif mode == 3:
            # kill and refill medium block, biased to overloaded + rattler region
            cc, m, G = contact_info(base, tol=1e-7)
            # rattlers
            ratt = np.where(cc == 0)[0]
            k = int(rng.integers(4, 11))
            idxset = set()
            if len(ratt) > 0 and rng.random() < 0.6:
                # kill rattler + its nearest neighbors
                r0 = int(ratt[int(rng.integers(0, len(ratt)))])
                dots = base @ base[r0]
                nn = np.argsort(-dots)[:k]
                for a in nn:
                    idxset.add(int(a))
            else:
                order_hi = np.argsort(-cc)
                top = order_hi[:30]
                nh = min(k // 2 + 1, len(top))
                perm = rng.permutation(top)[:nh]
                for a in perm:
                    idxset.add(int(a))
                while len(idxset) < k:
                    idxset.add(int(rng.integers(0, n)))
            idx = np.array(sorted(idxset), dtype=int)
            Xs = base.copy()
            Xs[idx] = random_sphere(len(idx), d, rng)
            Xs += 0.010 * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xs = smooth_adam(Xs, steps=2000, a0=35, a1=280, lr0=0.005, lr1=0.0005, t_deadline=EXPLORE_END)
        elif mode == 4:
            # large kill for topology change
            k = int(rng.integers(10, 18))
            idx = np.array(sorted(rng.choice(n, size=k, replace=False)), dtype=int)
            Xs = base.copy()
            Xs[idx] = random_sphere(k, d, rng)
            Xs += 0.012 * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xs = smooth_adam(Xs, steps=2200, a0=30, a1=260, lr0=0.006, lr1=0.0005, t_deadline=EXPLORE_END)
        else:
            # explicit contact rewiring: break a max contact, make a second-shell pair
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            # contact pairs
            ci, cj = np.where(G > m - 1e-9)
            # second shell: slack in [1e-5, 8e-3]
            iu = iu_all0
            cfull = G[iu]
            second = np.where((cfull < m - 1e-5) & (cfull > m - 8e-3))[0]
            Xs = base.copy()
            if len(ci) > 0:
                pick = int(rng.integers(0, len(ci)))
                i = int(ci[pick]); j = int(cj[pick])
                # push i away from j tangentially
                step = float(rng.uniform(0.02, 0.06))
                dir_i = base[j] - (base[j] @ Xs[i]) * Xs[i]
                dir_j = base[i] - (base[i] @ Xs[j]) * Xs[j]
                Xs[i] -= step * dir_i
                Xs[j] -= step * dir_j
            if len(second) > 0:
                pick2 = int(second[int(rng.integers(0, len(second)))])
                p = int(iu[0][pick2]); q = int(iu[1][pick2])
                step2 = float(rng.uniform(0.01, 0.04))
                # move p toward q
                dir_p = base[q] - (base[q] @ Xs[p]) * Xs[p]
                dir_q = base[p] - (base[p] @ Xs[q]) * Xs[q]
                Xs[p] += step2 * dir_p
                Xs[q] += step2 * dir_q
            Xs += 0.008 * rng.standard_normal((n, d))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
            Xs = smooth_adam(Xs, steps=1800, a0=45, a1=320, lr0=0.005, lr1=0.0004, t_deadline=EXPLORE_END)
        if Xs is None:
            continue
        v = max_cos(Xs)
        # decide LP
        do_lp = False
        lp_time = 0.7
        if v < bestV:
            do_lp = True
            lp_time = 1.6
        elif v < curV + 3.0 * T + 0.0008:
            # polish with prob to save time
            if (trial % 2 == 0) or (v < curV):
                do_lp = True
                lp_time = 0.7
        elif rng.random() < 0.03:
            do_lp = True
            lp_time = 0.6
        if do_lp:
            remain = EXPLORE_END - time.time()
            if remain < 0.4:
                # no time for LP, just compare unpolished
                if v < bestV:
                    bestV = v
                    bestX = Xs.copy()
                if v < curV:
                    curV = v
                    curX = Xs.copy()
                continue
            lp_time = min(lp_time, remain - 0.05)
            if lp_time < 0.25:
                if v < bestV:
                    bestV = v
                    bestX = Xs.copy()
                continue
            Xp = lp_polish(Xs, lp_time)
            vp = max_cos(Xp)
            if vp < bestV - 1e-12:
                bestV = vp
                bestX = Xp.copy()
                curX = Xp.copy()
                curV = vp
            else:
                # metropolis on walker
                if vp <= curV:
                    curX = Xp.copy()
                    curV = vp
                else:
                    if T > 0:
                        prob = float(np.exp(-(vp - curV) / T)) if (vp - curV) / T < 20 else 0.0
                    else:
                        prob = 0.0
                    if rng.random() < prob:
                        curX = Xp.copy()
                        curV = vp
                    else:
                        # still allow unpolished if it is better than walker
                        if v < curV:
                            curX = Xs.copy()
                            curV = v
        else:
            if v < bestV - 1e-12:
                bestV = v
                bestX = Xs.copy()
                curX = Xs.copy()
                curV = v
            elif v < curV:
                curX = Xs.copy()
                curV = v
        # keep walker from drifting too far
        if curV > bestV + 0.005:
            # reset to best with small shake
            curX = bestX + 0.01 * rng.standard_normal((n, d))
            curX /= np.linalg.norm(curX, axis=1, keepdims=True)
            curV = max_cos(curX)
    # final thorough polish of best
    remain = DEADLINE - (time.time() - t0) - 1.0
    if remain > 1.0:
        bestX = lp_polish(bestX, min(remain * 0.7, 20.0))
        bestV = max_cos(bestX)
        # micro shake ensemble to squeeze epsilon
        t_end = t0 + DEADLINE - 0.5
        rng2 = np.random.default_rng(123)
        while time.time() < t_end:
            sig = 2e-4
            Xt = bestX + sig * rng2.standard_normal((n, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            left = t_end - time.time()
            if left < 0.4:
                break
            Xp = lp_polish(Xt, min(0.8, left - 0.05))
            vp = max_cos(Xp)
            if vp < bestV:
                bestV = vp
                bestX = Xp.copy()
            else:
                break
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
