import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def contacts_cnt(x, thresh):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    cnt = np.sum(g > m - thresh, axis=1)
    return m, cnt

def lp_polish(x, seconds, r0=1.5e-3, rmax=0.05, cap=2600):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0[0], iu0[1]
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.06:
            break
        G = x @ x.T
        cfull = G[iuI, iuJ]
        m = float(cfull.max())
        if m < best:
            best = m
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(sel.sum())
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.argpartition(-cfull, cap)[:cap]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = cap
        I = iuI[sel]; J = iuJ[sel]
        c = cfull[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([ (I[:, None]*d + np.arange(d)), (J[:, None]*d + np.arange(d)), np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius*1.5, rmax)
        else:
            radius *= 0.5
    return x

def smooth_refine(x0, tau1=0.04, tau2=0.012, mi1=70, mi2=100):
    n, d = x0.shape
    flat = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    for tau, mi in ((tau1, mi1), (tau2, mi2)):
        def fun(f):
            X = f.reshape(n, d)
            norms = np.linalg.norm(X, axis=1)
            norms = np.maximum(norms, 1e-12)
            Xn = X / norms[:, None]
            G = Xn @ Xn.T
            np.fill_diagonal(G, -10.0)
            m = float(G.max())
            E = np.exp((G - m) / tau)
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if S <= 0 or not np.isfinite(S):
                S = 1e-300
            val = m + tau * np.log(S / 2.0 + 1e-300)
            W = E / S
            Gw = W @ Xn
            gXn = 2.0 * Gw
            dot = (gXn * Xn).sum(axis=1, keepdims=True)
            gX = (gXn - dot * Xn) / norms[:, None]
            pen = 0.1 * float(((norms - 1.0) ** 2).sum())
            val = val + pen
            gX = gX + (0.2 * (norms - 1.0))[:, None] * Xn
            return val, gX.ravel()
        try:
            res = minimize(fun, flat, jac=True, method="L-BFGS-B",
                           options={"maxiter": mi, "ftol": 1e-14, "gtol": 1e-7, "maxfun": mi*3})
            flat = res.x
        except Exception:
            break
    X = flat.reshape(n, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def random_sphere(rng, k, d):
    a = rng.randn(k, d)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    return a

def solve():
    rng = np.random.RandomState(20261103)
    t0 = time.time()
    DEADLINE = t0 + 172.0
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = random_sphere(rng, 162, 8)
    xb = np.asarray(xb, dtype=float).reshape(162, 8)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    x_best = xb.copy()
    # current walker (SA) starts at best
    x_cur = xb.copy()
    cur_val = best_val
    n, d = 162, 8
    # SA temperature schedule
    T0, T1 = 0.004, 0.00025
    trial = 0
    # precompute total explore time
    EXPLORE_END = DEADLINE - 12.0
    while time.time() < EXPLORE_END:
        trial += 1
        frac = (time.time() - t0) / max(1e-9, (EXPLORE_END - t0))
        Temp = T0 * (T1 / T0) ** min(1.0, max(0.0, frac))
        mode = trial % 8
        # contact info of walker
        try:
            _, cntW = contacts_cnt(x_cur, thresh=2e-3)
            _, cntT = contacts_cnt(x_cur, thresh=1e-7)
        except Exception:
            cntW = np.ones(n); cntT = np.ones(n)
        x0 = x_cur.copy()
        if mode == 0:
            # fresh random basin (topology reset)
            # alternate pure / antipodal / layered
            r = rng.rand()
            if r < 0.6:
                x0 = random_sphere(rng, n, d)
            elif r < 0.8:
                h = random_sphere(rng, 81, d)
                x0 = np.vstack([h, -h])
                # shuffle to avoid ordering bias
                rng.shuffle(x0)
            else:
                # two random caps + random rest
                x0 = random_sphere(rng, n, d)
        elif mode == 1:
            # large subset replacement from walker
            k = int(rng.randint(8, 28))
            idx = rng.choice(n, size=k, replace=False)
            # keep some structure: replace half randomly, kick half hard
            k1 = k // 2
            x0[idx[:k1]] = random_sphere(rng, k1, d)
            x0[idx[k1:]] = x0[idx[k1:]] + float(rng.uniform(0.2, 0.5)) * rng.randn(k-k1, d)
            x0 = x0 + 0.008 * rng.randn(n, d)
        elif mode == 2:
            # worst-pair split + overload shake
            G = x_cur @ x_cur.T
            np.fill_diagonal(G, -2.0)
            # find a few worst pairs
            flat_idx = np.argpartition(-G.ravel(), 5)[:5]
            pairs = [(int(i // n), int(i % n)) for i in flat_idx]
            for (a, b) in pairs:
                if a == b:
                    continue
                va, vb = x_cur[a], x_cur[b]
                # push apart tangentially
                direction = va - vb
                direction = direction - va * float(va @ direction)
                dn = np.linalg.norm(direction)
                if dn > 1e-12:
                    direction /= dn
                    step = float(rng.uniform(0.05, 0.15))
                    x0[a] = x0[a] + step * direction
                    x0[b] = x0[b] - step * (vb - va - vb*float(vb @ (vb-va)))/(np.linalg.norm(vb-va)+1e-12)
            k = int(rng.randint(5, 12))
            idx = np.argsort(cntW)[-k:]
            x0[idx] = x0[idx] + float(rng.uniform(0.05, 0.16)) * rng.randn(k, d)
            x0 = x0 + 0.005 * rng.randn(n, d)
        elif mode == 3:
            # hole re-insertion: move 1-3 points to deepest holes
            nh = int(rng.randint(1, 4))
            # candidates
            cand = random_sphere(rng, 4000, d)
            chosen = []
            avail = np.ones(n, dtype=bool)
            # pick points to move: prefer tight-contact + random
            order = np.argsort(-cntW)
            pool = list(order[:20]) + [int(rng.randint(0, n)) for _ in range(5)]
            for hh in range(nh):
                ri = int(rng.choice(pool))
                others_mask = np.ones(n, dtype=bool); others_mask[ri] = False
                # for hole use current x0 without ri? use x_cur
                others = x_cur[others_mask]
                dots = cand @ others.T
                worst = dots.max(axis=1)
                # enforce separation among chosen
                for q in chosen:
                    worst = np.maximum(worst, (cand @ q) - 0.15)
                bi = int(np.argmin(worst))
                x0[ri] = cand[bi]
                chosen.append(cand[bi].copy())
                # remove candidate neighbourhood to diversify
            x0 = x0 + 0.004 * rng.randn(n, d)
        elif mode == 4:
            # medium global kick from walker
            sigma = float(rng.uniform(0.04, 0.12))
            x0 = x0 + sigma * rng.randn(n, d)
        elif mode == 5:
            # shake underloaded + overloaded mix
            k = int(rng.randint(6, 16))
            kh = k // 2
            idx_h = np.argsort(cntW)[-kh:] if kh > 0 else np.array([], dtype=int)
            idx_l = np.argsort(cntW)[:k-kh]
            idx = np.concatenate([idx_h, idx_l])
            x0[idx] = x0[idx] + float(rng.uniform(0.08, 0.22)) * rng.randn(len(idx), d)
            x0 = x0 + 0.006 * rng.randn(n, d)
        elif mode == 6:
            # small jitter of walker (local basin tweak)
            sigma = float(rng.uniform(0.008, 0.035))
            x0 = x0 + sigma * rng.randn(n, d)
        else:
            # crossover: mix walker and best
            alpha = rng.rand()
            # random orthogonal-ish mix: interpolate + noise
            if rng.rand() < 0.5:
                # block crossover
                k = int(rng.randint(20, 60))
                idx = rng.choice(n, size=k, replace=False)
                x0 = x_cur.copy()
                # take those rows from best with small rotation noise
                x0[idx] = x_best[idx] + 0.02 * rng.randn(k, d)
            else:
                sigma = float(rng.uniform(0.03, 0.08))
                # average directions
                x0 = x_cur + 0.3 * (x_best - x_cur) + sigma * rng.randn(n, d)
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        tau1 = float(rng.uniform(0.025, 0.06))
        tau2 = float(rng.uniform(0.007, 0.02))
        try:
            xs = smooth_refine(x0, tau1=tau1, tau2=tau2, mi1=60, mi2=90)
        except Exception:
            continue
        try:
            vs = max_cos(xs)
        except Exception:
            continue
        # decide LP: skip if hopelessly bad to save time
        # allow worse states, so threshold relative to cur
        if vs > cur_val + 0.025 and vs > best_val + 0.015:
            # still SA-move the walker occasionally without LP
            # accept as walker step if not too bad
            if vs < cur_val or rng.rand() < np.exp((cur_val - vs) / max(Temp, 1e-9)):
                x_cur, cur_val = xs, vs
            # update best just in case
            if vs < best_val - 1e-12:
                x_best, best_val = xs.copy(), vs
            continue
        if time.time() > EXPLORE_END:
            # no time for LP
            vl, xl = vs, xs
        else:
            remain = EXPLORE_END - time.time()
            budget = 1.2 if remain > 2.0 else max(0.35, remain * 0.5)
            try:
                xl = lp_polish(xs, budget)
                vl = max_cos(xl)
            except Exception:
                continue
        # SA acceptance on walker
        accepted = False
        if vl < cur_val:
            accepted = True
        else:
            try:
                p_acc = float(np.exp((cur_val - vl) / max(Temp, 1e-9)))
            except Exception:
                p_acc = 0.0
            if rng.rand() < p_acc:
                accepted = True
        if accepted:
            x_cur, cur_val = xl.copy(), vl
            # if walker was fresh random and improved a lot, keep exploring there
        if vl < best_val - 1e-12:
            x_best, best_val = xl.copy(), vl
        elif vs < best_val - 1e-12 and vs < vl:
            x_best, best_val = xs.copy(), vs
        # occasional restart of walker to best if walker drifts too bad
        if cur_val > best_val + 0.02 and rng.rand() < 0.35:
            x_cur, cur_val = x_best.copy(), best_val
    # final long polish of incumbent (crash-proof)
    try:
        remain = DEADLINE - time.time() - 0.8
        if remain > 2.0:
            x_best = lp_polish(x_best, min(remain, 10.0), r0=1e-3)
    except Exception:
        pass
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
