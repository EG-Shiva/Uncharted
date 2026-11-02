import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contact_counts(X, tol=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = (G > m - tol)
    return C.sum(axis=1), m, G

def _adam_quench(X0, betas, iters_list, lrs, rng=None):
    n, d = X0.shape
    Y = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    iu = np.triu_indices(n, 1)
    # off-diag mask for sum
    for beta, iters, lr0 in zip(betas, iters_list, lrs):
        m1 = np.zeros((n, d))
        v1 = np.zeros((n, d))
        t = 0
        for s in range(iters):
            G = Y @ Y.T
            gv = G[iu]
            mx = gv.max()
            e = np.exp(beta * (gv - mx))
            S = e.sum()
            if S <= 0 or not np.isfinite(S):
                break
            w = e / S
            W = np.zeros((n, n))
            W[iu] = w
            W = W + W.T
            GradY = W @ Y
            dot = (GradY * Y).sum(axis=1, keepdims=True)
            Grad = GradY - dot * Y
            # Adam
            t += 1
            m1 = 0.9 * m1 + 0.1 * Grad
            v1 = 0.999 * v1 + 0.001 * (Grad * Grad)
            mh = m1 / (1.0 - 0.9 ** t)
            vh = v1 / (1.0 - 0.999 ** t)
            step = lr0 * mh / (np.sqrt(vh) + 1e-8)
            # project step to tangent
            sdot = (step * Y).sum(axis=1, keepdims=True)
            stept = step - sdot * Y
            # decay
            # slight linear decay
            Y = Y - stept
            nr = np.linalg.norm(Y, axis=1, keepdims=True)
            nr = np.maximum(nr, 1e-12)
            Y = Y / nr
    return Y

def _greedy_max_tune(X0, steps=400, lr=0.003):
    Y = X0 / np.linalg.norm(X0, axis=1, keepdims=True)
    n, d = Y.shape
    best = _max_cos(Y)
    bestY = Y.copy()
    cur_lr = lr
    for s in range(steps):
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        # active band shrinks
        band = 2e-4 if s < steps // 2 else 5e-5
        mask = (G > m - band)
        # degree
        deg = mask.sum(axis=1)
        # gradient of active max: sum of partner tangential comps
        # Grad_i = sum_{j in N(i)} (Yj - gij Yi)
        # compute efficiently
        # W active 0/1
        A = mask.astype(np.float64)
        # sum Yj
        SY = A @ Y  # (n,d)
        # sum gij over neighbors
        # g = G*mask summed
        sg = (G * A).sum(axis=1, keepdims=True)  # (n,1)
        Grad = SY - sg * Y
        # normalize by degree to avoid overload bias
        scale = np.maximum(deg[:, None], 1.0)
        Grad = Grad / scale
        # tangent already, but re-project tiny normal due to numerics
        dot = (Grad * Y).sum(axis=1, keepdims=True)
        Grad = Grad - dot * Y
        gn = np.linalg.norm(Grad)
        if gn < 1e-15 or not np.isfinite(gn):
            cur_lr *= 0.5
            continue
        Ytry = Y - cur_lr * Grad
        Ytry /= np.linalg.norm(Ytry, axis=1, keepdims=True)
        v = _max_cos(Ytry)
        if v < best:
            best = v
            bestY = Ytry.copy()
            Y = Ytry
            cur_lr *= 1.05
            if cur_lr > 0.01:
                cur_lr = 0.01
        else:
            cur_lr *= 0.5
            if cur_lr < 1e-6:
                cur_lr = 1e-6
                # randomize direction a bit? keep going
                Y = Ytry  # allow sideways move occasionally? no, revert
                Y = bestY.copy() + 0.0005 * np.random.standard_normal((n, d))
                Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                # actually use deterministic small kick every stall? revert to best
                Y = bestY.copy()
    return bestY

def _largest_void(cands, Xref):
    # cands (M,d) normalized, Xref (K,d)
    dots = cands @ Xref.T  # M x K
    mx = dots.max(axis=1)
    return mx

def solve():
    t0 = time.time()
    LIMIT = 168.0
    rng = np.random.default_rng(0)
    n, d = 78, 15
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            v = _max_cos(a)
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)

    # quick baseline greedy squeeze (cheap, may shave 1e-12)
    # do tiny adam to ensure local snap without heavy cost
    if time.time() - t0 < LIMIT - 100:
        try:
            q = _adam_quench(bestX, [600.0], [300], [0.005])
            v = _max_cos(q)
            if v < bestV:
                bestV = v
                bestX = q
        except Exception:
            pass

    it = 0
    # pre-generate candidate pool base? generate on fly
    while time.time() - t0 < LIMIT - 8:
        it += 1
        elapsed = time.time() - t0
        remain = LIMIT - 8 - elapsed
        if remain < 4:
            break
        # choose mode
        mode = it % 5
        try:
            cnts, mcur, Gcur = _contact_counts(bestX, tol=1e-9)
        except Exception:
            cnts = np.ones(n)
        order = np.argsort(cnts, kind='stable')
        Xtry = bestX.copy()
        if mode == 0:
            # global small jitter
            sigma = float(0.02 + 0.03 * rng.random())
            Xtry = bestX + sigma * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        elif mode == 4:
            # medium global jitter
            sigma = float(0.06 + 0.08 * rng.random())
            Xtry = bestX + sigma * rng.standard_normal((n, d))
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
        else:
            if mode == 1:
                k = 1
            elif mode == 2:
                k = 2
            else:
                k = 3
            # pool of loosest
            pool = order[:16]
            chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
            chosen = np.sort(np.asarray(chosen))
            # build reference set excluding chosen
            mask_keep = np.ones(n, dtype=bool)
            mask_keep[chosen] = False
            Xkeep = bestX[mask_keep]
            # sequentially insert voids
            placed = []
            for idx in chosen:
                M = 3000
                cands = rng.standard_normal((M, d))
                cands /= np.linalg.norm(cands, axis=1, keepdims=True)
                if len(placed) > 0:
                    P = np.vstack([Xkeep] + placed)
                else:
                    P = Xkeep
                scores = _largest_void(cands, P)
                bi = int(np.argmin(scores))
                vdir = cands[bi]
                # local refine: few hill steps on sphere minimizing max dot
                # sample around best
                best_c = vdir
                best_s = float(scores[bi])
                for _ in range(60):
                    prop = best_c + 0.02 * rng.standard_normal(d)
                    prop /= np.linalg.norm(prop)
                    s = float((prop @ P.T).max())
                    if s < best_s:
                        best_s = s
                        best_c = prop
                Xtry[idx] = best_c
                placed.append(best_c[None, :])
            # jitter the rest slightly
            sigma = float(0.008 + 0.02 * rng.random())
            noise = rng.standard_normal((n, d)) * sigma
            # don't jitter freshly placed much
            for idx in chosen:
                noise[idx] *= 0.15
            Xtry = Xtry + noise
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)

        # quench: adapt iters to remaining time (fast)
        # use 2-stage anneal
        try:
            # fewer iters when little time
            if remain < 20:
                Xq = _adam_quench(Xtry, [300.0, 900.0], [300, 300], [0.012, 0.006])
            else:
                Xq = _adam_quench(Xtry, [250.0, 800.0, 1600.0], [400, 400, 250], [0.014, 0.008, 0.004])
            # greedy squeeze short
            Xq2 = _greedy_max_tune(Xq, steps=150, lr=0.002)
            v = _max_cos(Xq2)
            # also check pre-squeeze
            vq = _max_cos(Xq)
            if vq < v:
                v = vq
                Xq2 = Xq
            if v < bestV:
                bestV = v
                bestX = Xq2.copy()
        except Exception:
            continue
        # occasional intensify best with longer squeeze if improved recently? keep loop

    # final tiny intensification on best if time left
    if time.time() - t0 < LIMIT - 2:
        try:
            Xt = _greedy_max_tune(bestX, steps=250, lr=0.0015)
            v = _max_cos(Xt)
            if v < bestV:
                bestV = v
                bestX = Xt
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    # guard NaN
    bestX = np.where(np.isfinite(bestX), bestX, 0.0)
    nr = np.linalg.norm(bestX, axis=1, keepdims=True)
    nr = np.where(nr < 1e-12, 1.0, nr)
    bestX = bestX / nr
    return bestX
