import time
import numpy as np

def _norm(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return A / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _full_polish(Y0, steps, beta0, beta1, lr0, lr1, deadline=1e18):
    Y = Y0.copy()
    M = np.zeros_like(Y)
    V = np.zeros_like(Y)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lbb0 = np.log(beta0); lbb1 = np.log(beta1)
    llr0 = np.log(lr0); llr1 = np.log(lr1)
    b1t = 1.0; b2t = 1.0
    best_val = _max_cos(Y)
    best_Y = Y.copy()
    n = Y.shape[0]
    for t in range(1, steps + 1):
        if (t % 200 == 0) and (time.time() > deadline):
            break
        frac = (t - 1) / max(1, steps - 1)
        beta = float(np.exp(lbb0 + (lbb1 - lbb0) * frac))
        lr = float(np.exp(llr0 + (llr1 - llr0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        G -= gmax
        G *= beta
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s = float(G.sum())
        if not np.isfinite(s) or s <= 0:
            continue
        G /= s
        Gr = G @ Y
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gr -= Y * dot
        b1t *= b1; b2t *= b2
        M *= b1; M += (1.0 - b1) * Gr
        V *= b2; V += (1.0 - b2) * (Gr * Gr)
        Mh = M / (1.0 - b1t)
        Vh = V / (1.0 - b2t)
        Y -= lr * Mh / (np.sqrt(Vh) + eps)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Y /= rn
        if (t % 200 == 0) or (t == steps):
            v = _max_cos(Y)
            if v < best_val:
                best_val = v
                best_Y = Y.copy()
    return best_Y, best_val

def _single_refine(y0, F, steps, beta0, beta1, lr0, lr1, rng=None):
    # minimize LSE(F @ y) over sphere; F:(m,22), y:(22,)
    y = y0.copy()
    nrm = float(np.linalg.norm(y))
    if nrm == 0 or not np.isfinite(nrm):
        y = rng.standard_normal(22) if rng is not None else np.random.randn(22)
        nrm = float(np.linalg.norm(y))
    y /= nrm
    m = np.zeros(22); v = np.zeros(22)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lbb0 = np.log(beta0); lbb1 = np.log(beta1)
    llr0 = np.log(lr0); llr1 = np.log(lr1)
    b1t = 1.0; b2t = 1.0
    for t in range(1, steps + 1):
        frac = (t - 1) / max(1, steps - 1)
        beta = float(np.exp(lbb0 + (lbb1 - lbb0) * frac))
        lr = float(np.exp(llr0 + (llr1 - llr0) * frac))
        d = F @ y
        dm = float(d.max())
        w = np.exp(beta * (d - dm))
        sw = float(w.sum())
        if not np.isfinite(sw) or sw <= 0:
            continue
        w /= sw
        g = w @ F
        g -= y * float(g @ y)
        b1t *= b1; b2t *= b2
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g * g)
        mh = m / (1.0 - b1t); vh = v / (1.0 - b2t)
        y -= lr * mh / (np.sqrt(vh) + eps)
        nrm = float(np.linalg.norm(y))
        if nrm == 0 or not np.isfinite(nrm):
            y = F[0] * 0.0
            y[0] = 1.0
            nrm = 1.0
        y /= nrm
        # check every 50 iterations (cheap, also satisfies time-check cadence)
        if (t % 50 == 0):
            pass
    return y

def solve():
    t0 = time.time()
    deadline_cluster = t0 + 160.0
    deadline_hard = t0 + 174.0
    rng = np.random.default_rng(0)

    X0 = None; best0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (59, 22):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            vv = _max_cos(a)
            if vv < best0:
                best0 = vv
                X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = _norm(rng.standard_normal((59, 22)))
        best0 = _max_cos(X0)
    x_best = X0.copy()
    best_val = best0

    # precomputed pool >=20000 random unit vectors
    POOL_N = 20000
    Pool = _norm(rng.standard_normal((POOL_N, 22)))

    # contact degrees of current best (margin for neighbours)
    def contact_order(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        deg = np.sum(G > (m - 0.008), axis=1)
        order = np.argsort(-deg)
        return G, m, deg, order

    Gc, mc, degc, orderc = contact_order(x_best)

    nclust = 0
    # cycle through high-contact centers
    while time.time() < deadline_cluster - 4.0:
        if nclust % 25 == 0:
            Gc, mc, degc, orderc = contact_order(x_best)
        # pick center: cycle through top-30 highly connected
        topk = orderc[:30]
        center = int(topk[nclust % len(topk)])
        # neighbours: contact-graph neighbours of center
        grow = Gc[center].copy()
        grow[center] = -2.0
        # contact threshold for neighbour candidacy
        cand_nb = np.where(grow > (mc - 0.010))[0]
        if len(cand_nb) < 5:
            cand_nb = np.argsort(-grow)[:12]
        k = 6 + (nclust % 4)  # 6..9 removed (center + 5..8 neighbours)
        kneb = min(k - 1, len(cand_nb))
        # random subset among contacts, deterministic via rng
        perm = rng.permutation(len(cand_nb))
        chosen_nb = np.sort(cand_nb[perm[:kneb]])
        removed = np.sort(np.concatenate(([center], chosen_nb)))
        kact = len(removed)
        keep_mask = np.ones(59, dtype=bool)
        keep_mask[removed] = False
        Ffrozen = x_best[keep_mask]  # (59-kact, 22)

        # sequential reinsertion
        new_pts = []
        # for diversity: alternate best vs random-among-top
        use_random_top = (nclust % 2 == 1)
        ok = True
        for pos in range(kact):
            if time.time() > deadline_cluster - 2.0:
                ok = False
                break
            if len(new_pts) == 0:
                Fcur = Ffrozen
            else:
                Fcur = np.vstack([Ffrozen] + [np.atleast_2d(p) for p in new_pts])
            # one matrix multiplication scoring
            S = Pool @ Fcur.T
            rowmax = S.max(axis=1)
            # best 100
            part = np.argpartition(rowmax, 100)[:100]
            # order those 100 by score
            sub = rowmax[part]
            o = np.argsort(sub)
            top100 = part[o]
            # choose 3 starts: best, one of top-10, one of top-100
            starts = []
            starts.append(Pool[top100[0]])
            starts.append(Pool[top100[int(rng.integers(1, 10))]])
            starts.append(Pool[top100[int(rng.integers(10, 100))]])
            if use_random_top and pos == 0:
                # force topology change: start from random top-20 rather than global best
                starts[0] = Pool[top100[int(rng.integers(0, 20))]]
            best_y = None; best_s = 1e9
            for s0 in starts:
                y = _single_refine(s0, Fcur, steps=350, beta0=25.0, beta1=300.0,
                                   lr0=0.03, lr1=0.002, rng=rng)
                sc = float((Fcur @ y).max())
                if sc < best_s:
                    best_s = sc
                    best_y = y
            new_pts.append(best_y)
            # time check every vacancy; inner 50-iter checks are inside _single_refine cadence
        if not ok:
            break
        # assemble trial
        Xtrial = np.empty_like(x_best)
        Xtrial[keep_mask] = Ffrozen
        for ii, idx in enumerate(removed):
            Xtrial[idx] = new_pts[ii]
        Xtrial = _norm(Xtrial)
        # unfreeze short polish 1500 steps
        Yp, vp = _full_polish(Xtrial, steps=1500, beta0=45.0, beta1=520.0,
                              lr0=0.012, lr1=0.0012, deadline=deadline_cluster)
        nclust += 1
        if vp < best_val - 1e-12:
            best_val = vp
            x_best = Yp.copy()
            Gc, mc, degc, orderc = contact_order(x_best)
        # also track intermediate best inside polish via Yp already best-tracked
        if time.time() > deadline_cluster - 4.0:
            break

    # final short polish of best if time remains
    if time.time() < deadline_hard - 8.0:
        remain = deadline_hard - time.time()
        steps = 4000 if remain > 12 else 1500
        Yp, vp = _full_polish(x_best, steps=steps, beta0=60.0, beta1=700.0,
                              lr0=0.008, lr1=0.0008, deadline=deadline_hard - 1.0)
        if vp < best_val:
            best_val = vp
            x_best = Yp.copy()
    x_best = _norm(x_best)
    return x_best
