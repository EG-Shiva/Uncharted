import time
import numpy as np

def _normr(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return A / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 160.0

    # load start
    X0 = None
    bv0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (59, 22):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _normr(a)
            v = _max_cos(a)
            if v < bv0:
                bv0 = v
                X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = _normr(rng.standard_normal((59, 22)))
        bv0 = _max_cos(X0)
    x_best = X0.copy()
    best_val = float(bv0)

    # precomputed pool >=20000
    POOL_N = 22000
    Pool = _normr(rng.standard_normal((POOL_N, 22)))

    # ---------- full polish (1500-step class) ----------
    def polish_full(Y0, steps, beta0, beta1, lr0, lr1):
        Y = Y0.copy()
        M = np.zeros_like(Y)
        V = np.zeros_like(Y)
        b1, b2, eps = 0.9, 0.999, 1e-8
        Gtmp = Y @ Y.T
        np.fill_diagonal(Gtmp, -10.0)
        bval = float(Gtmp.max())
        bY = Y.copy()
        lb = (np.log(beta1) - np.log(beta0)) / max(steps - 1, 1)
        ll = (np.log(lr1) - np.log(lr0)) / max(steps - 1, 1)
        lbo = np.log(beta0); llo = np.log(lr0)
        b1t = 1.0; b2t = 1.0
        n = Y.shape[0]
        for t in range(1, steps + 1):
            beta = float(np.exp(lbo + lb * (t - 1)))
            lr = float(np.exp(llo + ll * (t - 1)))
            G = Y @ Y.T
            np.fill_diagonal(G, -10.0)
            gmax = float(G.max())
            if gmax < bval:
                bval = gmax
                bY = Y.copy()
            G -= gmax
            G *= beta
            np.exp(G, out=G)
            np.fill_diagonal(G, 0.0)
            s = float(G.sum())
            if s <= 0 or not np.isfinite(s):
                continue
            G /= s
            Gr = G @ Y
            dot = np.sum(Gr * Y, axis=1, keepdims=True)
            Gr -= Y * dot
            b1t *= b1; b2t *= b2
            M *= b1; M += (1.0 - b1) * Gr
            V *= b2; V += (1.0 - b2) * (Gr * Gr)
            Y -= lr * (M / (1.0 - b1t)) / (np.sqrt(V / (1.0 - b2t)) + eps)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn[rn == 0] = 1.0
            Y /= rn
            if (t % 50 == 0):
                if time.time() > deadline:
                    break
                # cheap intermediate best check
                Ge = Y @ Y.T
                np.fill_diagonal(Ge, -10.0)
                ve = float(Ge.max())
                if ve < bval:
                    bval = ve
                    bY = Y.copy()
        Gf = bY @ bY.T
        np.fill_diagonal(Gf, -10.0)
        vf = float(Gf.max())
        # bY already best-tracked; ensure value
        bval = min(bval, vf)
        return bY, bval

    # ---------- partial polish: move M only, F frozen ----------
    def polish_partial(F, M0, steps, beta0, beta1, lr0, lr1):
        M_ = M0.copy()
        m = M_.shape[0]
        nF = F.shape[0]
        MA = np.zeros_like(M_)
        VA = np.zeros_like(M_)
        b1, b2, eps = 0.9, 0.999, 1e-8
        def pmax(Mt):
            a = Mt @ F.T
            am = float(a.max()) if a.size else -2.0
            B = Mt @ Mt.T
            np.fill_diagonal(B, -2.0)
            bm = float(B.max()) if m > 1 else -2.0
            return am if am > bm else bm
        bval = pmax(M_)
        bM = M_.copy()
        lb = (np.log(beta1) - np.log(beta0)) / max(steps - 1, 1)
        ll = (np.log(lr1) - np.log(lr0)) / max(steps - 1, 1)
        lbo = np.log(beta0); llo = np.log(lr0)
        b1t = 1.0; b2t = 1.0
        for t in range(1, steps + 1):
            beta = float(np.exp(lbo + lb * (t - 1)))
            lr = float(np.exp(llo + ll * (t - 1)))
            Gmf = M_ @ F.T
            Gmm = M_ @ M_.T
            # gmax over involved pairs
            gmax = float(Gmf.max()) if Gmf.size else -2.0
            if m > 1:
                tmp = Gmm.copy()
                np.fill_diagonal(tmp, -2.0)
                gm2 = float(tmp.max())
                if gm2 > gmax:
                    gmax = gm2
            cur = gmax
            if cur < bval:
                bval = cur
                bM = M_.copy()
            # weights
            Wmf = np.exp(beta * (Gmf - gmax))
            Wmm = np.exp(beta * (Gmm - gmax))
            np.fill_diagonal(Wmm, 0.0)
            s = float(Wmf.sum() + Wmm.sum())
            if s <= 0 or not np.isfinite(s):
                continue
            Gr = (Wmf @ F + Wmm @ M_) / s
            dot = np.sum(Gr * M_, axis=1, keepdims=True)
            Gr -= M_ * dot
            b1t *= b1; b2t *= b2
            MA *= b1; MA += (1.0 - b1) * Gr
            VA *= b2; VA += (1.0 - b2) * (Gr * Gr)
            M_ -= lr * (MA / (1.0 - b1t)) / (np.sqrt(VA / (1.0 - b2t)) + eps)
            rn = np.linalg.norm(M_, axis=1, keepdims=True)
            rn[rn == 0] = 1.0
            M_ /= rn
            if (t % 50 == 0):
                if time.time() > deadline:
                    break
                ve = pmax(M_)
                if ve < bval:
                    bval = ve
                    bM = M_.copy()
        ve = pmax(bM)
        # also check last
        vl = pmax(M_)
        if vl < ve:
            bM = M_.copy()
            bval = vl
        else:
            bval = ve
        return bM, bval

    # order centres by contact degree
    def contact_order(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        deg = np.sum(G > (m - 0.006), axis=1)
        rm = G.max(axis=1)
        order = np.argsort(-deg - 1e-6 * rm)
        return order, G, m

    order, G0, m0 = contact_order(x_best)

    n_attempt = 0
    max_attempts = 60
    # cycle clusters
    oi = 0
    while n_attempt < max_attempts:
        if time.time() > deadline - 8.0:
            break
        # refresh order every 10 attempts from current best
        if n_attempt % 10 == 0:
            order, Gc, mc = contact_order(x_best)
            oi = 0
        center = int(order[oi % 59])
        oi += 1
        # cluster size 6..9
        csize = 6 + (n_attempt % 4)  # 6,7,8,9
        # neighbours: top contacts by cosine
        row = (x_best @ x_best.T)[center].copy()
        row[center] = -2.0
        nbrs = np.argsort(-row)[:csize - 1]
        removed = np.array([center] + nbrs.tolist(), dtype=int)
        frozen_idx = np.array([i for i in range(59) if i not in set(removed.tolist())], dtype=int)
        F = x_best[frozen_idx].copy()
        msz = len(removed)
        # sequential greedy reinsertion using pool
        Curr = F.copy()  # (Nf,22)
        Minits = []
        for j in range(msz):
            # one matmul
            S = Pool @ Curr.T  # (POOL_N, Nf+j)
            rmax = S.max(axis=1)
            # best 100
            part = np.argpartition(rmax, 100)[:100]
            vals100 = rmax[part]
            srt = np.argsort(vals100)
            # diversify: pick k-th best among top 10
            k = (n_attempt + j) % 10
            kk = srt[k] if k < len(srt) else srt[0]
            best_p = part[kk]
            Minits.append(Pool[best_p].copy())
            Curr = np.vstack([Curr, Pool[best_p][None, :]])
            # time check every vacancy is cheap; check every 50 iterations required -> checked in optim loops;
            # outer check occasionally
        M0 = np.stack(Minits, axis=0)
        # locally optimise only reinserted points: 300-500 annealed steps (use 400)
        if time.time() > deadline - 6.0:
            break
        Mopt, _ = polish_partial(F, M0, steps=400, beta0=60.0, beta1=600.0, lr0=0.02, lr1=0.002)
        if time.time() > deadline - 5.0:
            break
        Xcand = np.empty_like(x_best)
        Xcand[frozen_idx] = F
        Xcand[removed] = Mopt
        Xcand = _normr(Xcand)
        # unfreeze: short 1500-step polish
        Ypol, vpol = polish_full(Xcand, steps=1500, beta0=80.0, beta1=600.0, lr0=0.006, lr1=0.0008)
        n_attempt += 1
        if vpol < best_val - 1e-12:
            best_val = float(vpol)
            x_best = _normr(Ypol.copy())
            # refresh contact order immediately
            order, Gc, mc = contact_order(x_best)
            oi = 0
        # time check every 50 iterations done inside polish; outer check each attempt
        if n_attempt % 1 == 0 and time.time() > deadline - 4.0:
            break

    # final quick polish if time remains
    if time.time() < deadline - 4.0:
        try:
            Yf, vf = polish_full(x_best, steps=1200, beta0=150.0, beta1=900.0, lr0=0.004, lr1=0.0005)
            if vf < best_val:
                best_val = vf
                x_best = Yf.copy()
        except Exception:
            pass

    x_best = _normr(x_best)
    return x_best
