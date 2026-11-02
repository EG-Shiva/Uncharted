import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(7)
    n0, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n0, d):
                a = _norm(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((n0, d))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    gbest = cands[bi].copy()
    gval = float(vals[bi])

    # ---------- core operators ----------
    def adam_lse(X, tau, steps, lr0):
        m = np.zeros_like(X)
        v = np.zeros_like(X)
        cnt = 0
        for s in range(steps):
            if (s & 63) == 0 and time.time() - t0 > LIM - 10:
                break
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            gmax = float(G.max())
            E = np.exp((G - gmax) / tau)
            np.fill_diagonal(E, 0.0)
            S = float(E.sum())
            if not np.finite(S) or S <= 0:
                continue
            Gr = (E @ X) / S
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dot * X
            m[:] = 0.9 * m + 0.1 * Gt
            v[:] = 0.999 * v + 0.001 * (Gt * Gt)
            cnt += 1
            c1 = 1.0 - 0.9 ** cnt
            c2 = 1.0 - 0.999 ** cnt
            mh = m / c1
            vh = v / c2
            lr = lr0 * (1.0 - 0.5 * s / max(1, steps))
            X -= lr * mh / (np.sqrt(vh) + 1e-8)
            # renormalize
            nr = np.linalg.norm(X, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            X /= nr
        return X

    def anneal(X, level="mid"):
        # level controls length
        if level == "short":
            sched = [(0.006, 250, 0.02), (0.002, 300, 0.012)]
        elif level == "mid":
            sched = [(0.010, 350, 0.025), (0.004, 450, 0.015), (0.0015, 450, 0.008)]
        elif level == "long":
            sched = [(0.012, 500, 0.03), (0.005, 700, 0.015), (0.002, 700, 0.008), (0.0008, 500, 0.004)]
        else:
            sched = [(0.006, 250, 0.02), (0.002, 300, 0.012)]
        for tau, steps, lr in sched:
            adam_lse(X, tau, steps, lr)
            if time.time() - t0 > LIM - 10:
                break
        return X

    def minimax(X, steps, lr0):
        cur = _max_cos(X)
        lr = lr0
        nn = X.shape[0]
        for it in range(steps):
            if (it & 15) == 0 and time.time() - t0 > LIM - 8:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            delta = max(3.0 * lr, 5e-7)
            mask = G > (gmax - delta)
            A = mask.astype(np.float64)
            Gr = A @ X
            dd = (Gr * X).sum(axis=1, keepdims=True)
            Gt = Gr - dd * X
            nr = np.linalg.norm(Gt, axis=1, keepdims=True)
            act = nr[:, 0] > 1e-14
            if not np.any(act):
                break
            step = np.zeros_like(Gt)
            step[act] = Gt[act] / (nr[act] + 1e-12) * lr
            Xn = _norm(X - step)
            nv = _max_cos(Xn)
            if nv < cur:
                X[:] = Xn
                cur = nv
                lr *= 1.08
            else:
                lr *= 0.5
                if lr < 1e-12:
                    break
        return X, cur

    def find_hole(X, n_cand=6000, refine=150):
        nn = X.shape[0]
        cand = rng.standard_normal((n_cand, d))
        cand = _norm(cand)
        S = cand @ X.T
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        c = cand[j].copy()
        # refine: minimize LSE of dots
        tau = 0.005
        for _ in range(refine):
            s = X @ c
            m = float(s.max())
            e = np.exp((s - m) / tau)
            w = e / (e.sum() + 1e-300)
            g = w @ X  # direction to avoid
            # tangent component
            gt = g - float(g @ c) * c
            ng = float(np.linalg.norm(gt))
            if ng < 1e-12:
                break
            gt /= ng
            c = c - 0.05 * gt
            nnr = float(np.linalg.norm(c))
            if nnr == 0 or not np.isfinite(nnr):
                c = _norm(rng.standard_normal((1, d)))[0]
                break
            c /= nnr
        return c

    def worst_order(X, k):
        nn = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = G.max(axis=1)
        return np.argsort(-mx)[:k]

    def prune_to_74(Xbig):
        # Xbig has >74 rows, delete worst to 74 using combined score
        while Xbig.shape[0] > 74:
            nn = Xbig.shape[0]
            G = Xbig @ Xbig.T
            np.fill_diagonal(G, -2.0)
            M = float(G.max())
            deg = (G > (M - 1e-7)).sum(axis=1)
            # sum top-5
            part = np.sort(G, axis=1)[:, -5:].sum(axis=1)
            mx = G.max(axis=1)
            # lexicographic worst: larger mx, larger deg, larger part
            # compute rank score
            order = np.lexsort((part, deg, mx))  # ascending best first
            # delete last one (worst)
            worst = int(order[-1])
            Xbig = np.delete(Xbig, worst, axis=0)
        return Xbig

    def squeeze_pair(X, budget=0.6):
        start = time.time()
        cur = _max_cos(X)
        nn = X.shape[0]
        step = 2e-5
        while time.time() - start < budget and time.time() - t0 < LIM - 4:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // nn; j = k % nn
            c = float(G[i, j])
            xi = X[i]; xj = X[j]
            gi = xj - c * xi
            gj = xi - c * xj
            ni = float(np.linalg.norm(gi)); nj = float(np.linalg.norm(gj))
            if ni < 1e-12 or nj < 1e-12:
                break
            gi /= ni; gj /= nj
            improved = False
            for mode in (0, 1, 2):
                Y = X.copy()
                if mode == 0:
                    Y[i] = xi - step * gi
                    Y[j] = xj - step * gj
                elif mode == 1:
                    Y[i] = xi - step * 2.0 * gi
                else:
                    Y[j] = xj - step * 2.0 * gj
                Y[i] /= np.linalg.norm(Y[i]); Y[j] /= np.linalg.norm(Y[j])
                # quick check only if promising: compute max involving i,j first?
                v = _max_cos(Y)
                if v < cur:
                    X[:] = Y
                    cur = v
                    improved = True
                    break
            if improved:
                step = min(step * 1.2, 5e-5)
            else:
                step *= 0.5
                if step < 1e-11:
                    break
        return X, cur

    # initial repolish of incumbent a bit
    gbest = anneal(gbest.copy(), level="mid")
    gbest, gv = minimax(gbest, 300, 3e-4)
    gbest, gv = squeeze_pair(gbest, budget=0.5)
    gv = _max_cos(gbest)
    if gv < gval:
        gval = gv
    else:
        # keep original best if anneal hurt (should not)
        pass

    it = 0
    # alternate overshoot / undershoot until time
    while time.time() - t0 < LIM - 22:
        it += 1
        if it % 2 == 1:
            # ---- overshoot: 74 -> 76..78 -> 74 ----
            k = 2 + (it % 3)  # cycles 3,2,4? gives 2..4
            if k < 2: k = 2
            if k > 4: k = 4
            Xb = gbest.copy()
            extras = []
            for _ in range(k):
                # find hole w.r.t current + extras
                tmp = Xb if not extras else np.vstack([Xb] + extras)
                h = find_hole(tmp, n_cand=5000, refine=120)
                extras.append(h.reshape(1, d))
            Xbig = np.vstack([Xb] + extras)
            Xbig = _norm(Xbig)
            # anneal harder problem
            anneal(Xbig, level="mid" if Xbig.shape[0] <= 77 else "short")
            # extra pressure: short minimax on big
            try:
                Xbig, _ = minimax(Xbig, 150, 2e-4)
            except Exception:
                pass
            Xnew = prune_to_74(Xbig)
            anneal(Xnew, level="short")
            Xnew, v1 = minimax(Xnew, 250, 2e-4)
            Xnew, v1 = squeeze_pair(Xnew, budget=0.3)
            v1 = _max_cos(Xnew)
            if v1 < gval:
                gval = v1
                gbest = Xnew.copy()
        else:
            # ---- undershoot: 74 -> 70 -> 74 staged growth ----
            drop = worst_order(gbest, 4)
            mask = np.ones(74, dtype=bool)
            mask[drop] = False
            Xsmall = gbest[mask].copy()  # 70
            # jitter small a bit to reorganise
            Xsmall = Xsmall + 0.005 * rng.standard_normal(Xsmall.shape)
            Xsmall = _norm(Xsmall)
            anneal(Xsmall, level="short")
            # reinsert one at a time with polish between
            for _ in range(4):
                h = find_hole(Xsmall, n_cand=5000, refine=120)
                Xsmall = np.vstack([Xsmall, h.reshape(1, d)])
                Xsmall = _norm(Xsmall)
                anneal(Xsmall, level="short")
                Xsmall, _ = minimax(Xsmall, 120, 2e-4)
                if time.time() - t0 > LIM - 22:
                    break
            if Xsmall.shape[0] != 74:
                continue
            Xsmall, v2 = squeeze_pair(Xsmall, budget=0.3)
            v2 = _max_cos(Xsmall)
            if v2 < gval:
                gval = v2
                gbest = Xsmall.copy()
        # occasional random-kick basin hop to add diversity
        if it % 6 == 0 and time.time() - t0 < LIM - 25:
            Xr = gbest + 0.02 * rng.standard_normal(gbest.shape)
            Xr = _norm(Xr)
            anneal(Xr, level="short")
            Xr, vr = minimax(Xr, 200, 2e-4)
            vr = _max_cos(Xr)
            if vr < gval:
                gval = vr
                gbest = Xr.copy()

    # ---------- final intensive polish ----------
    # long anneal + minimax + squeeze
    if time.time() - t0 < LIM - 4:
        try:
            anneal(gbest, level="long")
        except Exception:
            pass
    if time.time() - t0 < LIM - 3:
        try:
            gbest, gval2 = minimax(gbest, 600, 2e-4)
            gval = float(min(gval, _max_cos(gbest)))
        except Exception:
            pass
    if time.time() - t0 < LIM - 2:
        try:
            gbest, _ = squeeze_pair(gbest, budget=max(0.5, LIM - 2 - (time.time() - t0)))
        except Exception:
            pass

    # tiny shrink guard (no-op, keeps strictly feasible) and normalize
    gbest = _norm(gbest)
    return gbest
