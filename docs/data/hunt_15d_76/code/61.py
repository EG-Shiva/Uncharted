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

def _adam_lse(X, steps, b0, b1, lr0, lr1, deadline):
    n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        if (k & 63) == 0 and time.time() > deadline:
            break
        f = k / max(1, steps - 1)
        beta = b0 * ((b1 / b0) ** f)
        T = 1.0 / beta
        lr = lr0 * ((lr1 / lr0) ** f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp((G - gm) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        m = b1a * m + (1 - b1a) * Gt
        v = b2a * v + (1 - b2a) * (Gt * Gt)
        bc1 = 1 - b1a ** (k + 1)
        bc2 = 1 - b2a ** (k + 1)
        mh = m / bc1
        vh = v / bc2
        step = lr * mh / (np.sqrt(vh) + eps)
        X = X - step
        X = _norm(X)
    return X

def _refine_hole(v, Y, iters=100, T0=0.008, lr0=0.08):
    v = v / (np.linalg.norm(v) + 1e-12)
    for k in range(iters):
        f = k / max(1, iters - 1)
        Tk = T0 * (0.2 ** f)
        lrk = lr0 * (0.3 ** f)
        s = Y @ v
        sm = float(s.max())
        w = np.exp((s - sm) / Tk)
        sw = float(w.sum())
        if sw <= 0 or not np.isfinite(sw):
            break
        w /= sw
        g = w @ Y
        gv = float(g @ v)
        g = g - gv * v
        v = v - lrk * g
        nv = float(np.linalg.norm(v))
        if nv == 0 or not np.isfinite(nv):
            break
        v = v / nv
    return v

def _local_adam(cur, subset, steps, beta, lr, deadline):
    # optimize only rows in subset, freeze rest
    n, d = cur.shape
    sub = np.asarray(sorted(list(subset)), dtype=int)
    if len(sub) == 0:
        return cur
    m = np.zeros((len(sub), d))
    v = np.zeros((len(sub), d))
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    T = 1.0 / beta
    for k in range(steps):
        if (k & 31) == 0 and time.time() > deadline:
            break
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        gm = float(G.max())
        E = np.exp((G - gm) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (E @ cur) / S
        Gt = Gr[sub] - np.sum(Gr[sub] * cur[sub], axis=1, keepdims=True) * cur[sub]
        m = b1a * m + (1 - b1a) * Gt
        v = b2a * v + (1 - b2a) * (Gt * Gt)
        bc1 = 1 - b1a ** (k + 1)
        bc2 = 1 - b2a ** (k + 1)
        step = lr * (m / bc1) / (np.sqrt(v / bc2) + eps)
        cur[sub] = cur[sub] - step
        # renormalize subset rows
        nn = np.linalg.norm(cur[sub], axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        cur[sub] = cur[sub] / nn
    return cur

def _lp_polish(X, trust=0.004, maxc=1500):
    # capped LP polish: minimise t s.t. linearized pairs <= t, |dx|<=trust, x.dx=0
    try:
        from scipy.optimize import linprog
    except Exception:
        return X, False
    n, d = X.shape
    try:
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        iu0, iu1 = np.triu_indices(n, 1)
        s = G[iu0, iu1]
        order = np.argsort(-s)
        nc = int(min(maxc, len(s)))
        sel = order[:nc]
        si = iu0[sel]
        sj = iu1[sel]
        b_ub = -s[sel]
        p = nc
        Nv = n * d + 1
        # dense is ok for n<=76, p<=1500
        Aub = np.zeros((p, Nv), dtype=float)
        # fill rows: vectorized loop
        for r in range(p):
            i = int(si[r]); j = int(sj[r])
            Aub[r, i * d:(i + 1) * d] = X[j]
            Aub[r, j * d:(j + 1) * d] = X[i]
            Aub[r, Nv - 1] = -1.0
        c = np.zeros(Nv, dtype=float)
        c[Nv - 1] = 1.0
        Aeq = np.zeros((n, Nv), dtype=float)
        for i in range(n):
            Aeq[i, i * d:(i + 1) * d] = X[i]
        beq = np.zeros(n, dtype=float)
        bounds = [(-trust, trust)] * (n * d) + [(-1.0, 1.0)]
        res = linprog(c, A_ub=Aub, b_ub=b_ub, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')
        if not res.success:
            return X, False
        y = res.x
        dx = y[:n * d].reshape(n, d)
        # apply
        Xn = _norm(X + dx)
        return Xn, True
    except Exception:
        return X, False

def solve():
    import time
    import numpy as np
    t0 = time.time()
    deadline = t0 + 150.0
    hard_end = t0 + 172.0

    def normed(a):
        a = np.asarray(a, dtype=float)
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        return a

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    fb = None
    fbv = 1e9
    if xb is not None and xb.shape == (76, 15):
        try:
            xb = normed(xb)
            fb = xb.copy(); fbv = _maxcos(fb)
        except Exception:
            pass
    if xr is not None and xr.shape == (76, 15):
        try:
            xr = normed(xr)
            vv = _maxcos(xr)
            if vv < fbv:
                fb = xr.copy(); fbv = vv
        except Exception:
            pass
    if fb is None:
        rng0 = np.random.default_rng(0)
        fb = normed(rng0.normal(size=(76, 15)))
        fbv = _maxcos(fb)
    bestX = fb.copy()
    bestV = float(fbv)

    # identify 11 zero-contact (largest slack) points
    try:
        G0 = bestX @ bestX.T
        np.fill_diagonal(G0, -2.0)
        M0 = float(G0.max())
        rmax = G0.max(axis=1)
        slack = M0 - rmax
        ratt = np.argsort(slack)[-11:]
        ratt = [int(x) for x in ratt]
    except Exception:
        ratt = list(range(65, 76))
    keep = [i for i in range(76) if i not in set(ratt)]
    # 65-point core
    Xc = bestX[keep].copy()
    Xc = _norm(Xc)

    rng = np.random.default_rng(777)
    # annealed log-sum-exp Adam beta 50->600, 3000 steps
    try:
        Xc = _adam_lse(Xc, steps=3000, b0=50.0, b1=600.0, lr0=0.02, lr1=0.003, deadline=deadline - 60.0)
    except Exception:
        pass
    try:
        core_opt = _maxcos(Xc)
    except Exception:
        core_opt = bestV
    # if core not below record, push a bit more
    if core_opt > 0.16 and time.time() < deadline - 55:
        try:
            Xc = _adam_lse(Xc, steps=800, b0=200.0, b1=800.0, lr0=0.008, lr1=0.001, deadline=deadline - 50.0)
            core_opt = _maxcos(Xc)
        except Exception:
            pass

    cur = Xc.copy()
    # slowly raised threshold starting at core optimum
    final_cap = 0.172
    # reinsert 11 points one at a time
    for k in range(11):
        if time.time() > deadline - 20.0:
            break
        if cur.shape[0] >= 76:
            break
        frac = (cur.shape[0] - 65 + 1) / 11.0
        thresh = core_opt + (final_cap - core_opt) * frac + 0.01
        Y = cur
        # sample 20000 random hole candidates
        NS = 20000
        best_scores = []
        # chunked
        chunk = 5000
        cand_best_v = None
        cand_best_s = 1e9
        # keep top-3 distinct candidates
        topV = [None, None, None]
        topS = [1e9, 1e9, 1e9]
        for s0 in range(0, NS, chunk):
            if time.time() > deadline - 15.0:
                break
            cs = min(chunk, NS - s0)
            C = rng.normal(size=(cs, 15))
            C = _norm(C)
            S = C @ Y.T
            sc = S.max(axis=1)
            # update global best and top3
            for idx in range(cs):
                val = float(sc[idx])
                if val < topS[2]:
                    # insert sorted
                    if val < topS[0]:
                        topS[2] = topS[1]; topV[2] = topV[1]
                        topS[1] = topS[0]; topV[1] = topV[0]
                        topS[0] = val; topV[0] = C[idx].copy()
                    elif val < topS[1]:
                        topS[2] = topS[1]; topV[2] = topV[1]
                        topS[1] = val; topV[1] = C[idx].copy()
                    else:
                        topS[2] = val; topV[2] = C[idx].copy()
        accepted = False
        # try top3 holes in order
        for ti in range(3):
            if time.time() > deadline - 12.0:
                break
            v0 = topV[ti]
            if v0 is None:
                continue
            try:
                v0 = _refine_hole(v0, Y, iters=100, T0=0.008, lr0=0.08)
            except Exception:
                pass
            trial = np.vstack([Y, v0.reshape(1, 15)])
            trial = _norm(trial)
            # locally re-optimize inserted + neighbours cos>0.05, freeze rest
            try:
                s = trial[-1] @ trial[:-1].T
                nbr = set([trial.shape[0] - 1])
                for j, val in enumerate(s):
                    if float(val) > 0.05:
                        nbr.add(int(j))
                trial = _local_adam(trial, nbr, steps=250, beta=250.0, lr=0.01, deadline=deadline - 8.0)
            except Exception:
                pass
            # capped LP polish (at most 1500 constraints, ~10s)
            if time.time() < deadline - 10.0:
                try:
                    # use fewer constraints for speed but <=1500
                    Tn, ok = _lp_polish(trial, trust=0.004, maxc=1200)
                    if ok:
                        # accept LP only if not worse by much
                        v_old = _maxcos(trial)
                        v_new = _maxcos(Tn)
                        if v_new <= v_old + 0.002:
                            trial = Tn
                            # second shrink step
                            if time.time() < deadline - 9.0:
                                Tn2, ok2 = _lp_polish(trial, trust=0.002, maxc=1200)
                                if ok2 and _maxcos(Tn2) <= _maxcos(trial):
                                    trial = Tn2
                except Exception:
                    pass
            try:
                vm = _maxcos(trial)
            except Exception:
                continue
            if vm <= thresh + 1e-12:
                cur = trial
                accepted = True
                break
            else:
                # remember best of the 3 if none under thresh
                if ti == 0:
                    fallback_trial = trial.copy()
                    fallback_v = vm
                else:
                    if vm < fallback_v:
                        fallback_trial = trial.copy()
                        fallback_v = vm
        if not accepted:
            # accept best fallback to keep completing (still progress to full set)
            try:
                cur = fallback_trial.copy()
            except Exception:
                # single best hole path
                v0 = topV[0]
                if v0 is None:
                    break
                v0 = _refine_hole(v0, Y, iters=100)
                cur = _norm(np.vstack([Y, v0.reshape(1, 15)]))

    # finish: if complete, short global polish
    if cur.shape == (76, 15):
        try:
            cur = _norm(cur)
            vc = _maxcos(cur)
        except Exception:
            vc = 1e9
        # global anneal short
        if time.time() < hard_end - 12.0:
            try:
                cur2 = _adam_lse(cur.copy(), steps=1200, b0=150.0, b1=800.0, lr0=0.008, lr1=0.001, deadline=hard_end - 6.0)
                v2 = _maxcos(cur2)
                if v2 < vc:
                    cur = cur2; vc = v2
            except Exception:
                pass
        # final LP attempt if time
        if time.time() < hard_end - 6.0:
            try:
                Tn, ok = _lp_polish(cur, trust=0.003, maxc=1500)
                if ok and _maxcos(Tn) < vc:
                    cur = Tn; vc = _maxcos(cur)
            except Exception:
                pass
        if vc < bestV:
            bestX = cur.copy(); bestV = vc
        # else keep best fallback; but if cur is valid and bestX is fb, return better of the two
        # ensure we return the best complete set found (cur vs fb)
        if bestV > _maxcos(fb) and _maxcos(cur) < bestV:
            bestX = cur.copy()
        # if our reconstruction failed to beat fb, return fb (fallback)
        try:
            if _maxcos(cur) < _maxcos(bestX):
                bestX = cur.copy()
        except Exception:
            pass
    else:
        # incomplete: fallback to best
        bestX = fb.copy()

    try:
        bestX = np.asarray(bestX, dtype=float)
        if bestX.shape != (76, 15):
            bestX = fb.copy()
        bestX = _norm(bestX)
        if not np.all(np.isfinite(bestX)):
            bestX = fb.copy()
    except Exception:
        bestX = fb.copy()
    return bestX
