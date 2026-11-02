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

def _anneal_sgd(X, steps, T0, T1, lr0, lr1, deadline):
    n, d = X.shape
    try:
        lT = np.log(T1 / T0)
        ll = np.log(lr1 / lr0)
    except Exception:
        return X
    for k in range(steps):
        if (k & 127) == 0 and time.time() > deadline:
            break
        f = k / max(1, steps - 1)
        T = T0 * np.exp(lT * f)
        lr = lr0 * np.exp(ll * f)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if S <= 0 or not np.isfinite(S):
            break
        E /= S
        Gr = E @ X
        # tangential projection
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        X = X - lr * Gt
        # renormalize
        nn = np.linalg.norm(X, axis=1, keepdims=True)
        nn[nn == 0] = 1.0
        X = X / nn
    return X

def _refine_hole(v, Y, iters=200, T=0.008, lr=0.08):
    v = v / (np.linalg.norm(v) + 1e-12)
    for k in range(iters):
        f = k / max(1, iters - 1)
        Tk = T * (0.2 ** f)
        lrk = lr * (0.2 ** f)
        s = Y @ v
        m = float(s.max())
        w = np.exp((s - m) / Tk)
        sw = float(w.sum())
        if sw <= 0 or not np.isfinite(sw):
            break
        w /= sw
        g = w @ Y
        gv = float(g @ v)
        g = g - gv * v
        ng = float(np.linalg.norm(g))
        if ng > 1e-12:
            # scale to avoid huge steps
            sc = min(1.0, ng * 5.0) / (ng + 1e-12)
            g = g * sc * ng  # == normalized*min(1,ng*5)
            # simpler: g = g / (ng+eps) * min(1,ng*5)
            # recompute directly for stability
            pass
        v = v - lrk * g
        nv = float(np.linalg.norm(v))
        if nv == 0 or not np.isfinite(nv):
            break
        v = v / nv
    return v

def solve():
    import time
    import numpy as np
    t0 = time.time()
    deadline = t0 + 172.0
    final_reserve = 10.0

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
            v = _maxcos(xb)
            fb = xb.copy()
            fbv = v
        except Exception:
            pass
    if xr is not None and xr.shape == (76, 15):
        try:
            xr = normed(xr)
            v = _maxcos(xr)
            if v < fbv:
                fb = xr.copy()
                fbv = v
        except Exception:
            pass
    if fb is None:
        rng0 = np.random.default_rng(0)
        fb = normed(rng0.normal(size=(76, 15)))
        fbv = _maxcos(fb)

    bestX = fb.copy()
    bestV = float(fbv)
    baseX = fb.copy()

    # analyse base to get rattlers + hubs
    try:
        G0 = baseX @ baseX.T
        np.fill_diagonal(G0, -2.0)
        M0 = float(G0.max())
        row_max = G0.max(axis=1)
        slack = M0 - row_max
        rattler_order = np.argsort(slack)  # ascending small slack first
        rattlers11 = rattler_order[-11:]  # largest slack
        # near-contact counts for hubs
        thr = M0 - 0.015
        counts = (G0 > thr).sum(axis=1)
        hub_order = np.argsort(-counts)
    except Exception:
        rattlers11 = np.arange(11)
        hub_order = np.arange(76)

    removal_sizes = [16, 20, 24, 28]
    # time-boxed runs
    for ri, R in enumerate(removal_sizes):
        if time.time() > deadline - final_reserve - 8.0:
            break
        # per-run deadline: divide remaining (minus reserve) by remaining runs
        remain = (deadline - final_reserve) - time.time()
        nleft = len(removal_sizes) - ri
        run_budget = remain / nleft
        run_budget = min(run_budget, 34.0)
        run_deadline = time.time() + run_budget
        if run_budget < 6.0:
            break
        try:
            # build removal set: 11 rattlers + hubs to fill R
            rem = set(int(x) for x in rattlers11.tolist())
            for h in hub_order.tolist():
                if len(rem) >= R:
                    break
                rem.add(int(h))
            # if still short (overlap), fill with highest slack not in set
            if len(rem) < R:
                for idx in rattler_order[::-1].tolist():
                    if len(rem) >= R:
                        break
                    rem.add(int(idx))
            rem_list = sorted(list(rem))
            keep = [i for i in range(76) if i not in rem]
            # shuffle deletion order deterministically
            rngR = np.random.default_rng(1000 + R)
            del_order = rem_list.copy()
            rngR.shuffle(del_order)
            kcore = len(keep)
            Xc = baseX[keep].copy()
            Xc = _norm(Xc)
            # aggressive core anneal: aim <0.15
            # two stages
            inner_dl = run_deadline - (R * 1.1) - 2.0
            if inner_dl < time.time() + 1.0:
                inner_dl = time.time() + 1.0
            # stage 1
            Xc = _anneal_sgd(Xc, steps=2200, T0=0.018, T1=0.0012, lr0=5.0, lr1=0.20, deadline=min(inner_dl, run_deadline))
            if time.time() < inner_dl - 0.5:
                # extra push if still high
                try:
                    Gc = Xc @ Xc.T
                    np.fill_diagonal(Gc, -2.0)
                    mc = float(Gc.max())
                except Exception:
                    mc = 1.0
                if mc > 0.15 and time.time() < inner_dl - 1.0:
                    Xc = _anneal_sgd(Xc, steps=1200, T0=0.010, T1=0.0006, lr0=3.0, lr1=0.10, deadline=min(inner_dl, run_deadline))
            cur = Xc.copy()
            ok = True
            for ins_idx, _orig in enumerate(del_order):
                if time.time() > run_deadline:
                    ok = False
                    break
                Y = cur
                m = Y.shape[0]
                # sample >=50000 candidates
                NS = 50000
                # chunked sampling to bound memory
                chunk = 10000
                best_score = 1e9
                best_v = None
                # use rngR for candidates
                for s0 in range(0, NS, chunk):
                    cs = min(chunk, NS - s0)
                    C = rngR.normal(size=(cs, 15))
                    C = _norm(C)
                    # scores = max_j C.Y
                    S = C @ Y.T
                    sc = S.max(axis=1)
                    bi = int(np.argmin(sc))
                    if float(sc[bi]) < best_score:
                        best_score = float(sc[bi])
                        best_v = C[bi].copy()
                if best_v is None:
                    ok = False
                    break
                # local hole optimization 200 steps
                best_v = _refine_hole(best_v, Y, iters=200, T=0.008, lr=0.08)
                # insert
                cur = np.vstack([cur, best_v.reshape(1, 15)])
                cur = _norm(cur)
                # re-polish growing code 300-500 steps (use 350)
                polish_dl = run_deadline
                # leave estimated time for remaining insertions: ~0.9s each
                # if very late, reduce steps
                nrem = len(del_order) - ins_idx - 1
                # quick adapt: if little time, fewer steps
                tleft = polish_dl - time.time()
                if tleft < (nrem * 0.8 + 1.0):
                    steps_p = 150
                else:
                    steps_p = 350
                cur = _anneal_sgd(cur, steps=steps_p, T0=0.005, T1=0.0008, lr0=1.0, lr1=0.10, deadline=polish_dl)
                if time.time() > run_deadline:
                    # if this was last insertion ok else abort?
                    if ins_idx < len(del_order) - 1:
                        # check if we still can continue; if exceeded, mark incomplete
                        pass
            if not ok:
                continue
            if cur.shape != (76, 15):
                continue
            cur = _norm(cur)
            vcur = _maxcos(cur)
            if vcur < bestV - 1e-12:
                bestV = float(vcur)
                bestX = cur.copy()
                baseX = cur.copy()
                # refresh rattlers/hubs for next R? keep original base? update to new best
                try:
                    G0 = baseX @ baseX.T
                    np.fill_diagonal(G0, -2.0)
                    M0 = float(G0.max())
                    row_max = G0.max(axis=1)
                    slack = M0 - row_max
                    rattler_order = np.argsort(slack)
                    rattlers11 = rattler_order[-11:]
                    thr = M0 - 0.015
                    counts = (G0 > thr).sum(axis=1)
                    hub_order = np.argsort(-counts)
                except Exception:
                    pass
        except Exception:
            continue

    # final short polish of incumbent (LP-free, fast) in remaining time
    try:
        if time.time() < deadline - 1.5:
            dl = deadline - 1.0
            # gentle anneal
            bestX = _anneal_sgd(bestX, steps=1800, T0=0.0025, T1=0.0004, lr0=0.6, lr1=0.05, deadline=dl)
            v = _maxcos(bestX)
            if v < bestV:
                bestV = float(v)
            else:
                # keep best (anneal may have slightly worsened if cut off mid-schedule;
                # re-evaluate: if worse by tiny, keep annealed only if better)
                # _anneal modifies in place copy? we already overwrote bestX.
                # To be safe, compare and revert to baseX if worse and baseX better
                if v > bestV + 1e-9:
                    # revert to stored base if it was better
                    try:
                        vb = _maxcos(baseX)
                        if vb <= v:
                            # choose the better of baseX and fb
                            if _maxcos(fb) < min(v, vb):
                                bestX = fb.copy()
                            else:
                                bestX = baseX.copy() if vb <= v else bestX
                        # else keep annealed (it is better than base)
                    except Exception:
                        pass
    except Exception:
        pass

    # ensure valid output
    try:
        bestX = np.asarray(bestX, dtype=float)
        if bestX.shape != (76, 15):
            bestX = normed(fb)
        bestX = _norm(bestX)
        # guard NaN
        if not np.all(np.isfinite(bestX)):
            bestX = normed(fb)
    except Exception:
        bestX = normed(fb)
    return bestX
