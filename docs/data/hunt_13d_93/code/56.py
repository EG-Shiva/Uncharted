import time
import numpy as np

def solve():
    t0 = time.time()
    BUDGET = 170.0
    n, d = 93, 13
    rng = np.random.default_rng(42)

    def _norm(X):
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return X / nr

    def _maxcos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    def _rowmax(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return np.max(G, axis=1)

    iu0, iu1 = np.triu_indices(n, 1)
    M = len(iu0)

    # load baseline
    base = None
    base_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            a = _norm(a)
            v = _maxcos(a)
            if v < base_val:
                base_val = v
                base = a.copy()
        except Exception:
            pass
    if base is None:
        base = _norm(rng.standard_normal((n, d)))
        base_val = _maxcos(base)

    g_best = base.copy()
    g_val = float(base_val)

    # ---------- annealed LSE Adam ----------
    def adam_lse(X0, steps, tau0, tau1, lr0, lr1):
        X = _norm(X0.copy())
        m = np.zeros_like(X)
        vv = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        bv = _maxcos(X)
        bX = X.copy()
        ltau = np.log(tau0), np.log(tau1)
        llr = np.log(lr0), np.log(lr1)
        for s in range(steps):
            if time.time() - t0 > BUDGET - 18:
                break
            f = s / max(1, steps - 1)
            tau = float(np.exp(ltau[0] + f * (ltau[1] - ltau[0])))
            lr = float(np.exp(llr[0] + f * (llr[1] - llr[0])))
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            c = G[iu0, iu1]
            cm = float(np.max(c))
            # weights
            E = np.exp((c - cm) / tau)
            S = float(np.sum(E))
            if not np.finite(S) or S <= 0:
                continue
            W = E / S
            grad = np.zeros_like(X)
            np.add.at(grad, iu0, W[:, None] * X[iu1])
            np.add.at(grad, iu1, W[:, None] * X[iu0])
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            if not np.all(np.finite(gt)):
                continue
            tt += 1
            m[:] = b1 * m + (1 - b1) * gt
            vv[:] = b2 * vv + (1 - b2) * (gt * gt)
            mh = m / (1 - b1 ** tt)
            vh = vv / (1 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            step -= np.einsum('ij,ij->i', step, X)[:, None] * X
            X = _norm(X - step)
            # track infrequently to save time
            if (s & 31) == 31 or s == steps - 1:
                cur = _maxcos(X)
                if cur < bv:
                    bv = cur
                    bX = X.copy()
        return bX, bv

    def hard_topk(X0, iters, K, step0):
        X = _norm(X0.copy())
        bv = _maxcos(X)
        bX = X.copy()
        step = float(step0)
        grad = np.empty_like(X)
        for it in range(iters):
            if time.time() - t0 > BUDGET - 16:
                break
            G = X @ X.T
            c = G[iu0, iu1]
            Kk = int(min(K, M))
            if Kk < M:
                idx = np.argpartition(-c, Kk)[:Kk]
            else:
                idx = np.arange(M)
            I = iu0[idx]; J = iu1[idx]
            grad.fill(0.0)
            np.add.at(grad, I, X[J])
            np.add.at(grad, J, X[I])
            grad /= float(len(idx))
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            rn = np.linalg.norm(gt, axis=1, keepdims=True)
            nz = rn[:, 0] > 1e-14
            direc = np.zeros_like(gt)
            direc[nz] = gt[nz] / rn[nz]
            Xt = _norm(X - step * direc)
            vv2 = _maxcos(Xt)
            if vv2 < bv:
                bv = vv2; bX = Xt.copy(); X = Xt
                step = min(step * 1.3, 0.005)
            else:
                step *= 0.6
                if step < 5e-7:
                    step = float(step0) * 0.3
                    # tiny tangent kick to escape flat
                    R = rng.standard_normal(X.shape) * 2e-4
                    R -= np.einsum('ij,ij->i', R, X)[:, None] * X
                    X = _norm(X + R)
        return bX, bv

    def void_refine(v0, Y, steps=120):
        v = v0 / (np.linalg.norm(v0) + 1e-300)
        for s in range(steps):
            if time.time() - t0 > BUDGET - 16:
                break
            f = s / max(1, steps - 1)
            tau = 0.008 * ((0.0008 / 0.008) ** f)
            lr = 0.08 * ((0.008 / 0.08) ** f)
            dd = Y @ v
            m0 = float(np.max(dd))
            e = np.exp((dd - m0) / tau)
            se = float(np.sum(e))
            if not np.finite(se) or se <= 0:
                break
            w = e / se
            g = w @ Y
            gt = g - float(g @ v) * v
            v = v - lr * gt
            nr = float(np.linalg.norm(v))
            if not np.finite(nr) or nr < 1e-12:
                break
            v = v / nr
        return v / (np.linalg.norm(v) + 1e-300)

    def find_void(Y, ntry=4000):
        # random sampling + refine top few
        R = _norm(rng.standard_normal((ntry, d)))
        D = R @ Y.T
        md = D.max(axis=1)
        order = np.argsort(md)[:4]
        best_v = R[order[0]]
        best_s = float(md[order[0]])
        for k in order:
            v = void_refine(R[k], Y, steps=100)
            s = float(np.max(Y @ v))
            if s < best_s:
                best_s = s
                best_v = v
        return best_v

    # quick initial polish of loaded best
    try:
        _X, _v = hard_topk(g_best, iters=80, K=250, step0=0.0008)
        if _v < g_val:
            g_val = _v; g_best = _X.copy()
    except Exception:
        pass

    # ---------- basin hopping ----------
    sigmas = [0.022, 0.009, 0.032, 0.013, 0.005, 0.018]
    NCYC = 60
    second_best = None
    second_val = 1e9
    for cyc in range(NCYC):
        if time.time() - t0 > BUDGET - 28:
            break
        # choose start: mostly global best, sometimes second
        if second_best is not None and (cyc % 5 == 4):
            start = second_best
        else:
            start = g_best
        # identify jammed cloud from start
        try:
            G = start @ start.T
            np.fill_diagonal(G, -2.0)
            c_all = G[iu0, iu1]
            Ktop = 400
            idx_top = np.argpartition(-c_all, Ktop)[:Ktop]
            involved = np.unique(np.concatenate([iu0[idx_top], iu1[idx_top]]))
        except Exception:
            involved = np.array([], dtype=int)
        sigma = sigmas[cyc % len(sigmas)]
        Xp = start.copy()
        noise = rng.standard_normal(Xp.shape) * sigma
        # boost jammed points
        boost = np.full(n, 0.45, dtype=np.float64)
        if len(involved) > 0:
            boost[involved] = 1.6
        noise *= boost[:, None]
        noise -= np.einsum('ij,ij->i', noise, Xp)[:, None] * Xp
        # occasional big shake on one random loaded point
        if cyc % 3 == 1 and len(involved) > 0:
            j = involved[int(rng.integers(len(involved)))]
            big = rng.standard_normal(d) * 0.08
            big -= float(big @ Xp[j]) * Xp[j]
            noise[j] += big
        Xp = _norm(Xp + noise)

        # every 4th cycle: delete-and-reinsert worst 1-2 points
        if cyc % 4 == 2 or cyc % 4 == 3:
            try:
                rm = _rowmax(Xp)
                nw = 2 if (cyc % 4 == 3) else 1
                worst = np.argsort(-rm)[:nw]
                keep = [i for i in range(n) if i not in set(worst.tolist())]
                Y = Xp[keep]
                for w in worst:
                    v = find_void(Y, ntry=3000)
                    Xp[w] = v
                    Y = np.vstack([Y, v[None, :]])
            except Exception:
                pass

        # anneal: larger tau early to engage cloud, smaller later
        # vary schedule by cycle to diversify basins
        if cyc % 2 == 0:
            Xo, vo = adam_lse(Xp, steps=420, tau0=0.008, tau1=0.0007, lr0=0.003, lr1=0.0004)
        else:
            Xo, vo = adam_lse(Xp, steps=420, tau0=0.004, tau1=0.0004, lr0=0.002, lr1=0.0003)
        # snap with hard max
        try:
            K = 300 if cyc % 2 == 0 else 120
            Xh, vh = hard_topk(Xo, iters=100, K=K, step0=0.001)
        except Exception:
            Xh, vh = Xo, vo
        if vh < g_val - 1e-12:
            g_val = vh; g_best = Xh.copy()
        elif vh < second_val and vh > g_val + 1e-9:
            # keep diverse near-best
            if np.linalg.norm(Xh - g_best) > 1e-3:
                second_val = vh; second_best = Xh.copy()
        # also consider vo
        if vo < g_val - 1e-12:
            g_val = vo; g_best = Xo.copy()

    # ---------- final intensive polish ----------
    # long anneal from global best with small shake ensemble
    cands = [g_best]
    if second_best is not None:
        cands.append(second_best)
    # add tiny shakes of best
    for _ in range(2):
        if time.time() - t0 > BUDGET - 24:
            break
        R = rng.standard_normal(g_best.shape) * 0.004
        R -= np.einsum('ij,ij->i', R, g_best)[:, None] * g_best
        cands.append(_norm(g_best + R))
    polished = []
    for X0 in cands:
        if time.time() - t0 > BUDGET - 22:
            break
        try:
            X1, v1 = adam_lse(X0, steps=600, tau0=0.005, tau1=0.0004, lr0=0.002, lr1=0.00025)
            X2, v2 = hard_topk(X1, iters=160, K=250, step0=0.0009)
            X3, v3 = hard_topk(X2, iters=160, K=60, step0=0.0005)
            bestX, bestV = (X3, v3) if v3 < v2 else (X2, v2)
            if v1 < bestV:
                bestX, bestV = X1, v1
        except Exception:
            continue
        polished.append((bestV, bestX))
        if bestV < g_val:
            g_val = bestV; g_best = bestX.copy()
    # pure hill climbing on true max with per-point micro moves
    try:
        X = g_best.copy()
        cur = g_val
        iters = 4000
        for it in range(iters):
            if time.time() - t0 > BUDGET - 3:
                break
            # pick point involved in max or random
            if it % 4 == 0:
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                ij = np.unravel_index(int(np.argmax(G)), G.shape)
                pi = ij[int(rng.integers(2))]
            else:
                pi = int(rng.integers(n))
            scale = 3e-4 * (0.5 ** (it / iters)) + 1e-5
            r = rng.standard_normal(d) * scale
            r -= float(r @ X[pi]) * X[pi]
            Xt = X.copy()
            Xt[pi] = (Xt[pi] + r) / (np.linalg.norm(Xt[pi]) + 1e-300)
            # fast delta eval: only row pi matters for increase? need full max
            G2 = Xt @ Xt.T
            np.fill_diagonal(G2, -2.0)
            v2 = float(np.max(G2))
            if v2 < cur:
                cur = v2; X = Xt
                g_val = cur; g_best = X.copy()
    except Exception:
        pass

    g_best = _norm(g_best)
    # tiny shrink guard (strictly feasible, avoid exact ties issues)
    return np.asarray(g_best, dtype=np.float64)
