import time
import numpy as np

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 93, 13
    rng = np.random.default_rng(0)

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

    # load baseline
    base = None
    base_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, d):
                continue
            if not np.all(np.isfinite(a)):
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

    # ---------- void refinement (single vector vs fixed set) ----------
    def refine_void_single(v0, Y, steps=80, tau0=0.01, tau1=0.0008, lr0=0.1, lr1=0.01):
        v = v0 / (np.linalg.norm(v0) + 1e-300)
        for s in range(steps):
            if time.time() - t0 > BUDGET - 3:
                break
            f = s / max(1, steps - 1)
            tau = tau0 * ((tau1 / tau0) ** f)
            lr = lr0 * ((lr1 / lr0) ** f)
            dd = Y @ v
            m0 = float(np.max(dd))
            e = np.exp((dd - m0) / tau)
            se = float(np.sum(e))
            if not np.isfinite(se) or se <= 0:
                break
            w = e / se
            gv = w @ Y
            gt = gv - float(gv @ v) * v
            v = v - lr * gt
            nr = float(np.linalg.norm(v))
            if not np.isfinite(nr) or nr < 1e-12:
                break
            v = v / nr
        return v / (np.linalg.norm(v) + 1e-300)

    def best_void(Y, n_sample=25000, n_cand=8):
        S = rng.standard_normal((n_sample, d))
        S /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-300)
        sc = np.max(S @ Y.T, axis=1)
        idx = np.argpartition(sc, n_cand)[:n_cand]
        bv = None
        bs = 1e9
        for bi in idx:
            vr = refine_void_single(S[bi], Y, steps=70)
            s = float(np.max(Y @ vr))
            if s < bs:
                bs = s
                bv = vr
        return bv

    # ---------- annealed soft-max Adam ----------
    def smooth_adam(X0, steps, tau0, tau1, lr0, lr1):
        X = _norm(X0.copy())
        N = X.shape[0]
        j0, j1 = np.triu_indices(N, 1)
        m = np.zeros_like(X)
        vv = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        bv = _maxcos(X)
        bX = X.copy()
        tt = 0
        for s in range(steps):
            if time.time() - t0 > BUDGET - 3:
                break
            f = s / max(1, steps - 1)
            tau = tau0 * ((tau1 / tau0) ** f) if tau0 != tau1 else tau0
            lr = lr0 * ((lr1 / lr0) ** f) if lr0 != lr1 else lr0
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            c = G[j0, j1]
            mx = float(np.max(c))
            # check best occasionally
            if (s % 40 == 0) and mx < bv:
                bv = mx
                bX = X.copy()
            E = np.exp((c - mx) / tau)
            S = float(np.sum(E))
            if not np.isfinite(S) or S <= 0:
                break
            W = np.zeros((N, N), dtype=np.float64)
            W[j0, j1] = E
            W[j1, j0] = E
            grad = (W @ X) / S
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            tt += 1
            m = b1 * m + (1 - b1) * gt
            vv = b2 * vv + (1 - b2) * (gt * gt)
            mh = m / (1 - b1 ** tt)
            vh = vv / (1 - b2 ** tt)
            upd = mh / (np.sqrt(vh) + eps)
            # tangent projection
            upd = upd - np.einsum('ij,ij->i', upd, X)[:, None] * X
            X = _norm(X - lr * upd)
        cur = _maxcos(X)
        if cur < bv:
            bv = cur
            bX = X.copy()
        return bX, bv

    # ---------- hard top-K descent ----------
    def hard_topk(X0, iters=220, K=40, step0=0.008):
        X = _norm(X0.copy())
        N = X.shape[0]
        j0, j1 = np.triu_indices(N, 1)
        bv = _maxcos(X)
        bX = X.copy()
        step = float(step0)
        for it in range(iters):
            if time.time() - t0 > BUDGET - 3:
                break
            G = X @ X.T
            c = G[j0, j1]
            Kk = int(min(K, c.size))
            if Kk < c.size:
                idx = np.argpartition(-c, Kk)[:Kk]
            else:
                idx = np.arange(c.size)
            I = j0[idx]; J = j1[idx]
            grad = np.zeros_like(X)
            np.add.at(grad, I, X[J])
            np.add.at(grad, J, X[I])
            grad /= float(len(idx))
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            ng = float(np.linalg.norm(gt))
            if not np.isfinite(ng) or ng < 1e-15:
                # random tangent kick
                R = rng.standard_normal(X.shape) * 1e-4
                R -= np.einsum('ij,ij->i', R, X)[:, None] * X
                X = _norm(X + R)
                continue
            direc = gt / ng
            Xt = _norm(X - step * direc * np.sqrt(float(N)))
            v = _maxcos(Xt)
            if v < bv:
                bv = v
                bX = Xt.copy()
                X = Xt
                step = min(step * 1.12, 0.02)
            else:
                step *= 0.55
                if step < 1e-11:
                    step = float(step0) * 0.4
                    R = rng.standard_normal(X.shape) * 8e-5
                    R -= np.einsum('ij,ij->i', R, X)[:, None] * X
                    X = _norm(X + R)
        return bX, bv

    def polish(X0):
        # two-stage: soft then hard
        X1, _ = smooth_adam(X0, steps=320, tau0=0.008, tau1=0.0012, lr0=0.02, lr1=0.006)
        X2, v2 = hard_topk(X1, iters=160, K=60, step0=0.006)
        # second finer round
        X3, _ = smooth_adam(X2, steps=200, tau0=0.002, tau1=0.0004, lr0=0.008, lr1=0.003)
        X4, v4 = hard_topk(X3, iters=120, K=25, step0=0.004)
        if v4 < v2:
            return X4, v4
        else:
            # return best of the two
            v2c = _maxcos(X2)
            if v2c < v4:
                return X2, v2c
            return X4, v4

    # initial polish of base (quick)
    try:
        g_best, g_val = polish(g_best)
    except Exception:
        pass
    # keep safe copy
    best_archive = g_best.copy()
    arch_val = float(g_val)

    # ---------- N-overshoot homotopy (98 -> 93) once if time allows ----------
    try:
        if time.time() - t0 < BUDGET - 60:
            Xbig = g_best.copy()
            # add 5 voids greedily
            for _ in range(5):
                v = best_void(Xbig, n_sample=20000, n_cand=6)
                Xbig = _norm(np.vstack([Xbig, v[None, :]]))
            Xbig, _ = smooth_adam(Xbig, steps=700, tau0=0.012, tau1=0.002, lr0=0.015, lr1=0.006)
            # greedy delete back to 93 with brief repolish
            curX = Xbig
            while curX.shape[0] > n:
                if time.time() - t0 > BUDGET - 40:
                    break
                rm = _rowmax(curX)
                order = np.argsort(-rm)
                cands = order[:min(10, curX.shape[0])].tolist()
                bc = 1e9
                bXc = None
                for ci in cands:
                    Xd = np.delete(curX, ci, axis=0)
                    Xd2, _ = smooth_adam(Xd, steps=90, tau0=0.003, tau1=0.001, lr0=0.008, lr1=0.004)
                    val = _maxcos(Xd2)
                    if val < bc:
                        bc = val
                        bXc = Xd2
                if bXc is None:
                    break
                curX = bXc
            if curX.shape[0] == n:
                curX, cv = polish(curX)
                if cv < g_val:
                    g_val = cv
                    g_best = curX.copy()
    except Exception:
        pass

    # ---------- main basin-hopping with surgery ----------
    trial = 0
    k_cycle = [2, 3, 5, 1, 8, 4, 6, 2, 3, 10]
    sig_cycle = [0.015, 0.03, 0.008, 0.05, 0.02, 0.07]
    while time.time() - t0 < BUDGET - 6:
        trial += 1
        if time.time() - t0 > BUDGET - 6:
            break
        mode = trial % 3  # 0:surgery worst+random, 1:shake, 2:surgery max-pair focused
        try:
            if mode == 1:
                sig = sig_cycle[(trial // 3) % len(sig_cycle)]
                R = rng.standard_normal((n, d)) * sig
                R -= np.einsum('ij,ij->i', R, g_best)[:, None] * g_best
                X0 = _norm(g_best + R)
                X1, _ = smooth_adam(X0, steps=380, tau0=0.007, tau1=0.0008, lr0=0.018, lr1=0.004)
                X2, v2 = hard_topk(X1, iters=170, K=50, step0=0.006)
            else:
                k = k_cycle[(trial // 3) % len(k_cycle)]
                k = int(min(k, 12))
                G = g_best @ g_best.T
                np.fill_diagonal(G, -2.0)
                rm = np.max(G, axis=1)
                order_desc = np.argsort(-rm)
                order_asc = np.argsort(rm)
                # max pair
                flat = int(np.argmax(G))
                ai, bi = flat // n, flat % n
                delset = set()
                if mode == 2:
                    delset.add(int(ai)); 
                    if k >= 2:
                        delset.add(int(bi))
                    # fill with loosest / random
                    for idx in order_asc.tolist():
                        if len(delset) >= k:
                            break
                        if idx not in delset and rng.random() < 0.7:
                            delset.add(int(idx))
                    # if still short, add worst
                    for idx in order_desc.tolist():
                        if len(delset) >= k:
                            break
                        delset.add(int(idx))
                else:
                    # worst + random mix
                    delset.add(int(order_desc[0]))
                    # add random among top-15 worst
                    topw = order_desc[:min(15, n)].tolist()
                    while len(delset) < k:
                        c = topw[int(rng.integers(0, len(topw)))]
                        delset.add(int(c))
                        if len(delset) >= k:
                            break
                        # also inject a rattler
                        c2 = order_asc[int(rng.integers(0, min(30, n)))]
                        delset.add(int(c2))
                        if len(delset) >= k:
                            break
                del_idx = sorted(list(delset))[:k]
                mask = np.ones(n, dtype=bool)
                mask[np.array(del_idx, dtype=int)] = False
                Xkeep = g_best[mask].copy()
                # greedy void reinsert
                for _ in range(k):
                    if time.time() - t0 > BUDGET - 6:
                        break
                    v = best_void(Xkeep, n_sample=22000, n_cand=7)
                    Xkeep = _norm(np.vstack([Xkeep, v[None, :]]))
                X1, _ = smooth_adam(Xkeep, steps=360, tau0=0.008, tau1=0.0009, lr0=0.018, lr1=0.004)
                X2, v2 = hard_topk(X1, iters=150, K=55, step0=0.006)
            if v2 < g_val - 1e-12:
                g_val = float(v2)
                g_best = X2.copy()
                best_archive = g_best.copy()
                arch_val = float(g_val)
            # occasional diversification: if stuck many trials, big shake from archive
            if trial % 12 == 0 and time.time() - t0 < BUDGET - 20:
                R = rng.standard_normal((n, d)) * 0.08
                R -= np.einsum('ij,ij->i', R, g_best)[:, None] * g_best
                X0 = _norm(g_best + R)
                Xd, vd = smooth_adam(X0, steps=250, tau0=0.01, tau1=0.002, lr0=0.02, lr1=0.006)
                vd2 = _maxcos(Xd)
                # only adopt as restart base if not too bad (within 0.01)
                if vd2 < g_val + 0.01:
                    # short polish and maybe replace g_best base for next trials without losing archive
                    Xd2, vd2b = hard_topk(Xd, iters=100, K=60, step0=0.006)
                    if vd2b < g_val:
                        g_val = float(vd2b)
                        g_best = Xd2.copy()
                    else:
                        # move base slightly to new basin for exploration
                        g_best = Xd2.copy()
                        # restore if next trials fail? keep archive
                        pass
                # ensure g_best never drifts too far from archive
                if _maxcos(g_best) > arch_val + 0.008:
                    g_best = best_archive.copy()
        except Exception:
            continue

    # restore archive if drifted
    try:
        if _maxcos(g_best) > arch_val:
            g_best = best_archive.copy()
            g_val = float(arch_val)
    except Exception:
        pass
    # final quick hard polish if time
    try:
        if time.time() - t0 < BUDGET - 1:
            G = g_best @ g_best.T
            np.fill_diagonal(G, -2.0)
            # tiny tangent descent on current max pairs only
            Xf, vf = hard_topk(g_best, iters=80, K=15, step0=0.003)
            if vf < g_val:
                g_best = Xf.copy()
    except Exception:
        pass

    g_best = _norm(np.asarray(g_best, dtype=np.float64))
    # guard NaN
    g_best[~np.isfinite(g_best)] = 0.0
    # if any bad row, replace with random
    nr = np.linalg.norm(g_best, axis=1)
    for i in range(n):
        if not np.isfinite(nr[i]) or nr[i] < 1e-9:
            v = rng.standard_normal(d)
            v /= (np.linalg.norm(v) + 1e-300)
            g_best[i] = v
    g_best = _norm(g_best)
    return g_best
