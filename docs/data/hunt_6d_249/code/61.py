import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    BUDGET = 172.0
    T_END = t0 + BUDGET
    FINAL_RES = 18.0
    rng = np.random.default_rng(0)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)
    P = len(I0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(an)):
                    cands.append(an)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    bestX = cands[bi].copy()
    bestVal = float(vals[bi])

    from scipy.optimize import minimize

    def soft_anneal(Xinit, taus, iters, creg=0.05):
        Xf = Xinit.ravel().copy()
        for tau, mi in zip(taus, iters):
            if time.time() > T_END - FINAL_RES:
                break
            tau = float(tau)
            def fg(xf):
                X = xf.reshape(n, d)
                r = np.sqrt(np.sum(X * X, axis=1))
                r = np.maximum(r, 1e-12)
                Xn = X / r[:, None]
                G = Xn @ Xn.T
                g = G[I0, J0]
                gm = float(np.max(g))
                e = np.exp((g - gm) / tau)
                S = float(np.sum(e))
                Lsoft = gm + tau * np.log(S + 1e-300)
                w = e / (S + 1e-300)
                W = np.zeros((n, n))
                W[I0, J0] = w
                W = W + W.T
                A = W @ Xn
                proj = np.sum(A * Xn, axis=1)
                G0 = (A - proj[:, None] * Xn) / r[:, None]
                diff = r - 1.0
                L = Lsoft + creg * float(np.sum(diff * diff))
                Gr = G0 + (2.0 * creg * diff)[:, None] * Xn
                return float(L), Gr.ravel()
            try:
                res = minimize(fg, Xf, jac=True, method='L-BFGS-B',
                    options={'maxiter': int(mi), 'maxfun': int(mi * 5), 'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
                Xf = res.x
            except Exception:
                break
            Xm = Xf.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Xm, axis=1, keepdims=True), 1e-12)
            Xf = (Xm / rn).ravel()
            if time.time() > T_END - 2:
                break
        Xo = Xf.reshape(n, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def riesz_anneal(Xinit, s=8.0, mi=60, creg=0.02):
        Xf = Xinit.ravel().copy()
        s = float(s)
        def fg(xf):
            X = xf.reshape(n, d)
            r = np.sqrt(np.sum(X * X, axis=1))
            r = np.maximum(r, 1e-12)
            Xn = X / r[:, None]
            G = Xn @ Xn.T
            # squared distance
            R2 = 2.0 - 2.0 * G
            np.fill_diagonal(R2, 2.0)
            R2 = np.maximum(R2, 1e-6)
            # energy sum over i<j of R2^{-s/2}
            pw = R2 ** (-s / 2.0)
            # mean energy for scaling
            E = float(np.sum(pw[I0, J0])) / float(P)
            # dE/dg: s * R2^{-s/2-1}
            coef = s * (R2 ** (-s / 2.0 - 1.0))
            np.fill_diagonal(coef, 0.0)
            # W = dE/dG / P
            W = coef / float(P)
            A = W @ Xn
            proj = np.sum(A * Xn, axis=1)
            G0 = (A - proj[:, None] * Xn) / r[:, None]
            diff = r - 1.0
            L = E + creg * float(np.sum(diff * diff))
            Gr = G0 + (2.0 * creg * diff)[:, None] * Xn
            return float(L), Gr.ravel()
        try:
            res = minimize(fg, Xf, jac=True, method='L-BFGS-B',
                options={'maxiter': int(mi), 'maxfun': int(mi * 4), 'ftol': 1e-14, 'gtol': 1e-6, 'maxcor': 8})
            Xf = res.x
        except Exception:
            pass
        Xo = Xf.reshape(n, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def refine_single(pt, Xfix):
        p = pt.copy()
        pn = np.linalg.norm(p)
        if pn < 1e-12:
            p = rng.standard_normal(size=(d,))
            p /= np.linalg.norm(p)
        else:
            p /= pn
        for tau, steps, lr in [(0.01, 40, 0.4), (0.003, 40, 0.25), (0.001, 30, 0.15)]:
            for _ in range(steps):
                dots = Xfix @ p
                gm = float(np.max(dots))
                e = np.exp((dots - gm) / tau)
                se = float(np.sum(e)) + 1e-300
                w = e / se
                grad = w @ Xfix
                gt = grad - float(np.dot(grad, p)) * p
                gn = float(np.linalg.norm(gt))
                if gn < 1e-12:
                    break
                gt /= gn
                step = lr
                bestd = float(np.max(Xfix @ p))
                improved = False
                for _ls in range(3):
                    q = p - step * gt
                    q /= (np.linalg.norm(q) + 1e-12)
                    nd = float(np.max(Xfix @ q))
                    if nd < bestd:
                        p = q
                        improved = True
                        break
                    step *= 0.5
                if not improved:
                    # tiny drift to explore
                    q = p - 0.02 * gt
                    q /= (np.linalg.norm(q) + 1e-12)
                    nd = float(np.max(Xfix @ q))
                    if nd < bestd + 0.004:
                        p = q
            if time.time() > T_END - FINAL_RES:
                break
        return p

    def greedy_refill(Xbase, del_idx, pool_size=30000):
        del_idx = np.asarray(del_idx, dtype=int)
        k = len(del_idx)
        if k == 0:
            return Xbase.copy()
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[del_idx] = False
        Xkeep = Xbase[keep_mask]
        Cp = rng.standard_normal(size=(pool_size, d))
        Cp /= np.maximum(np.linalg.norm(Cp, axis=1, keepdims=True), 1e-12)
        # curmax vs keep
        curmax = np.empty(pool_size)
        BS = 10000
        for s in range(0, pool_size, BS):
            e = min(s + BS, pool_size)
            D = Cp[s:e] @ Xkeep.T
            curmax[s:e] = D.max(axis=1)
        used = np.zeros(pool_size, dtype=bool)
        new_pts = []
        for _ in range(k):
            curmax[used] = np.inf
            b = int(np.argmin(curmax))
            used[b] = True
            pt = Cp[b].copy()
            if len(new_pts) == 0:
                Xfix = Xkeep
            else:
                Xfix = np.vstack([Xkeep, np.stack(new_pts, axis=0)])
            pt = refine_single(pt, Xfix)
            new_pts.append(pt)
            dd = Cp @ pt
            # update: max with new pt
            np.maximum(curmax, dd, out=curmax)
            curmax[used] = np.inf
            if time.time() > T_END - FINAL_RES:
                break
        # if early break and fewer points, fill randomly
        while len(new_pts) < k:
            pt = rng.standard_normal(size=(d,))
            pt /= np.linalg.norm(pt)
            new_pts.append(pt)
        Xt = Xbase.copy()
        Xt[del_idx] = np.stack(new_pts, axis=0)
        return Xt

    def deep_hole_vs(Xfix, ntry=20000):
        C = rng.standard_normal(size=(ntry, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        D = C @ Xfix.T
        cm = D.max(axis=1)
        b = int(np.argmin(cm))
        pt = C[b].copy()
        pt = refine_single(pt, Xfix)
        return pt

    def top_set(X, K=1200):
        G = X @ X.T
        g = G[I0, J0]
        if K >= len(g):
            idx = np.arange(len(g))
        else:
            idx = np.argpartition(g, -K)[-K:]
        codes = I0[idx].astype(np.int64) * n + J0[idx].astype(np.int64)
        return set(codes.tolist())

    def jacc_dist(a, b):
        if not a and not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        if union == 0:
            return 0.0
        return 1.0 - inter / union

    archive = [(bestVal, bestX.copy(), top_set(bestX, 1200))]

    def try_archive(val, X):
        s = top_set(X, 1200)
        md = 1.0
        for _, _, s2 in archive:
            dd = jacc_dist(s, s2)
            if dd < md:
                md = dd
        if md < 0.06 and val >= bestVal:
            return False
        archive.append((val, X.copy(), s))
        if len(archive) > 10:
            # keep best + diverse
            archive.sort(key=lambda t: t[0])
            keep = archive[:4]
            rest = archive[4:]
            # sort rest by distance to best
            bs = keep[0][2]
            rest.sort(key=lambda t: jacc_dist(t[2], bs), reverse=True)
            del archive[:]
            archive.extend(keep + rest[:6])
        return True

    def pick_base():
        if len(archive) == 1 or rng.random() < 0.7:
            return bestX
        else:
            j = int(rng.integers(len(archive)))
            return archive[j][1]

    def stats(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        rm = np.max(G, axis=1)
        thr = m - 0.002
        # counts
        cnt = np.sum(G > thr, axis=1)
        return m, rm, cnt, G

    # quick initial jam
    if time.time() < T_END - FINAL_RES - 10:
        try:
            Y = soft_anneal(bestX, [0.002], [40])
            v = _max_cos(Y)
            if v < bestVal:
                bestVal = v; bestX = Y.copy()
                try_archive(v, Y)
        except Exception:
            pass

    it = 0
    while time.time() < T_END - FINAL_RES:
        it += 1
        op = it % 5
        try:
            base = pick_base()
            if op == 1:
                # Riesz spreading then soft jam (balance stress, kill rattlers)
                s_choice = [6.0, 8.0, 12.0][int(rng.integers(3))]
                Y1 = riesz_anneal(base, s=s_choice, mi=50)
                Y = soft_anneal(Y1, [0.003, 0.0012], [60, 60])
                v = _max_cos(Y)
                if v < bestVal:
                    bestVal = v; bestX = Y.copy()
                    try_archive(v, Y)
                elif v < bestVal + 0.002:
                    try_archive(v, Y)
                # also try pure riesz+tiny jam from best even if worse to diversify
            elif op == 2:
                # spherical cap delete-refill
                u = rng.standard_normal(size=(d,))
                u /= (np.linalg.norm(u) + 1e-12)
                dots = base @ u
                k = int(rng.integers(10, 20))
                del_idx = np.argsort(dots)[-k:]
                Xt = greedy_refill(base, del_idx, pool_size=30000)
                Y = soft_anneal(Xt, [0.003, 0.0012], [55, 55])
                v = _max_cos(Y)
                if v < bestVal:
                    bestVal = v; bestX = Y.copy()
                    try_archive(v, Y)
                elif v < bestVal + 0.002:
                    try_archive(v, Y)
            elif op == 3:
                # hub delete (overconstrained points)
                m, rm, cnt, G = stats(base)
                k = int(rng.integers(7, 13))
                # order by cnt then rm
                order = np.lexsort((rm, cnt))
                del_idx = order[-k:]
                Xt = greedy_refill(base, del_idx, pool_size=30000)
                Y = soft_anneal(Xt, [0.003, 0.001], [55, 55])
                v = _max_cos(Y)
                if v < bestVal:
                    bestVal = v; bestX = Y.copy()
                    try_archive(v, Y)
                elif v < bestVal + 0.002:
                    try_archive(v, Y)
            elif op == 4:
                # rattler relocation to deep holes (low-cost topology change)
                m, rm, cnt, G = stats(base)
                nr = int(rng.integers(3, 7))
                ridx = np.argsort(rm)[:nr]
                Xt = base.copy()
                for r_ in ridx:
                    mask = np.ones(n, dtype=bool)
                    mask[int(r_)] = False
                    Xfix = Xt[mask]
                    pt = deep_hole_vs(Xfix, ntry=15000)
                    Xt[int(r_)] = pt
                    if time.time() > T_END - FINAL_RES:
                        break
                Y = soft_anneal(Xt, [0.0025, 0.001], [50, 50])
                v = _max_cos(Y)
                if v < bestVal:
                    bestVal = v; bestX = Y.copy()
                    try_archive(v, Y)
                elif v < bestVal + 0.002:
                    try_archive(v, Y)
            else:  # op 0 : global shake
                sigma = 0.10 + 0.06 * float(rng.random())
                # larger shake occasionally
                if rng.random() < 0.2:
                    sigma = 0.30
                pert = base + sigma * rng.standard_normal(size=(n, d))
                pert /= np.maximum(np.linalg.norm(pert, axis=1, keepdims=True), 1e-12)
                Y = soft_anneal(pert, [0.003, 0.0012], [55, 55])
                v = _max_cos(Y)
                if v < bestVal:
                    bestVal = v; bestX = Y.copy()
                    try_archive(v, Y)
                elif v < bestVal + 0.002:
                    try_archive(v, Y)
        except Exception:
            continue
        # periodic full jam of best
        if it % 12 == 0 and time.time() < T_END - FINAL_RES - 5:
            try:
                Y = soft_anneal(bestX, [0.001, 0.0005], [60, 60])
                v = _max_cos(Y)
                if v < bestVal:
                    bestVal = v; bestX = Y.copy()
            except Exception:
                pass

    # final reserve: intensive polish of best + top archive entries
    try:
        # sort archive by val
        archive.sort(key=lambda t: t[0])
        # polish best few and keep global best
        nfin = min(3, len(archive))
        for fi in range(nfin):
            if time.time() > T_END - 3:
                break
            X0 = archive[fi][1] if fi < len(archive) else bestX
            # use smallest taus
            Y = soft_anneal(X0, [0.0015, 0.0006, 0.0003], [70, 70, 60])
            v = _max_cos(Y)
            if v < bestVal:
                bestVal = v; bestX = Y.copy()
        # last tiny polish of best
        if time.time() < T_END - 2:
            Y = soft_anneal(bestX, [0.00035], [70])
            v = _max_cos(Y)
            if v < bestVal:
                bestVal = v; bestX = Y.copy()
    except Exception:
        pass
    bestX /= np.maximum(np.linalg.norm(bestX, axis=1, keepdims=True), 1e-12)
    return bestX
