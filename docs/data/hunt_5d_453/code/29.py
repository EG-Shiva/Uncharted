import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rm = np.max(G, axis=1)
    m = float(np.max(rm))
    return rm, m, G

def _big_hole(Xref, rng, n_cand=50000, chunk=10000):
    d = Xref.shape[1]
    best_v = float('inf')
    best_h = None
    n_chunks = (n_cand + chunk - 1) // chunk
    for _ in range(n_chunks):
        cs = chunk if _ < n_chunks - 1 or n_cand % chunk == 0 else n_cand % chunk
        S = rng.standard_normal((cs, d))
        S /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-30)
        D = S @ Xref.T
        md = np.max(D, axis=1)
        bi = int(np.argmin(md))
        if float(md[bi]) < best_v:
            best_v = float(md[bi])
            best_h = S[bi].copy()
    return best_h

def _refine_hole(h, Xref, tau=1e-4, steps=130, lr0=0.04, lr1=0.005):
    h = h / (np.linalg.norm(h) + 1e-30)
    for it in range(steps):
        dots = Xref @ h
        m = float(np.max(dots))
        with np.errstate(over='ignore'):
            w = np.exp((dots - m) / tau)
        s = float(np.sum(w))
        if not np.isfinite(s) or s <= 0:
            break
        grad = (w @ Xref) / s
        Gt = grad - float(grad @ h) * h
        n = float(np.linalg.norm(Gt))
        if not np.isfinite(n) or n < 1e-14:
            break
        Gt /= n
        frac = it / max(1, steps - 1)
        lr = lr0 + (lr1 - lr0) * frac
        h = h - lr * Gt
        n2 = float(np.linalg.norm(h))
        if n2 == 0 or not np.isfinite(n2):
            break
        h /= n2
    return h

def _anneal(X, steps, tau0, tau1, lr0, lr1, mu, t_end):
    n, d = X.shape
    v = np.zeros_like(X)
    ltau = np.log(tau1 / tau0)
    llr = np.log(lr1 / lr0)
    for it in range(steps):
        if (it & 63) == 0 and time.time() > t_end:
            break
        frac = it / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * frac)
        lr = lr0 * np.exp(llr * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if not np.isfinite(m):
            break
        with np.errstate(over='ignore', invalid='ignore'):
            P = np.exp((G - m) / tau)
        np.fill_diagonal(P, 0.0)
        s = float(np.sum(P))
        if not np.isfinite(s) or s <= 0:
            break
        P /= s
        Grad = P @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        mn = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
        if not np.isfinite(mn) or mn < 1e-16:
            break
        Gt /= mn
        v = mu * v + (1.0 - mu) * Gt
        X = X - lr * v
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-30)
    return X

def _shave(X, t_end, eps_init=6e-8, max_iters=3000):
    n, d = X.shape
    bestV = _max_cos(X)
    eps = float(eps_init)
    it = 0
    while it < max_iters and time.time() < t_end:
        it += 1
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < bestV:
            bestV = m
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        xi = X[i].copy(); xj = X[j].copy()
        cij = float(xi @ xj)
        cij = max(-1.0, min(1.0, cij))
        ti = xj - cij * xi
        ni = float(np.linalg.norm(ti))
        if ni < 1e-12:
            ti = np.random.randn(d); ti -= float(ti @ xi) * xi
            ni = float(np.linalg.norm(ti)) + 1e-30
        ti /= ni
        tj = xi - cij * xj
        nj = float(np.linalg.norm(tj))
        if nj < 1e-12:
            tj = np.random.randn(d); tj -= float(tj @ xj) * xj
            nj = float(np.linalg.norm(tj)) + 1e-30
        tj /= nj
        Ri = G[i, :].copy(); Rj = G[j, :].copy()
        G[i, :] = -2.0; G[:, i] = -2.0
        G[j, :] = -2.0; G[:, j] = -2.0
        m_rest = float(np.max(G))
        r_imax = float(np.max(np.delete(Ri, [i, j])))
        r_jmax = float(np.max(np.delete(Rj, [i, j])))
        base_i = m_rest if m_rest > r_jmax else r_jmax
        base_j = m_rest if m_rest > r_imax else r_imax
        best_m2 = m; bvi = None; bvj = None; mode = 0
        for sc in (eps, eps * 0.5, eps * 0.25):
            vi = xi - sc * ti; vi /= (np.linalg.norm(vi) + 1e-30)
            vj = xj - sc * tj; vj /= (np.linalg.norm(vj) + 1e-30)
            di = X @ vi; dj = X @ vj
            di[i] = -2.0; di[j] = -2.0
            dj[i] = -2.0; dj[j] = -2.0
            mc = m_rest
            a = float(np.max(di))
            if a > mc: mc = a
            b = float(np.max(dj))
            if b > mc: mc = b
            c2 = float(vi @ vj)
            if c2 > mc: mc = c2
            if mc < best_m2 - 1e-15:
                best_m2 = mc; bvi = vi; bvj = vj; mode = 0
                break
        if best_m2 >= m - 1e-15:
            for sc in (eps, eps * 0.5):
                vi = xi - sc * ti; vi /= (np.linalg.norm(vi) + 1e-30)
                di = X @ vi; di[i] = -2.0
                mc = base_i
                a = float(np.max(di))
                if a > mc: mc = a
                if mc < best_m2 - 1e-15:
                    best_m2 = mc; bvi = vi; bvj = None; mode = 1
                    break
            for sc in (eps, eps * 0.5):
                vj = xj - sc * tj; vj /= (np.linalg.norm(vj) + 1e-30)
                dj = X @ vj; dj[j] = -2.0
                mc = base_j
                a = float(np.max(dj))
                if a > mc: mc = a
                if mc < best_m2 - 1e-15:
                    best_m2 = mc; bvi = None; bvj = vj; mode = 2
                    break
        if best_m2 < m - 1e-15:
            if mode == 0:
                X[i] = bvi; X[j] = bvj
            elif mode == 1:
                X[i] = bvi
            else:
                X[j] = bvj
            bestV = best_m2
            eps *= 0.985
            if eps < 1e-13: eps = 1e-13
        else:
            eps *= 0.6
            if eps < 4e-14:
                break
    return X, bestV

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    HOP_END = t0 + 142.0
    np.random.seed(1)
    rng = np.random.default_rng(1)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestX = None; bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                continue
            cc = cc / np.linalg.norm(cc, axis=1, keepdims=True)
            if not np.all(np.isfinite(cc)):
                continue
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv; bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((453, 5))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    n, d = bestX.shape
    elites = [(bestV, bestX.copy())]
    curX = bestX.copy(); curV = bestV
    ks = [15, 25, 40]
    it = 0
    while time.time() < HOP_END - 4.0:
        it += 1
        if HOP_END - time.time() < 3.0:
            break
        # choose start: round-robin elites, mostly cur
        if len(elites) > 1 and (it % 4 == 0):
            sidx = (it // 4) % len(elites)
            start = elites[sidx][1].copy()
        else:
            start = curX.copy()
        k = ks[it % 3]
        try:
            rm, m0, _ = _row_max(start)
        except Exception:
            continue
        order = np.argsort(-rm)
        k_worst = max(1, k // 2)
        worst_set = set(int(v) for v in order[:k_worst])
        # rattlers: no contact within 1e-9
        ratt = np.where(rm < m0 - 1e-9)[0]
        if len(ratt) < k:
            ratt = np.where(rm < m0 - 1e-7)[0]
        rng.shuffle(ratt) if len(ratt) > 0 else None
        chosen = set(worst_set)
        for v in ratt:
            if len(chosen) >= k:
                break
            chosen.add(int(v))
        while len(chosen) < k:
            chosen.add(int(rng.integers(0, n)))
        chosen = np.array(sorted(chosen)[:k], dtype=int)
        mask = np.ones(n, dtype=bool); mask[chosen] = False
        retained = start[mask].copy()
        cur_set = retained.copy()
        # reinsert one at a time
        n_cand = 50000 + (it % 3) * 10000  # 50k,60k,70k
        ok = True
        new_pts = []
        for t in range(k):
            if time.time() > HOP_END - 2.0:
                ok = False
                break
            try:
                h = _big_hole(cur_set, rng, n_cand=n_cand, chunk=10000)
                h = _refine_hole(h, cur_set, tau=1e-4, steps=130)
            except Exception:
                h = rng.standard_normal(d); h /= np.linalg.norm(h)
            new_pts.append(h)
            cur_set = np.vstack([cur_set, h.reshape(1, -1)])
        if not ok:
            continue
        Xnew = np.vstack([retained, np.array(new_pts)])
        # full anneal 300-800 steps by k
        steps = 300 if k == 15 else (500 if k == 25 else 750)
        try:
            Xnew = _anneal(Xnew, steps, 3.0e-4, 5.0e-5, 0.004, 0.001, 0.86, min(HOP_END, time.time() + 2.5))
        except Exception:
            continue
        try:
            Xnew, vpol = _shave(Xnew, min(HOP_END, time.time() + 0.7), eps_init=5e-8, max_iters=1500)
        except Exception:
            vpol = _max_cos(Xnew)
        if not np.all(np.isfinite(Xnew)):
            continue
        # update best / elites
        if vpol < bestV:
            bestV = vpol; bestX = Xnew.copy()
            elites.append((vpol, Xnew.copy()))
            elites.sort(key=lambda t: t[0])
            # keep best 3 distinct by value
            filt = []
            for vv2, xx2 in elites:
                if all(abs(vv2 - f[0]) > 1e-12 for f in filt):
                    filt.append((vv2, xx2))
                if len(filt) >= 3:
                    break
            # if collapsed, just keep top3
            if len(filt) < min(3, len(elites)):
                filt = elites[:3]
            elites = filt[:3]
            curX = Xnew.copy(); curV = vpol
        else:
            # elite archive insertion if good enough for top3
            if len(elites) < 3 or vpol < elites[-1][0] - 5e-9:
                distinct = all(abs(vpol - e[0]) > 1e-12 for e in elites)
                if distinct or len(elites) < 3:
                    elites.append((vpol, Xnew.copy()))
                    elites.sort(key=lambda t: t[0])
                    elites = elites[:3]
            # sideways acceptance within 2e-4 of best
            if vpol < bestV + 2e-4:
                curX = Xnew.copy(); curV = vpol
            else:
                # occasional random restart from elite to diversify
                if (it % 7 == 0) and len(elites) > 1:
                    pi = int(rng.integers(0, len(elites)))
                    curX = elites[pi][1].copy(); curV = elites[pi][0]
    # final 30s polish overall best
    try:
        Xb = bestX.copy(); Vb = bestV
        while time.time() < DEAD - 1.0:
            rem = DEAD - time.time()
            if rem < 1.2:
                break
            Xb = _anneal(Xb, 220, 1.5e-4, 3.0e-5, 0.002, 0.0005, 0.88, DEAD - 0.8)
            if time.time() > DEAD - 0.9:
                break
            Xb, Vb2 = _shave(Xb, min(DEAD - 0.5, time.time() + 2.0), eps_init=4e-8, max_iters=4000)
            vnow = _max_cos(Xb)
            if vnow < bestV:
                bestV = vnow; bestX = Xb.copy()
            # stop if no progress and little time
            if time.time() > DEAD - 2.0:
                break
    except Exception:
        pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
