import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    HOP_END = t0 + 142.0  # reserve final 30s
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 453, 5

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                continue
            nn = np.sqrt(np.sum(cc * cc, axis=1, keepdims=True))
            nn[nn == 0] = 1.0
            cc = cc / nn
            if not np.all(np.isfinite(cc)):
                continue
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)

    elites = [(bestV, bestX.copy())]  # list of (val, X), keep best 3

    def full_anneal(X, steps, tau0, tau1, lr0, lr1, mu, t_lim):
        v = np.zeros_like(X)
        ltau = np.log(tau1 / tau0)
        llr = np.log(lr1 / lr0)
        for it in range(steps):
            if it % 40 == 0 and time.time() > t_lim:
                break
            frac = it / max(steps - 1, 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            # (G-m)<=0 safe
            with np.errstate(over='ignore', under='ignore'):
                P = np.exp((G - m) / tau)
            np.fill_diagonal(P, 0.0)
            s = float(np.sum(P))
            if not np.isfinite(s) or s <= 0:
                break
            # scale invariant after global norm, keep as is
            Grad = P @ X
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            nrm = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
            if not np.isfinite(nrm) or nrm < 1e-16:
                break
            Gt /= nrm
            v = mu * v + (1.0 - mu) * Gt
            X = X - lr * v
            X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X

    def largest_hole(Xref, n_cand):
        mref = Xref.shape[0]
        best_val = float("inf")
        best_vec = None
        batch = 10000
        rem = int(n_cand)
        while rem > 0:
            b = batch if rem > batch else rem
            C = rng.standard_normal((b, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            D = C @ Xref.T
            rm = np.max(D, axis=1)
            idx = int(np.argmin(rm))
            if float(rm[idx]) < best_val:
                best_val = float(rm[idx])
                best_vec = C[idx].copy()
            rem -= b
        return best_vec, best_val

    def refine_hole(y, Xref, steps=130, tau=1e-4):
        lr0, lr1 = 0.020, 0.003
        for it in range(steps):
            s = Xref @ y
            ms = float(np.max(s))
            with np.errstate(over='ignore', under='ignore'):
                e = np.exp((s - ms) / tau)
            se = float(np.sum(e))
            if not np.isfinite(se) or se <= 0:
                break
            w = e / se
            grad = w @ Xref
            gd = float(grad @ y)
            gt = grad - gd * y
            nr = float(np.linalg.norm(gt))
            if not np.isfinite(nr) or nr < 1e-14:
                break
            gt /= nr
            frac = it / max(steps - 1, 1)
            lr = lr0 * (1.0 - frac) + lr1 * frac
            y = y - lr * gt
            ny = float(np.linalg.norm(y))
            if ny == 0 or not np.isfinite(ny):
                break
            y = y / ny
        return y

    ks = [15, 25, 40]
    hop = 0
    # hop loop
    while time.time() < HOP_END and hop < 30:
        if time.time() > HOP_END - 4.0:
            break
        k = ks[hop % len(ks)]
        # round-robin over elites
        elites_sorted = sorted(elites, key=lambda t: t[0])
        elites = elites_sorted
        baseV, baseX = elites[hop % len(elites)]
        Xb = baseX.copy()
        # per-point worst
        G0 = Xb @ Xb.T
        np.fill_diagonal(G0, -2.0)
        per = np.max(G0, axis=1)
        m0 = float(np.max(per))
        order_worst = np.argsort(per)[::-1]  # descending
        # rattlers: slack > 1e-9
        ratt = np.where((m0 - per) > 1e-9)[0]
        # mixture: half worst, half random rattlers
        n_worst = k // 2
        n_rat = k - n_worst
        sel = set()
        for ii in range(n_worst):
            sel.add(int(order_worst[ii]))
        # random rattlers not already selected
        if ratt.size > 0:
            # shuffle deterministically via rng
            perm = rng.permutation(ratt.size)
            cnt = 0
            for pi in perm:
                if cnt >= n_rat:
                    break
                idx = int(ratt[pi])
                if idx not in sel:
                    sel.add(idx)
                    cnt += 1
            # if not enough, fill random indices
            while cnt < n_rat:
                idx = int(rng.integers(0, n))
                if idx not in sel:
                    sel.add(idx)
                    cnt += 1
        else:
            while len(sel) < k:
                sel.add(int(rng.integers(0, n)))
            # trim if over? already k
        # if sel still < k (overlap worst/rattler), fill random
        while len(sel) < k:
            sel.add(int(rng.integers(0, n)))
        sel_list = list(sel)[:k]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[sel_list] = False
        Xkeep = Xb[keep_mask].copy()  # (n-k,5)
        # reinsert one at a time: random order
        ins_order = rng.permutation(k)
        # n_cand in 50k-100k
        if k <= 15:
            n_cand = 80000
        elif k <= 25:
            n_cand = 60000
        else:
            n_cand = 50000
        Xcur = Xkeep
        # time budget for this hop insert phase
        hop_lim = min(HOP_END, DEAD - 32.0)
        aborted = False
        for _ in ins_order:
            if time.time() > hop_lim:
                aborted = True
                break
            y0, _hv = largest_hole(Xcur, n_cand)
            if y0 is None:
                y0 = rng.standard_normal(d)
                y0 /= np.linalg.norm(y0)
            y1 = refine_hole(y0, Xcur, steps=130, tau=1e-4)
            Xcur = np.vstack([Xcur, y1.reshape(1, d)])
        if aborted or Xcur.shape[0] != n:
            hop += 1
            continue
        # re-anneal full config 300-800 steps
        if k == 15:
            steps_full = 400
        elif k == 25:
            steps_full = 550
        else:
            steps_full = 700
        # vary tau slightly per hop for diversity
        tau0 = 2.0e-3 if (hop % 2 == 0) else 1.2e-3
        tau1 = 3.0e-4 if (hop % 2 == 0) else 2.0e-4
        Xnew = full_anneal(Xcur, steps_full, tau0, tau1, 0.020, 0.004, 0.85, hop_lim)
        Vnew = _max_cos(Xnew)
        # accept if within 2e-4 of incumbent best
        # note bestV is global best
        if Vnew < bestV:
            bestV = Vnew
            bestX = Xnew.copy()
        if Vnew <= bestV + 2e-4:
            # add to elite archive (keep best three distinct)
            # distinct: require Frobenius distance to existing >1e-6 else keep better
            dup = False
            for ev, ex in elites:
                if abs(ev - Vnew) < 1e-12:
                    # check closeness
                    dd = float(np.mean(np.abs(ex - Xnew)))
                    if dd < 1e-7:
                        dup = True
                        break
            if not dup:
                elites.append((Vnew, Xnew.copy()))
                elites = sorted(elites, key=lambda t: t[0])[:3]
        hop += 1

    # final 30s polish of overall best
    try:
        X = bestX.copy()
        # two-stage low-temp anneal
        t_rem = DEAD - time.time()
        if t_rem > 4.0:
            X = full_anneal(X, 600, 8e-4, 2e-4, 0.012, 0.003, 0.88, DEAD - 8.0)
            v = _max_cos(X)
            if v < bestV:
                bestV = v
                bestX = X.copy()
            else:
                # keep trajectory best? keep bestX as global best, continue from improved-or-not
                # continue polishing from current X if close
                if v <= bestV + 5e-5:
                    pass
                else:
                    X = bestX.copy()
        # greedy worst-pair micro shave until DEAD-1
        eps = 8e-7
        it = 0
        while time.time() < DEAD - 1.0:
            it += 1
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            if m < bestV:
                bestV = m
                bestX = X.copy()
            flat = int(np.argmax(G))
            i = flat // n
            j = flat % n
            xi = X[i].copy(); xj = X[j].copy()
            cij = float(xi @ xj)
            cij = max(-1.0, min(1.0, cij))
            ti = xj - cij * xi
            ni = float(np.linalg.norm(ti))
            if ni < 1e-12:
                ti = rng.standard_normal(d)
                ti -= float(ti @ xi) * xi
                ni = float(np.linalg.norm(ti)) + 1e-30
            ti /= ni
            tj = xi - cij * xj
            nj = float(np.linalg.norm(tj))
            if nj < 1e-12:
                tj = rng.standard_normal(d)
                tj -= float(tj @ xj) * xj
                nj = float(np.linalg.norm(tj)) + 1e-30
            tj /= nj
            improved = False
            for sc in (eps, eps * 0.5):
                vi = xi - sc * ti
                vi /= np.linalg.norm(vi)
                vj = xj - sc * tj
                vj /= np.linalg.norm(vj)
                # fast eval of affected maxima
                di = X @ vi
                dj = X @ vj
                di[i] = -2.0; di[j] = -2.0
                dj[i] = -2.0; dj[j] = -2.0
                mc = float(np.max(di))
                b2 = float(np.max(dj))
                if b2 > mc:
                    mc = b2
                c2 = float(vi @ vj)
                if c2 > mc:
                    mc = c2
                # rest max excluding i,j
                # compute rest max quickly: we have G, mask i,j
                # use precomputed m_rest approx: max of G excluding rows i,j cols?
                # for safety compute full max only if promising
                if mc < m - 1e-15:
                    # verify full
                    X[i] = vi; X[j] = vj
                    G2 = X @ X.T
                    np.fill_diagonal(G2, -2.0)
                    m2 = float(np.max(G2))
                    if m2 < m - 1e-15:
                        if m2 < bestV:
                            bestV = m2
                            bestX = X.copy()
                        improved = True
                        break
                    else:
                        X[i] = xi; X[j] = xj
            if improved:
                eps = min(eps * 1.03, 2e-6)
            else:
                eps *= 0.85
                if eps < 1e-10:
                    eps = 1e-10
                    # tiny random kick to one of them occasionally
                    if it % 50 == 0:
                        kk = i if (it % 100 == 0) else j
                        rr = rng.standard_normal(d)
                        rr -= float(rr @ X[kk]) * X[kk]
                        rn = float(np.linalg.norm(rr))
                        if rn > 1e-12:
                            rr /= rn
                            X[kk] = X[kk] + 5e-10 * rr
                            X[kk] /= float(np.linalg.norm(X[kk]))
            if it % 200 == 0 and time.time() > DEAD - 1.0:
                break
    except Exception:
        pass

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
