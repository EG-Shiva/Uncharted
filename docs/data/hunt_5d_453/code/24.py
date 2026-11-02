import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    DEAD = t0 + 170.0
    rng = np.random.default_rng(0)
    np.random.seed(0)

    # load starts
    cands = []
    try:
        xb = np.load("best.npy")
        cands.append(np.asarray(xb, dtype=float))
    except Exception:
        pass
    try:
        xr = np.load("record.npy")
        cands.append(np.asarray(xr, dtype=float))
    except Exception:
        pass
    valid = []
    for c in cands:
        try:
            if c.shape != (453, 5):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            if not np.all(np.isfinite(c)):
                continue
            valid.append((_max_cos(c), c))
        except Exception:
            continue
    if len(valid) == 0:
        x0 = rng.standard_normal((453, 5))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        valid.append((_max_cos(x0), x0))
    valid.sort(key=lambda t: t[0])
    best_val = float(valid[0][0])
    best_X = valid[0][1].copy()
    n, d = best_X.shape  # 453,5

    # current state
    X = best_X.copy()
    G = X @ X.T
    np.fill_diagonal(G, -2.0)

    # helper to recompute G fully
    def recompute_G():
        nonlocal G
        G = X @ X.T
        np.fill_diagonal(G, -2.0)

    # greedy micro-shaving on current X,G with adaptive eps, time-bounded
    # updates best_val/best_X via closure lists
    def greedy_budget(time_limit, eps_init, max_iters=1000000):
        nonlocal X, G, best_val, best_X
        eps = float(eps_init)
        it = 0
        no_imp = 0
        # local copies for speed
        while it < max_iters and time.time() < time_limit:
            # periodic full refresh to kill drift
            if it % 5000 == 4999:
                Gr = X @ X.T
                np.fill_diagonal(Gr, -2.0)
                G = Gr
            # find worst pair
            flat = int(np.argmax(G))
            i = flat // n
            j = flat % n
            m = float(G[i, j])
            if m < best_val:
                best_val = m
                best_X = X.copy()
            # time check occasionally
            if it % 500 == 0 and time.time() >= time_limit:
                break
            it += 1

            xi = X[i]
            xj = X[j]
            cij = float(xi @ xj)
            if cij > 1.0:
                cij = 1.0
            elif cij < -1.0:
                cij = -1.0
            # tangent directions away
            ti = xj - cij * xi
            ni = float(np.linalg.norm(ti))
            if ni < 1e-14:
                # random orthogonal
                ti = rng.standard_normal(d)
                ti -= float(ti @ xi) * xi
                ni = float(np.linalg.norm(ti)) + 1e-30
            ti /= ni
            tj = xi - cij * xj
            nj = float(np.linalg.norm(tj))
            if nj < 1e-14:
                tj = rng.standard_normal(d)
                tj -= float(tj @ xj) * xj
                nj = float(np.linalg.norm(tj)) + 1e-30
            tj /= nj

            # m_rest excluding i,j : mask rows/cols
            ri = G[i, :].copy()
            rj = G[j, :].copy()
            G[i, :] = -2.0; G[:, i] = -2.0
            G[j, :] = -2.0; G[:, j] = -2.0
            m_rest = float(np.max(G))
            # restore
            G[i, :] = ri; G[:, i] = ri
            G[j, :] = rj; G[:, j] = rj

            # duplicate / jammed case: use simultaneous tiny step if tied
            if m - m_rest < 1.2e-12:
                # find tied set (pairs within 2e-09 of max)
                # get row maxima to collect indices cheaply: scan G for > m-2e-9
                # limit to avoid blowup
                # do one joint gradient micro-step with eps
                # build mask of pairs
                # To avoid O(n^2) extra, reuse G scan: find indices where G high
                # Use where on upper? full scan once
                idx = np.argwhere(G > m - 2e-09)
                # idx includes both (a,b) and (b,a); collect unique, cap 40
                uniq = []
                seen = set()
                for a, b in idx:
                    if a == b:
                        continue
                    if len(uniq) > 40:
                        break
                    if a not in seen:
                        seen.add(a); uniq.append(int(a))
                    if len(uniq) > 40:
                        break
                    if b not in seen:
                        seen.add(b); uniq.append(int(b))
                if len(uniq) >= 2 and len(uniq) <= 40:
                    s = eps if eps < 5e-07 else 5e-07
                    Xnew = X.copy()
                    for k in uniq:
                        # repulsion from tied neighbours
                        # neighbours: l with G[k,l] > m-2e-9
                        row = G[k, :]
                        nbr = np.where(row > m - 2e-09)[0]
                        if len(nbr) == 0:
                            continue
                        gvec = np.zeros(d)
                        for l in nbr:
                            ckl = float(G[k, l])
                            gvec += X[l] - ckl * X[k]
                        # project (already tangent) normalize
                        gn = float(np.linalg.norm(gvec))
                        if gn < 1e-14 or not np.isfinite(gn):
                            continue
                        gvec /= gn
                        xk = X[k] - s * gvec
                        xk /= np.linalg.norm(xk)
                        Xnew[k] = xk
                    Gn = Xnew @ Xnew.T
                    np.fill_diagonal(Gn, -2.0)
                    m2 = float(np.max(Gn))
                    if m2 < m - 1e-15:
                        X = Xnew
                        G = Gn
                        if m2 < best_val:
                            best_val = m2
                            best_X = X.copy()
                        eps = min(eps * 1.05, 2e-06)
                        no_imp = 0
                        continue
                    else:
                        eps *= 0.6
                        no_imp += 1
                        if eps < 3e-13:
                            return it
                        continue
                # else fall through to normal single-pair tries
                # need ri/rj row excl for m_rest_i/j below; recompute after? ri,rj still valid
                pass

            # row excl maxima for single-move bounds
            # row i excl j (ri[i]=-2 already)
            # compute quickly
            # ri[j]=m, need exclude
            a1 = float(np.max(ri[np.arange(n) != j])) if n > 1 else -2.0
            # Actually ri includes -2 at i; max over k!=j gives max involving i excl j
            # similarly rj
            a2 = float(np.max(rj[np.arange(n) != i])) if n > 1 else -2.0
            # m_rest_i = max(m_rest, a2)  (pairs with j unchanged)
            m_rest_i = m_rest if m_rest >= a2 else a2
            m_rest_j = m_rest if m_rest >= a1 else a1

            best_m2 = m
            best_kind = -1  # 0 both, 1 i-only, 2 j-only
            best_vi = None
            best_vj = None

            # try scales: both moves
            # use 3 scales, singles 2 scales
            scales_both = (eps, eps * 0.35, eps * 0.12)
            scales_single = (eps, eps * 0.35)
            # pre-store xi,xj for reuse
            for s in scales_both:
                inv = 1.0 / np.sqrt(1.0 + s * s)
                vi = (xi - s * ti) * inv
                vj = (xj - s * tj) * inv
                di = X @ vi
                dj = X @ vj
                # mask i,j
                # di[i] ~1, di[j] old dot irrelevant
                # compute max excl i,j without extra alloc: view
                # manual max via partition: use np.max on masked copy? n small so copy ok
                # do inline:
                mi = -2.0
                # loop in numpy: set temp? we can compute max with masking via max of two slices?
                # simplest: di[i]=-2 etc on copy? di is fresh array, can modify
                di[i] = -2.0; di[j] = -2.0
                dj[i] = -2.0; dj[j] = -2.0
                a = float(np.max(di))
                b = float(np.max(dj))
                c2 = float(vi @ vj)
                mc = m_rest
                if a > mc: mc = a
                if b > mc: mc = b
                if c2 > mc: mc = c2
                if mc < best_m2 - 1e-15:
                    best_m2 = mc
                    best_kind = 0
                    best_vi = vi
                    best_vj = vj
                    # don't break, try smaller scales to see even better? keep best
            # singles (only if not already much better? still try to find best)
            for s in scales_single:
                inv = 1.0 / np.sqrt(1.0 + s * s)
                vi = (xi - s * ti) * inv
                di = X @ vi
                di[i] = -2.0
                # keep j entry (valid)
                a = float(np.max(di))
                mc = m_rest_i if m_rest_i >= a else a
                if mc < best_m2 - 1e-15:
                    best_m2 = mc
                    best_kind = 1
                    best_vi = vi
                    best_vj = None
            for s in scales_single:
                inv = 1.0 / np.sqrt(1.0 + s * s)
                vj = (xj - s * tj) * inv
                dj = X @ vj
                dj[j] = -2.0
                a = float(np.max(dj))
                mc = m_rest_j if m_rest_j >= a else a
                if mc < best_m2 - 1e-15:
                    best_m2 = mc
                    best_kind = 2
                    best_vi = None
                    best_vj = vj

            if best_kind >= 0:
                # apply
                if best_kind == 0:
                    X[i] = best_vi
                    X[j] = best_vj
                    # update G rows i,j
                    rowi = X @ X[i]
                    rowj = X @ X[j]
                    rowi[i] = -2.0
                    rowj[j] = -2.0
                    # cij entry
                    # rowi[j] should be vi·vj, rowj[i] same
                    cnew = float(X[i] @ X[j])
                    rowi[j] = cnew
                    rowj[i] = cnew
                    G[i, :] = rowi; G[:, i] = rowi
                    G[j, :] = rowj; G[:, j] = rowj
                    G[i, i] = -2.0; G[j, j] = -2.0
                    G[i, j] = cnew; G[j, i] = cnew
                elif best_kind == 1:
                    X[i] = best_vi
                    rowi = X @ X[i]
                    rowi[i] = -2.0
                    G[i, :] = rowi; G[:, i] = rowi
                    G[i, i] = -2.0
                else:
                    X[j] = best_vj
                    rowj = X @ X[j]
                    rowj[j] = -2.0
                    G[j, :] = rowj; G[:, j] = rowj
                    G[j, j] = -2.0
                if best_m2 < best_val:
                    best_val = best_m2
                    best_X = X.copy()
                eps = min(eps * 1.03, 3e-06)
                no_imp = 0
            else:
                eps *= 0.62
                no_imp += 1
                if eps < 5e-13:
                    return it
                if no_imp > 400:
                    # tiny random tangential jitter to worst point to escape
                    k = i if (it % 2 == 0) else j
                    rnd = rng.standard_normal(d)
                    rnd -= float(rnd @ X[k]) * X[k]
                    rn = float(np.linalg.norm(rnd))
                    if rn > 1e-12:
                        rnd /= rn
                        X[k] = X[k] + 2e-12 * rnd
                        X[k] /= np.linalg.norm(X[k])
                        rowk = X @ X[k]
                        rowk[k] = -2.0
                        G[k, :] = rowk; G[:, k] = rowk
                        G[k, k] = -2.0
                    eps = max(eps, 2e-12)
                    no_imp = 0
        return it

    # initial squeeze from loaded best
    # give ~25s to initial basin
    greedy_budget(min(DEAD - 60.0, time.time() + 25.0), 4e-07)

    # basin hopping: kicks to best
    sigmas = [2e-05, 5e-05, 1e-05, 8e-05, 3e-05, 1.2e-04, 1.5e-05, 6e-05]
    cyc = 0
    while time.time() < DEAD - 4.0:
        cyc += 1
        sig = sigmas[(cyc - 1) % len(sigmas)]
        # start from global best
        X = best_X.copy()
        # find worst points to bias kick
        # row max
        rm = np.max(G if False else (best_X @ best_X.T + np.eye(n) * -3.0), axis=1) if False else None
        # compute row max quickly
        Gb = best_X @ best_X.T
        np.fill_diagonal(Gb, -2.0)
        rm = np.max(Gb, axis=1)
        order = np.argsort(-rm)
        worst = order[:12]
        noise = rng.standard_normal((n, d)) * (sig * 0.4)
        noise[worst] += rng.standard_normal((len(worst), d)) * sig
        # also kick the absolute worst pair a bit more every 4th cycle
        if cyc % 4 == 0:
            flat = int(np.argmax(Gb))
            ii = flat // n; jj = flat % n
            noise[ii] += rng.standard_normal(d) * sig * 1.5
            noise[jj] += rng.standard_normal(d) * sig * 1.5
        X = X + noise
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # short greedy from kicked state; smaller eps for small sig
        eps0 = 3e-07 if sig < 4e-05 else 5e-07
        tlim = min(DEAD - 3.0, time.time() + 9.0)
        if cyc % 4 == 0:
            tlim = min(DEAD - 3.0, time.time() + 12.0)
        greedy_budget(tlim, eps0, max_iters=60000)
        # loop continues; best_X tracked inside

    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
