import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _tangent_bases(X):
    n, d = X.shape
    Bs = np.empty((n, d, d-1))
    for i in range(n):
        x = X[i]
        k = int(np.argmin(np.abs(x)))
        norm2 = 2.0 - 2.0*x[k]
        if norm2 < 1e-12:
            norm2 = 1e-12
        norm = np.sqrt(norm2)
        u = -x / norm
        u[k] += 1.0 / norm
        # columns j != k of H = I - 2 u u^T
        idx = [j for j in range(d) if j != k]
        uj = u[idx]  # (d-1,)
        B = -2.0 * np.outer(u, uj)  # (d, d-1)
        for c, j in enumerate(idx):
            B[j, c] += 1.0
        Bs[i] = B
    return Bs

def _pairs_within(X, thresh, topcap=2500):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu, ju = np.triu_indices(n, 1)
    vals = G[iu, ju]
    mask = vals >= (m - thresh)
    pi = iu[mask]; pj = ju[mask]; pv = vals[mask]
    if len(pi) > topcap:
        o = np.argsort(-pv)[:topcap]
        pi, pj, pv = pi[o], pj[o], pv[o]
    return pi, pj, pv, m

def _extract_marginals(res, p):
    try:
        # new scipy: res.ineqlin.marginals
        mg = None
        if hasattr(res, 'ineqlin'):
            il = res.ineqlin
            if hasattr(il, 'marginals'):
                mg = np.asarray(il.marginals, dtype=float).ravel()
            elif isinstance(il, dict) and 'marginals' in il:
                mg = np.asarray(il['marginals'], dtype=float).ravel()
        if mg is None and isinstance(res, dict) and 'ineqlin' in res:
            il = res['ineqlin']
            if isinstance(il, dict) and 'marginals' in il:
                mg = np.asarray(il['marginals'], dtype=float).ravel()
            elif hasattr(il, 'marginals'):
                mg = np.asarray(il.marginals, dtype=float).ravel()
        # old scipy: res['marginals'] or res.con
        if mg is None and hasattr(res, 'marginals'):
            mg = np.asarray(res.marginals, dtype=float).ravel()
        if mg is not None and mg.size >= p:
            return mg[:p]
    except Exception:
        pass
    return None

def _lp_step(X, m_ref, pi, pj, forced_mask, delta, rise, drop):
    from scipy.optimize import linprog
    n, d = X.shape
    N = n*(d-1)
    p = len(pi)
    if p == 0:
        return X, None, False
    Bs = _tangent_bases(X)
    G = X @ X.T
    dots = G[pi, pj]
    # bounds
    bounds = [(-delta, delta)]*N + [(-1.0, m_ref+rise)]
    c = np.zeros(N+1)
    c[-1] = 1.0
    Aub = np.zeros((p, N+1))
    bub = np.zeros(p)
    for q in range(p):
        i = int(pi[q]); j = int(pj[q])
        ci = Bs[i].T @ X[j]  # 21
        cj = Bs[j].T @ X[i]
        Aub[q, i*(d-1):(i+1)*(d-1)] = ci
        Aub[q, j*(d-1):(j+1)*(d-1)] = cj
        if forced_mask is not None and forced_mask[q]:
            Aub[q, -1] = 0.0
            bub[q] = (m_ref - drop) - float(dots[q])
        else:
            Aub[q, -1] = -1.0
            bub[q] = -float(dots[q])
    try:
        res = linprog(c, A_ub=Aub, b_ub=bub, bounds=bounds, method='highs')
    except Exception:
        return X, None, False
    if res.status != 0 or res.x is None:
        return X, None, False
    try:
        xx = np.asarray(res.x, dtype=float).ravel()
    except Exception:
        return X, None, False
    if xx.size != N+1 or not np.all(np.finite(xx)):
        return X, None, False
    y = xx[:-1].reshape(n, d-1)
    dx = np.zeros_like(X)
    for i in range(n):
        dx[i] = Bs[i] @ y[i]
    # trust-region safeguard: clip large steps from linearization error
    Xn = _normed(X + dx)
    mg = _extract_marginals(res, p)
    return Xn, mg, True

def _adam_polish(X0, steps, beta0, beta1, lr0, lr1, seed=0):
    X = X0.copy()
    n, d = X.shape
    M = np.zeros_like(X); V = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1t, b2t = 1.0, 1.0
    best_val = _max_cos(X); best_X = X.copy()
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s/float(max(1, steps-1))
        beta = float(np.exp(lb0+(lb1-lb0)*f))
        lr = float(np.exp(ll0+(ll1-ll0)*f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta*(G-m))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum())
        if not np.isfinite(S) or S < 1e-30:
            continue
        Gr = (W @ X)/S
        dot = (Gr*X).sum(axis=1, keepdims=True)
        Gr = Gr - dot*X
        b1t *= b1; b2t *= b2
        M *= b1; M += (1.0-b1)*Gr
        V *= b2; V += (1.0-b2)*(Gr*Gr)
        Mh = M/(1.0-b1t); Vh = V/(1.0-b2t)
        X = X - lr*Mh/(np.sqrt(Vh)+eps)
        X = _normed(X)
        if (s % 80 == 79) or (s == steps-1):
            v = _max_cos(X)
            if v < best_val:
                best_val = v; best_X = X.copy()
    return best_X, best_val

def solve():
    import time
    import numpy as np
    t0 = time.time()
    np.random.seed(0)
    def left(limit=158.0):
        return (time.time()-t0) < limit
    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a = _normed(a)
                # remove zero rows guard
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        best_X = _normed(np.random.randn(59, 22))
    else:
        vals = [_max_cos(c) for c in cands]
        best_X = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(best_X)
    m0_global = best_val

    # Stage A: baseline LP to get stresses (pairs within 1e-5)
    stresses = None
    try:
        pi0, pj0, pv0, m0 = _pairs_within(best_X, 1e-5, topcap=2500)
        if len(pi0) > 0 and len(pi0) < 3000:
            Xn, mg, ok = _lp_step(best_X, m0, pi0, pj0, None, delta=0.004, rise=0.001, drop=0.0)
            if ok and mg is not None:
                stresses = np.abs(mg)
            elif ok:
                stresses = None
    except Exception:
        stresses = None

    # Stage B: forced contact-breaking trials
    rng = np.random.RandomState(0)
    n_trials = 14
    for trial in range(n_trials):
        if not left(135.0):
            break
        try:
            # refresh pairs around current best
            piW, pjW, pvW, mcur = _pairs_within(best_X, 0.003, topcap=1400)
            if len(piW) == 0:
                continue
            # rank by closeness to max (proxy for stress if marginals unavailable)
            if stresses is not None and trial < 4:
                # use stored stresses mapped by pair identity is stale; use fresh random-weighted top
                order = np.argsort(-pvW)
            else:
                order = np.argsort(-pvW)
            # select 20-50 dominant contacts: top slice + random subset
            k = int(20 + (trial*13 % 31))  # 20..50 deterministic variety
            k = min(k, len(piW))
            # dominant = top 2k candidates
            cand_top = order[:min(len(order), max(k*3, 60))]
            # randomized subset among dominant (stress/contact subset)
            sel_idx = rng.choice(cand_top, size=k, replace=False)
            sel_set = set(sel_idx.tolist())
            forced_mask = np.zeros(len(piW), dtype=bool)
            forced_mask[sel_idx] = True
            # parameters vary per trial
            delta = [0.008, 0.012, 0.006, 0.015][trial % 4]
            drop = 2e-4 + (trial % 3)*1e-4
            rise = 1e-4
            Xcur = best_X.copy()
            m_ref = float(mcur)  # incumbent max for this trial
            # sequence of 5-10 relinearized LPs
            K = 6 + (trial % 3)  # 6..8
            ok_all = True
            for it in range(K):
                if not left(135.0):
                    break
                # re-evaluate pairs each relinearization (keep forced identity by (i,j)? map)
                # simpler: recompute pairs and keep forcing on pairs closest to original selected pairs
                # carry forced pairs as explicit (i,j) list
                if it == 0:
                    fpi = piW[sel_idx].copy(); fpj = pjW[sel_idx].copy()
                    cpi, cpj = piW, pjW
                else:
                    cpi2, cpj2, _, _ = _pairs_within(Xcur, 0.003, topcap=1400)
                    if len(cpi2) == 0:
                        ok_all = False; break
                    cpi, cpj = cpi2, cpj2
                    # rebuild forced mask by matching (i,j) pairs
                    s = set(zip(fpi.tolist(), fpj.tolist()))
                    forced_mask = np.array([((int(a), int(b)) in s or (int(b), int(a)) in s) for a, b in zip(cpi, cpj)], dtype=bool)
                    # if none matched (pairs drifted), force the previously selected index pairs directly:
                    # add missing forced pairs to constraint set
                    have = set(zip(cpi.tolist(), cpj.tolist()))
                    haveU = set((min(a,b), max(a,b)) for a,b in have)
                    extra_i, extra_j = [], []
                    for a,b in zip(fpi.tolist(), fpj.tolist()):
                        key = (min(a,b), max(a,b))
                        if key not in haveU:
                            extra_i.append(a); extra_j.append(b)
                    if len(extra_i) > 0:
                        cpi = np.concatenate([cpi, np.array(extra_i, dtype=int)])
                        cpj = np.concatenate([cpj, np.array(extra_j, dtype=int)])
                        forced_mask = np.concatenate([forced_mask, np.ones(len(extra_i), dtype=bool)])
                Xn, mg2, ok = _lp_step(Xcur, m_ref, cpi, cpj, forced_mask, delta=delta, rise=rise, drop=drop)
                if not ok:
                    # retry once with larger trust region
                    Xn2, mg2b, ok2 = _lp_step(Xcur, m_ref, cpi, cpj, forced_mask, delta=delta*1.8, rise=rise*1.5, drop=drop)
                    if not ok2:
                        ok_all = False; break
                    Xn = Xn2
                Xcur = Xn
                # shrink delta slightly for stability
                delta *= 0.92
            # follow with short Adam polish
            Xp, vp = _adam_polish(Xcur, steps=900, beta0=40.0, beta1=350.0, lr0=0.006, lr1=0.0006)
            # also try a second polish seed from same Xcur with different schedule if time
            if vp < best_val and left(140.0):
                best_val = vp; best_X = Xp.copy(); m0_global = best_val
                # update stresses for next trials using fresh LP marginals around new best
                try:
                    piN, pjN, _, mN = _pairs_within(best_X, 1e-5, topcap=2500)
                    if 0 < len(piN) < 3000:
                        _, mgN, okN = _lp_step(best_X, mN, piN, pjN, None, delta=0.004, rise=0.001, drop=0.0)
                        if okN and mgN is not None:
                            stresses = np.abs(mgN)
                except Exception:
                    pass
            else:
                # keep best-tracking inside polish already; check Xcur itself
                vc = _max_cos(Xp)
                if vc < best_val:
                    best_val = vc; best_X = Xp.copy()
            # opportunistic second subset from same trial: different random subset polish only
            if left(135.0) and trial % 2 == 0:
                try:
                    Xp2, vp2 = _adam_polish(Xcur, steps=600, beta0=80.0, beta1=500.0, lr0=0.004, lr1=0.0004)
                    if vp2 < best_val:
                        best_val = vp2; best_X = Xp2.copy()
                except Exception:
                    pass
        except Exception:
            continue

    # Stage C: final fine polish of best (short, firm clock)
    if left(155.0):
        try:
            Xf, vf = _adam_polish(best_X, steps=1500, beta0=60.0, beta1=600.0, lr0=0.004, lr1=0.0002)
            if vf < best_val:
                best_val = vf; best_X = Xf.copy()
        except Exception:
            pass
    if left(160.0):
        try:
            Xf, vf = _adam_polish(best_X, steps=800, beta0=150.0, beta1=800.0, lr0=0.002, lr1=0.0001)
            if vf < best_val:
                best_val = vf; best_X = Xf.copy()
        except Exception:
            pass
    best_X = _normed(best_X)
    return best_X
