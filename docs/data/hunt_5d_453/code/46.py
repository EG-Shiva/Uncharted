import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def _contact_degrees(X, thresh=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    # upper triangle contacts
    mask = G > (m - thresh)
    # degrees (count both sides, divide by 2 since G symmetric, diag excluded)
    deg = np.sum(mask, axis=1) // 1
    # mask is symmetric, sum counts each edge twice except we counted full matrix;
    # each contact appears twice (i,j and j,i), so divide by... actually sum over row
    # counts each neighbor once (since row includes both). No division needed except diag=0.
    # The above //1 keeps it.
    return m, G, np.asarray(deg, dtype=int)

def _skew_from_p(p):
    S = np.zeros((5, 5), dtype=np.float64)
    k = 0
    for a in range(5):
        for b in range(a + 1, 5):
            v = float(p[k]); k += 1
            S[a, b] = v
            S[b, a] = -v
    return S

def _cayley_R(p):
    S = _skew_from_p(p)
    I = np.eye(5, dtype=np.float64)
    A = I - S
    B = I + S
    try:
        R = np.linalg.solve(A, B)
    except Exception:
        return I
    return R

def _cluster_max(Xc_rot, Xo):
    # max over internal and cross only
    # internal
    Gcc = Xc_rot @ Xc_rot.T
    np.fill_diagonal(Gcc, -2.0)
    m1 = float(np.max(Gcc))
    Gco = Xc_rot @ Xo.T
    m2 = float(np.max(Gco)) if Gco.size else -2.0
    return m1 if m1 > m2 else m2

def _anneal(X, steps, tau0, tau1, lr0, lr1, mu, t_end, rng, holder):
    n = X.shape[0]
    v = np.zeros_like(X)
    ltau = np.log(tau1 / tau0)
    llr = np.log(lr1 / lr0)
    for it in range(steps):
        if (it & 31) == 0 and time.time() > t_end:
            break
        frac = it / max(steps - 1, 1)
        tau = tau0 * np.exp(ltau * frac)
        lr = lr0 * np.exp(llr * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < holder[0]:
            holder[0] = m
            holder[1] = X.copy()
        inv = 1.0 / tau
        with np.errstate(over='ignore'):
            P = np.exp((G - m) * inv)
        np.fill_diagonal(P, 0.0)
        s = float(np.sum(P))
        if not np.isfinite(s) or s <= 0:
            break
        P /= s
        Grad = P @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        mn = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
        if not np.isfinite(mn) or mn < 1e-18:
            break
        Gt /= mn
        v = mu * v + (1.0 - mu) * Gt
        X = X - lr * v
        X /= np.sqrt(np.sum(X * X, axis=1, keepdims=True))
    return X

def _shave(X, iters, eps_init, t_end, rng, holder):
    n, d = X.shape
    eps = float(eps_init)
    for it in range(iters):
        if (it & 31) == 0 and time.time() > t_end:
            break
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < holder[0]:
            holder[0] = m
            holder[1] = X.copy()
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        xi = X[i].copy(); xj = X[j].copy()
        cij = float(xi @ xj)
        cij = max(-1.0, min(1.0, cij))
        ti = xj - cij * xi
        ni = float(np.linalg.norm(ti))
        if ni < 1e-12:
            ti = rng.standard_normal(d); ti -= float(ti @ xi) * xi
            ni = float(np.linalg.norm(ti)) + 1e-30
        ti /= ni
        tj = xi - cij * xj
        nj = float(np.linalg.norm(tj))
        if nj < 1e-12:
            tj = rng.standard_normal(d); tj -= float(tj @ xj) * xj
            nj = float(np.linalg.norm(tj)) + 1e-30
        tj /= nj
        G[i, :] = -2.0; G[:, i] = -2.0
        G[j, :] = -2.0; G[:, j] = -2.0
        m_rest = float(np.max(G))
        improved = False
        for sc in (eps, eps * 0.5, eps * 0.25):
            vi = xi - sc * ti; vi /= float(np.linalg.norm(vi))
            vj = xj - sc * tj; vj /= float(np.linalg.norm(vj))
            di = X @ vi
            dj = X @ vj
            a = float(np.max(np.delete(di, [i, j])))
            b = float(np.max(np.delete(dj, [i, j])))
            c2 = float(vi @ vj)
            mc = m_rest
            if a > mc: mc = a
            if b > mc: mc = b
            if c2 > mc: mc = c2
            if mc < m - 1e-15:
                X[i] = vi; X[j] = vj
                if mc < holder[0]:
                    holder[0] = mc; holder[1] = X.copy()
                eps = min(eps * 1.04, 3e-3)
                improved = True
                break
        if not improved:
            eps *= 0.9
            if eps < 2e-10:
                eps = 2e-10
                k = i if (it & 1) == 0 else j
                rnd = rng.standard_normal(d)
                rnd -= float(rnd @ X[k]) * X[k]
                rn = float(np.linalg.norm(rnd))
                if rn > 1e-12:
                    rnd /= rn
                    X[k] = X[k] + 5e-10 * rnd
                    X[k] /= float(np.linalg.norm(X[k]))
    return X

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(0)
    n, d = 453, 5
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestV = float("inf")
    bestX = None
    for cc in cands:
        try:
            if cc.shape != (n, d):
                continue
            cc = cc / np.linalg.norm(cc, axis=1, keepdims=True)
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
    X = bestX.copy()
    holder = [bestV, bestX.copy()]

    # ---------- collective rigid-cluster moves ----------
    # time budget for this phase
    CLUSTER_END = t0 + 95.0
    # compute hubs
    try:
        m0, G0, deg = _contact_degrees(X)
    except Exception:
        deg = np.zeros(n, dtype=int)
    hubs = np.argsort(deg)[::-1]
    # keep top ~25 hubs with deg>0
    hub_list = [int(h) for h in hubs[:30] if deg[h] > 0]
    if len(hub_list) < 8:
        hub_list = [int(h) for h in hubs[:12]]
    sizes = [12, 20, 32]
    # precompute dot order per hub on the fly (changes after moves, recompute periodically)
    cluster_count = 0
    for round_id in range(3):
        if time.time() > CLUSTER_END:
            break
        # recompute degrees every round
        try:
            _, _, deg = _contact_degrees(X)
            hubs = np.argsort(deg)[::-1]
            hub_list = [int(h) for h in hubs[:30] if deg[h] > 0]
            if len(hub_list) < 8:
                hub_list = [int(h) for h in hubs[:12]]
        except Exception:
            pass
        for hi, hub in enumerate(hub_list):
            if time.time() > CLUSTER_END:
                break
            # vary size
            k = sizes[(hi + round_id) % len(sizes)]
            # cap: k nearest by dot to hub
            dots = X @ X[hub]
            order = np.argsort(dots)[::-1]
            idx = order[:k]
            # sort idx for stable masking
            mask = np.zeros(n, dtype=bool)
            mask[idx] = True
            Xc = X[idx].copy()
            Xo = X[~mask].copy()
            m_c0 = _cluster_max(Xc, Xo)
            m_glob = _max_cos(X)
            # random coordinate search over 10 skew params
            # scale depends on round
            sigmas = [0.035, 0.018, 0.008]
            sigma = sigmas[round_id % len(sigmas)]
            best_p = np.zeros(10, dtype=np.float64)
            best_c = m_c0
            best_R = np.eye(5)
            # number of trials adapts to time left
            n_trials = 220
            if time.time() > CLUSTER_END - 20:
                n_trials = 120
            for t in range(n_trials):
                if (t & 63) == 0 and time.time() > CLUSTER_END:
                    break
                p = rng.standard_normal(10) * sigma
                R = _cayley_R(p)
                Xcr = Xc @ R.T
                c = _cluster_max(Xcr, Xo)
                if c < best_c:
                    best_c = c
                    best_p = p.copy()
                    best_R = R.copy()
                    if best_c < m_c0 - 8e-4:
                        # big win, stop early
                        pass
            # hill climb from best
            if best_c < m_c0 - 1e-12:
                step = sigma * 0.35
                for t in range(80):
                    if (t & 31) == 0 and time.time() > CLUSTER_END:
                        break
                    p2 = best_p + rng.standard_normal(10) * step
                    R2 = _cayley_R(p2)
                    Xcr2 = Xc @ R2.T
                    c2 = _cluster_max(Xcr2, Xo)
                    if c2 < best_c:
                        best_c = c2
                        best_p = p2
                        best_R = R2
                    else:
                        step *= 0.985
                        if step < 1e-5:
                            step = 1e-5
            # optionally try reflection variant: R * flip
            # quick few tries of improper rotation (det -1) around cap
            if time.time() < CLUSTER_END - 5:
                for _ in range(12):
                    if time.time() > CLUSTER_END:
                        break
                    p = rng.standard_normal(10) * sigma * 0.7
                    R = _cayley_R(p)
                    # flip last axis
                    Rf = R.copy(); Rf[:, 4] *= -1.0
                    Xcr = Xc @ Rf.T
                    # reflection changes internal chirality; internal max may change slightly
                    # due to exact isometry internal max unchanged for pure reflection?
                    # (distances preserved) so only cross matters; still worth testing
                    c = _cluster_max(Xcr, Xo)
                    if c < best_c:
                        best_c = c
                        best_R = Rf.copy()
                        best_p = p.copy()
            gain = m_c0 - best_c
            if gain > 1e-12:
                # accept if not blowing up global beyond tolerance
                # global after = max(m_out, best_c); m_out <= m_glob
                # best_c <= m_c0 <= m_glob, so global won't increase; accept
                X[idx] = (Xc @ best_R.T)
                # renormalize (should stay unit, but guard)
                X[idx] /= np.linalg.norm(X[idx], axis=1, keepdims=True)
                cluster_count += 1
                # counter-displace neighbours: small jitter on shell just outside cap
                shell = order[k:k + 25]
                # push shell slightly away from rotated cluster centre
                # cheap: tiny tangent noise scaled by gain
                amp = min(0.004, 0.0005 + gain * 0.5)
                for s in shell[:10]:
                    if time.time() > CLUSTER_END:
                        break
                    rndv = rng.standard_normal(d)
                    rndv -= float(rndv @ X[s]) * X[s]
                    rn = float(np.linalg.norm(rndv))
                    if rn > 1e-12:
                        rndv /= rn
                        X[s] = X[s] + amp * 0.25 * rndv
                        X[s] /= float(np.linalg.norm(X[s]))
                # brief local softmax to relax (cheap)
                tlim = min(time.time() + 1.2, CLUSTER_END, DEAD - 60)
                X = _anneal(X, steps=90, tau0=0.0025, tau1=0.0006, lr0=0.012, lr1=0.003,
                            mu=0.86, t_end=tlim, rng=rng, holder=holder)
                # update holder best into X if holder improved a lot? keep X as is (annealed)
                # but ensure X tracks best basin: if holder much better, jump
                if holder[0] < _max_cos(X) - 5e-7:
                    X = holder[1].copy()
            # occasional rattler re-insertion
            if (cluster_count % 6 == 5) and time.time() < CLUSTER_END - 5:
                try:
                    rm, _, dgr = _contact_degrees(X)
                    rats = np.where(dgr <= 0)[0]
                    if len(rats) > 0:
                        # find large hole: random candidates minimizing max dot
                        nc = 8000
                        C = rng.standard_normal((nc, d))
                        C /= np.linalg.norm(C, axis=1, keepdims=True)
                        D = C @ X.T
                        md = np.max(D, axis=1)
                        bi = int(np.argmin(md))
                        # place worst rattler there
                        X[rats[0]] = C[bi]
                        # quick check
                        nv = _max_cos(X)
                        if nv < holder[0]:
                            holder[0] = nv; holder[1] = X.copy()
                except Exception:
                    pass
    # sync to best if anneal found better
    if holder[0] < _max_cos(X):
        X = holder[1].copy()

    # ---------- final polish: softmax + shave alternation ----------
    # use remaining time
    while time.time() < DEAD - 4.0:
        rem = DEAD - time.time()
        if rem < 5:
            break
        # anneal block
        tlim = time.time() + min(6.0, rem - 3.0)
        if tlim <= time.time():
            break
        X = _anneal(X, steps=260, tau0=0.003, tau1=0.00035, lr0=0.016, lr1=0.0025,
                    mu=0.88, t_end=tlim, rng=rng, holder=holder)
        if time.time() > DEAD - 4.0:
            break
        tlim2 = min(DEAD - 2.0, time.time() + 4.0)
        X = _shave(X, iters=600, eps_init=8e-4, t_end=tlim2, rng=rng, holder=holder)
        # hole surgery on worst point if stalled
        if time.time() < DEAD - 10:
            try:
                G = X @ X.T; np.fill_diagonal(G, -2.0)
                per = np.max(G, axis=1)
                w = int(np.argmax(per))
                # propose hole replacement for w only
                nc = 12000
                C = rng.standard_normal((nc, d))
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                Xkeep = np.delete(X, w, axis=0)
                D = C @ Xkeep.T
                md = np.max(D, axis=1)
                bi = int(np.argmin(md))
                old = X[w].copy()
                X[w] = C[bi]
                nv = _max_cos(X)
                if nv < holder[0]:
                    holder[0] = nv; holder[1] = X.copy()
                else:
                    # revert unless close (topology change attempt)
                    # keep with small prob if within 1e-4 to allow basin hop
                    cur = _max_cos(X)
                    base = float(np.max(np.delete(per, w).max(), 0)) if False else holder[0]
                    if cur > holder[0] + 2e-4:
                        X[w] = old
                    else:
                        # keep and anneal briefly
                        tlim3 = min(DEAD - 4.0, time.time() + 1.5)
                        X = _anneal(X, steps=70, tau0=0.002, tau1=0.0006, lr0=0.01, lr1=0.003,
                                    mu=0.86, t_end=tlim3, rng=rng, holder=holder)
            except Exception:
                pass
        if holder[0] < 0.83:
            break

    Xbest = holder[1].copy()
    Xbest /= np.linalg.norm(Xbest, axis=1, keepdims=True)
    # strict finite guard + tiny shrink not needed (score uses normalized rows)
    Xbest = np.where(np.isfinite(Xbest), Xbest, 0.0)
    nn = np.linalg.norm(Xbest, axis=1, keepdims=True)
    nn[nn == 0] = 1.0
    Xbest = Xbest / nn
    return Xbest
