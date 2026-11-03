import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_with_duals(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    last_marg = None
    last_I = None
    last_J = None
    itn = 0
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14 or itn > 80:
            break
        itn += 1
        c = (x @ x.T)[iu]
        m = float(c.max())
        thr = max(20 * radius, 1e-9)
        sel = c > m - thr
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            order = np.argsort(-c[sel])[:3000]
            I, J = I[order], J[order]
            cvals = c[sel][order]
        else:
            cvals = c[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cvals, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        # capture marginals
        try:
            il = getattr(res, "ineqlin", None)
            marg = None
            if il is not None:
                if hasattr(il, "marginals"):
                    marg = np.asarray(il.marginals, dtype=float)
                elif isinstance(il, dict) and "marginals" in il:
                    marg = np.asarray(il["marginals"], dtype=float)
            if marg is not None and marg.shape[0] == p:
                last_marg = marg.copy()
                last_I = I.copy()
                last_J = J.copy()
        except Exception:
            pass
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    # if no marginal captured, keep None
    return best_x, best, last_marg, last_I, last_J

def _dual_mass(n, marg, I, J, X, fallback_thresh=1e-7):
    if marg is not None and I is not None and J is not None and len(marg) == len(I):
        mass = np.zeros(n)
        am = np.abs(marg)
        # ignore non-finite
        am[~np.isfinite(am)] = 0.0
        for k in range(len(I)):
            mass[I[k]] += am[k]
            mass[J[k]] += am[k]
        if mass.sum() > 0:
            return mass
    # fallback: contact degree at 1e-7
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    deg = (G > m - 1e-7).sum(axis=1).astype(float)
    if deg.sum() == 0:
        deg = (G > m - 1e-4).sum(axis=1).astype(float)
    return deg

def _select_cluster(X, mass, target, tried, rng):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    adj = G > m - 1e-7
    # candidate seeds: top mass
    order = np.argsort(-mass)
    # consider top 30 seeds
    seeds = order[:30]
    best_c = None
    best_score = -1e18
    for s in seeds:
        s = int(s)
        cl = {s}
        # greedy expansion
        while len(cl) < target:
            # frontier neighbours
            neigh = set()
            for u in list(cl):
                nbrs = np.where(adj[u])[0]
                for v in nbrs:
                    if v not in cl:
                        neigh.add(int(v))
            if neigh:
                # pick highest mass
                pick = max(neigh, key=lambda v: mass[v])
                cl.add(pick)
            else:
                # connect via tightest dot to cluster
                cllist = list(cl)
                # max dot to cluster among outside
                sub = G[np.ix_(np.array([i for i in range(n) if i not in cl]), np.array(cllist))]
                # sub shape (n-|cl|, |cl|), take max over cluster axis
                mx = sub.max(axis=1)
                outs = np.array([i for i in range(n) if i not in cl])
                pick = outs[int(np.argmax(mx))]
                cl.add(int(pick))
        cllist = sorted(cl)
        tot = float(mass[cllist].sum())
        # diversity penalty: overlap with tried
        pen = 0.0
        for t in tried:
            ov = len(set(cllist) & set(t))
            if ov > 0:
                # penalize heavily if large overlap
                pen += tot * 0.15 * (ov / max(len(cllist), 1))
                if ov >= int(0.7 * target):
                    pen += tot * 0.5
        score = tot - pen + rng.random() * 1e-9
        if score > best_score:
            best_score = score
            best_c = cllist
    return best_c

def _tangent_bases(Xc):
    m, d = Xc.shape
    Bs = []
    for i in range(m):
        xi = Xc[i]
        A = np.eye(d) - np.outer(xi, xi)
        U, S, Vt = np.linalg.svd(A)
        # largest d-1 singular values ~1
        B = U[:, :d - 1]
        # ensure tangent: B.T xi ~0; re-orthogonalize
        Bs.append(B)
    return Bs

def _null_samples(X, cluster, Bs, rng, nsamp=4, thetas=(0.05, 0.15)):
    n, d = X.shape
    m = len(cluster)
    cmap = {g: k for k, g in enumerate(cluster)}
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    mx = float(G.max())
    # active contacts involving cluster: within 0.015 of max
    act = []
    for ii, gi in enumerate(cluster):
        row = G[gi]
        for gj in range(n):
            if gj == gi:
                continue
            if row[gj] > mx - 0.015:
                act.append((gi, gj))
    # build reduced Jacobian
    rdim = m * (d - 1)
    p = len(act)
    if p == 0:
        # no constraints: full tangent null
        Vnull = np.eye(rdim)
    else:
        Jr = np.zeros((p, rdim))
        for k, (a, b) in enumerate(act):
            xa = X[a]
            xb = X[b]
            if a in cmap:
                ia = cmap[a]
                Ba = Bs[ia]
                Jr[k, ia * (d - 1):(ia + 1) * (d - 1)] = Ba.T @ xb
            # if both in cluster, second endpoint contributes too (a is first but b may also be in cluster)
            # Note act includes ordered? we added only a in cluster as first, but b could be in cluster too.
            # To avoid double counting, handle symmetric: if b in cluster and b != a
            if b in cmap:
                ib = cmap[b]
                Bb = Bs[ib]
                # avoid double-adding when a==? a in cluster already did one side; add other side
                # if both in cluster, the row has two blocks; the above already filled ia block with xb,
                # now fill ib block with xa. If a==b impossible.
                # But if we iterated with a as cluster point, and b also cluster, this row will appear twice
                # (once from each endpoint). Deduplicate: only keep unique unordered.
                Jr[k, ib * (d - 1):(ib + 1) * (d - 1)] += (Bb.T @ xa) if (a in cmap and b in cmap and True) else 0.0
                # Actually if both in cluster, above line adds second block; fine. Deduplication handled below.
        # deduplicate unordered pairs (keep one row per unordered)
        # Build unique key set
        seen = {}
        keep = []
        for k, (a, b) in enumerate(act):
            key = (min(a, b), max(a, b))
            if key not in seen:
                seen[key] = k
                keep.append(k)
        Jr = Jr[keep, :]
        p = Jr.shape[0]
        try:
            U, S, Vt = np.linalg.svd(Jr, full_matrices=False)
        except Exception:
            Vnull = np.eye(rdim)
            S = np.ones(min(Jr.shape))
            Vt = np.eye(rdim)[:min(Jr.shape), :]
            U = None
        # null = singular vectors with small S
        tol = 0.3
        # number of small
        nsmall = int(np.sum(S < tol))
        # ensure at least 8 and at most rdim
        want = max(8, nsmall)
        want = min(want, rdim)
        # take bottom `want` vectors (smallest S are at end)
        # Vt rows ordered descending S, so bottom = last `want`
        Vnull = Vt[-want:, :].T  # rdim x want? Actually Vt shape (min(p,rdim) x rdim); if p<rdim, Vt has p rows only
        if Vnull.shape[1] < want:
            # pad with random orthogonal complement (just use identity complement approx: random)
            extra = want - Vnull.shape[1]
            pad = rng.standard_normal((rdim, extra))
            # orthogonalize against Vnull roughly
            Vnull = np.concatenate([Vnull, pad], axis=1)
        else:
            Vnull = Vnull  # rdim x want
        # Vnull columns are basis; we want row basis for sampling: y = Vnull @ coeff
    outs = []
    for s in range(nsamp):
        theta = float(rng.uniform(thetas[0], thetas[1]))
        if p == 0:
            coeff = rng.standard_normal((rdim,))
            y = coeff
        else:
            w = Vnull.shape[1]
            coeff = rng.standard_normal((w,))
            y = Vnull @ coeff  # rdim
        Y = y.reshape(m, d - 1)
        DX = np.zeros((m, d))
        for ii in range(m):
            DX[ii] = Bs[ii] @ Y[ii]
        # scale to rms theta
        rms = float(np.sqrt((DX * DX).sum() / max(m, 1)))
        if rms < 1e-12 or not np.isfinite(rms):
            DX = rng.standard_normal((m, d))
            # project to tangent
            for ii in range(m):
                xi = X[cluster[ii]]
                DX[ii] -= np.dot(DX[ii], xi) * xi
            rms = float(np.sqrt((DX * DX).sum() / max(m, 1)))
            if rms < 1e-12:
                continue
        DX *= (theta / rms)
        outs.append(DX)
    return outs

def _cluster_relax(X, cluster, steps, lr, T0, T1, old_pairs, thresh=0.38, pw=25.0, rng=None):
    n, d = X.shape
    m = len(cluster)
    cset = np.zeros(n, dtype=bool)
    cset[cluster] = True
    pos = -np.ones(n, dtype=int)
    for k, g in enumerate(cluster):
        pos[g] = k
    X = X.copy()
    m_ad = np.zeros((m, d))
    v_ad = np.zeros((m, d))
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    # precompute old pair list as (kpos, gj)
    op = []
    for (a, b) in old_pairs:
        if a in pos and pos[a] >= 0:
            op.append((int(pos[a]), int(b)))
        elif b in pos and pos[b] >= 0:
            # ensure unordered but gradient symmetric; store with cluster side first
            # dot same, gradient for cluster side uses other point
            op.append((int(pos[b]), int(a)))
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 1e-12:
            T = 1e-12
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        # mask to pairs involving cluster for softmax normalizer
        # find M = max over involving cluster
        sub = G[np.ix_(np.array(cluster), np.arange(n))]
        M = float(sub.max())
        E = np.exp((G - M) / T)
        np.fill_diagonal(E, 0.0)
        # zero pairs with neither in cluster
        # build mask: keep if i in cluster or j in cluster
        # efficient: zero out submatrix of frozen-frozen
        frozen = np.array([i for i in range(n) if not cset[i]])
        if len(frozen) > 0:
            E[np.ix_(frozen, frozen)] = 0.0
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad_full = (E @ X) / S  # n x d
        Gt = np.zeros((m, d))
        for k, g in enumerate(cluster):
            gg = Grad_full[g]
            xg = X[g]
            Gt[k] = gg - np.dot(gg, xg) * xg
        # penalty gradient for old contacts above thresh
        if op:
            for (kp, gj) in op:
                gi = cluster[kp]
                dot = float(np.dot(X[gi], X[gj]))
                if dot > thresh:
                    # gradient of (dot-thresh)^2 w.r.t X[gi] is 2*(dot-thresh)*X[gj]
                    gpen = 2.0 * pw * (dot - thresh) * X[gj]
                    # tangent project
                    gpen_t = gpen - np.dot(gpen, X[gi]) * X[gi]
                    Gt[kp] += gpen_t / max(S, 1.0) * 5.0 + gpen_t * 0.2
                    # scale: keep penalty comparable; use direct addition
        b1p *= beta1
        b2p *= beta2
        m_ad = beta1 * m_ad + (1.0 - beta1) * Gt
        v_ad = beta2 * v_ad + (1.0 - beta2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        for k, g in enumerate(cluster):
            X[g] = X[g] - lr * step[k]
            # renormalize
            nr = np.linalg.norm(X[g])
            if not np.isfinite(nr) or nr < 1e-12:
                X[g] = X[g] / 1.0
            else:
                X[g] /= nr
    return _norm(X)

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(7)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal(size=(n, d)))
        best_val = _max_cos(best)
    # initial dual-guided polish
    remain = DEAD - time.time()
    init_t = min(12.0, max(4.0, remain - 60.0)) if remain > 20 else min(5.0, max(2.0, remain - 15))
    try:
        best, best_val, marg, Ii, Jj = _lp_with_duals(best, init_t, deadline=DEAD - 2.0)
    except Exception:
        marg, Ii, Jj = None, None, None
        best_val = _max_cos(best)
    mass = _dual_mass(n, marg, Ii, Jj, best)
    tried = []
    trial = 0
    # cycle through clusters
    while time.time() < DEAD - 12.0:
        trial += 1
        if trial > 25:
            break
        target = 12 + (trial % 7)  # 12..18
        try:
            cluster = _select_cluster(best, mass, target, tried, rng)
        except Exception:
            break
        if cluster is None:
            break
        tried.append(sorted([int(c) for c in cluster]))
        # old contacts for penalty: pairs at 1e-6 within max involving cluster
        try:
            G0 = best @ best.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(G0.max())
            old_pairs = []
            cset0 = set(cluster)
            for a in cluster:
                for b in range(n):
                    if b == a:
                        continue
                    if G0[a, b] > m0 - 1e-6:
                        # store unordered once (a<b to avoid dup, but cluster relax handles both)
                        if a < b:
                            old_pairs.append((int(a), int(b)))
                        elif b not in cset0:
                            old_pairs.append((int(a), int(b)))
            # tangent bases
            Xc = best[np.array(cluster)]
            Bs = _tangent_bases(Xc)
            DXs = _null_samples(best, cluster, Bs, rng, nsamp=4, thetas=(0.05, 0.15))
        except Exception:
            continue
        cands = []
        for DX in DXs:
            if time.time() > DEAD - 12.0:
                break
            try:
                Xp = best.copy()
                for k, g in enumerate(cluster):
                    v = best[g] + DX[k]
                    nr = np.linalg.norm(v)
                    if not np.isfinite(nr) or nr == 0:
                        v = best[g]
                    else:
                        v = v / nr
                    Xp[g] = v
                # cluster-only softmax + penalty (bounded steps for speed)
                Xp2 = _cluster_relax(Xp, cluster, steps=420, lr=0.015, T0=0.010, T1=0.0012,
                                    old_pairs=old_pairs, thresh=0.38, pw=25.0, rng=rng)
                cands.append(Xp2)
            except Exception:
                continue
        if not cands:
            continue
        # pick best pre-LP candidate
        cands.sort(key=lambda Xc_: _max_cos(Xc_))
        top = cands[0]
        # strictly time-capped LP polish ~6s per cluster
        lp_t = 6.0
        if DEAD - time.time() < lp_t + 4.0:
            lp_t = max(2.0, DEAD - time.time() - 3.0)
        try:
            Xpol, vpol, marg2, I2, J2 = _lp_with_duals(top, lp_t, deadline=DEAD - 2.0)
        except Exception:
            continue
        if vpol < best_val - 1e-12:
            best = Xpol.copy()
            best_val = vpol
            mass = _dual_mass(n, marg2, I2, J2, best)
            # keep tried for diversity but best updated
        else:
            # optionally update mass to newest if it has larger dual info? keep best's mass
            # if vpol close, still keep best; do not update mass
            pass
        # safety: if time low break
        if DEAD - time.time() < 10.0:
            break
    best = _norm(best)
    # sanitize
    nr = np.linalg.norm(best, axis=1)
    bad = ~(np.isfinite(nr) & (nr > 1e-12))
    if np.any(bad):
        rb = rng.standard_normal(size=(int(np.sum(bad)), d))
        best[bad] = _norm(rb)
    best = _norm(best)
    return best
