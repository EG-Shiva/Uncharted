import time
import numpy as np

def _max_cos(Y):
    G = Y @ Y.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _adam_polish(Y0, steps, beta0, beta1, lr0, lr1, deadline=1e18):
    n, d = Y0.shape
    Y = Y0.copy()
    M = np.zeros_like(Y)
    V = np.zeros_like(Y)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    G0 = Y @ Y.T
    np.fill_diagonal(G0, -10.0)
    best_val = float(G0.max())
    best_Y = Y.copy()
    if steps <= 1:
        return best_Y, best_val
    lb = np.log(beta1 / beta0) / (steps - 1)
    ll = np.log(lr1 / lr0) / (steps - 1)
    lbb0 = np.log(beta0)
    llr0 = np.log(lr0)
    b1t = 1.0
    b2t = 1.0
    for t in range(1, steps + 1):
        if t % 200 == 0 and time.time() > deadline:
            break
        beta = float(np.exp(lbb0 + lb * (t - 1)))
        lr = float(np.exp(llr0 + ll * (t - 1)))
        G = Y @ Y.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        if gmax < best_val:
            best_val = gmax
            best_Y = Y.copy()
        G -= gmax
        G *= beta
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        s = float(G.sum())
        if s <= 0 or not np.isfinite(s):
            continue
        G /= s
        Gr = G @ Y
        dot = np.sum(Gr * Y, axis=1, keepdims=True)
        Gr -= Y * dot
        b1t *= b1
        b2t *= b2
        M *= b1
        M += (1.0 - b1) * Gr
        V *= b2
        V += (1.0 - b2) * (Gr * Gr)
        Mh = M / (1.0 - b1t)
        Vh = V / (1.0 - b2t)
        Y -= lr * Mh / (np.sqrt(Vh) + eps)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Y /= rn
    Gf = Y @ Y.T
    np.fill_diagonal(Gf, -10.0)
    vf = float(Gf.max())
    if vf < best_val:
        best_val = vf
        best_Y = Y.copy()
    return best_Y, best_val

def _tangent_Vs(X):
    n, d = X.shape
    dm = d - 1
    Vs = np.empty((n, d, dm), dtype=np.float64)
    for i in range(n):
        x = X[i]
        k = int(np.argmax(np.abs(x)))
        idx = [j for j in range(d) if j != k]
        idx = np.array(idx, dtype=int)
        # V[a,c] = delta_{a,idx[c]} - x[a]*x[idx[c]]
        Vi = -np.outer(x, x[idx])
        # add 1 at (idx[c], c)
        Vi[idx, np.arange(dm)] += 1.0
        Vs[i] = Vi
    return Vs

def _pairs_above(X, thresh):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    ii, jj = np.where(np.triu(G > thresh, k=1))
    c = G[ii, jj]
    order = np.argsort(-c)
    return ii[order], jj[order], c[order], G

def _solve_linprog(A, b, cobj, bounds):
    try:
        from scipy.optimize import linprog
        res = linprog(cobj, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        return res
    except Exception:
        return None

def _get_marginals(res, m):
    try:
        if res is None:
            return None
        # try various scipy layouts
        if hasattr(res, "ineqlin"):
            iq = res.ineqlin
            if hasattr(iq, "marginals"):
                v = np.asarray(iq.marginals, dtype=float).ravel()
                if v.size == m:
                    return v
            try:
                v = np.asarray(iq["marginals"], dtype=float).ravel()
                if v.size == m:
                    return v
            except Exception:
                pass
        try:
            d = dict(res)
            if "ineqlin" in d:
                iq = d["ineqlin"]
                try:
                    v = np.asarray(iq.marginals, dtype=float).ravel()
                    if v.size == m:
                        return v
                except Exception:
                    pass
        except Exception:
            pass
        if hasattr(res, "marginals"):
            v = np.asarray(res.marginals, dtype=float).ravel()
            if v.size >= m:
                return v[:m]
        # fallback: use slack complementarity proxy -> None
        return None
    except Exception:
        return None

def _standard_lp_step(X, Mref, window, trust):
    n, d = X.shape
    N = n * (d - 1)
    Vs = _tangent_Vs(X)
    thresh = Mref - window
    ii, jj, cvals, _ = _pairs_above(X, thresh)
    m = len(ii)
    if m == 0:
        return None, None, None
    # cap constraints to keep LP small: keep tightest 1200
    if m > 1200:
        ii, jj, cvals = ii[:1200], jj[:1200], cvals[:1200]
        m = 1200
    dm = d - 1
    A = np.zeros((m, N + 1), dtype=np.float64)
    b = np.zeros(m, dtype=np.float64)
    xj_all = X[jj]  # (m,d)
    xi_all = X[ii]
    for r in range(m):
        i = int(ii[r]); j = int(jj[r])
        Vi = Vs[i]  # d x dm
        Vj = Vs[j]
        # Vi^T x_j
        A[r, i*dm:(i+1)*dm] = Vi.T @ X[j]
        A[r, j*dm:(j+1)*dm] = Vj.T @ X[i]
        A[r, N] = -1.0
        b[r] = -float(cvals[r])
    cobj = np.zeros(N + 1, dtype=np.float64)
    cobj[N] = 1.0
    blo = Mref - 0.01
    bhi = Mref + 0.001
    bounds = [(-trust, trust)] * N + [(blo, bhi)]
    res = _solve_linprog(A, b, cobj, bounds)
    if res is None or getattr(res, "status", 1) != 0 or res.x is None:
        return None, (ii, jj, cvals), None
    marg = _get_marginals(res, m)
    y = np.asarray(res.x[:N], dtype=float)
    U = np.zeros_like(X)
    for i in range(n):
        U[i] = Vs[i] @ y[i*dm:(i+1)*dm]
    Xn = _normalize(X + U)
    return Xn, (ii, jj, cvals), marg

def _forced_sequence(X0, M0, Sset, drop=2e-4, allow=1e-4, trust=0.01, iters=6):
    # Sset: set of (i,j) tuples with i<j to force down
    S = set()
    for a, b_ in Sset:
        i, j = (int(a), int(b_)) if a < b_ else (int(b_), int(a))
        S.add((i, j))
    Xcur = X0.copy()
    n, d = Xcur.shape
    dm = d - 1
    N = n * dm
    target = M0 - drop
    t_allow = M0 + allow
    for it in range(iters):
        Vs = _tangent_Vs(Xcur)
        G = Xcur @ Xcur.T
        np.fill_diagonal(G, -2.0)
        # pool: pairs above M0-0.002 (guard) plus S
        mask = G > (M0 - 0.002)
        pi, pj = np.where(np.triu(mask, k=1))
        # union with S
        # build dict for quick lookup of c
        # assemble lists
        plist = []
        for k in range(len(pi)):
            i = int(pi[k]); j = int(pj[k])
            plist.append((i, j, float(G[i, j])))
        for (i, j) in S:
            # add if not already above threshold
            if G[i, j] <= (M0 - 0.002):
                plist.append((i, j, float(G[i, j])))
        m = len(plist)
        if m == 0:
            break
        if m > 1500:
            # keep tightest 1500 (largest c), but always keep S
            # sort non-S by c
            s_lookup = S
            nonS = [t for t in plist if (t[0], t[1]) not in s_lookup]
            sL = [t for t in plist if (t[0], t[1]) in s_lookup]
            nonS.sort(key=lambda t: -t[2])
            keep = 1500 - len(sL)
            plist = sL + nonS[:max(keep, 0)]
            m = len(plist)
        # split into forced and free
        is_forced = np.array([(p[0], p[1]) in S for p in plist], dtype=bool)
        A = np.zeros((m, N + 1), dtype=np.float64)
        b = np.zeros(m, dtype=np.float64)
        for r, (i, j, c) in enumerate(plist):
            A[r, i*dm:(i+1)*dm] = Vs[i].T @ Xcur[j]
            A[r, j*dm:(j+1)*dm] = Vs[j].T @ Xcur[i]
            if is_forced[r]:
                A[r, N] = 0.0
                b[r] = target - c
            else:
                A[r, N] = -1.0
                b[r] = -c
        cobj = np.zeros(N + 1, dtype=np.float64)
        cobj[N] = 1.0
        bounds = [(-trust, trust)] * N + [(M0 - 0.01, t_allow)]
        res = _solve_linprog(A, b, cobj, bounds)
        if res is None or getattr(res, "status", 1) != 0 or res.x is None:
            break
        y = np.asarray(res.x[:N], dtype=float)
        if not np.all(np.isfinite(y)):
            break
        U = np.zeros_like(Xcur)
        for i in range(n):
            U[i] = Vs[i] @ y[i*dm:(i+1)*dm]
        Xcur = _normalize(Xcur + U)
        # early stop if forced satisfied with margin in actual (nonlinear) values
        G2 = Xcur @ Xcur.T
        mx = -1
        for (i, j) in S:
            v = float(G2[i, j])
            if v > mx:
                mx = v
        # if all forced below target+5e-5, we can stop forcing early
        if mx < target + 5e-05:
            break
    return Xcur

def solve():
    rng = np.random.default_rng(0)
    t0 = time.time()
    deadline = t0 + 165.0

    X0 = None
    best0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (59, 22):
                continue
            if not np.all(np.isfinite(a)):
                continue
            a = _normalize(a)
            v = _max_cos(a)
            if v < best0:
                best0 = v
                X0 = a.copy()
        except Exception:
            continue
    if X0 is None:
        X0 = _normalize(rng.standard_normal((59, 22)))
        best0 = _max_cos(X0)

    x_best = X0.copy()
    best_val = best0
    M0 = best_val

    # --- standard trust LP to get equilibrium stresses / marginals ---
    tight_ii = tight_jj = tight_c = None
    marg = None
    try:
        if time.time() < deadline - 30:
            _, info, marg_ = _standard_lp_step(x_best, M0, window=1e-5, trust=0.01)
            if info is not None:
                tight_ii, tight_jj, tight_c = info
                marg = marg_
    except Exception:
        marg = None

    # build contact list within 1e-5
    try:
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        ii0, jj0 = np.where(np.triu(G > mcur - 1e-5, k=1))
        contacts = list(zip([int(a) for a in ii0], [int(a) for a in jj0]))
        cvals0 = [float(G[i, j]) for (i, j) in contacts]
    except Exception:
        contacts = []
        cvals0 = []

    # order by marginals if available
    order = None
    if marg is not None and tight_ii is not None and len(contacts) > 0:
        try:
            # marg corresponds to tight_ii/jj ordering (sorted by -c)
            am = np.abs(np.asarray(marg, dtype=float))
            idx = np.argsort(-am)
            # map to contact tuples
            ranked = [(int(tight_ii[k]), int(tight_jj[k])) for k in idx if np.isfinite(am[k])]
            # keep only those still in contacts set or just use ranked
            if len(ranked) >= 20:
                ranked_contacts = ranked
            else:
                ranked_contacts = contacts
            order = ranked_contacts
        except Exception:
            order = contacts
    else:
        order = contacts
    if order is None or len(order) == 0:
        order = contacts
    if len(order) == 0:
        # fallback: tightest 200 pairs
        try:
            G = x_best @ x_best.T
            np.fill_diagonal(G, -2.0)
            tri = np.triu(G, k=1)
            flat = tri[np.triu_indices(59, k=1)]
            idx = np.argsort(-flat)[:200]
            rows, cols = np.triu_indices(59, k=1)
            order = [(int(rows[k]), int(cols[k])) for k in idx]
        except Exception:
            order = []

    n_trials = 0
    # forced facet trials with several random stress/contact subsets
    # firm clock limit: leave 18s for final polish
    trial = 0
    # predefine subset sizes 20..50
    while time.time() < deadline - 22.0 and trial < 40:
        trial += 1
        try:
            if len(order) == 0:
                break
            ssize = int(rng.integers(20, 51))
            ssize = min(ssize, len(order))
            # selection: mix of top-marginal and random
            if marg is not None and len(order) > ssize:
                # order is already ranked by |marginal|
                top_pool = order[:min(len(order), 100)]
                # take half from top deterministically (shifted by trial) + random rest
                k_top = ssize // 2
                start = ((trial - 1) * 7) % max(1, len(top_pool) - k_top + 1) if len(top_pool) > k_top else 0
                sel = top_pool[start:start + k_top]
                # fill remainder randomly from order
                rem = ssize - len(sel)
                if rem > 0:
                    cand = [p for p in order if p not in sel]
                    if len(cand) > 0:
                        ch = rng.choice(len(cand), size=min(rem, len(cand)), replace=False)
                        sel = sel + [cand[k] for k in ch]
                Sset = sel
            else:
                ch = rng.choice(len(order), size=ssize, replace=False)
                Sset = [order[k] for k in ch]
            # sequence of relinearized LPs
            Xf = _forced_sequence(x_best, M0, Sset, drop=2e-4, allow=1e-4, trust=0.01, iters=6)
            vf = _max_cos(Xf)
            # short Adam polish after forced move (even if vf worse, polish may find new basin)
            Xp, vp = _adam_polish(Xf, steps=1800, beta0=80.0, beta1=600.0, lr0=0.005, lr1=0.0008, deadline=deadline - 18.0)
            # also try a tiny extra contact-style greedy? skip, keep time
            if vp < best_val:
                best_val = vp
                x_best = Xp.copy()
                M0 = best_val
                # refresh order around new best (cheap, without LP every time)
                try:
                    G2 = x_best @ x_best.T
                    np.fill_diagonal(G2, -2.0)
                    m2 = float(G2.max())
                    a2, b2 = np.where(np.triu(G2 > m2 - 1e-5, k=1))
                    contacts2 = list(zip([int(a) for a in a2], [int(a) for a in b2]))
                    if len(contacts2) >= 20:
                        order = contacts2
                        # re-rank by random (marginals stale) -> keep as is, shuffle deterministically
                        # keep marg=None after improvement to use random subsets
                        marg = None
                except Exception:
                    pass
            elif vf < M0 + 5e-05:
                # even non-improving forced points may be useful as restart seeds occasionally
                # polish once more with different init? skip to save time
                pass
            n_trials += 1
        except Exception:
            continue

    # final polish of best
    try:
        if time.time() < deadline - 15.0:
            Xp, vp = _adam_polish(x_best, steps=6000, beta0=120.0, beta1=900.0, lr0=0.004, lr1=0.0004, deadline=deadline - 2.0)
            if vp < best_val:
                best_val = vp
                x_best = Xp.copy()
    except Exception:
        pass

    x_best = _normalize(x_best)
    return x_best
