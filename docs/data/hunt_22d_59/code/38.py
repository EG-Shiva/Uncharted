import numpy as np
import time

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _bases(X):
    n, d = X.shape
    B = np.empty((n, d, d-1), dtype=float)
    Eye = np.eye(d)
    for i in range(n):
        x = X[i]
        k = int(np.argmax(np.abs(x)))
        cols = [j for j in range(d) if j != k]
        M = Eye[:, cols] - np.outer(x, x[cols])
        Q, _ = np.linalg.qr(M)
        B[i] = Q[:, :d-1]
    return B

def _solve_lp(c, Aub, bub, bounds, timelim=8.0):
    from scipy.optimize import linprog
    try:
        res = linprog(c, A_ub=Aub, b_ub=bub, bounds=bounds, method="highs",
                      options={"time_limit": timelim, "presolve": True})
    except Exception:
        return None
    return res

def _extract_duals(res, m):
    # returns array length m or None
    try:
        for attr in ("ineqlin", "ineq_lin", "A_ub"):
            if hasattr(res, attr):
                obj = getattr(res, attr)
                # obj may be dict-like or object with marginals
                if isinstance(obj, dict) and "marginals" in obj:
                    v = np.asarray(obj["marginals"], dtype=float).ravel()
                    if v.size == m:
                        return np.abs(v)
                if hasattr(obj, "marginals"):
                    v = np.asarray(obj.marginals, dtype=float).ravel()
                    if v.size == m:
                        return np.abs(v)
                if hasattr(obj, "dual"):
                    v = np.asarray(obj.dual, dtype=float).ravel()
                    if v.size == m:
                        return np.abs(v)
        # scipy newer: res.marginals dict
        if hasattr(res, "marginals"):
            mg = getattr(res, "marginals")
            if isinstance(mg, dict):
                for k in ("ineqlin", "A_ub", "inequality"):
                    if k in mg:
                        e = mg[k]
                        if isinstance(e, dict) and "marginals" in e:
                            v = np.asarray(e["marginals"], dtype=float).ravel()
                            if v.size == m:
                                return np.abs(v)
                        if hasattr(e, "marginals"):
                            v = np.asarray(e.marginals, dtype=float).ravel()
                            if v.size == m:
                                return np.abs(v)
    except Exception:
        pass
    # try res.con / res.slack based fallback -> None
    return None

def _adam_polish(Y0, steps, beta0, beta1, lr0, lr1):
    n, d = Y0.shape
    Y = Y0.copy()
    M = np.zeros_like(Y)
    V = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    G0 = Y @ Y.T
    np.fill_diagonal(G0, -10.0)
    best_val = float(G0.max())
    best_Y = Y.copy()
    if steps <= 1:
        return best_Y, best_val
    lb = np.log(beta1 / beta0) / (steps - 1)
    ll = np.log(lr1 / lr0) / (steps - 1)
    lbb0, llr0 = np.log(beta0), np.log(lr0)
    b1t, b2t = 1.0, 1.0
    for t in range(1, steps + 1):
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
        if not np.isfinite(s) or s <= 0:
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

def _forced_sequence(X0, force_set, M0, drop=2e-4, rise=1e-4, n_iter=6, radius=0.004, timelim=6.0):
    n, d = X0.shape
    dd = d - 1
    Xcur = X0.copy()
    fset = set(force_set)
    Nvars = n * dd + 1
    c = np.zeros(Nvars)
    c[-1] = 1.0
    for it in range(n_iter):
        B = _bases(Xcur)
        G = Xcur @ Xcur.T
        # candidate other pairs
        iu = np.triu_indices(n, 1)
        gvals = G[iu]
        # active threshold for others
        thr = M0 - 8e-4
        other_pairs = []
        other_g = []
        for k in range(len(gvals)):
            g = float(gvals[k])
            if g < thr:
                continue
            p, q = int(iu[0][k]), int(iu[1][k])
            if (p, q) in fset or (q, p) in fset:
                continue
            other_pairs.append((p, q))
            other_g.append(g)
        force_pairs = []
        force_g = []
        for (p, q) in fset:
            force_pairs.append((p, q))
            force_g.append(float(G[p, q]))
        mF = len(force_pairs)
        mO = len(other_pairs)
        m = mF + mO
        if m == 0:
            return Xcur
        Aub = np.zeros((m, Nvars), dtype=float)
        bub = np.zeros(m, dtype=float)
        # forced rows: coeff*a <= M0-drop-g
        for r in range(mF):
            p, q = force_pairs[r]
            g = force_g[r]
            up = B[p].T @ Xcur[q]
            vq = B[q].T @ Xcur[p]
            Aub[r, p*dd:(p+1)*dd] = up
            Aub[r, q*dd:(q+1)*dd] = vq
            bub[r] = (M0 - drop) - g
        # other rows: coeff*a - t <= -g
        for r2 in range(mO):
            r = mF + r2
            p, q = other_pairs[r2]
            g = other_g[r2]
            up = B[p].T @ Xcur[q]
            vq = B[q].T @ Xcur[p]
            Aub[r, p*dd:(p+1)*dd] = up
            Aub[r, q*dd:(q+1)*dd] = vq
            Aub[r, -1] = -1.0
            bub[r] = -g
        bounds = [(-radius, radius)] * (n*dd) + [(M0 - 0.005, M0 + rise)]
        res = _solve_lp(c, Aub, bub, bounds, timelim=timelim)
        if res is None or not hasattr(res, "x") or res.x is None:
            return None
        try:
            status = int(getattr(res, "status", 0))
        except Exception:
            status = 0
        if status != 0:
            return None
        xv = np.asarray(res.x, dtype=float).ravel()
        if xv.size != Nvars or not np.all(np.isfinite(xv)):
            return None
        a = xv[:-1].reshape(n, dd)
        # apply
        Xn = np.empty_like(Xcur)
        for i in range(n):
            dx = B[i] @ a[i]
            v = Xcur[i] + dx
            nn = np.linalg.norm(v)
            if nn < 1e-12 or not np.isfinite(nn):
                return None
            Xn[i] = v / nn
        Xcur = Xn
        # early infeasibility check on true values
        Gc = Xcur @ Xcur.T
        np.fill_diagonal(Gc, -2.0)
        # if others blew up far beyond allowed, abort
        if float(Gc.max()) > M0 + rise + 5e-4:
            # still allow to continue? abort this trial
            pass
    return Xcur

def solve():
    t0 = time.time()
    np.random.seed(0)
    DEADLINE = 165.0
    def left(b=0.0):
        return (time.time() - t0) < (DEADLINE - b)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a = _normed(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        best_X = _normed(np.random.randn(59, 22))
    else:
        vals = [_max_cos(c) for c in cands]
        best_X = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(best_X)
    M0_global = best_val
    X_inc = best_X.copy()

    n, d = 59, 22
    dd = d - 1
    # --- initial trust LP to get duals ---
    duals = None
    active_pairs = []
    try:
        B0 = _bases(X_inc)
        G0 = X_inc @ X_inc.T
        iu = np.triu_indices(n, 1)
        gv = G0[iu]
        mcur = float(np.max(np.where(np.eye(n, dtype=bool), -2.0, G0)))
        # pairs within 1e-5 of max
        sel = np.where(gv >= mcur - 1e-5)[0]
        ap = [(int(iu[0][k]), int(iu[1][k])) for k in sel]
        ag = [float(gv[k]) for k in sel]
        active_pairs = ap
        m = len(ap)
        if m > 0 and left(30.0):
            Nvars = n * dd + 1
            c = np.zeros(Nvars); c[-1] = 1.0
            Aub = np.zeros((m, Nvars))
            bub = np.zeros(m)
            for r in range(m):
                p, q = ap[r]; g = ag[r]
                up = B0[p].T @ X_inc[q]
                vq = B0[q].T @ X_inc[p]
                Aub[r, p*dd:(p+1)*dd] = up
                Aub[r, q*dd:(q+1)*dd] = vq
                Aub[r, -1] = -1.0
                bub[r] = -g
            radius0 = 0.004
            bounds = [(-radius0, radius0)] * (n*dd) + [(-1.0, 1.0)]
            res = _solve_lp(c, Aub, bub, bounds, timelim=8.0)
            if res is not None and res.x is not None:
                duals = _extract_duals(res, m)
    except Exception:
        duals = None

    rng = np.random.RandomState(0)
    # build subsets of 20-50 contacts
    subsets = []
    try:
        if duals is not None and len(duals) == len(active_pairs) and np.any(np.isfinite(duals)):
            order = np.argsort(-np.asarray(duals))
            ap_arr = active_pairs
            Klist = [30, 45, 25, 50, 20, 35]
            for K in Klist:
                K2 = min(K, len(ap_arr))
                subsets.append([ap_arr[order[i]] for i in range(K2)])
            # random weighted by duals among top 150
            topN = min(150, len(ap_arr))
            top_idx = order[:topN]
            w = np.asarray(duals)[top_idx]
            w = np.maximum(w, 0)
            if w.sum() <= 0:
                w = np.ones_like(w)
            w = w / w.sum()
            for rep in range(3):
                K2 = int(30 + rep * 5)
                K2 = min(K2, len(ap_arr))
                ch = rng.choice(topN, size=K2, replace=False, p=w)
                subsets.append([ap_arr[top_idx[c]] for c in ch])
            # point-concentrated: highest stress points
            stress = np.zeros(n)
            for r, (p, q) in enumerate(ap_arr):
                stress[p] += float(duals[r]); stress[q] += float(duals[r])
            for _ in range(2):
                piv = int(np.argmax(stress + rng.randn(n) * 1e-12))
                # take contacts incident to piv + random partner point
                piv2 = int(rng.randint(0, n))
                cand = [e for e in ap_arr if e[0] == piv or e[1] == piv or e[0] == piv2 or e[1] == piv2]
                rng.shuffle(cand)
                subsets.append(cand[:40] if len(cand) >= 20 else cand)
        else:
            raise ValueError("no duals")
    except Exception:
        # randomized subsets among tightest pairs
        try:
            G0 = X_inc @ X_inc.T
            iu = np.triu_indices(n, 1)
            gv = G0[iu]
            order = np.argsort(-gv)
            tight = [(int(iu[0][order[i]]), int(iu[1][order[i]])) for i in range(min(250, len(order)))]
            for K in [20, 30, 40, 50, 30, 35]:
                idx = rng.choice(len(tight), size=min(K, len(tight)), replace=False)
                subsets.append([tight[i] for i in idx])
        except Exception:
            subsets = []

    # forced trials
    drop, rise = 2e-4, 1e-4
    trial_id = 0
    for S in subsets:
        if not left(35.0):
            break
        if len(S) < 15:
            continue
        # normalize pair order
        S = [(min(p, q), max(p, q)) for (p, q) in S]
        # deduplicate preserving
        S = list(dict.fromkeys(S))
        trial_id += 1
        try:
            Xf = _forced_sequence(X_inc, S, M0_global, drop=drop, rise=rise, n_iter=6, radius=0.004, timelim=5.0)
        except Exception:
            Xf = None
        if Xf is None:
            continue
        # short Adam polish after forced move
        try:
            steps = 1800 if left(50.0) else 800
            Xp, vp = _adam_polish(Xf, steps=steps, beta0=30.0, beta1=320.0, lr0=0.006, lr1=0.0006)
        except Exception:
            continue
        if vp < best_val - 1e-12:
            best_val = vp
            best_X = Xp.copy()
            M0_global = best_val
            X_inc = best_X.copy()
            # refresh active? keep going with new incumbent
        if not left(12.0):
            break

    # extra random-subset trials if time remains and no improvement stress
    extra = 0
    while left(30.0) and extra < 4:
        extra += 1
        try:
            G0 = X_inc @ X_inc.T
            iu = np.triu_indices(n, 1)
            gv = G0[iu]
            order = np.argsort(-gv)
            tight = [(int(iu[0][order[i]]), int(iu[1][order[i]])) for i in range(min(300, len(order)))]
            K = int(20 + rng.rand() * 30)
            idx = rng.choice(len(tight), size=K, replace=False)
            S = [tight[i] for i in idx]
            Xf = _forced_sequence(X_inc, S, M0_global, drop=2e-4, rise=1e-4, n_iter=5, radius=0.005, timelim=4.0)
            if Xf is None:
                continue
            Xp, vp = _adam_polish(Xf, steps=1500, beta0=25.0, beta1=300.0, lr0=0.005, lr1=0.0005)
            if vp < best_val:
                best_val = vp
                best_X = Xp.copy()
                M0_global = best_val
                X_inc = best_X.copy()
        except Exception:
            continue

    # final fine polish on best
    if left(8.0):
        try:
            Xp, vp = _adam_polish(best_X, steps=2000, beta0=60.0, beta1=500.0, lr0=0.004, lr1=0.0003)
            if vp < best_val:
                best_val = vp
                best_X = Xp.copy()
        except Exception:
            pass

    best_X = _normed(best_X)
    return best_X
