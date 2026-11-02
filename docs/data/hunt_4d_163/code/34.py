import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _lp_polish(x, deadline, r0=2e-3, cap=1400):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                sel[int(np.argmax(c))] = True
                cnt = 1
            if cnt > cap:
                idx = np.argsort(-c)[:cap]
                s2 = np.zeros_like(sel)
                s2[idx] = True
                sel = s2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            cols[:, :d] = I[:, None] * d + ar
            cols[:, d:2 * d] = J[:, None] * d + ar
            cols[:, 2 * d] = n * d
            cols = cols.ravel()
            vals = np.empty((p, 2 * d + 1), dtype=float)
            vals[:, :d] = x[J]
            vals[:, d:2 * d] = x[I]
            vals[:, 2 * d] = -1.0
            vals = vals.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x

def _lbfgs(x_init, deadline, taus=(0.004, 0.0012, 0.0005), iters=(90, 90, 110)):
    x = _norm(x_init)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.3:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            GY = (GX - (GX * X).sum(axis=1, keepdims=True) * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 4),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
        except Exception:
            break
    Yf = y.reshape(n, d)
    return _norm(Yf)

def _anneal(x0, steps, T0, T1, lr, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 63) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lr_k = lr * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        Gr = (E @ x) / S
        Gr -= (Gr * x).sum(axis=1, keepdims=True) * x
        vel = mu * vel + (1.0 - mu) * Gr
        x = x - lr_k * vel
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.standard_normal((n, d)) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x = _norm(x)
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max()), G

def _find_hole(Xf, rng, n_samples=20000):
    Y = rng.standard_normal((n_samples, 4))
    Y = _norm(Y)
    D = Y @ Xf.T
    vals = D.max(axis=1)
    idx = int(np.argmin(vals))
    return Y[idx], float(vals[idx])

def _contact_hash(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    A = G > m - tol
    deg = sorted(np.count_nonzero(A, axis=1).tolist())
    e = int(np.count_nonzero(A) // 2)
    q = int(round(m * 1e9))
    return (e, tuple(deg), q)

def _rand_rot(rng):
    A = rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    Q = Q * np.sign(d + 1e-30)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _icp_align(A, B, rng, restarts=3, iters=22):
    n = A.shape[0]
    bestR = np.eye(4)
    bestS = -1e99
    for r in range(restarts + 1):
        if r == 0:
            R = np.eye(4)
        else:
            R = _rand_rot(rng)
        for _ in range(iters):
            Br = B @ R
            S = Br @ A.T
            idx = np.argmax(S, axis=1)
            Ac = A[idx]
            M = B.T @ Ac
            try:
                U, _, Vt = np.linalg.svd(M, full_matrices=False)
                Rn = U @ Vt
            except Exception:
                break
            if np.linalg.det(Rn) < 0:
                U[:, -1] *= -1
                Rn = U @ Vt
            R = R @ Rn
            # need update: B@R vs? our iteration uses B@R; compose correctly:
            # we solved for delta on Br? simplify: recompute with new R in next loop
            # Actually M = B.T @ Ac gives optimal R directly (not delta), so set R=Rn
            R = Rn
            break_ind = _
            # run a few refinement steps with R=Rn then assignment again
            for __ in range(6):
                Br2 = B @ R
                S2 = Br2 @ A.T
                idx2 = np.argmax(S2, axis=1)
                Ac2 = A[idx2]
                M2 = B.T @ Ac2
                try:
                    U2, _, Vt2 = np.linalg.svd(M2, full_matrices=False)
                    Rn2 = U2 @ Vt2
                except Exception:
                    break
                if np.linalg.det(Rn2) < 0:
                    U2[:, -1] *= -1
                    Rn2 = U2 @ Vt2
                if np.max(np.abs(Rn2 - R)) < 1e-9:
                    R = Rn2
                    break
                R = Rn2
            break
        Br = B @ R
        S = Br @ A.T
        sc = float(S.max(axis=1).mean())
        if sc > bestS:
            bestS = sc
            bestR = R.copy()
    return bestR

def _greedy_maximin(U, k, rng):
    M = U.shape[0]
    G = U @ U.T
    np.fill_diagonal(G, -2.0)
    # start from most isolated point (min max-dot) or random
    pm = G.max(axis=1)
    s0 = int(np.argmin(pm))
    # randomize start occasionally for diversity
    if rng.random() < 0.4:
        s0 = int(rng.integers(0, M))
    sel = [s0]
    selmask = np.zeros(M, dtype=bool)
    selmask[s0] = True
    # best[i] = max dot to selected set
    best = G[:, s0].copy()
    for _ in range(1, k):
        # choose unselected with minimal best (most distant)
        cand = np.where(~selmask)[0]
        j = cand[int(np.argmin(best[cand]))]
        sel.append(int(j))
        selmask[j] = True
        if len(sel) >= k:
            break
        best = np.maximum(best, G[:, j])
    return np.array(sel, dtype=int)

def _swap_improve(U, sel, nosel, iters=60, rng=None):
    # try to lower max cosine by swaps
    sel = sel.copy()
    def val(s):
        X = U[np.array(s)]
        return _max_cos(X)
    cur = val(sel)
    set_sel = sel.tolist()
    set_no = nosel.tolist()
    for _ in range(iters):
        X = U[np.array(set_sel)]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m <= cur:
            cur = m
        # find worst pair
        ij = np.unravel_index(int(np.argmax(G)), G.shape)
        worst = [ij[0], ij[1]]
        improved = False
        for wpos in worst:
            widx = set_sel[wpos]
            # try random candidates from nosel
            order = np.argsort(U[set_no] @ U[widx])  # most distant first
            for t in order[:12]:
                cand = set_no[int(t)]
                trial = set_sel.copy()
                trial[wpos] = cand
                v = _max_cos(U[np.array(trial)])
                if v < cur - 1e-12:
                    # do swap
                    old = set_sel[wpos]
                    set_sel[wpos] = cand
                    # update no list
                    loc = set_no.index(cand)
                    set_no[loc] = old
                    cur = v
                    improved = True
                    break
            if improved:
                break
        if not improved:
            # random swap attempt
            if rng is not None:
                a = int(rng.integers(0, len(set_sel)))
                b = int(rng.integers(0, len(set_no)))
                trial = set_sel.copy()
                trial[a] = set_no[b]
                v = _max_cos(U[np.array(trial)])
                if v < cur:
                    old = set_sel[a]
                    set_sel[a] = set_no[b]
                    set_no[b] = old
                    cur = v
                else:
                    break
            else:
                break
    return np.array(set_sel, dtype=int), cur

def _quick_polish(X, deadline, rng, do_lp=True):
    # anneal + lbfgs + (optional short lp)
    rem = deadline - time.time()
    if rem <= 0.2:
        return X, _max_cos(X)
    steps = 500 if rem > 3 else 250
    try:
        Xa, _ = _anneal(X, steps, 0.004, 0.0004, 0.35, 0.85, rng, noise0=0.002, deadline=deadline - 0.6)
    except Exception:
        Xa = X
    if time.time() > deadline - 0.5:
        return Xa, _max_cos(Xa)
    try:
        Xb = _lbfgs(Xa, deadline - (0.4 if do_lp else 0.05), taus=(0.003, 0.0008), iters=(70, 90))
    except Exception:
        Xb = Xa
    if do_lp and time.time() < deadline - 0.3:
        try:
            Xb = _lp_polish(Xb, deadline, r0=1e-3)
        except Exception:
            pass
    try:
        v = _max_cos(Xb)
    except Exception:
        return X, 1.0
    return Xb, v

def solve():
    t0 = time.time()
    DEAD = t0 + 174.0
    SEARCH_END = t0 + 146.0
    rng = np.random.default_rng(1634)
    best_x = None
    best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (163, 4):
                continue
            a = _norm(a)
            if not np.all(np.isfinite(a)):
                continue
            v = _max_cos(a)
            if v < best_v:
                best_v = v
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = _norm(rng.standard_normal((163, 4)))
        best_v = _max_cos(best_x)
    archive = []
    hashes = set()
    def add_arch(X, v):
        try:
            h = _contact_hash(X)
        except Exception:
            h = None
        archive.append((v, X.copy(), h))
        if h is not None:
            hashes.add(h)
    add_arch(best_x, best_v)
    # light diverse seeds by perturbation + polish
    for si in range(2):
        if time.time() > SEARCH_END - 20:
            break
        try:
            cur = best_x + rng.standard_normal(best_x.shape) * (0.05 + 0.04 * si)
            cur = _norm(cur)
            Xp, vp = _quick_polish(cur, min(SEARCH_END, time.time() + 6.0), rng, do_lp=True)
            add_arch(Xp, vp)
            if vp < best_v:
                best_v, best_x = vp, Xp.copy()
        except Exception:
            continue
    # N-continuation paths back to 163
    def trace_path(startX, Ns, tag):
        nonlocal best_v, best_x
        X = startX.copy()
        for N in Ns:
            if time.time() > SEARCH_END - 8:
                break
            ncur = X.shape[0]
            if N > ncur:
                # insert into holes
                for _ in range(N - ncur):
                    h, _ = _find_hole(X, rng, n_samples=18000)
                    # small jitter on hole
                    h = _norm((h + rng.standard_normal(4) * 0.005)[None, :])[0]
                    X = np.vstack([X, h[None, :]])
            elif N < ncur:
                # delete worst-contact points
                try:
                    ppm, _, _ = _per_point_max(X)
                    k = ncur - N
                    worst = np.argsort(ppm)[-k:]
                    mask = np.ones(ncur, dtype=bool)
                    mask[worst] = False
                    X = X[mask]
                except Exception:
                    X = X[:N]
            # repolish at this N (short)
            try:
                dl = min(SEARCH_END, time.time() + 5.0)
                X, v = _quick_polish(X, dl, rng, do_lp=(X.shape[0] == 163))
                if X.shape[0] == 163:
                    add_arch(X, v)
                    if v < best_v:
                        best_v, best_x = v, X.copy()
            except Exception:
                continue
        return X
    # path down then up: 163 -> 157 -> 163 (different deletions/insertions)
    if time.time() < SEARCH_END - 10:
        try:
            Ns_down_up = [162, 160, 158, 156, 158, 160, 162, 163]
            trace_path(best_x, Ns_down_up, "du")
        except Exception:
            pass
    # path up then down: 163 -> 171 -> 163
    if time.time() < SEARCH_END - 10:
        try:
            # start from second archive member if exists for diversity
            sX = archive[1][1] if len(archive) > 1 else best_x
            Ns_up_dn = [164, 166, 168, 170, 168, 166, 164, 163]
            trace_path(sX, Ns_up_dn, "ud")
        except Exception:
            pass
    # orthogonal crossover loop prioritising unexplored graphs
    # keep archive sorted
    n_cross = 0
    while time.time() < SEARCH_END - 4:
        try:
            archive.sort(key=lambda t: t[0])
            # keep top 8
            if len(archive) > 10:
                archive = archive[:10]
            if len(archive) < 2:
                break
            # pick two distinct parents: bias to diverse hashes / ranks
            # choose i,j randomly among top
            ii = int(rng.integers(0, min(len(archive), 6)))
            jj = int(rng.integers(0, min(len(archive), 6)))
            if ii == jj:
                jj = (jj + 1) % min(len(archive), 6)
            A = archive[ii][1]
            B = archive[jj][1]
            if A.shape[0] != 163 or B.shape[0] != 163:
                continue
            R = _icp_align(A, B, rng, restarts=2, iters=8)
            Br = _norm(B @ R)
            U = np.vstack([A, Br])  # 326 x 4
            # dedup very close duplicates
            sel = _greedy_maximin(U, 163, rng)
            allidx = np.arange(U.shape[0])
            nosel = np.array([i for i in allidx if i not in set(sel.tolist())], dtype=int)
            sel2, _ = _swap_improve(U, sel, nosel, iters=25, rng=rng)
            child = _norm(U[sel2])
            # anneal + polish child
            dl = min(SEARCH_END, time.time() + 5.5)
            Xc, vc = _quick_polish(child, dl, rng, do_lp=True)
            # extra anneal with different temp if not improved
            if vc >= best_v - 1e-9 and time.time() < SEARCH_END - 5:
                try:
                    Xa2, _ = _anneal(child, 600, 0.006, 0.0003, 0.4, 0.8, rng, noise0=0.004, deadline=dl - 1.0)
                    Xb2 = _lbfgs(Xa2, dl - 0.5)
                    Xb2 = _lp_polish(Xb2, dl, r0=1e-3)
                    vc2 = _max_cos(Xb2)
                    if vc2 < vc:
                        Xc, vc = Xb2, vc2
                except Exception:
                    pass
            h = _contact_hash(Xc)
            is_new = (h not in hashes)
            add_arch(Xc, vc)
            if vc < best_v:
                best_v, best_x = vc, Xc.copy()
            else:
                # keep novel graphs even if slightly worse to preserve diversity
                pass
            n_cross += 1
            if n_cross > 14:
                # force exploitation: perturb best with hole ejection occasionally
                if n_cross % 4 == 0 and time.time() < SEARCH_END - 6:
                    try:
                        ppm, mval, G = _per_point_max(best_x)
                        k = 3
                        worst = np.argsort(ppm)[-k:]
                        mask = np.ones(163, dtype=bool)
                        mask[worst] = False
                        Xk = best_x[mask]
                        Xa = Xk.copy()
                        for _ in range(k):
                            hh, _ = _find_hole(Xa, rng, n_samples=18000)
                            Xa = np.vstack([Xa, hh[None, :]])
                        Xp, vp = _quick_polish(_norm(Xa), min(SEARCH_END, time.time() + 5.0), rng, True)
                        add_arch(Xp, vp)
                        if vp < best_v:
                            best_v, best_x = vp, Xp.copy()
                    except Exception:
                        pass
        except Exception:
            continue
    # final polish best
    try:
        archive.sort(key=lambda t: t[0])
        if archive and archive[0][0] < best_v:
            best_v, best_x = archive[0][0], archive[0][1].copy()
    except Exception:
        pass
    try:
        rem = DEAD - time.time()
        if rem > 6:
            sd = time.time() + min(12.0, rem - 7.0)
            best_x = _lbfgs(best_x, sd, taus=(0.004, 0.0015, 0.0006, 0.0003), iters=(90, 90, 90, 80))
    except Exception:
        pass
    try:
        best_x = _lp_polish(best_x, DEAD - 0.4, r0=2e-3)
    except Exception:
        pass
    best_x = _norm(best_x)
    return best_x
