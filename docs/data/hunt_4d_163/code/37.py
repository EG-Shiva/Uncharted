import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _rand_rot(rng):
    A = rng.randn(4, 4)
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    Q = Q * np.sign(d + 1e-12)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _hopf_code(n_list, thetas, twists, phases, pattern, rng, rot=None):
    pts = []
    for nk, th, tw, ph in zip(n_list, thetas, twists, phases):
        nk = int(nk)
        if nk <= 0:
            continue
        ct, st = float(np.cos(th)), float(np.sin(th))
        ct = max(ct, 1e-3); st = max(st, 1e-3)
        cot = ct / st
        # target nx/ny = cot for square cells
        nx = int(round(np.sqrt(max(nk * cot, 1.0))))
        nx = max(1, min(nk, nx))
        ny = int((nk + nx - 1) // nx)
        ox, oy = float(ph[0]), float(ph[1])
        uu = []
        vv = []
        for ix in range(nx):
            for iy in range(ny):
                u = (ix + ox) / nx % 1.0
                v = (iy + oy) / ny % 1.0
                if pattern == "tri":
                    u = (u + 0.5 * (iy % 2) / nx) % 1.0
                uu.append(u); vv.append(v)
        uu = np.array(uu); vv = np.array(vv)
        if len(uu) > nk:
            idx = rng.choice(len(uu), nk, replace=False)
            uu = uu[idx]; vv = vv[idx]
        # twist shear + tiny jitter
        vv = (vv + tw * uu) % 1.0
        uu = (uu + rng.randn(len(uu)) * 0.002) % 1.0
        vv = (vv + rng.randn(len(vv)) * 0.002) % 1.0
        a = 2 * np.pi * uu
        b = 2 * np.pi * vv
        X = np.stack([ct * np.cos(a), ct * np.sin(a), st * np.cos(b), st * np.sin(b)], axis=1)
        pts.append(X)
    X = np.vstack(pts)
    # fix count mismatch
    if X.shape[0] > 163:
        idx = rng.choice(X.shape[0], 163, replace=False)
        X = X[idx]
    elif X.shape[0] < 163:
        extra = rng.randn(163 - X.shape[0], 4)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        X = np.vstack([X, extra])
    if rot is not None:
        X = X @ rot.T
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X

def _hurwitz_pool(max_norm=39, R=6):
    vecs = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            for c in range(-R, R + 1):
                for d in range(-R, R + 1):
                    n2 = a*a+b*b+c*c+d*d
                    if 0 < n2 <= max_norm:
                        vecs.append((float(a), float(b), float(c), float(d)))
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            for c in range(-R, R + 1):
                for d in range(-R, R + 1):
                    x = a + 0.5; y = b + 0.5; z = c + 0.5; w = d + 0.5
                    n2 = x*x+y*y+z*z+w*w
                    if 0 < n2 <= max_norm:
                        vecs.append((x, y, z, w))
    P = np.array(vecs, dtype=np.float64)
    nrm = np.linalg.norm(P, axis=1, keepdims=True)
    P = P / np.maximum(nrm, 1e-12)
    # dedup directions (keep antipodes distinct, collapse same ray)
    seen = set()
    keep = []
    for i in range(P.shape[0]):
        key = tuple(np.round(P[i], 5))
        if key not in seen:
            seen.add(key)
            keep.append(i)
    P = P[np.array(keep)]
    return P

def _greedy_select(P, m, rng):
    M = P.shape[0]
    seed = int(rng.randint(M))
    sel = [seed]
    selmask = np.zeros(M, dtype=bool)
    selmask[seed] = True
    cur = P @ P[seed]
    cur[seed] = 1e9
    for _ in range(1, m):
        # pick unselected minimizing cur max-dot
        best = -1; bestv = 1e9
        # argmin over unselected
        # use masked array
        tmp = cur.copy()
        tmp[selmask] = 1e9
        j = int(np.argmin(tmp))
        sel.append(j)
        selmask[j] = True
        if len(sel) >= m:
            break
        d = P @ P[j]
        cur = np.maximum(cur, d)
        cur[selmask] = 1e9
    S = P[np.array(sel)]
    return S, sel

def _swap_improve(P, sel_idx, iters, rng):
    sel_idx = list(sel_idx)
    m = len(sel_idx)
    S = P[np.array(sel_idx)].copy()
    def score_of(S_):
        G = S_ @ S_.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())
    cur_score = score_of(S)
    M = P.shape[0]
    selset = set(sel_idx)
    for _ in range(iters):
        G = S @ S.T
        np.fill_diagonal(G, -2.0)
        ij = int(np.argmax(G))
        i = ij // m; j = ij % m
        # alternate which endpoint to replace
        tgt = i if rng.rand() < 0.5 else j
        rest = np.delete(S, tgt, axis=0)
        # scores of all pool candidates vs rest
        D = P @ rest.T  # M x (m-1)
        mx = D.max(axis=1)
        # mask already selected (except tgt) to avoid duplicates
        for s in sel_idx:
            if s == sel_idx[tgt]:
                continue
            mx[s] = 1e9
        c = int(np.argmin(mx))
        if float(mx[c]) < cur_score - 1e-12:
            sel_idx[tgt] = c
            S[tgt] = P[c]
            cur_score = score_of(S)
        else:
            # try the other endpoint once
            tgt2 = j if tgt == i else i
            rest2 = np.delete(S, tgt2, axis=0)
            D2 = P @ rest2.T
            mx2 = D2.max(axis=1)
            for s in sel_idx:
                if s == sel_idx[tgt2]:
                    continue
                mx2[s] = 1e9
            c2 = int(np.argmin(mx2))
            if float(mx2[c2]) < cur_score - 1e-12:
                sel_idx[tgt2] = c2
                S[tgt2] = P[c2]
                cur_score = score_of(S)
            else:
                # random kick occasionally
                if rng.rand() < 0.1:
                    rc = int(rng.randint(M))
                    if rc not in selset:
                        # replace random point if it helps locally
                        rt = int(rng.randint(m))
                        rrest = np.delete(S, rt, axis=0)
                        v = float((rrest @ P[rc]).max())
                        if v < cur_score:
                            sel_idx[rt] = rc
                            S[rt] = P[rc]
                            cur_score = score_of(S)
    return S, cur_score

def _gd_softmax(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    best = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 255) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12:
            T = 1e-12
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _lp_polish(x, deadline, r0=1e-3, cap=1200):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) == 0:
                sel[np.argmax(c)] = True
            if int(sel.sum()) > cap:
                idx = np.argsort(-c)[:cap]
                s2 = np.zeros_like(sel)
                s2[idx] = True
                sel = s2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            tmpc = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None] * d + ar
            tmpc[:, d:2 * d] = J[:, None] * d + ar
            tmpc[:, 2 * d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2 * d] = x[I]
            cc[:, 2 * d] = -1.0
            vals = cc.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            vv = _max_cos(y)
            if vv < best:
                x, best = y, vv
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _lbfgs_smooth(x_init, deadline, taus=(0.003, 0.001, 0.00045), iters=(80, 80, 100)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.4:
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
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 5),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
            Yt = y.reshape(n, d)
            nt = np.maximum(np.linalg.norm(Yt, axis=1, keepdims=True), 1e-12)
            y = (Yt / nt).ravel()
        except Exception:
            break
    Xf = y.reshape(n, d)
    Xf /= np.maximum(np.linalg.norm(Xf, axis=1, keepdims=True), 1e-12)
    return Xf

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SCREEN_END = t0 + 95.0
    POLISH_END = DEADLINE - 4.0
    rng = np.random.RandomState(20261102)
    n = 163
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, 4) or not np.all(np.isfinite(a)):
                continue
            a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = rng.randn(n, 4)
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best_val = _max_cos(best_x)
    base_best = best_x.copy()

    cands = []
    # --- Hopf / Clifford torus structured inits ---
    Kvals = [4, 5, 6, 7, 8, 5, 6, 7]
    patts = ["square", "tri", "square", "tri", "square", "tri", "square", "square"]
    for ci in range(28):
        if time.time() > SCREEN_END - 30:
            break
        K = Kvals[ci % len(Kvals)]
        patt = patts[ci % len(patts)]
        # allocations summing to 163
        base = n // K
        alloc = np.full(K, base, dtype=int)
        rem = n - int(alloc.sum())
        for k in range(rem):
            alloc[rng.randint(K)] += 1
        # random rebalancing for some configs
        if ci % 3 == 2:
            mv = rng.randint(1, 4)
            for _ in range(mv):
                a_ = rng.randint(K); b_ = rng.randint(K)
                if alloc[a_] > 5:
                    alloc[a_] -= 1; alloc[b_] += 1
        qs = (np.arange(K) + 0.5) / K
        thetas = np.arcsin(np.sqrt(np.clip(qs, 1e-6, 1 - 1e-6)))
        thetas = thetas + rng.randn(K) * 0.04
        thetas = np.clip(thetas, 0.08, np.pi/2 - 0.08)
        twists = rng.rand(K)
        phases = rng.rand(K, 2)
        rot = _rand_rot(rng)
        X = _hopf_code(alloc, thetas, twists, phases, patt, rng, rot)
        v = _max_cos(X)
        cands.append((v, X))
    # --- Hurwitz / D4 pool + greedy maximin ---
    try:
        P = _hurwitz_pool(max_norm=39, R=6)
        # augment with random uniform to fill gaps
        R2 = rng.randn(2500, 4)
        R2 /= np.linalg.norm(R2, axis=1, keepdims=True)
        P = np.vstack([P, R2])
        for rep in range(3):
            if time.time() > SCREEN_END - 20:
                break
            S, sel = _greedy_select(P, n, rng)
            S2, sc2 = _swap_improve(P, sel, iters=60, rng=rng)
            v = _max_cos(S2)
            cands.append((v, S2.copy()))
    except Exception:
        pass
    # also add perturbed incumbent for safety
    try:
        pert = base_best + rng.randn(n, 4) * 0.05
        pert /= np.linalg.norm(pert, axis=1, keepdims=True)
        cands.append((_max_cos(pert), pert))
    except Exception:
        pass

    # quick screening polish
    cands.sort(key=lambda t: t[0])
    # keep top ~10 for screening
    cands = cands[:12]
    screened = []
    for v0, X0 in cands:
        if time.time() > SCREEN_END:
            break
        try:
            Xs, vs = _gd_softmax(X0, steps=1100, T0=0.010, T1=0.0006, lr0=0.036, mu=0.86,
                                 rng=rng, noise0=0.0015, deadline=SCREEN_END)
            screened.append((vs, Xs))
            if vs < best_val:
                best_val = vs; best_x = Xs.copy()
        except Exception:
            screened.append((v0, X0))
    if len(screened) == 0:
        screened = [(best_val, best_x.copy())]
    screened.sort(key=lambda t: t[0])
    # full polish top candidates + incumbent basin
    tops = screened[:3]
    # ensure incumbent included
    tops.append((best_val, best_x.copy()))
    # dedup-ish: keep best 3 unique
    tops = sorted(tops, key=lambda t: t[0])[:3]
    for v0, X0 in tops:
        if time.time() > POLISH_END - 2:
            break
        try:
            Xc = X0.copy()
            # stage 1 anneal
            Xc, vc = _gd_softmax(Xc, steps=2600, T0=0.008, T1=0.0004, lr0=0.034, mu=0.87,
                                 rng=rng, noise0=0.0008, deadline=POLISH_END - 8)
            if vc < best_val:
                best_val = vc; best_x = Xc.copy()
            # stage 2 lbfgs
            if time.time() < POLISH_END - 8:
                Xl = _lbfgs_smooth(Xc, POLISH_END - 6)
                vl = _max_cos(Xl)
                if vl < best_val:
                    best_val = vl; best_x = Xl.copy()
                    Xc = Xl
                elif vl < vc + 0.002:
                    Xc = Xl
            # stage 3 LP
            if time.time() < POLISH_END - 2:
                Xp, vp = _lp_polish(Xc, POLISH_END, r0=1e-3)
                if vp < best_val:
                    best_val = vp; best_x = Xp.copy()
            # stage 4 short re-anneal at low T to escape LP corner
            if time.time() < POLISH_END - 3:
                Xr, vr = _gd_softmax(best_x, steps=700, T0=0.0012, T1=0.0002, lr0=0.020, mu=0.88,
                                     rng=rng, noise0=0.0, deadline=POLISH_END - 1)
                if vr < best_val:
                    best_val = vr; best_x = Xr.copy()
                    if time.time() < POLISH_END - 1:
                        Xp2, vp2 = _lp_polish(best_x, POLISH_END, r0=5e-4)
                        if vp2 < best_val:
                            best_val = vp2; best_x = Xp2.copy()
        except Exception:
            continue
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    # tiny shrink guard
    return best_x
