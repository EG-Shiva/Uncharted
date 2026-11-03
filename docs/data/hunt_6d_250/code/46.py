import time
import itertools
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam_full(X0, steps, b0, b1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1m, b2m, eps = 0.9, 0.999, 1e-8
    t = 0
    lb0, lb1 = np.log(b0), np.log(b1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(lr0 + (lr1 - lr0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = float(np.max(G))
        G -= cm
        G *= beta
        np.clip(G, -30.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G))
        if S < 1e-100 or not np.isfinite(S):
            continue
        GX = (G @ X) * (2.0 / S)
        GX -= np.sum(GX * X, axis=1, keepdims=True) * X
        t += 1
        m = b1m * m + (1.0 - b1m) * GX
        v = b2m * v + (1.0 - b2m) * (GX * GX)
        c1 = 1.0 - b1m ** t
        c2 = 1.0 - b2m ** t
        step = (lr / c1) * m / (np.sqrt(v / c2) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 5.0 * lr + 1e-12
        step *= np.minimum(1.0, cap / sn)
        X -= step
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
        md = np.sum(m * X, axis=1, keepdims=True)
        m -= md * X
    return X

def _lp_polish(x, deadline, k_top=4500, r0=1.5e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline - 1.0 and radius > 1e-14:
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        m = float(np.max(c_all))
        sel = c_all > m - max(20.0 * radius, 1e-6)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > k_top:
            idx = np.argpartition(-c_all, k_top)[:k_top]
            I = iu0[idx]; J = iu1[idx]; c_sel = c_all[idx]
        else:
            I = iu0[sel]; J = iu1[sel]; c_sel = c_all[sel]
        p = I.shape[0]
        nv = n * d + 1
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        vv = _max_cos(y)
        if vv < best - 1e-12:
            x, best = y, vv
            radius = min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
        if time.time() > deadline - 1.5:
            break
    return x

def _build_orbit_data():
    # fixed blocks
    A12 = np.zeros((12, 6))
    k = 0
    for i in range(6):
        for s in (1.0, -1.0):
            A12[k, i] = s
            k += 1
    AA = []
    for (i, j) in itertools.combinations(range(6), 2):
        for sa, sb in itertools.product((1.0, -1.0), repeat=2):
            r = np.zeros(6); r[i] = sa; r[j] = sb
            AA.append(r)
    AA = np.array(AA)  # 60x6
    AAA = []
    for comb in itertools.combinations(range(6), 3):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            r = np.zeros(6)
            for idx, s in zip(comb, signs):
                r[idx] = s
            AAA.append(r)
    AAA = np.array(AAA)  # 160x6
    A6 = np.zeros((64, 6))
    for mask in range(64):
        for j in range(6):
            A6[mask, j] = 1.0 if ((mask >> j) & 1) else -1.0
    # AB patterns 120
    pA = []; pB = []; sA = []; sB = []
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            for sa, sb in itertools.product((1.0, -1.0), repeat=2):
                pA.append(i); pB.append(j); sA.append(sa); sB.append(sb)
    pA = np.array(pA, dtype=np.int64); pB = np.array(pB, dtype=np.int64)
    sA = np.array(sA); sB = np.array(sB)
    return A12, AA, AAA, A6, pA, pB, sA, sB

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    SYM_END = t0 + 62.0
    HOP_END = t0 + 145.0
    rng = np.random.default_rng(0)
    n, d = 250, 6
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape == (n, d):
                z = z / np.linalg.norm(z, axis=1, keepdims=True)
                if np.all(np.isfinite(z)):
                    cands.append(z)
        except Exception:
            pass
    if not cands:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)
    # ---- symmetric phase ----
    try:
        A12, AA, AAA, A6, pA, pB, sA, sB = _build_orbit_data()
    except Exception:
        A12 = AA = AAA = A6 = None
        pA = pB = sA = sB = None
    combos = [
        (["AB", "AB"], 10),
        (["AB", "A6", "AA"], 6),
        (["AAA", "AA", "A"], 18),
        (["AAA", "A6", "A"], 14),
    ]
    beta_sym = 2000.0
    for ci, (types, F) in enumerate(combos):
        if time.time() > SYM_END:
            break
        nAB = sum(1 for t in types if t == "AB")
        # fixed sizes
        try:
            # initial ratios and free points
            t_init = np.full(nAB, 1.0)
            # diversify per combo deterministically
            if ci == 1:
                t_init[:] = 0.85
            elif ci == 2:
                pass
            free_init = rng.standard_normal((F, d))
            x0 = np.concatenate([t_init, free_init.ravel()]) if nAB > 0 else free_init.ravel()
            bounds = ([(0.35, 2.8)] * nAB + [(None, None)] * (F * d)) if nAB > 0 else [(None, None)] * (F * d)
            # precompute fixed raw blocks list in order
            fixed_mats = []
            for tp in types:
                if tp == "A":
                    fixed_mats.append(("F", A12))
                elif tp == "AA":
                    fixed_mats.append(("F", AA))
                elif tp == "AAA":
                    fixed_mats.append(("F", AAA))
                elif tp == "A6":
                    fixed_mats.append(("F", A6))
                elif tp == "AB":
                    fixed_mats.append(("AB", None))
            def fg(Yflat, _fm=fixed_mats, _nAB=nAB, _F=F):
                rats = Yflat[:_nAB] if _nAB > 0 else np.zeros(0)
                free_raw = Yflat[_nAB:].reshape(_F, d) if _F > 0 else np.zeros((0, d))
                Rparts = []
                ai = 0
                for kind, mat in _fm:
                    if kind == "F":
                        Rparts.append(mat)
                    else:
                        t = float(rats[ai]); ai += 1
                        M = pA.shape[0]
                        Rm = np.zeros((M, 6))
                        Rm[np.arange(M), pA] = sA * 1.0
                        Rm[np.arange(M), pB] = sB * t
                        Rparts.append(Rm)
                if _F > 0:
                    Rparts.append(free_raw)
                R = np.vstack(Rparts) if len(Rparts) else np.zeros((0, d))
                nrm = np.linalg.norm(R, axis=1, keepdims=True)
                nrm = np.maximum(nrm, 1e-12)
                X = R / nrm
                G = X @ X.T
                np.fill_diagonal(G, -10.0)
                Mv = float(np.max(G))
                D = (G - Mv) * beta_sym
                np.clip(D, -30.0, 0.0, out=D)
                np.exp(D, out=D)
                np.fill_diagonal(D, 0.0)
                S = float(np.sum(D))
                if S <= 0 or not np.isfinite(S):
                    S = 1e-300
                loss = Mv + np.log(S) / beta_sym
                W = D / S
                GX = 2.0 * (W @ X)
                dot = np.sum(GX * X, axis=1, keepdims=True)
                GR = (GX - dot * X) / nrm
                # split grads
                grad_rats = np.zeros(_nAB)
                grad_free = np.zeros((_F, d))
                off = 0
                ai = 0
                for kind, mat in _fm:
                    if kind == "F":
                        mm = mat.shape[0]
                        off += mm
                    else:
                        M = pA.shape[0]
                        Gb = GR[off:off + M]
                        g = float(np.sum(Gb[np.arange(M), pB] * sB))
                        grad_rats[ai] = g
                        ai += 1
                        off += M
                if _F > 0:
                    grad_free = GR[off:off + _F]
                grad = np.concatenate([grad_rats, grad_free.ravel()]) if _nAB > 0 else grad_free.ravel()
                return loss, grad
            # run L-BFGS-B with limited budget
            maxiter = 120 if (nAB + F * d) > 60 else 200
            res = minimize(lambda y: fg(y), x0, jac=True, method="L-BFGS-B",
                           bounds=bounds,
                           options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-8, "maxfun": maxiter * 4})
            Yf = res.x
            # rebuild X
            rats = Yf[:nAB] if nAB > 0 else np.zeros(0)
            free_raw = Yf[nAB:].reshape(F, d) if F > 0 else np.zeros((0, d))
            Rparts = []
            ai = 0
            for kind, mat in fixed_mats:
                if kind == "F":
                    Rparts.append(mat.astype(float))
                else:
                    t = float(rats[ai]); ai += 1
                    M = pA.shape[0]
                    Rm = np.zeros((M, 6))
                    Rm[np.arange(M), pA] = sA * 1.0
                    Rm[np.arange(M), pB] = sB * t
                    Rparts.append(Rm)
            if F > 0:
                Rparts.append(free_raw)
            R = np.vstack(Rparts)
            R /= np.maximum(np.linalg.norm(R, axis=1, keepdims=True), 1e-12)
            # R should be 250 rows
            if R.shape[0] != 250:
                continue
            # break symmetry slightly then short Adam to evaluate true potential
            Xc = R / np.linalg.norm(R, axis=1, keepdims=True)
            # small symmetry-breaking noise + short Adam
            Xn = Xc + rng.standard_normal(Xc.shape) * 1e-3
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            try:
                Xa = _adam_full(Xn, 250, 2000.0, 20000.0, 0.004, 0.0005)
                va = _max_cos(Xa)
            except Exception:
                va = _max_cos(Xc); Xa = Xc
            v_raw = _max_cos(Xc)
            # keep global best from either raw symmetric or broken
            if v_raw < best_val and time.time() < HARD - 5:
                best, best_val = Xc, v_raw
            if va < best_val:
                best, best_val = Xa, va
        except Exception:
            continue
    # ---- symmetry-breaking full polish / basin hopping from global best ----
    iu0, iu1 = np.triu_indices(n, 1)
    # quick anneal of best
    if time.time() < HOP_END:
        try:
            cur = _adam_full(best, 600, 2000.0, 30000.0, 0.004, 0.0003)
            v = _max_cos(cur)
            if v < best_val:
                best, best_val = cur, v
        except Exception:
            pass
    trial = 0
    while time.time() < HOP_END and trial < 120:
        trial += 1
        if time.time() > HARD - 32.0:
            break
        Y = best.copy()
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            gv = G[iu0, iu1]
            kk = int(np.argmax(gv))
            wa, wb = int(iu0[kk]), int(iu1[kk])
            rowmax = G.max(axis=1)
        except Exception:
            wa, wb = 0, 1
            rowmax = np.zeros(n)
        mode = trial % 4
        if mode == 0:
            k = int(4 + rng.integers(0, 10))
            idx = np.argsort(-rowmax)[:k]
            sigma = 0.18 + 0.15 * float(rng.random())
            Y[idx] += rng.standard_normal((k, d)) * sigma
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps, b0, b1, lr0 = 650, 1500.0, 25000.0, 0.004
        elif mode == 1:
            k = int(6 + rng.integers(0, 10))
            try:
                sc = G[wa] + G[wb]
                idx = np.argsort(-sc)[:k]
                idx[0] = wa
                if k > 1:
                    idx[1] = wb
            except Exception:
                idx = rng.choice(n, size=k, replace=False)
            sigma = 0.12 + 0.12 * float(rng.random())
            Y[idx] += rng.standard_normal((k, d)) * sigma
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps, b0, b1, lr0 = 650, 2500.0, 35000.0, 0.0035
        elif mode == 2:
            Y += rng.standard_normal((n, d)) * 0.010
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            k = int(6 + rng.integers(0, 12))
            idx = rng.choice(n, size=k, replace=False)
            sigma = 0.28 + 0.20 * float(rng.random())
            Y[idx] += rng.standard_normal((k, d)) * sigma
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps, b0, b1, lr0 = 700, 2000.0, 30000.0, 0.004
        else:
            u = rng.standard_normal(d); u /= np.linalg.norm(u) + 1e-16
            dots = best @ u
            k = int(10 + rng.integers(0, 12))
            thr = np.sort(dots)[-k]
            sel = np.where(dots >= thr)[0]
            w = rng.standard_normal(d)
            w -= float(w @ u) * u
            wn = float(np.linalg.norm(w))
            if wn < 1e-12:
                w = np.zeros(d); w[0] = 1.0
            else:
                w /= wn
            shift = 0.07 + 0.08 * float(rng.random())
            Y[sel] += w * shift + rng.standard_normal((sel.shape[0], d)) * 0.05
            Y += rng.standard_normal((n, d)) * 0.007
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps, b0, b1, lr0 = 650, 2000.0, 40000.0, 0.0035
        try:
            Y2 = _adam_full(Y, steps, b0, b1, lr0, 0.0002)
        except Exception:
            continue
        try:
            v = _max_cos(Y2)
        except Exception:
            continue
        if v < best_val - 1e-12:
            best, best_val = Y2, v
    if time.time() < HARD - 28.0:
        try:
            Yf = _adam_full(best, 900, 8000.0, 80000.0, 0.002, 0.0001)
            vf = _max_cos(Yf)
            if vf < best_val:
                best, best_val = Yf, vf
        except Exception:
            pass
    if time.time() < HARD - 3.0:
        try:
            deadline = min(HARD - 1.0, time.time() + 22.0)
            Yl = _lp_polish(best, deadline, k_top=4500, r0=1.5e-3)
            vl = _max_cos(Yl)
            if vl < best_val:
                best, best_val = Yl, vl
            if time.time() < HARD - 6.0:
                try:
                    Yf2 = _adam_full(best, 450, 15000.0, 100000.0, 0.001, 0.0001)
                    v2 = _max_cos(Yf2)
                    if v2 < best_val:
                        best, best_val = Yf2, v2
                except Exception:
                    pass
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
