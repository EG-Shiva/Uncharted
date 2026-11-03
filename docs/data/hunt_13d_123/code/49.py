import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _expand_c13(R, S):
    # R (9,13), S (6,13) -> X (123,13)
    n_orb = R.shape[0]
    X = np.empty((n_orb * 13 + S.shape[0], 13), dtype=float)
    idx = 0
    for k in range(n_orb):
        rk = R[k]
        for s in range(13):
            X[idx] = np.roll(rk, s)
            idx += 1
    X[idx:] = S
    return X

def _anneal_c13(R0, S0, steps, beta0, beta1, lr0, lr1, mu=0.8):
    R = R0.copy()
    S = S0.copy()
    vR = np.zeros_like(R)
    vS = np.zeros_like(S)
    n_orb = R.shape[0]
    n_spec = S.shape[0]
    N = n_orb * 13 + n_spec
    for st in range(steps):
        f = st / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        X = _expand_c13(R, S)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Z = float(W.sum() * 0.5) + 1e-30
        gradX = (W @ X) / Z
        # tangential
        dot = np.sum(gradX * X, axis=1, keepdims=True)
        gt = gradX - dot * X
        # map to R: mean of inverse rolls
        gR = np.empty_like(R)
        for k in range(n_orb):
            acc = np.zeros(13)
            base = k * 13
            for s in range(13):
                acc += np.roll(gt[base + s], -s)
            acc /= 13.0
            # project to tangent of R[k]
            acc = acc - float(acc @ R[k]) * R[k]
            gR[k] = acc
        gS = gt[n_orb * 13:]
        # re-project S (already tangential, but ensure)
        dotS = np.sum(gS * S, axis=1, keepdims=True)
        gS = gS - dotS * S
        vR = mu * vR + gR
        vS = mu * vS + gS
        R = R - lr * vR
        S = S - lr * vS
        R = _normalize(R)
        S = _normalize(S)
    X = _expand_c13(R, S)
    return R, S, X

def _expand_c12(B, C):
    # B (norb,13), C (nspec,13); each B orbit -> 12 points
    norb = B.shape[0]
    nspec = C.shape[0]
    N = norb * 12 + nspec
    X = np.empty((N, 13), dtype=float)
    angs = 2.0 * np.pi * np.arange(12) / 12.0
    cs = np.cos(angs); sn = np.sin(angs)
    idx = 0
    for k in range(norb):
        b = B[k]
        z = b[12]
        for t in range(12):
            c = cs[t]; s = sn[t]
            row = np.empty(13)
            for j in range(6):
                bx = b[2*j]; by = b[2*j+1]
                row[2*j] = c*bx - s*by
                row[2*j+1] = s*bx + c*by
            row[12] = z
            X[idx] = row
            idx += 1
    X[idx:] = C
    return X

def _anneal_c12(B0, C0, steps, beta0, beta1, lr0, lr1, mu=0.8):
    B = B0.copy(); C = C0.copy()
    vB = np.zeros_like(B); vC = np.zeros_like(C)
    norb = B.shape[0]
    angs = 2.0 * np.pi * np.arange(12) / 12.0
    cs = np.cos(angs); sn = np.sin(angs)
    for st in range(steps):
        f = st / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        X = _expand_c12(B, C)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Z = float(W.sum() * 0.5) + 1e-30
        gradX = (W @ X) / Z
        dot = np.sum(gradX * X, axis=1, keepdims=True)
        gt = gradX - dot * X
        gB = np.empty_like(B)
        for k in range(norb):
            base = k * 12
            acc = np.zeros(13)
            for t in range(12):
                c = cs[t]; s = sn[t]
                g = gt[base + t]
                # rotate grad by -t: gr @ Rot_t  == Rot_{-t}(gr)
                for j in range(6):
                    gx = g[2*j]; gy = g[2*j+1]
                    acc[2*j] += c*gx + s*gy
                    acc[2*j+1] += -s*gx + c*gy
                acc[12] += g[12]
            acc /= 12.0
            acc = acc - float(acc @ B[k]) * B[k]
            gB[k] = acc
        gC = gt[norb*12:]
        dotC = np.sum(gC * C, axis=1, keepdims=True)
        gC = gC - dotC * C
        vB = mu * vB + gB
        vC = mu * vC + gC
        B = B - lr * vB
        C = C - lr * vC
        B = _normalize(B)
        C = _normalize(C)
    return B, C, _expand_c12(B, C)

def _anneal_free(X0, steps, beta0, beta1, lr0, lr1, mu=0.85):
    v = X0.copy()
    vel = np.zeros_like(v)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 * (beta1 / beta0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = v @ v.T
        np.fill_diagonal(G, -10.0)
        gmax = float(G.max())
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        Z = float(W.sum() * 0.5) + 1e-30
        grad = (W @ v) / Z
        dot = np.sum(grad * v, axis=1, keepdims=True)
        gt = grad - dot * v
        # normalize step by mean row weight to avoid blowup
        vel = mu * vel + gt
        v = v - lr * vel
        v = _normalize(v)
    return v

def _lp_polish(x, seconds, p_cap=2200):
    t_end = time.time() + seconds
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = p_cap
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([(I[:, None]*d + np.arange(d)).reshape(p, d),
                                   (J[:, None]*d + np.arange(d)).reshape(p, d),
                                   np.full((p,1), n*d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() >= t_end:
            break
    return best_x

def solve():
    t0 = time.time()
    n, d = 123, 13
    rng = np.random.RandomState(0)
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _normalize(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _normalize(rng.randn(n, d))
        best_val = _max_cos(best_x)

    def remain():
        return 168.0 - (time.time() - t0)

    # --- symmetric C13 search ---
    sym_best = None
    sym_val = 1e9
    try:
        for ri in range(4):
            if remain() < 60:
                break
            R0 = _normalize(rng.randn(9, 13))
            S0 = _normalize(rng.randn(6, 13))
            # vary schedule slightly per restart
            b0 = 25.0 + 5*ri
            _, _, Xsym = _anneal_c13(R0, S0, steps=1800, beta0=b0, beta1=600.0, lr0=0.02, lr1=0.002, mu=0.8)
            v = _max_cos(Xsym)
            if v < sym_val:
                sym_val = v
                sym_best = Xsym.copy()
    except Exception:
        pass
    # --- symmetric C12 search ---
    try:
        for ri in range(2):
            if remain() < 55:
                break
            B0 = _normalize(rng.randn(10, 13))
            C0 = _normalize(rng.randn(3, 13))
            _, _, Xsym2 = _anneal_c12(B0, C0, steps=1500, beta0=25.0, beta1=500.0, lr0=0.02, lr1=0.002, mu=0.8)
            v = _max_cos(Xsym2)
            if v < sym_val:
                sym_val = v
                sym_best = Xsym2.copy()
    except Exception:
        pass

    # --- symmetry breaking + free relax ---
    cand = None
    cand_val = best_val
    if sym_best is not None:
        # try direct sym and broken versions
        for sig, steps in [(0.0, 2200), (0.004, 2200), (0.010, 1800)]:
            if remain() < 45:
                break
            try:
                if sig == 0.0:
                    X0 = sym_best.copy()
                else:
                    X0 = _normalize(sym_best + sig * rng.randn(n, d))
                Xf = _anneal_free(X0, steps=steps, beta0=40.0, beta1=600.0, lr0=0.010, lr1=0.0008, mu=0.85)
                v = _max_cos(Xf)
                if v < cand_val:
                    cand_val = v
                    cand = Xf.copy()
            except Exception:
                continue
    if cand is not None and cand_val < best_val:
        best_val = cand_val
        best_x = cand.copy()
    elif sym_best is not None and sym_val < best_val:
        best_val = sym_val
        best_x = sym_best.copy()

    # --- final unrestricted LP polish on current best ---
    r = remain()
    if r > 8:
        try:
            # polish the best we hold (either record or symmetric-derived)
            # if symmetric path failed, this just re-polishes record briefly
            sec = min(r - 2.0, 45.0)
            best_x = _lp_polish(best_x, sec)
            best_val = _max_cos(best_x)
        except Exception:
            pass
    best_x = _normalize(best_x)
    return best_x
