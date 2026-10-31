import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _nrm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _mc(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam(X, steps, beta0, beta1, lr0, lr1, mu=0.88, deadline=None):
    n, d = X.shape
    vel = np.zeros_like(X)
    lb0, lb1 = np.log(beta0), np.log(beta1)
    for s in range(steps):
        if deadline is not None and (s % 200 == 0) and time.time() > deadline:
            break
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = lr0 * (1.0 - f) + lr1 * f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = (E @ X) / rs
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        sc = float(np.mean(np.linalg.norm(gt, axis=1))) + 1e-12
        gt = gt / sc
        vel = mu * vel + (1.0 - mu) * gt
        X = _nrm(X - lr * vel)
    return X

def _lp(x, deadline):
    x = _nrm(x.copy())
    n, d = x.shape
    best = _mc(x)
    rad = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and rad > 1e-14:
        C = (x @ x.T)[iu]
        m = float(np.max(C))
        sel = C > m - max(20.0 * rad, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        cs = C[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        Aeq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            r = linprog(obj, A_ub=Aub, b_ub=-cs, A_eq=Aeq, b_eq=np.zeros(n),
                        bounds=[(-rad, rad)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if r.status != 0:
            rad *= 0.5
            continue
        y = _nrm(x + r.x[:-1].reshape(n, d))
        v = _mc(y)
        if v < best:
            x, best = y, v
            rad = min(rad * 1.5, 0.05)
        else:
            rad *= 0.5
    return x

def _tangent_bases(X):
    n = X.shape[0]
    B = np.empty((n, 4, 3))
    I4 = np.eye(4)
    for i in range(n):
        xi = X[i]
        k = int(np.argmin(np.abs(xi)))
        E = np.delete(I4, k, axis=1)  # 4x3
        P = E - xi[:, None] * (xi @ E)[None, :]
        Q, _ = np.linalg.qr(P)
        # ensure orthogonal to xi (fix sign if needed)
        B[i] = Q
    return B

def _ref_single(x0, Xr, steps=70, b0=45.0, b1=260.0, lr0=0.09, lr1=0.006, rng=None):
    s = x0 / (np.linalg.norm(x0) + 1e-30)
    lb0, lb1 = np.log(b0), np.log(b1)
    for it in range(steps):
        f = it / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = lr0 * (1.0 - f) + lr1 * f
        dots = Xr @ s
        cm = float(np.max(dots))
        w = np.exp(np.clip(beta * (dots - cm), -25.0, 0.0))
        w = w / (np.sum(w) + 1e-30)
        g = w @ Xr
        gt = g - np.dot(g, s) * s
        gn = np.linalg.norm(gt) + 1e-12
        gt = gt / gn
        s = s - lr * gt
        ns = np.linalg.norm(s)
        if ns < 1e-12:
            s = rng.standard_normal(4) if rng is not None else np.random.randn(4)
            ns = np.linalg.norm(s)
        s = s / ns
    return s

def _best_void(Xref, rng, ns=10000):
    C = _nrm(rng.standard_normal((ns, 4)))
    D = C @ Xref.T
    m = D.max(axis=1)
    c0 = C[int(np.argmin(m))]
    return _ref_single(c0, Xref, rng=rng)

def _hash_deg(X, thr=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    cmax = float(np.max(G))
    iu = np.triu_indices(X.shape[0], 1)
    C = G[iu]
    # degrees with slack<thr
    Adj = G > cmax - thr
    deg = tuple(sorted(np.sum(Adj, axis=1).astype(int).tolist()))
    top = np.sort(C)[-30:]
    qt = tuple((np.round(top * 1e6)).astype(int).tolist())
    return (deg, qt), cmax

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    SEARCH = t0 + 144.0
    rng = np.random.default_rng(77168)
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.load(f).astype(float)
            if c.shape == (168, 4):
                c = _nrm(c)
                if np.all(np.isfinite(c)):
                    cands.append(c)
        except Exception:
            pass
    if not cands:
        xb = _nrm(rng.standard_normal((168, 4)))
        bv = _mc(xb)
    else:
        vs = [_mc(c) for c in cands]
        xb = cands[int(np.argmin(vs))].copy()
        bv = float(min(vs))
    n, d = 168, 4
    xcur = xb.copy(); cv = bv
    tabu = set()
    hh, _ = _hash_deg(xb)
    tabu.add(hh)
    Temp = 0.0022
    trial = 0
    iu_all = np.triu_indices(n, 1)
    while time.time() < SEARCH:
        trial += 1
        if time.time() > SEARCH - 1.0:
            break
        # base selection
        if trial % 4 == 0 and cv < bv + 0.005:
            base = xcur.copy()
        else:
            base = xb.copy()
        # --- analyse active contacts ---
        try:
            G0 = base @ base.T
            np.fill_diagonal(G0, -2.0)
            cmax0 = float(np.max(G0))
            C0 = G0[iu_all]
            sel0 = C0 > cmax0 - 1e-7
            I0 = iu_all[0][sel0]; J0 = iu_all[1][sel0]
            p0 = len(I0)
            deg0 = np.sum(G0 > cmax0 - 1e-7, axis=1)
        except Exception:
            base = _nrm(base + 0.05 * rng.standard_normal((n, d)))
            X = _adam(base, 700, 30., 420., 0.016, 0.0013, deadline=SEARCH)
            v = _mc(X)
            if v < bv:
                xb, bv = X.copy(), float(v)
            continue
        # tangent bases + rigidity
        try:
            B = _tangent_bases(base)  # n,4,3
            N3 = n * 3
            R = np.zeros((max(p0, 1), N3))
            if p0 > 0:
                # Bi^T xj
                for r in range(p0):
                    i = int(I0[r]); j = int(J0[r])
                    R[r, i * 3:(i + 1) * 3] = B[i].T @ base[j]
                    R[r, j * 3:(j + 1) * 3] = B[j].T @ base[i]
                # SVD
                # feasible: 500x504
                U, S, Vt = np.linalg.svd(R, full_matrices=False)
                K = int(min(18, Vt.shape[0], max(10, Vt.shape[0] // 2)))
                # last K rows = smallest singular values
                modes = Vt[-K:, :]  # K x N3
                svals = S[-K:]
                # stress: last left vector
                w = np.abs(U[:, -1]) if U.shape[1] > 0 else np.ones(p0) / max(p0, 1)
            else:
                modes = None; w = None; svals = None
        except Exception:
            modes = None; w = None
        # per-point stress
        if p0 > 0 and w is not None:
            pstress = np.zeros(n)
            np.add.at(pstress, I0, w)
            np.add.at(pstress, J0, w)
        else:
            pstress = deg0.astype(float)
        order_hi = np.argsort(-(deg0 * 2.0 + pstress / (np.max(pstress) + 1e-12) * 5.0), kind="stable")
        # decide kick type
        do_void = (trial % 3 == 0)  # overload surgery trials
        Xk = base.copy()
        # --- overload delete/repel on void trials ---
        if do_void and p0 > 0:
            kdel = 3 + (trial % 3)  # 3-5
            vict = order_hi[:kdel].copy()
            # include contact neighbours of worst point for repelling
            worst = int(order_hi[0])
            nbrs = np.where(G0[worst] > cmax0 - 1e-7)[0]
            # repelling jitter to neighbours (excluding victims)
            rep = [a for a in nbrs.tolist() if a not in vict.tolist()][:6]
            keep_mask = np.ones(n, dtype=bool); keep_mask[vict] = False
            cur_set = Xk[keep_mask]
            newp = []
            for s_idx in range(kdel):
                Xr = cur_set if s_idx == 0 else np.vstack([cur_set, np.array(newp)])
                newp.append(_best_void(Xr, rng, ns=9000))
            Xk[vict] = np.array(newp)
            if len(rep) > 0:
                Xk[rep] = _nrm(Xk[rep] + 0.12 * rng.standard_normal((len(rep), 4)))
        # --- floppy-mode kick ---
        amp_target = float(rng.uniform(0.03, 0.15))
        # random combo of smallest modes
        if modes is not None and p0 > 10:
            K = modes.shape[0]
            coeff = rng.standard_normal(K)
            # weight floppier modes more: divide by (sval+eps)
            try:
                inv = 1.0 / (svals + 1e-6)
                inv = inv / np.max(inv)
                coeff = coeff * (0.5 + 0.5 * inv)
            except Exception:
                pass
            coeff = coeff / (np.linalg.norm(coeff) + 1e-12)
            vcomb = coeff @ modes  # N3
            Tm = vcomb.reshape(n, 3)
            D = np.empty((n, 4))
            for i in range(n):
                D[i] = B[i] @ Tm[i]
            rms = float(np.sqrt(np.mean(np.sum(D ** 2, axis=1)))) + 1e-12
            D = D / rms
            # adaptive amplitude to break >=30 contacts and max<0.90
            amp = amp_target
            best_try = None
            for adapt in range(6):
                Xt = _nrm(base + amp * D + (0.008 * rng.standard_normal((n, 4)) if do_void else 0.0))
                # for void trials D applied on top of void reinsertion: use Xk as base for D?
                # recompute with Xk base for void case: shift Xk along D
                if do_void:
                    Xt = _nrm(Xk + amp * D)
                Gt = Xt @ Xt.T
                np.fill_diagonal(Gt, -2.0)
                mt = float(np.max(Gt))
                # count old contacts broken: old pairs slack>1e-3
                # old pair cos values under new config
                old_cos_new = Gt[I0, J0] if p0 > 0 else np.array([mt])
                nbroken = int(np.sum((cmax0 - old_cos_new) > 1e-3)) if p0 > 0 else 100
                if nbroken >= 30 and mt < 0.90:
                    best_try = Xt
                    break
                if nbroken < 30 and mt < 0.88:
                    amp = min(amp * 1.45, 0.28)
                elif mt >= 0.90:
                    amp = amp * 0.6
                    if amp < 0.015:
                        best_try = Xt
                        break
                else:
                    best_try = Xt
                    break
            Xk = best_try if best_try is not None else _nrm(base + amp_target * D)
            # extra neighbour jitter of overloaded nodes
            if not do_void and trial % 2 == 0:
                ov = order_hi[:4]
                Xk[ov] = _nrm(Xk[ov] + 0.05 * rng.standard_normal((len(ov), 4)))
        else:
            # fallback isotropic
            Xk = _nrm(base + amp_target * rng.standard_normal((n, d)))
        # --- re-anneal with LSE ---
        if do_void:
            st, kw = 1050, dict(beta0=24., beta1=460., lr0=0.017, lr1=0.0012)
        elif trial % 5 == 0:
            st, kw = 1100, dict(beta0=22., beta1=480., lr0=0.019, lr1=0.0011)
        else:
            st, kw = 850, dict(beta0=30., beta1=420., lr0=0.016, lr1=0.0013)
        try:
            Xo = _adam(Xk, st, deadline=SEARCH, **kw)
        except Exception:
            continue
        v = _mc(Xo)
        hh2, _ = _hash_deg(Xo)
        seen = hh2 in tabu
        if seen:
            # tabu: extra random nudge + short anneal, don't LP
            Xo = _adam(_nrm(Xo + 0.02 * rng.standard_normal((n, d))), 350, 35., 350., 0.012, 0.002, deadline=SEARCH)
            v = _mc(Xo)
            hh2, _ = _hash_deg(Xo)
        tabu.add(hh2)
        if len(tabu) > 6000:
            # prune randomly (keep last 3000 by recreating)
            tabu = set(list(tabu)[-3000:])
        if v < cv or rng.random() < np.exp(-(v - cv) / Temp):
            xcur, cv = Xo.copy(), float(v)
        if v < bv:
            bv = float(v); xb = Xo.copy()
            # opportunistic LP with bigger budget
            if time.time() < SEARCH - 0.4:
                try:
                    bud = 2.2
                    Xp = _lp(Xo, min(time.time() + bud, SEARCH + 3.0, HARD - 7.0))
                    vp = _mc(Xp)
                    hh3, _ = _hash_deg(Xp)
                    tabu.add(hh3)
                    if vp < cv:
                        xcur, cv = Xp.copy(), float(vp)
                    if vp < bv:
                        bv = float(vp); xb = Xp.copy()
                except Exception:
                    pass
        else:
            # near-best LP occasionally
            if (not seen) and v < bv + 0.002 and (trial % 4 == 0) and time.time() < SEARCH - 0.4:
                try:
                    Xp = _lp(Xo, min(time.time() + 0.9, SEARCH + 3.0, HARD - 7.0))
                    vp = _mc(Xp)
                    hh3, _ = _hash_deg(Xp)
                    tabu.add(hh3)
                    if vp < cv:
                        xcur, cv = Xp.copy(), float(vp)
                    if vp < bv:
                        bv = float(vp); xb = Xp.copy()
                except Exception:
                    pass
            elif v < bv + 0.006 and (trial % 11 == 0) and time.time() < SEARCH - 0.4:
                try:
                    Xp = _lp(Xo, min(time.time() + 0.6, SEARCH + 3.0, HARD - 7.0))
                    vp = _mc(Xp)
                    if vp < bv:
                        bv = float(vp); xb = Xp.copy()
                    if vp < cv:
                        xcur, cv = Xp.copy(), float(vp)
                except Exception:
                    pass
    if time.time() < HARD - 2.0:
        try:
            rem = min(HARD - 1.0, time.time() + max(6.0, HARD - time.time() - 1.0))
            xb = _lp(xb, rem)
        except Exception:
            pass
    return _nrm(xb)
