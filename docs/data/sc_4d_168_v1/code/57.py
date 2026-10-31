import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _nm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _mc(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    SEARCH = t0 + 145.0
    rng = np.random.default_rng(20261031)
    # load incumbent
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.load(f).astype(float)
            if c.shape == (168, 4):
                c = _nm(c)
                if np.all(np.isfinite(c)):
                    cands.append(c)
        except Exception:
            pass
    if not cands:
        xb = _nm(rng.standard_normal(size=(168, 4)))
        bv = _mc(xb)
    else:
        vs = [_mc(c) for c in cands]
        bi = int(np.argmin(vs))
        xb = cands[bi].copy()
        bv = float(vs[bi])

    # ---------- helpers (closure over rng) ----------
    def _adam(X, steps, beta0, beta1, lr0, lr1, mu=0.88):
        n, d = X.shape
        vel = np.zeros_like(X)
        lb0, lb1 = np.log(beta0), np.log(beta1)
        for s in range(steps):
            if s % 150 == 0 and time.time() > SEARCH:
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
            X = _nm(X - lr * vel)
        return X

    def _lp(x, deadline):
        x = _nm(x.copy())
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
            y = _nm(x + r.x[:-1].reshape(n, d))
            v = _mc(y)
            if v < best:
                x, best = y, v
                rad = min(rad * 1.5, 0.05)
            else:
                rad *= 0.5
        return x

    def _ref_single(x0, Xr, steps=70, b0=45.0, b1=260.0, lr0=0.09, lr1=0.006):
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
                s = rng.standard_normal(size=(4,))
                ns = np.linalg.norm(s)
            s = s / ns
        return s

    def _gen_holes(Xref, n_sample=30000, n_raw=300, n_cand=12):
        # random sphere samples
        C = _nm(rng.standard_normal(size=(n_sample, 4)))
        D = C @ Xref.T
        m = D.max(axis=1)
        if n_raw >= n_sample:
            idx = np.argsort(m)[:n_cand]
            cand0 = C[idx]
            out = []
            for c0 in cand0:
                out.append(_ref_single(c0, Xref))
            out = np.array(out)
            q = (out @ Xref.T).max(axis=1)
            return out[np.argsort(q)]
        part = np.argpartition(m, n_raw)[:n_raw]
        Cp = C[part]
        mp = m[part]
        order = np.argsort(mp)
        Cp = Cp[order]
        # greedy diverse pick
        picked = []
        for c0 in Cp:
            if len(picked) >= n_cand:
                break
            if not picked:
                picked.append(c0)
            else:
                P = np.array(picked)
                if float(np.max(P @ c0)) < 0.96:
                    picked.append(c0)
        # fill if not enough
        if len(picked) < n_cand:
            for c0 in Cp:
                if len(picked) >= n_cand:
                    break
                dup = False
                for p0 in picked:
                    if np.dot(p0, c0) > 0.999:
                        dup = True
                        break
                if not dup and not any(np.allclose(c0, p0) for p0 in picked):
                    # avoid exact dup; check dot
                    if all(float(np.dot(p0, c0)) < 0.999 for p0 in picked) or len(picked) == 0:
                        pass
                    picked.append(c0)
                    if len(picked) >= n_cand:
                        break
        P0 = np.array(picked[:n_cand])
        # local hole refinement
        out = []
        for c0 in P0:
            out.append(_ref_single(c0, Xref))
        out = np.array(out)
        q = (out @ Xref.T).max(axis=1)
        return out[np.argsort(q)]

    BEAM = 6
    HCAND = 12
    KEEP_PER = 4
    SHORT_STEPS = 200

    cyc = 0
    while time.time() < SEARCH:
        cyc += 1
        if time.time() > SEARCH - 3.0:
            break
        # choose Nsmall 160-166
        Nsmall = int(rng.integers(160, 167))
        # occasionally force 162-164 for depth
        if cyc % 3 == 0:
            Nsmall = int(rng.integers(162, 165))
        kdel = 168 - Nsmall
        base = xb.copy()
        try:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            deg = np.sum(G > cmax - 0.004, axis=1)
            # stress: sum of excess over threshold
            order_hi = np.argsort(-deg, kind="stable")
        except Exception:
            order_hi = np.arange(168)
        vict = order_hi[:kdel].copy()
        # diversify: swap 1-2 victims with next candidates
        if kdel > 0 and rng.random() < 0.7:
            pool = order_hi[kdel:kdel + 12]
            nswap = min(2, kdel, len(pool))
            if nswap > 0 and len(pool) > 0:
                si = rng.choice(kdel, size=nswap, replace=False)
                pi = rng.choice(len(pool), size=nswap, replace=False)
                vict[si] = pool[pi]
        keep = np.ones(168, dtype=bool)
        keep[vict] = False
        Xs = base[keep].copy()
        # fully polish smaller code (unjammed -> lower max)
        try:
            Xs = _adam(Xs, 900, 20.0, 500.0, 0.018, 0.0012)
        except Exception:
            continue
        if time.time() < SEARCH - 1.0 and cyc % 2 == 0:
            try:
                Xs = _lp(Xs, min(time.time() + 0.7, SEARCH))
            except Exception:
                pass
        vsmall = _mc(Xs)
        beam = [Xs]
        bvals = [vsmall]
        # beam reinsertion one at a time
        ok = True
        for s in range(Nsmall, 168):
            if time.time() > SEARCH - 1.5:
                ok = False
                break
            poolX = []
            poolV = []
            for bi_, Xp in enumerate(beam):
                bv_ = bvals[bi_]
                try:
                    holes = _gen_holes(Xp, n_sample=30000, n_raw=300, n_cand=HCAND)
                except Exception:
                    continue
                # raw insertion quality
                dots = holes @ Xp.T
                md = dots.max(axis=1)
                raw = np.maximum(bv_, md)
                o = np.argsort(raw)[:KEEP_PER]
                for hi in o:
                    h = holes[hi]
                    Xn = np.vstack([Xp, h])
                    try:
                        Xp2 = _adam(Xn, SHORT_STEPS, 25.0, 450.0, 0.017, 0.0012)
                    except Exception:
                        continue
                    v2 = _mc(Xp2)
                    poolX.append(Xp2)
                    poolV.append(v2)
                    if time.time() > SEARCH - 1.0:
                        break
                if time.time() > SEARCH - 1.0:
                    break
            if not poolX:
                ok = False
                break
            o2 = np.argsort(np.array(poolV))[:BEAM]
            beam = [poolX[i] for i in o2]
            bvals = [poolV[i] for i in o2]
        if not ok:
            continue
        # final polish best few completions
        o3 = np.argsort(np.array(bvals))
        nfin = min(2, len(o3))
        for fi in range(nfin):
            if time.time() > SEARCH - 0.5:
                break
            Xf = beam[o3[fi]].copy()
            vf = bvals[o3[fi]]
            # longer anneal if promising
            if vf < bv + 0.004 or fi == 0:
                try:
                    Xf = _adam(Xf, 700, 28.0, 450.0, 0.016, 0.0012)
                    vf = _mc(Xf)
                except Exception:
                    pass
                if time.time() < SEARCH - 0.3:
                    try:
                        bud = 1.0 if vf < bv + 0.001 else 0.5
                        Xp2 = _lp(Xf, min(time.time() + bud, SEARCH + 2.0, HARD - 7.0))
                        vf2 = _mc(Xp2)
                        if vf2 <= vf:
                            Xf, vf = Xp2, vf2
                    except Exception:
                        pass
                if vf < bv:
                    bv = float(vf)
                    xb = Xf.copy()
        # opportunistic: if beam best beats global without long polish, keep
        try:
            if min(bvals) < bv:
                bv = float(min(bvals))
                xb = beam[int(np.argmin(np.array(bvals)))].copy()
        except Exception:
            pass
    if time.time() < HARD - 2.0:
        try:
            rem = min(HARD - 1.0, time.time() + max(6.0, HARD - time.time() - 1.0))
            Xl = _lp(xb, rem)
            vl = _mc(Xl)
            if vl < bv:
                xb = Xl
        except Exception:
            pass
    return _nm(xb)
