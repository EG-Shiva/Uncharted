import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    x = x / n
    return np.nan_to_num(x)

def _units(m):
    import math
    return [k for k in range(1, m) if math.gcd(k, m) == 1]

def _free_quench(y, steps, t0, t1, lr0=0.32, mu=0.9):
    n, d = y.shape
    vel = np.zeros((n, d))
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = t0 * (1 - frac) + t1 * frac
        if T < 1e-12:
            T = 1e-12
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = (G - m) / T
        np.clip(E, -15, 0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = W @ y
        dot = np.einsum('ij,ij->i', grad, y)
        grad -= dot[:, None] * y
        grad /= rs
        vel = mu * vel + grad
        lr = lr0 * (1.0 - 0.6 * frac)
        y = y - lr * vel
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
    return y

def _lp_polish(x, t_end, max_pairs=1500):
    x = _proj(x)
    best = _max_cos(x)
    n, d = x.shape
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > max_pairs:
            idx = np.argpartition(-c[sel], max_pairs)[:max_pairs]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
        else:
            csv = c[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if time.time() >= t_end:
            break
    return x

def solve():
    T0 = time.time()
    HARD = 150.0
    def tl():
        return HARD - (time.time() - T0)
    rng = np.random.default_rng(0)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = np.load("record.npy")
    b0 = _proj(np.nan_to_num(np.asarray(b0, dtype=float)))
    if b0.shape != (54, 21):
        b0 = _proj(rng.normal(size=(54, 21)))
    try:
        r0 = _proj(np.asarray(np.load("record.npy"), dtype=float))
        if r0.shape == (54, 21):
            if _max_cos(r0) < _max_cos(b0):
                b0 = r0
    except Exception:
        pass
    best = b0.copy()
    bv = _max_cos(best)
    target = 0.0666634002447
    if bv > target:
        target = bv
        bv = float(min(bv, target))

    n, d = 54, 21
    nplanes = 10

    def build_tables(mults, m):
        th = np.array(mults, dtype=float) * (2 * np.pi / m)
        ks = np.arange(m)[None, :]
        thv = th[:, None]
        C = np.cos(thv * ks)
        S = np.sin(thv * ks)
        return C, S

    def expand(Sseeds, C, S, m):
        ns = Sseeds.shape[0]
        Xb = np.empty((ns, m, 21))
        for p in range(nplanes):
            a0 = Sseeds[:, 2 * p]
            a1 = Sseeds[:, 2 * p + 1]
            c = C[p]
            s = S[p]
            Xb[:, :, 2 * p] = a0[:, None] * c[None, :] - a1[:, None] * s[None, :]
            Xb[:, :, 2 * p + 1] = a0[:, None] * s[None, :] + a1[:, None] * c[None, :]
        Xb[:, :, 20] = Sseeds[:, 20:21]
        return Xb.reshape(ns * m, 21), Xb

    def sym_quench(Sinit, C, S, m, steps, t0, t1, lr0=0.03):
        ns = Sinit.shape[0]
        Sn = Sinit / (np.linalg.norm(Sinit, axis=1, keepdims=True) + 1e-12)
        mt = np.zeros_like(Sn)
        vt = np.zeros_like(Sn)
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = 0
        for it in range(steps):
            frac = it / max(1, steps - 1)
            T = t0 * (1 - frac) + t1 * frac
            if T < 5e-5:
                T = 5e-5
            X, Xb = expand(Sn, C, S, m)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mx = float(G.max())
            E = (G - mx) / T
            np.clip(E, -15, 0, out=E)
            W = np.exp(E)
            np.fill_diagonal(W, 0.0)
            sumU = float(W.sum() / 2.0) + 1e-30
            gradX = (W @ X) / sumU
            gradXb = gradX.reshape(ns, m, 21)
            gradS = np.empty_like(Sn)
            for p in range(nplanes):
                b0m = gradXb[:, :, 2 * p]
                b1m = gradXb[:, :, 2 * p + 1]
                c = C[p]
                s = S[p]
                gradS[:, 2 * p] = (b0m * c[None, :] + b1m * s[None, :]).sum(axis=1)
                gradS[:, 2 * p + 1] = (-b0m * s[None, :] + b1m * c[None, :]).sum(axis=1)
            gradS[:, 20] = gradXb[:, :, 20].sum(axis=1)
            dot = np.einsum('ij,ij->i', gradS, Sn)
            gradS -= dot[:, None] * Sn
            # Adam
            t += 1
            mt = b1 * mt + (1 - b1) * gradS
            vt = b2 * vt + (1 - b2) * (gradS * gradS)
            mh = mt / (1 - b1 ** t)
            vh = vt / (1 - b2 ** t)
            lr = lr0 * (1.0 - 0.5 * frac)
            step = lr * mh / (np.sqrt(vh) + eps)
            # project step to tangent
            sdot = np.einsum('ij,ij->i', step, Sn)
            step -= sdot[:, None] * Sn
            # clip step
            Sn = Sn - step
            Sn = Sn / (np.linalg.norm(Sn, axis=1, keepdims=True) + 1e-12)
            # keep axis from collapsing to +-1 (avoid short orbits): clamp
            # ensure plane energy not zero: if a seed becomes axial, nudge
            if (it % 200) == 199:
                pr = np.sqrt((Sn[:, :20] ** 2).sum(axis=1))
                tiny = pr < 0.2
                if np.any(tiny):
                    Sn[tiny, :20] += 0.05 * rng.normal(size=(int(tiny.sum()), 20))
                    Sn[tiny] /= (np.linalg.norm(Sn[tiny], axis=1, keepdims=True) + 1e-12)
        X, _ = expand(Sn, C, S, m)
        X = _proj(X)
        return Sn, X

    units18 = _units(18)
    units27 = _units(27)

    trial = 0
    # sequence of short trials
    while tl() > 12 and trial < 60:
        trial += 1
        cyc = (trial - 1) % 4
        if cyc == 0:
            m, anti = 18, False
            ns_eff, ns = 3, 3
        elif cyc == 1:
            m, anti = 18, True
            ns_eff, ns = 2, 3
        elif cyc == 2:
            m, anti = 27, False
            ns_eff, ns = 2, 2
        else:
            m, anti = 27, True
            ns_eff, ns = 1, 2
        if m == 18:
            mults = [units18[int(rng.integers(0, len(units18)))] for _ in range(nplanes)]
        else:
            mults = [units27[int(rng.integers(0, len(units27)))] for _ in range(nplanes)]
        C, S = build_tables(mults, m)
        Sinit_eff = rng.normal(size=(ns_eff, 21))
        Sinit_eff /= (np.linalg.norm(Sinit_eff, axis=1, keepdims=True) + 1e-12)
        # anneal schedule similar K 80->1000 => T 0.0125 -> 0.001
        steps_sym = 900 if tl() > 60 else 600
        # vary temperature slightly per trial
        t0 = 0.012 if (trial % 2 == 0) else 0.008
        t1 = 0.001 if (trial % 3 != 0) else 0.0005
        lr0 = 0.03 if (trial % 2 == 0) else 0.05
        try:
            Seff, _ = sym_quench(Sinit_eff, C, S, m, steps_sym, t0, t1, lr0)
        except Exception:
            continue
        if anti:
            if m == 18:
                Sfull = np.vstack([Seff[0:1], Seff[1:2], -Seff[0:1]])
            else:
                Sfull = np.vstack([Seff[0:1], -Seff[0:1]])
        else:
            Sfull = Seff
        Xsym, _ = expand(Sfull / (np.linalg.norm(Sfull, axis=1, keepdims=True) + 1e-12), C, S, m)
        Xsym = _proj(Xsym)
        vsym = _max_cos(Xsym)
        # break symmetry
        nz = rng.normal(size=(54, 21)) * 1e-3
        nz -= (nz * Xsym).sum(axis=1, keepdims=True) * Xsym
        Y = _proj(Xsym + nz)
        try:
            Y = _free_quench(Y, steps=350, t0=0.008, t1=0.0008, lr0=0.32, mu=0.9)
        except Exception:
            continue
        vf = _max_cos(Y)
        # keep only if promising: LP polish if within margin of target/best
        thresh = min(bv, target) + 0.02
        # also always polish the best free so far occasionally
        do_lp = (vf < thresh) and (tl() > 10)
        # limit LP to save time: only if vf < 0.12
        if vf > 0.12:
            do_lp = False
        if do_lp:
            t_end = time.time() + min(2.5, tl() - 4.0)
            if t_end > time.time():
                try:
                    Y2 = _lp_polish(Y, t_end)
                    v2 = _max_cos(Y2)
                except Exception:
                    v2 = vf
                    Y2 = Y
                if v2 < vf:
                    Y, vf = Y2, v2
        if vf < bv:
            best, bv = Y.copy(), vf
            # early exit if clearly new record with margin
            if bv < 0.0666634002447 - 1e-6:
                # spend remaining a bit to polish best further
                if tl() > 10:
                    t_end = time.time() + min(6.0, tl() - 2.0)
                    try:
                        Yp = _lp_polish(best, t_end)
                        vp = _max_cos(Yp)
                        if vp < bv:
                            best, bv = Yp, vp
                    except Exception:
                        pass
                break
        if tl() <= 8:
            break
    # final short squeeze on best if time
    if tl() > 6:
        try:
            t_end = time.time() + min(5.0, tl() - 1.5)
            Yp = _lp_polish(best, t_end)
            vp = _max_cos(Yp)
            if vp < bv:
                best, bv = Yp, vp
        except Exception:
            pass
    best = _proj(best)
    return best
