import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _proj(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return x / nrm

def _quench(y, steps, lr0, T0, T1, mu=0.82):
    n, d = y.shape
    vel = np.zeros((n, d))
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (1.0 - frac) + T1 * frac
        if T < 1e-12:
            T = 1e-12
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = (G - m) / T
        np.clip(E, -20.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        grad = W @ y
        dot = np.einsum('ij,ij->i', grad, y)
        grad -= dot[:, None] * y
        # normalize per-point for stable step
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        grad /= gn
        vel = mu * vel + (1.0 - mu) * grad + 0.0 * vel * 0.0
        # use momentum form: vel = mu*vel + grad
        # redo correctly:
        # (keep simple heavy-ball)
        # we already did EMA; convert to step
        lr = lr0 * (1.0 - 0.6 * frac)
        y = y - lr * (grad + mu * vel * 0.5)
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
    return y

def _quench2(y, steps, lr0, T0, T1, mu=0.85):
    # heavy-ball variant, slightly different dynamics for diversity
    n, d = y.shape
    vel = np.zeros((n, d))
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = T0 * (1.0 - frac) + T1 * frac
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = (G - m) / (T + 1e-12)
        np.clip(E, -20.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ y) / rs
        dot = np.einsum('ij,ij->i', grad, y)
        grad -= dot[:, None] * y
        vel = mu * vel + grad
        lr = lr0 * (1.0 - 0.7 * frac)
        y = y - lr * vel
        nrm = np.linalg.norm(y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        y /= nrm
    return y

def _lp_polish(x, seconds, t0limit=155.0, t_start=0.0):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if time.time() - t_start > t0limit:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2200:
            idx = np.argpartition(-c[sel], 2200)[:2200]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
            p = 2200
        else:
            csv = c[sel]
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
                          method="highs", options={"time_limit": 2.0})
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
            if radius < 2e-7 and v >= best:
                # no progress, stop early
                pass
    return x

def solve():
    T_START = time.time()
    HARD = 150.0  # hard internal deadline
    rng = np.random.default_rng(7)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = None
    try:
        r0 = np.load("record.npy")
    except Exception:
        r0 = None
    cands = []
    if b0 is not None:
        try:
            b0 = np.nan_to_num(np.asarray(b0, dtype=float))
            if b0.shape == (54, 21):
                cands.append(_proj(b0))
        except Exception:
            pass
    if r0 is not None:
        try:
            r0 = np.nan_to_num(np.asarray(r0, dtype=float))
            if r0.shape == (54, 21):
                cands.append(_proj(r0))
        except Exception:
            pass
    if len(cands) == 0:
        cands.append(_proj(rng.normal(size=(54, 21))))
    best = None
    bv = 1e9
    for c in cands:
        v = _max_cos(c)
        if v < bv:
            bv = v
            best = c.copy()
    n, d = 54, 21

    def tl():
        return HARD - (time.time() - T_START)

    # hopping parameters
    strengths = [0.02, 0.06, 0.15, 0.30, 0.03, 0.10, 0.22, 0.05]
    it = 0
    # reserve ~25s for final LP
    while tl() > 30.0:
        it += 1
        if tl() < 5:
            break
        s = strengths[(it - 1) % len(strengths)]
        y = best.copy()
        mode = it % 4
        if mode == 0 or mode == 2:
            # tangent shake of subset
            k = 6 + int(rng.integers(0, 6)) if mode == 0 else 10 + int(rng.integers(0, 8))
            # pick targets: mix of worst + random for diversity
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            order = np.argsort(-rowmax)
            targets = set(order[:3].tolist())
            while len(targets) < k:
                targets.add(int(rng.integers(0, n)))
            for i in list(targets):
                v = rng.normal(size=d)
                v -= (v @ y[i]) * y[i]
                nv = np.linalg.norm(v)
                if nv < 1e-12:
                    continue
                v /= nv
                ang = s * (0.5 + rng.random())
                y[i] = np.cos(ang) * y[i] + np.sin(ang) * v
            # small global tangent jitter
            nz = rng.normal(size=(n, d)) * (s * 0.15)
            nz -= (np.einsum('ij,ij->i', nz, y))[:, None] * y
            y = y + nz
            y = _proj(y)
        elif mode == 1:
            # worst-point void reinsertion (best-of-many)
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            rowmax = G.max(axis=1)
            if it % 8 == 0:
                order = np.argsort(-rowmax)
                iw = int(order[int(rng.integers(0, 5))])
            else:
                iw = int(np.argmax(rowmax))
            mask = np.ones(n, dtype=bool)
            mask[iw] = False
            Yr = y[mask]
            K = 2000
            C = rng.normal(size=(K, d))
            C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
            D = C @ Yr.T
            scores = D.max(axis=1)
            ib = int(np.argmin(scores))
            y[iw] = C[ib]
            # jitter others slightly
            nz = rng.normal(size=(n, d)) * (s * 0.1)
            nz -= (np.einsum('ij,ij->i', nz, y))[:, None] * y
            y = y + nz
            y = _proj(y)
        else:
            # global jitter + random rotation of one point
            nz = rng.normal(size=(n, d)) * s
            nz -= (np.einsum('ij,ij->i', nz, y))[:, None] * y
            y = _proj(y + nz)
            # also fully randomize one of the worst 5 occasionally
            if it % 2 == 0:
                G = y @ y.T
                np.fill_diagonal(G, -2.0)
                rowmax = G.max(axis=1)
                order = np.argsort(-rowmax)
                iw = int(order[int(rng.integers(0, 5))])
                v = rng.normal(size=d)
                v /= (np.linalg.norm(v) + 1e-12)
                y[iw] = v

        # annealed quench: choose temp schedule by strength
        if s > 0.12:
            steps1, lr1, Ta, Tb = 900, 0.35, 0.03, 0.002
        elif s > 0.05:
            steps1, lr1, Ta, Tb = 1100, 0.30, 0.02, 0.0012
        else:
            steps1, lr1, Ta, Tb = 800, 0.25, 0.012, 0.0008
        # alternate quench dynamics
        if it % 2 == 0:
            yq = _quench(y, steps=steps1, lr0=lr1, T0=Ta, T1=Tb)
        else:
            yq = _quench2(y, steps=steps1, lr0=lr1, T0=Ta, T1=Tb)
        vq = _max_cos(yq)
        if vq < bv:
            best, bv = yq, vq
            continue
        # if promising, refine with low-T quench
        if vq < bv + 0.004 and tl() > 32:
            yq2 = _quench2(yq, steps=700, lr0=0.12, T0=0.006, T1=0.0006, mu=0.9)
            vq2 = _max_cos(yq2)
            if vq2 < bv:
                best, bv = yq2, vq2
                continue
            # also try alternate
            if vq2 < bv + 0.001 and vq2 < vq:
                # keep as base for next hops? no, keep best unchanged
                pass
        # opportunistic: if very close, try tiny shake + short quench chain
        # (skip to save time)

    # final squeeze: short low-T quench then LP polish on best
    if tl() > 8:
        try:
            yf = _quench2(best.copy(), steps=600, lr0=0.10, T0=0.005, T1=0.0005, mu=0.9)
            vf = _max_cos(yf)
            if vf < bv:
                best, bv = yf, vf
        except Exception:
            pass
    if tl() > 8:
        try:
            secs = min(22.0, tl() - 4.0)
            if secs > 4:
                yp = _lp_polish(best.copy(), seconds=secs, t0limit=HARD - 2.0, t_start=T_START)
                vp = _max_cos(yp)
                if vp <= bv:
                    best, bv = yp, vp
        except Exception:
            pass
    best = _proj(best)
    return best
