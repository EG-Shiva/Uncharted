import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, deadline=None, r0=2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0
    while time.time() - start < seconds and radius > 1e-14:
        if deadline is not None and time.time() > deadline - 0.6:
            break
        G = x @ x.T
        c_all = G[iuI, iuJ]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            # keep largest 2500
            thv = np.partition(c_all, -2500)[-2500]
            sel = c_all >= thv
            # subsample if still too many due to ties
            cnt2 = int(np.count_nonzero(sel))
            if cnt2 > 3200:
                idxw = np.where(sel)[0]
                # take evenly spaced to keep determinism + spread
                step = max(1, len(idxw) // 2800)
                keep = idxw[::step][:2800]
                ns = np.zeros_like(sel)
                ns[keep] = True
                sel = ns
        I = iuI[sel]; J = iuJ[sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cd = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def _riesz_descent(x, s0, s1, steps, lr, mu, rng, jitter=0.0):
    n, d = x.shape
    vel = np.zeros_like(x)
    for k in range(steps):
        frac = k / max(steps - 1, 1)
        s = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        expo = s * 0.5 + 1.0
        G = x @ x.T
        np.fill_diagonal(G, 2.0)  # D2= -2 on diag -> large, weight ~0
        D2 = 2.0 - 2.0 * G
        # D2 diag = -2 -> clip
        np.fill_diagonal(D2, 10.0)
        D2 = np.maximum(D2, 1e-12)
        d2min = float(D2.min())
        # normalized weights in (0,1]
        # (d2min/D2)^expo
        with np.errstate(under='ignore', over='ignore', divide='ignore'):
            W = np.exp(expo * (np.log(d2min) - np.log(D2)))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-300
        Gx = G.copy()
        np.fill_diagonal(Gx, 0.0)
        # grad tangent: W@x - ((W*Gx).sum)*x
        Wx = W @ x
        coeff = (W * Gx).sum(axis=1, keepdims=True)
        gt = Wx - coeff * x
        gt /= rs
        # clip per-row norm to 1 for stability
        gn = np.linalg.norm(gt, axis=1, keepdims=True) + 1e-300
        np.clip(gn, 1.0, None, out=gn)
        gt /= gn
        vel = mu * vel + (1.0 - mu) * gt
        cl = 1.0 - 0.5 * frac
        x = x - lr * cl * vel
        if jitter > 0:
            x = x + jitter * (rng.standard_normal(size=x.shape) * 0.01)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _void_points(x_keep, m, rng, S=25000):
    y = rng.standard_normal(size=(S, 4))
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    scores = (y @ x_keep.T).max(axis=1)
    order = np.argsort(scores)
    picked = []
    for idx in order:
        if len(picked) >= m:
            break
        cand = y[idx]
        ok = True
        for q in picked:
            if float(cand @ q) > 0.80:
                ok = False
                break
        if ok:
            picked.append(cand.copy())
    k = 0
    while len(picked) < m:
        picked.append(y[order[k]].copy())
        k += 1
    P = np.array(picked)
    # local repulsion refinement vs kept set (few steps)
    for _ in range(120):
        D = P @ x_keep.T  # m x nk
        mx = D.max(axis=1, keepdims=True)
        W = np.exp(40.0 * (D - mx))
        Gdir = W @ x_keep  # m x 4
        # project
        dot = (Gdir * P).sum(axis=1, keepdims=True)
        Gt = Gdir - dot * P
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-300
        P = P - 0.05 * Gt / nrm
        P /= np.linalg.norm(P, axis=1, keepdims=True)
    return P

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    DEADLINE = t0 + 170.0
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.standard_normal(size=(163, 4))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (163, 4):
        tmp = rng.standard_normal(size=(163, 4))
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best = _max_cos(xb)
    # also try record
    try:
        xr = np.asarray(np.load("record.npy"), dtype=float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        vr = _max_cos(xr)
        if vr < best:
            best = vr
            xb = xr.copy()
    except Exception:
        pass
    x_best = xb.copy()
    x_cur = xb.copy()
    cur = best
    n, d = 163, 4
    it = 0
    # quick initial polish to calibrate (short)
    # main basin loop; reserve ~16s final
    while time.time() < DEADLINE - 17.0:
        it += 1
        if time.time() > DEADLINE - 19.0:
            break
        # choose base
        if rng.random() < 0.75 or it < 4:
            base = x_best
            base_val = best
        else:
            base = x_cur
            base_val = cur
        x_try = base.copy()
        r = rng.random()
        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        rowmax = G0.max(axis=1)
        # contact degree with adaptive thr
        thr_deg = base_val - 0.012
        deg = (G0 > thr_deg).sum(axis=1)
        if r < 0.30:
            sigma = 10.0 ** rng.uniform(-3.4, -1.3)
            x_try = base + sigma * rng.standard_normal(size=(n, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.58:
            # patch shake around overloaded or random center
            if rng.random() < 0.65:
                center = int(np.argmax(deg + rng.random(n) * 0.7))
            else:
                center = int(rng.integers(n))
            k = int(rng.integers(8, 30))
            dots = base @ base[center]
            idx = np.argsort(-dots)[:k]
            sigma = float(rng.uniform(0.02, 0.10))
            x_try[idx] = base[idx] + sigma * rng.standard_normal(size=(k, d))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        elif r < 0.82:
            # rattler / overloaded relocation to voids
            # pick m worst: low slack (high rowmax) + high degree
            score = rowmax + 0.002 * deg + rng.random(n) * 0.0005
            m = int(rng.integers(1, 4))
            # bias: sometimes pick pure rattlers (low rowmax) to move? actually rattlers have low rowmax, don't move worst; we want move worst.
            # alternate: move worst m
            reloc = np.argsort(-score)[:m]
            # occasionally move a random low-degree (rattler) instead
            if rng.random() < 0.25:
                # lowest degree
                reloc = np.argsort(deg + rng.random(n) * 0.3)[:m]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[reloc] = False
            voids = _void_points(base[keep_mask], m, rng, S=22000)
            voids = voids + 0.004 * rng.standard_normal(size=voids.shape)
            voids /= np.linalg.norm(voids, axis=1, keepdims=True)
            x_try[reloc] = voids
            if rng.random() < 0.6:
                sig = 0.008
                nm = n - m
                noise = np.zeros((n, d))
                noise[keep_mask] = sig * rng.standard_normal(size=(nm, d))
                x_try = x_try + noise
                x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        else:
            # contact-pair split: take worst pair, push apart
            iu = np.triu_indices(n, 1)
            gv = G0[iu]
            w = int(np.argmax(gv))
            i, j = int(iu[0][w]), int(iu[1][w])
            # midpoint
            mid = base[i] + base[j]
            mn = np.linalg.norm(mid)
            if mn < 1e-9:
                mid = rng.standard_normal(size=(4,))
                mn = np.linalg.norm(mid)
            mid /= mn
            # orthogonal direction separating them
            diff = base[i] - base[j]
            diff = diff - float(diff @ mid) * mid
            dn = np.linalg.norm(diff)
            if dn < 1e-9:
                diff = rng.standard_normal(size=(4,))
                diff = diff - float(diff @ mid) * mid
                dn = np.linalg.norm(diff)
            diff /= dn
            amp = float(rng.uniform(0.02, 0.07))
            x_try[i] = base[i] + amp * diff + 0.01 * rng.standard_normal(size=(4,))
            x_try[j] = base[j] - amp * diff + 0.01 * rng.standard_normal(size=(4,))
            # shake neighbours of i,j slightly
            if rng.random() < 0.7:
                dots = (base @ base[i] + base @ base[j]) * 0.5
                idx2 = np.argsort(-dots)[:12]
                x_try[idx2] = x_try[idx2] + 0.015 * rng.standard_normal(size=(len(idx2), 4))
            x_try /= np.linalg.norm(x_try, axis=1, keepdims=True)
        # Riesz annealing hypers
        s0 = float(rng.choice([6.0, 10.0, 14.0, 22.0]))
        s1 = float(s0 * rng.uniform(2.0, 5.0))
        s1 = min(s1, 90.0)
        steps = int(rng.integers(500, 1400))
        lr = float(rng.uniform(0.03, 0.12))
        mu = float(rng.uniform(0.5, 0.9))
        # time guard: estimate cost; skip if too close
        x_try = _riesz_descent(x_try, s0, s1, steps, lr, mu, rng, jitter=0.0)
        if time.time() > DEADLINE - 18.5:
            v = _max_cos(x_try)
        else:
            sec = float(rng.uniform(0.7, 1.6))
            sec = min(sec, max(0.5, DEADLINE - 17.0 - time.time()))
            if sec > 0.35:
                x_try = _lp_polish(x_try, sec, deadline=DEADLINE - 16.0)
            v = _max_cos(x_try)
        if v < best - 1e-12:
            x_best = x_try.copy()
            best = v
            x_cur = x_try.copy()
            cur = v
        elif v < cur + 0.002 and rng.random() < 0.5:
            x_cur = x_try.copy()
            cur = v
        if it % 15 == 0 and cur > best + 0.0025:
            x_cur = x_best.copy()
            cur = best
    remain = DEADLINE - time.time() - 0.8
    if remain > 2.0:
        x_best = _lp_polish(x_best, min(remain, 15.0), deadline=DEADLINE)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
