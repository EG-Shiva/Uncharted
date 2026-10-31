import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    x = np.asarray(x, dtype=float)
    x[~np.isfinite(x)] = 0.0
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    x = x / n
    bad = ~np.all(np.isfinite(x), axis=1)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contacts(x, thr=1e-9):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    cnt = (g >= m - thr).sum(axis=1)
    pw = g.max(axis=1)
    return m, pw, cnt, g

def _anneal(x, steps, tau0, tau1, lr0, lr1, mov=None, seed_off=0):
    n, d = x.shape
    if mov is None:
        m = np.zeros_like(x); v = np.zeros_like(x)
        idx = None
    else:
        mov = np.asarray(mov, dtype=int)
        m = np.zeros((len(mov), d)); v = np.zeros((len(mov), d))
        idx = mov
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac if tau1 != tau0 else tau0
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        G -= gm
        G /= tau
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(G.sum())
        if not np.isfinite(S) or S <= 0:
            break
        G /= S
        grad = G @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        g = gt if idx is None else gt[idx]
        m *= b1; m += (1 - b1) * g
        v *= b2; v += (1 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1); bc2 = 1.0 - b2 ** (k + 1)
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        if idx is None:
            x = x - lr * step
            x = _norm(x)
        else:
            x[idx] = x[idx] - lr * step
            x[idx] = x[idx] / np.linalg.norm(x[idx], axis=1, keepdims=True)
    return x

def _rand_orth(rng):
    A = rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    s = np.sign(d + 1e-30)
    Q = Q * s[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _find_void(x_fixed, rng, n_samples=8000):
    cand = rng.standard_normal((n_samples, 4))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    sims = cand @ x_fixed.T
    best = sims.max(axis=1)
    i = int(best.argmin())
    return cand[i]

def _lp_polish(x, seconds, deadline):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    rng = np.random.default_rng(12345)
    t0 = time.time()
    deadline = t0 + 170.0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(np.asarray(xb, dtype=float))
    if xr is not None:
        cands.append(np.asarray(xr, dtype=float))
    if len(cands) == 0:
        cands.append(_norm(rng.standard_normal((272, 4))))
    # sanitize shapes
    clean = []
    for c in cands:
        c = np.asarray(c, dtype=float)
        if c.shape != (272, 4):
            flat = c.ravel()
            need = 272 * 4
            if flat.size < need:
                flat = np.concatenate([flat, rng.standard_normal((need - flat.size,))])
            c = flat[:need].reshape(272, 4)
        clean.append(_norm(c))
    vals = [_max_cos(c) for c in clean]
    x_best = clean[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    Q0 = _rand_orth(rng)

    trial = 0
    # Stage 1: diverse basin search
    while time.time() < t0 + 110 and time.time() < deadline - 55:
        trial += 1
        mode = trial % 6
        if mode == 0:
            x = _norm(rng.standard_normal((272, 4)))
            steps, tau0, tau1, lr0, lr1 = 700, 0.02, 0.001, 0.015, 0.0008
        elif mode == 1:
            sig = 0.04 if (trial % 2 == 0) else 0.10
            x = x_best + sig * rng.standard_normal((272, 4))
            x = _norm(x)
            steps, tau0, tau1, lr0, lr1 = 500, 0.012, 0.001, 0.010, 0.0008
        elif mode == 2:
            # coordinated cap rotation
            x = x_best.copy()
            u = rng.standard_normal((4,)); u /= np.linalg.norm(u)
            proj = x @ u
            frac = 0.20 + 0.15 * rng.random()
            thr = np.quantile(proj, 1.0 - frac)
            sel = np.where(proj >= thr)[0]
            Q = _rand_orth(rng)
            # interpolate: small rotation toward Q
            ang = 0.25 + 0.45 * rng.random()
            # build rotation via exp of skew: use Q slerp approx: R = I*cos+...
            # simpler: rotate cap points by fixed orthogonal with blending
            R = np.eye(4) * np.cos(ang) + Q * np.sin(ang)
            # orthonormalize R via QR to keep rigid
            Rq, _ = np.linalg.qr(R)
            if np.linalg.det(Rq) < 0:
                Rq[:, 0] = -Rq[:, 0]
            x[sel] = x[sel] @ Rq.T
            x = _norm(x)
            # small jitter
            x = _norm(x + 0.01 * rng.standard_normal(x.shape))
            steps, tau0, tau1, lr0, lr1 = 500, 0.012, 0.0009, 0.010, 0.0007
        elif mode == 3:
            # hub+rattler patch kill + greedy void refill
            x = x_best.copy()
            m, pw, cnt, g = _contacts(x)
            order_hi = np.argsort(-pw)
            order_lo = np.argsort(pw)
            nk = 6 + (trial % 5)  # 6..10
            victims = set()
            for k in range(nk // 2):
                victims.add(int(order_hi[k]))
            for k in range(nk - len(victims)):
                victims.add(int(order_lo[k]))
            # add nearest neighbours of worst hub for coordinated move
            worst = int(order_hi[0])
            drow = g[worst].copy()
            nn = np.argsort(-drow)[:3]
            for q in nn:
                if len(victims) < nk + 3 and rng.random() < 0.5:
                    victims.add(int(q))
            victims = list(victims)
            for v in victims:
                mask = np.ones(272, dtype=bool); mask[v] = False
                x[v] = _find_void(x[mask], rng, n_samples=6000)
            steps, tau0, tau1, lr0, lr1 = 550, 0.015, 0.001, 0.012, 0.0008
        elif mode == 4:
            # global random rotation + jitter (symmetry escape)
            x = x_best @ _rand_orth(rng).T
            x = _norm(x + 0.02 * rng.standard_normal(x.shape))
            steps, tau0, tau1, lr0, lr1 = 500, 0.010, 0.0008, 0.009, 0.0007
        else:
            # antipodal-ish shuffle: flip random hemisphere
            x = x_best.copy()
            u = rng.standard_normal((4,)); u /= np.linalg.norm(u)
            proj = x @ u
            sel = np.where(proj > 0)[0]
            # random orthogonal on that hemisphere + jitter
            R = _rand_orth(rng)
            ang = 0.1 + 0.2 * rng.random()
            R = np.eye(4) + ang * (R - R.T)
            Rq, _ = np.linalg.qr(R)
            x[sel] = x[sel] @ Rq.T
            x = _norm(x + 0.015 * rng.standard_normal(x.shape))
            steps, tau0, tau1, lr0, lr1 = 500, 0.012, 0.001, 0.010, 0.0008
        x = _anneal(x, steps, tau0, tau1, lr0, lr1)
        v = _max_cos(x)
        if v < best_val:
            best_val = v
            x_best = x.copy()
            # quick opportunistic LP snap if promising and time allows
            if time.time() < t0 + 90 and (best_val < 0.89816):
                xs = _lp_polish(x_best, 4.0, deadline - 1.0)
                vs = _max_cos(xs)
                if vs <= best_val:
                    x_best, best_val = xs, vs

    # Stage 2: balance hubs/rattlers with patch anneal
    for it in range(6):
        if time.time() > deadline - 40:
            break
        m, pw, cnt, g = _contacts(x_best)
        # rattlers: cnt==0 ; hubs: top contacts
        ratt = np.where(cnt == 0)[0]
        hubs = np.argsort(-pw)[:4]
        x = x_best.copy()
        moved = False
        for r in ratt:
            mask = np.ones(272, dtype=bool); mask[r] = False
            x[r] = _find_void(x[mask], rng, n_samples=8000)
            moved = True
        # disperse hub neighbourhood slightly
        for h in hubs:
            row = (x @ x[h])
            row[h] = -2
            nn = np.argsort(-row)[:6]
            for q in nn:
                x[q] = _norm(x[q][None, :] + 0.005 * rng.standard_normal((1, 4)))[0]
            moved = True
        if not moved:
            # kill worst 3 into voids
            for k in range(3):
                G = x @ x.T; np.fill_diagonal(G, -2.0)
                idx = int(G.argmax()); ii, jj = divmod(idx, 272)
                victim = ii if rng.random() < 0.5 else jj
                mask = np.ones(272, dtype=bool); mask[victim] = False
                x[victim] = _find_void(x[mask], rng, n_samples=8000)
        # patch anneal: worst 40 points jointly at low temp
        m2, pw2, _, _ = _contacts(x)
        mov = np.argsort(-pw2)[:48]
        x = _anneal(x, 350, 0.008, 0.0006, 0.006, 0.0005, mov=mov)
        # full gentle anneal
        x = _anneal(x, 250, 0.005, 0.0005, 0.004, 0.0004)
        v = _max_cos(x)
        if v < best_val:
            best_val = v
            x_best = x.copy()

    remain = deadline - time.time() - 2.0
    if remain > 5:
        xp = _lp_polish(x_best, min(remain, 35.0), deadline - 1.0)
        vp = _max_cos(xp)
        if vp <= best_val:
            x_best, best_val = xp, vp
    if not np.all(np.isfinite(x_best)):
        x_best = clean[0]
    x_best = _norm(x_best)
    return x_best
