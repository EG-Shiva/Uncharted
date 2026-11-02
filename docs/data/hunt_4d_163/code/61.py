import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x

def _anneal(x0, steps, T0, T1, lr0, mu, rng, active=None, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    if active is None:
        act = np.ones(n, dtype=bool)
    else:
        act = np.asarray(active, dtype=bool)
    na = int(act.sum())
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    bv = float(G.max())
    bx = x.copy()
    if T0 <= 0: T0 = 1e-12
    if T1 <= 0: T1 = 1e-12
    lratio = np.log(T1 / T0)
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 1e-12: T = 1e-12
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
        if na != n:
            grad[~act] = 0.0
            v[~act] = 0.0
        v = mu * v + (1.0 - mu) * grad
        x[act] = x[act] - lr * v[act]
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0 and na > 0:
                nz = rng.randn(na, d) * amp
                xa = x[act]
                nz -= (nz * xa).sum(axis=1, keepdims=True) * xa
                x[act] = xa + nz
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        x /= np.maximum(xn, 1e-12)
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _lp_polish(x, deadline, r0=0.004, cap=1300):
    x = _norm(x.copy())
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
            if int(np.count_nonzero(sel)) == 0:
                sel[int(np.argmax(c))] = True
            if int(sel.sum()) > cap:
                idx = np.argsort(-c)[:cap]
                s2 = np.zeros_like(sel, dtype=bool)
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

def _degs(x, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    deg = (G > m - tol).sum(axis=1)
    return m, deg, G

def _refine_hole(c0, Y, iters=120, T=0.004, lr=0.08):
    c = c0 / max(float(np.linalg.norm(c0)), 1e-12)
    for _ in range(iters):
        dots = Y @ c
        m = float(dots.max())
        e = np.exp((dots - m) / T)
        s = float(e.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w = e / s
        g = w @ Y
        g = g - float(g @ c) * c
        c = c - lr * g
        nrm = float(np.linalg.norm(c))
        if nrm < 1e-12:
            break
        c = c / nrm
    return c

def _sample_hole(Y, rng, n_sample=12000):
    C = rng.randn(n_sample, 4)
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    D = Y @ C.T
    mx = D.max(axis=0)
    j = int(np.argmin(mx))
    c = _refine_hole(C[j], Y)
    return c

def _small_rot(rng, eps):
    M = np.eye(4) + rng.randn(4, 4) * eps
    Q, R = np.linalg.qr(M)
    Q = Q * np.sign(np.diag(R) + 1e-12)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.RandomState(1234567)
    n = 163
    best_x = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (n, 4) or not np.all(np.isfinite(a)):
                continue
            a = _norm(a)
            v = _max_cos(a)
            if v < best_val:
                best_val = v
                best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = _norm(rng.randn(n, 4))
        best_val = _max_cos(best_x)
    # initial squeeze of incumbent
    if time.time() < DEADLINE - 20:
        try:
            best_x, best_val = _lp_polish(best_x, min(DEADLINE - 15, time.time() + 8.0), r0=0.003)
        except Exception:
            pass
    elite = [(best_val, best_x.copy())]
    trial = 0
    # fast hopping loop, leave ~14s for final polish
    while time.time() < DEADLINE - 14.0:
        trial += 1
        # pick source: mostly best, sometimes 2nd elite
        if len(elite) > 1 and (trial % 5 == 0):
            src = elite[1][1]
        else:
            src = best_x
        x = src.copy()
        mode = trial % 4
        try:
            if mode == 0:
                # global tangent kick + anneal
                sig = 0.012 if (trial % 8) else 0.025
                nz = rng.randn(n, 4) * sig
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = _norm(x + nz)
                x, v = _anneal(x, steps=700, T0=0.006, T1=0.0003, lr0=0.09, mu=0.85, rng=rng, noise0=0.003, deadline=min(time.time()+2.5, DEADLINE-14))
                x, v = _anneal(x, steps=500, T0=0.0012, T1=0.00012, lr0=0.05, mu=0.88, rng=rng, noise0=0.0, deadline=min(time.time()+2.0, DEADLINE-14))
            elif mode == 1:
                # rigid cap rotation
                u = rng.randn(4); u /= np.linalg.norm(u)
                dots = x @ u
                capsize = int(rng.choice([8, 12, 18, 26]))
                idx = np.argsort(-dots)[:capsize]
                Q = _small_rot(rng, 0.06 + 0.04 * rng.rand())
                # rotate cap points around, keep on sphere
                x[idx] = _norm(x[idx] @ Q.T)
                # small noise to cap
                nz = rng.randn(capsize, 4) * 0.008
                xc = x[idx]
                nz -= (nz * xc).sum(axis=1, keepdims=True) * xc
                x[idx] = _norm(xc + nz)
                x, v = _anneal(x, steps=800, T0=0.004, T1=0.0002, lr0=0.08, mu=0.86, rng=rng, noise0=0.002, deadline=min(time.time()+2.5, DEADLINE-14))
                x, v = _anneal(x, steps=400, T0=0.0008, T1=0.0001, lr0=0.05, mu=0.88, rng=rng, noise0=0.0, deadline=min(time.time()+2.0, DEADLINE-14))
            elif mode == 2:
                # stressed hub deletion + hole reinsertion (1-3 points)
                try:
                    _, deg, _ = _degs(x, tol=1e-9)
                except Exception:
                    deg = np.zeros(n, dtype=int)
                k = int(rng.choice([1, 2, 2, 3, 4]))
                jit = rng.randn(n) * 0.6
                order = np.argsort(-deg.astype(float) - jit * 0.0 + jit * 0.0)
                # degree + jitter ranking
                rank = deg.astype(float) + jit * 0.5
                del_idx = np.argsort(-rank)[:k]
                keep = np.ones(n, dtype=bool); keep[del_idx] = False
                Y = x[keep].copy()
                # relax remainder briefly
                Y, _ = _anneal(Y, steps=400, T0=0.004, T1=0.0004, lr0=0.08, mu=0.85, rng=rng, noise0=0.002, deadline=min(time.time()+1.5, DEADLINE-14))
                cur = Y
                for _t in range(k):
                    hc = _sample_hole(cur, rng, n_sample=12000)
                    cur = _norm(np.vstack([cur, hc.reshape(1, 4)]))
                x = cur
                # local anneal of inserted + neighbours then global
                x, v = _anneal(x, steps=700, T0=0.003, T1=0.0002, lr0=0.07, mu=0.86, rng=rng, noise0=0.001, deadline=min(time.time()+2.5, DEADLINE-14))
                x, v = _anneal(x, steps=400, T0=0.0008, T1=0.0001, lr0=0.05, mu=0.88, rng=rng, noise0=0.0, deadline=min(time.time()+1.5, DEADLINE-14))
            else:
                # worst-pair avalanche: kick neighbourhood of max pair
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                ij = int(np.argmax(G))
                a = ij // n; b = ij % n
                dvec = (x @ x[a]) + (x @ x[b])
                nbr = np.where(dvec > 1.2)[0]
                if len(nbr) < 4:
                    nbr = np.argsort(-dvec)[:8]
                # bigger kick to neighbourhood
                nz = np.zeros_like(x)
                nz[nbr] = rng.randn(len(nbr), 4) * 0.03
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = _norm(x + nz)
                act = np.ones(n, dtype=bool)  # anneal all but with noise concentrated early
                x, v = _anneal(x, steps=800, T0=0.005, T1=0.0002, lr0=0.085, mu=0.85, rng=rng, active=act, noise0=0.002, deadline=min(time.time()+2.5, DEADLINE-14))
                x, v = _anneal(x, steps=400, T0=0.0009, T1=0.0001, lr0=0.05, mu=0.88, rng=rng, noise0=0.0, deadline=min(time.time()+1.5, DEADLINE-14))
        except Exception:
            continue
        try:
            vcur = _max_cos(x)
        except Exception:
            continue
        # track elite (keep 3)
        if len(elite) < 3 or vcur < elite[-1][0] + 0.002:
            elite.append((vcur, x.copy()))
            elite = sorted(elite, key=lambda z: z[0])[:3]
        # LP only if promising (saves time) or strictly better
        if vcur < best_val - 2e-9:
            # squeeze immediately
            try:
                x2, v2 = _lp_polish(x, min(DEADLINE - 8.0, time.time() + 6.0), r0=0.003)
            except Exception:
                x2, v2 = x, vcur
            if v2 < best_val:
                best_x, best_val = x2.copy(), float(v2)
                elite = sorted([(best_val, best_x.copy())] + elite, key=lambda z: z[0])[:3]
        elif vcur < best_val + 0.0004 and (trial % 7 == 0) and time.time() < DEADLINE - 25:
            try:
                x2, v2 = _lp_polish(x, min(DEADLINE - 14.0, time.time() + 4.0), r0=0.0025)
            except Exception:
                continue
            if v2 < best_val:
                best_x, best_val = x2.copy(), float(v2)
                elite = sorted([(best_val, best_x.copy())] + elite, key=lambda z: z[0])[:3]
        # early exit if excellent improvement? continue to use time for further polish
    # final long polish on best
    try:
        if time.time() < DEADLINE - 1.0:
            best_x, best_val = _lp_polish(best_x, DEADLINE - 0.5, r0=0.002)
    except Exception:
        pass
    best_x = _norm(best_x)
    return best_x
