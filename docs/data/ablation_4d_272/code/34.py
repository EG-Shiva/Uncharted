import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, r0=2e-3, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[order]] = True
            sel = keep
        c = c_all[sel]
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth(x, steps, a0, a1, s0, s1, mu=0.85):
    n, d = x.shape
    vel = np.zeros_like(x)
    la0 = np.log(max(a0, 1e-12)); la1 = np.log(max(a1, 1e-12))
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = float(np.exp(la0 + (la1 - la0) * frac))
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            x = x - (smax / nrm) * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _subset_smooth(x, movable, steps, a0, a1, s0, s1, mu=0.7):
    movable = np.asarray(movable, dtype=int)
    M = len(movable)
    if M == 0:
        return x
    vel = np.zeros((M, 4))
    la0 = np.log(a0); la1 = np.log(a1)
    for it in range(steps):
        frac = it / max(steps - 1, 1)
        alpha = float(np.exp(la0 + (la1 - la0) * frac))
        smax = s0 * (s1 / s0) ** frac if s0 != s1 else s0
        Xm = x[movable]
        D = Xm @ x.T
        D[np.arange(M), movable] = -2.0
        m = float(D.max())
        W = np.exp(alpha * (D - m))
        grad = W @ x
        dot = np.sum(grad * Xm, axis=1, keepdims=True)
        gt = grad - dot * Xm
        vel = mu * vel + gt
        nrm = float(np.linalg.norm(vel, axis=1).max())
        if nrm > 1e-14:
            Xn = Xm - (smax / nrm) * vel
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            x[movable] = Xn
    return x

def _hole(x, rng, n_cand=8000, refine=80):
    n = x.shape[0]
    C = rng.standard_normal((n_cand, 4))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    best_i = 0
    best_v = 1e9
    chunk = 2000
    dm = np.empty(n_cand)
    for s in range(0, n_cand, chunk):
        e = min(s + chunk, n_cand)
        D = C[s:e] @ x.T
        dm[s:e] = D.max(axis=1)
    i = int(np.argmin(dm))
    c = C[i].copy()
    vel = np.zeros(4)
    alpha = 120.0
    for it in range(refine):
        dots = x @ c
        m = float(dots.max())
        W = np.exp(alpha * (dots - m))
        g = W @ x
        gt = g - float(g @ c) * c
        nn = float(np.linalg.norm(gt))
        if nn < 1e-14:
            break
        gt /= nn
        vel = 0.6 * vel + gt
        nn2 = float(np.linalg.norm(vel))
        if nn2 < 1e-14:
            break
        step = 0.02 * (1.0 - it / refine) + 0.002
        c = c - (step / nn2) * vel
        nc = float(np.linalg.norm(c))
        if nc < 1e-12:
            break
        c /= nc
    return c

def _two_phase(x):
    x = _smooth(x, steps=350, a0=12.0, a1=90.0, s0=0.06, s1=0.008, mu=0.85)
    x = _smooth(x, steps=350, a0=110.0, a1=650.0, s0=0.03, s1=0.004, mu=0.85)
    return x

def _givens4(rng, angle):
    A = rng.standard_normal((4, 4))
    Q, _ = np.linalg.qr(A)
    # two random orthonormal vectors spanning plane
    e1 = Q[:, 0]
    e2 = Q[:, 1]
    R = np.eye(4) + (np.cos(angle) - 1.0) * (np.outer(e1, e1) + np.outer(e2, e2)) + np.sin(angle) * (np.outer(e2, e1) - np.outer(e1, e2))
    return R

def _clifford_init(rng, kind=0):
    pts = []
    if kind == 0:
        T = 17; P = 16
        ts = (np.arange(T) + 0.5) * (np.pi / 2.0) / T
        ts += 0.02 * rng.standard_normal(T)
        ts = np.clip(ts, 0.03, np.pi/2 - 0.03)
        for k in range(T):
            t = ts[k]
            oa = rng.uniform(0, 2*np.pi)
            ob = rng.uniform(0, 2*np.pi)
            ct = np.cos(t); st = np.sin(t)
            for j in range(P):
                a = 2*np.pi*(j % 4)/4.0 + oa + 0.05*rng.standard_normal()
                b = 2*np.pi*(j // 4)/4.0 + ob + 0.05*rng.standard_normal()
                pts.append([ct*np.cos(a), ct*np.sin(a), st*np.cos(b), st*np.sin(b)])
    else:
        T = 16; P = 17
        ts = (np.arange(T) + 0.5) * (np.pi / 2.0) / T
        ts += 0.02 * rng.standard_normal(T)
        ts = np.clip(ts, 0.03, np.pi/2 - 0.03)
        for k in range(T):
            t = ts[k]
            oa = rng.uniform(0, 2*np.pi)
            ct = np.cos(t); st = np.sin(t)
            for j in range(P):
                a = 2*np.pi*j/17.0 + oa + 0.03*rng.standard_normal()
                b = 2*np.pi*j*7.0/17.0 + oa*0.7 + 0.03*rng.standard_normal()
                pts.append([ct*np.cos(a), ct*np.sin(a), st*np.cos(b), st*np.sin(b)])
    X = np.array(pts, dtype=float)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def solve():
    t0 = time.time()
    BUDGET = 168.0
    t_end = t0 + BUDGET
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (272, 4):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        r = rng.standard_normal((272, 4))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        cands.append(r)
    best_x = min(cands, key=lambda z: _max_cos(z)).copy()
    best_v = _max_cos(best_x)
    cur_x = best_x.copy()
    cur_v = best_v

    trial = 0
    while True:
        if time.time() > t_end - 14.0:
            break
        trial += 1
        op = int(rng.integers(0, 6))
        # occasionally fresh clifford
        if trial % 9 == 0:
            op = 5
        try:
            if op == 0:
                # cap deletion / regrow : large geographic void
                K = int(rng.integers(22, 55))
                u = rng.standard_normal(4); u /= np.linalg.norm(u)
                dots = cur_x @ u
                del_idx = np.argsort(-dots)[:K]
                keep = np.array([i for i in range(272) if i not in set(del_idx.tolist())])
                xs = cur_x[keep].copy()
                xs = xs + 0.005 * rng.standard_normal(xs.shape)
                xs /= np.linalg.norm(xs, axis=1, keepdims=True)
                xs = _smooth(xs, steps=200, a0=15.0, a1=120.0, s0=0.05, s1=0.006)
                # regrow
                x = xs
                while x.shape[0] < 272:
                    if time.time() > t_end - 14.0:
                        break
                    c = _hole(x, rng, n_cand=6000, refine=60)
                    x = np.vstack([x, c.reshape(1, 4)])
                    nn_idx = x.shape[0]-1
                    dots2 = x[:-1] @ x[-1]
                    nb = min(16, x.shape[0]-1)
                    neigh = np.argsort(-dots2)[:nb]
                    mov = np.concatenate([neigh, np.array([nn_idx])])
                    x = _subset_smooth(x, mov, steps=60, a0=80.0, a1=500.0, s0=0.04, s1=0.005)
                if x.shape[0] != 272:
                    continue
                x = _smooth(x, steps=250, a0=20.0, a1=600.0, s0=0.03, s1=0.004)
                x = _lp_polish(x, seconds=min(1.2, max(0.4, t_end - 14.0 - time.time())), r0=2e-3)
            elif op == 1:
                # hemisphere twist: rotate one side coherently
                u = rng.standard_normal(4); u /= np.linalg.norm(u)
                s = cur_x @ u
                mask = s > 0
                if mask.sum() < 20 or mask.sum() > 252:
                    continue
                ang = float(rng.uniform(0.12, 0.38))
                R = _givens4(rng, ang)
                x = cur_x.copy()
                x[mask] = x[mask] @ R.T
                x = x + float(rng.uniform(0.004, 0.016)) * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                x = _two_phase(x)
                x = _lp_polish(x, seconds=min(1.2, max(0.4, t_end - 14.0 - time.time())), r0=2e-3)
            elif op == 2:
                # cluster breakup: densest neighborhood to holes
                G = cur_x @ cur_x.T
                np.fill_diagonal(G, -2.0)
                m = float(G.max())
                cnt = np.sum(G > (m - 2e-3), axis=1)
                seed = int(np.argmax(cnt + 0.1*rng.standard_normal(272)))
                drow = G[seed]
                K = int(rng.integers(12, 24))
                victims = np.argsort(-drow)[:K]
                # include seed itself
                victims = np.unique(np.concatenate([np.array([seed]), victims]))[:K]
                x = cur_x.copy()
                # sequential far holes
                placed = []
                for vi in victims:
                    # temporary set without vi
                    tmp = np.delete(x, vi, axis=0)
                    # avoid placed holes being close: penalize
                    c = _hole(tmp, rng, n_cand=6000, refine=60)
                    # enforce separation from already placed
                    tries = 0
                    while tries < 3 and any(float(c @ p) > 0.90 for p in placed):
                        c2 = _hole(tmp, rng, n_cand=4000, refine=40)
                        c = c2
                        tries += 1
                    placed.append(c)
                    x[vi] = c
                x = x + 0.004 * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                x = _two_phase(x)
                x = _lp_polish(x, seconds=min(1.2, max(0.4, t_end - 14.0 - time.time())), r0=2e-3)
            elif op == 3:
                # partial restart: keep random subset
                keep_n = int(rng.integers(195, 235))
                perm = rng.permutation(272)
                keep = np.sort(perm[:keep_n])
                xs = cur_x[keep].copy()
                need = 272 - keep_n
                fresh = rng.standard_normal((need, 4))
                fresh /= np.linalg.norm(fresh, axis=1, keepdims=True)
                x = np.vstack([xs, fresh])
                # shuffle order to avoid bias
                x = x[rng.permutation(272)]
                x = _two_phase(x)
                x = _lp_polish(x, seconds=min(1.2, max(0.4, t_end - 14.0 - time.time())), r0=2e-3)
            elif op == 4:
                # global jitter + balanced anneal (coordinated)
                sig = float(rng.uniform(0.02, 0.06))
                x = cur_x + sig * rng.standard_normal(cur_x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                # long balanced phase to share load
                x = _smooth(x, steps=400, a0=8.0, a1=60.0, s0=0.07, s1=0.008, mu=0.9)
                x = _smooth(x, steps=300, a0=100.0, a1=700.0, s0=0.03, s1=0.003, mu=0.85)
                x = _lp_polish(x, seconds=min(1.4, max(0.4, t_end - 14.0 - time.time())), r0=2e-3)
            else:
                # clifford fresh start
                kind = int(rng.integers(0, 2))
                x = _clifford_init(rng, kind=kind)
                x = x + 0.01 * rng.standard_normal(x.shape)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                x = _two_phase(x)
                x = _lp_polish(x, seconds=min(1.4, max(0.4, t_end - 14.0 - time.time())), r0=3e-3)
            v = _max_cos(x)
            if v < best_v - 1e-12:
                best_v = v
                best_x = x.copy()
                cur_x = x.copy(); cur_v = v
            else:
                # walker acceptance: accept if close, else sometimes random restart to best
                if v < cur_v or rng.random() < 0.12:
                    cur_x = x.copy(); cur_v = v
                # periodic pull back to best if wandering too far
                if cur_v > best_v + 0.004 and rng.random() < 0.5:
                    cur_x = best_x.copy() + 0.01 * rng.standard_normal(best_x.shape)
                    cur_x /= np.linalg.norm(cur_x, axis=1, keepdims=True)
                    cur_v = _max_cos(cur_x)
                if trial % 25 == 0 and rng.random() < 0.4:
                    cur_x = best_x.copy()
                    cur_v = best_v
        except Exception:
            continue

    # final polish of best with remaining time
    try:
        rem = t_end - time.time()
        if rem > 1.0:
            best_x = _lp_polish(best_x, seconds=min(rem - 0.7, 10.0), r0=1.5e-3, cap=4000)
            best_v = _max_cos(best_x)
            # one more sharp smooth + polish if time
            rem2 = t_end - time.time()
            if rem2 > 2.0:
                best_x = _smooth(best_x, steps=200, a0=200.0, a1=800.0, s0=0.012, s1=0.002)
                best_x = _lp_polish(best_x, seconds=min(rem2 - 0.5, 6.0), r0=1e-3, cap=4000)
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
