import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _refine_one(x0, Others, steps=60, beta0=3000.0, beta1=20000.0, lr0=0.03, lr1=0.005):
    x = x0 / max(float(np.linalg.norm(x0)), 1e-12)
    lb0, lb1 = np.log(beta0), np.log(beta1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(lr0 + (lr1 - lr0) * f)
        d = Others @ x
        M = float(np.max(d))
        E = np.exp(np.clip((d - M) * beta, -30.0, 0.0))
        S = float(np.sum(E)) + 1e-300
        W = E / S
        g = Others.T @ W
        g = g - float(np.dot(g, x)) * x
        x = x - lr * g
        n = float(np.linalg.norm(x))
        if n < 1e-12 or not np.isfinite(n):
            x = x0 / max(float(np.linalg.norm(x0)), 1e-12)
            continue
        x /= n
    return x

def _joint_block_adam(Block0, Frozen, steps=80, beta=10000.0, lr=0.008):
    B = Block0.copy()
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    K = B.shape[0]
    m = np.zeros_like(B)
    v = np.zeros_like(B)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    Ik = np.eye(K)
    for s in range(steps):
        Dfb = B @ Frozen.T
        Dbb = B @ B.T
        M = float(max(np.max(Dfb), np.max(Dbb - Ik * 10.0)))
        Efb = np.exp(np.clip((Dfb - M) * beta, -30.0, 0.0))
        Ebb = np.exp(np.clip((Dbb - M) * beta, -30.0, 0.0))
        np.fill_diagonal(Ebb, 0.0)
        S = float(np.sum(Efb) + 0.5 * np.sum(Ebb)) + 1e-300
        Wfb = Efb / S
        Wbb = Ebb / S
        G = Wfb @ Frozen + Wbb @ B
        G = G - np.sum(G * B, axis=1, keepdims=True) * B
        t += 1
        m = b1 * m + (1.0 - b1) * G
        v = b2 * v + (1.0 - b2) * (G * G)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        step = (lr / c1) * m / (np.sqrt(v / c2) + eps)
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 5.0 * lr + 1e-12
        step = step * np.minimum(1.0, cap / sn)
        B = B - step
        B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return B

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
        if S < 1e-100:
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

def _lp_polish(x, deadline, k_top=4500, r0=1.2e-3):
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
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y /= yn
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

def solve():
    t_start = time.time()
    REGROW_END = t_start + 135.0
    FINAL_END = t_start + 172.0
    n, d = 250, 6
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    rng0 = np.random.default_rng(0)
    if not cands:
        z = rng0.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)
    # short collective settle
    try:
        if time.time() < t_start + 25:
            cur = _adam_full(best, 80, 4000.0, 15000.0, 0.008, 0.002)
            v = _max_cos(cur)
            if v < best_val and np.all(np.isfinite(cur)):
                best, best_val = cur, v
    except Exception:
        pass
    sizes_cycle = [25, 15, 40, 25, 30, 20, 40, 15, 25, 40]
    trial = 0
    while time.time() < REGROW_END and trial < 40:
        if time.time() > REGROW_END - 5:
            break
        Kwant = sizes_cycle[trial % len(sizes_cycle)]
        # also try exact spec sizes 15,25,40 cycling already covers
        seed = 1000 + trial * 17
        rng = np.random.default_rng(seed)
        trial += 1
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            a = k // n; b = k % n
            maxv = float(G[a, b])
            thresh = maxv - 2e-5
            Iu, Ju = np.where(np.triu(G, 1) > thresh)
            adj = [[] for _ in range(n)]
            for u, vv in zip(Iu, Ju):
                u = int(u); vv = int(vv)
                adj[u].append(vv); adj[vv].append(u)
            seen = np.zeros(n, dtype=bool)
            stack = [int(a)]
            seen[int(a)] = True
            comp = []
            while stack:
                u = stack.pop(); comp.append(u)
                for w in adj[u]:
                    if not seen[w]:
                        seen[w] = True; stack.append(w)
            rowmax = G.max(axis=1)
            comp_sorted = sorted(comp, key=lambda u: -rowmax[u])
            victims = [int(a), int(b)]
            for u in comp_sorted:
                if len(victims) >= Kwant:
                    break
                if u not in victims:
                    victims.append(int(u))
            if len(victims) < Kwant:
                order = np.argsort(-rowmax)
                for u in order:
                    u = int(u)
                    if u not in victims:
                        victims.append(u)
                    if len(victims) >= Kwant:
                        break
            victims = victims[:Kwant]
            K = len(victims)
            vset = set(victims)
            keep_idx = [i for i in range(n) if i not in vset]
            Frozen = best[keep_idx].copy()
            Block_list = []
            aborted = False
            for ii in range(K):
                if time.time() > REGROW_END - 2:
                    aborted = True; break
                if Block_list:
                    Bp = np.stack(Block_list, axis=0)
                    C = np.vstack([Frozen, Bp])
                else:
                    C = Frozen
                cand = rng.standard_normal((8192, d))
                cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
                D = cand @ C.T
                mx = D.max(axis=1)
                bi = int(np.argmin(mx))
                x0 = cand[bi]
                xr = _refine_one(x0, C, steps=60)
                Block_list.append(xr)
                if len(Block_list) % 5 == 0 and len(Block_list) >= 5:
                    try:
                        Bp2 = np.stack(Block_list, axis=0)
                        Bp2 = _joint_block_adam(Bp2, Frozen, steps=60, beta=9000.0, lr=0.008)
                        Block_list = [Bp2[i] for i in range(Bp2.shape[0])]
                    except Exception:
                        pass
            if aborted or len(Block_list) != K:
                continue
            Bp = np.stack(Block_list, axis=0)
            try:
                Bp = _joint_block_adam(Bp, Frozen, steps=100, beta=12000.0, lr=0.006)
            except Exception:
                pass
            Xnew = best.copy()
            for vi, xv in zip(victims, Bp):
                nv = float(np.linalg.norm(xv))
                if nv < 1e-12 or not np.isfinite(nv):
                    continue
                Xnew[vi] = xv / nv
            try:
                Xnew = _adam_full(Xnew, 60, 6000.0, 15000.0, 0.006, 0.0015)
            except Exception:
                pass
            vnew = _max_cos(Xnew)
            if vnew < best_val and np.all(np.isfinite(Xnew)):
                best = Xnew; best_val = vnew
        except Exception:
            continue
    # final collective polish
    try:
        if time.time() < FINAL_END - 8:
            cur = _adam_full(best, 120, 5000.0, 30000.0, 0.006, 0.001)
            v = _max_cos(cur)
            if v < best_val and np.all(np.isfinite(cur)):
                best, best_val = cur, v
    except Exception:
        pass
    try:
        if time.time() < FINAL_END - 3:
            cur2 = _lp_polish(best, FINAL_END - 1.0)
            v2 = _max_cos(cur2)
            if v2 < best_val and np.all(np.isfinite(cur2)):
                best, best_val = cur2, v2
    except Exception:
        pass
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    best = np.where(np.isfinite(best), best, 0.0)
    nn = np.linalg.norm(best, axis=1, keepdims=True)
    nn[nn < 1e-12] = 1.0
    best = best / nn
    return best
