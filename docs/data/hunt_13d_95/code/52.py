import time
import numpy as np
from collections import deque
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, radius_init=1e-3, max_pairs=1300):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.08:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > max_pairs:
            idx = np.argpartition(c, -max_pairs)[-max_pairs:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _local_min_hole(h, others, steps=140, beta=200.0):
    h = h / np.linalg.norm(h)
    best_h = h.copy()
    best_m = float((others @ h).max())
    cur = h.copy()
    for s in range(steps):
        dots = others @ cur
        m = float(dots.max())
        w = np.exp(beta * (dots - m))
        sw = w.sum()
        if not np.isfinite(sw) or sw == 0:
            break
        w /= sw
        grad = w @ others
        gt = grad - float(grad @ cur) * cur
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-12 or not np.isfinite(nrm):
            break
        gt /= nrm
        f = s / max(1, steps - 1)
        lr = 0.02 * (1 - f) + 0.001 * f
        cur = cur - lr * gt
        nr = float(np.linalg.norm(cur))
        if nr == 0 or not np.isfinite(nr):
            break
        cur /= nr
        m2 = float((others @ cur).max())
        if m2 < best_m:
            best_m = m2
            best_h = cur.copy()
    return best_h


def _find_hole(others, rng, n_samples=20000):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_samples, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    md = dots.max(axis=1)
    # take top few and locally minimise
    k = 4
    idx = np.argpartition(md, k - 1)[:k]
    # order by md
    idx = idx[np.argsort(md[idx])]
    best_h = None
    best_m = np.inf
    for j in idx:
        h0 = cand[int(j)]
        h = _local_min_hole(h0, others, steps=140, beta=200.0)
        m = float((others @ h).max())
        if m < best_m:
            best_m = m
            best_h = h
    return best_h


def _refine_inserted(X, R_idx, steps=550, beta0=50.0, beta1=800.0, lr0=0.03, lr1=0.0025, mu=0.9):
    n, d = X.shape
    K = len(R_idx)
    if K == 0:
        return X
    Rmask = np.zeros(n, dtype=bool)
    Rmask[R_idx] = True
    Rel = (Rmask[:, None] | Rmask[None, :])
    np.fill_diagonal(Rel, False)
    V = np.zeros((K, d))
    # map row -> position in V
    pos = np.full(n, -1, dtype=int)
    pos[R_idx] = np.arange(K)
    lB0, lB1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lB0 + (lB1 - lB0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        # max over relevant pairs for shift
        rel_vals = G[Rel]
        if rel_vals.size == 0:
            break
        m = float(rel_vals.max())
        W = np.exp(beta * (G - m))
        W[~Rel] = 0.0
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw == 0:
            break
        W /= sw
        Grad = W @ X  # n x d
        Gi = Grad[R_idx]  # K x d
        Xi = X[R_idx]
        dot = np.sum(Gi * Xi, axis=1, keepdims=True)
        Gt = Gi - dot * Xi
        gn = float(np.linalg.norm(Gt))
        if not np.isfinite(gn) or gn == 0:
            break
        Gt = Gt / (gn / np.sqrt(float(K)))
        V = mu * V + Gt
        Xn = Xi - lr * V
        Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
        X[R_idx] = Xn
    return X


def _build_removal(X, target_K, hub, rng):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    mc = float(G.max())
    adj = G > (mc - 1e-9)
    np.fill_diagonal(adj, False)
    cc = adj.sum(axis=1)
    low = np.where(cc < 8)[0].tolist()
    Rset = set(int(v) for v in low)
    Rset.add(int(hub))
    q = deque([int(hub)])
    visited = set([int(hub)])
    # also mark low as visited to avoid immediate re-add confusion
    for v in low:
        visited.add(int(v))
    guard = 0
    while len(Rset) < target_K and guard < 5000:
        guard += 1
        if len(q) == 0:
            # frontier neighbours of Rset
            cand = []
            for u in list(Rset):
                nbrs = np.where(adj[u])[0].tolist()
                for v in nbrs:
                    if v not in Rset:
                        cand.append(int(v))
            if not cand:
                rem = [i for i in range(n) if i not in Rset]
                if not rem:
                    break
                # prefer low contact among remainder
                rem_cc = cc[np.array(rem)]
                order = np.argsort(rem_cc)
                # random among 8 lowest
                pick = int(rng.integers(0, min(len(rem), 8)))
                Rset.add(int(np.array(rem)[order[pick]]))
                continue
            # dedup + shuffle
            cand = list(set(cand))
            rng.shuffle(cand)
            Rset.add(int(cand[0]))
            if int(cand[0]) not in visited:
                visited.add(int(cand[0]))
                q.append(int(cand[0]))
        else:
            u = q.popleft()
            nbrs = np.where(adj[u])[0].tolist()
            rng.shuffle(nbrs)
            for v in nbrs:
                v = int(v)
                if v not in visited:
                    visited.add(v)
                    q.append(v)
                if v not in Rset and len(Rset) < target_K:
                    Rset.add(v)
            if len(nbrs) == 0:
                # isolated: add random low-contact point
                rem = [i for i in range(n) if i not in Rset]
                if rem:
                    rem_cc = cc[np.array(rem)]
                    order = np.argsort(rem_cc)
                    pick = int(rng.integers(0, min(len(rem), 8)))
                    nv = int(np.array(rem)[order[pick]])
                    Rset.add(nv)
                    if nv not in visited:
                        visited.add(nv)
                        q.append(nv)
    Rlist = list(Rset)
    if len(Rlist) > target_K:
        rng.shuffle(Rlist)
        if int(hub) in Rlist:
            filt = [int(hub)] + [v for v in Rlist if v != int(hub)][:target_K - 1]
            Rlist = filt
        else:
            Rlist = Rlist[:target_K]
    rng.shuffle(Rlist)
    return np.array(sorted(Rlist), dtype=int), cc, mc


def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    EXPL_END = t0 + 138.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (95, 13):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (95, 13):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        rng0 = np.random.default_rng(0)
        z = rng0.standard_normal((95, 13))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best_val = float(vals[bi])
    n, d = 95, 13

    trial = 0
    # hub 44 first as directed, then vary
    while time.time() < EXPL_END and time.time() < deadline - 32:
        trial += 1
        if time.time() > EXPL_END - 2.0:
            break
        rng = np.random.default_rng(1000 + trial * 131)
        # contact info of current best
        G0 = x_best @ x_best.T
        np.fill_diagonal(G0, -2.0)
        mc0 = float(G0.max())
        cc0 = (G0 > mc0 - 1e-9).sum(axis=1)
        order_hub = np.argsort(-cc0)
        if trial == 1:
            hub = 44 if 0 <= 44 < n else int(order_hub[0])
        elif trial == 2:
            hub = int(order_hub[0])
        elif trial <= 6:
            hub = int(order_hub[int(rng.integers(0, min(5, n)))])
        else:
            if rng.random() < 0.7:
                hub = int(order_hub[int(rng.integers(0, min(8, n)))])
            else:
                hub = int(rng.integers(0, n))
        target_K = int(rng.integers(15, 23))  # 15..22
        R_idx, cc_, _ = _build_removal(x_best, target_K, hub, rng)
        if len(R_idx) < 15:
            # pad to 15 with random
            rem = [i for i in range(n) if i not in set(R_idx.tolist())]
            rng.shuffle(rem)
            need = 15 - len(R_idx)
            R_idx = np.array(sorted(R_idx.tolist() + rem[:need]), dtype=int)
        if len(R_idx) > 22:
            R_idx = np.array(sorted(rng.choice(R_idx, size=22, replace=False)), dtype=int)
        K = len(R_idx)
        Rset = set(R_idx.tolist())
        # random insertion order
        ins_order = R_idx.copy()
        rng.shuffle(ins_order)
        X_new = x_best.copy()
        core_mask = np.ones(n, dtype=bool)
        core_mask[R_idx] = False
        others = X_new[core_mask].copy()
        # greedy hole insertion one at a time
        for r in ins_order:
            if time.time() > deadline - 28:
                break
            h = _find_hole(others, rng, n_samples=20000)
            X_new[int(r)] = h
            others = np.concatenate([others, h[None, :]], axis=0)
        # joint refine only inserted vs frozen core
        X_new = _refine_inserted(X_new, np.array(sorted(ins_order), dtype=int),
                                 steps=550, beta0=50.0, beta1=800.0,
                                 lr0=0.03, lr1=0.0025, mu=0.9)
        v_mid = _max_cos(X_new)
        if v_mid > best_val + 0.025:
            continue
        t_rem = deadline - time.time()
        if t_rem < 8:
            break
        # short LP polish unfrozen
        lp_sec = 1.4
        if v_mid < best_val:
            lp_sec = 2.0
        lp_sec = min(lp_sec, max(0.7, t_rem - 26.0))
        if lp_sec < 0.6:
            continue
        Xp = _lp_polish(X_new, seconds=lp_sec, radius_init=1e-3)
        v2 = _max_cos(Xp)
        if v2 < best_val - 1e-12:
            x_best = Xp
            best_val = v2

    t_rem = deadline - time.time()
    if t_rem > 3.0:
        x_best = _lp_polish(x_best, seconds=max(1.0, t_rem - 1.0), radius_init=2e-3)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
