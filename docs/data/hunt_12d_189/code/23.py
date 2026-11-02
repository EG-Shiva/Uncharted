import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _pt_max(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1)

def _lp_polish(x, deadline, radius_init=8e-4, max_pairs=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thr = m - max(8.0 * radius, 1.5e-4)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > max_pairs:
            idx = np.argpartition(-c, max_pairs)[:max_pairs]
            mask = np.zeros_like(c, dtype=bool)
            mask[idx] = True
            sel = mask
            cnt = max_pairs
        I, J = iu[0][sel], iu[1][sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 1.5})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return x, best

def _anneal_fast(x, steps, a0, a1, l0, l1, noise=0.0, rng=None):
    n, d = x.shape
    for s in range(steps):
        f = s / (steps - 1) if steps > 1 else 0.0
        alpha = a0 * (a1 / a0) ** f
        lr = l0 + (l1 - l0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # exp in stable way
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True) + 1e-30
        S = (W @ x) / row
        dot = np.sum(S * x, axis=1, keepdims=True)
        Gt = S - dot * x
        nr = np.linalg.norm(Gt, axis=1, keepdims=True)
        Gt = Gt * np.minimum(1.0, 1.0 / np.maximum(1.0, nr))
        x = x - lr * Gt
        if noise > 0:
            sig = noise * (1.0 - f)
            if sig > 1e-12:
                R = rng.standard_normal((n, d))
                R = R - np.sum(R * x, axis=1, keepdims=True) * x
                x = x + sig * R
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _deep_hole(x, exclude, M, rng):
    n, d = x.shape
    best_v = 2.0
    best_r = None
    chunk = 5000
    done = 0
    while done < M:
        c = min(chunk, M - done)
        R = rng.standard_normal((c, d))
        R = R / np.linalg.norm(R, axis=1, keepdims=True)
        D = R @ x.T
        if exclude is not None:
            if np.isscalar(exclude):
                D[:, int(exclude)] = -2.0
            else:
                # list
                for e in exclude:
                    D[:, int(e)] = -2.0
        vals = np.max(D, axis=1)
        j = int(np.argmin(vals))
        if float(vals[j]) < best_v:
            best_v = float(vals[j])
            best_r = R[j].copy()
        done += c
    # local refinement: push away from nearest neighbours
    r = best_r
    for _ in range(12):
        dots = x @ r
        if exclude is not None:
            if np.isscalar(exclude):
                dots[int(exclude)] = -2.0
        # 8 nearest
        k = 8
        idx = np.argpartition(-dots, k)[:k]
        g = np.sum(x[idx] - dots[idx, None] * r[None, :], axis=0)
        # g points toward neighbours, move opposite tangentially
        g = g - np.dot(g, r) * r
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        step = 0.02
        rn = r - step * g / ng
        rn = rn / np.linalg.norm(rn)
        # accept if better
        dm = float(np.max(x @ rn)) if exclude is None else float(np.max(np.delete(x @ rn, exclude if np.isscalar(exclude) else list(exclude))))
        # for list case approximate: recompute with mask
        if exclude is not None and not np.isscalar(exclude):
            dd = x @ rn
            dd[list(exclude)] = -2.0
            dm = float(np.max(dd))
        elif exclude is not None:
            dd = x @ rn
            dd[int(exclude)] = -2.0
            dm = float(np.max(dd))
        if dm < best_v:
            best_v = dm
            r = rn
        else:
            break
    return r

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    soft_dead = t0 + 155.0
    rng = np.random.default_rng(987654321)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=np.float64)
            if a.shape == (189, 12):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append((_max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((189, 12))
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_val, best_x = cands[0][0], cands[0][1].copy()
    n, d = 189, 12
    trial = 0
    while time.time() < soft_dead and trial < 600:
        mode = trial % 5
        x = best_x.copy()
        # quick stats for hub/loose detection
        G0 = x @ x.T
        np.fill_diagonal(G0, -2.0)
        pt = np.max(G0, axis=1)
        m0 = float(np.max(pt))
        # contact counts with tight threshold
        thr_c = m0 - 3e-4
        cnt = np.sum(G0 > thr_c, axis=1)
        order_pt = np.argsort(pt)  # loose first
        order_cnt = np.argsort(cnt)
        if mode == 0:
            # hub deletion: worst by count+pt
            nh = 3 + (trial % 3)  # 3..5
            # pick top nh by cnt, tie by pt
            hubs = np.argsort(cnt * 10.0 + pt)[-nh:]
            for h in hubs:
                r = _deep_hole(x, int(h), 10000, rng)
                x[int(h)] = r
            steps, a0, a1, l0, l1, nz = 260, 25.0, 220.0, 0.30, 0.02, 0.0
        elif mode == 1:
            # rattler refill + hub jitter
            k = 5
            loose = order_pt[:k]
            for idx in loose:
                r = _deep_hole(x, int(idx), 8000, rng)
                x[int(idx)] = r
            # jitter overloaded hubs slightly
            hubs = order_cnt[-4:]
            x[hubs] += 0.015 * rng.standard_normal((len(hubs), d))
            x[hubs] /= np.linalg.norm(x[hubs], axis=1, keepdims=True)
            steps, a0, a1, l0, l1, nz = 240, 30.0, 240.0, 0.28, 0.018, 0.002
        elif mode == 2:
            # cap shake along random direction
            v = rng.standard_normal(d); v /= np.linalg.norm(v)
            proj = x @ v
            q = 0.78 + 0.02 * (trial % 5)
            th = np.quantile(proj, q)
            idxs = np.where(proj > th)[0]
            sigma = 0.04 + 0.01 * (trial % 3)
            x[idxs] += sigma * rng.standard_normal((len(idxs), d))
            x[idxs] /= np.linalg.norm(x[idxs], axis=1, keepdims=True)
            # hole-fill single worst
            w = int(order_pt[-1])
            # don't fill worst, fill loosest to keep diversity
            l = int(order_pt[0])
            r = _deep_hole(x, l, 6000, rng)
            x[l] = r
            steps, a0, a1, l0, l1, nz = 260, 28.0, 250.0, 0.30, 0.018, 0.0
        elif mode == 3:
            # worst-pair surgery: for top 3 pairs move one endpoint to hole
            iu = np.triu_indices(n, 1)
            # find top pairs quickly (partial)
            # use flattened argpartition on G0 upper? approximate via pt
            # brute force top 3 via max search iteratively
            Gc = G0.copy()
            for _ in range(3):
                ai, aj = np.unravel_index(int(np.argmax(Gc)), Gc.shape)
                # replace the one with larger pt
                victim = ai if pt[ai] > pt[aj] else aj
                r = _deep_hole(x, int(victim), 8000, rng)
                x[int(victim)] = r
                # mask out pairs involving victim for next search
                Gc[victim, :] = -2.0; Gc[:, victim] = -2.0
                # update pt roughly
            steps, a0, a1, l0, l1, nz = 280, 22.0, 200.0, 0.32, 0.02, 0.0
        else:
            # global moderate shake of random subset
            sel = rng.choice(n, size=28, replace=False)
            sigma = 0.012 + 0.005 * (trial % 3)
            x[sel] += sigma * rng.standard_normal((len(sel), d))
            x[sel] /= np.linalg.norm(x[sel], axis=1, keepdims=True)
            steps, a0, a1, l0, l1, nz = 220, 35.0, 260.0, 0.22, 0.015, 0.001
        x = _anneal_fast(x, steps, a0, a1, l0, l1, noise=nz, rng=rng)
        v = _max_cos(x)
        if v < best_val - 1e-12:
            best_val = v
            best_x = x.copy()
        trial += 1
    # final squeeze with remaining time
    if time.time() < deadline - 1.5:
        rem = min(deadline - 0.8, time.time() + 14.0)
        best_x, v2 = _lp_polish(best_x, rem, radius_init=6e-4)
        best_val = min(best_val, v2)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
