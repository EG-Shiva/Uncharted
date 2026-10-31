import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        C = (x @ x.T)[iu]
        m = float(np.max(C))
        sel = C > m - max(25.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-C[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-14:
            x, best = y, v
            radius = min(radius * 1.6, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 168.0
    SEARCH_END = t0 + 132.0
    rng = np.random.RandomState(42)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(168, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    try:
        xr = np.load("record.npy").astype(float)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr
    except Exception:
        pass
    n, d = 168, 4
    if xb.shape != (168, 4):
        xb = rng.randn(168, 4)
        xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = _max_cos(xb)
    x_best = xb.copy()

    def refine(x_init, steps, mu=0.9, beta0=30.0, beta1=400.0, lr0=0.012, lr1=0.0008):
        x = x_init.copy()
        vel = np.zeros_like(x)
        for s in range(steps):
            if s > 200 and time.time() >= SEARCH_END:
                break
            frac = s / max(1, steps - 1)
            beta = beta0 * (beta1 / beta0) ** frac
            lr = lr0 * (1.0 - frac) + lr1 * frac
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -20.0, 0.0))
            np.fill_diagonal(E, 0.0)
            grad = E @ x
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x

    def hole_candidates(X_keep, n_sample=20000, n_ref=40):
        S = rng.randn(n_sample, 4)
        S /= np.linalg.norm(S, axis=1, keepdims=True)
        D = S @ X_keep.T
        worst = D.max(axis=1)
        idx = np.argpartition(worst, n_ref)[:n_ref]
        cand = S[idx]
        # local ascent: minimize smooth max to X_keep
        for _ in range(60):
            D2 = cand @ X_keep.T
            m = D2.max(axis=1, keepdims=True)
            # softmax weights over keep points
            beta = 60.0
            W = np.exp(np.clip(beta * (D2 - m), -20, 0))
            W /= (W.sum(axis=1, keepdims=True) + 1e-30)
            grad = W @ X_keep
            # project: we want to decrease max dot, so step opposite grad tangentially
            dot = np.sum(grad * cand, axis=1, keepdims=True)
            gt = grad - dot * cand
            cand = cand - 0.05 * gt
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        Df = cand @ X_keep.T
        wf = Df.max(axis=1)
        order = np.argsort(wf)
        return cand[order]

    def reinsert_cluster(x, cluster):
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[cluster] = False
        cur = x[keep_mask]
        holes = hole_candidates(cur, n_sample=20000, n_ref=40)
        # assign sequentially, refreshing every 2 inserts
        new_pts = []
        hi = 0
        for k in range(len(cluster)):
            if k > 0 and k % 2 == 0:
                tmp = np.vstack([cur] + [np.array(new_pts)])
                holes = hole_candidates(tmp, n_sample=15000, n_ref=20)
                hi = 0
            new_pts.append(holes[hi])
            hi += 1
        x2 = x.copy()
        x2[cluster] = np.array(new_pts)
        return x2

    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 1.5:
            break
        x = x_best.copy()
        typ = trial % 8
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        deg = np.sum(G > cmax - 0.008, axis=1)

        if typ == 0:
            x = x + 0.015 * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 700
            x = refine(x, steps)
        elif typ == 1:
            x = x + 0.060 * rng.randn(n, d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps = 1000
            x = refine(x, steps)
        elif typ == 2:
            order = np.argsort(-deg, kind="stable")
            i0 = order[rng.randint(0, 12) % n]
            nbr = np.argsort(-G[i0], kind="stable")[:8]
            x[nbr] += 0.15 * rng.randn(len(nbr), d)
            x[i0] += 0.15 * rng.randn(d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = refine(x, 1000)
        elif typ == 3:
            order = np.argsort(-deg, kind="stable")
            i0 = order[rng.randint(0, 10) % n]
            nn = np.argsort(-G[i0], kind="stable")[:rng.randint(3, 6)]
            cluster = np.concatenate([[i0], nn])
            # dedupe
            cluster = np.unique(cluster)
            x = reinsert_cluster(x, cluster)
            # small jitter after insert
            x[cluster] += 0.01 * rng.randn(len(cluster), d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = refine(x, 1200, beta0=25.0, beta1=500.0, lr0=0.014, lr1=0.0007)
        elif typ == 4:
            idx = rng.choice(n, 15, replace=False)
            x[idx] += 0.22 * rng.randn(len(idx), d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = refine(x, 900)
        elif typ == 5:
            deg0 = np.sum(G > cmax - 1e-4, axis=1)
            loose = np.where(deg0 <= 4)[0]
            if len(loose) == 0:
                loose = np.argsort(deg0)[:4]
            li = loose[rng.randint(0, len(loose))]
            X_keep = np.delete(x, li, axis=0)
            h = hole_candidates(X_keep, n_sample=18000, n_ref=12)[0]
            x[li] = h
            G2 = x @ x.T
            np.fill_diagonal(G2, -2.0)
            nbr = np.argsort(-G2[li], kind="stable")[:5]
            x[nbr] += 0.05 * rng.randn(len(nbr), d)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = refine(x, 900)
        elif typ == 6:
            # worst-pair split: push apart top pairs
            iu = np.triu_indices(n, 1)
            flat = G[iu]
            k = 6
            top = np.argpartition(-flat, k)[:k]
            I = iu[0][top]; J = iu[1][top]
            for a, b in zip(I, J):
                # tangent push
                dab = x[a] - flat[np.where((iu[0]==a)&(iu[1]==b))[0][0] if False else 0]*x[b] if False else x[a]-x[b]
                # simpler: move a away from b along tangent at a
                rab = x[b] - np.dot(x[b], x[a]) * x[a]
                rba = x[a] - np.dot(x[a], x[b]) * x[b]
                na = np.linalg.norm(rab)+1e-12; nb = np.linalg.norm(rba)+1e-12
                x[a] -= 0.06 * rab / na
                x[b] -= 0.06 * rba / nb
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = refine(x, 900)
        else:
            # large jump, occasional fresh
            if trial % 24 == 7:
                x = rng.randn(n, d)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                x = refine(x, 1800, beta0=20.0, beta1=350.0, lr0=0.014, lr1=0.0008)
            else:
                x = x + 0.14 * rng.randn(n, d)
                x /= np.linalg.norm(x, axis=1, keepdims=True)
                x = refine(x, 1100)

        v = _max_cos(x)
        if v < best_val - 2e-12:
            # quick LP squeeze to get true basin value if promising and time allows
            if v < best_val - 5e-7 and time.time() < SEARCH_END - 8:
                xq = _lp_polish(x, time.time() + 2.5)
                vq = _max_cos(xq)
                if vq < v:
                    x, v = xq, vq
            if v < best_val:
                best_val = v
                x_best = x.copy()

    # final tightening: high-beta anneal then LP with remaining time
    if time.time() < HARD_END - 3:
        try:
            x_best = refine(x_best, 400, beta0=150.0, beta1=900.0, lr0=0.004, lr1=0.0003)
        except Exception:
            pass
        v = _max_cos(x_best)
        if v < best_val:
            best_val = v
        deadline = min(HARD_END - 1.0, time.time() + 28.0)
        if deadline > time.time() + 1.0:
            x_best = _lp_polish(x_best, deadline)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
