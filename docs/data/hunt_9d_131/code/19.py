import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lse_max(X, tau=0.008):
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = float(np.max(G))
    E = np.exp((G - m) / max(tau, 1e-12))
    np.fill_diagonal(E, 0.0)
    S = float(np.sum(E)) / 2.0
    if S <= 0 or not np.finite(S):
        return m
    return m + tau * np.log(S)

def _anneal(X, steps, lr0, lr1, tau0, tau1):
    n, d = X.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        lr = lr0 * ((lr1 / lr0) ** f) if lr0 > 0 and lr1 > 0 else lr0
        tau = tau0 * (1 - f) + tau1 * f
        if tau < 1e-9:
            tau = 1e-9
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        E = np.exp((G - m) / tau)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if S <= 0 or not np.isfinite(S):
            break
        P = E / S
        Gr = P @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Ft = Gr - dot * X
        mx = float(np.max(np.sqrt(np.sum(Ft * Ft, axis=1))))
        if mx < 1e-12 or not np.isfinite(mx):
            break
        X = X - (lr / mx) * Ft
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _spherical_kmeans(Xc, K, rng, restarts=20, iters=12):
    N, d = Xc.shape
    best_score = -1e100
    best_labels = None
    best_C = None
    for _ in range(restarts):
        idx = rng.choice(N, size=K, replace=False)
        C = Xc[idx].copy()
        labels = np.zeros(N, dtype=int)
        for _it in range(iters):
            S = Xc @ C.T
            labels = np.argmax(S, axis=1)
            Cn = np.zeros_like(C)
            for k in range(K):
                m = labels == k
                if np.any(m):
                    v = np.sum(Xc[m], axis=0)
                    nrm = np.linalg.norm(v)
                    if nrm < 1e-12:
                        v = Xc[rng.integers(0, N)]
                        nrm = np.linalg.norm(v)
                    Cn[k] = v / nrm
                else:
                    v = Xc[rng.integers(0, N)]
                    Cn[k] = v / np.linalg.norm(v)
            if np.allclose(Cn, C, atol=1e-8):
                C = Cn
                break
            C = Cn
        S = Xc @ C.T
        sc = float(np.sum(np.max(S, axis=1)))
        if sc > best_score:
            best_score = sc
            best_labels = np.argmax(S, axis=1).copy()
            best_C = C.copy()
    return best_labels, best_C

def _lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end and radius > 1e-14:
        Gfull = x @ x.T
        c_all = Gfull[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        if p > 5000:
            o = np.argsort(-c)[:5000]
            I, J, c = I[o], J[o], c[o]
            p = 5000
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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

def solve():
    t0 = time.time()
    TIME_LIMIT = 178.0
    LP_BUDGET = 26.0
    explore_end = t0 + TIME_LIMIT - LP_BUDGET - 2.0
    rng = np.random.default_rng(0)
    n, d = 131, 9
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v /= np.linalg.norm(v, axis=1, keepdims=True)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best = v.copy()
        except Exception:
            pass
    if best is None:
        best = rng.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    # elite pool of 5: list of (hard, soft, X)
    def soft_of(X):
        return _lse_max(X, tau=0.008)
    elites = [(best_val, soft_of(best), best.copy())]

    def push_elite(h, s, X):
        elites.append((h, s, X.copy()))
        elites.sort(key=lambda z: z[0])
        # diversity prune: if two elites too close, drop worse
        # use mean abs dot of sorted? cheap: Frobenius after best alignment sign? just keep top 5
        del elites[5:]

    it = 0
    thetas = [0.15, 0.3, 0.6]
    while time.time() < explore_end:
        it += 1
        # pick base: 70% best, else random elite
        if len(elites) > 1 and rng.random() < 0.30:
            bidx = int(rng.integers(0, len(elites)))
            base = elites[bidx][2]
        else:
            base = elites[0][2]
        incumbent = elites[0][0]
        Xb = base.copy()
        mv = _max_cos(Xb)
        # core mask: points with a near-max contact
        G0 = Xb @ Xb.T
        np.fill_diagonal(G0, -2.0)
        rowmax = np.max(G0, axis=1)
        core_mask = rowmax >= (mv - 1e-6)
        core_idx = np.where(core_mask)[0]
        # if too few/many, fallback to all
        if core_idx.size < 30 or core_idx.size > 130:
            core_idx = np.arange(n)
        Xc = Xb[core_idx]
        Nc = Xc.shape[0]
        # choose K 8..12
        K = int(rng.integers(8, 13))
        K = min(K, Nc - 1)
        if K < 2:
            K = 2
        try:
            labels, _ = _spherical_kmeans(Xc, K, rng, restarts=20, iters=12)
        except Exception:
            # fallback small shake
            Y = Xb + 0.02 * rng.standard_normal(Xb.shape)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            Y = _anneal(Y, steps=1200, lr0=0.02, lr1=0.002, tau0=0.03, tau1=0.004)
            h = _max_cos(Y); s = _lse_max(Y)
            if h < elites[0][0] - 1e-12 or s < elites[0][1] - 1e-9:
                push_elite(h, s, Y)
            elif h < elites[-1][0] - 2e-6 and len(elites) < 5:
                push_elite(h, s, Y)
            continue
        # collect clusters size 5..18
        uniq, counts = np.unique(labels, return_counts=True)
        cand_clusters = [u for u, c in zip(uniq, counts) if 5 <= c <= 18]
        if len(cand_clusters) == 0:
            # pick cluster closest to range
            # prefer 5..18 proximity
            best_u = None; best_pen = 1e9
            for u, c in zip(uniq, counts):
                pen = 0 if 5 <= c <= 18 else (5 - c if c < 5 else c - 18)
                # add randomness
                pen = pen + rng.random() * 0.5
                if pen < best_pen:
                    best_pen = pen; best_u = u
            cand_clusters = [best_u]
        cu = cand_clusters[int(rng.integers(0, len(cand_clusters)))]
        local_pos = np.where(labels == cu)[0]
        glob = core_idx[local_pos]
        # build two-plane: u = centroid dir, v = random orth
        cen = np.sum(Xb[glob], axis=0)
        nrm = np.linalg.norm(cen)
        if nrm < 1e-12:
            cen = Xb[glob[0]]
            nrm = np.linalg.norm(cen)
        u = cen / nrm
        rv = rng.standard_normal(d)
        rv -= float(rv @ u) * u
        rn = np.linalg.norm(rv)
        if rn < 1e-12:
            rv = rng.standard_normal(d); rv -= float(rv @ u) * u; rn = np.linalg.norm(rv)
        v = rv / rn
        Y = Xb.copy()
        do_reflect = rng.random() < 0.25
        if do_reflect:
            hvec = rng.standard_normal(d); hvec /= np.linalg.norm(hvec)
            Xcl = Y[glob]
            proj = Xcl @ hvec
            Xcl = Xcl - 2.0 * proj[:, None] * hvec[None, :]
            Xcl /= np.linalg.norm(Xcl, axis=1, keepdims=True)
            Y[glob] = Xcl
        else:
            th = thetas[int(rng.integers(0, len(thetas)))]
            if rng.random() < 0.5:
                th = -th
            co, si = float(np.cos(th)), float(np.sin(th))
            Xcl = Y[glob]
            A = Xcl @ u
            B = Xcl @ v
            Ap = A * co - B * si
            Bp = A * si + B * co
            Xcl = Xcl + (Ap - A)[:, None] * u[None, :] + (Bp - B)[:, None] * v[None, :]
            Xcl /= np.linalg.norm(Xcl, axis=1, keepdims=True)
            Y[glob] = Xcl
        # annealed weighted repulsion 1200-2000 steps
        rpick = rng.random()
        if rpick < 0.4:
            steps = 1200
        elif rpick < 0.75:
            steps = 1500
        else:
            steps = 2000
            if time.time() > explore_end - 20:
                steps = 1200
        # tau/lr schedule jitter
        tau0 = 0.025 + 0.01 * rng.random()
        Y = _anneal(Y, steps=steps, lr0=0.022, lr1=0.0018, tau0=tau0, tau1=0.0035)
        h = _max_cos(Y); s = _lse_max(Y)
        # softened acceptance into pool of five
        if h < elites[0][0] - 1e-12:
            push_elite(h, s, Y)
        elif s < elites[0][1] - 5e-07 and h < elites[0][0] + 1e-4:
            # soft improvement with hard within tolerance -> diversity
            # avoid duplicate: require hard different enough or just keep if pool not full
            if len(elites) < 5 or h < elites[-1][0] - 1e-07 or s < elites[-1][1] - 1e-07:
                push_elite(h, s, Y)
        elif len(elites) < 5 and h < elites[-1][0]:
            push_elite(h, s, Y)
        # occasional hard-near-best diverse keep
        if time.time() >= explore_end:
            break
    # final: LP-polish only candidates within 1e-4 of incumbent
    elites.sort(key=lambda z: z[0])
    if len(elites) == 0:
        out = best
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out
    thresh = elites[0][0] + 1e-4
    cands = [e for e in elites if e[0] <= thresh][:5]
    t_end = t0 + TIME_LIMIT - 1.0
    polished = []
    for i, (h, s, X) in enumerate(cands):
        if time.time() >= t_end:
            polished.append((h, X))
            continue
        Xs = _anneal(X.copy(), steps=120, lr0=0.008, lr1=0.001, tau0=0.012, tau1=0.002)
        remain = t_end - time.time()
        left = len(cands) - i
        per = remain / max(1, left)
        per = min(max(per, 2.0), 10.0)
        try:
            Xp = _lp_polish(Xs, seconds=per, t_end=t_end)
        except Exception:
            Xp = Xs
        hp = _max_cos(Xp)
        polished.append((hp, Xp))
        # also keep original if LP hurt
        if hp > h:
            polished.append((h, X))
    # also consider unpolished elites beyond thresh in case LP hurt everything
    for h, s, X in elites:
        polished.append((h, X))
    polished.sort(key=lambda z: z[0])
    out = polished[0][1].copy()
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out
