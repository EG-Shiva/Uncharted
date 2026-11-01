import time
import numpy as np

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _softmax_quench(X, steps, T0, T1, lr, mu=0.9, t_deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(steps):
        if t_deadline is not None and (s % 200 == 0) and time.time() > t_deadline:
            break
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_X = X.copy()
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
    Gf = X @ X.T
    np.fill_diagonal(Gf, -2.0)
    mf = float(Gf.max())
    if mf < best_m:
        return X
    return best_X

def _jam_quench(X, t_end, steps, lr, mu=0.9, t_start=None, bg=0.02, Tbg=0.008, t_deadline=None):
    # minimize sum relu(G - t(s))^2 + bg * softmax, t(s) annealed to t_end
    n, d = X.shape
    V = np.zeros_like(X)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    m0 = float(G0.max())
    best_m = m0
    best_X = X.copy()
    if t_start is None:
        t_start = m0
    for s in range(steps):
        if t_deadline is not None and (s % 200 == 0) and time.time() > t_deadline:
            break
        frac = s / max(1, steps - 1)
        t_cur = t_start + (t_end - t_start) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_X = X.copy()
        # violation weights
        W = G - t_cur
        W[W < 0] = 0.0
        np.fill_diagonal(W, 0.0)
        GradV = W @ X
        # weak background softmax to move rattlers
        Gb = G.copy()
        np.fill_diagonal(Gb, -1e100)
        Q = (Gb - m) / Tbg
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        GradB = (E @ X) / rs
        Grad = GradV + bg * GradB
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.5
        if np.any(bad):
            Gt[bad] *= (2.5 / gn[bad])
        V = mu * V + Gt
        X = X - lr * V
        X = _norm_rows(X)
    return best_X

def _single_refine(s, R, steps=220, T=0.004, lr=0.12, mu=0.8):
    s = s / (np.linalg.norm(s) + 1e-12)
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = e.sum() + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        s = s / (np.linalg.norm(s) + 1e-12)
    return s

def _find_hole(R, rng, n_sample=8000, n_cand=3):
    Z = _norm_rows(rng.standard_normal(size=(n_sample, R.shape[1])))
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None
    best_v = 1e100
    for k in idx:
        s0 = Z[k]
        s1 = _single_refine(s0, R, steps=200, T=0.005, lr=0.12)
        s1 = _single_refine(s1, R, steps=200, T=0.0015, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    return best_s / (np.linalg.norm(best_s) + 1e-12)

def _crossover_child(A, B, rng):
    # align B to A then greedy subset selection from pool
    try:
        M = B.T @ A
        U, _, Vt = np.linalg.svd(M)
        Q = U @ Vt
        Bal = B @ Q
        Bal = _norm_rows(Bal)
    except Exception:
        Bal = B
    pool = np.vstack([A, Bal])
    P = len(pool)
    start = int(rng.integers(P))
    chosen = [start]
    # greedy farthest
    for _ in range(71):
        if len(chosen) >= 71:
            break
        C = pool[chosen]
        # max cos to chosen for each pool element
        S = pool @ C.T
        # ignore already chosen by setting large
        mask = np.ones(P, dtype=bool)
        mask[np.array(chosen)] = False
        if not np.any(mask):
            break
        best_val = np.max(S, axis=1)
        best_val[np.array(chosen)] = 1e100
        nxt = int(np.argmin(best_val))
        chosen.append(nxt)
    Xc = pool[np.array(chosen[:71])]
    return _norm_rows(Xc)

def _lp_polish(x, seconds, deadline=None):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm_rows(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 1500:
            order = np.argsort(-cmat[sel])[:1500]
            I, J = I[order], J[order]
            csel = cmat[sel][order]
        else:
            csel = cmat[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    TIME_LIMIT = 173.0
    rng = np.random.default_rng(12345)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = _norm_rows(np.asarray(np.load(fn), dtype=np.float64))
            if a.shape == (71, 12):
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_norm_rows(rng.standard_normal(size=(71, 12))))
    bestX = min(cands, key=lambda z: _max_cos(z)).copy()
    bestVal = _max_cos(bestX)
    curX = bestX.copy()
    curVal = bestVal
    archive = [(bestVal, bestX.copy())]
    if len(cands) > 1:
        for c in cands[1:]:
            archive.append((_max_cos(c), c.copy()))
        archive = sorted(archive, key=lambda t: t[0])[:6]

    def push_archive(v, X):
        archive.append((v, X.copy()))
        archive.sort(key=lambda t: t[0])
        del archive[6:]

    explore_deadline = t0 + 128.0
    bh_T = 0.0012
    it = 0
    while time.time() < explore_deadline:
        it += 1
        if time.time() > explore_deadline - 1.0:
            break
        r = rng.random()
        if r < 0.55:
            base = bestX
        elif len(archive) > 1 and r < 0.85:
            k = int(rng.integers(0, len(archive)))
            base = archive[k][1]
        else:
            base = curX
        mv = rng.random()
        Xt = None
        if mv < 0.30:
            # cluster-void kill: remove k nearest to random center
            k = int(rng.choice([5, 7, 9, 12, 15]))
            k = min(k, 20)
            c = int(rng.integers(71))
            d2 = np.sum((base - base[c]) ** 2, axis=1)
            victims = np.argsort(d2)[:k]
            mask = np.ones(71, dtype=bool)
            mask[victims] = False
            Xt_base = base[mask]
            for _ in range(k):
                s_new = _find_hole(Xt_base, rng, n_sample=7000, n_cand=3)
                Xt_base = np.vstack([Xt_base, s_new[None, :]])
            Xt = Xt_base
        elif mv < 0.45:
            # hub kill 2-4
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            thr = bestVal - 0.008
            cnt = (G > thr).sum(axis=1)
            order = np.argsort(-cnt)
            topk = order[:10]
            nrem = int(rng.choice([2, 3, 4]))
            victims = list(rng.choice(topk, size=nrem, replace=False))
            mask = np.ones(71, dtype=bool)
            mask[np.array(victims)] = False
            Xt_base = base[mask]
            for _ in range(nrem):
                s_new = _find_hole(Xt_base, rng, n_sample=7000, n_cand=3)
                Xt_base = np.vstack([Xt_base, s_new[None, :]])
            Xt = Xt_base
        elif mv < 0.60:
            # crossover
            if len(archive) >= 2:
                a_idx = int(rng.integers(len(archive)))
                b_idx = int(rng.integers(len(archive)))
                if b_idx == a_idx:
                    b_idx = (b_idx + 1) % len(archive)
                A = archive[a_idx][1]
                B = archive[b_idx][1]
                Xt = _crossover_child(A, B, rng)
                # small noise
                Xt = Xt + rng.standard_normal(size=(71, 12)) * 0.015
                Xt = _norm_rows(Xt)
            else:
                Xt = base.copy()
                idx = rng.choice(71, size=10, replace=False)
                Xt[idx] = Xt[idx] + rng.standard_normal(size=(10, 12)) * 0.35
                Xt = _norm_rows(Xt)
        elif mv < 0.75:
            # fresh random restart
            Xt = _norm_rows(rng.standard_normal(size=(71, 12)))
        else:
            # large shake
            Xt = base.copy()
            k = int(rng.choice([6, 10, 15, 20]))
            idx = rng.choice(71, size=k, replace=False)
            sigma = float(np.exp(rng.uniform(np.log(0.1), np.log(0.7))))
            if rng.random() < 0.1:
                Xt[idx] = _norm_rows(rng.standard_normal(size=(k, 12)))
            else:
                Xt[idx] = Xt[idx] + rng.standard_normal(size=(k, 12)) * sigma
                Xt = _norm_rows(Xt)
        # jam quench toward below best
        gap = float(rng.choice([0.0008, 0.0015, 0.0025, 0.004]))
        t_end = bestVal - gap
        # for fresh restarts use two-stage jam
        if mv >= 0.60 and mv < 0.75:
            steps = int(rng.choice([2000, 2800]))
            lr = float(rng.choice([0.05, 0.07, 0.04]))
            Xt = _jam_quench(Xt, t_end, steps=steps, lr=lr, mu=0.9, bg=0.03, Tbg=0.01, t_deadline=explore_deadline)
        else:
            steps = int(rng.choice([1400, 2000, 2600]))
            lr = float(rng.choice([0.05, 0.07, 0.035]))
            bg = float(rng.choice([0.01, 0.02, 0.04]))
            Xt = _jam_quench(Xt, t_end, steps=steps, lr=lr, mu=0.9, bg=bg, Tbg=0.008, t_deadline=explore_deadline)
        v = _max_cos(Xt)
        # short softmax tighten if promising
        if v < bestVal + 0.004 and time.time() < explore_deadline - 1:
            Xt2 = _softmax_quench(Xt.copy(), steps=500, T0=0.0008, T1=0.00015, lr=0.02, t_deadline=explore_deadline)
            v2 = _max_cos(Xt2)
            if v2 < v:
                Xt, v = Xt2, v2
        if v < bestVal:
            bestVal = v
            bestX = Xt.copy()
            curX = Xt.copy()
            curVal = v
            push_archive(v, Xt)
        else:
            if v < bestVal + 0.005:
                if len(archive) < 6 or v < archive[-1][0]:
                    push_archive(v, Xt)
            if v < curVal or rng.random() < np.exp(-(v - curVal) / bh_T):
                curX = Xt.copy()
                curVal = v
        if it % 30 == 0:
            curX = bestX.copy()
            curVal = bestVal

    # final polish of elites
    archive_sorted = sorted(archive, key=lambda t: t[0])[:4]
    remain = TIME_LIMIT - (time.time() - t0)
    if remain > 14 and len(archive_sorted) > 0:
        per = min(3.0, (remain - 11.0) / max(1, len(archive_sorted)))
        for (av, ax) in list(archive_sorted):
            if time.time() - t0 > TIME_LIMIT - 12:
                break
            y = _softmax_quench(ax.copy(), steps=1200, T0=0.001, T1=0.00012, lr=0.02, t_deadline=t0 + TIME_LIMIT - 11)
            vy = _max_cos(y)
            cand = y if vy < av else ax
            # extra jam squeeze at tight target
            yj = _jam_quench(cand.copy(), bestVal - 0.0005, steps=900, lr=0.04, mu=0.9, bg=0.005, Tbg=0.005, t_deadline=t0 + TIME_LIMIT - 11)
            vj = _max_cos(yj)
            if vj < vy:
                cand, vy = yj, vj
            y2 = _lp_polish(cand, seconds=max(0.8, per), deadline=t0 + TIME_LIMIT - 11)
            v2 = _max_cos(y2)
            if v2 < bestVal:
                bestVal = v2
                bestX = y2.copy()
            elif vy < bestVal:
                bestVal = vy
                bestX = y.copy()
    if time.time() - t0 < TIME_LIMIT - 2:
        y = _softmax_quench(bestX.copy(), steps=2000, T0=0.001, T1=0.0001, lr=0.018, t_deadline=t0 + TIME_LIMIT - 7)
        vy = _max_cos(y)
        if vy < bestVal:
            bestVal = vy
            bestX = y.copy()
        rem2 = TIME_LIMIT - (time.time() - t0)
        if rem2 > 2:
            y2 = _lp_polish(bestX, seconds=rem2 - 0.5, deadline=t0 + TIME_LIMIT - 0.3)
            v2 = _max_cos(y2)
            if v2 < bestVal:
                bestX = y2.copy()
                bestVal = v2
    bestX = _norm_rows(bestX)
    return bestX
