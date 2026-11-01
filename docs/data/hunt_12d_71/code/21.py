import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _relax_softmax(X, steps, T0, T1, lr, mu=0.9, t_deadline=None):
    n, d = X.shape
    V = np.zeros_like(X)
    lt0 = np.log(T0); lt1 = np.log(T1)
    Gtmp = X @ X.T
    np.fill_diagonal(Gtmp, -2.0)
    best_m = float(Gtmp.max())
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
    else:
        return best_X

def _single_refine(s, R, steps=250, T=0.004, lr=0.12, mu=0.8):
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

def _find_hole(R, rng, n_sample=10000, n_cand=3):
    n, d = R.shape
    Z = _norm_rows(rng.standard_normal(size=(n_sample, d)))
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None
    best_v = 1e100
    for k in idx:
        s0 = Z[k]
        s1 = _single_refine(s0, R, steps=220, T=0.005, lr=0.12)
        s1 = _single_refine(s1, R, steps=220, T=0.0015, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    return best_s / (np.linalg.norm(best_s) + 1e-12)

def _lp_polish(x, seconds, deadline=None):
    x = _norm_rows(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1.2e-3
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
        # cap huge systems
        if p > 1500:
            # keep tightest 1500
            order = np.argsort(-cmat[sel])[:1500]
            I, J = I[order], J[order]
            csel = cmat[sel][order]
            p = 1500
        else:
            csel = cmat[sel]
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
    rng = np.random.default_rng(0)
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
        try:
            xb = _norm_rows(np.asarray(xb, dtype=np.float64))
            if xb.shape == (71, 12):
                cands.append(xb)
        except Exception:
            pass
    if xr is not None:
        try:
            xr = _norm_rows(np.asarray(xr, dtype=np.float64))
            if xr.shape == (71, 12):
                cands.append(xr)
        except Exception:
            pass
    if len(cands) == 0:
        cands.append(_norm_rows(rng.standard_normal(size=(71, 12))))
    bestX = min(cands, key=lambda z: _max_cos(z)).copy()
    bestVal = _max_cos(bestX)
    curX = bestX.copy()
    curVal = bestVal
    archive = [(bestVal, bestX.copy())]

    def push_archive(v, X):
        archive.append((v, X.copy()))
        archive.sort(key=lambda t: t[0])
        del archive[5:]

    explore_deadline = t0 + 124.0
    bh_T = 0.0015
    it = 0
    while time.time() < explore_deadline:
        it += 1
        if time.time() > explore_deadline:
            break
        # pick base
        r = rng.random()
        if r < 0.6:
            base = bestX
        elif r < 0.8 and len(archive) > 1:
            k = int(rng.integers(1, len(archive)))
            base = archive[k][1]
        else:
            base = curX
        mv = rng.random()
        if mv < 0.22:
            # hole reinsertion: remove 1-2 hubs
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            thr = bestVal - 0.008
            cnt = (G > thr).sum(axis=1)
            order = np.argsort(-cnt)
            topk = order[:8]
            nrem = 2 if (it % 3 == 0) else 1
            victims = []
            victims.append(int(topk[0]))
            if nrem == 2:
                victims.append(int(topk[1]))
                if rng.random() < 0.3:
                    victims = [int(rng.choice(topk)), int(rng.choice(topk))]
            victims = sorted(set(victims))
            mask = np.ones(71, dtype=bool)
            mask[victims] = False
            Xt_base = base[mask]
            for _ in range(len(victims)):
                s_new = _find_hole(Xt_base, rng, n_sample=10000, n_cand=3)
                Xt_base = np.vstack([Xt_base, s_new[None, :]])
            Xt = Xt_base
            T0 = float(rng.choice([0.008, 0.012, 0.006, 0.015]))
            T1 = float(rng.choice([0.0008, 0.0004, 0.0012, 0.0006]))
            lr = float(rng.choice([0.04, 0.06, 0.03, 0.05]))
            steps = int(rng.choice([1800, 2500, 3200, 2000]))
            Xt = _relax_softmax(Xt, steps=steps, T0=T0, T1=T1, lr=lr, mu=0.9, t_deadline=explore_deadline)
        elif mv < 0.55:
            # hub-targeted shake
            Xt = base.copy()
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            thr = bestVal - 0.012
            cnt = (G > thr).sum(axis=1)
            hubs = np.argsort(-cnt)[:8]
            k = int(rng.choice([2, 3, 3, 4]))
            k = min(k, 8)
            idx = hubs[rng.choice(8, size=k, replace=False)]
            # occasionally add a random index
            if rng.random() < 0.3:
                idx[0] = int(rng.integers(71))
            ls = rng.uniform(np.log(0.02), np.log(0.40))
            sigma = float(np.exp(ls))
            Xt[idx] = Xt[idx] + rng.standard_normal(size=(k, 12)) * sigma
            Xt = _norm_rows(Xt)
            T0 = float(rng.choice([0.01, 0.007, 0.015, 0.012]))
            T1 = float(rng.choice([0.0005, 0.001, 0.0003, 0.0007]))
            lr = float(rng.choice([0.05, 0.03, 0.07]))
            steps = int(rng.choice([1500, 2200, 3000]))
            Xt = _relax_softmax(Xt, steps=steps, T0=T0, T1=T1, lr=lr, mu=0.9, t_deadline=explore_deadline)
        else:
            Xt = base.copy()
            k = int(rng.choice([1, 2, 2, 3, 4]))
            idx = rng.choice(71, size=k, replace=False)
            ls = rng.uniform(np.log(0.015), np.log(0.45))
            sigma = float(np.exp(ls))
            if rng.random() < 0.07:
                Xt[idx] = _norm_rows(rng.standard_normal(size=(k, 12)))
            else:
                Xt[idx] = Xt[idx] + rng.standard_normal(size=(k, 12)) * sigma
                Xt = _norm_rows(Xt)
            T0 = float(rng.choice([0.01, 0.007, 0.015]))
            T1 = float(rng.choice([0.0005, 0.001, 0.0003]))
            lr = float(rng.choice([0.05, 0.03, 0.07]))
            steps = int(rng.choice([1500, 2200, 3000]))
            Xt = _relax_softmax(Xt, steps=steps, T0=T0, T1=T1, lr=lr, mu=0.9, t_deadline=explore_deadline)
        v = _max_cos(Xt)
        if v < bestVal + 0.003 and time.time() < explore_deadline - 1:
            Xt2 = _relax_softmax(Xt.copy(), steps=600, T0=0.0008, T1=0.0002, lr=0.02, mu=0.9, t_deadline=explore_deadline)
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
            if v < bestVal + 0.004:
                if len(archive) < 5 or v < archive[-1][0]:
                    push_archive(v, Xt)
            if v < curVal or rng.random() < np.exp(-(v - curVal) / bh_T):
                curX = Xt.copy()
                curVal = v
        if it % 25 == 0:
            curX = bestX.copy()
            curVal = bestVal

    archive_sorted = sorted(archive, key=lambda t: t[0])[:4]
    remain = TIME_LIMIT - (time.time() - t0)
    if remain > 15 and len(archive_sorted) > 1:
        per = min(3.5, (remain - 12.0) / max(1, len(archive_sorted)))
        for (av, ax) in list(archive_sorted):
            if time.time() - t0 > TIME_LIMIT - 13:
                break
            y = _relax_softmax(ax.copy(), steps=1300, T0=0.001, T1=0.00015, lr=0.02, mu=0.9, t_deadline=t0 + TIME_LIMIT - 12)
            vy = _max_cos(y)
            cand = y if vy < av else ax
            y2 = _lp_polish(cand, seconds=max(0.8, per), deadline=t0 + TIME_LIMIT - 12)
            v2 = _max_cos(y2)
            if v2 < bestVal:
                bestVal = v2
                bestX = y2.copy()
            elif vy < bestVal:
                bestVal = vy
                bestX = y.copy()
    if time.time() - t0 < TIME_LIMIT - 2:
        y = _relax_softmax(bestX.copy(), steps=2200, T0=0.0012, T1=0.00012, lr=0.018, mu=0.9, t_deadline=t0 + TIME_LIMIT - 8)
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
