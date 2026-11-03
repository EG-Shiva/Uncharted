import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, p_cap=1200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1.2e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    itc = 0
    while time.time() - start < seconds and radius > 1e-14 and itc < 40:
        itc += 1
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = c_all.max()
        window = max(25 * radius, 2e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
        I = iu0[sel]; J = iu1[sel]
        csel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)), (J[:, None] * d + np.arange(d)), np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            tl = max(0.3, seconds - (time.time() - start))
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            best = v; best_x = y.copy(); x = y
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return best_x

def _stress_weights(X, M, eps=1e-6):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    sel = c > M - eps
    I = iu0[sel]; J = iu1[sel]
    p = len(I)
    if p == 0:
        deg = np.zeros(n)
        return np.ones(n), np.zeros(n), deg, I, J, np.ones(0)
    if p > 2500:
        idx = np.argpartition(-c[sel], 2500 - 1)[:2500]
        I = I[idx]; J = J[idx]
        p = 2500
    Gij = (X[I] * X[J]).sum(axis=1)
    w = np.ones(p)
    mu = 5e-3
    eta = 0.08
    R = np.zeros((n, d))
    for _ in range(300):
        Vij = X[J] - Gij[:, None] * X[I]
        Vji = X[I] - Gij[:, None] * X[J]
        R.fill(0.0)
        np.add.at(R, I, w[:, None] * Vij)
        np.add.at(R, J, w[:, None] * Vji)
        grad = np.sum(Vij * R[I], axis=1) + np.sum(Vji * R[J], axis=1) + mu * (w - 1.0)
        w = w - eta * grad
        np.maximum(w, 0.0, out=w)
    # final residuals and per-point stress
    Vij = X[J] - Gij[:, None] * X[I]
    Vji = X[I] - Gij[:, None] * X[J]
    R.fill(0.0)
    np.add.at(R, I, w[:, None] * Vij)
    np.add.at(R, J, w[:, None] * Vji)
    S = np.zeros(n)
    np.add.at(S, I, w)
    np.add.at(S, J, w)
    Rn = np.linalg.norm(R, axis=1)
    # contact degree at 1e-9 for rattler detection
    deg = (G > M - 1e-9).sum(axis=1)
    return S, Rn, deg, I, J, w

def _repack_patch(Xb, free_idx, seconds, rng):
    t0 = time.time()
    n, d = Xb.shape
    free_idx = np.asarray(free_idx, dtype=int)
    k = len(free_idx)
    frozen_mask = np.ones(n, bool)
    frozen_mask[free_idx] = False
    Xf = Xb[frozen_mask].copy()
    nf = Xf.shape[0]
    Y = Xb[free_idx].copy()
    # delete: hole-search reinsertion against frozen + sequential placement
    placed = []
    for a in range(k):
        cand = rng.normal(size=(600, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        if nf > 0:
            mf = cand @ Xf.T
            m1 = mf.max(axis=1)
        else:
            m1 = np.full(600, -2.0)
        if placed:
            P = np.stack(placed, axis=0)
            mp = cand @ P.T
            m2 = mp.max(axis=1)
            m = np.maximum(m1, m2)
        else:
            m = m1
        # small jitter among best 8
        order = np.argpartition(m, 8)[:8]
        pick = order[int(rng.integers(0, 8))]
        v = cand[pick] + 0.002 * rng.normal(size=(d,))
        v /= np.linalg.norm(v)
        placed.append(v)
        Y[a] = v
        if time.time() - t0 >= max(0.5, seconds * 0.15):
            # fill rest randomly if time tight
            for b in range(a + 1, k):
                vv = rng.normal(size=(d,))
                vv /= np.linalg.norm(vv)
                Y[b] = vv
            break
    def full_max(Yc):
        Gff = Yc @ Yc.T
        np.fill_diagonal(Gff, -2.0)
        mff = Gff.max() if k > 1 else -2.0
        Gfx = Yc @ Xf.T
        mfx = Gfx.max() if nf > 0 else -2.0
        return max(mff, mfx)
    bestY = Y.copy()
    bestv = full_max(bestY)
    T0, T1 = 0.016, 0.0022
    lr0, lr1 = 0.45, 0.05
    step = 0
    while True:
        el = time.time() - t0
        if el >= seconds:
            break
        frac = el / max(seconds, 1e-9)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        # batch of inner steps without time check
        inner = 25
        for _ in range(inner):
            frac2 = min(1.0, (time.time() - t0) / max(seconds, 1e-9))
            T2 = T0 + (T1 - T0) * frac2
            lr2 = lr0 + (lr1 - lr0) * frac2
            Gff = Y @ Y.T
            np.fill_diagonal(Gff, -10.0)
            Gfx = Y @ Xf.T if nf > 0 else np.zeros((k, 0))
            m = Gff.max()
            if nf > 0:
                m = max(m, float(Gfx.max()))
            Eff = np.exp((Gff - m) / T2)
            np.fill_diagonal(Eff, 0.0)
            if nf > 0:
                Efx = np.exp((Gfx - m) / T2)
                row = Eff.sum(axis=1, keepdims=True) + Efx.sum(axis=1, keepdims=True) + 1e-20
                Avg = (Eff @ Y + Efx @ Xf) / row
            else:
                row = Eff.sum(axis=1, keepdims=True) + 1e-20
                Avg = (Eff @ Y) / row
            dot = (Avg * Y).sum(axis=1, keepdims=True)
            Gt = Avg - dot * Y
            Y = Y - lr2 * Gt
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        step += 1
        if step % 2 == 0:
            v = full_max(Y)
            if v < bestv:
                bestv = v
                bestY = Y.copy()
        if time.time() - t0 >= seconds:
            break
    Xt = Xb.copy()
    Xt[free_idx] = bestY
    Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
    return Xt

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(7)
    n, d = 123, 13
    X0 = None; bv = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, float)
            if a.shape == (n, d):
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                v = _max_cos(a)
                if v < bv:
                    bv = v; X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.normal(size=(n, d)); X0 /= np.linalg.norm(X0, axis=1, keepdims=True); bv = _max_cos(X0)
    best = X0.copy(); best_val = float(bv)
    # quick initial tighten (capped)
    try:
        if time.time() - t0 < LIM - 70:
            cand = _lp_polish(best, 2.0)
            v = _max_cos(cand)
            if v < best_val:
                best = cand; best_val = v
    except Exception:
        pass
    if time.time() - t0 >= LIM - 60:
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        return best
    # stress analysis on best
    try:
        M = _max_cos(best)
        S, Rn, deg, I, J, w = _stress_weights(best, M, eps=1e-6)
    except Exception:
        S = np.zeros(n); Rn = np.zeros(n); deg = np.zeros(n); M = best_val
    Gfull = best @ best.T
    np.fill_diagonal(Gfull, -2.0)
    rattler = int(np.argmin(deg))
    # nearest neighbours of rattler
    gr = (best @ best[rattler]).copy(); gr[rattler] = -2.0
    nn_ratt = np.argsort(-gr)
    # hub order: lowest total stress first (excluding rattler)
    hub_order = np.argsort(S)
    hub_order = [h for h in hub_order if h != rattler]
    # also consider high-residual as alternative hub
    res_order = np.argsort(-Rn)
    res_order = [h for h in res_order if h != rattler]
    patches = []
    # build at most 3 patches, each 10-18 pts: rattler + 5 nearest + hub neighbourhood
    hub_cands = []
    if len(hub_order) > 0:
        hub_cands.append(hub_order[0])
    if len(hub_order) > 1:
        hub_cands.append(hub_order[1])
    if len(res_order) > 0 and res_order[0] not in hub_cands:
        hub_cands.append(res_order[0])
    else:
        if len(hub_order) > 2:
            hub_cands.append(hub_order[2])
    hub_cands = hub_cands[:3]
    for hc in hub_cands:
        s = set()
        s.add(rattler)
        for q in nn_ratt[:5]:
            s.add(int(q))
        s.add(int(hc))
        gh = (best @ best[int(hc)]).copy(); gh[int(hc)] = -2.0
        nb = np.argsort(-gh)[:8]
        for q in nb:
            if len(s) >= 16:
                break
            s.add(int(q))
        # pad/trim to 10-18
        if len(s) < 10:
            for q in hub_order:
                if q not in s:
                    s.add(int(q))
                if len(s) >= 12:
                    break
        patches.append(sorted(list(s))[:18])
    # at most three patches
    patches = patches[:3]
    for pi, patch in enumerate(patches):
        remain = LIM - (time.time() - t0)
        if remain < 8:
            break
        repack_sec = min(14.0, max(2.0, remain - 12.0))
        # strictly capped repack: use at most 14, here use min
        repack_sec = min(repack_sec, 14.0)
        if repack_sec < 1.5:
            break
        try:
            trial = _repack_patch(best, patch, repack_sec, rng)
        except Exception:
            continue
        if time.time() - t0 >= LIM - 7:
            v = _max_cos(trial)
            if v < best_val:
                best = trial; best_val = v
            break
        # LP polish whole config at most 5 s
        remain2 = LIM - (time.time() - t0)
        lp_sec = min(5.0, max(1.0, remain2 - 4.0))
        try:
            trial2 = _lp_polish(trial, lp_sec)
        except Exception:
            trial2 = trial
        v = _max_cos(trial2)
        # accept only if improves
        if v < best_val - 1e-14:
            best = trial2; best_val = v
            # refresh stress for remaining patches? keep same patches but update M
            try:
                M = best_val
                S2, Rn2, _, _, _, _ = _stress_weights(best, M, eps=1e-6)
                S, Rn = S2, Rn2
            except Exception:
                pass
        if time.time() - t0 >= LIM - 8:
            break
    # final short polish with remaining time (capped)
    try:
        remain = LIM - (time.time() - t0) - 0.8
        if remain > 1.0:
            fsec = min(remain, 8.0)
            pol = _lp_polish(best, fsec)
            pv = _max_cos(pol)
            if pv < best_val:
                best = pol; best_val = pv
    except Exception:
        pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
