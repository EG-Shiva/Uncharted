import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth_opt(X0, beta, maxiter, maxcor=10):
    n, d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.empty_like(D)
        np.exp(D, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if S <= 0 or not np.isfinite(S):
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-12,
                                "maxfun": maxiter * 5, "maxcor": maxcor})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return Y

def _lp_polish(x, seconds, t0):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < seconds and radius > 1e-14:
        cmat = x @ x.T
        iu_c = cmat[iu]
        m = float(iu_c.max())
        sel = iu_c > m - max(20.0 * radius, 5e-7)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 4000:
            thr = np.sort(iu_c[sel])[-4000]
            sel = iu_c >= thr
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-iu_c[sel], A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
    return x

def _refine_single(y, Z, beta=4000.0, iters=60, step0=0.06):
    y = np.asarray(y, dtype=np.float64).copy()
    yn = np.linalg.norm(y)
    if yn < 1e-12:
        y = Z[0] * 0.0
        y[0] = 1.0
    else:
        y = y / yn
    dots = Z @ y
    best_m = float(dots.max())
    best_y = y.copy()
    cur = y.copy()
    for it in range(iters):
        dots = Z @ cur
        M = float(dots.max())
        D = beta * (dots - M)
        np.clip(D, -30.0, 0.0, out=D)
        w = np.exp(D)
        s = float(w.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ Z
        gt = g - float(g @ cur) * cur
        n = float(np.linalg.norm(gt))
        if n < 1e-14:
            break
        gt = gt / n
        step = step0 / (1.0 + it * 0.08)
        nxt = cur - step * gt
        nn = np.linalg.norm(nxt)
        if nn < 1e-12:
            continue
        nxt = nxt / nn
        dn = Z @ nxt
        mn = float(dn.max())
        if mn < best_m:
            best_m = mn
            best_y = nxt.copy()
        cur = nxt
    return best_y

def _block_max(Yb, Xf):
    if Yb.shape[0] == 0:
        return -2.0
    Gbf = Yb @ Xf.T
    m1 = float(Gbf.max()) if Gbf.size else -2.0
    b = Yb.shape[0]
    if b > 1:
        Gbb = Yb @ Yb.T
        np.fill_diagonal(Gbb, -2.0)
        m2 = float(Gbb.max())
    else:
        m2 = -2.0
    return m1 if m1 > m2 else m2

def _joint_block_opt(Yb, Xf, beta=4000.0, iters=30, step0=0.03):
    Yb = np.asarray(Yb, dtype=np.float64).copy()
    Yb = Yb / np.maximum(np.linalg.norm(Yb, axis=1, keepdims=True), 1e-12)
    b = Yb.shape[0]
    if b == 0:
        return Yb
    best = Yb.copy()
    best_v = _block_max(Yb, Xf)
    Y = Yb.copy()
    for it in range(iters):
        Gbf = Y @ Xf.T
        Gbb = Y @ Y.T
        if b > 1:
            np.fill_diagonal(Gbb, -2.0)
        M = float(Gbf.max()) if Gbf.size else -2.0
        if b > 1:
            m2 = float(Gbb.max())
            if m2 > M:
                M = m2
        Abf = beta * (Gbf - M)
        np.clip(Abf, -30.0, 0.0, out=Abf)
        Wbf = np.exp(Abf)
        if b > 1:
            Abb = beta * (Gbb - M)
            np.clip(Abb, -30.0, 0.0, out=Abb)
            Wbb = np.exp(Abb)
            np.fill_diagonal(Wbb, 0.0)
        else:
            Wbb = np.zeros((1, 1))
        S = float(Wbf.sum() + Wbb.sum())
        if S <= 0 or not np.isfinite(S):
            break
        Wbf /= S
        Wbb /= S
        Grad = Wbf @ Xf
        if b > 1:
            Grad = Grad + 2.0 * (Wbb @ Y)
        dot = np.sum(Grad * Y, axis=1, keepdims=True)
        Gt = Grad - dot * Y
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt = Gt / nr
        step = step0 / (1.0 + it * 0.10)
        Y = Y - step * Gt
        Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        v = _block_max(Y, Xf)
        if v < best_v:
            best_v = v
            best = Y.copy()
    return best

def _select_remove(X, k, max_val, wa, wb):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    thr = max_val - 2e-5
    iu0 = np.triu_indices(n, 1)
    vals = G[iu0]
    mask = vals > thr
    adj = [[] for _ in range(n)]
    I = iu0[0][mask]
    J = iu0[1][mask]
    for a, b in zip(I.tolist(), J.tolist()):
        adj[a].append(b)
        adj[b].append(a)
    seen = {int(wa)}
    queue = [int(wa)]
    idx = 0
    while idx < len(queue):
        u = queue[idx]
        idx += 1
        for w in adj[u]:
            if w not in seen:
                seen.add(w)
                queue.append(w)
    # ensure wb included
    if int(wb) not in seen:
        seen.add(int(wb))
        queue.append(int(wb))
    rowmax = G.max(axis=1)
    comp = list(seen)
    comp_sorted = sorted(comp, key=lambda v: -float(rowmax[v]))
    ordered = []
    for v in (int(wa), int(wb)):
        if v not in ordered:
            ordered.append(v)
    for v in comp_sorted:
        if v not in ordered:
            ordered.append(v)
    if len(ordered) < k:
        order_all = np.argsort(-rowmax)
        for v in order_all.tolist():
            v = int(v)
            if v not in ordered:
                ordered.append(v)
            if len(ordered) >= k:
                break
    return ordered[:k]

def solve():
    t_start = time.time()
    TIME_BUDGET = 168.0
    def remaining():
        return TIME_BUDGET - (time.time() - t_start)
    n, d = 250, 6
    rng0 = np.random.default_rng(7)
    cands = []
    try:
        z = np.asarray(np.load("best.npy"), dtype=np.float64)
        if z.shape == (n, d):
            z = z / np.linalg.norm(z, axis=1, keepdims=True)
            cands.append(z)
    except Exception:
        pass
    try:
        z = np.asarray(np.load("record.npy"), dtype=np.float64)
        if z.shape == (n, d):
            z = z / np.linalg.norm(z, axis=1, keepdims=True)
            cands.append(z)
    except Exception:
        pass
    if not cands:
        z = rng0.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        cands.append(z)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)
    # sizes to try: 15, 25, 40 as required, cycled
    sizes = [25, 15, 40, 25, 40, 15, 30, 20, 35]
    attempt = 0
    # strict time box for regrowth phase ~140s from start
    while (time.time() - t_start) < 140.0 and remaining() > 28.0 and attempt < 12:
        k = int(sizes[attempt % len(sizes)])
        seed = 1000 + attempt * 131
        rng = np.random.default_rng(seed)
        attempt += 1
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            gv = G[iu]
            kk = int(np.argmax(gv))
            wa, wb = int(iu[0][kk]), int(iu[1][kk])
            cur_max = float(gv[kk])
            chosen = _select_remove(best, k, cur_max, wa, wb)
            # safety: must include bottleneck
            if wa not in chosen:
                chosen[0] = wa
            if wb not in chosen:
                chosen[1 % len(chosen)] = wb
            mask = np.ones(n, dtype=bool)
            mask[np.array(chosen, dtype=int)] = False
            F = best[mask].copy()
            # need enough time for this attempt
            need = 4.0 + k * 0.35
            if remaining() < need + 20.0:
                break
            deadline = time.time() + min(need + 6.0, remaining() - 20.0)
            # joint deep-hole regrowth
            inserted = np.zeros((0, d))
            ok = True
            for i in range(k):
                if time.time() > deadline:
                    ok = False
                    break
                if inserted.shape[0] == 0:
                    Z = F
                else:
                    Z = np.vstack([F, inserted])
                Nc = 8192
                cand = rng.standard_normal((Nc, d))
                cand /= np.linalg.norm(cand, axis=1, keepdims=True)
                dots = cand @ Z.T
                mx = dots.max(axis=1)
                bi = int(np.argmin(mx))
                y = cand[bi].copy()
                y = _refine_single(y, Z, beta=4000.0, iters=60, step0=0.06)
                if inserted.shape[0] == 0:
                    inserted = y.reshape(1, d)
                else:
                    inserted = np.vstack([inserted, y.reshape(1, d)])
                if (inserted.shape[0] % 5 == 0) or (inserted.shape[0] == k):
                    if time.time() > deadline:
                        break
                    inserted = _joint_block_opt(inserted, F, beta=4000.0, iters=30, step0=0.03)
            if not ok or inserted.shape[0] != k:
                continue
            Xnew = np.vstack([F, inserted])
            Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
            # short collective polish (smooth)
            if time.time() > deadline:
                continue
            try:
                beta_c = 8000.0 if (attempt % 2 == 0) else 12000.0
                Xpol = _smooth_opt(Xnew, beta_c, 70, maxcor=10)
            except Exception:
                Xpol = Xnew
            v = _max_cos(Xpol)
            if v < best_val - 1e-12:
                best = Xpol.copy()
                best_val = v
                # quick LP squeeze if good
                if remaining() > 24.0:
                    try:
                        sq = _lp_polish(best, min(2.0, remaining() - 22.0), time.time())
                        vs = _max_cos(sq)
                        if vs < best_val:
                            best = sq
                            best_val = vs
                    except Exception:
                        pass
            elif v < best_val + 5e-6 and remaining() > 70.0 and (attempt % 3 == 0):
                # sideways drift to new basin (allow small uphill)
                best = Xpol.copy()
                best_val = v
        except Exception:
            continue
    if remaining() > 6.0:
        try:
            polished = _lp_polish(best, remaining() - 1.5, time.time())
            vp = _max_cos(polished)
            if vp < best_val:
                best = polished
                best_val = vp
        except Exception:
            pass
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    best = np.where(np.isfinite(best), best, 0.0)
    best = best / np.maximum(np.linalg.norm(best, axis=1, keepdims=True), 1e-12)
    return best
