import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _stats(X, win=2e-3):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = G > (m - win)
    cnt = int(np.count_nonzero(C) // 2)
    deg = C.sum(axis=1)
    md = int(deg.max())
    return m, cnt, md

def _fp_sorted(X):
    iu = np.triu_indices(X.shape[0], 1)
    v = (X @ X.T)[iu]
    v = np.sort(v)[::-1]
    return v

def _softmax_descend(y, steps, T0, T1, lr0, lr1, deadline):
    n, d = y.shape
    for s in range(steps):
        if (s % 50 == 0) and (time.time() > deadline):
            break
        frac = s / max(steps - 1, 1)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / max(T, 1e-6))
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-20
        Avg = (E @ y) / row
        dot = (Avg * y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * y
        y = y - lr * Gt
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y /= yn
    return y

def _power_descend(y, steps, p0, p1, lr0, lr1, deadline):
    n, d = y.shape
    for s in range(steps):
        if (s % 50 == 0) and (time.time() > deadline):
            break
        frac = s / max(steps - 1, 1)
        p = p0 + (p1 - p0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        # power weights, emphasize close pairs
        W = np.clip(1.0 - G + 0.05, 1e-3, 3.0) ** (-p)
        np.fill_diagonal(W, 0.0)
        # downweight very far to avoid drift: keep top influence
        S = (W * G).sum(axis=1)
        A = W @ y
        F = A - S[:, None] * y
        cnt = W.sum(axis=1, keepdims=True) + 1.0
        F /= cnt
        # clip
        nrm = np.linalg.norm(F, axis=1, keepdims=True) + 1e-20
        F = F / np.maximum(nrm, 1.0)
        y = y - lr * F
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y /= yn
    return y

def _lp_polish(x, seconds, p_cap=1200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 8e-4
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while (time.time() - start < seconds) and (radius > 1e-14):
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
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
        p = int(sel.sum())
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([(I[:, None] * d + np.arange(d)).reshape(p, d),
                               (J[:, None] * d + np.arange(d)).reshape(p, d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() - start >= seconds:
            break
    return best_x

def _d13_roots(d=13):
    rows = []
    for i in range(d):
        for j in range(i + 1, d):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(d)
                    v[i] = si; v[j] = sj
                    rows.append(v)
    A = np.stack(rows, axis=0)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    return A

def _gen_seed(kind, rng, n, d, Droots):
    if kind == "randn":
        Y = rng.normal(size=(n, d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    elif kind == "d13":
        idx = rng.choice(Droots.shape[0], size=n, replace=False)
        Y = Droots[idx].copy()
        Y += 1e-3 * rng.normal(size=Y.shape)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    elif kind == "cube":
        Y = rng.integers(0, 2, size=(n, d)).astype(float) * 2.0 - 1.0
        # random sign flips already; add jitter for diversity
        Y += 0.05 * rng.normal(size=Y.shape)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    elif kind == "ternary":
        # base pattern with levels 1/2, optimize ratio r
        supp = rng.integers(2, 6, size=n)
        base_sign = np.zeros((n, d))
        base_lvl = np.zeros((n, d), dtype=int)
        for i in range(n):
            k = int(supp[i])
            cols = rng.choice(d, size=k, replace=False)
            for c in cols:
                s = 1.0 if rng.random() < 0.5 else -1.0
                lvl = 1 if rng.random() < 0.5 else 2
                base_sign[i, c] = s
                base_lvl[i, c] = lvl
        best_r = 0.5
        best_v = 1e9
        best_Y = None
        for r in (0.25, 0.45, 0.65, 0.85):
            Y = np.zeros((n, d))
            Y[base_lvl == 1] = base_sign[base_lvl == 1] * r
            Y[base_lvl == 2] = base_sign[base_lvl == 2] * 1.0
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            v = _max_cos(Y)
            if v < best_v:
                best_v = v
                best_r = r
                best_Y = Y.copy()
        return best_Y
    elif kind == "orth":
        A = rng.normal(size=(d, d))
        Q, _ = np.linalg.qr(A)
        Y = rng.normal(size=(n, d)) @ Q.T
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    else:
        Y = rng.normal(size=(n, d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y

def solve():
    t0 = time.time()
    EXP_END = t0 + 110.0
    LP_END = t0 + 155.0
    HARD_END = t0 + 168.0
    n, d = 123, 13
    np.random.seed(12345)
    rng = np.random.default_rng(12345)
    # baseline best (verified)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val:
                        best_val = float(v)
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    try:
        Droots = _d13_roots(d)
    except Exception:
        Droots = rng.normal(size=(312, d))
        Droots /= np.linalg.norm(Droots, axis=1, keepdims=True)
    kinds = ["randn", "d13", "ternary", "cube", "orth", "randn", "ternary", "d13"]
    archive = []  # list of (val, cnt, maxdeg, fp, X)
    fps = []
    it = 0
    # exploration loop
    while time.time() < EXP_END:
        if time.time() > HARD_END - 20:
            break
        kind = kinds[it % len(kinds)]
        try:
            Y = _gen_seed(kind, rng, n, d, Droots)
        except Exception:
            it += 1
            continue
        # continuation schedule by it
        try:
            dl = min(EXP_END - 0.5, time.time() + 8.0)
            mode = it % 4
            if mode == 0:
                Y = _softmax_descend(Y, steps=220, T0=0.03, T1=0.004, lr0=0.5, lr1=0.05, deadline=dl)
            elif mode == 1:
                Y = _softmax_descend(Y, steps=260, T0=0.015, T1=0.002, lr0=0.35, lr1=0.03, deadline=dl)
            elif mode == 2:
                Y = _power_descend(Y, steps=220, p0=2.0, p1=9.0, lr0=0.6, lr1=0.05, deadline=dl)
            else:
                Y = _power_descend(Y, steps=260, p0=3.0, p1=12.0, lr0=0.4, lr1=0.03, deadline=dl)
                Y = _softmax_descend(Y, steps=80, T0=0.008, T1=0.002, lr0=0.15, lr1=0.02, deadline=dl)
        except Exception:
            it += 1
            continue
        if time.time() > EXP_END:
            # still evaluate quickly then break
            pass
        try:
            v, cnt, md = _stats(Y)
            fp = _fp_sorted(Y)
        except Exception:
            it += 1
            continue
        # diversity check: distinct if sorted-gram distance > 0.3 to all with close val
        distinct = True
        for (av, ac, amd, afp, _) in archive:
            if abs(av - v) < 0.005:
                try:
                    dist = float(np.linalg.norm(fp - afp))
                except Exception:
                    dist = 1e9
                if dist < 0.3:
                    distinct = False
                    break
        if distinct:
            archive.append((float(v), int(cnt), int(md), fp, Y.copy()))
            # cap archive to 40, drop worst
            if len(archive) > 40:
                archive.sort(key=lambda e: e[0])
                archive = archive[:40]
        it += 1
        if it >= 60:
            break
        if time.time() >= EXP_END:
            break
    # select few structurally distinct best for LP
    archive.sort(key=lambda e: e[0])
    selected = []
    for e in archive:
        v, cnt, md, fp, X = e
        ok = True
        for _, _, _, sfp, _ in selected:
            try:
                if float(np.linalg.norm(fp - sfp)) < 0.8:
                    ok = False
                    break
            except Exception:
                pass
        if ok:
            selected.append(e)
        if len(selected) >= 3:
            break
    # LP polishing, strictly time-capped
    for si, (v, cnt, md, fp, X) in enumerate(selected):
        now = time.time()
        if now > LP_END - 1.0 or now > HARD_END - 8.0:
            break
        # only polish if promising: within 0.02 of best or best archive
        if v > best_val + 0.02 and si > 0:
            continue
        secs = min(3.0, LP_END - now, HARD_END - 5.0 - now)
        if secs < 0.5:
            break
        try:
            P = _lp_polish(X, seconds=float(secs), p_cap=1200)
            pv = _max_cos(P)
            if pv < best_val:
                best_val = float(pv)
                best = P.copy()
        except Exception:
            continue
        if time.time() > LP_END or time.time() > HARD_END - 5.0:
            break
    # unconditional return of best verified array
    try:
        best = np.asarray(best, dtype=float)
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        # replace any non-finite rows with random
        nr = np.linalg.norm(best, axis=1)
        bad = ~(np.isfinite(nr)) | (nr < 1e-12)
        if np.any(bad):
            R = rng.normal(size=(int(bad.sum()), d))
            R /= np.linalg.norm(R, axis=1, keepdims=True)
            best[bad] = R
            best = best / np.linalg.norm(best, axis=1, keepdims=True)
    except Exception:
        rng2 = np.random.default_rng(999)
        best = rng2.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
