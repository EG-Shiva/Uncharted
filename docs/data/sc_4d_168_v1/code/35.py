import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _nm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _refine(x_init, steps, beta0=30.0, beta1=500.0, lr0=0.016, lr1=0.0008, mu=0.88, deadline=None):
    x = x_init.copy()
    vel = np.zeros_like(x)
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    for s in range(steps):
        if deadline is not None and (s & 63) == 0 and time.time() > deadline:
            break
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = lr0 * (1.0 - f) + lr1 * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = (E @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        sc = np.mean(np.linalg.norm(gt, axis=1)) + 1e-12
        gt = gt / sc
        vel = mu * vel + (1.0 - mu) * gt
        x = x - lr * vel
        x = _nm(x)
    return x

def _lp(x, deadline):
    x = _nm(x.copy())
    n, d = x.shape
    best = _mc(x)
    rad = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and rad > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * rad, 1e-9)
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
        Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        Aeq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            r = linprog(obj, A_ub=Aub, b_ub=-c[sel], A_eq=Aeq, b_eq=np.zeros(n),
                        bounds=[(-rad, rad)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if r.status != 0:
            rad *= 0.5
            continue
        y = _nm(x + r.x[:-1].reshape(n, d))
        v = _mc(y)
        if v < best:
            x, best, rad = y, v, min(rad * 1.5, 0.05)
        else:
            rad *= 0.5
    return x

def _loads(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = np.sum(G > (m - 1e-3), axis=1)
    exc = np.sum(np.clip(G - 0.80, 0.0, None), axis=1)
    load = cnt + 20.0 * exc
    return load, m, G

def _polish_single(s0, Xref, steps=90, beta0=50.0, beta1=300.0, lr0=0.08, lr1=0.005):
    s = s0 / (np.linalg.norm(s0) + 1e-30)
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    for it in range(steps):
        f = it / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = lr0 * (1.0 - f) + lr1 * f
        dots = Xref @ s
        cm = float(np.max(dots))
        w = np.exp(np.clip(beta * (dots - cm), -25.0, 0.0))
        w /= (np.sum(w) + 1e-30)
        g = w @ Xref
        gt = g - np.dot(g, s) * s
        gn = np.linalg.norm(gt) + 1e-12
        s = s - lr * (gt / gn)
        ns = np.linalg.norm(s)
        if ns < 1e-12:
            s = np.array([1.0, 0.0, 0.0, 0.0]); ns = 1.0
        s = s / ns
    return s

def _deepest_holes(Xref, k, rng, nsamp=20000, polish=True):
    C = rng.normal(size=(nsamp, 4))
    C = _nm(C)
    D = C @ Xref.T
    mx = D.max(axis=1)
    order = np.argsort(mx)
    # greedy diverse pick
    picked = []
    for idx in order:
        c = C[idx]
        ok = True
        for p in picked:
            if float(np.dot(c, p)) > 0.90:
                ok = False
                break
        if ok:
            picked.append(c)
        if len(picked) >= k:
            break
    while len(picked) < k:
        picked.append(C[order[len(picked) % len(order)]])
    P = np.stack(picked, axis=0)
    if polish:
        for i in range(k):
            try:
                P[i] = _polish_single(P[i], Xref, steps=80)
            except Exception:
                pass
    return P

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    SEARCH = t0 + 146.0
    rng = np.random.default_rng(1681717)
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.load(f).astype(float)
            if c.shape == (168, 4):
                c = _nm(c)
                if np.all(np.isfinite(c)):
                    cands.append(c)
        except Exception:
            pass
    if not cands:
        xb = _nm(rng.normal(size=(168, 4)))
    else:
        vs = [_mc(c) for c in cands]
        xb = cands[int(np.argmin(vs))].copy()
    bv = _mc(xb)
    x_best = xb.copy()

    def try_update(X, lp_budget=0.0):
        nonlocal bv, x_best, xb
        v = _mc(X)
        if v < bv and np.all(np.isfinite(X)):
            bv = v
            x_best = X.copy()
            xb = X.copy()
            return True
        return False

    # initial squeeze of incumbent to have tight baseline
    try:
        if time.time() < SEARCH - 5:
            Xp = _lp(xb, min(time.time() + 2.0, SEARCH, HARD - 2))
            try_update(_nm(Xp))
    except Exception:
        pass

    cyc = 0
    # alternate overshoot (even) / undershoot (odd)
    while time.time() < SEARCH and cyc < 60:
        cyc += 1
        if time.time() > SEARCH - 3:
            break
        if cyc % 2 == 0:
            # OVERSHOOT: add 1-3 points
            add_k = 1 + ((cyc // 2) % 3)  # 1,2,3 cycle
            try:
                holes = _deepest_holes(xb, add_k, rng, nsamp=18000, polish=True)
            except Exception:
                continue
            try:
                Xen = _nm(np.vstack([xb, holes]))
                # refine enlarged hard (allow worse max)
                Xen = _refine(Xen, steps=700, beta0=28.0, beta1=520.0, lr0=0.016, lr1=0.0009, deadline=SEARCH - 2)
                # delete point with largest contact load
                ld, _, _ = _loads(Xen)
                # never delete all; delete one worst
                del_idx = int(np.argmax(ld))
                mask = np.ones(Xen.shape[0], dtype=bool)
                mask[del_idx] = False
                X168 = _nm(Xen[mask])
                # handle size: Xen is 168+add_k, deleting 1 leaves 167+add_k
                # need exactly 168: if add_k>1 delete add_k points by load order
                if X168.shape[0] != 168:
                    order = np.argsort(-ld)
                    todel = set(order[:add_k].tolist())
                    mask2 = np.ones(Xen.shape[0], dtype=bool)
                    for ii in todel:
                        mask2[ii] = False
                    X168 = _nm(Xen[mask2])
                # short joint refine + LSE
                X168 = _refine(X168, steps=450, beta0=30.0, beta1=480.0, lr0=0.014, lr1=0.0008, deadline=SEARCH - 1)
                improved = try_update(X168)
                # LP polish if competitive
                v168 = _mc(X168)
                if v168 < bv + 0.0015 and time.time() < SEARCH - 1:
                    try:
                        Xp = _lp(X168, min(time.time() + (1.4 if improved or v168 < bv + 2e-4 else 0.6), SEARCH + 1, HARD - 3))
                        try_update(_nm(Xp))
                    except Exception:
                        pass
            except Exception:
                continue
        else:
            # UNDERSHOOT: delete 8-18 highest-load points
            del_k = 8 + ((cyc * 7) % 11)  # 8..18
            try:
                ld, _, _ = _loads(xb)
                order = np.argsort(-ld)
                todel = set(order[:del_k].tolist())
                mask = np.ones(168, dtype=bool)
                for ii in todel:
                    mask[ii] = False
                Xsmall = _nm(xb[mask])
                # polish small to its own optimum
                Xsmall = _refine(Xsmall, steps=650, beta0=28.0, beta1=480.0, lr0=0.016, lr1=0.0009, deadline=SEARCH - 2)
                # grow back 2-4 at a time
                cur = Xsmall
                batch = 3 if (cyc % 4 != 3) else 2
                while cur.shape[0] < 168 and time.time() < SEARCH - 2:
                    need = 168 - cur.shape[0]
                    nb = int(min(batch, need))
                    # vary batch occasionally between 2-4
                    holes = _deepest_holes(cur, nb, rng, nsamp=15000, polish=True)
                    cur = _nm(np.vstack([cur, holes]))
                    cur = _refine(cur, steps=220, beta0=30.0, beta1=300.0, lr0=0.014, lr1=0.0012, deadline=SEARCH - 1)
                if cur.shape[0] != 168:
                    continue
                cur = _refine(cur, steps=450, beta0=30.0, beta1=520.0, lr0=0.014, lr1=0.0008, deadline=SEARCH - 1)
                improved = try_update(cur)
                vcur = _mc(cur)
                if vcur < bv + 0.0015 and time.time() < SEARCH - 1:
                    try:
                        Xp = _lp(cur, min(time.time() + (1.4 if improved or vcur < bv + 2e-4 else 0.6), SEARCH + 1, HARD - 3))
                        try_update(_nm(Xp))
                    except Exception:
                        pass
            except Exception:
                continue
        # seed each cycle from best descendant (xb already updated only on improvement)
        # occasionally perturb best slightly to avoid re-finding same holes with same rng state:
        # (rng state already advances, so holes differ)

    if time.time() < HARD - 2:
        try:
            Xf = _lp(x_best, min(HARD - 1.0, time.time() + max(4.0, HARD - time.time() - 1.0)))
            Xf = _nm(Xf)
            if _mc(Xf) < _mc(x_best):
                x_best = Xf
        except Exception:
            pass
    return _nm(x_best)
