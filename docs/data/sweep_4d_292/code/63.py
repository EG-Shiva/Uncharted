import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contacts_deg(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu0 = np.triu_indices(x.shape[0], 1)
    c = G[iu0]
    sel = c > m - tol
    deg = np.zeros(x.shape[0], dtype=int)
    I = iu0[0][sel]; J = iu0[1][sel]
    for i, j in zip(I.tolist(), J.tolist()):
        deg[i] += 1
        deg[j] += 1
    return deg, m

def _anneal_masked(x, mask, steps, beta0, beta1, lr0, lr1, mu=0.85):
    x = _norm(x.copy())
    vel = np.zeros_like(x)
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return x
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    ll = np.log(lr1 / lr0) if abs(lr1 - lr0) > 1e-18 else 0.0
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * np.exp(ll * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        r = W.sum(axis=1, keepdims=True)
        np.maximum(r, 1e-30, out=r)
        A = (W @ x) / r
        At = A - np.sum(A * x, axis=1, keepdims=True) * x
        vel = mu * vel + (1.0 - mu) * At
        bias = 1.0 - mu ** (s + 1)
        step = lr * (vel / max(bias, 1e-12))
        step[~m] = 0.0
        x = x - step
        x = _norm(x)
    return x

def _refine_cands(h, Xkeep, steps=150, beta0=60.0, beta1=450.0, lr0=0.03, lr1=0.005):
    h = _norm(h.copy())
    vel = np.zeros_like(h)
    mu = 0.8
    lb = np.log(beta1 / beta0)
    ll = np.log(lr1 / lr0)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * np.exp(ll * frac)
        D = h @ Xkeep.T
        dm = D.max(axis=1, keepdims=True)
        W = np.exp(beta * (D - dm))
        r = W.sum(axis=1, keepdims=True)
        np.maximum(r, 1e-30, out=r)
        A = (W @ Xkeep) / r
        At = A - np.sum(A * h, axis=1, keepdims=True) * h
        vel = mu * vel + (1.0 - mu) * At
        bias = 1.0 - mu ** (s + 1)
        h = h - lr * (vel / max(bias, 1e-12))
        h = _norm(h)
    return h

def _refine_sel(sel, Xkeep, steps=200, beta0=100.0, beta1=600.0, lr0=0.02, lr1=0.003):
    h = _norm(sel.copy())
    vel = np.zeros_like(h)
    mu = 0.82
    lb = np.log(beta1 / beta0)
    ll = np.log(lr1 / lr0)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * np.exp(ll * frac)
        Dk = h @ Xkeep.T
        Hh = h @ h.T
        np.fill_diagonal(Hh, -2.0)
        mk = Dk.max(axis=1, keepdims=True)
        mh = Hh.max(axis=1, keepdims=True)
        mall = np.maximum(mk, mh)
        Wk = np.exp(beta * (Dk - mall))
        Wh = np.exp(beta * (Hh - mall))
        np.fill_diagonal(Wh, 0.0)
        rk = Wk.sum(axis=1, keepdims=True)
        rh = Wh.sum(axis=1, keepdims=True)
        r = rk + rh
        np.maximum(r, 1e-30, out=r)
        A = (Wk @ Xkeep + Wh @ h) / r
        At = A - np.sum(A * h, axis=1, keepdims=True) * h
        vel = mu * vel + (1.0 - mu) * At
        bias = 1.0 - mu ** (s + 1)
        h = h - lr * (vel / max(bias, 1e-12))
        h = _norm(h)
    return h

def _lp_polish(x, seconds, r0=8e-4):
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    if seconds <= 0:
        return xb
    start = time.time()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = c.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if not np.any(sel):
            break
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = I[:, None] * d + np.arange(d)
        cj = J[:, None] * d + np.arange(d)
        cols = np.concatenate([ci, cj, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    DEAD = 172.0
    rng = np.random.default_rng(20261031)
    def _load(p):
        try:
            a = np.load(p)
            a = np.asarray(a, dtype=float)
            if a.shape == (292, 4) and np.all(np.isfinite(a)):
                n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1.0
                return a / n
        except Exception:
            pass
        return None
    best = _load("best.npy")
    rec = _load("record.npy")
    if best is None:
        best = _norm(rng.standard_normal((292, 4)))
    if rec is not None and _max_cos(rec) < _max_cos(best):
        best = rec.copy()
    best_val = _max_cos(best)
    # quick initial LP touch if record not yet polished
    # main surgery loop
    it = 0
    # precompute nothing
    while time.time() - t0 < DEAD - 14.0:
        it += 1
        time_left = DEAD - (time.time() - t0)
        if time_left < 8.0:
            break
        src = best.copy()
        deg, msrc = _contacts_deg(src, tol=1e-7)
        ratt = np.where(deg == 0)[0].tolist()
        low = np.where(deg <= 3)[0].tolist()
        high = np.where(deg >= 8)[0].tolist()
        if len(high) < 3:
            order_d = np.argsort(-deg)
            high = order_d[:max(10, len(high))].tolist()
        if len(low) < 2:
            order_a = np.argsort(deg)
            low = order_a[:10].tolist()
        # deletion set size 3-4
        k = 4 if (it % 2 == 1) else 3
        dell = []
        # always include rattler (or lowest degree as pseudo-rattler)
        if ratt:
            # pick one rattler deterministically cycling
            dell.append(int(ratt[(it - 1) % len(ratt)]))
        else:
            order_a = np.argsort(deg)
            dell.append(int(order_a[(it - 1) % 292]))
        # variant composition
        mode = it % 3
        rng.shuffle(high); rng.shuffle(low)
        if mode == 0:
            for i in high:
                if len(dell) >= k: break
                if int(i) not in dell: dell.append(int(i))
        elif mode == 1:
            # 2 high + 1 low
            nh = 0
            for i in high:
                if nh >= min(2, k - 1): break
                if int(i) not in dell:
                    dell.append(int(i)); nh += 1
            for i in low:
                if len(dell) >= k: break
                if int(i) not in dell: dell.append(int(i))
        else:
            # 1 high + rest random
            for i in high:
                if len(dell) >= 2: break
                if int(i) not in dell: dell.append(int(i))
            pool = list(range(292))
            rng.shuffle(pool)
            for i in pool:
                if len(dell) >= k: break
                if int(i) not in dell: dell.append(int(i))
        # fill if short
        if len(dell) < k:
            pool = list(range(292)); rng.shuffle(pool)
            for i in pool:
                if len(dell) >= k: break
                if int(i) not in dell: dell.append(int(i))
        dell = dell[:k]
        keep_idx = [i for i in range(292) if i not in set(dell)]
        Xkeep = src[keep_idx]
        T = best_val
        # massive quasi-random sampling
        N_CAND = 50000
        cand = rng.standard_normal((N_CAND, 4))
        cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
        bscore = np.empty(N_CAND)
        cs = 8192
        for a in range(0, N_CAND, cs):
            b = min(a + cs, N_CAND)
            D = cand[a:b] @ Xkeep.T
            bscore[a:b] = D.max(axis=1)
        order = np.argsort(bscore)
        H = 400
        h0 = cand[order[:H]]
        try:
            hR = _refine_cands(h0, Xkeep, steps=120, beta0=60.0, beta1=450.0)
        except Exception:
            hR = h0
        D2 = hR @ Xkeep.T
        s2 = D2.max(axis=1)
        o2 = np.argsort(s2)
        hR = hR[o2]; s2 = s2[o2]
        # greedy pick with mutual constraint
        sel = []
        found = False
        for relax in [0.0, 0.001, 0.002, 0.004, 0.008, 0.02]:
            sel = []
            Tl = T + relax
            for idx in range(H):
                if len(sel) >= k: break
                if s2[idx] > Tl + 0.002 and len(sel) < k and relax < 0.008:
                    # allow slightly worse keep-score only in later relax
                    if s2[idx] > Tl:
                        continue
                c = hR[idx]
                ok = True
                for s in sel:
                    if float(s @ c) > Tl:
                        ok = False; break
                if ok:
                    sel.append(c)
            if len(sel) >= k:
                found = True
                break
        if len(sel) < k:
            # fill greedily minimizing max
            while len(sel) < k:
                bc = None; bm = 1e9
                for idx in range(H):
                    c = hR[idx]
                    skip = False
                    for s in sel:
                        if np.linalg.norm(c - s) < 1e-9:
                            skip = True; break
                    if skip: continue
                    m1 = float(s2[idx])
                    m2 = -2.0
                    for s in sel:
                        m2 = max(m2, float(s @ c))
                    mm = max(m1, m2)
                    if mm < bm:
                        bm = mm; bc = c
                if bc is None: break
                sel.append(bc)
        while len(sel) < k:
            r = rng.standard_normal(4); r = r / np.linalg.norm(r)
            sel.append(r)
        sel = np.array(sel[:k])
        # locally refine with keep frozen
        try:
            sel = _refine_sel(sel, Xkeep, steps=180)
        except Exception:
            pass
        Xnew = np.vstack([Xkeep, sel])
        Xnew = _norm(Xnew)
        nkeep = Xkeep.shape[0]
        # staged thawing masks
        # stage 1: moved + neighbours cos>0.55
        try:
            Dins = Xkeep @ sel.T
            neigh1 = np.where(Dins.max(axis=1) > 0.55)[0]
            mask1 = np.zeros(292, dtype=bool)
            mask1[nkeep:] = True
            mask1[neigh1] = True
            Xnew = _anneal_masked(Xnew, mask1, steps=350, beta0=100.0, beta1=1500.0, lr0=0.02, lr1=0.002)
            # stage 2: shell cos>0.2
            sel2 = _norm(Xnew[nkeep:])
            Dins2 = Xnew[:nkeep] @ sel2.T
            neigh2 = np.where(Dins2.max(axis=1) > 0.20)[0]
            mask2 = np.zeros(292, dtype=bool)
            mask2[nkeep:] = True
            mask2[neigh2] = True
            # if shell is almost all, just do full
            if mask2.sum() > 260:
                mask2[:] = True
            Xnew = _anneal_masked(Xnew, mask2, steps=350, beta0=100.0, beta1=1500.0, lr0=0.015, lr1=0.0015)
            # stage 3: all
            mask3 = np.ones(292, dtype=bool)
            Xnew = _anneal_masked(Xnew, mask3, steps=450, beta0=100.0, beta1=1500.0, lr0=0.012, lr1=0.001)
        except Exception:
            pass
        v = _max_cos(Xnew)
        # short LP polish
        tl2 = DEAD - (time.time() - t0)
        lp_sec = min(2.5, max(0.6, tl2 - 10.0))
        if lp_sec > 0.4:
            try:
                Xnew = _lp_polish(Xnew, seconds=lp_sec)
                v = _max_cos(Xnew)
            except Exception:
                pass
        if v < best_val - 1e-12:
            best = _norm(Xnew.copy())
            best_val = v
    # final polish on best with remaining time
    tl = DEAD - (time.time() - t0)
    if tl > 3.0:
        try:
            best = _lp_polish(best, seconds=min(6.0, tl - 1.0))
            best_val = _max_cos(best)
        except Exception:
            pass
    best = _norm(best)
    return best
