import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _anneal_full(x, steps, beta0, beta1, lr0, lr1, mu=0.85):
    x = _norm(x.copy())
    vel = np.zeros_like(x)
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
        x = x - lr * (vel / max(bias, 1e-12))
        x = _norm(x)
    return x

def _anneal_masked(x, mask, steps, beta0, beta1, lr0, lr1, mu=0.85):
    x = _norm(x.copy())
    vel = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    ll = np.log(lr1 / lr0) if abs(lr1 - lr0) > 1e-18 else 0.0
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return x
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

def _lp_polish_deadline(x, deadline, r0=8e-4):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = c.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if not np.any(sel):
            break
        I = iu0[0][sel]
        J = iu0[1][sel]
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
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb

def _contacts(x, tol=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(x.shape[0], 1)
    c = G[iu]
    sel = c > m - tol
    s = set(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
    deg = np.zeros(x.shape[0], dtype=int)
    for (i, j) in s:
        deg[i] += 1
        deg[j] += 1
    return s, deg, m

def _refine_holes(holes, Xkeep, steps=150, lr0=0.03, lr1=0.005, beta0=60.0, beta1=450.0):
    h = _norm(holes.copy())
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

def solve():
    t0 = time.time()
    DEAD = 172.0
    rng = np.random.default_rng(20261031)

    def _load(p):
        try:
            a = np.load(p)
            a = np.asarray(a, dtype=float)
            if a.shape == (292, 4) and np.all(np.isfinite(a)):
                n = np.linalg.norm(a, axis=1, keepdims=True)
                n[n == 0] = 1.0
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
    incumbent = best.copy()
    try:
        inc_contacts, _, _ = _contacts(incumbent, tol=1e-7)
    except Exception:
        inc_contacts = set()
    cur = best.copy()
    cur_val = best_val
    archive = [incumbent.copy()]
    arch_vals = [best_val]
    ks = [4, 8, 12, 16, 20]
    it = 0
    while time.time() - t0 < DEAD - 14.0:
        it += 1
        time_left = DEAD - (time.time() - t0)
        if time_left < 14.0:
            break
        k = ks[(it - 1) % len(ks)]
        try:
            if len(archive) > 1 and rng.random() < 0.3:
                src_idx = int(rng.integers(0, len(archive)))
                src = archive[src_idx].copy()
            else:
                src = cur.copy()
            cset, deg, msrc = _contacts(src, tol=1e-7)
            ratt = np.where(deg == 0)[0].tolist()
            low = np.where(deg <= 4)[0].tolist()
            high = np.where(deg >= 8)[0].tolist()
            dell = []
            def _add(pool, nwant):
                cand = [i for i in pool if i not in dell]
                rng.shuffle(cand)
                for i in cand[:nwant]:
                    dell.append(int(i))
            if ratt:
                _add(ratt, min(len(ratt), 1 + k // 10))
            nlow = max(1, int(round(0.35 * k))) if low else 0
            nhigh = max(1, int(round(0.35 * k))) if high else 0
            _add(low, nlow)
            _add(high, nhigh)
            extra = []
            for i in low + high:
                if i not in dell:
                    extra.append(i)
            rng.shuffle(extra)
            for i in extra:
                if len(dell) >= k:
                    break
                dell.append(int(i))
            rest = list(range(292))
            rng.shuffle(rest)
            for i in rest:
                if len(dell) >= k:
                    break
                if int(i) not in dell:
                    dell.append(int(i))
            dell = dell[:k]
            keep_idx = [i for i in range(292) if i not in set(dell)]
            Xkeep = src[keep_idx]
            N_CAND = 65536
            cand = rng.standard_normal((N_CAND, 4))
            cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
            bscore = np.empty(N_CAND, dtype=float)
            cs = 8192
            for a in range(0, N_CAND, cs):
                b = min(a + cs, N_CAND)
                D = cand[a:b] @ Xkeep.T
                bscore[a:b] = D.max(axis=1)
            order = np.argsort(bscore, kind="stable")
            H = 300
            h0 = cand[order[:H]]
            hR = _refine_holes(h0, Xkeep, steps=150)
            D2 = hR @ Xkeep.T
            s2 = D2.max(axis=1)
            o2 = np.argsort(s2, kind="stable")
            hR = hR[o2]
            s2 = s2[o2]
            sel_idx = []
            sel_vecs = []
            for tt in range(k):
                bi = -1
                bm = 1e9
                for idx in range(H):
                    if idx in sel_idx:
                        continue
                    c = hR[idx]
                    m1 = float(s2[idx])
                    m2 = -2.0
                    for s in sel_vecs:
                        d = float(np.dot(c, s))
                        if d > m2:
                            m2 = d
                    mm = m1 if m1 > m2 else m2
                    if mm < bm:
                        bm = mm
                        bi = idx
                if bi < 0:
                    break
                sel_idx.append(bi)
                sel_vecs.append(hR[bi])
            while len(sel_vecs) < k:
                r = rng.standard_normal((1, 4))
                r = r / np.linalg.norm(r)
                sel_vecs.append(r[0])
            Sel = np.array(sel_vecs[:k])
            Xnew = np.vstack([Xkeep, Sel])
            Xnew = _norm(Xnew)
            nkeep = Xkeep.shape[0]
            Gki = Xnew[:nkeep] @ Xnew[nkeep:].T
            neigh = np.where(Gki.max(axis=1) > 0.75)[0]
            mask = np.zeros(292, dtype=bool)
            mask[nkeep:] = True
            mask[neigh] = True
            Xnew = _anneal_masked(Xnew, mask, steps=200, beta0=40.0, beta1=350.0, lr0=0.03, lr1=0.004)
            Xnew = _anneal_full(Xnew, steps=300, beta0=30.0, beta1=500.0, lr0=0.025, lr1=0.003)
            lp_dead = min(time.time() + 2.5, t0 + DEAD - 9.0)
            if lp_dead > time.time() + 0.5:
                Xnew = _lp_polish_deadline(Xnew, lp_dead, r0=8e-4)
            vnew = _max_cos(Xnew)
            try:
                nset, _, _ = _contacts(Xnew, tol=1e-7)
                diff = len(nset.symmetric_difference(inc_contacts))
            except Exception:
                diff = 0
                nset = set()
            is_new_basin = diff >= 20
            if is_new_basin and len(archive) < 12:
                archive.append(Xnew.copy())
                arch_vals.append(vnew)
            if vnew < best_val:
                best_val = vnew
                best = Xnew.copy()
                if is_new_basin or vnew < cur_val:
                    pass
            if vnew < cur_val:
                cur = Xnew.copy()
                cur_val = vnew
            elif is_new_basin and vnew < cur_val + 0.002 and rng.random() < 0.5:
                cur = Xnew.copy()
                cur_val = vnew
            elif rng.random() < 0.05 and vnew < cur_val + 0.001:
                cur = Xnew.copy()
                cur_val = vnew
        except Exception:
            continue
    try:
        final_dead = min(time.time() + max(1.0, DEAD - (time.time() - t0) - 1.0), t0 + DEAD - 0.5)
        if final_dead > time.time() + 0.5:
            best = _lp_polish_deadline(best, final_dead, r0=6e-4)
            best_val = _max_cos(best)
    except Exception:
        pass
    best = _norm(np.asarray(best, dtype=float))
    return best
