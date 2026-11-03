import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, max_pairs=1800):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best_val = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    iu_r, iu_c = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_full = G[iu_r, iu_c]
        m = c_full.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_full > thr
        p = int(sel.sum())
        if p == 0:
            radius *= 0.5
            continue
        if p > max_pairs:
            idx = np.argpartition(-c_full, max_pairs)[:max_pairs]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
            p = max_pairs
        I = iu_r[sel]
        J = iu_c[sel]
        c_sel = c_full[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(),
                 (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel,
                          A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if np.any(~np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        v = _max_cos(y)
        if v < best_val:
            best_val = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
            if time.time() >= deadline:
                break
        else:
            radius *= 0.5
    return best_x

def _soft_descent(x, steps, a0, a1, lr0, lr1, mu=0.8, max_move=0.05):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    vel = np.zeros_like(x)
    la = np.log(a1 / a0) if a0 > 0 and a1 > 0 else 0.0
    for s in range(steps):
        f = s / max(steps - 1, 1)
        alpha = a0 * np.exp(la * f)
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_t = grad - dot * x
        vel = mu * vel + (1.0 - mu) * grad_t
        step = lr * (grad_t + 0.5 * vel)
        nm = np.linalg.norm(step, axis=1, keepdims=True)
        scale = np.minimum(1.0, max_move / (nm + 1e-16))
        step = step * scale
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _soft_modes(x, k=36, thresh=0.002, alpha=300.0):
    try:
        from scipy.sparse.linalg import eigsh, LinearOperator
    except Exception:
        return None, None
    n, d = x.shape
    N = n * d
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    iu_r, iu_c = np.triu_indices(n, 1)
    c = G[iu_r, iu_c]
    sel = c > m - thresh
    I = iu_r[sel]; J = iu_c[sel]
    cv = c[sel]
    if len(I) > 3000:
        idx = np.argpartition(-cv, 3000)[:3000]
        I = I[idx]; J = J[idx]; cv = cv[idx]
    if len(I) < 50:
        return None, None
    w = np.exp(alpha * (cv - m))
    # rotation basis
    pairs = [(a, b) for a in range(d) for b in range(a + 1, d)]
    R = np.zeros((N, len(pairs)))
    for col, (a, b) in enumerate(pairs):
        blk = np.zeros((n, d))
        blk[:, a] = x[:, b]
        blk[:, b] = -x[:, a]
        R[:, col] = blk.ravel()
    try:
        Qr, _ = np.linalg.qr(R, mode='reduced')
    except Exception:
        return None, None
    xc = x.copy()
    Ic = I.copy(); Jc = J.copy(); wc = w.copy()
    Qc = Qr.copy()
    def matvec(v):
        V = v.reshape(n, d)
        dot = np.sum(V * xc, axis=1, keepdims=True)
        Vt = V - dot * xc
        Vn = V - Vt
        av = np.sum(Vt[Ic] * xc[Jc], axis=1) + np.sum(xc[Ic] * Vt[Jc], axis=1)
        wav = wc * av
        Hv = np.zeros((n, d))
        np.add.at(Hv, Ic, wav[:, None] * xc[Jc])
        np.add.at(Hv, Jc, wav[:, None] * xc[Ic])
        doth = np.sum(Hv * xc, axis=1, keepdims=True)
        Hvt = Hv - doth * xc
        coeff = Qc.T @ v
        rot = (Qc @ coeff).reshape(n, d)
        out = Hvt + 10.0 * Vn + 10.0 * rot
        return out.ravel()
    try:
        Op = LinearOperator((N, N), matvec=lambda v: matvec(v), dtype=float)
        kk = min(k, N - 2)
        vals, vecs = eigsh(Op, k=kk, which='SM', ncv=min(N - 1, 3 * kk + 20), maxiter=250, tol=1e-3)
        order = np.argsort(vals)
        return vals[order], vecs[:, order]
    except Exception:
        return None, None

def _kmeans_assign(x, k, rng, iters=8):
    n, d = x.shape
    ci = rng.choice(n, size=k, replace=False)
    C = x[ci].copy()
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        dots = x @ C.T
        assign = np.argmax(dots, axis=1)
        for c in range(k):
            m = assign == c
            if np.any(m):
                v = x[m].sum(axis=0)
                nv = np.linalg.norm(v)
                if nv > 1e-12:
                    C[c] = v / nv
    return assign

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    EXPLORE_END = t0 + 144.0
    rng = np.random.default_rng(1234567)
    try:
        best = np.load("best.npy").astype(float)
    except Exception:
        try:
            best = np.load("record.npy").astype(float)
        except Exception:
            best = rng.standard_normal((102, 12))
    n, d = 102, 12
    if best.shape != (n, d):
        try:
            rec = np.load("record.npy").astype(float)
            if rec.shape == (n, d):
                best = rec
            else:
                raise ValueError("bad")
        except Exception:
            tmp = rng.standard_normal((n, d))
            m0 = min(best.shape[0], n); m1 = min(best.shape[1], d)
            tmp[:m0, :m1] = best[:m0, :m1]
            best = tmp
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best_val = _max_cos(best)
    cur = best.copy()
    cur_val = best_val
    iu_r, iu_c = np.triu_indices(n, 1)

    def edit_dist(a, b, thr):
        Ga = a @ a.T; Gb = b @ b.T
        ma = _max_cos(a); mb = _max_cos(b)
        # fixed threshold near best to compare topology
        T = min(best_val + 0.001, 0.5)
        ea = Ga[iu_r, iu_c] > T
        eb = Gb[iu_r, iu_c] > T
        union = np.sum(ea | eb)
        if union == 0:
            return 0.0
        return float(np.sum(ea ^ eb)) / float(union)

    def best_hole(others, kcand=2000):
        cand = rng.standard_normal((kcand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ others.T
        worst = dots.max(axis=1)
        return cand[np.argmin(worst)]

    modes = None
    modes_base = None
    modes_age = 100
    it = 0
    TEMP0 = 0.0022
    while time.time() < EXPLORE_END:
        it += 1
        if time.time() >= HARD_END - 15.0:
            break
        if it % 15 == 0 and rng.random() < 0.4:
            cur = best.copy(); cur_val = best_val
            modes_age = 100
        # choose move type
        r = rng.random()
        if r < 0.50:
            mtype = 'spectral'
        elif r < 0.80:
            mtype = 'cluster'
        else:
            mtype = 'hole'
        trial = cur.copy()
        if mtype == 'spectral':
            need = (modes is None) or (modes_age >= 4)
            if need:
                if time.time() > EXPLORE_END - 2.0:
                    continue
                vals, vecs = _soft_modes(cur, k=36)
                if vecs is not None:
                    modes = vecs
                    modes_age = 0
                else:
                    modes = None
                    modes_age = 100
            if modes is not None:
                kk = modes.shape[1]
                nm = int(rng.integers(20, min(37, kk + 1)))
                if nm < 4:
                    nm = kk
                cols = rng.choice(kk, size=nm, replace=False)
                coeff = rng.standard_normal(nm)
                delta = (modes[:, cols] @ coeff).reshape(n, d)
                # remove global rotation / normal drift explicitly
                dotd = np.sum(delta * cur, axis=1, keepdims=True)
                delta = delta - dotd * cur
                amp = float(rng.uniform(0.02, 0.15))
                dn = np.linalg.norm(delta)
                if dn < 1e-12 or not np.isfinite(dn):
                    trial = cur + 0.03 * rng.standard_normal((n, d))
                else:
                    delta *= (amp * np.sqrt(n) / dn)
                    trial = cur + delta
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                modes_age += 1
                trial = _soft_descent(trial, 380, 14.0, 110.0, 0.011, 0.0035)
            else:
                trial = cur + 0.03 * rng.standard_normal((n, d))
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                trial = _soft_descent(trial, 400, 12.0, 100.0, 0.010, 0.004)
        elif mtype == 'cluster':
            kcl = int(rng.integers(4, 7))
            try:
                assign = _kmeans_assign(cur, kcl, rng, iters=7)
            except Exception:
                assign = rng.integers(0, kcl, size=n)
            cl = int(rng.integers(0, kcl))
            idx = np.where(assign == cl)[0]
            if len(idx) == 0 or len(idx) == n:
                # PC1 split fallback
                try:
                    _, _, Vt = np.linalg.svd(cur - cur.mean(axis=0), full_matrices=False)
                    pc1 = cur @ Vt[0]
                    idx = np.where(pc1 > np.median(pc1))[0]
                    if rng.random() < 0.5:
                        idx = np.where(pc1 <= np.median(pc1))[0]
                except Exception:
                    idx = rng.choice(n, size=n // 4, replace=False)
            a, b = rng.choice(d, size=2, replace=False)
            th = float(rng.uniform(0.03, 0.20)) * (1.0 if rng.random() < 0.5 else -1.0)
            co, si = np.cos(th), np.sin(th)
            xa = trial[idx, a].copy(); xb = trial[idx, b].copy()
            trial[idx, a] = co * xa - si * xb
            trial[idx, b] = si * xa + co * xb
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _soft_descent(trial, 420, 13.0, 115.0, 0.011, 0.0035)
        else:
            Gc = cur @ cur.T
            np.fill_diagonal(Gc, -2.0)
            pm = Gc.max(axis=1)
            order = np.argsort(-pm)
            c0 = int(order[int(rng.integers(0, 8))])
            mask = np.ones(n, dtype=bool); mask[c0] = False
            trial[c0] = best_hole(trial[mask], 2000)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            trial = _soft_descent(trial, 420, 12.0, 115.0, 0.010, 0.0035)

        v_soft = _max_cos(trial)
        # LP only for promising candidates
        do_lp = (v_soft < 0.269) and (time.time() < EXPLORE_END)
        if do_lp:
            dl = min(time.time() + 1.1, EXPLORE_END + 0.5, HARD_END - 15.0)
            if dl - time.time() < 0.3:
                polished = trial; v = v_soft
            else:
                polished = _lp_polish(trial, dl)
                v = _max_cos(polished)
        else:
            # allow diverse wandering without LP occasionally
            polished = trial; v = v_soft
            # skip far-worse without LP (save time) unless diverse
            if v > best_val + 0.018 and v > cur_val + 0.012:
                if rng.random() > 0.03:
                    continue
        frac = (time.time() - t0) / max(EXPLORE_END - t0, 1.0)
        Temp = TEMP0 * (1.0 - 0.7 * frac)
        if v < best_val - 1e-12:
            best = polished / np.linalg.norm(polished, axis=1, keepdims=True)
            best_val = v
            cur = best.copy(); cur_val = v
            modes_age = 100
        else:
            acc = False
            if v < cur_val:
                acc = True
            else:
                try:
                    ed = edit_dist(polished, cur, best_val)
                except Exception:
                    ed = 0.0
                if v < best_val + 0.002 and ed > 0.15:
                    # favour topology change
                    acc = rng.random() < 0.55
                else:
                    acc = rng.random() < np.exp((cur_val - v) / max(Temp, 1e-4))
            if acc:
                cur = polished / np.linalg.norm(polished, axis=1, keepdims=True)
                cur_val = v
                modes_age = 100 if rng.random() < 0.3 else modes_age + 1

    remain = HARD_END - time.time()
    if remain > 5.0:
        try:
            best = _lp_polish(best, time.time() + remain - 1.2, max_pairs=2500)
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
