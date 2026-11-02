import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _anneal(x0, steps, T0, T1, lr, mu, rng, deadline=None):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    if T0 <= 0 or T1 <= 0:
        T0 = max(T0, 1e-12); T1 = max(T1, 1e-12)
    lratio = np.log(T1 / T0)
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lrk = lr * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        Gr = (E @ x) / S
        Gr -= (Gr * x).sum(axis=1, keepdims=True) * x
        vel = mu * vel + (1.0 - mu) * Gr
        x = x - lrk * vel
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _lbfgs(x_init, deadline, taus=(0.002, 0.0008, 0.0003), iters=(60, 60, 80)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.4:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            GY = (GX - (GX * X).sum(axis=1, keepdims=True) * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 4),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
        except Exception:
            break
    Yf = y.reshape(n, d)
    Yf /= np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
    return Yf

def _lp_polish(x, deadline, r0=0.002, cap=1200):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if int(sel.sum()) == 0:
                sel[int(np.argmax(c))] = True
            if int(sel.sum()) > cap:
                idx = np.argsort(-c)[:cap]
                s2 = np.zeros_like(sel, dtype=bool)
                s2[idx] = True
                sel = s2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            cols[:, :d] = I[:, None] * d + ar
            cols[:, d:2 * d] = J[:, None] * d + ar
            cols[:, 2 * d] = n * d
            cols = cols.ravel()
            vals = np.empty((p, 2 * d + 1), dtype=float)
            vals[:, :d] = x[J]
            vals[:, d:2 * d] = x[I]
            vals[:, 2 * d] = -1.0
            vals = vals.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            yy = x + res.x[:-1].reshape(n, d)
            yy /= np.maximum(np.linalg.norm(yy, axis=1, keepdims=True), 1e-12)
            v = _max_cos(yy)
            if v < best:
                x, best = yy, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _refine_hole(y0, Xf, iters=200):
    y = y0 / max(np.linalg.norm(y0), 1e-12)
    dots = Xf @ y
    vy = float(dots.max())
    step = 0.05
    for _ in range(iters):
        dots = Xf @ y
        j = int(np.argmax(dots))
        dj = float(dots[j])
        g = Xf[j] - dj * y
        ng = float(np.linalg.norm(g))
        if ng < 1e-13:
            break
        g /= ng
        improved = False
        for s in (step, step * 0.3):
            yn = y - s * g
            yn /= max(np.linalg.norm(yn), 1e-12)
            vn = float((Xf @ yn).max())
            if vn < vy:
                y, vy = yn, vn
                improved = True
                break
        if improved:
            step = min(step * 1.05, 0.05)
        else:
            step *= 0.92
            if step < 1e-5:
                break
    return y, vy

def _greedy_pack(T, rng, deadline, max_total=220000, batch=6000):
    n_target = 163
    A = np.zeros((n_target, 4))
    k = 0
    total = 0
    stall_batches = 0
    while k < n_target and time.time() < deadline and total < max_total:
        Y = rng.randn(batch, 4)
        Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        total += batch
        if k > 0:
            Xf = A[:k]
            D = Y @ Xf.T
            vals = D.max(axis=1)
        else:
            vals = np.full(batch, -2.0)
        order = np.argsort(vals)
        added = 0
        for idx in order:
            if vals[idx] >= T:
                break
            y = Y[idx]
            if k > 0:
                # fast check against current set (vals outdated after additions)
                if added > 0:
                    if float((A[:k] @ y).max()) >= T:
                        continue
                # else vals already guarantees ok
            A[k] = y
            k += 1
            added += 1
            if k >= n_target:
                break
        if added == 0:
            stall_batches += 1
        else:
            stall_batches = 0
        # if stalled, try refined deep holes
        if stall_batches >= 1 and k >= 1 and k < n_target:
            if time.time() > deadline:
                break
            Xf = A[:k].copy()
            # take few best candidates and refine
            bidx = order[:min(8, batch)]
            for bi in bidx:
                try:
                    yr, vr = _refine_hole(Y[bi], Xf, iters=180)
                except Exception:
                    continue
                if vr < T and float((A[:k] @ yr).max()) < T:
                    A[k] = yr
                    k += 1
                    Xf = A[:k].copy()
                    added += 1
                    if k >= n_target:
                        break
            if added > 0:
                stall_batches = 0
            else:
                # extra random refined attempts
                for _ in range(4):
                    y0 = rng.randn(4)
                    y0 /= max(np.linalg.norm(y0), 1e-12)
                    try:
                        yr, vr = _refine_hole(y0, Xf, iters=200)
                    except Exception:
                        continue
                    if vr < T and float((A[:k] @ yr).max()) < T:
                        A[k] = yr
                        k += 1
                        Xf = A[:k].copy()
                        added += 1
                        if k >= n_target:
                            break
                if added > 0:
                    stall_batches = 0
        # adapt batch size late (smaller) to save time
        if k > 140 and batch > 3000:
            batch = 3000
    if k < n_target:
        return None
    return A.copy()

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 128.0
    n = 163
    base_rng = np.random.RandomState(20261102)
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = base_rng.randn(n, 4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (n, 4) or not np.all(np.isfinite(bx)):
        bx = base_rng.randn(n, 4)
    bx = bx / np.maximum(np.linalg.norm(bx, axis=1, keepdims=True), 1e-12)
    try:
        best_val = _max_cos(bx)
    except Exception:
        best_val = 1.0
    best_x = bx.copy()
    thresholds = [0.8542, 0.8548, 0.8552, 0.8556, 0.8560, 0.8563, 0.8550, 0.8558]
    attempt = 0
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 40:
            break
        T = thresholds[attempt % len(thresholds)]
        # tiny jitter to diversify
        jit = (base_rng.rand() - 0.5) * 0.0002
        Tt = T + jit
        seed = 1000 + attempt * 7919
        rng = np.random.RandomState(seed)
        gdead = min(SEARCH_END, time.time() + 9.0)
        try:
            pack = _greedy_pack(Tt, rng, deadline=gdead)
        except Exception:
            pack = None
        attempt += 1
        if pack is None:
            continue
        pack /= np.maximum(np.linalg.norm(pack, axis=1, keepdims=True), 1e-12)
        try:
            pv = _max_cos(pack)
        except Exception:
            continue
        # polish pipeline with remaining budget
        cur = pack
        cur_v = pv
        try:
            if time.time() < SEARCH_END - 1:
                T0 = 0.006 if (attempt % 2 == 0) else 0.009
                cur_a, va = _anneal(cur, steps=1500, T0=T0, T1=0.0003, lr=0.032, mu=0.86,
                                    rng=rng, deadline=min(SEARCH_END, time.time() + 3.0))
                if va < cur_v:
                    cur, cur_v = cur_a, va
        except Exception:
            pass
        try:
            if time.time() < SEARCH_END - 0.5:
                cur_l = _lbfgs(cur, deadline=min(SEARCH_END, time.time() + 2.5))
                vl = _max_cos(cur_l)
                if vl < cur_v:
                    cur, cur_v = cur_l, vl
        except Exception:
            pass
        # LP polish only if promising or new topology near record
        do_lp = (cur_v < best_val + 0.003)
        if do_lp and time.time() < DEADLINE - 8:
            try:
                lp_x, lp_v = _lp_polish(cur, deadline=min(DEADLINE - 2.0, time.time() + 3.0), r0=0.0025)
                if lp_v < cur_v:
                    cur, cur_v = lp_x, lp_v
            except Exception:
                pass
        if cur_v < best_val - 1e-12:
            best_val = cur_v
            best_x = cur.copy()
            # early exit if big win
            if best_val < 0.8545 and time.time() > t0 + 60:
                pass
        # if failed threshold but close, keep trying denser
    # final short polish of best if time remains
    if time.time() < DEADLINE - 6:
        try:
            fa, va = _anneal(best_x, steps=800, T0=0.002, T1=0.0002, lr=0.02, mu=0.87,
                             rng=base_rng, deadline=DEADLINE - 4.0)
            if va < best_val:
                best_val = va
                best_x = fa
        except Exception:
            pass
    if time.time() < DEADLINE - 3:
        try:
            lp_x, lp_v = _lp_polish(best_x, deadline=DEADLINE - 0.7, r0=0.0015)
            if lp_v < best_val:
                best_val = lp_v
                best_x = lp_x
        except Exception:
            pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    # shrink by tiny margin guard against rounding (keep strictly feasible, negligible score change)
    return best_x
