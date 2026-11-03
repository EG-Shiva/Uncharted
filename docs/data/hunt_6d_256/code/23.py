import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_sm(x0, steps, K0, K1, lr):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 150 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2e-3, floor=1e-4, cap=4000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0 = np.triu_indices(n, 1)
    iuI, iuJ = iu0[0], iu0[1]
    ar = np.arange(d)
    eq_rows = np.repeat(np.arange(n), d)
    eq_cols = np.arange(n * d)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        margin = max(20.0 * radius, floor)
        sel = c > (m - margin)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c[idx], cap)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
            cnt = cap
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (eq_rows, eq_cols)), shape=(n, nv)).tocsr()
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        val = _max_cos(y)
        if val < best:
            x, best = y, val
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x

def _single_sweep(x, rng, deadline, top_k=16, batch=384, sigma0=0.01):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    while time.time() < deadline:
        prow = G.max(axis=1)
        order = np.argsort(-prow)[:top_k]
        improved = False
        for ii in order:
            if time.time() >= deadline:
                break
            i = int(ii)
            cur_row = float(prow[i])
            if cur_row < best - 2e-3:
                continue
            G2max = float(np.max(np.delete(np.delete(G, i, axis=0), i, axis=1)))
            target = min(cur_row, best)
            nbr = np.where(G[i] > target - 8e-3)[0]
            nbr = nbr[nbr != i]
            rep = np.zeros(d)
            if len(nbr) > 0:
                w = np.exp(600.0 * (G[i, nbr] - target))
                w /= w.sum()
                rep = w @ x[nbr]
                rep = rep - np.dot(rep, x[i]) * x[i]
                rn = np.linalg.norm(rep)
                if rn > 1e-12:
                    rep /= rn
                else:
                    rep[:] = 0
            R = rng.normal(size=(batch, d))
            R -= np.outer(R @ x[i], x[i])
            rn = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
            R /= rn
            scales = np.empty(batch)
            scales[:batch//2] = sigma0
            scales[batch//2:3*batch//4] = sigma0 * 0.3
            scales[3*batch//4:] = sigma0 * 2.5
            if np.linalg.norm(rep) > 1e-12:
                for k_, sc in enumerate([sigma0*0.5, sigma0, sigma0*2.0, sigma0*4.0]):
                    if k_ < batch:
                        R[k_] = rep
                        scales[k_] = sc
            Y = x[i][None, :] + R * scales[:, None]
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            D = Y @ x.T
            D[:, i] = -2.0
            new_rows = D.max(axis=1)
            new_glob = np.maximum(new_rows, G2max)
            kb = int(np.argmin(new_glob))
            if new_glob[kb] < best - 1e-12:
                x[i] = Y[kb]
                best = float(new_glob[kb])
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                prow = G.max(axis=1)
                improved = True
            elif new_rows[kb] < cur_row - 1e-9 and new_glob[kb] <= best + 1e-12:
                x[i] = Y[kb]
                G = x @ x.T
                np.fill_diagonal(G, -2.0)
                prow = G.max(axis=1)
                improved = True
        if not improved:
            sigma0 *= 0.5
            if sigma0 < 3e-4:
                break
            if time.time() >= deadline:
                break
    return x

def _pair_split_refine(x, rng, deadline, batch=512, sigma0=0.012, n_pairs=4):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    while time.time() < deadline:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        flat = G.copy()
        pairs = []
        F = G.copy()
        for _ in range(n_pairs):
            ai = int(np.argmax(F)) // n
            aj = int(np.argmax(F)) % n
            if float(F[ai, aj]) < -1.5:
                break
            pairs.append((ai, aj))
            F[ai, :] = -2.0; F[:, ai] = -2.0
            F[aj, :] = -2.0; F[:, aj] = -2.0
        if not pairs:
            break
        any_imp = False
        for (i, j) in pairs:
            if time.time() >= deadline:
                break
            mask = np.ones(n, dtype=bool); mask[i] = False; mask[j] = False
            Xr = x[mask]
            Gr = Xr @ Xr.T
            np.fill_diagonal(Gr, -2.0)
            other_max = float(Gr.max()) if Gr.size else -2.0
            # directions away from each other
            ti = x[j] - np.dot(x[j], x[i]) * x[i]
            ti /= (np.linalg.norm(ti) + 1e-12)
            tj = x[i] - np.dot(x[i], x[j]) * x[j]
            tj /= (np.linalg.norm(tj) + 1e-12)
            Ri = rng.normal(size=(batch, d))
            Ri -= np.outer(Ri @ x[i], x[i])
            Ri /= (np.linalg.norm(Ri, axis=1, keepdims=True) + 1e-12)
            Rj = rng.normal(size=(batch, d))
            Rj -= np.outer(Rj @ x[j], x[j])
            Rj /= (np.linalg.norm(Rj, axis=1, keepdims=True) + 1e-12)
            si = np.empty(batch); sj = np.empty(batch)
            # mix: half repulsion-biased, half random
            for k in range(batch):
                r = k % 4
                if r == 0:
                    s = sigma0 * (0.5 + rng.random()*1.5)
                elif r == 1:
                    s = sigma0 * 0.3
                elif r == 2:
                    s = sigma0 * 2.0
                else:
                    s = sigma0 * 4.0
                si[k] = s; sj[k] = s * (0.5 + rng.random())
            # first few: pure repulsion
            for k_, sc in enumerate([sigma0*0.5, sigma0, sigma0*2.0, sigma0*3.0]):
                if k_ < batch:
                    Ri[k_] = -ti; Rj[k_] = -tj
                    si[k_] = sc; sj[k_] = sc
            Yi = x[i][None, :] + Ri * si[:, None]
            Yi /= np.linalg.norm(Yi, axis=1, keepdims=True)
            Yj = x[j][None, :] + Rj * sj[:, None]
            Yj /= np.linalg.norm(Yj, axis=1, keepdims=True)
            Di = Yi @ Xr.T
            Dj = Yj @ Xr.T
            ri = Di.max(axis=1)
            rj = Dj.max(axis=1)
            pij = np.sum(Yi * Yj, axis=1)
            ng = np.maximum(np.maximum(ri, rj), pij)
            ng = np.maximum(ng, other_max)
            kb = int(np.argmin(ng))
            if ng[kb] < best - 1e-12:
                x[i] = Yi[kb]; x[j] = Yj[kb]
                best = float(ng[kb])
                any_imp = True
        if not any_imp:
            sigma0 *= 0.6
            if sigma0 < 5e-4:
                break
    return x

def _void_reinsert(x, rng, n_cand=8192):
    n, d = x.shape
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    prow = G.max(axis=1)
    i = int(np.argmax(prow))
    mask = np.ones(n, dtype=bool); mask[i] = False
    Xr = x[mask]
    C = rng.normal(size=(n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = C @ Xr.T
    rm = D.max(axis=1)
    k = int(np.argmin(rm))
    Gr = Xr @ Xr.T
    np.fill_diagonal(Gr, -2.0)
    other = float(Gr.max())
    new_global = max(float(rm[k]), other)
    cur = _max_cos(x)
    if new_global < cur:
        x = x.copy()
        x[i] = C[k]
    return x

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_END = t0 + 172.0
    ANNEAL_END = t0 + 140.0
    rng = _np.random.default_rng(7)
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (256, 6) and _np.all(_np.isfinite(a)):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                vv = _max_cos(a)
                if vv < best:
                    best = vv
                    x_best = a.copy()
        except Exception:
            continue
    if x_best is None:
        x_best = rng.normal(size=(256, 6))
        x_best /= _np.linalg.norm(x_best, axis=1, keepdims=True)
        best = _max_cos(x_best)
    n, d = 256, 6
    # quick symmetric attack on isolated pair before main loop
    try:
        x_best = _pair_split_refine(x_best, rng, min(_time.time()+6.0, ANNEAL_END-20), batch=512, sigma0=0.015, n_pairs=3)
        best = _max_cos(x_best)
    except Exception:
        pass
    try:
        x_best = _lp_polish(x_best, min(_time.time()+5.0, ANNEAL_END), r0=1.5e-3, floor=8e-5)
        best = _max_cos(x_best)
    except Exception:
        pass
    cycle = 0
    while _time.time() < ANNEAL_END:
        cycle += 1
        rem = ANNEAL_END - _time.time()
        if rem < 3:
            break
        mode = cycle % 6
        x = x_best.copy()
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        prow = G.max(axis=1)
        cnt = _np.sum(G > gmax - 5e-4, axis=1)
        order = _np.argsort(-prow)
        try:
            if mode == 0:
                sig = 0.008
                x += sig * rng.normal(size=(n, d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = (450, 150.0, 2500.0, 0.006)
            elif mode == 1:
                # kick worst cluster
                wk = order[:12]
                x[wk] += 0.03 * rng.normal(size=(12, d))
                x[_np.random.default_rng(cycle).choice(n, 4, replace=False)] += 0.01 * rng.normal(size=(4, d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = (500, 60.0, 1500.0, 0.009)
            elif mode == 2:
                # pair-split kick on worst pair
                ai = int(_np.argmax(G)) // n; aj = int(_np.argmax(G)) % n
                ti = x[aj] - _np.dot(x[aj], x[ai]) * x[ai]; ti /= (_np.linalg.norm(ti)+1e-12)
                tj = x[ai] - _np.dot(x[ai], x[aj]) * x[aj]; tj /= (_np.linalg.norm(tj)+1e-12)
                x[ai] = x[ai] - 0.02*ti + 0.008*rng.normal(size=d)
                x[aj] = x[aj] - 0.02*tj + 0.008*rng.normal(size=d)
                x += 0.004 * rng.normal(size=(n, d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = (450, 120.0, 3000.0, 0.007)
            elif mode == 3:
                # void reinsertion
                x = _void_reinsert(x, rng, n_cand=8192)
                # small noise to neighbours of void
                x += 0.005 * rng.normal(size=(n, d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = (450, 100.0, 2000.0, 0.007)
            elif mode == 4:
                # large escape kick
                sel = rng.choice(n, size=10, replace=False)
                x[sel] += 0.12 * rng.normal(size=(10, d))
                x /= _np.linalg.norm(x, axis=1, keepdims=True)
                steps, K0, K1, lr = (600, 30.0, 800.0, 0.012)
            else:
                # pure exploitation, no kick
                steps, K0, K1, lr = (400, 400.0, 4000.0, 0.004)
            if rem < 25:
                steps = min(steps, 300)
            xb, vb = _adam_sm(x, steps, K0, K1, lr)
        except Exception:
            continue
        # joint + single refinement
        try:
            tref = _time.time() + min(2.0, max(0.5, rem-1.0))
            xb = _pair_split_refine(xb, rng, tref, batch=256, sigma0=0.008, n_pairs=2)
        except Exception:
            pass
        try:
            tref2 = _time.time() + min(1.5, max(0.5, ANNEAL_END - _time.time()))
            if tref2 > _time.time() + 0.2:
                xb = _single_sweep(xb, rng, tref2, top_k=12, batch=256, sigma0=0.006)
        except Exception:
            pass
        try:
            vb = _max_cos(xb)
        except Exception:
            continue
        if vb < best - 1e-12:
            x_best = xb
            best = vb
            # opportunistic LP to lock basin
            try:
                if _time.time() < ANNEAL_END - 6:
                    x_best = _lp_polish(x_best, min(_time.time()+3.0, ANNEAL_END), r0=1.2e-3, floor=5e-5)
                    best = _max_cos(x_best)
            except Exception:
                pass
    # final polish
    try:
        if _time.time() < T_END - 10:
            xb, vb = _adam_sm(x_best, 500, 400.0, 3500.0, 0.004)
            if vb < best:
                x_best, best = xb, vb
    except Exception:
        pass
    try:
        if _time.time() < T_END - 6:
            x_best = _pair_split_refine(x_best, rng, T_END-4.0, batch=512, sigma0=0.004, n_pairs=4)
    except Exception:
        pass
    try:
        if _time.time() < T_END - 4:
            x_best = _single_sweep(x_best, rng, T_END-2.0, top_k=24, batch=384, sigma0=0.003)
    except Exception:
        pass
    try:
        if _time.time() < T_END - 1.5:
            x_best = _lp_polish(x_best, T_END-0.8, r0=7e-4, floor=3e-5, cap=5000)
    except Exception:
        pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = _np.where(_np.isfinite(x_best), x_best, 0.0)
    nr = _np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
