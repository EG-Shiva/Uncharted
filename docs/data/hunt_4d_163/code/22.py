import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _rowmax(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g

def _lp_polish(x, deadline, r0=0.004, cap=1200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
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
            thresh = max(20.0 * radius, 1e-9)
            sel = c > m - thresh
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                sel[np.argmax(c)] = True
                cnt = 1
            if cnt > cap:
                # keep largest
                idx = np.argpartition(-c, cap)[:cap]
                sel = np.zeros_like(sel, dtype=bool)
                sel[idx] = True
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            tmpc = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None] * d + ar
            tmpc[:, d:2 * d] = J[:, None] * d + ar
            tmpc[:, 2 * d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2 * d] = x[I]
            cc[:, 2 * d] = -1.0
            vals = cc.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _hb_smooth(x0, steps, T0, T1, lr0, mu, rng, deadline=None):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    best = x.copy()
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    best_val = float(G.max())
    if T0 > 0 and T1 > 0:
        lratio = np.log(T1 / T0)
    else:
        lratio = 0.0
    for k in range(steps):
        if deadline is not None and (k & 63) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        # normalize grad scale
        vel = mu * vel + (1.0 - mu) * grad
        x = x - lr * (0.5 * grad + 0.5 * vel)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < best_val:
            best_val = m
            best = x.copy()
    return best, best_val

def _lbfgs_smooth(x_init, deadline, tau=0.0012, maxiter=100):
    x_init = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x_init.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x_init.ravel().copy()
    def fg(yflat):
        Y = yflat.reshape(n, d)
        rn = np.linalg.norm(Y, axis=1, keepdims=True)
        rn = np.maximum(rn, 1e-12)
        X = Y / rn
        g = (X @ X.T)[iu0, iu1]
        m = float(g.max())
        e = np.exp((g - m) / tau)
        s = float(e.sum())
        if s <= 0 or not np.isfinite(s):
            return float(m), np.zeros_like(yflat)
        L = m + tau * np.log(s / Np)
        w = e / s
        W = np.zeros((n, n), dtype=float)
        W[iu0, iu1] = w
        W = W + W.T
        GX = W @ X
        dot = (GX * X).sum(axis=1, keepdims=True)
        GY = (GX - dot * X) / rn
        return float(L), GY.ravel()
    try:
        res = minimize(fg, y, method="L-BFGS-B", jac=True,
                       options={"maxiter": int(maxiter), "maxfun": int(maxiter * 6),
                                "ftol": 1e-14, "gtol": 1e-8})
        y = res.x
    except Exception:
        pass
    Y = y.reshape(n, d)
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return Y

def _deep_hole(others, rng, nc=10000, refine_rounds=3):
    d = others.shape[1]
    cand = rng.randn(nc, d)
    cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
    # chunk to bound memory
    chunk = 4000
    m_all = np.empty(nc)
    for a in range(0, nc, chunk):
        b = min(nc, a + chunk)
        dots = cand[a:b] @ others.T
        m_all[a:b] = dots.max(axis=1)
    bi = int(np.argmin(m_all))
    best_v = float(m_all[bi])
    best_c = cand[bi].copy()
    rad = 0.3
    for _ in range(refine_rounds):
        prop = rng.randn(2500, d) * (rad * 0.5)
        prop -= (prop @ best_c)[:, None] * best_c[None, :]
        prop = best_c[None, :] + prop
        prop /= np.maximum(np.linalg.norm(prop, axis=1, keepdims=True), 1e-12)
        dots = prop @ others.T
        m = dots.max(axis=1)
        j = int(np.argmin(m))
        if float(m[j]) < best_v:
            best_v = float(m[j])
            best_c = prop[j].copy()
        rad *= 0.45
    return best_c

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    LOOP_END = HARD_END - 15.0
    rng = np.random.RandomState(20261102)
    cands = []
    try:
        xb = np.load("best.npy").astype(np.float64)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        cands.append(xb)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(np.float64)
        xr = xr / np.linalg.norm(xr, axis=1, keepdims=True)
        cands.append(xr)
    except Exception:
        pass
    if len(cands) == 0:
        tmp = rng.randn(163, 4)
        tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        cands.append(tmp)
    best = min(cands, key=lambda z: _max_cos(z)).copy()
    best_val = _max_cos(best)
    n, d = best.shape
    cur = best.copy()
    cur_val = best_val
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while time.time() < LOOP_END:
        it += 1
        if time.time() > LOOP_END:
            break
        # occasional restart to best
        base = cur
        if rng.rand() < 0.08:
            base = best
        x = base.copy()
        mode = rng.rand()
        if mode < 0.32:
            # light: small shake + 1-2 hole reinsert + short smooth
            sigma = 0.002 + rng.rand() * 0.010
            x = x + rng.randn(n, d) * sigma
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            Gt = x @ x.T
            np.fill_diagonal(Gt, -2.0)
            rowmax = Gt.max(axis=1)
            order = np.argsort(-rowmax)
            k = 1 if rng.rand() < 0.55 else 2
            used = set()
            for t in range(k):
                if time.time() > LOOP_END:
                    break
                if rng.rand() < 0.75:
                    pool = order[:10]
                    victim = int(pool[rng.randint(0, len(pool))])
                else:
                    victim = int(rng.randint(0, n))
                if victim in used:
                    continue
                used.add(victim)
                mask = np.ones(n, dtype=bool); mask[victim] = False
                hc = _deep_hole(x[mask], rng, nc=7000 + rng.randint(0, 5000), refine_rounds=3)
                x[victim] = hc
            s0 = 30.0 / 1000.0 + rng.rand() * 0.03
            # convert to temperature: use absolute T
            T0 = 0.004 + rng.rand() * 0.006
            T1 = 0.0004 + rng.rand() * 0.0008
            steps = 130 + rng.randint(0, 120)
            lr = 0.12 + rng.rand() * 0.15
            mu = 0.70 + rng.rand() * 0.20
            # _hb expects T; map s0 style
            x, _ = _hb_smooth(x, steps, T0, T1, lr * 0.15, mu, rng, deadline=LOOP_END)
            if time.time() > LOOP_END:
                v = _max_cos(x)
                if v < best_val:
                    best, best_val = x, v
                break
            # occasional lbfgs
            if rng.rand() < 0.25 and time.time() < LOOP_END - 2.0:
                tau = float(rng.choice([0.002, 0.0012, 0.0006]))
                x = _lbfgs_smooth(x, LOOP_END, tau=tau, maxiter=80)
            x, _ = _lp_polish(x, min(LOOP_END, time.time() + 0.85), r0=0.0035)
        elif mode < 0.68:
            # cluster + edge break: topology change
            Gc = base @ base.T
            np.fill_diagonal(Gc, -2.0)
            rowmax = Gc.max(axis=1)
            w = int(np.argmax(rowmax)) if rng.rand() < 0.85 else int(rng.randint(0, n))
            msize = int(rng.choice([4, 5, 6, 7, 8, 9]))
            neigh = np.argsort(-Gc[w])[:msize]
            x = base + rng.randn(n, d) * 0.004
            common = rng.randn(d) * (0.04 + rng.rand() * 0.08)
            x[neigh] += common[None, :]
            x[neigh] += rng.randn(len(neigh), d) * (0.008 + rng.rand() * 0.018)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            # pair-break top edges
            Ge = x @ x.T
            np.fill_diagonal(Ge, -2.0)
            # get top edges
            flat = Ge[iu0, iu1]
            topk = np.argpartition(-flat, min(40, len(flat)-1))[:40]
            # pick 2-3 edges to break
            nb = int(rng.choice([2, 2, 3, 3]))
            for _ in range(nb):
                ei = int(topk[rng.randint(0, len(topk))])
                a = int(iu0[ei]); b_ = int(iu1[ei])
                s = 0.015 + rng.rand() * 0.03
                da = x[a] - x[b_]
                db = -da
                # tangentize
                da = da - (da @ x[a]) * x[a]
                db = db - (db @ x[b_]) * x[b_]
                na = np.linalg.norm(da) + 1e-12
                nb_ = np.linalg.norm(db) + 1e-12
                x[a] += (da / na) * s
                x[b_] += (db / nb_) * s
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            # hole reinsert 2-4
            Gt2 = x @ x.T
            np.fill_diagonal(Gt2, -2.0)
            rm2 = Gt2.max(axis=1)
            order = np.argsort(-rm2)
            k = int(rng.choice([2, 2, 3, 3, 4]))
            used = set()
            for t in range(k):
                if time.time() > LOOP_END:
                    break
                rr = rng.rand()
                if rr < 0.5:
                    pool = order[:10]
                    victim = int(pool[rng.randint(0, len(pool))])
                elif rr < 0.75:
                    victim = int(neigh[rng.randint(0, len(neigh))])
                else:
                    victim = int(rng.randint(0, n))
                if victim in used:
                    continue
                used.add(victim)
                mask = np.ones(n, dtype=bool); mask[victim] = False
                hc = _deep_hole(x[mask], rng, nc=10000 + rng.randint(0, 7000), refine_rounds=4)
                x[victim] = hc
            T0 = 0.003 + rng.rand() * 0.005
            T1 = 0.0003 + rng.rand() * 0.0007
            steps = 180 + rng.randint(0, 150)
            lr = 0.14 + rng.rand() * 0.18
            mu = 0.72 + rng.rand() * 0.20
            x, _ = _hb_smooth(x, steps, T0, T1, lr * 0.15, mu, rng, deadline=LOOP_END)
            if time.time() > LOOP_END:
                v = _max_cos(x)
                if v < best_val:
                    best, best_val = x, v
                break
            if rng.rand() < 0.40 and time.time() < LOOP_END - 2.0:
                tau = float(rng.choice([0.0015, 0.0008, 0.0004]))
                x = _lbfgs_smooth(x, LOOP_END, tau=tau, maxiter=90)
            x, _ = _lp_polish(x, min(LOOP_END, time.time() + 1.0), r0=0.005)
        else:
            # big kick: large noise + many holes
            sigma = 0.045 + rng.rand() * 0.06
            x = base + rng.randn(n, d) * sigma
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            Gt = x @ x.T
            np.fill_diagonal(Gt, -2.0)
            rowmax = Gt.max(axis=1)
            order = np.argsort(-rowmax)
            k = int(rng.choice([3, 4, 4, 5]))
            used = set()
            for t in range(k):
                if time.time() > LOOP_END:
                    break
                if rng.rand() < 0.6:
                    pool = order[:12]
                    victim = int(pool[rng.randint(0, len(pool))])
                else:
                    victim = int(rng.randint(0, n))
                if victim in used:
                    continue
                used.add(victim)
                mask = np.ones(n, dtype=bool); mask[victim] = False
                hc = _deep_hole(x[mask], rng, nc=9000 + rng.randint(0, 7000), refine_rounds=3)
                x[victim] = hc
            T0 = 0.005 + rng.rand() * 0.006
            T1 = 0.0004 + rng.rand() * 0.0008
            steps = 220 + rng.randint(0, 150)
            lr = 0.15 + rng.rand() * 0.18
            x, _ = _hb_smooth(x, steps, T0, T1, lr * 0.15, 0.78, rng, deadline=LOOP_END)
            if time.time() > LOOP_END:
                v = _max_cos(x)
                if v < best_val:
                    best, best_val = x, v
                break
            if rng.rand() < 0.30 and time.time() < LOOP_END - 2.0:
                tau = float(rng.choice([0.002, 0.001, 0.0005]))
                x = _lbfgs_smooth(x, LOOP_END, tau=tau, maxiter=100)
            x, _ = _lp_polish(x, min(LOOP_END, time.time() + 1.05), r0=0.006)
        v = _max_cos(x)
        if v < best_val - 1e-12:
            best, best_val = x.copy(), v
            cur, cur_val = x.copy(), v
            # extra squeeze
            if time.time() < LOOP_END - 2.0:
                x2, v2 = _lp_polish(x, min(LOOP_END, time.time() + 0.7), r0=0.0018)
                vv = _max_cos(x2) if isinstance(v2, float) else v2
                # _lp returns tuple
                if vv < best_val:
                    best, best_val = x2.copy(), vv
                    cur, cur_val = x2.copy(), vv
        elif v < cur_val:
            cur, cur_val = x.copy(), v
        else:
            Temp = 0.0011 * (0.996 ** (it // 2)) + 7e-05
            if rng.rand() < np.exp(-(v - cur_val) / Temp) * 0.5:
                cur, cur_val = x.copy(), v
            if rng.rand() < 0.07:
                cur, cur_val = best.copy(), best_val
    remain = HARD_END - time.time() - 0.6
    if remain > 2.0:
        half = remain * 0.55
        best, best_val = _lp_polish(best, time.time() + half, r0=0.0035)
        remain2 = HARD_END - time.time() - 0.3
        if remain2 > 1.0:
            best, best_val = _lp_polish(best, time.time() + remain2, r0=0.0009)
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
