import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _sig_deg(X, tol=1e-4):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    M = G > (m - tol)
    deg = M.sum(axis=1).astype(int)
    return tuple(sorted(deg.tolist())) + (round(m, 9),)

def _refine_hole(Y, p, iters=120, beta=100.0, lr0=0.12, lr1=0.004):
    p = p / (np.linalg.norm(p) + 1e-18)
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        lr = lr0 * (lr1 / lr0) ** frac
        g = Y @ p
        gm = g.max()
        w = np.exp(beta * (g - gm))
        s = w.sum() + 1e-18
        A = (w @ Y) / s
        At = A - float(A @ p) * p
        p = p - lr * At
        n = np.linalg.norm(p)
        if n == 0:
            break
        p = p / n
    return p

def _block_quench(Y, B, steps, beta0, beta1, lr0, lr1, t_lim=None):
    k = B.shape[0]
    m = np.zeros_like(B); v = np.zeros_like(B)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    Yn = Y
    for t in range(1, steps + 1):
        if t_lim is not None and (t & 63) == 0 and time.time() > t_lim:
            break
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        C = B @ Yn.T
        S = B @ B.T
        gmax = C.max()
        sm = S.copy()
        np.fill_diagonal(sm, -10.0)
        if sm.max() > gmax:
            gmax = float(sm.max())
        Wc = np.exp(beta * (C - gmax))
        Ws = np.exp(beta * (sm - gmax))
        np.fill_diagonal(Ws, 0.0)
        tot = Wc.sum() + Ws.sum() + 1e-18
        Grad = (Wc @ Yn + Ws @ B) / tot
        dot = (Grad * B).sum(axis=1, keepdims=True)
        Gt = Grad - dot * B
        m *= b1; m += (1 - b1) * Gt
        v *= b2; v += (1 - b2) * (Gt * Gt)
        bc1 = 1 - b1 ** t; bc2 = 1 - b2 ** t
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        B = B - lr * step
        B /= np.sqrt((B * B).sum(axis=1, keepdims=True)) + 1e-18
    return B

def _free_quench(X, steps, beta0, beta1, lr0, lr1, t_lim=None):
    n = X.shape[0]
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        if t_lim is not None and (t & 127) == 0 and time.time() > t_lim:
            break
        frac = t / steps
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = np.exp(ll0 + (ll1 - ll0) * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum() + 1e-18
        Grad = (W @ X) / S
        # scale to per-point magnitude (balance)
        Grad = Grad * float(n)
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1; m += (1 - b1) * Gt
        v *= b2; v += (1 - b2) * (Gt * Gt)
        bc1 = 1 - b1 ** t; bc2 = 1 - b2 ** t
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        X = X - lr * step
        X /= np.sqrt((X * X).sum(axis=1, keepdims=True)) + 1e-18
    return X

def _hard(X, steps=180, lr0=0.0025, lr1=0.00012, tol0=2e-4, tol1=1e-6):
    X = X.copy()
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        tol = tol0 + (tol1 - tol0) * frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        M = (G > (mx - tol)).astype(np.float64)
        np.fill_diagonal(M, 0.0)
        cnt = M.sum(axis=1, keepdims=True) + 1e-12
        act = (cnt[:, 0] > 0.5)
        if not np.any(act):
            continue
        Grad = (M @ X) / cnt
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        nn = np.sqrt((Gt * Gt).sum(axis=1, keepdims=True)) + 1e-12
        st = np.zeros_like(Gt)
        st[act] = Gt[act] / nn[act]
        X = X - lr * st
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-18
    return X

def _lp(x, seconds, r0=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2500:
            idx = np.argsort(c[sel])[-2500:]
            I, J = I[idx], J[idx]
            cc = c[sel][idx]
        else:
            cc = c[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time(); BUDGET = 175.0
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape != (168, 8):
                continue
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if not np.all(np.isfinite(a)):
                continue
            va = _max_cos(a)
            if va < best_val:
                best_val = va; best = a.copy()
        except Exception:
            pass
    if best is None:
        r0 = np.random.RandomState(0)
        best = r0.randn(168, 8); best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    rng = np.random.RandomState(20261103)
    cur = best.copy(); cur_val = best_val
    archive = set()
    try:
        archive.add(_sig_deg(best))
    except Exception:
        pass
    ks = [10, 16, 24]
    it = 0
    # time guards
    while time.time() < t0 + BUDGET - 18:
        it += 1
        if time.time() + 6.0 > t0 + BUDGET - 12:
            break
        try:
            # choose base: mostly cur/best
            u0 = rng.rand()
            if u0 < 0.55:
                base = cur.copy()
            elif u0 < 0.85:
                base = best.copy()
            else:
                # random restart perturbation of best
                base = best + rng.randn(168, 8) * 0.02
                base /= np.linalg.norm(base, axis=1, keepdims=True)
            base_val = _max_cos(base)
            # choose center: hub or random direction
            if rng.rand() < 0.5:
                G = base @ base.T
                np.fill_diagonal(G, -2.0)
                per = G.max(axis=1)
                hub = int(np.argmax(per))
                c = base[hub].copy()
            else:
                c = rng.randn(8); c /= np.linalg.norm(c)
            k = ks[int(rng.randint(len(ks)))]
            dots = base @ c
            rem_idx = np.argsort(dots)[-k:]
            keep_mask = np.ones(168, bool); keep_mask[rem_idx] = False
            Y = base[keep_mask].copy()
            # 20000 probes
            P = rng.randn(20000, 8)
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            # scores vs frozen
            CH = 5000
            bs = np.empty(20000)
            for a0 in range(0, 20000, CH):
                b1 = min(a0 + CH, 20000)
                bs[a0:b1] = (P[a0:b1] @ Y.T).max(axis=1)
            order = np.argsort(bs)
            holes = []
            # scan first 4000 deepest
            lim = min(4000, 20000)
            for oi in range(lim):
                if len(holes) >= k:
                    break
                p = P[order[oi]]
                if holes:
                    mh = 0.0
                    for q in holes:
                        v = abs(float(p @ q))
                        if v > mh:
                            mh = v
                    if mh > 0.85:
                        continue
                Yext = Y if len(holes) == 0 else np.vstack([Y, np.array(holes)])
                pr = _refine_hole(Yext, p, iters=110, beta=85.0 + rng.rand() * 45.0)
                if holes:
                    dh = max(float(pr @ q) for q in holes)
                    ah = max(abs(float(pr @ q)) for q in holes)
                    if dh > 0.70 or ah > 0.86:
                        continue
                holes.append(pr)
            # fallback fill
            fi = lim
            while len(holes) < k and fi < 20000:
                p = P[order[fi]]; fi += 1
                Yext = Y if len(holes) == 0 else np.vstack([Y, np.array(holes)])
                pr = _refine_hole(Yext, p, iters=80, beta=100.0)
                holes.append(pr)
            while len(holes) < k:
                p = rng.randn(8); p /= np.linalg.norm(p)
                Yext = Y if len(holes) == 0 else np.vstack([Y, np.array(holes)])
                holes.append(_refine_hole(Yext, p, iters=80, beta=100.0))
            B = np.array(holes, dtype=np.float64)
            # joint block optimisation (high beta soft-min)
            tlim_b = time.time() + 1.2
            B = _block_quench(Y, B, steps=420, beta0=140.0, beta1=650.0, lr0=0.012, lr1=0.0012, t_lim=tlim_b)
            # second block pass with higher beta
            B = _block_quench(Y, B, steps=220, beta0=350.0, beta1=900.0, lr0=0.004, lr1=0.0006, t_lim=time.time()+0.6)
            Xnew = np.vstack([Y, B])
            Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
            # unfreeze balanced quench
            Xnew = _free_quench(Xnew, steps=650, beta0=28.0, beta1=480.0, lr0=0.005, lr1=0.0005, t_lim=time.time()+0.9)
            Xnew = _hard(Xnew, steps=150, lr0=0.0025, lr1=0.00012)
            # short LP
            remain = (t0 + BUDGET - 12) - time.time()
            if remain > 4.0:
                Xnew = _lp(Xnew, seconds=min(1.1, remain - 3.0))
            vn = _max_cos(Xnew)
            if not np.all(np.isfinite(Xnew)):
                continue
            # basin rejection at 1e-4
            try:
                sg = _sig_deg(Xnew)
            except Exception:
                sg = None
            if sg is not None and sg in archive:
                # duplicate basin: only accept if strictly better than best (allow)
                if not (vn < best_val - 1e-12):
                    continue
            if sg is not None:
                archive.add(sg)
                if len(archive) > 4000:
                    # prune randomly by rebuilding (keep last)
                    archive = set(list(archive)[-2000:])
            # acceptance
            if vn < best_val:
                best_val = vn; best = Xnew.copy()
                cur = Xnew.copy(); cur_val = vn
            elif vn < best_val + 2e-6 and rng.rand() < 0.35:
                cur = Xnew.copy(); cur_val = vn
            elif vn < cur_val:
                cur = Xnew.copy(); cur_val = vn
        except Exception:
            continue
    # final polish of best
    try:
        if time.time() < t0 + BUDGET - 3:
            best = _hard(best, steps=220, lr0=0.002, lr1=0.00008)
            rem = (t0 + BUDGET - 1.5) - time.time()
            if rem > 0.8:
                best = _lp(best, seconds=min(3.0, rem))
            best_val = _max_cos(best)
    except Exception:
        pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
