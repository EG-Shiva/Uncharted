import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _adam(X0, steps, b0, b1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1m = 0.9
    b2m = 0.999
    eps = 1e-8
    t = 0
    lb0 = np.log(b0)
    lb1 = np.log(b1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = float(np.max(G))
        G -= cm
        G *= beta
        np.clip(G, -30.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G))
        if S < 1e-100:
            continue
        GX = G @ X
        GX *= (2.0 / S)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GX -= dot * X
        t += 1
        m *= b1m
        m += (1.0 - b1m) * GX
        v *= b2m
        v += (1.0 - b2m) * (GX * GX)
        c1 = 1.0 - (b1m ** t)
        c2 = 1.0 - (b2m ** t)
        denom = np.sqrt(v / c2) + eps
        step = (lr / c1) * m / denom
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 5.0 * lr + 1e-12
        sc = np.minimum(1.0, cap / sn)
        step *= sc
        X -= step
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
        md = np.sum(m * X, axis=1, keepdims=True)
        m -= md * X
    return X

def _lp_polish(x, deadline, k_top=4500, r0=1.2e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline - 1.0 and radius > 1e-14:
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        m = float(np.max(c_all))
        # wide window to tie plateau
        sel = c_all > m - max(20.0 * radius, 1e-6)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > k_top:
            # keep top k_top
            idx = np.argpartition(-c_all, k_top)[:k_top]
            I = iu0[idx]; J = iu1[idx]; c_sel = c_all[idx]
        else:
            I = iu0[sel]; J = iu1[sel]; c_sel = c_all[sel]
        p = I.shape[0]
        nv = n * d + 1
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq,
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
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        vv = _max_cos(y)
        if vv < best - 1e-12:
            x, best = y, vv
            radius = min(radius * 1.5, 0.012)
        else:
            radius *= 0.5
        if time.time() > deadline - 1.5:
            break
    return x

def _block_void(Y_cur, sel, rng, ns=5000):
    n, d = Y_cur.shape
    sel = np.asarray(sel, dtype=int)
    mask = np.ones(n, dtype=bool)
    mask[sel] = False
    Others = Y_cur[mask]
    R = rng.standard_normal((ns, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True) + 1e-16
    D = R @ Others.T
    md = D.max(axis=1)
    order = np.argsort(md)
    Y = Y_cur.copy()
    picked = []
    for oi in order:
        cand = R[oi]
        ok = True
        for pc in picked:
            if float(cand @ pc) > 0.90:
                ok = False
                break
        if not ok:
            continue
        picked.append(cand)
        if len(picked) >= len(sel):
            break
    for k2, idx in enumerate(sel):
        if k2 < len(picked):
            Y[idx] = picked[k2]
        else:
            vv = rng.standard_normal(d)
            vv -= float(np.dot(vv, Y[idx])) * Y[idx]
            Y[idx] = Y[idx] + vv * 0.1
            Y[idx] /= np.linalg.norm(Y[idx]) + 1e-16
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-16
    return Y

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    HOP_END = t0 + 130.0
    rng = np.random.default_rng(12345)
    n, d = 250, 6
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        X = rng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        cands.append(X)
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)
    def time_left():
        return HARD - time.time()

    # quick settle
    try:
        if time.time() < t0 + 30:
            Y = _adam(best, 450, 400.0, 6000.0, 0.004, 0.0015)
            v = _max_cos(Y)
            if v < best_val:
                best, best_val = Y, v
    except Exception:
        pass

    cur = best.copy()
    cur_val = best_val
    sigma = 0.03
    fail = 0

    while time.time() < HOP_END and time_left() > 32:
        r = float(rng.random())
        try:
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            order = np.argsort(-rm)
            if r < 0.36:
                # block void reinsertion: 4-8 worst
                k = int(rng.integers(4, 9))
                # sometimes include random rattlers too
                if float(rng.random()) < 0.3:
                    sel = list(order[:k-2]) + list(rng.choice(order[50:], size=2, replace=False))
                else:
                    sel = [int(o) for o in order[:k]]
                Y = _block_void(cur, sel, rng, ns=5000)
                Y2 = _adam(Y, 260, 900.0, 9000.0, 0.003, 0.001)
            elif r < 0.66:
                # coherent cap shift: take neighborhood of a seed
                seed = int(order[int(rng.integers(0, 8))]) if float(rng.random()) < 0.7 else int(rng.integers(0, n))
                c = cur[seed]
                dots = cur @ c
                cap_k = int(rng.integers(10, 22))
                cap_idx = np.argpartition(-dots, cap_k)[:cap_k]
                # common tangent direction
                tu = rng.standard_normal(d)
                tu -= float(np.dot(tu, c)) * c
                tn = float(np.linalg.norm(tu)) + 1e-16
                tu /= tn
                mag = sigma * (1.0 + 2.0 * float(rng.random()))
                Y = cur.copy()
                for idx in cap_idx:
                    ind = rng.standard_normal(d)
                    ind -= float(np.dot(ind, Y[idx])) * Y[idx]
                    inn = float(np.linalg.norm(ind)) + 1e-16
                    ind /= inn
                    shift = tu * mag * (0.5 + float(rng.random())) + ind * sigma * 0.5
                    # project shift to tangent at Y[idx]
                    shift -= float(np.dot(shift, Y[idx])) * Y[idx]
                    Y[idx] = Y[idx] + shift
                Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-16
                Y2 = _adam(Y, 260, 900.0, 9000.0, 0.003, 0.001)
            elif r < 0.86:
                # bottleneck pair split + neighbor agitation
                iu = np.triu_indices(n, 1)
                # find top pair
                # use G already
                flat = np.argmax(G)
                ai, aj = divmod(int(flat), n)
                if ai == aj:
                    # fallback
                    ai, aj = int(order[0]), int(np.argmax(G[int(order[0])]))
                alpha = 0.02 + 0.03 * float(rng.random())
                Y = cur.copy()
                # push apart along difference
                for (idx, sgn) in ((ai, 1.0), (aj, -1.0)):
                    oth = cur[aj] if idx == ai else cur[ai]
                    dv = Y[idx] - oth
                    dv -= float(np.dot(dv, Y[idx])) * Y[idx]
                    dn = float(np.linalg.norm(dv)) + 1e-16
                    dv /= dn
                    Y[idx] = Y[idx] + dv * alpha * (0.7 + 0.6 * float(rng.random()))
                # agitate their nearest neighbors
                for anchor in (ai, aj):
                    dots2 = cur @ cur[anchor]
                    nn = np.argpartition(-dots2, 8)[:8]
                    for idx in nn:
                        if idx == ai or idx == aj:
                            continue
                        if float(rng.random()) < 0.6:
                            nz = rng.standard_normal(d)
                            nz -= float(np.dot(nz, Y[idx])) * Y[idx]
                            nz /= float(np.linalg.norm(nz)) + 1e-16
                            Y[idx] = Y[idx] + nz * sigma * 0.6
                Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-16
                Y2 = _adam(Y, 300, 1000.0, 10000.0, 0.0025, 0.0009)
            else:
                Y = cur + rng.standard_normal((n, d)) * (sigma * 0.4)
                Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-16
                Y2 = _adam(Y, 300, 700.0, 9000.0, 0.003, 0.001)
            v = _max_cos(Y2)
        except Exception:
            fail += 1
            continue
        if v < best_val - 1e-12:
            best, best_val = Y2, v
            cur, cur_val = Y2, v
            fail = 0
            sigma = max(0.008, sigma * 0.94)
        elif v < cur_val - 1e-12:
            cur, cur_val = Y2, v
            fail = 0
        else:
            fail += 1
            if fail >= 7:
                cur = best.copy()
                cur_val = best_val
                sigma = min(0.06, sigma * 1.3)
                fail = 0

    try:
        if time_left() > 14:
            Y = _adam(best, 400, 2500.0, 16000.0, 0.0012, 0.0005)
            v = _max_cos(Y)
            if v < best_val:
                best, best_val = Y, v
    except Exception:
        pass
    try:
        if time_left() > 9:
            Y = _lp_polish(best, HARD - 4.0, k_top=4500, r0=1.2e-3)
            v = _max_cos(Y)
            if v < best_val:
                best, best_val = Y, v
    except Exception:
        pass
    try:
        if time_left() > 4:
            Y = _adam(best, 200, 8000.0, 25000.0, 0.0006, 0.0002)
            v = _max_cos(Y)
            if v < best_val:
                best, best_val = Y, v
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
