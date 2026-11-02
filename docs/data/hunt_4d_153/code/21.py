import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(x, deadline, max_solves=120):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu0 = np.triu_indices(n, 1)
    solves = 0
    while time.time() < deadline and radius > 1e-14 and solves < max_solves:
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        win = max(20.0 * radius, 1e-9)
        sel = c > m - win
        if int(sel.sum()) > 1800:
            idx = np.argpartition(-c, 1800)[:1800]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(I)
        if p == 0:
            break
        cc = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        solves += 1
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def _smooth_opt(x0, T, maxiter, gtol=1e-5):
    n, d = x0.shape
    x0f = x0.ravel().copy()

    def fun_g(flat):
        X = flat.reshape(n, d)
        r = np.linalg.norm(X, axis=1, keepdims=True)
        r = np.maximum(r, 1e-12)
        Y = X / r
        G = Y @ Y.T
        # off-diag max for shift
        # set diag to very low for max
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        # sum over i<j
        iu = np.triu_indices(n, 1)
        s = float(E[iu].sum()) + 1e-300
        L = m + T * np.log(s)
        Pdiv = E / s
        gradY = Pdiv @ Y
        dot = np.sum(gradY * Y, axis=1, keepdims=True)
        gradX = (gradY - dot * Y) / r
        return L, gradX.ravel()
    try:
        res = minimize(fun_g, x0f, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-12, "gtol": gtol,
                                "maxfun": maxiter * 3})
        X = res.x.reshape(n, d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X
    except Exception:
        return x0


def _find_void(x_keep, rng, ns=12000, refine=30):
    R = rng.standard_normal(size=(ns, 4))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    S = R @ x_keep.T
    m = S.max(axis=1)
    v = R[int(np.argmin(m))].copy()
    for it in range(refine):
        dots = x_keep @ v
        mm = float(dots.max())
        T = 0.004
        w = np.exp((dots - mm) / T)
        w /= (w.sum() + 1e-12)
        g = w @ x_keep
        gt = g - float(g @ v) * v
        step = 0.06 / (1.0 + it * 0.2)
        v = v - step * gt
        nrm = np.linalg.norm(v)
        if nrm < 1e-12:
            v = rng.standard_normal(4)
            nrm = np.linalg.norm(v)
        v /= nrm
    return v


def _patch_from_worst(G, seed, size, m, rng):
    n = G.shape[0]
    adj = G > (m - 0.022)
    seen = np.zeros(n, dtype=bool)
    seen[seed] = True
    frontier = [seed]
    order = [seed]
    while frontier and len(order) < size:
        nxt = []
        for u in frontier:
            nb = np.where(adj[u])[0]
            # randomize order for diversity
            if len(nb) > 1:
                nb = nb[rng.permutation(len(nb))]
            for w in nb:
                if int(w) == int(u):
                    continue
                if not seen[w]:
                    seen[w] = True
                    order.append(int(w))
                    nxt.append(int(w))
                    if len(order) >= size:
                        break
            if len(order) >= size:
                break
        frontier = nxt
        if not frontier:
            break
    if len(order) < size:
        per = G.max(axis=1)
        for w in np.argsort(-per):
            if not seen[w]:
                seen[w] = True
                order.append(int(w))
                if len(order) >= size:
                    break
    return np.array(order[:size], dtype=int)


def solve():
    t0 = time.time()
    deadline = t0 + 175.0
    explore_end = t0 + 128.0
    rng = np.random.default_rng(777123)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None and xb.shape == (153, 4):
        cands.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (153, 4):
        cands.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands:
        R = rng.standard_normal(size=(153, 4))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        cands.append(R)
    vals = [_max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_x = cands[bi].copy()
    best_val = float(vals[bi])
    # tiny initial squeeze
    best_x, best_val = _lp_polish(best_x, min(deadline - 5, time.time() + 3.0), max_solves=20)

    cur_x = best_x.copy()
    cur_val = best_val
    n = 153
    trial = 0
    T_accept = 0.0010

    while time.time() < explore_end:
        trial += 1
        if trial % 11 == 0:
            base = best_x
            base_val = best_val
        else:
            base = cur_x
            base_val = cur_val
        if cur_val > best_val + 0.012 and (trial % 4 == 0):
            base = best_x
            base_val = best_val

        G0 = base @ base.T
        np.fill_diagonal(G0, -2.0)
        per = G0.max(axis=1)
        m0 = float(G0.max())
        # contacts count
        contacts = (G0 > m0 - 1e-9).sum(axis=1)
        rattlers = np.where(contacts == 0)[0]
        overloaded = np.where(contacts >= 7)[0]
        mode = trial % 7
        y = base.copy()

        if mode == 0:
            idx = np.argsort(-per)[:10]
            y[idx] += 0.08 * rng.standard_normal((10, 4))
            y += 0.004 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, T=0.014, maxiter=120)
            y = _smooth_opt(y, T=0.004, maxiter=120)
        elif mode == 1:
            w = int(np.argmax(per))
            if len(rattlers) > 0 and (trial % 3 == 0):
                w = int(rattlers[0])
            keep = np.delete(base, w, axis=0)
            v = _find_void(keep, rng, ns=14000, refine=30)
            y[w] = v + 0.005 * rng.standard_normal(4)
            y += 0.003 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, T=0.012, maxiter=130)
            y = _smooth_opt(y, T=0.003, maxiter=130)
        elif mode == 2:
            if len(rattlers) >= 1:
                ws = [int(rattlers[0])]
                top = np.argsort(-per)[:12]
                # add one worst not already
                for c in top:
                    if int(c) not in ws:
                        ws.append(int(c))
                        break
                ws = np.array(ws[:2])
            else:
                ws = np.argsort(-per)[:2]
            keep = np.delete(base, ws, axis=0)
            for w in ws:
                v = _find_void(keep, rng, ns=9000, refine=25)
                y[w] = v + 0.007 * rng.standard_normal(4)
                keep = np.vstack([keep, y[w] / np.linalg.norm(y[w])])
            y += 0.003 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, T=0.013, maxiter=130)
            y = _smooth_opt(y, T=0.0035, maxiter=120)
        elif mode == 3:
            if len(overloaded) > 0:
                seed = int(overloaded[rng.integers(len(overloaded))]) if trial % 2 == 0 else int(np.argmax(per))
            else:
                seed = int(np.argmax(per))
            Gf = base @ base.T
            np.fill_diagonal(Gf, -2.0)
            sz = int(rng.integers(9, 17))
            patch = _patch_from_worst(Gf, seed, sz, m0, rng)
            amp = 0.10 if (trial % 3 == 0) else 0.055
            y[patch] += amp * rng.standard_normal((len(patch), 4))
            iso = np.argsort(per)[:2]
            y[iso] += 0.03 * rng.standard_normal((2, 4))
            y += 0.002 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, T=0.016, maxiter=140)
            y = _smooth_opt(y, T=0.004, maxiter=120)
        elif mode == 4:
            amp = 0.07 if (trial % 3 == 0) else 0.03
            y = y + amp * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, T=0.020, maxiter=110)
            y = _smooth_opt(y, T=0.005, maxiter=110)
        elif mode == 5:
            top = np.argsort(-per)[:15]
            ws = [int(np.argmax(per))]
            extra = rng.choice(top, size=2, replace=False)
            for e in extra:
                if int(e) not in ws:
                    ws.append(int(e))
            ws = np.unique(np.array(ws))
            if len(rattlers) > 0 and (trial % 2 == 0):
                ws = np.unique(np.concatenate([ws, rattlers[:1]]))[:3]
            keep = np.delete(base, ws, axis=0)
            for w in ws:
                v = _find_void(keep, rng, ns=7000, refine=22)
                y[w] = v + 0.009 * rng.standard_normal(4)
                keep = np.vstack([keep, y[w] / np.linalg.norm(y[w])])
            y += 0.003 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, T=0.013, maxiter=120)
            y = _smooth_opt(y, T=0.003, maxiter=110)
        else:
            y = y + 0.012 * rng.standard_normal(y.shape)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = _smooth_opt(y, T=0.009, maxiter=130)
            y = _smooth_opt(y, T=0.0025, maxiter=130)

        v = _max_cos(y)
        if v < best_val - 1e-12:
            if v < best_val - 3e-6:
                y2, v2 = _lp_polish(y, min(deadline - 6.0, time.time() + 2.0), max_solves=20)
                if v2 < best_val:
                    best_x, best_val = y2, v2
                    cur_x, cur_val = y2.copy(), v2
                    continue
                elif v < best_val:
                    best_x, best_val = y, v
                    cur_x, cur_val = y.copy(), v
                    continue
            else:
                best_x, best_val = y, v
                cur_x, cur_val = y.copy(), v
                continue
        if v < cur_val:
            cur_x, cur_val = y, v
        elif rng.random() < np.exp((cur_val - v) / T_accept):
            cur_x, cur_val = y, v
        # occasional pull back
        if trial % 25 == 0 and cur_val > best_val + 0.004:
            cur_x, cur_val = best_x.copy(), best_val

    # final intensification: smooth at low T then LP
    y = best_x.copy()
    if time.time() < deadline - 12:
        y = _smooth_opt(y, T=0.0022, maxiter=250, gtol=1e-6)
        v = _max_cos(y)
        if v < best_val:
            best_x, best_val = y, v
    if time.time() < deadline - 8:
        y = _smooth_opt(best_x, T=0.0012, maxiter=250, gtol=1e-7)
        v = _max_cos(y)
        if v < best_val:
            best_x, best_val = y, v
    best_x, best_val = _lp_polish(best_x, deadline - 1.0, max_solves=200)
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
