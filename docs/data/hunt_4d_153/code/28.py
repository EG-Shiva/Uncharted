import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, deadline, max_solves=100):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    solves = 0
    while time.time() < deadline and radius > 1e-14 and solves < max_solves:
        G = x @ x.T
        c = G[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
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
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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


def _smooth_anneal(x, steps, T0, T1, lrfac):
    n, d = x.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -1.0)
        m = G.max()
        W = np.exp(np.clip((G - m) / T, -50, 0))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        row = W.sum(axis=1, keepdims=True) + 1e-12
        gt /= row
        lr = lrfac * T
        x = x - lr * gt
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _sym_optimize(J, K, m_vec, a0, t0_, p0, steps, T0, T1, lr0):
    a = a0.copy(); tt = t0_.copy(); pp = p0.copy()
    ma = np.zeros(J); va = np.zeros(J)
    mt = np.zeros(J); vt = np.zeros(J)
    mp = np.zeros(J); vp = np.zeros(J)
    b1, b2, eps = 0.9, 0.999, 1e-8
    base = 2.0 * np.pi * np.arange(K) / K
    mv = m_vec.astype(float)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 * (T1 / T0) ** frac
        # expand
        r = 1.0 / (1.0 + np.exp(-a))
        r = np.clip(r, 0.05, 0.95)
        ss = np.sqrt(np.maximum(1e-12, 1.0 - r * r))
        A = tt[:, None] + base[None, :]
        B = pp[:, None] + mv[:, None] * base[None, :]
        cA = np.cos(A); sA = np.sin(A); cB = np.cos(B); sB = np.sin(B)
        N = J * K
        X = np.empty((N, 4))
        X[:, 0] = (r[:, None] * cA).ravel()
        X[:, 1] = (r[:, None] * sA).ravel()
        X[:, 2] = (ss[:, None] * cB).ravel()
        X[:, 3] = (ss[:, None] * sB).ravel()
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = G.max()
        E = np.exp(np.clip((G - gm) / T, -50, 0))
        np.fill_diagonal(E, 0.0)
        S = E.sum() + 1e-20
        GX = (E @ X) / S
        g = GX.reshape(J, K, 4)
        g0 = g[:, :, 0]; g1 = g[:, :, 1]; g2 = g[:, :, 2]; g3 = g[:, :, 3]
        dLdt = r * np.sum(-g0 * sA + g1 * cA, axis=1)
        dLdp = ss * np.sum(-g2 * sB + g3 * cB, axis=1)
        sum1 = np.sum(g0 * cA + g1 * sA, axis=1)
        sum2 = np.sum(g2 * cB + g3 * sB, axis=1)
        dLdr = sum1 - (r / np.maximum(ss, 1e-6)) * sum2
        dLda = dLdr * r * (1.0 - r)
        lr_eff = lr0 * (0.3 + 0.7 * T / T0)
        # adam updates
        ma = b1 * ma + (1 - b1) * dLda
        va = b2 * va + (1 - b2) * dLda * dLda
        mt = b1 * mt + (1 - b1) * dLdt
        vt = b2 * vt + (1 - b2) * dLdt * dLdt
        mp = b1 * mp + (1 - b1) * dLdp
        vp = b2 * vp + (1 - b2) * dLdp * dLdp
        bc1 = 1.0 - b1 ** (s + 1); bc2 = 1.0 - b2 ** (s + 1)
        a -= lr_eff * (ma / bc1) / (np.sqrt(va / bc2) + eps)
        tt -= lr_eff * (mt / bc1) / (np.sqrt(vt / bc2) + eps)
        pp -= lr_eff * (mp / bc1) / (np.sqrt(vp / bc2) + eps)
        a = np.clip(a, -3.0, 3.0)
    # final expand
    r = 1.0 / (1.0 + np.exp(-a))
    r = np.clip(r, 0.05, 0.95)
    ss = np.sqrt(np.maximum(1e-12, 1.0 - r * r))
    A = tt[:, None] + base[None, :]
    B = pp[:, None] + mv[:, None] * base[None, :]
    X = np.empty((J * K, 4))
    X[:, 0] = (r[:, None] * np.cos(A)).ravel()
    X[:, 1] = (r[:, None] * np.sin(A)).ravel()
    X[:, 2] = (ss[:, None] * np.cos(B)).ravel()
    X[:, 3] = (ss[:, None] * np.sin(B)).ravel()
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, float(_max_cos(X))


def solve():
    t0 = time.time()
    deadline = t0 + 175.0
    sym_end = t0 + 62.0
    explore_end = t0 + 148.0
    rng = np.random.default_rng(7)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands0 = []
    if xb is not None and xb.shape == (153, 4):
        cands0.append(xb / np.linalg.norm(xb, axis=1, keepdims=True))
    if xr is not None and xr.shape == (153, 4):
        cands0.append(xr / np.linalg.norm(xr, axis=1, keepdims=True))
    if not cands0:
        R = rng.standard_normal(size=(153, 4))
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        cands0.append(R)
    vals0 = [_max_cos(c) for c in cands0]
    best_x = cands0[int(np.argmin(vals0))].copy()
    best_val = float(min(vals0))

    # ---- symmetric torus search ----
    sym_cands = []
    patterns = []
    # 9x17 family (primary)
    patterns.append((9, 17, np.full(9, 8, dtype=int)))
    patterns.append((9, 17, np.full(9, 1, dtype=int)))
    patterns.append((9, 17, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)))
    patterns.append((9, 17, np.array([8, 9, 8, 9, 8, 9, 8, 9, 8], dtype=int)))
    patterns.append((9, 17, np.array([8, 8, 8, 4, 4, 12, 12, 1, 1], dtype=int)))
    for _ in range(7):
        patterns.append((9, 17, rng.integers(1, 17, size=9)))
    # 17x9 family (transposed)
    allowed = np.array([1, 2, 4, 5, 7, 8])
    patterns.append((17, 9, np.full(17, 4, dtype=int)))
    patterns.append((17, 9, np.full(17, 1, dtype=int)))
    for _ in range(4):
        patterns.append((17, 9, allowed[rng.integers(0, len(allowed), size=17)]))

    for (J, K, mv) in patterns:
        if time.time() > sym_end:
            break
        for restart in range(2):
            if time.time() > sym_end:
                break
            r_init = rng.uniform(0.2, 0.8, size=J)
            a0 = np.log(r_init / (1.0 - r_init))
            tt0 = rng.uniform(0, 2 * np.pi, size=J)
            pp0 = rng.uniform(0, 2 * np.pi, size=J)
            steps = 1100 if J == 9 else 900
            try:
                Xsym, vsym = _sym_optimize(J, K, mv, a0, tt0, pp0,
                                           steps=steps, T0=0.02, T1=0.0012, lr0=0.035)
            except Exception:
                continue
            sym_cands.append((vsym, Xsym.copy()))
            if vsym < best_val - 1e-12:
                # keep as candidate; don't overwrite best yet (needs release)
                pass
    sym_cands.sort(key=lambda z: z[0])
    # release symmetry: unrestricted anneal + short LP for top sym candidates
    for idx in range(min(3, len(sym_cands))):
        if time.time() > explore_end - 20:
            break
        vs, Xs = sym_cands[idx]
        # only pursue if remotely competitive
        if vs > best_val + 0.025:
            continue
        y = Xs + 0.002 * rng.standard_normal(Xs.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y = _smooth_anneal(y, steps=500, T0=0.012, T1=0.0015, lrfac=1.2)
        y = _smooth_anneal(y, steps=150, T0=0.003, T1=0.001, lrfac=1.0)
        v = _max_cos(y)
        if v < best_val + 0.004:
            y2, v2 = _lp_polish(y, min(deadline - 10, time.time() + 4.0), max_solves=30)
            if v2 < v:
                y, v = y2, v2
        if v < best_val:
            best_x, best_val = y, v

    # ---- small-kick topology-retaining walk around best ----
    cur_x, cur_val = best_x.copy(), best_val
    trial = 0
    while time.time() < explore_end:
        trial += 1
        if trial % 12 == 0:
            base = best_x
        else:
            base = cur_x
        if cur_val > best_val + 0.010 and trial % 5 == 0:
            base = best_x
        y = base.copy()
        mode = trial % 4
        if mode == 0:
            y = y + 0.020 * rng.standard_normal(y.shape)
        elif mode == 1:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            idxw = np.argsort(-per)[:8]
            y[idxw] = y[idxw] + 0.05 * rng.standard_normal((8, 4))
            y = y + 0.004 * rng.standard_normal(y.shape)
        elif mode == 2:
            y = y + 0.035 * rng.standard_normal(y.shape)
        else:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            idxw = np.argsort(-per)[:4]
            idxc = np.argsort(per)[:3]
            y[idxw] = y[idxw] + 0.04 * rng.standard_normal((4, 4))
            y[idxc] = y[idxc] + 0.02 * rng.standard_normal((3, 4))
            y = y + 0.003 * rng.standard_normal(y.shape)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        y = _smooth_anneal(y, steps=380, T0=0.012, T1=0.0018, lrfac=1.2)
        y = _smooth_anneal(y, steps=100, T0=0.003, T1=0.0012, lrfac=1.0)
        v = _max_cos(y)
        if v < best_val - 1e-12:
            if v < best_val - 4e-6 and time.time() < deadline - 10:
                y2, v2 = _lp_polish(y, min(deadline - 8, time.time() + 2.0), max_solves=20)
                if v2 < best_val:
                    best_x, best_val = y2, v2
                else:
                    best_x, best_val = y, v
            else:
                best_x, best_val = y, v
            cur_x, cur_val = best_x.copy(), best_val
            continue
        if v < cur_val:
            cur_x, cur_val = y, v
        else:
            if rng.random() < np.exp((cur_val - v) / 0.0012):
                cur_x, cur_val = y, v
        if trial % 7 == 0 and v < best_val + 0.002 and time.time() < explore_end - 4:
            y2, v2 = _lp_polish(y, time.time() + 1.0, max_solves=10)
            if v2 < best_val - 1e-12:
                best_x, best_val = y2, v2
                cur_x, cur_val = y2.copy(), v2
            elif v2 < cur_val:
                cur_x, cur_val = y2, v2

    if time.time() < deadline - 0.5:
        best_x, best_val = _lp_polish(best_x, deadline - 0.4, max_solves=200)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
