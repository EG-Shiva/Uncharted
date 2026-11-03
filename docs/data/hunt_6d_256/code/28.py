import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_sm(x0, steps, K0, K1, lr, check=200):
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
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if t % check == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2e-3, cap=3500, margin_scale=20.0, floor=8e-5):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        margin = max(margin_scale * radius, floor)
        sel = c > (m - margin)
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            p = cap
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
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

def _sign_mat_full():
    M = np.ones((64, 6))
    for i in range(64):
        for k in range(6):
            if (i >> k) & 1:
                M[i, k] = -1.0
    return M

def _sign_mat_even():
    rows = []
    for i in range(64):
        b = [(i >> k) & 1 for k in range(6)]
        if sum(b) % 2 == 0:
            r = np.ones(6)
            for k in range(6):
                if b[k]:
                    r[k] = -1.0
            rows.append(r)
    return np.array(rows)  # 32x6

def _sym_adam(rng, sign_mat, n_proto, steps, K0, K1, lr):
    d = 6
    M = sign_mat.shape[0]
    P = rng.normal(size=(n_proto, d))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    # avoid near-zero coords (orbit collapse)
    P = np.where(np.abs(P) < 0.08, 0.08 * np.sign(P + 1e-9), P)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    m = np.zeros_like(P); v = np.zeros_like(P)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = 1e9
    best_X = None
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        X = (P[:, None, :] * sign_mat[None, :, :]).reshape(-1, d)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        # track
        if gmax < best_val:
            best_val = gmax
            best_X = X.copy()
        W = np.exp(np.clip(K * (G - gmax), -60, 0.0))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad_X = (W @ X) / S
        grad_P = np.zeros_like(P)
        for a in range(n_proto):
            blk = grad_X[a * M:(a + 1) * M]
            grad_P[a] = (blk * sign_mat).sum(axis=0) / M
        dot = np.sum(grad_P * P, axis=1, keepdims=True)
        gt = grad_P - dot * P
        b1p *= b1; b2p *= b2
        m *= b1; m += (1 - b1) * gt
        v *= b2; v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p); vhat = v / (1 - b2p)
        st = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(st * P, axis=1, keepdims=True)
        st = st - sdot * P
        P = P - st
        P /= np.linalg.norm(P, axis=1, keepdims=True)
    return best_X, best_val

def _procrustes_rotate(src, dst):
    # rotate src to align to dst: src @ R ~= dst
    Mmat = src.T @ dst
    try:
        U, _, Vt = np.linalg.svd(Mmat)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return src @ R
    except Exception:
        return src

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_END = t0 + 172.0
    ANNEAL_END = t0 + 138.0
    SYM_END = t0 + 32.0
    rng = _np.random.default_rng(0)
    x_best = None
    best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = _np.load(fn)
            a = _np.asarray(a, dtype=float)
            if a.shape == (256, 6) and _np.all(_np.isfinite(a)):
                a = a / _np.linalg.norm(a, axis=1, keepdims=True)
                G = a @ a.T
                _np.fill_diagonal(G, -2.0)
                vv = float(G.max())
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
    elites = [x_best.copy()]
    # --- symmetric jammed seeding ---
    try:
        SF = _sign_mat_full()
        SE = _sign_mat_even()
        cfgs = [(SF, 4, 700, 30.0, 600.0, 0.02), (SE, 8, 700, 30.0, 600.0, 0.02),
                (SF, 4, 600, 80.0, 1200.0, 0.015)]
        for (SM, np_, st_, k0_, k1_, lr_) in cfgs:
            if _time.time() > SYM_END - 3:
                break
            rs = _np.random.default_rng(int(rng.integers(1e9)))
            Xb, vb = _sym_adam(rs, SM, np_, st_, k0_, k1_, lr_)
            if Xb is None or not _np.all(_np.isfinite(Xb)):
                continue
            if vb > 0.85:  # collapsed / useless
                continue
            # break symmetry lightly and polish a bit
            Xn = Xb + 0.006 * rs.normal(size=Xb.shape)
            Xn /= _np.linalg.norm(Xn, axis=1, keepdims=True)
            rem = min(400, st_ // 2)
            Xp, vp = _adam_sm(Xn, rem, 40.0, 900.0, 0.008, check=200)
            elites.append(Xp.copy())
            if vp < best:
                best = vp
                x_best = Xp.copy()
    except Exception:
        pass
    # light LP squeeze of incumbent
    try:
        if _time.time() < SYM_END:
            x_best = _lp_polish(x_best, min(_time.time() + 4.0, SYM_END), r0=1.5e-3)
            best = _max_cos(x_best)
            elites[0] = x_best.copy()
    except Exception:
        pass
    cur = x_best.copy()
    cur_val = best
    cycle = 0
    while _time.time() < ANNEAL_END:
        cycle += 1
        rem = ANNEAL_END - _time.time()
        if rem < 2.0:
            break
        # pick base
        r = rng.random()
        if r < 0.60:
            base = cur.copy()
        elif r < 0.82:
            base = x_best.copy()
        else:
            base = elites[int(rng.integers(len(elites)))].copy()
        x = base
        G = x @ x.T
        _np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        prow = G.max(axis=1)
        order_row = _np.argsort(-prow)
        cnt = _np.sum(G > gmax - 1e-3, axis=1)
        order_cnt = _np.argsort(-cnt)
        mv = int(rng.integers(6))
        if mv == 0:
            # worst-pair split along tangent
            fi = int(_np.argmax(G)); ai = fi // n; aj = fi % n
            alpha = 0.05 + 0.05 * rng.random()
            gia = x[aj] - float(x[ai] @ x[aj]) * x[ai]
            gjb = x[ai] - float(x[ai] @ x[aj]) * x[aj]
            gia /= (_np.linalg.norm(gia) + 1e-12)
            gjb /= (_np.linalg.norm(gjb) + 1e-12)
            x[ai] = x[ai] - alpha * gia + 0.012 * rng.normal(size=(6,))
            x[aj] = x[aj] - alpha * gjb + 0.012 * rng.normal(size=(6,))
            hub = order_cnt[:6]
            x[hub] += 0.02 * rng.normal(size=(len(hub), 6))
            steps, K0, K1, lr = 500, 60.0, 2000.0, 0.008
        elif mv == 1:
            # coherent cap rotation
            u = rng.normal(size=(6,)); u /= _np.linalg.norm(u)
            dots = x @ u
            kcap = int(rng.integers(12, 30))
            idx = _np.argsort(-dots)[:kcap]
            A = rng.normal(size=(6, 6)); A = A - A.T
            th = float(rng.uniform(0.08, 0.32)) * (1 if rng.random() < 0.5 else -1)
            x[idx] = x[idx] + th * (A @ x[idx].T).T + 0.008 * rng.normal(size=(len(idx), 6))
            w = order_row[:6]
            x[w] += 0.012 * rng.normal(size=(6, 6))
            steps, K0, K1, lr = 500, 40.0, 1500.0, 0.009
        elif mv == 2:
            # void reinsertion of worst points
            nw = 3
            worst = order_row[:nw]
            Z = rng.normal(size=(12000, 6))
            Z /= _np.linalg.norm(Z, axis=1, keepdims=True)
            S = Z @ x.T
            for rr in worst:
                S[:, rr] = -2.0
            mm = S.max(axis=1)
            for rr in worst:
                bi = int(_np.argmin(mm))
                if mm[bi] < prow[rr] - 1e-6:
                    x[rr] = Z[bi]
                    mm[bi] = 2.0
            x += 0.006 * rng.normal(size=(n, 6))
            steps, K0, K1, lr = 550, 80.0, 2200.0, 0.007
        elif mv == 3:
            # rotation-aligned block transplant (crossover)
            donor = elites[int(rng.integers(len(elites)))]
            drot = _procrustes_rotate(donor, x)
            k = int(rng.integers(12, 33))
            ib = rng.choice(n, size=k, replace=False)
            ia = rng.choice(n, size=k, replace=False)
            x[ib] = drot[ia] + 0.004 * rng.normal(size=(k, 6))
            x += 0.004 * rng.normal(size=(n, 6))
            steps, K0, K1, lr = 550, 50.0, 1600.0, 0.008
        elif mv == 4:
            # sign-reflection jump (symmetry basin hop)
            k = int(rng.integers(10, 25))
            pick = rng.choice(n, size=k, replace=False)
            for pp in pick:
                c = int(rng.integers(6))
                x[pp, c] *= -1.0
            x[pick] += 0.01 * rng.normal(size=(k, 6))
            x += 0.004 * rng.normal(size=(n, 6))
            steps, K0, K1, lr = 550, 30.0, 1000.0, 0.009
        else:
            # large kick
            k = int(rng.integers(20, 34))
            pick = rng.choice(n, size=k, replace=False)
            x[pick] += 0.35 * rng.normal(size=(k, 6))
            x += 0.008 * rng.normal(size=(n, 6))
            steps, K0, K1, lr = 600, 30.0, 900.0, 0.011
        if rem < 25:
            steps = min(steps, 320)
        x /= _np.linalg.norm(x, axis=1, keepdims=True)
        xb, vb = _adam_sm(x, steps, K0, K1, lr)
        if vb < best - 1e-12 and _np.all(_np.isfinite(xb)):
            x_best = xb; best = vb
            cur = xb.copy(); cur_val = vb
            elites.append(xb.copy())
            if len(elites) > 6:
                elites.pop(1)
            try:
                if _time.time() < ANNEAL_END - 4:
                    xs = _lp_polish(x_best, min(_time.time() + 2.0, ANNEAL_END), r0=1.2e-3)
                    vs = _max_cos(xs)
                    if vs < best:
                        x_best = xs; best = vs
                        cur = xs.copy(); cur_val = vs
                        elites[-1] = xs.copy()
            except Exception:
                pass
        elif vb < cur_val - 1e-12:
            cur = xb.copy(); cur_val = vb
        elif vb < cur_val + 4e-4 and rng.random() < 0.12:
            cur = xb.copy(); cur_val = vb
    # final polish
    try:
        if _time.time() < T_END - 4:
            xb, vb = _adam_sm(x_best, 700, 300.0, 3500.0, 0.004)
            if vb < best:
                x_best, best = xb, vb
    except Exception:
        pass
    try:
        if _time.time() < T_END - 1.5:
            x_best = _lp_polish(x_best, T_END - 0.8, r0=1e-3)
    except Exception:
        pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    x_best = _np.where(_np.isfinite(x_best), x_best, 0.0)
    nr = _np.linalg.norm(x_best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    x_best = x_best / nr
    return x_best
