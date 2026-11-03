import time
import numpy as np

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_viol(x0, T, steps, lr, check=200):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        G = x @ x.T
        # violation
        D = G - T
        np.fill_diagonal(D, 0.0)
        np.maximum(D, 0.0, out=D)
        grad = 2.0 * (D @ x)
        # tangent
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
                # early exit if feasible with margin
                if best_val < T - 1e-4:
                    break
    return best_x, best_val

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
        W = np.exp(np.clip(K * (G - gmax), -60, 0.0))
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

def _lp_polish(x, deadline, r0=2e-3, cap=3000):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        margin = max(20.0 * radius, 8e-5)
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

def solve():
    import time as _time
    import numpy as _np
    t0 = _time.time()
    T_END = t0 + 172.0
    ANNEAL_END = t0 + 148.0
    rng = _np.random.default_rng(0)
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
    elites = [x_best.copy()]
    cur = x_best.copy()
    cur_val = best
    # initial squeeze of incumbent plateau with violation target
    try:
        Tt = best - 1.5e-4
        xv, vv = _adam_viol(cur, Tt, 500, 0.01)
        if vv < cur_val:
            cur, cur_val = xv, vv
        if vv < best:
            best = vv
            x_best = xv.copy()
            elites.append(xv.copy())
        xs, vs = _adam_sm(cur, 400, 60.0, 2500.0, 0.008)
        if vs < cur_val:
            cur, cur_val = xs, vs
        if vs < best:
            best = vs
            x_best = xs.copy()
            elites[0] = xs.copy()
    except Exception:
        pass
    cycle = 0
    while _time.time() < ANNEAL_END:
        cycle += 1
        rem = ANNEAL_END - _time.time()
        if rem < 3.0:
            break
        r = rng.random()
        # choose base
        if r < 0.45:
            base = cur.copy()
        elif r < 0.7:
            base = x_best.copy()
        elif r < 0.85 and len(elites) > 1:
            base = elites[int(rng.integers(len(elites)))].copy()
        else:
            base = rng.normal(size=(n, d))
            base /= _np.linalg.norm(base, axis=1, keepdims=True)
        x = base
        # diagnose worst structure
        try:
            G = x @ x.T
            _np.fill_diagonal(G, -2.0)
            gmax = float(G.max())
            prow = G.max(axis=1)
            order = _np.argsort(-prow)
        except Exception:
            order = rng.permutation(n)
            gmax = 0.75
            prow = _np.full(n, gmax)
        mv = int(rng.integers(5))
        if mv == 0:
            # void reinsert worst k then compress
            k = int(rng.integers(4, 18))
            worst = order[:k].copy()
            # sample candidates
            nc = 20000
            Z = rng.normal(size=(nc, 6))
            Z /= _np.linalg.norm(Z, axis=1, keepdims=True)
            S = Z @ x.T
            for rr in worst:
                S[:, rr] = -2.0
            mm = S.max(axis=1)
            # sequential greedy assign
            for rr in worst:
                bi = int(_np.argmin(mm))
                # accept if better than current worst for rr
                if mm[bi] < prow[rr] + 0.02:
                    x[rr] = Z[bi]
                    # update mm approx
                    mm[bi] = 2.0
                else:
                    x[rr] = Z[bi]
                    mm[bi] = 2.0
            x += 0.004 * rng.normal(size=(n, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            Tt = min(best - 8e-5, float(gmax) - 3e-4)
            steps_v, lr_v = 550, 0.012
            steps_s, K0, K1, lr_s = 400, 80.0, 2500.0, 0.008
        elif mv == 1:
            # cap rotation + kick
            u = rng.normal(size=(6,))
            u /= _np.linalg.norm(u)
            dots = x @ u
            kcap = int(rng.integers(15, 45))
            idx = _np.argsort(-dots)[:kcap]
            A = rng.normal(size=(6, 6))
            A = A - A.T
            th = float(rng.uniform(0.1, 0.4)) * (1 if rng.random() < 0.5 else -1)
            x[idx] = x[idx] + th * (A @ x[idx].T).T + 0.01 * rng.normal(size=(len(idx), 6))
            # kick global worst
            w = order[:8]
            x[w] += 0.03 * rng.normal(size=(len(w), 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            Tt = min(best - 8e-5, float(gmax) - 2e-4)
            steps_v, lr_v = 500, 0.01
            steps_s, K0, K1, lr_s = 400, 50.0, 2000.0, 0.008
        elif mv == 2:
            # large Gaussian kick to change topology
            sig = float(rng.uniform(0.05, 0.18))
            x = x + sig * rng.normal(size=(n, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            Tt = best - 1e-4
            steps_v, lr_v = 600, 0.012
            steps_s, K0, K1, lr_s = 450, 60.0, 2200.0, 0.007
        elif mv == 3:
            # antipodal enforcement jump: pair up and enforce +/- then release
            perm = rng.permutation(n)
            # force antipodal pairs for half then perturb
            for ii in range(0, n, 2):
                a = perm[ii]; b = perm[ii + 1]
                if rng.random() < 0.3:
                    # make b antipode of a with noise
                    x[b] = -x[a] + 0.02 * rng.normal(size=(6,))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            x += 0.008 * rng.normal(size=(n, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            Tt = min(best - 5e-5, float(gmax) - 1e-4)
            steps_v, lr_v = 550, 0.01
            steps_s, K0, K1, lr_s = 400, 40.0, 1500.0, 0.009
        else:
            # from-scratch-ish partial resample: replace random block with fresh spread points
            k = int(rng.integers(20, 50))
            ib = rng.choice(n, size=k, replace=False)
            Z = rng.normal(size=(k, 6))
            Z /= _np.linalg.norm(Z, axis=1, keepdims=True)
            x[ib] = Z
            x += 0.005 * rng.normal(size=(n, 6))
            x /= _np.linalg.norm(x, axis=1, keepdims=True)
            Tt = best - 1e-4
            steps_v, lr_v = 600, 0.012
            steps_s, K0, K1, lr_s = 400, 70.0, 2500.0, 0.008
        # anneal: violation then softmax
        try:
            xv, vv = _adam_viol(x, float(Tt), steps_v, lr_v)
        except Exception:
            continue
        try:
            xs, vs = _adam_sm(xv, steps_s, K0, K1, lr_s)
        except Exception:
            xs, vs = xv, vv
        # acceptance: basin hopping (accept if better than cur or with prob)
        if vs < cur_val - 1e-12:
            cur, cur_val = xs, vs
        elif rng.random() < 0.12 and vs < cur_val + 8e-4:
            cur, cur_val = xs, vs
        if vs < best - 1e-12:
            best = vs
            x_best = xs.copy()
            elites.append(xs.copy())
            if len(elites) > 8:
                # keep diverse: drop oldest non-best
                elites.pop(1)
        elif vv < best - 1e-12:
            best = vv
            x_best = xv.copy()
            elites.append(xv.copy())
            if len(elites) > 8:
                elites.pop(1)
        # keep elites bounded, occasionally refresh cur to best
        if cycle % 12 == 0:
            cur = x_best.copy()
            cur_val = best
    # final squeeze: aggressive softmax + LP if time
    try:
        if _time.time() < T_END - 6:
            xs, vs = _adam_sm(x_best, 700, 100.0, 4000.0, 0.006)
            if vs < best:
                best = vs
                x_best = xs.copy()
    except Exception:
        pass
    try:
        if _time.time() < T_END - 12:
            x_best = _lp_polish(x_best, T_END - 1.5, r0=1.5e-3, cap=3000)
            best = _max_cos(x_best)
    except Exception:
        pass
    x_best = x_best / _np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
