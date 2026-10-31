import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _lp_polish(x, seconds, deadline=1e100):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-14:
            break
        G = (x @ x.T)[iu]
        m = G.max()
        sel = G > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-G[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def _smooth_opt(x0, steps, beta0, beta1, lr0):
    n, d = x0.shape
    x = _norm(x0.copy())
    m1 = np.zeros_like(x)
    v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lrb = np.log(beta1 / beta0) if beta0 > 0 else 0.0
    for k in range(steps):
        beta = beta0 * np.exp(lrb * (k / max(steps - 1, 1)))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mv = G.max()
        W = np.exp(beta * (G - mv))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if S < 1e-30:
            S = 1e-30
        Gr = (W @ x) / S
        dot = (Gr * x).sum(axis=1, keepdims=True)
        Gp = Gr - dot * x
        m1 = b1 * m1 + (1 - b1) * Gp
        v1 = b2 * v1 + (1 - b2) * (Gp * Gp)
        bc1 = 1 - b1 ** (k + 1)
        bc2 = 1 - b2 ** (k + 1)
        mh = m1 / bc1
        vh = v1 / bc2
        step = lr0 * mh / (np.sqrt(vh) + eps)
        # clip extreme steps per row
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 0.05 / (sn + 1e-12))
        # only clip if huge
        step = step * np.where(sn > 0.05, clip, 1.0)
        x = x - step
        x = _norm(x)
    return x


def _greedy_refine(x0, steps=220, lr0=0.012):
    x = _norm(x0.copy())
    lr = lr0
    for k in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        frac = k / max(steps - 1, 1)
        tol = 0.012 * (1 - frac) + 0.0004
        mask = G >= m - tol
        deg = mask.sum(axis=1, keepdims=True)
        Gr = (mask @ x) / np.maximum(deg, 1)
        Gp = Gr - (Gr * x).sum(axis=1, keepdims=True) * x
        nr = np.linalg.norm(Gp, axis=1, keepdims=True) + 1e-12
        active = (deg[:, 0] > 0).astype(float)[:, None]
        x = x - lr * (Gp / nr) * active
        x = _norm(x)
        lr *= 0.996
    return x


def _contacts(x, thr=1e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = G.max()
    mask = G >= m - thr
    cnt = mask.sum(axis=1)
    return m, mask, cnt


def _find_holes(Xkeep, n_hole, rng, n_cand=20000):
    cand = rng.normal(size=(n_cand, 4))
    cand = _norm(cand)
    # chunked to limit memory
    best = np.empty(n_cand)
    cs = 5000
    for a in range(0, n_cand, cs):
        b = min(a + cs, n_cand)
        D = cand[a:b] @ Xkeep.T
        best[a:b] = D.max(axis=1)
    idx = np.argsort(best)
    sel = []
    for i in idx:
        if len(sel) >= n_hole:
            break
        c = cand[i]
        ok = True
        for s in sel:
            if float(c @ s) > 0.92:
                ok = False
                break
        if ok:
            sel.append(c)
    if len(sel) < n_hole:
        for i in idx:
            if len(sel) >= n_hole:
                break
            c = cand[i]
            dup = False
            for s in sel:
                if float(c @ s) > 0.999:
                    dup = True
                    break
            if not dup:
                sel.append(c)
    return np.array(sel)


def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    mid_deadline = t0 + 128.0
    rng = np.random.default_rng(0)
    xb = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (292, 4):
                a = _norm(a)
                v = _max_cos(a)
                if xb is None or v < _max_cos(xb):
                    xb = a.copy()
        except Exception:
            continue
    if xb is None:
        r = rng.normal(size=(292, 4))
        xb = _norm(r)
    best_val = _max_cos(xb)
    x_best = xb.copy()

    sigmas = [0.008, 0.022, 0.055, 0.10]
    it = 0
    # cache contacts of best
    _, _, cnt_best = _contacts(x_best)
    while time.time() < mid_deadline:
        it += 1
        sigma = sigmas[(it - 1) % len(sigmas)]
        # base shake
        x_try = x_best + sigma * rng.normal(size=x_best.shape)
        x_try = _norm(x_try)
        # targeted moves on odd iterations
        if it % 2 == 0:
            try:
                m, mask, cnt = _contacts(x_best)
            except Exception:
                cnt = cnt_best
            order = np.argsort(cnt)  # ascending
            k_rep = 3
            rep_idx = order[:k_rep]
            keep_mask = np.ones(292, dtype=bool)
            keep_mask[rep_idx] = False
            Xkeep = x_best[keep_mask]
            holes = _find_holes(Xkeep, k_rep, rng, n_cand=16000)
            if holes.shape[0] == k_rep:
                x_try[rep_idx] = holes + 0.004 * rng.normal(size=(k_rep, 4))
                x_try = _norm(x_try)
            # unjam overloaded neighbourhood every 4th
            if it % 4 == 0:
                try:
                    worst = int(np.argmax(cnt))
                    neigh = np.where(mask[worst])[0]
                    # second order
                    extra = 0.06
                    x_try[worst] += extra * rng.normal(size=(4,))
                    if len(neigh) > 0:
                        x_try[neigh] += (extra * 0.7) * rng.normal(size=(len(neigh), 4))
                    x_try = _norm(x_try)
                except Exception:
                    pass
        # annealed smooth optimisation
        if sigma > 0.05:
            steps, b0, b1, lr = 850, 18.0, 450.0, 0.004
        elif sigma > 0.02:
            steps, b0, b1, lr = 700, 20.0, 420.0, 0.0035
        else:
            steps, b0, b1, lr = 550, 25.0, 380.0, 0.003
        # time guard: skip if too little time for this iter
        if deadline - time.time() < 8:
            break
        try:
            x_opt = _smooth_opt(x_try, steps, b0, b1, lr)
            x_opt = _greedy_refine(x_opt, steps=180, lr0=0.010)
        except Exception:
            continue
        try:
            v = _max_cos(x_opt)
        except Exception:
            continue
        # promising? refine with short LP
        if v < best_val - 2e-6:
            # clearly better basin, polish it
            remain = min(6.0, max(1.0, deadline - time.time() - 25.0))
            if remain < 1.0:
                x_best, best_val = x_opt, v
                _, _, cnt_best = _contacts(x_best)
                continue
            try:
                x_pol = _lp_polish(x_opt, seconds=remain, deadline=deadline - 22.0)
                vp = _max_cos(x_pol)
            except Exception:
                x_pol, vp = x_opt, v
            if vp < best_val:
                x_best, best_val = x_pol, vp
                try:
                    _, _, cnt_best = _contacts(x_best)
                except Exception:
                    pass
        elif v < best_val + 0.0012 and (it % 5 == 0):
            # occasional gamble: polish near-miss to see true optimum
            remain = min(2.5, max(0.8, deadline - time.time() - 25.0))
            if remain >= 0.8 and time.time() < mid_deadline - 10:
                try:
                    x_pol = _lp_polish(x_opt, seconds=remain, deadline=deadline - 22.0)
                    vp = _max_cos(x_pol)
                    if vp < best_val:
                        x_best, best_val = x_pol, vp
                        try:
                            _, _, cnt_best = _contacts(x_best)
                        except Exception:
                            pass
                except Exception:
                    pass
        # small time check
        if time.time() > mid_deadline:
            break

    # final polish of best with remaining time
    rem = deadline - time.time()
    if rem > 3:
        try:
            x_best = _lp_polish(x_best, seconds=min(rem - 1.0, 40.0), deadline=deadline - 1.0)
            best_val = _max_cos(x_best)
        except Exception:
            pass
    x_best = _norm(x_best)
    # safety: remove NaN
    x_best = np.nan_to_num(x_best, nan=1.0, posinf=1.0, neginf=-1.0)
    x_best = _norm(x_best)
    return x_best
