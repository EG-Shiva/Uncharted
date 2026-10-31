import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    SEARCH_END = t0 + 132.0
    rng = np.random.RandomState(20261031)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(168, 4)
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr)
        xb = _norm(xb)
        if _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        xb = _norm(xb)
    if xb.shape != (168, 4):
        tmp = rng.randn(168, 4)
        xb = _norm(tmp)
    xb = _norm(xb)
    best_val = _max_cos(xb)
    x_best = xb.copy()

    # ---------- helpers (closure over rng) ----------
    def refine(x_init, steps, mu=0.9, beta0=25.0, beta1=500.0, lr0=0.014, lr1=0.0007):
        x = x_init.copy()
        vel = np.zeros_like(x)
        for s in range(steps):
            if time.time() > HARD_END - 2.0:
                break
            frac = s / max(1, steps - 1)
            beta = beta0 * (beta1 / beta0) ** frac
            lr = lr0 * (1.0 - frac) + lr1 * frac
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -20.0, 0.0))
            np.fill_diagonal(E, 0.0)
            rs = np.sum(E, axis=1, keepdims=True) + 1e-12
            grad = (E @ x) / rs
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x /= (np.linalg.norm(x, axis=1, keepdims=True) + 1e-30)
        return x

    def lp_polish(x, deadline):
        x = _norm(x.copy())
        n, d = x.shape
        best = _max_cos(x)
        radius = 1.5e-3
        iu = np.triu_indices(n, 1)
        while time.time() < deadline and radius > 1e-14:
            C = (x @ x.T)[iu]
            m = float(np.max(C))
            sel = C > m - max(20.0 * radius, 1e-9)
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
                res = linprog(obj, A_ub=A_ub, b_ub=-C[sel], A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                break
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= (np.linalg.norm(y, axis=1, keepdims=True) + 1e-30)
            v = _max_cos(y)
            if v < best - 1e-14:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return x, best

    def contact_load(X):
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        deg = np.sum(G > (m - 1e-3), axis=1).astype(float)
        excess = np.sum(np.maximum(0.0, G - 0.80), axis=1)
        return deg + 8.0 * excess, m

    def opt_hole(X_keep, n_sample=6000, iters=90):
        S = rng.randn(n_sample, 4)
        S /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-30)
        D = S @ X_keep.T
        worst = D.max(axis=1)
        s = S[np.argmin(worst)].copy()
        b0, b1 = 30.0, 250.0
        lr0, lr1 = 0.08, 0.005
        for it in range(iters):
            f = it / max(1, iters - 1)
            beta = b0 * (b1 / b0) ** f
            lr = lr0 * (1.0 - f) + lr1 * f
            dots = X_keep @ s
            cm = float(np.max(dots))
            w = np.exp(np.clip(beta * (dots - cm), -20.0, 0.0))
            w /= (np.sum(w) + 1e-30)
            g = w @ X_keep
            gt = g - np.dot(g, s) * s
            s = s - lr * gt
            ns = np.linalg.norm(s)
            if ns < 1e-12:
                s = rng.randn(4)
                ns = np.linalg.norm(s)
            s /= ns
        return s

    # initial squeeze of incumbent
    try:
        xb2, vb2 = lp_polish(x_best, min(t0 + 10.0, HARD_END - 20.0))
        if vb2 < best_val:
            x_best, best_val = xb2, vb2
    except Exception:
        pass

    cycle = 0
    # alternate overshoot / undershoot
    while time.time() < SEARCH_END:
        cycle += 1
        if time.time() > SEARCH_END - 2.0:
            break
        if cycle % 2 == 1:
            # ---- OVERSHOOT branch ----
            k_add = [1, 2, 3][((cycle // 2)) % 3]
            X_enl = x_best.copy()
            for _ in range(k_add):
                if time.time() > SEARCH_END:
                    break
                s = opt_hole(X_enl, n_sample=6000, iters=90)
                X_enl = np.vstack([X_enl, s[None, :]])
            # refine enlarged code hard (worse max is expected)
            steps_big = 900 if k_add <= 2 else 1050
            # shorten if low on time
            if SEARCH_END - time.time() < 20:
                steps_big = min(steps_big, 500)
            X_enl = refine(X_enl, steps_big, beta0=25.0, beta1=450.0, lr0=0.013, lr1=0.0008)
            # delete k_add points with largest contact load
            loads, _ = contact_load(X_enl)
            del_idx = np.argsort(-loads)[:k_add]
            keep = np.ones(X_enl.shape[0], dtype=bool)
            keep[del_idx] = False
            X_new = _norm(X_enl[keep])
            X_new = refine(X_new, 650, beta0=25.0, beta1=550.0, lr0=0.012, lr1=0.0006)
            v = _max_cos(X_new)
            # opportunistic LP if competitive
            if v < best_val + 0.004 and time.time() < SEARCH_END - 3.0:
                try:
                    Xlp, vlp = lp_polish(X_new, min(time.time() + 4.0, SEARCH_END + 6.0))
                    if vlp < v:
                        X_new, v = Xlp, vlp
                except Exception:
                    pass
            if v < best_val - 1e-12:
                x_best, best_val = X_new, v
        else:
            # ---- UNDERSHOOT branch ----
            del_n = [8, 13, 18][((cycle // 2)) % 3]
            loads, _ = contact_load(x_best)
            del_idx = np.argsort(-loads)[:del_n]
            keep = np.ones(168, dtype=bool)
            keep[del_idx] = False
            cur = _norm(x_best[keep].copy())
            # polish small code to its own optimum
            cur = refine(cur, 600, beta0=25.0, beta1=450.0, lr0=0.013, lr1=0.0008)
            # grow back to 168
            batch = 2 + (cycle % 3)  # 2..4 insertions per short refinement
            since = 0
            while cur.shape[0] < 168:
                if time.time() > SEARCH_END:
                    break
                # faster hole search during growth
                s = opt_hole(cur, n_sample=5000, iters=70)
                cur = np.vstack([cur, s[None, :]])
                since += 1
                is_last = (cur.shape[0] == 168)
                if since >= batch or is_last:
                    st = 700 if is_last else 280
                    cur = refine(cur, st, beta0=25.0, beta1=500.0, lr0=0.012, lr1=0.0007)
                    since = 0
            cur = _norm(cur)
            v = _max_cos(cur)
            if v < best_val + 0.004 and time.time() < SEARCH_END - 3.0:
                try:
                    Xlp, vlp = lp_polish(cur, min(time.time() + 4.0, SEARCH_END + 6.0))
                    if vlp < v:
                        cur, v = Xlp, vlp
                except Exception:
                    pass
            if v < best_val - 1e-12:
                x_best, best_val = cur, v
        # safety: keep shape
        if x_best.shape != (168, 4):
            x_best = _norm(x_best[:168] if x_best.shape[0] >= 168 else np.vstack([x_best, rng.randn(168 - x_best.shape[0], 4)]))

    # final intensive LSE + LP from best descendant
    if time.time() < HARD_END - 12.0:
        x_best = refine(x_best, 2200, beta0=20.0, beta1=700.0, lr0=0.010, lr1=0.0005)
        bv = _max_cos(x_best)
        if bv < best_val:
            best_val = bv
    if time.time() < HARD_END - 2.0:
        try:
            Xlp, vlp = lp_polish(x_best, HARD_END - 1.0)
            if vlp < best_val:
                x_best, best_val = Xlp, vlp
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
