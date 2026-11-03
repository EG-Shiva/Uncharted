import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _adam_quench(X, steps, T0, T1, lr, lam_iso=0.05, deadline=1e18):
    n, d = X.shape
    X = _norm(X)
    m_ad = np.zeros_like(X)
    v_ad = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0
    b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    eye = np.eye(d) / d
    for t in range(1, steps + 1):
        if (t & 127) == 0 and time.time() > deadline:
            break
        T = T0 * np.exp(logr * (t - 1))
        if T < 7e-4:
            T = 7e-4
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        E = np.exp((G - mv) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ X) / S
        # isotropy / centroid regularizer, decay to 0 in last 25%
        frac = t / steps
        if lam_iso > 0 and frac < 0.75:
            lam = lam_iso * (1.0 - frac / 0.75)
            mu = X.mean(axis=0, keepdims=True)  # 1xd
            C = (X.T @ X) / n
            D = C - eye
            extra = (2.0 * np.repeat(mu, n, axis=0) / n + 4.0 * (X @ D) / n)
            Grad = Grad + lam * extra
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        b1p *= b1
        b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        lrt = lr * (1.0 - 0.6 * frac)
        X = X - lrt * step
        X = _norm(X)
    return X

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _mc(x)
    cur_best = x.copy()
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    itn = 0
    while time.time() - start < seconds and time.time() < deadline:
        if radius < 1e-13 or itn >= 60:
            break
        itn += 1
        c = (x @ x.T)[iu]
        m = float(c.max())
        thr = max(20.0 * radius, 5e-5)
        sel = c > m - thr
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            o = np.argsort(-c[sel])[:3000]
            I, J = I[o], J[o]
            cv = c[sel][o]
        else:
            cv = c[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _mc(y)
        if v < best:
            best = v
            cur_best = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return cur_best

def _hole_greedy(X, rng, C=8000):
    # one pass over worst 12 points: replace by best of C random holes if improves
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rowmax = G.max(axis=1)
    order = np.argsort(-rowmax)[:12]
    cur = float(G.max())
    for vv in order:
        vv = int(vv)
        mask = np.ones(n, dtype=bool)
        mask[vv] = False
        Xt = X[mask]
        cand = _norm(rng.normal(size=(C, d)))
        md = (cand @ Xt.T).max(axis=1)
        bi = int(np.argmin(md))
        nv = float(md[bi])
        if nv < cur - 1e-7:
            X[vv] = cand[bi]
            cur = _mc(X)
    return X

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(7)
    n, d = 154, 10
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _mc(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.normal(size=(n, d)))
        best_val = _mc(best)
    global_best = best.copy()
    global_val = best_val
    cur_exp = best.copy()
    cur_val = best_val

    EXPL_END = DEAD - 30.0
    trial = 0
    while time.time() < EXPL_END:
        trial += 1
        time_left = EXPL_END - time.time()
        if time_left < 1.5:
            break
        r = rng.random()
        Y = cur_exp.copy()
        if r < 0.45:
            sigma = 0.25 + rng.random() * 0.35
            Y = _norm(Y + sigma * rng.normal(size=(n, d)))
            steps = 850
            T0 = 0.035 + rng.random() * 0.025
            T1 = 0.0035
            lr = 0.028
            lam = 0.06
        elif r < 0.75:
            k = int(18 + rng.random() * 22)
            idx = rng.choice(n, size=k, replace=False)
            Y[idx] = _norm(rng.normal(size=(k, d)))
            # small shake on rest
            Y = _norm(Y + 0.04 * rng.normal(size=(n, d)))
            steps = 900
            T0 = 0.04
            T1 = 0.003
            lr = 0.03
            lam = 0.08
        else:
            # shake global best directly (fresh basin)
            sigma = 0.20 + rng.random() * 0.30
            Y = _norm(global_best + sigma * rng.normal(size=(n, d)))
            steps = 800
            T0 = 0.03 + rng.random() * 0.02
            T1 = 0.004
            lr = 0.025
            lam = 0.04
        try:
            Y = _adam_quench(Y, steps=steps, T0=T0, T1=T1, lr=lr, lam_iso=lam, deadline=EXPL_END)
        except Exception:
            continue
        # cheap hole greedy on most trials (every other to save time)
        if trial % 2 == 0:
            try:
                Y = _hole_greedy(Y, rng, C=6000)
                # short low-T snap after hole move
                Y = _adam_quench(Y, steps=180, T0=0.006, T1=0.0012, lr=0.015, lam_iso=0.0, deadline=EXPL_END)
            except Exception:
                pass
        try:
            v = _mc(Y)
        except Exception:
            continue
        # wandering acceptance: allow small uphill vs explorer
        if v < cur_val + 0.006 or v < global_val:
            cur_exp = Y.copy()
            cur_val = v
        if v < global_val - 1e-9:
            global_val = v
            global_best = Y.copy()
        # strong stagnation: restart explorer from global with kick
        if trial % 25 == 0 and time.time() < EXPL_END - 5:
            # recenter explorer partly
            cur_exp = _norm(global_best + 0.35 * rng.normal(size=(n, d)))
            cur_val = _mc(cur_exp)
    # final squeeze: LP polish top candidates
    # polish global best with remaining time (keep 2s buffer)
    remain = DEAD - time.time() - 1.5
    if remain > 3.0:
        try:
            lp_time = min(remain, 26.0)
            global_best = _lp_polish(global_best, lp_time, deadline=DEAD - 1.0)
            global_val = _mc(global_best)
        except Exception:
            pass
    # also try polishing explorer if different and pick better
    remain2 = DEAD - time.time() - 1.0
    if remain2 > 4.0:
        try:
            alt = _lp_polish(cur_exp, min(6.0, remain2), deadline=DEAD - 0.5)
            av = _mc(alt)
            if av < global_val:
                global_best = alt
                global_val = av
        except Exception:
            pass
    return _norm(global_best)
