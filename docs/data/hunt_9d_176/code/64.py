import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def quench(X, steps, beta0, beta1, lr0, lr1, lam0, lam1, mu, deadline):
    n, d = X.shape
    c = n / d
    I9 = np.eye(d)
    for s in range(steps):
        if (s & 15) == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        lam = lam0 * (lam1 / lam0) ** frac if lam1 > 0 else lam0
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # global LogSumExp weights, reuse for row-softmax
        E = np.exp((G - m) * beta)
        # diag entry exp(beta*(-2-m)) is tiny; zero it
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        rowsum = E.sum(axis=1)
        rowsum = np.maximum(rowsum, 1e-300)
        # smooth per-point worst (up to additive const m)
        srow = np.log(rowsum) / beta
        ms = float(srow.mean())
        coeff = 2.0 * (srow - ms) / n  # d Var / ds
        Q = E / rowsum[:, None]
        C = coeff[:, None] * Q
        W = E / (S * 0.5) + mu * (C + C.T)
        Gr = W @ X
        # moment balancing: ||X^T X - c I||_F^2, grad = 4 X (X^T X - cI)
        if lam != 0.0:
            XtX = X.T @ X
            Emat = XtX - c * I9
            Gr = Gr + (4.0 * lam) * (X @ Emat)
        # tangent projection
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        gnorm = float(np.linalg.norm(Gt) / np.sqrt(n))
        if not np.isfinite(gnorm) or gnorm < 1e-16:
            continue
        X = X - (lr / gnorm) * Gt
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        X = X / nr
    return X

def hist_vec(X, bins=20):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    v = G[iu]
    h, _ = np.histogram(v, bins=bins, range=(-1.0, 1.0))
    h = h.astype(float) / max(1, v.size)
    return h

def lp_polish_deadline(x, deadline, r_init=0.004, cap=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline:
            break
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argpartition(-c_all[idx], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx[order]] = True
            sel = keep
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols[:, 2 * d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD = t0 + 170.0
    SCREEN_END = t0 + 143.0
    LP_START = t0 + 145.0
    n, d = 176, 9
    rng = np.random.default_rng(20261103)
    # fallback
    base = None
    base_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    a = a / nr
                    v = max_cos(a)
                    if v < base_v:
                        base_v = v
                        base = a.copy()
        except Exception:
            pass
    # 16-24 independent starts
    NST = 20
    starts = []
    for k in range(NST):
        if time.time() > SCREEN_END:
            break
        X = rng.standard_normal((n, d))
        # low-discrepancy flavour: mix random rotation for half
        if k % 2 == 1:
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
            X = X @ Q
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        starts.append(X)
    screened = []
    screened_v = []
    # cheap stage-1 quench for each start, no LP
    for X in starts:
        if time.time() > SCREEN_END:
            break
        try:
            Y = quench(X, steps=550, beta0=55.0, beta1=150.0,
                       lr0=0.06, lr1=0.012, lam0=2.0e-3, lam1=2.0e-4,
                       mu=5.0, deadline=SCREEN_END)
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            v = max_cos(Y)
        except Exception:
            continue
        if time.time() > HARD:
            break
        screened.append(Y)
        screened_v.append(v)
        if time.time() > SCREEN_END:
            break
    if len(screened) == 0:
        # nothing screened: fall back
        if base is not None:
            Xf = base.copy()
        else:
            Xf = rng.standard_normal((n, d))
            Xf /= np.linalg.norm(Xf, axis=1, keepdims=True)
        # single LP polish in final 25s if time
        try:
            if time.time() < HARD - 1.0:
                Xf = lp_polish_deadline(Xf, HARD - 1.0)
        except Exception:
            pass
        Xf /= np.linalg.norm(Xf, axis=1, keepdims=True)
        return Xf
    screened_v = np.array(screened_v)
    order = np.argsort(screened_v)
    # keep top 6
    K6 = min(6, len(order))
    top_idx = order[:K6].tolist()
    # diversity: histogram distance, preserve mutually distant elites
    hists = [hist_vec(screened[i]) for i in top_idx]
    sel = [0]  # index into top_idx list (best first)
    while len(sel) < min(3, K6):
        best_cand = -1
        best_d = -1.0
        for c in range(K6):
            if c in sel:
                continue
            dm = min(float(np.linalg.norm(hists[c] - hists[s])) for s in sel)
            # slight preference for better score
            score_adj = dm - 0.02 * c
            if score_adj > best_d:
                best_d = score_adj
                best_cand = c
        if best_cand < 0:
            break
        sel.append(best_cand)
    elites = [screened[top_idx[c]] for c in sel]
    elite_v = [screened_v[top_idx[c]] for c in sel]
    # progressively sharper quenches only on top three
    sharp = [(110.0, 380.0, 0.030, 0.005, 5.0e-4, 5.0e-5, 10.0, 1100),
             (160.0, 500.0, 0.018, 0.003, 1.5e-4, 2.0e-5, 14.0, 900)]
    for e in range(len(elites)):
        if time.time() > SCREEN_END:
            break
        X = elites[e]
        for (b0, b1, l0, l1, lm0, lm1, mu, st) in sharp:
            if time.time() > SCREEN_END:
                break
            try:
                X = quench(X, steps=st, beta0=b0, beta1=b1, lr0=l0, lr1=l1,
                           lam0=lm0, lam1=lm1, mu=mu, deadline=SCREEN_END)
            except Exception:
                break
            if time.time() > HARD:
                break
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        try:
            v = max_cos(X)
        except Exception:
            continue
        elites[e] = X
        elite_v[e] = v
    # rank elites
    bi = int(np.argmin(elite_v))
    new_best = elites[bi]
    new_v = float(elite_v[bi])
    if base is not None and base_v <= new_v:
        final = base.copy()
    else:
        final = new_best.copy()
    # wait until LP window / use remaining time: single LP polish in final 25s
    # if we finished early, do an extra cheap quench on final before LP? No LP until final window to obey brief.
    # single LP polish
    try:
        lp_deadline = HARD - 1.0
        # ensure we start LP with at most 25s left; if too early, idle with tiny extra quench? just run LP (it respects deadline)
        if time.time() < lp_deadline:
            # cap LP time to final 25 seconds worth: start no earlier than HARD-25? if earlier, do cheap extra quench until then
            while time.time() < max(LP_START, HARD - 25.0):
                if time.time() > HARD - 26.0:
                    break
                # gentle extra quench to use spare time without LP
                try:
                    final = quench(final, steps=120, beta0=200.0, beta1=350.0,
                                   lr0=0.008, lr1=0.003, lam0=5e-5, lam1=1e-5,
                                   mu=12.0, deadline=min(HARD - 25.0, lp_deadline))
                    final /= np.linalg.norm(final, axis=1, keepdims=True)
                except Exception:
                    break
                break
            final = lp_polish_deadline(final, lp_deadline, r_init=0.004)
    except Exception:
        pass
    final = final / np.linalg.norm(final, axis=1, keepdims=True)
    # fallback check
    try:
        vf = max_cos(final)
    except Exception:
        if base is not None:
            return base
        return final
    if base is not None and not np.isfinite(vf):
        return base
    if base is not None and base_v < vf:
        return base
    return final
