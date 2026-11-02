import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def lp_polish(X, seconds, t_deadline=None):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = max_cos(X)
    x_best = X.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if t_deadline is not None and time.time() >= t_deadline:
            break
        if radius < 1e-14:
            break
        c = (X @ X.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            X, best, x_best = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x_best


def tangent_bases(X):
    n, d = X.shape
    m = d - 1
    P = np.empty((n, d, m), dtype=np.float64)
    Eye = np.eye(d)
    for i in range(n):
        xi = X[i]
        k = int(np.argmax(np.abs(xi)))
        E = np.delete(Eye, k, axis=1)  # d x m
        # orthogonalize against xi
        coeff = xi @ E  # m
        Eorth = E - np.outer(xi, coeff)
        Q, _ = np.linalg.qr(Eorth)
        P[i] = Q
    return P


def build_R(X, P, I, J):
    n, d = X.shape
    m = d - 1
    p = len(I)
    Np = n * m
    R = np.zeros((p, Np), dtype=np.float64)
    # loop over contacts
    for k in range(p):
        i = int(I[k]); j = int(J[k])
        R[k, i * m:(i + 1) * m] = P[i].T @ X[j]
        R[k, j * m:(j + 1) * m] = P[j].T @ X[i]
    return R


def compute_soft_modes(X, eps=1e-6):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = float(np.max(G))
    iu = np.triu_indices(n, 1)
    c = G[iu]
    sel = c > m - eps
    # ensure reasonable size
    if np.sum(sel) < 30:
        sel = c > m - 1e-4
    if np.sum(sel) > 900:
        # keep tightest 900
        idx = np.argsort(-c)[:900]
        msel = np.zeros_like(sel)
        msel[idx] = True
        sel = msel
    I, J = iu[0][sel], iu[1][sel]
    P = tangent_bases(X)
    R = build_R(X, P, I, J)
    try:
        U, S, Vt = np.linalg.svd(R, full_matrices=False)
    except Exception:
        return P, R, None, None, None, (I, J)
    k5 = min(5, Vt.shape[0])
    Vt_tail = Vt[-k5:, :]
    # approximate equilibrium stress: minimal left singular vector
    w = None
    try:
        w = np.abs(U[:, -1])
        w = w / (np.sum(w) + 1e-12)
    except Exception:
        w = None
    return P, R, S, Vt_tail, w, (I, J)


def modal_displace(X, P, Vt_tail, amp, coeff):
    n, d = X.shape
    m = d - 1
    # coeff: (k5,) random combo, Vt_tail: (k5, Np)
    y = coeff @ Vt_tail  # Np
    nrm = float(np.linalg.norm(y))
    if nrm < 1e-12:
        return X
    y = y * (amp / nrm)
    Np = n * m
    dx = np.empty_like(X)
    for i in range(n):
        yb = y[i * m:(i + 1) * m]
        dx[i] = P[i] @ yb
    Y = X + dx
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y


def lse_adam(X0, steps, beta0, beta1, lr, t_deadline):
    n, d = X0.shape
    X = X0.copy()
    m1 = np.zeros_like(X)
    s1 = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    ratio = beta1 / max(beta0, 1e-12)
    for t in range(1, steps + 1):
        if (t & 31) == 0 and time.time() >= t_deadline:
            break
        frac = t / steps
        beta = beta0 * (ratio ** frac)
        lr_t = lr * (1.0 - 0.7 * frac)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        M = float(np.max(G))
        W = np.exp(beta * (G - M))
        np.fill_diagonal(W, 0.0)
        row = W.sum(axis=1, keepdims=True)
        Avg = (W @ X) / (row + 1e-12)
        dot = np.sum(Avg * X, axis=1, keepdims=True)
        Gt = Avg - dot * X
        m1 = b1 * m1 + (1.0 - b1) * Gt
        s1 = b2 * s1 + (1.0 - b2) * (Gt * Gt)
        mhat = m1 / (1.0 - b1 ** t)
        shat = s1 / (1.0 - b2 ** t)
        upd = mhat / (np.sqrt(shat) + eps)
        rdot = np.sum(upd * X, axis=1, keepdims=True)
        upd_t = upd - rdot * X
        X = X - lr_t * upd_t
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def contact_signature(X, thresh):
    G = X @ X.T
    np.fill_diagonal(G, -10.0)
    m = float(np.max(G))
    iu = np.triu_indices(X.shape[0], 1)
    sel = G[iu] > m - thresh
    idx = np.where(sel)[0]
    # encode pair index as single int using iu ordering
    return frozenset(idx.tolist()), m


def solve():
    t0 = time.time()
    T_HARD = 160.0
    T_SEARCH = T_HARD - 15.0
    t_dead_search = t0 + T_SEARCH
    t_dead_hard = t0 + T_HARD
    rng = np.random.default_rng(0)
    n, d = 54, 23
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    x_best = cands[bi].copy()
    best = float(vals[bi])

    # cached soft modes for current best basin
    try:
        P_cur, R_cur, S_cur, Vt_tail, w_stress, _ = compute_soft_modes(x_best, eps=1e-6)
        if Vt_tail is None:
            raise RuntimeError("svd fail")
    except Exception:
        P_cur = tangent_bases(x_best)
        Vt_tail = None

    seen_graphs = []
    # store incumbent graph to discard immediate returns
    try:
        sig0, _ = contact_signature(x_best, 5e-4)
        seen_graphs.append(sig0)
    except Exception:
        pass

    lp_used = 0
    MAX_LP = 5
    LP_SEC = 2.0
    SCREEN_LP = 0.003  # only LP if within best+SCREEN_LP
    trial = 0

    steps_choices = [300, 400, 500, 600]
    b0_choices = [40.0, 60.0, 80.0]
    b1_choices = [400.0, 700.0, 1100.0]
    lr_choices = [0.008, 0.012, 0.018]
    log_lo, log_hi = np.log(0.03), np.log(0.30)

    while time.time() < t_dead_search:
        trial += 1
        if time.time() > t_dead_search - 1.0:
            break
        # refresh modes occasionally if Vt missing
        if Vt_tail is None or P_cur is None:
            try:
                P_cur, R_cur, S_cur, Vt_tail, w_stress, _ = compute_soft_modes(x_best, eps=1e-6)
            except Exception:
                # fallback to gaussian kick
                Vt_tail = None

        # --- modal kick ---
        if Vt_tail is not None:
            k5 = Vt_tail.shape[0]
            amp = float(np.exp(rng.uniform(log_lo, log_hi)))
            coeff = rng.standard_normal(k5)
            try:
                x0 = modal_displace(x_best, P_cur, Vt_tail, amp, coeff)
            except Exception:
                x0 = x_best + 0.02 * rng.standard_normal((n, d))
                x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
        else:
            amp = float(rng.uniform(0.02, 0.08))
            x0 = x_best + amp * rng.standard_normal((n, d))
            x0 /= np.linalg.norm(x0, axis=1, keepdims=True)

        steps = int(steps_choices[rng.integers(0, len(steps_choices))])
        # shorten if little time left
        if t_dead_search - time.time() < 12.0 and steps > 350:
            steps = 350
        beta0 = float(b0_choices[rng.integers(0, len(b0_choices))])
        beta1 = float(b1_choices[rng.integers(0, len(b1_choices))])
        lr = float(lr_choices[rng.integers(0, len(lr_choices))])

        try:
            x1 = lse_adam(x0, steps=steps, beta0=beta0, beta1=beta1, lr=lr, t_deadline=t_dead_search)
        except Exception:
            continue
        v1 = max_cos(x1)
        if not np.isfinite(v1):
            continue
        if v1 > best + 0.012:
            continue

        # --- contact-graph clustering to discard returns ---
        try:
            sig, _ = contact_signature(x1, 5e-4)
            dup = False
            for s in seen_graphs[-200:]:
                if len(s) == 0 and len(sig) == 0:
                    dup = True
                    break
                inter = len(s & sig)
                union = len(s | sig)
                if union > 0 and inter / union > 0.92:
                    dup = True
                    break
            if not dup:
                seen_graphs.append(sig)
                if len(seen_graphs) > 400:
                    seen_graphs = seen_graphs[-400:]
            else:
                # repeated basin, discard unless strictly better
                if v1 >= best - 1e-12:
                    continue
        except Exception:
            pass

        if v1 < best - 1e-12:
            # direct improvement without LP; still try to keep
            best = v1
            x_best = x1.copy()
            try:
                P_cur, R_cur, S_cur, Vt_tail, w_stress, _ = compute_soft_modes(x_best, eps=1e-6)
            except Exception:
                pass
            # spend one LP only if we have budget and time to snap further
            if lp_used < MAX_LP and time.time() < t_dead_search - (LP_SEC + 1.0):
                try:
                    x2 = lp_polish(x1, seconds=min(LP_SEC, t_dead_search - time.time() - 0.2), t_deadline=t_dead_search)
                    v2 = max_cos(x2)
                    lp_used += 1
                    if v2 < best:
                        best = v2
                        x_best = x2.copy()
                        P_cur, R_cur, S_cur, Vt_tail, w_stress, _ = compute_soft_modes(x_best, eps=1e-6)
                except Exception:
                    pass
            continue

        # promising but not improving: limited LP snaps
        if v1 < best + SCREEN_LP and lp_used < MAX_LP:
            if time.time() > t_dead_search - (LP_SEC + 0.5):
                continue
            try:
                x2 = lp_polish(x1, seconds=min(LP_SEC, t_dead_search - time.time() - 0.2), t_deadline=t_dead_search)
                v2 = max_cos(x2)
                lp_used += 1
                # cluster LP result too
                try:
                    sig2, _ = contact_signature(x2, 5e-4)
                    seen_graphs.append(sig2)
                except Exception:
                    pass
                if v2 < best - 1e-12:
                    best = v2
                    x_best = x2.copy()
                    try:
                        P_cur, R_cur, S_cur, Vt_tail, w_stress, _ = compute_soft_modes(x_best, eps=1e-6)
                    except Exception:
                        pass
            except Exception:
                continue

    # reserve final polish
    time_left = t_dead_hard - time.time()
    if time_left > 2.0:
        try:
            x_best = lp_polish(x_best, seconds=min(time_left - 0.5, 15.0), t_deadline=t_dead_hard)
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
