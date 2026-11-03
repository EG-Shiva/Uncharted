import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def contact_count(X, thresh):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    return int(np.count_nonzero(c > thresh)), float(c.max())

def tangent_bases(X):
    n, d = X.shape
    dm = d - 1
    B = np.empty((n, d, dm))
    for i in range(n):
        x = X[i]
        M = np.eye(d) - np.outer(x, x)
        w, V = np.linalg.eigh(M)
        # smallest eigenvalue ~0, rest ~1; take largest dm
        B[i] = V[:, 1:] if w[0] < w[-1] else V[:, :dm]
        # ensure orthogonality
        # V from eigh is sorted ascending, so first is null
        # B = last dm columns
    return B

def adam_quench(X, steps=400, a0=40.0, a1=350.0, lr=0.003):
    x = X.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    vv = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        frac = t / max(steps - 1, 1)
        alpha = a0 * ((a1 / a0) ** frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        lrt = lr * (1.0 - 0.5 * frac)
        x = x - lrt * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if not np.all(np.isfinite(x)):
            break
    nr = np.linalg.norm(x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    return x / nr

def lp_polish_deadline(x, deadline, r_init=0.004, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
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
            idx_all = np.where(sel)[0]
            # keep largest `cap`
            part = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols_I = (I[:, None] * d + np.arange(d))
        cols_J = (J[:, None] * d + np.arange(d))
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = cols_I
        cols[:, d:2*d] = cols_J
        cols[:, 2*d] = n * d
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
        try:
            v = max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 165.0
    n, d = 176, 9
    # load incumbent
    inc = None
    inc_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    a = a / nr
                    v = max_cos(a)
                    if v < inc_v:
                        inc_v = v
                        inc = a.copy()
        except Exception:
            pass
    if inc is None:
        rng0 = np.random.default_rng(0)
        inc = rng0.standard_normal((n, d))
        inc /= np.linalg.norm(inc, axis=1, keepdims=True)
        inc_v = max_cos(inc)
    best_X = inc.copy()
    best_v = float(inc_v)
    old_max = best_v

    # --- build contact graph within 1e-7 ---
    try:
        G0 = best_X @ best_X.T
        iu_all = np.triu_indices(n, 1)
        c_all = G0[iu_all]
        m0 = float(c_all.max())
        sel = c_all > m0 - 1e-7
        cnt0 = int(np.count_nonzero(sel))
        # fallback: ensure at least 200 pairs and at most 3000
        if cnt0 < 200:
            k = min(1000, len(c_all))
            part = np.argpartition(c_all, -k)[-k:]
            I0 = iu_all[0][part]; J0 = iu_all[1][part]
        elif cnt0 > 3000:
            idx = np.where(sel)[0]
            # keep top 2500
            order = np.argpartition(-c_all[idx], 2500)[:2500]
            keep = idx[order]
            I0 = iu_all[0][keep]; J0 = iu_all[1][keep]
        else:
            I0 = iu_all[0][sel]; J0 = iu_all[1][sel]
        p = len(I0)
    except Exception:
        return best_X

    if time.time() > HARD_END - 30:
        return best_X

    # --- sparse contact Jacobian in tangent coords ---
    try:
        B = tangent_bases(best_X)  # n x 9 x 8
        dm = d - 1
        Ndim = n * dm
        J = np.zeros((p, Ndim), dtype=np.float64)
        # vectorized per pair loop
        for k in range(p):
            i = int(I0[k]); j = int(J0[k])
            # Ti^T xj , Tj^T xi
            vi = B[i].T @ best_X[j]  # 8
            vj = B[j].T @ best_X[i]
            J[k, i*dm:(i+1)*dm] = vi
            J[k, j*dm:(j+1)*dm] = vj
        # Gram matrix
        Gr = J.T @ J  # 1408 x 1408
        # symmetrize
        Gr = (Gr + Gr.T) * 0.5
        w, V = np.linalg.eigh(Gr)
        # V columns, ascending
        K = min(10, V.shape[1])
        modes = V[:, :K]  # Ndim x K
    except Exception:
        return best_X

    if time.time() > HARD_END - 30:
        return best_X

    # --- try signed displacements along first 5-10 modes ---
    try:
        nmodes = min(8, modes.shape[1])
        amps = [0.02, 0.05, 0.1, 0.2]
        cands = []  # (cnt, maxv, X)
        iu_t = np.triu_indices(n, 1)
        for mi in range(nmodes):
            if time.time() > HARD_END - 35:
                break
            v = modes[:, mi]  # Ndim
            Vb = v.reshape(n, dm)  # per-point coeffs
            # dx = B @ coeff
            Dx = np.einsum('nij,nj->ni', B, Vb)
            for sgn in (1.0, -1.0):
                for amp in amps:
                    Y = best_X + sgn * amp * Dx
                    nr = np.linalg.norm(Y, axis=1, keepdims=True)
                    nr[nr == 0] = 1.0
                    Y = Y / nr
                    # exact max only
                    G = Y @ Y.T
                    cc = G[iu_t]
                    mv = float(cc.max())
                    # explosion guard
                    if mv > old_max + 0.02 or not np.isfinite(mv):
                        continue
                    cnt = int(np.count_nonzero(cc > mv - 1e-7))
                    cands.append((cnt, mv, Y))
        # retain three moves that most reduce contacts without exploding max
        if len(cands) == 0:
            return best_X
        cands.sort(key=lambda t: (t[0], t[1]))
        top3 = cands[:3]
    except Exception:
        return best_X

    # --- quench each briefly with LogSumExp Adam ---
    quenched = []
    for (cnt, mv, Y) in top3:
        if time.time() > HARD_END - 30:
            break
        try:
            Q = adam_quench(Y, steps=400, a0=40.0, a1=350.0, lr=0.003)
            qv = max_cos(Q)
            # count contacts of quenched
            Gq = Q @ Q.T
            ccq = Gq[iu_t]
            qcnt = int(np.count_nonzero(ccq > qv - 1e-7))
            quenched.append((qcnt, qv, Q))
        except Exception:
            continue
    if len(quenched) == 0:
        return best_X
    quenched.sort(key=lambda t: (t[1], t[0]))
    best_q_cnt, best_q_v, best_q_X = quenched[0]

    # pick best quenched for LP polish (smallest max)
    # reserve one capped 25s LP polish
    lp_budget = 25.0
    lp_deadline = min(HARD_END, time.time() + lp_budget)
    # ensure at least a couple seconds
    if lp_deadline <= time.time() + 0.5:
        # no time for LP, compare directly
        if best_q_v < best_v:
            return best_q_X / np.linalg.norm(best_q_X, axis=1, keepdims=True)
        return best_X
    try:
        Z = lp_polish_deadline(best_q_X, lp_deadline, r_init=0.004, cap=2500)
        zv = max_cos(Z)
    except Exception:
        # fallback to quenched
        if best_q_v < best_v:
            return best_q_X / np.linalg.norm(best_q_X, axis=1, keepdims=True)
        return best_X

    # enforce hard deadline and incumbent rule
    if time.time() > HARD_END + 1:
        pass
    # return incumbent if no improvement
    if zv < best_v and np.all(np.isfinite(Z)):
        Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
        return Z
    # also check unpolished quenched in case LP worsened but quench improved
    if best_q_v < best_v and best_q_v < zv:
        return best_q_X / np.linalg.norm(best_q_X, axis=1, keepdims=True)
    return best_X
