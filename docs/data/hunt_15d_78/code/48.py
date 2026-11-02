import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos_vec(X):
    # X: (B,n,d) or (n,d)
    if X.ndim == 2:
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(G.max())
    else:
        B = X.shape[0]
        G = X @ X.transpose(0, 2, 1)
        for b in range(B):
            np.fill_diagonal(G[b], -2.0)
        return G.reshape(B, -1).max(axis=1)

def _lp_polish_single(x, deadline, r_init=0.008, r_max=0.03, cap=2500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    # fast max
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    best = float(G0.max())
    radius = r_init
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    while time.time() + 0.5 < deadline and radius > 1e-13:
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thresh = max(20.0 * radius, 1e-9)
        sel = c_all > m - thresh
        nnz = int(np.count_nonzero(sel))
        if nnz == 0:
            radius *= 0.5
            continue
        if nnz > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:cap]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
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
        G2 = y @ y.T
        np.fill_diagonal(G2, -2.0)
        v = float(G2.max())
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, r_max)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(12345)
    n, d = 78, 15
    B = 32
    # load fallback
    try:
        best0 = np.load("best.npy").astype(float)
        best0 /= np.linalg.norm(best0, axis=1, keepdims=True)
    except Exception:
        best0 = None
    try:
        rec = np.load("record.npy").astype(float)
        rec /= np.linalg.norm(rec, axis=1, keepdims=True)
    except Exception:
        rec = best0
    # fingerprint of record
    def topk_sorted(X, k=200):
        G = X @ X.T
        iu = np.triu_indices(n, 1)
        c = G[iu]
        c = np.sort(c)[::-1][:k]
        return c
    rec_top = topk_sorted(rec, 200) if rec is not None else None
    rec_max = _max_cos_vec(rec) if rec is not None else 0.168274222547

    # ---- batched multi-start ----
    # init random codes
    Y = rng.standard_normal((B, n, d))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    X = Y.copy()
    # a couple replicas seeded from perturbed record/best for diversity? spec says ignoring record
    # keep all random to search balanced basin (pure fresh)
    T = 800
    beta0, beta1 = 20.0, 300.0
    lam1, lam2 = 5.0, 8.0
    lr = 0.012
    b1, b2, eps = 0.9, 0.999, 1e-8
    m_ad = np.zeros_like(X)
    v_ad = np.zeros_like(X)
    I_d = np.eye(d) / d
    # precompute diag mask
    # Adam loop
    for t in range(1, T + 1):
        if time.time() > HARD_END - 55:
            break
        frac = t / T
        beta = beta0 * (beta1 / beta0) ** frac
        # anneal lr slightly
        alpha = lr * (1.0 - 0.6 * frac)
        # Gram
        G = X @ X.transpose(0, 2, 1)  # (B,n,n)
        # set diag to -2 for stability (broadcast)
        # fastest: fill diagonal
        # use stride trick: G[:, range, range] = -2
        G[:, np.arange(n), np.arange(n)] = -2.0
        mg = G.reshape(B, -1).max(axis=1)  # (B,)
        # global weights (ordered)
        # E = exp(beta*(G-mg))
        # to save memory compute in place
        E = np.exp(beta * (G - mg[:, None, None]))
        E[:, np.arange(n), np.arange(n)] = 0.0
        S = E.sum(axis=(1, 2))  # (B,)
        S = np.maximum(S, 1e-300)
        W = E / S[:, None, None]
        # grad L
        GrL = 2.0 * (W @ X)
        # personal smoothed
        mi = G.max(axis=2)  # (B,n)
        Er = np.exp(beta * (G - mi[:, :, None]))
        Er[:, np.arange(n), np.arange(n)] = 0.0
        Si = Er.sum(axis=2)  # (B,n)
        Si = np.maximum(Si, 1e-300)
        # p_i not needed fully except var; compute log
        # pi = mi + log(Si)/beta
        logSi = np.log(Si)
        pi = mi + logSi / beta
        meanp = pi.mean(axis=1)  # (B,)
        diffp = pi - meanp[:, None]
        coeff = 2.0 * diffp / n  # (B,n)
        WR = Er / Si[:, :, None]
        WRX = WR @ X  # (B,n,d)
        term1 = WRX * coeff[:, :, None]
        M = WR * coeff[:, :, None]  # M[k,j]=WR[k,j]*coeff[k]
        term2 = M.transpose(0, 2, 1) @ X
        GrV = term1 + term2
        # iso grad
        # C = X^T X / n
        XtX = X.transpose(0, 2, 1) @ X / n  # (B,d,d)
        Diff = XtX - I_d[None, :, :]
        GrI = (4.0 * lam1 / n) * (X @ Diff)
        Grad = GrL + GrI + lam2 * GrV
        # tangent project
        dot = (Grad * X).sum(axis=2, keepdims=True)
        Gt = Grad - dot * X
        # Adam
        m_ad = b1 * m_ad + (1 - b1) * Gt
        v_ad = b2 * v_ad + (1 - b2) * Gt * Gt
        bc1 = 1 - b1 ** t
        bc2 = 1 - b2 ** t
        mhat = m_ad / bc1
        vhat = v_ad / bc2
        step = alpha * mhat / (np.sqrt(vhat) + eps)
        # project step to tangent
        sdot = (step * X).sum(axis=2, keepdims=True)
        stept = step - sdot * X
        # limit step norm to avoid blowup
        # clip per-point norm to 0.05
        sn = np.linalg.norm(stept, axis=2, keepdims=True)
        clip = 0.05
        scale = np.minimum(1.0, clip / (sn + 1e-12))
        stept = stept * scale
        X = X - stept
        X /= np.linalg.norm(X, axis=2, keepdims=True)

    # ---- evaluate replicas ----
    # compute metrics
    Gf = X @ X.transpose(0, 2, 1)
    Gf[:, np.arange(n), np.arange(n)] = -2.0
    maxs = Gf.reshape(B, -1).max(axis=1)
    # covariance eigen spread
    XtXf = X.transpose(0, 2, 1) @ X / n
    eig_spread = np.zeros(B)
    for b in range(B):
        try:
            ev = np.linalg.eigvalsh(XtXf[b])
            eig_spread[b] = float(ev.max() / max(ev.min(), 1e-12))
        except Exception:
            eig_spread[b] = 1e9
    # contact degree variance (tol 0.005 for unpolished)
    deg_var = np.zeros(B)
    tops = np.zeros((B, 200))
    for b in range(B):
        Gb = Gf[b]
        m = maxs[b]
        cnt = (Gb > m - 0.005).sum(axis=1)
        deg_var[b] = float(cnt.var())
        c = Gb[np.triu_indices(n, 1)]
        c = np.sort(c)[::-1][:200]
        tops[b] = c
    # fingerprint distance to record
    if rec_top is not None:
        fdist = np.linalg.norm(tops - rec_top[None, :], axis=1)
    else:
        fdist = np.full(B, 1e9)
    # rank: primary max, penalize duplicates and anisotropy
    order = np.argsort(maxs)
    # select best 2-3 distinct
    selected = []
    for idx in order:
        # discard near-duplicates of record: close max AND close fingerprint
        is_dup = (abs(float(maxs[idx]) - float(rec_max)) < 2e-6) and (float(fdist[idx]) < 0.02)
        # also enforce not duplicate among selected
        dup2 = False
        for s in selected:
            if abs(float(maxs[idx]) - float(maxs[s])) < 1e-7 and float(np.linalg.norm(tops[idx] - tops[s])) < 0.01:
                dup2 = True
                break
        # prefer balanced: allow dup but deprioritize? we skip dups first pass
        if is_dup or dup2:
            continue
        selected.append(int(idx))
        if len(selected) >= 3:
            break
    # if too few, fill with best regardless
    if len(selected) < 2:
        for idx in order:
            if int(idx) not in selected:
                selected.append(int(idx))
            if len(selected) >= 3:
                break
    # also consider balanced bonus: sort selected by max + small tie-breaker on spread/var
    # LP-polish selected (up to 3, <=15s each)
    cands = []
    # include fallback
    if best0 is not None:
        cands.append((best0, _max_cos_vec(best0)))
    for s in selected[:3]:
        Xs = X[s] / np.linalg.norm(X[s], axis=1, keepdims=True)
        cands.append((Xs, float(maxs[s])))
    # sort cands by max to polish most promising first
    cands.sort(key=lambda z: z[1])
    # polish best 3 distinct
    best_x = None
    best_v = 1e9
    # keep initial best
    for (cx, cv) in cands:
        if cv < best_v:
            best_v = cv
            best_x = cx.copy()
    # time left for polish: reserve
    for k, (cx, cv) in enumerate(cands[:3]):
        remain = HARD_END - time.time()
        if remain < 4:
            break
        budget = min(15.0, remain - 2.0)
        if budget < 2.0:
            break
        deadline = time.time() + budget
        px, pv = _lp_polish_single(cx, deadline, r_init=0.008, r_max=0.03)
        if pv < best_v:
            best_v = pv
            best_x = px.copy()
        # early stop if clearly record-beating and time low? continue to use time
    # final fine polish of best
    remain = HARD_END - time.time()
    if remain > 3 and best_x is not None:
        px, pv = _lp_polish_single(best_x, HARD_END - 0.4, r_init=0.002, r_max=0.01)
        if pv < best_v:
            best_v = pv
            best_x = px.copy()
    if best_x is None:
        best_x = rec.copy() if rec is not None else X[0]
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    if not np.all(np.isfinite(best_x)):
        best_x = rec / np.linalg.norm(rec, axis=1, keepdims=True)
    return best_x
