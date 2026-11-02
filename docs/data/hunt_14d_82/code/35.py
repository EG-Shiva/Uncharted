import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, seconds, radius_init=5e-5, cap=1500, max_it=30, t_end=None):
    X = _norm(X.copy())
    n, d = X.shape
    best = _max_cos(X)
    radius = float(radius_init)
    iu0 = np.triu_indices(n, 1)
    it = 0
    start = time.time()
    while it < max_it:
        it += 1
        if time.time() - start >= seconds:
            break
        if t_end is not None and time.time() >= t_end:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        c_all = G[iu0]
        m = float(np.max(c_all))
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_all > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            vals = c_all[idx]
            part = np.argpartition(-vals, cap - 1)[:cap]
            keep = np.zeros_like(sel, dtype=bool)
            keep[idx[part]] = True
            sel = keep
        I = iu0[0][sel]; J = iu0[1][sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 1.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = X + res.x[:-1].reshape(n, d)
        Y = _norm(Y)
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def _tangent_bases(X):
    n, d = X.shape
    T = np.empty((n, d, d - 1))
    E = np.eye(d)
    for i in range(n):
        B = E.copy()
        B[:, 0] = X[i]
        Q, _ = np.linalg.qr(B)
        # ensure first column aligns with X[i]
        if np.dot(Q[:, 0], X[i]) < 0:
            Q[:, 0] = -Q[:, 0]
        # Q[:,1:] is orthogonal to Q[:,0]=+/-X[i]; fix sign already handled
        # re-orthogonalize check: if Q0 != X[i] due to dependence, force projection
        # but QR guarantees Q0 = X[i]/|X[i]| when B[:,0]=X[i] unless rank deficient;
        # in deficient case still orthogonal, accept.
        T[i] = Q[:, 1:]
    return T

def _rigidity_floppy_step(X, p_target=1000, eta=2e-6, flop_boost=1.0):
    n, d = X.shape
    dm1 = d - 1
    N = n * dm1
    iu = np.triu_indices(n, 1)
    G = X @ X.T
    c_all = G[iu]
    m = float(np.max(c_all))
    slack = m - c_all
    order = np.argsort(slack)
    # rigidity tolerance: include ~p_target pairs, clipped
    k = min(p_target, len(slack) - 1)
    tol_rig = float(slack[order[k]])
    tol_rig = float(np.clip(tol_rig, 5e-6, 3e-4))
    sel_rig = slack <= tol_rig
    # cap at 1300 largest (smallest slack)
    if int(sel_rig.sum()) > 1300:
        idx = np.where(sel_rig)[0]
        # keep smallest slack
        s = slack[idx]
        part = np.argpartition(s, 1300 - 1)[:1300]
        m2 = np.zeros_like(sel_rig, dtype=bool)
        m2[idx[part]] = True
        sel_rig = m2
    I = iu[0][sel_rig]; J = iu[1][sel_rig]
    p = len(I)
    # degrees at 1e-5 for signature / weighting
    sel5 = slack <= 1e-5
    deg = np.zeros(n, dtype=int)
    np.add.at(deg, iu[0][sel5], 1)
    np.add.at(deg, iu[1][sel5], 1)
    if p < 10:
        return None, m, deg, tol_rig, 0
    T = _tangent_bases(X)
    A = np.zeros((p, N))
    for r in range(p):
        i = int(I[r]); j = int(J[r])
        # Ti^T x_j
        a = T[i].T @ X[j]
        b = T[j].T @ X[i]
        A[r, i * dm1:(i + 1) * dm1] = a
        A[r, j * dm1:(j + 1) * dm1] = b
    # SVD for nullspace / rank
    try:
        _, S, Vt = np.linalg.svd(A, full_matrices=False)
    except Exception:
        return None, m, deg, tol_rig, 0
    rank = int(np.sum(S > 1e-3))
    # softmax weights near max for gradient
    c_rig = c_all[sel_rig]
    Sw = (c_rig - m) / max(eta, 1e-12)
    Sw = np.clip(Sw, -30.0, 0.0)
    W = np.exp(Sw)
    g = np.zeros(N)
    for r in range(p):
        w = W[r]
        if w < 1e-12:
            continue
        i = int(I[r]); j = int(J[r])
        g[i * dm1:(i + 1) * dm1] += w * (T[i].T @ X[j])
        g[j * dm1:(j + 1) * dm1] += w * (T[j].T @ X[i])
    ng = float(np.linalg.norm(g))
    if not np.isfinite(ng) or ng < 1e-18:
        return None, m, deg, tol_rig, rank
    g = g / ng
    # project onto floppy (null) and stiff
    if rank < N:
        F = Vt[rank:, :]
        coeff = F @ g
        g_flop = F.T @ coeff
        g_stiff = g - g_flop
    else:
        g_flop = np.zeros_like(g)
        g_stiff = g
    nfl = float(np.linalg.norm(g_flop))
    nst = float(np.linalg.norm(g_stiff))
    # emphasize floppy: y = flop + 0.25*stiff
    if nfl > 1e-12:
        y = g_flop + 0.25 * g_stiff
    else:
        # fully jammed direction: use stiff descent
        y = g
    # weight to relocate low-degree points: amplify rattlers
    wpts = 1.0 / (1.0 + deg.astype(float))
    # boost lowest: scale 0.3..3
    wpts = 0.3 + 2.7 * (wpts / max(wpts.max(), 1e-12))
    Y = y.reshape(n, dm1) * wpts[:, None]
    # map to ambient
    DX = np.zeros((n, d))
    for i in range(n):
        DX[i] = T[i] @ Y[i]
    # normalize so max per-point tangent norm = 1
    pn = np.linalg.norm(DX, axis=1)
    mx = float(pn.max())
    if not np.isfinite(mx) or mx < 1e-18:
        return None, m, deg, tol_rig, rank
    DX = DX / mx
    return DX, m, deg, tol_rig, rank

def solve():
    t0 = time.time()
    BUDGET = 168.0
    t_end = t0 + BUDGET
    n, d = 82, 14
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        rng0 = np.random.default_rng(0)
        xb = _norm(rng0.standard_normal((n, d)))
    else:
        cands.sort(key=lambda c: _max_cos(c))
        xb = cands[0].copy()
    # initial squeeze
    if t_end - time.time() > 15:
        xb = _lp_polish(xb, min(3.0, t_end - time.time() - 10.0), radius_init=2e-4, t_end=t_end - 10.0)
    best_val = _max_cos(xb)
    Xbest = xb.copy()
    cur = xb.copy()
    cur_val = best_val
    # event-driven rigidity loop
    it = 0
    etas = [3e-6, 1.5e-6, 5e-6, 8e-7]
    while time.time() < t_end - 9.0 and it < 60:
        it += 1
        base = cur if (it % 3 != 0) else Xbest
        eta = etas[it % len(etas)]
        # vary target size to span graph toward isostatic ~976
        p_target = 900 + (it % 4) * 100  # 900..1200
        out = _rigidity_floppy_step(base, p_target=p_target, eta=eta)
        if out[0] is None:
            # fallback small random kick on lowest-degree point
            _, m0, deg0, _, _ = out
            victim = int(np.argmin(deg0))
            rng = np.random.default_rng(it)
            trial = base.copy()
            trial[victim] = _norm(base[victim: victim + 1] + 0.005 * rng.standard_normal((1, d)))[0]
            trial = _norm(trial)
            v = _max_cos(trial)
            if v < cur_val:
                cur, cur_val = trial, v
            continue
        DX, m0, deg, tol_rig, rank = out
        # event line search along DX (floppy-descent) until new contact enters
        alphas = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.035]
        best_trial = None
        best_trial_v = 1e9
        # also track contact gain
        iu = np.triu_indices(n, 1)
        for a in alphas:
            if time.time() >= t_end - 8.0:
                break
            Y = base + a * DX
            Y = _norm(Y)
            v = _max_cos(Y)
            # prefer lower v; allow small increase if it creates contacts
            if v < best_trial_v:
                best_trial_v = v
                best_trial = Y
            # early accept if improvement
            if v < m0 - 2e-7:
                best_trial_v = v
                best_trial = Y
                break
        if best_trial is None:
            continue
        # accept if not too bad (event step may raise max slightly to gain rigidity)
        if best_trial_v > m0 + 0.004 and best_trial_v > cur_val + 0.002:
            # too destructive, try smaller alpha relocation of single worst rattler along DX
            order_low = np.argsort(deg)
            victim = int(order_low[0])
            Y = base.copy()
            Y[victim] = _norm((base[victim] + 0.005 * DX[victim])[None, :])[0]
            v2 = _max_cos(Y)
            if v2 < cur_val:
                cur, cur_val = Y, v2
            continue
        # polish the event trial briefly to convert new geometry into lower max
        budget_lp = 0.7
        if best_trial_v < best_val + 0.0005:
            budget_lp = 1.2
        if time.time() + budget_lp + 6.0 > t_end:
            budget_lp = max(0.2, t_end - 6.0 - time.time())
        if budget_lp < 0.15:
            # just track
            if best_trial_v < cur_val:
                cur, cur_val = best_trial, best_trial_v
            if best_trial_v < best_val:
                Xbest, best_val = best_trial.copy(), best_trial_v
            break
        # two-stage: moderate then micro
        t2 = _lp_polish(best_trial, budget_lp * 0.65, radius_init=6e-5, t_end=t_end - 5.0)
        t2 = _lp_polish(t2, budget_lp * 0.35, radius_init=6e-6, t_end=t_end - 5.0)
        v = _max_cos(t2)
        if v < best_val - 1e-15:
            Xbest, best_val = t2.copy(), v
            cur, cur_val = t2.copy(), v
        elif v < cur_val:
            cur, cur_val = t2.copy(), v
        else:
            # simulated-annealing accept to change basin
            Temp = 0.00035
            rng = np.random.default_rng(1000 + it)
            if rng.random() < np.exp((cur_val - v) / Temp):
                cur, cur_val = t2.copy(), v
        if it % 12 == 0:
            cur, cur_val = Xbest.copy(), best_val
    # final polish chain
    if time.time() < t_end - 0.6:
        Xbest = _lp_polish(Xbest, max(0.6, t_end - time.time() - 0.4), radius_init=4e-6, t_end=t_end - 0.2)
    if time.time() < t_end - 0.3:
        Xbest = _lp_polish(Xbest, max(0.4, t_end - time.time() - 0.1), radius_init=2e-4, t_end=t_end - 0.05)
    Xbest = _norm(Xbest)
    return Xbest
