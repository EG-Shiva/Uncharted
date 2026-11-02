import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos_full(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_tables(m, speeds):
    j = np.arange(m)
    ang = 2.0 * np.pi * j / float(m)
    Cc = np.empty((m, 3))
    Ss = np.empty((m, 3))
    for k, s in enumerate(speeds):
        Cc[:, k] = np.cos(ang * s)
        Ss[:, k] = np.sin(ang * s)
    return Cc, Ss

def _full_from_reps(reps, Cc, Ss, pole):
    R = reps.shape[0]
    m = Cc.shape[0]
    N = R * m + (1 if pole else 0)
    X = np.empty((N, 7), dtype=np.float64)
    r0 = reps[:, 0]; r1 = reps[:, 1]; r2 = reps[:, 2]
    r3 = reps[:, 3]; r4 = reps[:, 4]; r5 = reps[:, 5]; r6 = reps[:, 6]
    for j in range(m):
        c0, s0 = Cc[j, 0], Ss[j, 0]
        c1, s1 = Cc[j, 1], Ss[j, 1]
        c2, s2 = Cc[j, 2], Ss[j, 2]
        b = X[j*R:(j+1)*R]
        b[:, 0] = c0*r0 - s0*r1
        b[:, 1] = s0*r0 + c0*r1
        b[:, 2] = c1*r2 - s1*r3
        b[:, 3] = s1*r2 + c1*r3
        b[:, 4] = c2*r4 - s2*r5
        b[:, 5] = s2*r4 + c2*r5
        b[:, 6] = r6
    if pole:
        X[N-1, :] = 0.0
        X[N-1, 6] = 1.0
    return X

def _adam_restart(reps_init, Cc, Ss, pole, rng, deadline, steps=300, lr=0.02, tau0=0.02, tau1=0.004):
    R = reps_init.shape[0]
    m = Cc.shape[0]
    reps = reps_init.copy()
    mt = np.zeros_like(reps)
    vt = np.zeros_like(reps)
    b1, b2, eps = 0.9, 0.999, 1e-8
    best_reps = reps.copy()
    Xb = _full_from_reps(best_reps, Cc, Ss, pole)
    best_val = _max_cos_full(Xb)
    t = 0
    for step in range(steps):
        if time.time() > deadline:
            break
        frac = step / max(1, steps-1)
        tau = tau0 + (tau1 - tau0) * frac
        X = _full_from_reps(reps, Cc, Ss, pole)
        N = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cur_max = float(G.max())
        if cur_max < best_val:
            best_val = cur_max
            best_reps = reps.copy()
        # smooth weights
        # subtract max for stability
        E = np.exp((G - cur_max) / tau)
        np.fill_diagonal(E, 0.0)
        Z = E.sum() * 0.5
        if Z <= 0 or not np.isfinite(Z):
            break
        W = E / Z
        grad_full = W @ X  # (N,7)
        # map to reps
        grad_rep = np.zeros_like(reps)
        # first R*m rows are orbits, blocked by j
        for j in range(m):
            Gj = grad_full[j*R:(j+1)*R]
            c0, s0 = Cc[j, 0], Ss[j, 0]
            c1, s1 = Cc[j, 1], Ss[j, 1]
            c2, s2 = Cc[j, 2], Ss[j, 2]
            g0 = Gj[:, 0]; g1 = Gj[:, 1]; g2 = Gj[:, 2]
            g3 = Gj[:, 3]; g4 = Gj[:, 4]; g5 = Gj[:, 5]; g6 = Gj[:, 6]
            grad_rep[:, 0] += c0*g0 + s0*g1
            grad_rep[:, 1] += -s0*g0 + c0*g1
            grad_rep[:, 2] += c1*g2 + s1*g3
            grad_rep[:, 3] += -s1*g2 + c1*g3
            grad_rep[:, 4] += c2*g4 + s2*g5
            grad_rep[:, 5] += -s2*g4 + c2*g5
            grad_rep[:, 6] += g6
        # tangent projection
        dot = np.sum(grad_rep * reps, axis=1, keepdims=True)
        gt = grad_rep - dot * reps
        # Adam
        t += 1
        mt = b1*mt + (1-b1)*gt
        vt = b2*vt + (1-b2)*(gt*gt)
        mh = mt / (1 - b1**t)
        vh = vt / (1 - b2**t)
        reps = reps - lr * mh / (np.sqrt(vh) + eps)
        nrm = np.linalg.norm(reps, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        reps = reps / nrm
    # exact hill climb on worst orbits
    # evaluate best
    Xb = _full_from_reps(best_reps, Cc, Ss, pole)
    best_val = _max_cos_full(Xb)
    cur = best_reps.copy()
    # per-orbit worst for selection
    for _ in range(200):
        if time.time() > deadline:
            break
        Xc = _full_from_reps(cur, Cc, Ss, pole)
        Gc = Xc @ Xc.T
        np.fill_diagonal(Gc, -2.0)
        worst_pt = Gc.max(axis=1)
        Rm = cur.shape[0]
        # map points to orbits (first Rm*m)
        orb_worst = np.empty(Rm)
        for k in range(Rm):
            # indices k in each block j: j*Rm + k
            idx = np.arange(m)*Rm + k
            orb_worst[k] = worst_pt[idx].max()
        # pick among top-4 worst
        order = np.argsort(-orb_worst)
        kk = int(order[rng.integers(0, min(4, Rm))])
        sig = 0.01
        trial = cur.copy()
        trial[kk] = trial[kk] + rng.standard_normal(7)*sig
        trial[kk] /= np.linalg.norm(trial[kk])
        Xt = _full_from_reps(trial, Cc, Ss, pole)
        vt_ = _max_cos_full(Xt)
        if vt_ < best_val:
            best_val = vt_
            cur = trial
            best_reps = trial.copy()
    return best_reps, best_val

def _lp_polish(x, deadline, init_radius=5e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos_full(x)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        remain = deadline - time.time()
        if remain < 0.2:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        mcur = float(c.max())
        thresh = mcur - max(20.0*radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 3000:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -3000)[-3000:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 3000
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n*d + 1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.empty(p*(2*d+1), dtype=np.int64)
        vals = np.empty(p*(2*d+1), dtype=np.float64)
        Cb = cols.reshape(p, 2*d+1)
        Vb = vals.reshape(p, 2*d+1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None]*d + ar
        Cb[:, d:2*d] = J[:, None]*d + ar
        Cb[:, 2*d] = n*d
        Vb[:, :d] = x[J]
        Vb[:, d:2*d] = x[I]
        Vb[:, 2*d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        tl = min(2.0, max(0.3, remain-0.1))
        try:
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)]*(n*d)+[(None, None)],
                              method="highs", options={"time_limit": tl})
            except TypeError:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)]*(n*d)+[(None, None)],
                              method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos_full(y)
        if v < best:
            x = y
            best = v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD = t0 + 165.0
    SYM_END = t0 + 105.0
    rng = np.random.default_rng(0)
    # load incumbent
    x_best = None
    best = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos_full(a)
                    if v < best:
                        best = v
                        x_best = a.copy()
        except Exception:
            pass
    if x_best is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a
        best = _max_cos_full(a)
    # symmetric configs: (m, speeds, pole, R)
    configs = [
        (23, (1, 5, 11), True, 14),
        (23, (1, 4, 9), True, 14),
        (19, (1, 4, 9), False, 17),
        (23, (1, 2, 3), True, 14),
        (19, (1, 7, 11), False, 17),
        (23, (2, 7, 11), True, 14),
        (23, (1, 7, 10), True, 14),
        (19, (2, 5, 11), False, 17),
    ]
    tables = []
    for (m, sp, pole, R) in configs:
        Cc, Ss = _build_tables(m, sp)
        tables.append((Cc, Ss))
    sym_best_X = None
    sym_best = np.inf
    sym_best_reps = None
    sym_best_cfg = None
    ci = 0
    # quick random init loop round-robin
    while time.time() < SYM_END:
        idx = ci % len(configs)
        ci += 1
        m, sp, pole, R = configs[idx]
        Cc, Ss = tables[idx]
        reps_init = rng.standard_normal(size=(R, 7))
        reps_init /= np.linalg.norm(reps_init, axis=1, keepdims=True)
        # deadline for this restart: leave time; cap ~6s per restart
        rdead = min(HARD, time.time() + 6.0, SYM_END)
        if rdead - time.time() < 0.5:
            break
        try:
            reps_opt, val = _adam_restart(reps_init, Cc, Ss, pole, rng, rdead, steps=300)
        except Exception:
            continue
        if val < sym_best:
            sym_best = val
            sym_best_reps = reps_opt.copy()
            sym_best_cfg = idx
            sym_best_X = _full_from_reps(reps_opt, Cc, Ss, pole)
        # early stop if already beating record clearly? continue to improve
    # choose candidate to polish: best symmetric if found and competitive, else incumbent
    cand = None
    if sym_best_X is not None:
        cand = sym_best_X.copy()
    else:
        cand = x_best.copy()
    # LP polish candidate in full space (break symmetry allowed)
    try:
        polished = _lp_polish(cand, min(HARD, t0+163.0), init_radius=4e-3)
        pv = _max_cos_full(polished)
    except Exception:
        polished = cand
        pv = _max_cos_full(cand)
    # compare with global incumbent
    final = x_best
    final_best = best
    if pv < final_best:
        final = polished
        final_best = pv
    # also if symmetric unpolished better than polished (shouldn't), keep min
    if sym_best_X is not None and sym_best < final_best:
        # quick tiny polish of sym best already done; return sym best
        final = sym_best_X
        final_best = sym_best
    final = final / np.linalg.norm(final, axis=1, keepdims=True)
    # safety: if NaN, return incumbent
    if not np.all(np.isfinite(final)):
        final = x_best
    return final
