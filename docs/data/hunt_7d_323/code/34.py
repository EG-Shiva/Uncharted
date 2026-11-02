import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _build_CS(m, speeds):
    ks = np.arange(m, dtype=np.float64)
    sp = np.array(speeds, dtype=np.float64)
    ang = (2.0 * np.pi / m) * (ks[:, None] * sp[None, :])
    return np.cos(ang), np.sin(ang)

def _build_full(reps, C, S, add_pole):
    R = reps.shape[0]
    m = C.shape[0]
    C0, C1, C2 = C[:, 0], C[:, 1], C[:, 2]
    S0, S1, S2 = S[:, 0], S[:, 1], S[:, 2]
    a1 = reps[:, 0]; a2 = reps[:, 1]; a3 = reps[:, 2]; a4 = reps[:, 3]
    a5 = reps[:, 4]; a6 = reps[:, 5]; h = reps[:, 6]
    B1 = C0[:, None] * a1[None, :] - S0[:, None] * a2[None, :]
    B2 = S0[:, None] * a1[None, :] + C0[:, None] * a2[None, :]
    B3 = C1[:, None] * a3[None, :] - S1[:, None] * a4[None, :]
    B4 = S1[:, None] * a3[None, :] + C1[:, None] * a4[None, :]
    B5 = C2[:, None] * a5[None, :] - S2[:, None] * a6[None, :]
    B6 = S2[:, None] * a5[None, :] + C2[:, None] * a6[None, :]
    Xr = np.empty((R, m, 7), dtype=np.float64)
    Xr[:, :, 0] = B1.T
    Xr[:, :, 1] = B2.T
    Xr[:, :, 2] = B3.T
    Xr[:, :, 3] = B4.T
    Xr[:, :, 4] = B5.T
    Xr[:, :, 5] = B6.T
    Xr[:, :, 6] = h[:, None] * np.ones((R, m))
    X = Xr.reshape(R * m, 7)
    if add_pole:
        pole = np.zeros((1, 7), dtype=np.float64)
        pole[0, 6] = 1.0
        X = np.vstack([X, pole])
    # renorm for safety
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _sym_exact(reps, C, S, add_pole):
    X = _build_full(reps, C, S, add_pole)
    return _max_cos(X), X

def _adam_refine(reps0, C, S, add_pole, steps, lr, beta0, beta1, deadline, track):
    # track: dict with 'best_v','best_X','best_reps'
    reps = reps0.copy()
    R = reps.shape[0]
    m1 = np.zeros_like(reps)
    m2 = np.zeros_like(reps)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for st in range(1, steps + 1):
        if time.time() >= deadline - 0.05:
            break
        # anneal beta
        frac = st / max(1, steps)
        beta = beta0 + (beta1 - beta0) * frac
        X = _build_full(reps, C, S, add_pole)
        N = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cmax = float(np.max(G))
        E = np.exp(beta * (G - cmax))
        np.fill_diagonal(E, 0.0)
        s = float(np.sum(E)) * 0.5
        if not np.isfinite(s) or s <= 0:
            break
        W = E / s
        GF = W @ X  # (N,7)
        nrep = R * C.shape[0]
        Gr = GF[:nrep].reshape(R, C.shape[0], 7)
        G1 = Gr[:, :, 0]; G2 = Gr[:, :, 1]; G3 = Gr[:, :, 2]; G4 = Gr[:, :, 3]
        G5 = Gr[:, :, 4]; G6 = Gr[:, :, 5]; G7 = Gr[:, :, 6]
        C0 = C[:, 0]; S0 = S[:, 0]; C1 = C[:, 1]; S1 = S[:, 1]; C2 = C[:, 2]; S2 = S[:, 2]
        grad = np.empty_like(reps)
        grad[:, 0] = np.sum(C0[None, :] * G1 + S0[None, :] * G2, axis=1)
        grad[:, 1] = np.sum(-S0[None, :] * G1 + C0[None, :] * G2, axis=1)
        grad[:, 2] = np.sum(C1[None, :] * G3 + S1[None, :] * G4, axis=1)
        grad[:, 3] = np.sum(-S1[None, :] * G3 + C1[None, :] * G4, axis=1)
        grad[:, 4] = np.sum(C2[None, :] * G5 + S2[None, :] * G6, axis=1)
        grad[:, 5] = np.sum(-S2[None, :] * G5 + C2[None, :] * G6, axis=1)
        grad[:, 6] = np.sum(G7, axis=1)
        # tangent projection
        dot = np.sum(grad * reps, axis=1, keepdims=True)
        grad = grad - dot * reps
        # Adam
        m1 = b1 * m1 + (1 - b1) * grad
        m2 = b2 * m2 + (1 - b2) * (grad * grad)
        m1h = m1 / (1 - b1 ** st)
        m2h = m2 / (1 - b2 ** st)
        upd = m1h / (np.sqrt(m2h) + eps)
        # project update tangent
        dot2 = np.sum(upd * reps, axis=1, keepdims=True)
        upd = upd - dot2 * reps
        reps = reps - lr * upd
        n = np.linalg.norm(reps, axis=1, keepdims=True)
        n[n == 0] = 1.0
        reps = reps / n
        if (st % 25 == 0) or (st == steps):
            v = _max_cos(X)
            if v < track['best_v']:
                track['best_v'] = float(v)
                track['best_X'] = X.copy()
                track['best_reps'] = reps.copy()
                track['C'] = C; track['S'] = S; track['add_pole'] = add_pole
    return reps

def _lp_polish(x, deadline, init_radius=6e-3, max_pairs=2000, max_it=60):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    it = 0
    while it < max_it:
        it += 1
        if time.time() >= deadline - 0.25:
            break
        if radius < 1e-13:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(np.max(c))
        th = m - max(20.0 * radius, 1e-9)
        sel = c > th
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > max_pairs:
            idx = np.where(sel)[0]
            cv = c[idx]
            part = np.argpartition(cv, -max_pairs)[-max_pairs:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = max_pairs
        I = iuI[sel]; J = iuJ[sel]; cs = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        cr = cols.reshape(p, 2 * d + 1); vr = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        cr[:, :d] = I[:, None] * d + ar[None, :]
        cr[:, d:2 * d] = J[:, None] * d + ar[None, :]
        cr[:, 2 * d] = n * d
        vr[:, :d] = x[J]; vr[:, d:2 * d] = x[I]; vr[:, 2 * d] = -1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        tl = deadline - time.time() - 0.08
        if tl < 0.15:
            break
        tl = min(tl, 1.2)
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": tl})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True); yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x, best

def solve():
    t0 = time.time()
    SYM_END = t0 + 105.0
    FINAL_END = t0 + 174.0
    rng = np.random.default_rng(0)
    # incumbent
    x_inc = None; v_inc = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < v_inc:
                        v_inc = float(v); x_inc = a.copy()
        except Exception:
            pass
    if x_inc is None:
        x_inc = rng.standard_normal(size=(323, 7))
        x_inc /= np.linalg.norm(x_inc, axis=1, keepdims=True)
        v_inc = _max_cos(x_inc)
    x_best = x_inc.copy(); best = float(v_inc)

    strata = [
        (23, 14, (1, 5, 11), True),
        (23, 14, (1, 2, 5), True),
        (19, 17, (1, 4, 9), False),
        (19, 17, (1, 2, 4), False),
    ]
    track = {'best_v': np.inf, 'best_X': None, 'best_reps': None, 'C': None, 'S': None, 'add_pole': True}
    for si, (m, R, speeds, add_pole) in enumerate(strata):
        if time.time() >= SYM_END - 1.0:
            break
        C, S = _build_CS(m, speeds)
        strata_deadline = SYM_END - (len(strata) - si - 1) * 12.0
        strata_deadline = min(strata_deadline, SYM_END)
        # random screening
        n_screen = 250
        best_rep = None; best_v = np.inf
        for _ in range(n_screen):
            if time.time() >= strata_deadline - 3.0:
                break
            reps = rng.standard_normal(size=(R, 7))
            reps /= np.linalg.norm(reps, axis=1, keepdims=True)
            v, _ = _sym_exact(reps, C, S, add_pole)
            if v < best_v:
                best_v = v; best_rep = reps.copy()
            if v < track['best_v']:
                # quick record (X rebuilt)
                _, X = _sym_exact(reps, C, S, add_pole)
                track['best_v'] = float(v); track['best_X'] = X.copy()
                track['best_reps'] = reps.copy(); track['C'] = C; track['S'] = S; track['add_pole'] = add_pole
        if best_rep is None:
            continue
        # Adam restarts with kicks
        cur = best_rep.copy()
        # first long refine
        try:
            cur = _adam_refine(cur, C, S, add_pole, steps=350, lr=0.02, beta0=30.0, beta1=90.0, deadline=strata_deadline - 0.5, track=track)
        except Exception:
            pass
        nkick = 0
        while time.time() < strata_deadline - 0.8 and nkick < 6:
            nkick += 1
            if time.time() >= SYM_END - 1.0:
                break
            try:
                base = track['best_reps'] if track['best_reps'] is not None and track['C'] is not None and track['C'].shape[0] == m else cur
                # ensure shape matches
                if base.shape[0] != R:
                    base = cur
                sig = 0.05 + 0.03 * (nkick % 3)
                pert = base + rng.standard_normal(size=base.shape) * sig
                pert /= np.linalg.norm(pert, axis=1, keepdims=True)
                pert = _adam_refine(pert, C, S, add_pole, steps=180, lr=0.015, beta0=40.0, beta1=110.0, deadline=strata_deadline - 0.3, track=track)
            except Exception:
                continue
        # update global best from symmetric track
        if track['best_X'] is not None and track['best_v'] < best:
            best = float(track['best_v']); x_best = track['best_X'].copy()

    # LP polish: first symmetric best (symmetry-breaking), then incumbent
    remaining = FINAL_END - time.time()
    cands = []
    if track['best_X'] is not None:
        cands.append(track['best_X'].copy())
    cands.append(x_inc.copy())
    if best < v_inc:
        cands.append(x_best.copy())
    # dedupe by value later; polish each with split budget
    budget_each = max(4.0, (FINAL_END - time.time() - 1.0) / max(1, len(cands)))
    for ci, cand in enumerate(cands):
        if time.time() >= FINAL_END - 2.0:
            break
        dl = min(time.time() + budget_each, FINAL_END)
        # use smaller radius for near-optimal incumbent, larger for symmetric
        ir = 6e-3 if ci == 0 else 3e-3
        try:
            y, v = _lp_polish(cand, dl, init_radius=ir, max_pairs=2000, max_it=60)
            if v < best - 1e-12:
                best = float(v); x_best = y.copy()
        except Exception:
            continue
    # final tiny polish on overall best if time remains
    if time.time() < FINAL_END - 2.5:
        try:
            y, v = _lp_polish(x_best, FINAL_END, init_radius=1.2e-3, max_pairs=2000, max_it=40)
            if v < best:
                best = float(v); x_best = y.copy()
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    # safety: if somehow worse than incumbent, return incumbent
    try:
        vb = _max_cos(x_best)
        if vb > v_inc and x_inc is not None:
            return x_inc
    except Exception:
        return x_inc
    return x_best
