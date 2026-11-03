import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _lp_polish(x, seconds, deadline_abs=None, p_cap=2500, max_it=12):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-14 and it < max_it:
        it += 1
        if deadline_abs is not None and time.time() > deadline_abs - 0.5:
            break
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = p_cap
        I = iu0[sel]
        J = iu1[sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if deadline_abs is not None and time.time() > deadline_abs - 0.5:
            break
    return best_x


def _tangent_bases(x):
    n, d = x.shape
    dm = d - 1
    T = np.empty((n, d, dm))
    E = np.eye(d)[:, :dm]
    for i in range(n):
        xi = x[i]
        B = np.empty((d, d))
        B[:, 0] = xi
        B[:, 1:] = E
        Q, _ = np.linalg.qr(B)
        if np.dot(Q[:, 0], xi) < 0:
            Q[:, 0] = -Q[:, 0]
        T[i] = Q[:, 1:]
    return T


def _surgical_move(x, hub, drop_js, step_rad, rng, n_modes=40):
    n, d = x.shape
    dm = d - 1
    N = n * dm
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    thr = m - 1e-7
    iu0, iu1 = np.triu_indices(n, 1)
    c_all = G[iu0, iu1]
    sel = c_all > thr
    # cap to tightest 1200 if too many
    if int(sel.sum()) > 1200:
        idx = np.where(sel)[0]
        part = np.argpartition(-c_all[idx], 1200 - 1)[:1200]
        keep = np.zeros_like(sel)
        keep[idx[part]] = True
        sel = keep
    I_all = iu0[sel]
    J_all = iu1[sel]
    # partition into drop vs keep
    drop_set = set()
    for j in drop_js:
        a, b = (hub, int(j)) if hub < int(j) else (int(j), hub)
        drop_set.add((a, b))
    keep_mask = []
    drop_rows = []
    for k in range(len(I_all)):
        if (int(I_all[k]), int(J_all[k])) in drop_set:
            drop_rows.append(k)
        else:
            keep_mask.append(k)
    # if drop not in contact list (hub rattler etc), synthesize drop pairs from closest
    if len(drop_rows) == 0:
        # use closest pairs to hub as drop
        row = G[hub].copy()
        row[hub] = -2
        cj = np.argsort(-row)[:max(2, len(drop_js))]
        drop_js = [int(v) for v in cj]
        drop_set = set()
        for j in drop_js:
            a, b = (hub, j) if hub < j else (j, hub)
            drop_set.add((a, b))
        keep_mask = list(range(len(I_all)))
        # build synthetic drop list separately
        I_drop = np.array([min(hub, j) for j in drop_js], dtype=int)
        J_drop = np.array([max(hub, j) for j in drop_js], dtype=int)
        I_keep = I_all
        J_keep = J_all
    else:
        I_keep = I_all[keep_mask] if len(keep_mask) else np.zeros(0, dtype=int)
        J_keep = J_all[keep_mask] if len(keep_mask) else np.zeros(0, dtype=int)
        I_drop = I_all[drop_rows]
        J_drop = J_all[drop_rows]
    T = _tangent_bases(x)
    p_keep = len(I_keep)
    # build R_keep (p_keep x N)
    if p_keep > 0:
        Rk = np.empty((p_keep, N))
        for k in range(p_keep):
            i = int(I_keep[k]); j = int(J_keep[k])
            vi = T[i].T @ x[j]
            vj = T[j].T @ x[i]
            row = np.zeros(N)
            row[i * dm:(i + 1) * dm] = vi
            row[j * dm:(j + 1) * dm] = vj
            Rk[k] = row
    else:
        Rk = np.zeros((0, N))
    p_drop = len(I_drop)
    if p_drop > 0:
        Rd = np.empty((p_drop, N))
        for k in range(p_drop):
            i = int(I_drop[k]); j = int(J_drop[k])
            vi = T[i].T @ x[j]
            vj = T[j].T @ x[i]
            row = np.zeros(N)
            row[i * dm:(i + 1) * dm] = vi
            row[j * dm:(j + 1) * dm] = vj
            Rd[k] = row
    else:
        Rd = np.zeros((0, N))
    # null / low-energy basis of R_keep via dense SVD
    try:
        if p_keep == 0:
            Zb = np.eye(N)[:, :n_modes]
        elif p_keep >= 1:
            # Rk is p_keep x N; SVD
            _, S, Vt = np.linalg.svd(Rk, full_matrices=False)
            M = min(n_modes, Vt.shape[0], N)
            Zb = Vt[-M:, :].T.copy()  # N x M smallest
        else:
            Zb = np.eye(N)[:, :n_modes]
    except Exception:
        # fallback random tangent
        Zb = None
    if Zb is None:
        v = rng.normal(size=N)
    else:
        Mm = Zb.shape[1]
        if p_drop > 0:
            P = Rd @ Zb  # p_drop x M
            g = P.sum(axis=0)  # M
            nrm = float(np.linalg.norm(g))
            if nrm < 1e-12:
                w = rng.normal(size=Mm)
                w /= (np.linalg.norm(w) + 1e-20)
            else:
                w = -g / nrm
                # mix small random for diversity across cycles
                w = w + 0.25 * rng.normal(size=Mm)
                w /= (np.linalg.norm(w) + 1e-20)
        else:
            w = rng.normal(size=Mm)
            w /= (np.linalg.norm(w) + 1e-20)
        v = Zb @ w
    # reshape to dx
    V = v.reshape(n, dm)
    DX = np.einsum('ijk,ik->ij', T, V)
    norms = np.linalg.norm(DX, axis=1)
    mx = float(norms.max()) if norms.size else 0.0
    if mx < 1e-12:
        DX = rng.normal(size=(n, d))
        DX -= (DX * x).sum(axis=1, keepdims=True) * x
        norms = np.linalg.norm(DX, axis=1)
        mx = float(norms.max() + 1e-20)
    scale = step_rad / (mx + 1e-20)
    DX = DX * scale
    norms2 = np.linalg.norm(DX, axis=1)
    y = x.copy()
    for i in range(n):
        ni = float(norms2[i])
        if ni < 1e-14:
            continue
        y[i] = np.cos(ni) * x[i] + np.sin(ni) * (DX[i] / ni)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y


def solve():
    t0 = time.time()
    DEADLINE = 170.0
    abs_dead = t0 + DEADLINE
    np.random.seed(1)
    rng = np.random.default_rng(1)
    n, d = 123, 13
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_val:
                        best_val = v
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    # initial quick squeeze
    try:
        if time.time() < abs_dead - 30:
            cand = _lp_polish(best, 2.0, deadline_abs=abs_dead)
            v = _max_cos(cand)
            if v < best_val:
                best_val = v
                best = cand
    except Exception:
        pass
    steps_cycle = [0.03, 0.05, 0.08, 0.12, 0.15, 0.06, 0.10, 0.04]
    drop_cycle = [2, 3, 4, 5, 3, 2, 4, 3]
    it = 0
    # precompute hub order refresh periodically
    while time.time() < abs_dead - 12:
        it += 1
        # refresh contact ordering every iteration from incumbent
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            cmax = float(G.max())
            thr = cmax - 1e-7
            counts = (G > thr).sum(axis=1)
            wmax = G.max(axis=1)
            hubs = np.argsort(-counts)
            # mix worst-max and high-count for hub choice
            if it % 3 == 0:
                hubs = np.argsort(-wmax)
            top_hubs = hubs[:12]
            hub = int(top_hubs[(it - 1) % len(top_hubs)])
            # dropped-contact set: sample among hub's contacts
            row = G[hub]
            order = np.argsort(-row)
            # collect actual contacts
            cons = [int(j) for j in order if row[j] > thr and j != hub][:20]
            if len(cons) == 0:
                cons = [int(j) for j in order[:20] if j != hub]
            kdrop = drop_cycle[(it - 1) % len(drop_cycle)]
            kdrop = min(kdrop, len(cons))
            # cycle start offset for diversity
            off = (it * 3) % max(1, len(cons))
            drop_js = []
            for t in range(kdrop):
                drop_js.append(cons[(off + t) % len(cons)])
            step_rad = float(steps_cycle[(it - 1) % len(steps_cycle)])
            # occasional larger random hub for diversity
            if it % 9 == 0:
                hub = int(rng.integers(0, n))
                step_rad = float(rng.uniform(0.05, 0.13))
        except Exception:
            continue
        # surgical move
        try:
            trial = _surgical_move(best, hub, drop_js, step_rad, rng)
        except Exception:
            continue
        # tiny extra jitter to avoid exact symmetry
        try:
            trial = trial + 1e-4 * rng.normal(size=trial.shape)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
        except Exception:
            pass
        remaining = abs_dead - 12 - (time.time() - t0 + t0) + t0
        # capped LP polish 1.0-1.4s
        lp_sec = 1.3 if (abs_dead - time.time()) > 25 else 0.9
        lp_sec = min(lp_sec, max(0.5, abs_dead - 12 - time.time()))
        if lp_sec < 0.4:
            break
        try:
            trial_p = _lp_polish(trial, lp_sec, deadline_abs=abs_dead - 10)
        except Exception:
            continue
        try:
            v = _max_cos(trial_p)
        except Exception:
            continue
        if v < best_val - 1e-14:
            best_val = v
            best = trial_p.copy()
        # hard deadline check
        if time.time() > abs_dead - 12:
            break
        # safety: cap iterations
        if it > 60:
            break
    # final polish with remaining time (up to ~8s)
    try:
        rem = abs_dead - time.time() - 0.8
        if rem > 0.8:
            rem = min(rem, 8.0)
            polished = _lp_polish(best, rem, deadline_abs=abs_dead - 0.3)
            pv = _max_cos(polished)
            if pv < best_val:
                best = polished
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
