import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    # set diagonal to -2
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _lp_polish(X, deadline, init_radius=5e-3):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    bestX = X.copy()
    radius = float(init_radius)
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        now = time.time()
        if now >= deadline - 0.3:
            break
        G = X @ X.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            break
        if cnt > 3500:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -3500)[-3500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 3500
        I = iuI[sel]
        J = iuJ[sel]
        csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        cr = cols.reshape(p, 2 * d + 1)
        vr = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        cr[:, :d] = I[:, None] * d + ar
        cr[:, d:2 * d] = J[:, None] * d + ar
        cr[:, 2 * d] = n * d
        vr[:, :d] = X[J]
        vr[:, d:2 * d] = X[I]
        vr[:, 2 * d] = -1.0
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        tl = deadline - time.time() - 0.15
        if tl <= 0.05:
            break
        tl = min(tl, 8.0)
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": float(tl), "presolve": True})
        except Exception:
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                radius *= 0.5
                continue
            if time.time() >= deadline - 0.2:
                break
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y
            best = v
            bestX = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
        if time.time() >= deadline - 0.2:
            break
    return bestX


def _build_full(reps, c1, s1, c2, s2, c3, s3, pole):
    m = reps.shape[0]
    q = c1.shape[0]
    has_pole = pole is not None
    N = m * q + (1 if has_pole else 0)
    X = np.empty((N, 7), dtype=np.float64)
    idx = 0
    for j in range(m):
        x0, y0, x1, y1, x2, y2, h = reps[j]
        X[idx:idx + q, 0] = x0 * c1 - y0 * s1
        X[idx:idx + q, 1] = x0 * s1 + y0 * c1
        X[idx:idx + q, 2] = x1 * c2 - y1 * s2
        X[idx:idx + q, 3] = x1 * s2 + y1 * c2
        X[idx:idx + q, 4] = x2 * c3 - y2 * s3
        X[idx:idx + q, 5] = x2 * s3 + y2 * c3
        X[idx:idx + q, 6] = h
        idx += q
    if has_pole:
        X[idx] = pole
    return X


def _symmetric_anneal(m, q, speeds, pole_vec, deadline, rng, Xseed=None):
    theta = 2.0 * np.pi / q
    ks = np.arange(q, dtype=np.float64)
    s1, s2, s3 = speeds
    c1 = np.cos(ks * theta * s1); sn1 = np.sin(ks * theta * s1)
    c2 = np.cos(ks * theta * s2); sn2 = np.sin(ks * theta * s2)
    c3 = np.cos(ks * theta * s3); sn3 = np.sin(ks * theta * s3)

    def eval_reps(R):
        Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
        X = _build_full(Rn, c1, sn1, c2, sn2, c3, sn3, pole_vec)
        return _max_cos(X), X

    # init candidates
    bestR = None
    bestV = np.inf
    bestX = None
    # candidate 1: random
    Rrand = rng.standard_normal(size=(m, 7))
    Rrand /= np.linalg.norm(Rrand, axis=1, keepdims=True)
    v, Xf = eval_reps(Rrand)
    if v < bestV:
        bestV = v; bestR = Rrand.copy(); bestX = Xf
    # candidate 2: sampled from seed
    if Xseed is not None:
        try:
            idx = rng.choice(Xseed.shape[0], size=m, replace=False)
            Rs = Xseed[idx].copy()
            Rs /= np.linalg.norm(Rs, axis=1, keepdims=True)
            v2, Xf2 = eval_reps(Rs)
            if v2 < bestV:
                bestV = v2; bestR = Rs.copy(); bestX = Xf2
        except Exception:
            pass
    curR = bestR.copy()
    curV = bestV
    curX = bestX
    globR = curR.copy(); globV = curV; globX = curX.copy()
    sigma = 0.12
    it = 0
    noimp = 0
    while True:
        it += 1
        if it % 50 == 0:
            if time.time() >= deadline - 0.2:
                break
        # propose: perturb single orbit mostly, occasionally global
        prop = curR.copy()
        if (it % 25) == 0:
            prop = prop + rng.standard_normal(size=prop.shape) * (sigma * 0.5)
            for j in range(m):
                nrm = np.linalg.norm(prop[j])
                if nrm == 0:
                    prop[j] = curR[j]
                else:
                    prop[j] /= nrm
        else:
            j = int(rng.integers(0, m))
            noise = rng.standard_normal(size=7) * sigma
            prop[j] = prop[j] + noise
            nrm = np.linalg.norm(prop[j])
            if nrm == 0:
                prop[j] = curR[j]
            else:
                prop[j] /= nrm
        # eval
        Rn = prop  # already normalized (global case normalized above)
        # ensure normalized
        # quick renormalize all (cheap for m<=17)
        # already done, but do safe:
        # (skip extra norm for speed, prop rows are unit)
        Xp = _build_full(Rn, c1, sn1, c2, sn2, c3, sn3, pole_vec)
        G = Xp @ Xp.T
        np.fill_diagonal(G, -2.0)
        v = float(G.max())
        if v < curV:
            curR = Rn.copy() if Rn is not prop else prop
            # need copy to avoid alias
            curR = prop.copy()
            curV = v
            curX = Xp
            noimp = 0
            if v < globV:
                globV = v; globR = prop.copy(); globX = Xp.copy()
        else:
            noimp += 1
        if it % 500 == 0:
            sigma *= 0.93
            if sigma < 0.004:
                sigma = 0.004
            # if stuck long, kick sigma up once
            if noimp > 2500:
                sigma = min(0.08, sigma * 2.0 + 0.005)
                noimp = 0
        if it % 2000 == 0 and time.time() >= deadline - 0.5:
            break
    return globR, globV, globX


def solve():
    t0 = time.time()
    DEAD = t0 + 165.0
    rng = np.random.default_rng(12345)
    # load incumbent
    global_best = None
    global_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.any(~np.isfinite(a)):
                    continue
                v = _max_cos(a)
                if v < global_val:
                    global_val = v
                    global_best = a.copy()
        except Exception:
            continue
    if global_best is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        global_best = a
        global_val = _max_cos(a)
    Xseed = global_best.copy()
    # symmetric configs: (q, m, speeds, has_pole)
    configs = [
        (23, 14, (1, 5, 11), True),
        (23, 14, (1, 2, 4), True),
        (23, 14, (1, 3, 8), True),
        (19, 17, (1, 4, 9), False),
        (19, 17, (1, 2, 7), False),
    ]
    pole_north = np.array([0, 0, 0, 0, 0, 0, 1.0], dtype=np.float64)
    sym_best_X = None
    sym_best_V = np.inf
    # reserve time: symmetric phase up to 75s, but leave >=55s for polish
    sym_deadline = min(t0 + 75.0, DEAD - 55.0)
    if sym_deadline > time.time() + 5:
        per = (sym_deadline - time.time()) / len(configs)
        for (q, m, speeds, has_pole) in configs:
            if time.time() >= sym_deadline - 0.5:
                break
            sub_dead = min(time.time() + per, sym_deadline)
            # shrink per if behind? keep simple
            try:
                pole = pole_north if has_pole else None
                _, v, Xf = _symmetric_anneal(m, q, speeds, pole, sub_dead, rng, Xseed=Xseed)
                if v < sym_best_V:
                    sym_best_V = v
                    sym_best_X = Xf.copy()
                if v < global_val:
                    global_val = v
                    global_best = Xf.copy()
            except Exception:
                continue
            if time.time() >= DEAD - 60:
                break
    # LP-polish best symmetric with symmetry breaking
    if sym_best_X is not None and time.time() < DEAD - 12:
        try:
            dl = min(time.time() + 22.0, DEAD - 30.0)
            if dl > time.time() + 2:
                Xp = _lp_polish(sym_best_X, dl, init_radius=5e-3)
                v = _max_cos(Xp)
                if v < global_val:
                    global_val = v
                    global_best = Xp.copy()
                # keep sym polish even if not better than global? track anyway
                if v < sym_best_V:
                    sym_best_V = v
                    sym_best_X = Xp.copy()
        except Exception:
            pass
    # final polish of global incumbent
    if time.time() < DEAD - 3:
        try:
            remain = (DEAD - 3.0) - time.time()
            remain = min(remain, 45.0)
            if remain > 2:
                dl = time.time() + remain
                Xp = _lp_polish(global_best, dl, init_radius=3e-3)
                v = _max_cos(Xp)
                if v < global_val and np.all(np.isfinite(Xp)):
                    global_val = v
                    global_best = Xp.copy()
        except Exception:
            pass
    global_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    # shrink tiny margin not needed; ensure finite
    global_best[~np.isfinite(global_best)] = 0.0
    # renormalize after fix
    nrm = np.linalg.norm(global_best, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    global_best = global_best / nrm
    return global_best
