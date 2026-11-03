import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.sqrt(np.sum(X*X, axis=1, keepdims=True)) + 1e-30
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_lse(Y0, stages, deadline):
    Y = Y0.copy()
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    step = 0
    for beta, iters, lr in stages:
        for _ in range(iters):
            if time.time() > deadline:
                break
            nrm = np.sqrt(np.sum(Y*Y, axis=1, keepdims=True)) + 1e-30
            X = Y / nrm
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mx = float(G.max())
            E = np.exp(beta * (G - mx))
            np.fill_diagonal(E, 0.0)
            S = float(E.sum() / 2.0) + 1e-300
            GradX = (E @ X) / S
            dot = np.sum(GradX * X, axis=1, keepdims=True)
            GradY = (GradX - dot * X) / nrm
            step += 1
            m[:] = 0.9 * m + 0.1 * GradY
            v[:] = 0.999 * v + 0.001 * GradY * GradY
            mhat = m / (1.0 - 0.9 ** step)
            vhat = v / (1.0 - 0.999 ** step)
            Y -= lr * mhat / (np.sqrt(vhat) + 1e-8)
        if time.time() > deadline:
            break
    return _norm(Y)

def _lp_polish(x, deadline):
    x = _norm(x)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(c.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _e8_points():
    pts = []
    for i in range(8):
        for j in range(i+1, 8):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(8); v[i] = si; v[j] = sj
                    pts.append(v)
    for mask in range(256):
        cnt = 0
        vv = np.empty(8)
        for k in range(8):
            b = (mask >> k) & 1
            cnt += b
            vv[k] = -0.5 if b else 0.5
        if cnt % 2 == 0:
            pts.append(vv.copy())
    A = np.stack(pts, axis=0)
    return _norm(A)

def _tangent_noise(X, sigma, rng):
    N = rng.standard_normal(X.shape)
    dot = np.sum(N * X, axis=1, keepdims=True)
    Nt = N - dot * X
    return _norm(X + sigma * Nt)

def _refine_void(y0, Xf, beta=150, steps=250, lr=0.06):
    y = y0.copy()
    yn = float(np.linalg.norm(y)) + 1e-30
    y = y / yn
    for _ in range(steps):
        c = Xf @ y
        mx = float(c.max())
        e = np.exp(beta * (c - mx))
        S = float(e.sum()) + 1e-300
        g = (e[:, None] * Xf).sum(axis=0) / S
        g = g - float(g @ y) * y
        y = y - lr * g
        yn = float(np.linalg.norm(y)) + 1e-30
        if yn < 1e-12:
            break
        y = y / yn
    return y

def _find_void(Xf, rng):
    S = 8000
    C = rng.standard_normal((S, 8))
    C = _norm(C)
    vals = (C @ Xf.T).max(axis=1)
    idx = int(np.argmin(vals))
    y0 = C[idx].copy()
    return _refine_void(y0, Xf)

def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    rng0 = np.random.default_rng(0)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn).astype(float)
            if vv.shape == (n, d):
                vv = _norm(vv)
                c = _max_cos(vv)
                if c < best_val:
                    best_val = c
                    best = vv.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng0.standard_normal((n, d)))
        best_val = _max_cos(best)
    incumbent = best.copy()

    E8 = _e8_points()

    # build initial pool
    inits = []
    inits.append(("inc", incumbent.copy()))
    for sig, nm in ((0.02, "p02"), (0.08, "p08"), (0.25, "p25")):
        r = np.random.default_rng(10 + int(sig*100))
        inits.append((nm, _tangent_noise(incumbent, sig, r)))
    for s in range(2):
        r = np.random.default_rng(100 + s)
        inits.append((f"rnd{s}", _norm(r.standard_normal((n, d)))))
    for s in range(2):
        r = np.random.default_rng(200 + s)
        idx = r.choice(E8.shape[0], size=n, replace=False)
        Xe = E8[idx].copy()
        Xe = _tangent_noise(Xe, 0.01, r)
        inits.append((f"e8{s}", Xe))

    stages_short = [(40.0, 300, 0.02), (120.0, 300, 0.012), (300.0, 400, 0.007)]
    stages_mid = [(40.0, 350, 0.02), (120.0, 350, 0.012), (300.0, 450, 0.007), (600.0, 350, 0.004)]
    stages_long = [(60.0, 400, 0.018), (150.0, 400, 0.011), (350.0, 500, 0.007), (700.0, 400, 0.0035)]

    scored = []
    short_deadline = min(deadline - 90.0, t0 + 70.0)
    for nm, X0 in inits:
        if time.time() > short_deadline:
            break
        try:
            Xopt = _adam_lse(X0, stages_short, short_deadline)
        except Exception:
            continue
        v = _max_cos(Xopt)
        scored.append((v, Xopt))
        if v < best_val:
            best_val = v
            best = Xopt.copy()
    scored.sort(key=lambda t: t[0])
    # mid refinement on top 3
    top = scored[:3] if len(scored) >= 3 else scored
    refined = []
    mid_deadline = min(deadline - 70.0, t0 + 100.0)
    for v0, X0 in top:
        if time.time() > mid_deadline:
            break
        try:
            Xopt = _adam_lse(X0, stages_mid, mid_deadline)
        except Exception:
            Xopt = X0
        v = _max_cos(Xopt)
        refined.append((v, Xopt))
        if v < best_val:
            best_val = v
            best = Xopt.copy()
    if len(refined) == 0:
        refined = [(best_val, best.copy())]
    refined.sort(key=lambda t: t[0])
    cur = refined[0][1].copy()
    cur_val = float(refined[0][0])

    # rattler void reinsertion + hub dispersal
    vrng = np.random.default_rng(999)
    void_deadline = min(deadline - 30.0, t0 + 140.0)
    for rnd in range(5):
        if time.time() > void_deadline:
            break
        G = cur @ cur.T
        np.fill_diagonal(G, -2.0)
        mx = float(G.max())
        nearest = G.max(axis=1)
        slack = mx - nearest
        # rattlers: loose points
        order_loose = np.argsort(-slack)
        # degrees with 1e-4 band for hub detection
        deg = (G > mx - 1e-4).sum(axis=1)
        order_hub = np.argsort(-deg)
        nrep = 3 if rnd < 3 else 2
        # choose victims: loosest with slack > 2e-5, else random
        victims = []
        for ii in order_loose:
            if slack[ii] > 2e-5 and len(victims) < nrep:
                victims.append(int(ii))
        # if not enough, add a hub duplicate to disperse (perturb hub instead)
        Xnew = cur.copy()
        if len(victims) > 0:
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[victims] = False
            Xkeep = Xnew[keep_mask]
            cur_stack = Xkeep.copy()
            newpts = []
            for _ in victims:
                if time.time() > void_deadline:
                    break
                yv = _find_void(cur_stack, vrng)
                newpts.append(yv)
                cur_stack = np.vstack([cur_stack, yv])
            Xnew[keep_mask] = Xkeep
            # place new points at victim indices
            for vi, yv in zip(victims, newpts):
                Xnew[vi] = yv
            # small noise on worst hub to break jam
            hub = int(order_hub[0])
            if hub not in victims and deg[hub] >= 8:
                nn = vrng.standard_normal(d)
                nn = nn - float(nn @ Xnew[hub]) * Xnew[hub]
                nn = nn / (float(np.linalg.norm(nn)) + 1e-30)
                Xnew[hub] = _norm((Xnew[hub] + 0.01 * nn).reshape(1, -1))[0]
        else:
            # no clear rattlers: perturb top hubs slightly
            for k in range(min(3, n)):
                h = int(order_hub[k])
                nn = vrng.standard_normal(d)
                nn = nn - float(nn @ Xnew[h]) * Xnew[h]
                nn = nn / (float(np.linalg.norm(nn)) + 1e-30)
                Xnew[h] = Xnew[h] + 0.02 * nn
            Xnew = _norm(Xnew)
        try:
            Xopt = _adam_lse(Xnew, stages_short, void_deadline)
        except Exception:
            continue
        v = _max_cos(Xopt)
        if v < best_val:
            best_val = v
            best = Xopt.copy()
        # accept if not much worse to change basin
        if v < cur_val + 5e-4:
            cur = Xopt.copy()
            cur_val = v
        else:
            # try hub shake from cur
            pass

    # long final tune on current best candidates
    final_deadline = deadline - 12.0
    if time.time() < final_deadline:
        try:
            Xf = _adam_lse(best.copy(), stages_long, final_deadline)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val = vf
                best = Xf.copy()
        except Exception:
            pass
    if time.time() < deadline - 1.0:
        try:
            best = _lp_polish(best, deadline - 0.5)
            best_val = _max_cos(best)
        except Exception:
            pass
    best = _norm(best)
    return best
