import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    x = x / n
    x = np.where(np.isfinite(x), x, 0.0)
    n2 = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n2[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _personal_worst(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1), g

def _contact_counts(x, thresh=1e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    c = (g > m - thresh).sum(axis=1)
    return c, m

def _lp_polish(x, deadline):
    x = _normalize(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _normalize(y)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def _anneal_lbfgs(x_init, taus, maxiters, t0, time_limit):
    n, d = x_init.shape
    y = _normalize(x_init).ravel().copy()
    for tau, mit in zip(taus, maxiters):
        if time.time() - t0 > time_limit:
            break
        def fun(yflat, _tau=tau, _n=n, _d=d):
            Y = yflat.reshape(_n, _d)
            norms = np.linalg.norm(Y, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            X = Y / norms
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            Gmax = float(G.max())
            E = np.exp((G - Gmax) / _tau)
            np.fill_diagonal(E, 0.0)
            Esum = float(E.sum())
            if not np.isfinite(Esum) or Esum <= 0:
                Esum = 1e-300
            S = Esum * 0.5
            loss = Gmax + _tau * np.log(max(S, 1e-300))
            W = E / Esum
            GradX = 2.0 * (W @ X)
            dot = (GradX * X).sum(axis=1, keepdims=True)
            GradY = (GradX - dot * X) / norms
            return float(loss), GradY.ravel()
        try:
            res = minimize(fun, y, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(mit), "ftol": 1e-14, "gtol": 1e-7, "maxfun": int(mit * 12)})
            y = res.x
        except Exception:
            break
    return _normalize(y.reshape(n, d))

def _rand_rot4(rng):
    A = rng.standard_normal((4, 4))
    Q, R = np.linalg.qr(A)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def _best_void(others, rng, n_cand=6000):
    cand = rng.standard_normal((n_cand, 4))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    s = cand @ others.T
    scores = s.max(axis=1)
    bi = int(np.argmin(scores))
    b = cand[bi].copy()
    bs = float(scores[bi])
    step = 0.06
    for _ in range(250):
        prop = b + step * rng.standard_normal((32, 4))
        prop = prop / np.linalg.norm(prop, axis=1, keepdims=True)
        sc = (prop @ others.T).max(axis=1)
        j = int(np.argmin(sc))
        if sc[j] < bs:
            b, bs = prop[j], float(sc[j])
            step *= 1.05
        else:
            step *= 0.96
        if step < 1e-5:
            break
    return b

def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    rng = np.random.default_rng(20261031)
    try:
        xb0 = np.load("best.npy")
    except Exception:
        try:
            xb0 = np.load("record.npy")
        except Exception:
            xb0 = rng.standard_normal((272, 4))
    xb0 = _normalize(np.asarray(xb0, dtype=np.float64))
    if xb0.shape != (272, 4):
        xb0 = _normalize(rng.standard_normal((272, 4)))
    base_val = _max_cos(xb0)
    best_x = xb0.copy()
    best_val = base_val
    cur_x = xb0.copy()
    cur_val = base_val

    counts, _ = _contact_counts(best_x, thresh=1e-7)
    rattlers0 = np.where(counts == 0)[0]

    taus_base = [0.03, 0.015, 0.008, 0.004, 0.002]
    mits_base = [50, 50, 70, 90, 110]

    cands = [(base_val, xb0.copy())]

    TRIAL_END = t0 + 118.0
    it = 0
    Temp = 0.004
    while time.time() < TRIAL_END and time.time() < HARD_END - 50:
        it += 1
        remain = TRIAL_END - time.time()
        if remain <= 1.0:
            break
        mode = it % 6
        y = cur_x.copy()
        if mode == 0:
            sigma = 0.025 + 0.01 * (rng.random())
            y = y + sigma * rng.standard_normal(y.shape)
            y = _normalize(y)
        elif mode == 1:
            sigma = 0.06 + 0.04 * rng.random()
            y = y + sigma * rng.standard_normal(y.shape)
            cc, _ = _contact_counts(cur_x, thresh=1e-7)
            rr = np.where(cc == 0)[0]
            if len(rr) > 0:
                for r in rr:
                    y[r] = rng.standard_normal(4)
            pw, _ = _personal_worst(cur_x)
            order = np.argsort(-pw)
            for idx in order[:3]:
                y[idx] += 0.18 * rng.standard_normal(4)
            y = _normalize(y)
        elif mode == 2:
            k = int(rng.integers(12, 30))
            pw, _ = _personal_worst(cur_x)
            order = np.argsort(-pw)
            pool = order[:60]
            sel = pool[rng.choice(len(pool), size=min(k, len(pool)), replace=False)]
            # reinsert selected into voids sequentially
            for wi in sel:
                others = np.delete(y, wi, axis=0)
                y[wi] = _best_void(others, rng, n_cand=6000)
            y = _normalize(y)
        elif mode == 3:
            k = int(rng.integers(16, 36))
            sel = rng.choice(272, size=k, replace=False)
            Q = _rand_rot4(rng)
            mu = y[sel].mean(axis=0, keepdims=True)
            mu = mu / max(np.linalg.norm(mu), 1e-12)
            y[sel] = (y[sel] - mu) @ Q.T + mu
            y[sel] = y[sel] + 0.02 * rng.standard_normal((k, 4))
            y = _normalize(y)
        elif mode == 4:
            sigma = 0.10 + 0.08 * rng.random()
            k = int(rng.integers(20, 45))
            sel = rng.choice(272, size=k, replace=False)
            y[sel] += sigma * 2.0 * rng.standard_normal((k, 4))
            y += 0.015 * rng.standard_normal(y.shape)
            y = _normalize(y)
        else:
            # single worst void + small global shake
            pw, _ = _personal_worst(cur_x)
            order = np.argsort(-pw)
            nw = int(rng.integers(1, 4))
            for wi in order[:nw]:
                others = np.delete(y, wi, axis=0)
                y[wi] = _best_void(others, rng, n_cand=8000)
            y = y + 0.02 * rng.standard_normal(y.shape)
            y = _normalize(y)

        scale = 0.85 + 0.3 * rng.random()
        taus_t = [t * scale for t in taus_base]
        # shorten last stage if low on time
        tl = max(0.5, TRIAL_END - time.time())
        yopt = _anneal_lbfgs(y, taus_t, mits_base, time.time(), 1000.0 if tl > 20 else tl)
        # time limit inside anneal uses absolute? we pass elapsed-style: use t0-like
        # _anneal checks time.time()-t0>limit; we gave t0=time.time() so ok
        v = _max_cos(yopt)
        cands.append((v, yopt.copy()))
        if v < best_val:
            best_val = v
            best_x = yopt.copy()
        # basin-hopping acceptance for exploration
        if v < cur_val:
            cur_x, cur_val = yopt.copy(), v
        else:
            # accept worse with prob exp(-(v-cur)/Temp)
            if rng.random() < np.exp(-(v - cur_val) / max(Temp, 1e-6)):
                cur_x, cur_val = yopt.copy(), v
        Temp *= 0.985
        Temp = max(Temp, 0.0008)
        # occasional restart to best
        if it % 9 == 0 and rng.random() < 0.5:
            cur_x, cur_val = best_x.copy(), best_val

    cands.sort(key=lambda kv: kv[0])
    # keep top distinct + base
    uniq = []
    for v, xc in cands:
        ok = True
        for _, u in uniq:
            if _max_cos((xc + u) * 0.5) < 1e-10:
                ok = False
                break
        if ok:
            uniq.append((v, xc))
        if len(uniq) >= 5:
            break
    polished_x, polished_v = best_x.copy(), best_val
    for v, xc in uniq[:3]:
        if time.time() > HARD_END - 8:
            break
        share = max(10.0, (HARD_END - 6.0 - time.time()) / 2.0)
        deadline = min(HARD_END - 2.0, time.time() + share)
        xp, vp = _lp_polish(xc, deadline)
        if vp < polished_v:
            polished_v, polished_x = vp, xp.copy()
        if vp < best_val:
            best_val, best_x = vp, xp.copy()
    out = _normalize(polished_x)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    out = out / nrm
    return out
