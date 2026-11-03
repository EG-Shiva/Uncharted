import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)


def smooth_refine(X, steps, T0, T1, lr0, lr1, mu=0.9):
    n, d = X.shape
    V = np.zeros_like(X)
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = float(G.max())
        W = np.exp((G - mx) / T)
        np.fill_diagonal(W, 0.0)
        ss = W.sum()
        if not np.isfinite(ss) or ss == 0:
            break
        W /= ss
        Grad = W @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        gn = float(np.linalg.norm(Gt))
        if not np.isfinite(gn) or gn == 0:
            break
        Gt = Gt / (gn / np.sqrt(float(n)))
        V = mu * V + Gt
        X = X - lr * V
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def lp_polish(x, seconds, radius_init=1e-3, cap=1500):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt > cap:
            idx = np.argpartition(c, -cap)[-cap:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        elif cnt == 0:
            break
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def raw_hole(others, rng, n_samples=4000):
    d = others.shape[1]
    cand = rng.standard_normal(size=(n_samples, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    md = dots.max(axis=1)
    j = int(np.argmin(md))
    return cand[j]


def refine_hole(others, x0, steps=200, T0=0.01, T1=0.001, lr=0.05):
    x = x0.copy() / np.linalg.norm(x0)
    l0 = np.log(T0); l1 = np.log(T1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        T = float(np.exp(l0 + (l1 - l0) * f))
        dots = others @ x
        mx = float(dots.max())
        w = np.exp((dots - mx) / T)
        sw = w.sum()
        if not np.isfinite(sw) or sw == 0:
            break
        w /= sw
        g = w @ others
        gt = g - float(g @ x) * x
        nrm = float(np.linalg.norm(gt))
        if not np.isfinite(nrm) or nrm < 1e-12:
            break
        x = x - lr * gt / nrm * 0.2
        x /= np.linalg.norm(x)
    return x


def contacts_info(x, tol=2e-4):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    mc = float(G.max())
    cc = (G > mc - tol).sum(axis=1)
    ppm = G.max(axis=1)
    return G, mc, cc, ppm


def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline = t0 + BUDGET
    EXPL_END = t0 + 132.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    inits = []
    for cand in (xb, xr):
        if cand is not None and cand.shape == (95, 13):
            cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
            if np.all(np.isfinite(cand)):
                inits.append(cand)
    if not inits:
        r0 = rng.standard_normal((95, 13))
        r0 /= np.linalg.norm(r0, axis=1, keepdims=True)
        inits.append(r0)
    vals = [max_cos(c) for c in inits]
    bi = int(np.argmin(vals))
    x_best = inits[bi].copy()
    best_val = float(vals[bi])
    n, d = x_best.shape

    # elite pool for diversity (keep 2nd best start)
    elites = [(best_val, x_best.copy())]

    trial = 0
    while time.time() < EXPL_END and time.time() < deadline - 35:
        trial += 1
        remain = EXPL_END - time.time()
        if remain < 2.0:
            break
        # pick base: mostly best, sometimes second elite
        if len(elites) > 1 and (trial % 9 == 8) and rng.random() < 0.5:
            # pick random elite
            ei = int(rng.integers(0, len(elites)))
            base = elites[ei][1]
        else:
            base = x_best
        Xt = base.copy()
        mode = trial % 8
        if mode == 0:
            sigma = float(rng.choice([0.01, 0.02, 0.04, 0.08]))
            Xt = Xt + sigma * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, 800, 0.015, 0.001, 0.03, 0.003, mu=0.9)
        elif mode == 1:
            # worst-point kill + refined hole
            _, _, _, ppm = contacts_info(Xt)
            order = np.argsort(-ppm)
            k = int(rng.integers(1, 4))
            pool = list(order[:10]) + list(rng.choice(n, size=6, replace=False))
            idx = np.array(rng.choice(pool, size=min(k, len(pool)), replace=False))
            keep = np.ones(n, dtype=bool); keep[idx] = False
            others = Xt[keep]
            for j in idx:
                h0 = raw_hole(others, rng, n_samples=4000)
                h = refine_hole(others, h0, steps=250, T0=0.008, T1=0.0008, lr=0.08)
                # choose better of h0,h
                m0 = float((h0[None, :] @ others.T).max())
                m1 = float((h[None, :] @ others.T).max())
                Xt[j] = h if m1 < m0 else h0
                others = np.concatenate([others, Xt[j][None, :]], axis=0)
            Xt += 0.006 * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, 900, 0.012, 0.0008, 0.025, 0.002, mu=0.9)
        elif mode == 2:
            # hub neighbourhood block move
            G, mc, cc, ppm = contacts_info(Xt)
            hub = int(np.argmax(cc + rng.random(n) * 0.5))
            dots_h = G[hub].copy()
            dots_h[hub] = -10
            nn = int(rng.integers(6, 14))
            neigh = np.argsort(-dots_h)[:nn]
            grp = np.concatenate([[hub], neigh])
            common = rng.standard_normal(d,)
            common /= np.linalg.norm(common)
            s_c = float(rng.uniform(0.05, 0.18))
            s_i = float(rng.uniform(0.01, 0.05))
            Xt[grp] += s_c * common[None, :] + s_i * rng.standard_normal((len(grp), d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, 1000, 0.016, 0.0009, 0.03, 0.0022, mu=0.9)
        elif mode == 3:
            # forced separation of worst edges
            G = Xt @ Xt.T
            np.fill_diagonal(G, -2.0)
            iu = np.triu_indices(n, 1)
            c = G[iu]
            topk = np.argpartition(-c, min(8, len(c)-1))[:8]
            # push apart 2-4 worst pairs
            nk = int(rng.integers(1, 4))
            sel = rng.choice(topk, size=nk, replace=False)
            for s_ in sel:
                i, j = int(iu[0][s_]), int(iu[1][s_])
                dij = Xt[i] - Xt[j]
                nnr = float(np.linalg.norm(dij))
                if nnr < 1e-9:
                    dij = rng.standard_normal(d)
                    nnr = float(np.linalg.norm(dij))
                dij /= nnr
                eps = float(rng.uniform(0.02, 0.09))
                Xt[i] += eps * dij
                Xt[j] -= eps * dij
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, 900, 0.014, 0.0008, 0.028, 0.002, mu=0.9)
        elif mode == 4:
            # rattler / low-contact relocation
            G, mc, cc, ppm = contacts_info(Xt, tol=1e-3)
            low = np.where(cc <= 5)[0]
            if len(low) == 0:
                low = np.argsort(cc)[:6]
            k = min(len(low), int(rng.integers(1, 4)))
            idx = rng.choice(low, size=k, replace=False)
            keep = np.ones(n, dtype=bool); keep[idx] = False
            others = Xt[keep]
            for j in idx:
                h0 = raw_hole(others, rng, n_samples=5000)
                h = refine_hole(others, h0, steps=250, T0=0.008, T1=0.0008, lr=0.08)
                m0 = float((h0[None, :] @ others.T).max())
                m1 = float((h[None, :] @ others.T).max())
                Xt[j] = h if m1 < m0 else h0
                others = np.concatenate([others, Xt[j][None, :]], axis=0)
            Xt += 0.008 * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, 1000, 0.015, 0.0009, 0.03, 0.002, mu=0.9)
        elif mode == 5:
            sigma = float(rng.uniform(0.25, 0.55))
            Xt = Xt + sigma * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, 1300, 0.024, 0.001, 0.04, 0.0025, mu=0.9)
        elif mode == 6:
            # subset shake
            ksub = int(rng.integers(5, 18))
            idx = rng.choice(n, size=ksub, replace=False)
            sigma = float(rng.uniform(0.05, 0.20))
            Xt[idx] += sigma * rng.standard_normal((ksub, d))
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, 900, 0.013, 0.0008, 0.027, 0.002, mu=0.9)
        else:
            # random-subspace rotation of a block (topology change)
            ksub = int(rng.integers(8, 24))
            idx = rng.choice(n, size=ksub, replace=False)
            a = rng.standard_normal(d); a /= np.linalg.norm(a)
            b = rng.standard_normal(d); b -= float(b @ a) * a; b /= np.linalg.norm(b)
            ang = float(rng.uniform(0.1, 0.6))
            ca, sa = np.cos(ang), np.sin(ang)
            for j in idx:
                xa, xb_ = float(Xt[j] @ a), float(Xt[j] @ b)
                Xt[j] = Xt[j] + (xa * (ca - 1) - xb_ * sa) * a + (xa * sa + xb_ * (ca - 1)) * b
            Xt += 0.01 * rng.standard_normal(Xt.shape)
            Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
            Xt = smooth_refine(Xt, 1000, 0.016, 0.0009, 0.03, 0.002, mu=0.9)

        v_pre = max_cos(Xt)
        if v_pre > best_val + 0.03:
            continue
        t_rem = deadline - time.time()
        if t_rem < 6:
            break
        if v_pre < best_val:
            ps = 2.2
        elif v_pre < best_val + 0.005:
            ps = 1.4
        else:
            ps = 0.8
        ps = min(ps, max(0.6, t_rem - 32))
        Xt2 = lp_polish(Xt, seconds=ps, radius_init=1e-3)
        v2 = max_cos(Xt2)
        if v2 < best_val - 1e-12:
            x_best = Xt2
            best_val = v2
            elites.append((v2, Xt2.copy()))
            elites.sort(key=lambda e: e[0])
            elites = elites[:4]
        elif v2 < elites[-1][0] and len(elites) < 4:
            elites.append((v2, Xt2.copy()))
            elites.sort(key=lambda e: e[0])

    t_rem = deadline - time.time()
    if t_rem > 3:
        x_best = lp_polish(x_best, seconds=max(1.0, t_rem - 1.0), radius_init=2e-3)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
