import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _stats(x, delta=1e-3):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    pm = G.max(axis=1)
    deg = (G > m - delta).sum(axis=1).astype(float)
    return G, m, pm, deg

def _anneal(x, steps, beta0, beta1, lr0, lr1, mu):
    n = x.shape[0]
    x = _norm(x)
    vel = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    ll = np.log(lr1 / lr0) if abs(lr1 - lr0) > 1e-18 else 0.0
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * np.exp(ll * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        r = W.sum(axis=1, keepdims=True)
        np.maximum(r, 1e-30, out=r)
        A = (W @ x) / r
        adot = np.sum(A * x, axis=1, keepdims=True)
        At = A - adot * x
        vel = mu * vel + (1.0 - mu) * At
        bias = 1.0 - mu ** (s + 1)
        vh = vel / max(bias, 1e-12)
        x = x - lr * vh
        x = _norm(x)
    return x

def _find_holes(Xkeep, k, rng, n_cand=12000, sep=0.88):
    cand = rng.standard_normal((n_cand, 4))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    best = np.empty(n_cand)
    cs = 4000
    for a in range(0, n_cand, cs):
        b = min(a + cs, n_cand)
        D = cand[a:b] @ Xkeep.T
        best[a:b] = D.max(axis=1)
    order = np.argsort(best)  # smallest max-dot = largest hole first
    sel = []
    for idx in order:
        c = cand[idx]
        ok = True
        for s in sel:
            if float(s @ c) > sep:
                ok = False
                break
        if ok:
            sel.append(c)
            if len(sel) >= k:
                break
    j = 0
    while len(sel) < k and j < len(order):
        sel.append(cand[order[j]])
        j += 1
    return np.array(sel)

def _choose_delete(pm, deg, k, rng):
    n = pm.shape[0]
    # combine stress: rank by pm primarily, deg secondarily, with random tie
    jitter = 1e-6 * rng.standard_normal(n)
    score = pm + 0.002 * (deg / max(1.0, deg.max())) + jitter
    # branch over deletion orders: sometimes degree-first
    if rng.random() < 0.35:
        score = deg + 200.0 * pm + jitter
    order = np.argsort(-score)
    pool_sz = min(n, max(k + 6, 16))
    pool = order[:pool_sz]
    if k >= pool_sz:
        return order[:k].copy()
    # sample k from pool weighted to top
    perm = rng.permutation(pool)
    # ensure at least one top-3 included
    sel = set(perm[:k].tolist())
    if len(sel) < k:
        pass
    sel = list(sel)[:k]
    # force inclusion of worst point with prob 0.7
    if rng.random() < 0.7 and int(order[0]) not in sel:
        sel[0] = int(order[0])
    return np.array(sel, dtype=int)

def _slack_sig(x, K=900):
    n = x.shape[0]
    iu = np.triu_indices(n, 1)
    c = (x @ x.T)[iu]
    m = float(c.max())
    if K >= c.size:
        top = np.sort(c)[::-1]
    else:
        idx = np.argpartition(-c, K)[:K]
        top = np.sort(c[idx])[::-1]
    slack = m - top
    return m, slack

def _lp_polish(x, seconds, r0=8e-4):
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    xb = x.copy()
    if seconds <= 0:
        return xb
    start = time.time()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0]
            m = float(c.max())
            thresh = m - max(20.0 * radius, 1e-9)
            sel = c > thresh
            if not np.any(sel):
                break
            I = iu0[0][sel]
            J = iu0[1][sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            ci = I[:, None] * d + np.arange(d)
            cj = J[:, None] * d + np.arange(d)
            cols = np.concatenate([ci, cj, np.full((p, 1), n * d)], axis=1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y = _norm(y)
        try:
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            best = v
            xb = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb

def solve():
    t0 = time.time()
    rng = np.random.default_rng(20261031)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            z = rng.standard_normal((292, 4))
            x0 = z / np.linalg.norm(z, axis=1, keepdims=True)
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (292, 4) or not np.all(np.isfinite(x0)):
        z = rng.standard_normal((292, 4))
        x0 = z / np.linalg.norm(z, axis=1, keepdims=True)
    x0 = _norm(x0)
    try:
        r0 = np.load("record.npy")
        r0 = np.asarray(r0, dtype=float)
        if r0.shape == (292, 4) and np.all(np.isfinite(r0)):
            r0 = _norm(r0)
            if _max_cos(r0) < _max_cos(x0):
                x0 = r0
    except Exception:
        pass
    best = x0.copy()
    try:
        best_val = _max_cos(best)
    except Exception:
        best_val = 2.0
    cur = best.copy()
    cur_val = best_val
    n = 292
    sigs = []
    try:
        _, s0 = _slack_sig(best)
        sigs.append(s0.copy())
    except Exception:
        pass
    EXPLORE_END = 132.0
    T0, T1 = 0.0020, 1.5e-05
    it = 0
    while time.time() - t0 < EXPLORE_END:
        it += 1
        now = time.time() - t0
        if now > 164:
            break
        frac = now / max(1e-9, EXPLORE_END)
        Temp = T0 * (T1 / T0) ** frac
        u = rng.random()
        xt = None
        try:
            Gc, mc, pmc, degc = _stats(cur)
        except Exception:
            cur = best.copy()
            cur_val = best_val
            continue
        try:
            if u < 0.45:
                # down-up ladder: delete k then rebuild into holes
                if rng.random() < 0.25:
                    k = int(rng.integers(5, 13))  # deep dip to 280-287
                else:
                    k = int(rng.integers(1, 9))
                k = min(k, 14)
                dels = _choose_delete(pmc, degc, k, rng)
                mask = np.ones(n, dtype=bool)
                mask[dels] = False
                Xkeep = cur[mask]
                # relax smaller code (neighbouring N)
                Xs = _anneal(Xkeep, steps=320, beta0=16.0, beta1=380.0, lr0=0.040, lr1=0.0030, mu=0.85)
                # rebuild to 292 by inserting into largest optimised holes
                holes = _find_holes(Xs, k=k, rng=rng, n_cand=12000, sep=0.88)
                xt = np.empty((n, 4))
                xt[mask] = Xs
                xt[dels] = holes + 0.004 * rng.standard_normal((k, 4))
                xt = _norm(xt)
                xt = _anneal(xt, steps=480, beta0=18.0, beta1=450.0, lr0=0.040, lr1=0.0028, mu=0.86)
            elif u < 0.75:
                # up-down overshoot: add j then delete different set
                j = int(rng.integers(1, 9))  # N=293..300
                holes = _find_holes(cur, k=j, rng=rng, n_cand=12000, sep=0.86)
                Xup = np.vstack([cur, holes + 0.004 * rng.standard_normal((j, 4))])
                Xup = _norm(Xup)
                Xup = _anneal(Xup, steps=320, beta0=16.0, beta1=380.0, lr0=0.040, lr1=0.0030, mu=0.85)
                # delete j points with nontrivial cycle: stress-based, force >=1 original deleted
                try:
                    Gu, mu_, pmu, degu = _stats(Xup)
                except Exception:
                    continue
                # deletion scores on overshoot code
                cand_del = _choose_delete(pmu, degu, j, rng)
                # ensure not exactly the inserted block
                nup = n + j
                inserted = np.arange(n, nup)
                if set(cand_del.tolist()) == set(inserted.tolist()) and j > 1:
                    # swap one out for worst original
                    orig_order = np.argsort(-pmu[:n])
                    cand_del[0] = int(orig_order[0])
                maskup = np.ones(nup, dtype=bool)
                maskup[cand_del] = False
                Xdown = Xup[maskup]
                xt = _anneal(Xdown, steps=480, beta0=18.0, beta1=450.0, lr0=0.040, lr1=0.0028, mu=0.86)
            else:
                # direct neighbouring-N topology kick: kill-fill worst into hole + cluster jitter
                ns = int(rng.integers(2, 6))
                order = np.argsort(-(pmc + 0.002 * degc))
                pool = order[:16]
                src = rng.choice(pool, size=ns, replace=False)
                mask2 = np.ones(n, dtype=bool)
                mask2[src] = False
                Xkeep = cur[mask2]
                holes = _find_holes(Xkeep, k=ns, rng=rng, n_cand=14000, sep=0.88)
                xt = cur.copy()
                for s_, h_ in zip(src, holes):
                    xt[int(s_)] = h_ + 0.005 * rng.standard_normal(4)
                # small cluster shake to change contact graph
                c0 = int(rng.integers(0, n))
                kcl = int(rng.integers(5, 14))
                dots = xt @ xt[c0]
                S = np.argsort(-dots)[:kcl]
                xt[S] += 0.05 * rng.standard_normal((kcl, 4))
                xt = _norm(xt)
                xt = _anneal(xt, steps=520, beta0=20.0, beta1=480.0, lr0=0.040, lr1=0.0028, mu=0.86)
            # fine squeeze
            xt = _anneal(xt, steps=150, beta0=160.0, beta1=1100.0, lr0=0.009, lr1=0.0012, mu=0.88)
            vt = _max_cos(xt)
        except Exception:
            continue
        # slack-signature novelty (contact-graph comparison)
        novel = True
        try:
            _, sig = _slack_sig(xt)
            if len(sigs) > 0:
                dmin = min(float(np.mean((sig - s) ** 2)) for s in sigs[-24:])
                if dmin < 1e-14:
                    novel = False
            if len(sigs) < 60:
                sigs.append(sig.copy())
            else:
                if novel:
                    sigs[int(rng.integers(0, len(sigs)))] = sig.copy()
        except Exception:
            novel = True
        if vt < best_val - 1e-12:
            # score intermediate only after polish: short LP if novel or clearly better
            do_lp = (time.time() - t0) < 160.0 and (novel or vt < best_val - 5e-07)
            if do_lp and vt < best_val - 1e-07:
                remain = EXPLORE_END - (time.time() - t0)
                if remain > 1.2:
                    xp = _lp_polish(xt, seconds=min(0.9, max(0.35, remain * 0.06)), r0=8e-4)
                    try:
                        vp = _max_cos(xp)
                    except Exception:
                        vp = vt
                    if vp < vt:
                        xt, vt = xp, vp
                    try:
                        _, sig2 = _slack_sig(xt)
                        sigs.append(sig2.copy())
                        if len(sigs) > 80:
                            sigs = sigs[-80:]
                    except Exception:
                        pass
            if vt < best_val - 1e-12:
                best = xt.copy()
                best_val = vt
                cur = xt.copy()
                cur_val = vt
                continue
        # Metropolis on cur basin
        if vt < cur_val or rng.random() < np.exp((cur_val - vt) / max(Temp, 1e-12)):
            cur = xt.copy()
            cur_val = vt
        if cur_val > best_val + 0.006 and (it % 12 == 0) and rng.random() < 0.5:
            cur = best.copy() + 0.015 * rng.standard_normal((n, 4))
            cur = _norm(cur)
            try:
                cur_val = _max_cos(cur)
            except Exception:
                cur_val = best_val + 0.001
    # intensify around best
    while time.time() - t0 < 156.0:
        try:
            sigma = float(rng.choice(np.array([0.008, 0.018, 0.035])))
            xt = best + sigma * rng.standard_normal((n, 4))
            xt = _norm(xt)
            xt = _anneal(xt, steps=420, beta0=30.0, beta1=650.0, lr0=0.030, lr1=0.002, mu=0.87)
            xt = _anneal(xt, steps=130, beta0=260.0, beta1=1300.0, lr0=0.008, lr1=0.001, mu=0.88)
            vt = _max_cos(xt)
        except Exception:
            break
        if vt < best_val - 1e-12:
            best = xt.copy()
            best_val = vt
        if time.time() - t0 > 156:
            break
    remain = min(17.0, max(0.0, 172.0 - (time.time() - t0) - 0.5))
    if remain > 1.0:
        try:
            best = _lp_polish(best, seconds=remain, r0=7e-4)
        except Exception:
            pass
    best = _norm(best)
    return best
