import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _anneal(X, N, tau0, tau1, lr0, lr1, mu, t_end):
    n, d = X.shape
    v = np.zeros_like(X)
    ltau = np.log(tau1 / tau0) if tau1 != tau0 else 0.0
    llr = np.log(lr1 / lr0) if lr1 != lr0 else 0.0
    for it in range(N):
        if time.time() > t_end:
            break
        frac = it / max(1, N - 1)
        tau = tau0 * np.exp(ltau * frac)
        lr = lr0 * np.exp(llr * frac)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # softmax weights
        with np.errstate(over='ignore', under='ignore'):
            P = np.exp((G - m) / tau)
        np.fill_diagonal(P, 0.0)
        Grad = P @ X
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        rn = np.sqrt(np.sum(Gt * Gt, axis=1))
        mn = float(np.max(rn))
        if not np.isfinite(mn) or mn < 1e-14:
            break
        Gt /= mn
        v = mu * v + (1.0 - mu) * Gt
        X = X - lr * v
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
    return X

def _find_holes(X, exclude, rng, k, n_cand=24000):
    n, d = X.shape
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-30
    mask = np.ones(n, dtype=bool)
    mask[np.asarray(exclude, dtype=int)] = False
    Xe = X[mask]
    D = C @ Xe.T
    md = np.max(D, axis=1)
    order = np.argsort(md)
    sel = []
    for idx in order:
        if len(sel) >= k:
            break
        c = C[idx]
        ok = True
        for s in sel:
            if float(c @ s) > 0.92:
                ok = False
                break
        if ok:
            sel.append(c.copy())
    # refine each hole away from closest point
    out = []
    for h in sel:
        hh = h.copy()
        for _ in range(40):
            dots = Xe @ hh
            ci = int(np.argmax(dots))
            xc = Xe[ci]
            cc = float(dots[ci])
            t = xc - cc * hh
            tn = float(np.linalg.norm(t))
            if tn < 1e-12:
                break
            t /= tn
            hh = hh - 0.012 * t
            hh /= np.linalg.norm(hh) + 1e-30
        out.append(hh)
    return out

def _greedy_shave(x, steps, t_end, rng):
    n, d = x.shape
    eps = 1e-3
    bestV = _max_cos(x)
    bestX = x.copy()
    for it in range(steps):
        if time.time() > t_end:
            break
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < bestV:
            bestV = m
            bestX = x.copy()
        flat = int(np.argmax(G))
        i = flat // n
        j = flat % n
        cij = float(x[i] @ x[j])
        ti = x[j] - cij * x[i]
        ni = float(np.linalg.norm(ti))
        if ni < 1e-12:
            ti = rng.standard_normal(d)
            ti -= float(ti @ x[i]) * x[i]
            ni = float(np.linalg.norm(ti)) + 1e-12
        ti /= ni
        tj = x[i] - cij * x[j]
        nj = float(np.linalg.norm(tj))
        if nj < 1e-12:
            tj = rng.standard_normal(d)
            tj -= float(tj @ x[j]) * x[j]
            nj = float(np.linalg.norm(tj)) + 1e-12
        tj /= nj
        improved = False
        for scale in (eps, eps * 0.5, eps * 0.25):
            vi = x[i] - scale * ti
            vi /= np.linalg.norm(vi) + 1e-30
            vj = x[j] - scale * tj
            vj /= np.linalg.norm(vj) + 1e-30
            oi = x[i].copy(); oj = x[j].copy()
            x[i] = vi; x[j] = vj
            g2 = x @ x.T
            np.fill_diagonal(g2, -2.0)
            m2 = float(np.max(g2))
            if m2 < m - 1e-12:
                if m2 < bestV:
                    bestV = m2
                    bestX = x.copy()
                eps = min(scale * 1.1, 3e-3)
                improved = True
                break
            else:
                x[i] = oi; x[j] = oj
        if not improved:
            eps *= 0.96
            if eps < 5e-07:
                eps = 5e-07
                k = int(rng.integers(0, n))
                rnd = rng.standard_normal(d)
                rnd -= float(rnd @ x[k]) * x[k]
                rn = float(np.linalg.norm(rnd))
                if rn > 1e-12:
                    rnd /= rn
                    xn = x[k] + 1e-06 * rnd
                    xn /= np.linalg.norm(xn) + 1e-30
                    x[k] = xn
    return bestX, bestV

def _lp_polish(x, cur, t_end, r_init=3e-4):
    n, d = x.shape
    best = x.copy()
    best_val = cur
    radius = r_init
    iu_r, iu_c = np.triu_indices(n, 1)
    while time.time() < t_end and radius > 1e-13:
        if t_end - time.time() < 2.0:
            break
        Gfull = x @ x.T
        c = Gfull[iu_r, iu_c]
        m = float(np.max(c))
        delta = max(40.0 * radius, 4e-7)
        sel = c > m - delta
        cnt = int(np.count_nonzero(sel))
        if cnt > 7000:
            idx = np.argpartition(c, -7000)[-7000:]
            mk = np.zeros_like(sel)
            mk[idx] = True
            sel = mk
            cnt = 7000
        if cnt == 0:
            break
        I = iu_r[sel]; J = iu_c[sel]
        p = int(I.size)
        csel = c[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-30
        v = _max_cos(y)
        if v < best_val:
            x = y
            best = y.copy()
            best_val = v
            radius = min(radius * 1.5, 6e-4)
        else:
            radius *= 0.5
    return best, best_val

def solve():
    t0 = time.time()
    t_hop_end = t0 + 102.0
    t_shave_end = t0 + 118.0
    t_lp_end = t0 + 163.0
    t_end_total = t0 + 172.0
    rng = np.random.default_rng(1)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                continue
            cc = cc / (np.linalg.norm(cc, axis=1, keepdims=True) + 1e-30)
            if not np.all(np.isfinite(cc)):
                continue
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv
                bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((453, 5))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-30
        bestV = _max_cos(bestX)
    n, d = bestX.shape
    iu_r, iu_c = np.triu_indices(n, 1)

    taus = [(0.004, 0.0004), (0.002, 0.00015), (0.001, 0.0001), (0.006, 0.0006)]
    lrs = [(0.004, 0.0003), (0.0025, 0.0002), (0.0015, 0.00012)]
    sigmas = [0.004, 0.008, 0.012, 0.002, 0.006, 0.015]
    mus = [0.85, 0.88, 0.92]
    it = 0
    while time.time() < t_hop_end:
        it += 1
        if t_hop_end - time.time() < 0.8:
            break
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        pmax = np.max(G, axis=1)
        order = np.argsort(-pmax)
        op = int(rng.integers(0, 4))
        inner_end = min(time.time() + 3.0, t_hop_end)
        tau0, tau1 = taus[it % len(taus)]
        lr0, lr1 = lrs[it % len(lrs)]
        mu = float(mus[it % len(mus)])
        y = bestX.copy()
        if op == 0:
            # coordinated jitter of worst cluster + randoms
            m = int(rng.choice([5, 8, 12]))
            worst = order[:m]
            sset = set(int(v) for v in worst)
            # add neighbours of worst[0]
            w0 = int(order[0])
            row = G[w0].copy()
            nb = np.argpartition(row, -6)[-6:]
            for t_ in nb:
                sset.add(int(t_))
            for _ in range(3):
                sset.add(int(rng.integers(0, n)))
            sidx = np.array(list(sset), dtype=int)
            sigma = float(sigmas[it % len(sigmas)])
            noise = rng.standard_normal((len(sidx), d)) * sigma
            for k2, idx in enumerate(sidx):
                vv = noise[k2] - float(np.dot(noise[k2], y[idx])) * y[idx]
                y[idx] = y[idx] + vv
            y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-30
            y = _anneal(y, 280, tau0, tau1, lr0, lr1, mu, inner_end)
        elif op == 1:
            # multi-hole reinsertion of 2-3 worst points
            k = int(rng.choice([1, 2, 2, 3]))
            victims = [int(v) for v in order[:k]]
            try:
                holes = _find_holes(bestX, victims, rng, k)
            except Exception:
                continue
            if len(holes) < k:
                continue
            for vi, h in zip(victims, holes):
                y[vi] = h
            y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-30
            y = _anneal(y, 320, tau0, tau1, lr0, lr1, mu, inner_end)
        elif op == 2:
            # worst-pair split + neighbourhood shake
            flat = int(np.argmax(G))
            wi = flat // n; wj = flat % n
            # split pair apart
            cij = float(np.dot(y[wi], y[wj]))
            cij = max(-1.0, min(1.0, cij))
            for (a_, b_) in ((wi, wj), (wj, wi)):
                # direction of a away from b
                ref = bestX[b_]
                t = cij * bestX[a_] - ref
                # actually push a away from b: component of a orthogonal to b? use standard
                t2 = bestX[b_] - cij * bestX[a_]
                tn = float(np.linalg.norm(t2))
                if tn > 1e-12:
                    t2 /= tn
                    s = float(rng.choice([0.001, 0.002, 0.0035]))
                    y[a_] = y[a_] - s * t2
                    y[a_] /= np.linalg.norm(y[a_]) + 1e-30
            # shake neighbours
            for ww in (wi, wj):
                row = G[ww]
                nb = np.argpartition(row, -5)[-5:]
                for u in nb:
                    u = int(u)
                    if u == wi or u == wj:
                        continue
                    nz = rng.standard_normal(d) * 0.003
                    nz -= float(np.dot(nz, y[u])) * y[u]
                    y[u] = y[u] + nz
                    y[u] /= np.linalg.norm(y[u]) + 1e-30
            y = _anneal(y, 240, tau0, tau1, lr0, lr1, mu, inner_end)
        else:
            # global small shake + anneal (basin escape)
            sigma = 0.0025
            nz = rng.standard_normal((n, d)) * sigma
            nz -= np.sum(nz * y, axis=1, keepdims=True) * y
            # only shake a random half to keep some structure
            pick = rng.random(n) < 0.4
            y[pick] += nz[pick]
            y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-30
            y = _anneal(y, 300, tau0, tau1, lr0, lr1, mu, inner_end)
        if not np.all(np.isfinite(y)):
            continue
        v = _max_cos(y)
        if v < bestV:
            bestX = y
            bestV = v

    # greedy shave interlude
    if time.time() < t_shave_end:
        try:
            xg, vg = _greedy_shave(bestX.copy(), 600, t_shave_end, rng)
            if vg < bestV:
                bestX, bestV = xg, vg
        except Exception:
            pass

    # iterated LP polish with kicks
    if time.time() < t_lp_end:
        try:
            xlp, vlp = _lp_polish(bestX, bestV, min(t_lp_end, time.time() + 40.0), r_init=3e-4)
            if vlp < bestV:
                bestX, bestV = xlp, vlp
        except Exception:
            pass
        # small kick + repolish if time remains
        while time.time() < t_lp_end - 3.0:
            G = bestX @ bestX.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            i = flat // n; j = flat % n
            y = bestX.copy()
            for kk in (i, j):
                nz = rng.standard_normal(d) * 2e-5
                nz -= float(np.dot(nz, y[kk])) * y[kk]
                y[kk] = y[kk] + nz
                y[kk] /= np.linalg.norm(y[kk]) + 1e-30
            try:
                x2, v2 = _lp_polish(y, bestV, min(t_lp_end, time.time() + 12.0), r_init=1.2e-4)
            except Exception:
                break
            if v2 < bestV:
                bestX, bestV = x2, v2
            else:
                # try different radius once
                break

    # final micro hill climbing on worst pair
    sigmas2 = [3e-5, 1e-5, 1e-4, 2e-4, 5e-6, 5e-5]
    k = 0
    while time.time() < t_end_total:
        if k > 30000:
            break
        G = bestX @ bestX.T
        np.fill_diagonal(G, -2.0)
        flat = int(np.argmax(G))
        i = flat // n; j = flat % n
        sigma = float(sigmas2[k % len(sigmas2)])
        k += 1
        y = bestX.copy()
        for kk in (i, j):
            nz = rng.standard_normal(5) * sigma
            nz -= float(np.dot(nz, y[kk])) * y[kk]
            y[kk] = y[kk] + nz
            nn = float(np.linalg.norm(y[kk]))
            if nn == 0 or not np.isfinite(nn):
                y[kk] = bestX[kk]
            else:
                y[kk] /= nn
        if (k % 7) == 0:
            r = int(rng.integers(0, n))
            nz = rng.standard_normal(5) * (sigma * 2.0)
            nz -= float(np.dot(nz, y[r])) * y[r]
            y[r] = y[r] + nz
            nn = float(np.linalg.norm(y[r]))
            if nn != 0 and np.isfinite(nn):
                y[r] /= nn
        v = _max_cos(y)
        if v < bestV:
            bestX = y
            bestV = v

    bestX = bestX / (np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-30)
    return bestX
