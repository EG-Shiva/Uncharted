import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def _lp_polish(x, deadline, radius_init=1e-3):
    x = _norm(np.asarray(x, dtype=np.float64))
    n, d = x.shape
    best = _max_cos(x)
    radius = radius_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        cfull = G[iu0]
        m = float(cfull.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = cfull > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 1100:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:1100]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        colI = (I[:, None] * d + np.arange(d)).ravel()
        colJ = (J[:, None] * d + np.arange(d)).ravel()
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        cols[0::(2*d+1)] = 0  # placeholder, overwritten below
        # build cols vectorized
        cols = np.concatenate([ (I[:, None]*d + np.arange(d)).ravel() if False else None ])
        # simpler: build with concatenate of blocks
        cI = (I[:, None] * d + np.arange(d))
        cJ = (J[:, None] * d + np.arange(d))
        cT = np.full((p, 1), n*d)
        cols = np.concatenate([cI, cJ, cT], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _quench(X0, steps, b0, b1, lr0, lr1, mu, rng, noise=0.0):
    X = X0.copy()
    V = np.zeros_like(X)
    for t in range(steps):
        f = t / max(steps-1, 1)
        beta = b0 * (b1/b0)**f if b0>0 and b1>0 else b0
        lr = lr0 * (lr1/lr0)**f if lr0>0 and lr1>0 else lr0
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        # avoid overflow: clip argument
        A = beta*(G-m)
        np.clip(A, -50.0, 0.0, out=A)
        W = np.exp(A)
        np.fill_diagonal(W, 0.0)
        Gw = W @ X
        dot = np.sum(Gw*X, axis=1, keepdims=True)
        Gt = Gw - dot*X
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True)+1e-12
        Gt = Gt / nrm
        if noise>0:
            Z = rng.standard_normal(X.shape)
            Z -= np.sum(Z*X, axis=1, keepdims=True)*X
            zn = np.linalg.norm(Z, axis=1, keepdims=True)+1e-12
            Z /= zn
            Gt = (1.0-noise)*Gt + noise*Z
            # re-project
            Gt -= np.sum(Gt*X, axis=1, keepdims=True)*X
            n2 = np.linalg.norm(Gt, axis=1, keepdims=True)+1e-12
            Gt /= n2
        V = mu*V + (1.0-mu)*Gt
        # keep V tangential
        V -= np.sum(V*X, axis=1, keepdims=True)*X
        X = X - lr*V
        X = _norm(X)
    return X

def _hole(Y, rng, n_cand=4000):
    d = Y.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cand = _norm(cand)
    dots = cand @ Y.T
    md = dots.max(axis=1)
    return cand[np.argmin(md)]

def solve():
    rng = np.random.default_rng(20261101)
    t0 = time.time()
    LIM = 172.0
    SEARCH_END = t0 + 148.0
    n, d = 55, 21
    best = None
    best_score = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (55, 21):
                a = _norm(a)
                s = _max_cos(a)
                if s < best_score:
                    best_score = s
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_score = _max_cos(best)
    x_best = best.copy()
    cur = best.copy()
    cur_score = best_score
    T0, T1 = 0.0035, 0.00012
    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        remain = LIM - (time.time()-t0)
        if remain < 8:
            break
        frac = (time.time()-t0)/max(SEARCH_END-t0, 1e-9)
        frac = min(max(frac, 0.0), 1.0)
        Temp = T0*(T1/T0)**frac
        # base: mostly cur, periodic return to best
        if trial % 25 == 0:
            base = x_best
        else:
            base = cur
        prop = base.copy()
        mode = trial % 6
        if mode == 0:
            sig = 0.02 + 0.03*rng.random()
            prop = _norm(prop + sig*rng.standard_normal(prop.shape))
            prop = _quench(prop, steps=450, b0=80.0, b1=600.0, lr0=0.014, lr1=0.0012, mu=0.72, rng=rng, noise=0.015)
        elif mode == 1:
            sig = 0.09 + 0.16*rng.random()
            prop = _norm(prop + sig*rng.standard_normal(prop.shape))
            prop = _quench(prop, steps=650, b0=40.0, b1=800.0, lr0=0.02, lr1=0.0015, mu=0.8, rng=rng, noise=0.04)
        elif mode == 2:
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            order = np.argsort(-rm)
            k = 3 + int(rng.integers(0, 5))
            worst = order[:k]
            sig = np.full(n, 0.004)
            sig[worst] = 0.16 + 0.14*rng.random()
            nb = np.argsort(-G[worst[0]])[:7]
            for j in nb:
                sig[j] = max(sig[j], 0.05)
            prop = _norm(prop + sig[:, None]*rng.standard_normal(prop.shape))
            prop = _quench(prop, steps=550, b0=60.0, b1=750.0, lr0=0.016, lr1=0.0012, mu=0.78, rng=rng, noise=0.03)
        elif mode == 3:
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            rm = G.max(axis=1)
            order = np.argsort(-rm)
            nr = 1 + int(rng.integers(0, 2))
            for kk in range(nr):
                w = int(order[kk])
                mask = np.ones(n, dtype=bool); mask[w]=False
                h = _hole(prop[mask], rng, n_cand=4000)
                h = h + 0.008*rng.standard_normal(d)
                h /= np.linalg.norm(h)+1e-12
                prop[w]=h
            prop = _norm(prop + 0.006*rng.standard_normal(prop.shape))
            prop = _quench(prop, steps=500, b0=70.0, b1=700.0, lr0=0.015, lr1=0.001, mu=0.75, rng=rng, noise=0.015)
        elif mode == 4:
            fracs = 0.3+0.3*rng.random()
            k = int(n*fracs)
            sub = rng.choice(n, size=k, replace=False)
            sig = 0.07+0.13*rng.random()
            prop[sub] += sig*rng.standard_normal((k, d))
            prop = _norm(prop)
            prop = _quench(prop, steps=550, b0=50.0, b1=750.0, lr0=0.017, lr1=0.0013, mu=0.78, rng=rng, noise=0.035)
        else:
            # contact-break: pick a top pair and separate
            G = prop @ prop.T
            np.fill_diagonal(G, -2.0)
            # find max pair
            ij = np.argmax(G)
            ii, jj = divmod(int(ij), n)
            # push ii away from jj tangentially
            vij = prop[jj] - float(G[ii,jj])*prop[ii]
            vn = np.linalg.norm(vij)+1e-12
            vij /= vn
            prop[ii] -= (0.05+0.08*rng.random())*vij
            # push jj opposite
            vji = prop[ii] - float((prop[ii]@prop[jj]))*prop[jj]
            vjn = np.linalg.norm(vji)+1e-12
            if vjn>1e-12:
                prop[jj] -= (0.03+0.05*rng.random())*vji/vjn
            prop = _norm(prop + 0.02*rng.standard_normal(prop.shape))
            prop = _quench(prop, steps=600, b0=45.0, b1=850.0, lr0=0.018, lr1=0.0012, mu=0.8, rng=rng, noise=0.03)
        v = _max_cos(prop)
        improved_best = False
        if v < best_score - 1e-12:
            # gated short LP to squeeze
            if v < 0.075 and (LIM-(time.time()-t0)) > 10:
                dl = min(time.time()+1.2, t0+LIM-8.0)
                if dl > time.time()+0.3:
                    pq = _lp_polish(prop, deadline=dl, radius_init=7e-4)
                    vq = _max_cos(pq)
                    if vq < v:
                        prop, v = pq, vq
            if v < best_score:
                best_score = v
                x_best = prop.copy()
                improved_best = True
            cur, cur_score = prop.copy(), v
        else:
            # SA acceptance for cur
            if v < cur_score or rng.random() < np.exp((cur_score - v)/max(Temp,1e-9)):
                cur, cur_score = prop.copy(), v
            # occasional gated LP if very close to best (cheap topology fix)
            if v < best_score+0.0006 and v < 0.072 and (trial % 7 == 0):
                if (LIM-(time.time()-t0)) > 12:
                    dl = min(time.time()+0.9, t0+LIM-8.0)
                    if dl > time.time()+0.3:
                        pq = _lp_polish(prop, deadline=dl, radius_init=5e-4)
                        vq = _max_cos(pq)
                        if vq < best_score - 1e-12:
                            best_score = vq
                            x_best = pq.copy()
                            cur, cur_score = pq.copy(), vq
    # final polish
    remain = LIM-(time.time()-t0)
    if remain > 2.0:
        dl = time.time()+min(remain-0.7, 20.0)
        x_best = _lp_polish(x_best, deadline=dl, radius_init=9e-4)
    x_best = _norm(x_best)
    return x_best
