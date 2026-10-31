import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(n, 1e-12, out=n)
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _per_point(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    pm = G.max(axis=1)
    m = pm.max()
    deg = (G >= m - 1e-7).sum(axis=1)
    return m, pm, deg

def _anneal_row(x, steps, beta0, beta1, lr0, mu):
    n, d = x.shape
    v = np.zeros_like(x)
    if beta1 > beta0:
        lb = np.log(beta1 / beta0)
    else:
        lb = 0.0
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True) + 1e-300
        A = (W @ x) / S
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        v = mu * v + Gt
        vd = (v * x).sum(axis=1, keepdims=True)
        v = v - vd * x
        x = x - lr * v
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(xn, 1e-12, out=xn)
        x = x / xn
    return x

def _cheap_descent(x, steps=100, lr0=0.01, tol0=0.005):
    n, d = x.shape
    lr = lr0
    for t in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        band = tol0 * (1.0 - 0.8 * t / max(1, steps - 1))
        Mask = G > m - band
        deg = Mask.sum(axis=1)
        has = deg > 0
        if not np.any(has):
            break
        W = Mask.astype(float)
        D = np.maximum(deg, 1)[:, None]
        A = (W @ x) / D
        dot = (A * x).sum(axis=1, keepdims=True)
        Gt = A - dot * x
        Gt[~has] = 0.0
        x = x - lr * Gt
        xn = np.linalg.norm(x, axis=1, keepdims=True)
        np.maximum(xn, 1e-12, out=xn)
        x = x / xn
        lr *= 0.998
    return x

def _lp_polish(x, seconds, hard_deadline=1e100):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < hard_deadline:
        if radius < 1e-14:
            break
        C = (x @ x.T)[iu]
        m = C.max()
        sel = C > m - max(20 * radius, 1e-9)
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
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-C[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            x, best, xb = y, v, y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb

def _find_holes(Xkeep, k, rng, n_cand=20000, sep=0.90):
    cand = rng.normal(size=(n_cand, 4))
    cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
    best = np.empty(n_cand)
    cs = 5000
    for a in range(0, n_cand, cs):
        b = min(a + cs, n_cand)
        D = cand[a:b] @ Xkeep.T
        best[a:b] = D.max(axis=1)
    idx = np.argsort(best)
    sel = []
    for i in idx:
        if len(sel) >= k:
            break
        c = cand[i]
        ok = True
        for s in sel:
            if float(c @ s) > sep:
                ok = False
                break
        if ok:
            sel.append(c)
    if len(sel) < k:
        for i in idx:
            if len(sel) >= k:
                break
            c = cand[i]
            dup = False
            for s in sel:
                if float(c @ s) > 0.999:
                    dup = True
                    break
            if not dup:
                sel.append(c)
    return np.array(sel)

def _rand_orth(rng):
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    explore_end = t0 + 143.0
    rng = np.random.default_rng(0)
    n, d = 292, 4
    xb = None
    bv = 1e100
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                v = _max_cos(a)
                if v < bv:
                    bv = v
                    xb = a.copy()
        except Exception:
            continue
    if xb is None:
        xb = _norm(rng.normal(size=(n, d)))
        bv = _max_cos(xb)
    x_best = xb.copy()
    best = float(bv)
    elites = [x_best.copy()]
    evals = [best]
    try:
        _, permax_best, deg_best = _per_point(x_best)
    except Exception:
        permax_best = np.zeros(n); deg_best = np.zeros(n, dtype=int)
    order_worst = np.argsort(-permax_best)
    order_hi = np.argsort(-deg_best)

    it = 0
    while time.time() < explore_end:
        it += 1
        if deadline - time.time() < 10:
            break
        # pick base: mostly best, sometimes other elite
        if len(elites) > 1 and rng.random() < 0.30:
            bi = int(rng.integers(0, len(elites)))
            base = elites[bi]
        else:
            base = x_best
        X = base.copy()
        mv = rng.random()
        k = 0
        big = False
        if mv < 0.38:
            # kill and fill holes, variable size incl. large
            u = rng.random()
            if u < 0.25:
                k = int(rng.integers(2, 6))
            elif u < 0.55:
                k = int(rng.integers(6, 14))
            elif u < 0.85:
                k = int(rng.integers(14, 30))
            else:
                k = int(rng.integers(30, 55))
            k = min(k, 50)
            sig = float(0.004 + 0.02 * rng.random())
            X = _norm(X + sig * rng.normal(size=X.shape))
            victims = set()
            victims.add(int(order_worst[0]))
            victims.add(int(order_hi[0]))
            # add overloaded neighbours of worst
            try:
                Gw = X @ X.T
                Gw[range(n), range(n)] = -2.0
                w = int(order_worst[0])
                nn = np.argsort(-Gw[w])[:12]
                for ii in nn[:4]:
                    if len(victims) < k and rng.random() < 0.7:
                        victims.add(int(ii))
            except Exception:
                pass
            while len(victims) < k:
                u2 = rng.random()
                if u2 < 0.35:
                    victims.add(int(order_hi[int(rng.integers(0, min(n, 40)))]))
                elif u2 < 0.6:
                    victims.add(int(order_worst[int(rng.integers(0, min(n, 40)))]))
                else:
                    victims.add(int(rng.integers(0, n)))
            victims = np.array(sorted(list(victims))[:k], dtype=int)
            keep = np.ones(n, dtype=bool); keep[victims] = False
            try:
                holes = _find_holes(X[keep], len(victims), rng, n_cand=20000, sep=0.88)
                if holes.shape[0] == len(victims):
                    X[victims] = holes + 0.004 * rng.normal(size=(len(victims), 4))
                    X = _norm(X)
                else:
                    X[victims] = _norm(rng.normal(size=(len(victims), 4)))
            except Exception:
                X[victims] = _norm(rng.normal(size=(len(victims), 4)))
            big = (k >= 14)
        elif mv < 0.60:
            # global shake, log-uniform sigma
            ls = rng.uniform(np.log(0.008), np.log(0.45))
            sig = float(np.exp(ls))
            X = _norm(X + sig * rng.normal(size=X.shape))
            big = (sig > 0.12)
        elif mv < 0.75:
            # jammed cluster excision: neighbours of random overloaded centre
            try:
                G0 = X @ X.T
                G0[range(n), range(n)] = -2.0
                c = int(order_hi[int(rng.integers(0, 12))]) if rng.random() < 0.7 else int(rng.integers(0, n))
                nn = np.argsort(-G0[c])[:int(rng.integers(12, 32))]
                X[c] += 0.25 * rng.normal(size=(4,))
                X[nn] += 0.18 * rng.normal(size=(len(nn), 4))
                ri = rng.choice(n, size=6, replace=False)
                X[ri] += 0.02 * rng.normal(size=(6, 4))
                X = _norm(X)
            except Exception:
                X = _norm(X + 0.1 * rng.normal(size=X.shape))
            big = True
        elif mv < 0.87:
            # subset rigid rotation: preserves local packing, changes cross contacts
            msz = int(rng.integers(25, 90))
            idx = rng.choice(n, size=msz, replace=False)
            Q = _rand_orth(rng)
            # interpolate: small or full rotation
            if rng.random() < 0.5:
                X[idx] = X[idx] @ Q.T
            else:
                # small-angle: mix with identity
                a = float(rng.uniform(0.3, 0.9))
                R = a * Q + (1 - a) * np.eye(4)
                R, _ = np.linalg.qr(R)
                if np.linalg.det(R) < 0:
                    R[:, 0] = -R[:, 0]
                X[idx] = X[idx] @ R.T
            X[idx] += 0.01 * rng.normal(size=(msz, 4))
            X = _norm(X)
            big = True
        else:
            # crossover of two elites
            if len(elites) >= 2:
                i1, i2 = rng.choice(len(elites), size=2, replace=False)
                A, B = elites[i1], elites[i2]
            else:
                A = x_best; B = _norm(rng.normal(size=(n, d)))
            pa = rng.permutation(n); pb = rng.permutation(n)
            h = n // 2
            X = np.vstack([A[pa[:h]], B[pb[:h]]])
            # if odd, add one more
            if X.shape[0] < n:
                X = np.vstack([X, A[pa[h:h+1]]])
            X = _norm(X + 0.005 * rng.normal(size=X.shape))
            big = True
        # choose anneal params
        if big:
            steps, b0, b1, lr0, mu = 520, 12.0, 550.0, 0.005, 0.88
            if rng.random() < 0.3:
                steps = 700
        else:
            steps, b0, b1, lr0, mu = 340, 22.0, 420.0, 0.004, 0.86
            # randomize a bit
            b1 = float(rng.uniform(300, 600))
        try:
            Xopt = _anneal_row(X, steps, b0, b1, lr0, mu)
            Xopt = _cheap_descent(Xopt, steps=90, lr0=0.009, tol0=0.005)
        except Exception:
            continue
        try:
            v = _max_cos(Xopt)
        except Exception:
            continue
        if v < best - 1.5e-6:
            rem = min(6.0, max(1.0, deadline - time.time() - 26.0))
            try:
                Xpol = _lp_polish(Xopt, seconds=rem, hard_deadline=deadline - 22.0)
                vp = _max_cos(Xpol)
            except Exception:
                Xpol, vp = Xopt, v
            if vp < best:
                x_best, best = Xpol, float(vp)
                try:
                    _, permax_best, deg_best = _per_point(x_best)
                    order_worst = np.argsort(-permax_best)
                    order_hi = np.argsort(-deg_best)
                except Exception:
                    pass
                elites.append(x_best.copy()); evals.append(best)
                # keep top 6
                if len(elites) > 6:
                    o = np.argsort(np.array(evals))
                    elites = [elites[i] for i in o[:6]]
                    evals = [evals[i] for i in o[:6]]
        elif v < best + 0.0025:
            # maintain diversity pool
            if len(elites) < 6:
                # avoid duplicate insertion
                if v < min(evals) + 0.0025:
                    elites.append(Xopt.copy()); evals.append(float(v))
            else:
                # replace worst if better
                wi = int(np.argmax(np.array(evals)))
                if v < evals[wi] - 2e-5:
                    elites[wi] = Xopt.copy(); evals[wi] = float(v)
            # occasional LP on near-miss late
            if (it % 25 == 0) and time.time() < explore_end - 10 and (deadline - time.time()) > 30:
                if v < best + 2e-5:
                    try:
                        Xpol = _lp_polish(Xopt, seconds=1.5, hard_deadline=deadline - 22.0)
                        vp = _max_cos(Xpol)
                        if vp < best:
                            x_best, best = Xpol, float(vp)
                            try:
                                _, permax_best, deg_best = _per_point(x_best)
                                order_worst = np.argsort(-permax_best)
                                order_hi = np.argsort(-deg_best)
                            except Exception:
                                pass
                            elites.append(x_best.copy()); evals.append(best)
                            if len(elites) > 6:
                                o = np.argsort(np.array(evals))
                                elites = [elites[i] for i in o[:6]]
                                evals = [evals[i] for i in o[:6]]
                    except Exception:
                        pass
    # final squeeze
    try:
        rem = max(1.0, deadline - time.time() - 2.0)
        if rem > 1.0:
            x_best = _anneal_row(x_best, 200, 30.0, 600.0, 0.0025, 0.86)
            x_best = _cheap_descent(x_best, steps=120, lr0=0.006, tol0=0.004)
            x_best = _lp_polish(x_best, seconds=min(rem, 22.0), hard_deadline=deadline - 1.0)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
