import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    LIMIT = 160.0
    n, d = 93, 13
    rng = np.random.RandomState(0)

    def _norm(X):
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        return X / nr

    def _maxcos(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        return float(np.max(G))

    iu0, iu1 = np.triu_indices(n, 1)

    # load baseline
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _maxcos(a)
                    if v < best_val:
                        best_val = v
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.randn(n, d))
        best_val = _maxcos(best)

    def fingerprint(X, K=300):
        G = X @ X.T
        c = G[iu0, iu1]
        s = np.sort(c)[::-1][:K]
        return s.copy()

    def is_dup(fp, arch, tol=5e-5):
        for a in arch:
            if np.max(np.abs(fp - a)) < tol:
                return True
        return False

    archive = [fingerprint(best)]

    def build_clusters(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # choose threshold
        th = 1e-6
        slack = m - G
        np.fill_diagonal(slack, 1.0)
        cnt = int(np.sum(slack < th) // 2)
        if cnt < 20:
            th = 1e-5
            cnt = int(np.sum(slack < th) // 2)
        adj = slack < th
        visited = np.zeros(n, dtype=bool)
        comps = []
        for s in range(n):
            if visited[s]:
                continue
            stack = [s]
            visited[s] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                nbrs = np.where(adj[u])[0]
                for w in nbrs:
                    if not visited[w]:
                        visited[w] = True
                        stack.append(int(w))
            comps.append(np.array(comp, dtype=int))
        # if single giant component covering all, split by median projection
        if len(comps) == 1 and len(comps[0]) == n:
            u = rng.randn(d)
            u /= (np.linalg.norm(u) + 1e-300)
            proj = X @ u
            med = np.median(proj)
            i0 = np.where(proj <= med)[0]
            i1 = np.where(proj > med)[0]
            if len(i0) == 0 or len(i1) == 0:
                i0 = np.arange(n // 2)
                i1 = np.arange(n // 2, n)
            return [i0.astype(int), i1.astype(int)]
        # if one giant + isolates, keep as is (giant is a valid cluster)
        return comps

    clusters = build_clusters(best)

    def adam_softmax(X0, steps, tau0, tau1, lr0, lr1):
        X = _norm(X0.copy())
        m1 = np.zeros_like(X)
        m2 = np.zeros_like(X)
        b1, b2, eps = 0.9, 0.999, 1e-8
        tt = 0
        bv = _maxcos(X)
        bX = X.copy()
        for s in range(steps):
            f = s / max(1, steps - 1)
            tau = tau0 * ((tau1 / tau0) ** f)
            lr = lr0 * ((lr1 / lr0) ** f)
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            c = G[iu0, iu1]
            cm = float(np.max(c))
            E = np.exp((c - cm) / tau)
            S = float(np.sum(E))
            if not np.isfinite(S) or S <= 0:
                break
            w = E / S
            grad = np.zeros_like(X)
            np.add.at(grad, iu0, (w[:, None] * X[iu1]))
            np.add.at(grad, iu1, (w[:, None] * X[iu0]))
            dot = np.einsum('ij,ij->i', grad, X)[:, None]
            gt = grad - dot * X
            tt += 1
            m1 = b1 * m1 + (1 - b1) * gt
            m2 = b2 * m2 + (1 - b2) * (gt * gt)
            mh = m1 / (1 - b1 ** tt)
            vh = m2 / (1 - b2 ** tt)
            step = lr * mh / (np.sqrt(vh) + eps)
            step -= np.einsum('ij,ij->i', step, X)[:, None] * X
            X = _norm(X - step)
            if (s % 50 == 0) or (s == steps - 1):
                cur = _maxcos(X)
                if cur < bv:
                    bv = cur
                    bX = X.copy()
        return bX, bv

    def lp_attempt(x, radius, cap=2500):
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        width = max(20.0 * radius, 5e-9)
        thresh = m - width
        sel = c_all > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            return x, m, False
        if cnt > cap:
            idx = np.argpartition(-c_all, cap)[:cap]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = mask
            cnt = cap
        I = iu0[sel]
        J = iu1[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = np.arange(d)
        cols = np.concatenate([I[:, None] * d + cI, J[:, None] * d + cI, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            return x, m, False
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_all[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            return x, m, False
        if res.status != 0:
            return x, m, False
        dx = res.x[:-1].reshape(n, d)
        y = _norm(x + dx)
        v = _maxcos(y)
        return y, v, True

    def full_lp_polish(X0, deadline):
        x = _norm(X0.copy())
        cur = _maxcos(x)
        bx = x.copy()
        bv = cur
        radius = 2e-3
        it = 0
        while time.time() < deadline - 0.8 and it < 14:
            it += 1
            y, v, ok = lp_attempt(x, radius)
            if ok and v < cur - 1e-15:
                x = y
                cur = v
                if v < bv:
                    bv = v
                    bx = y.copy()
                radius = min(radius * 1.4, 3e-3)
            else:
                radius *= 0.5
                if radius < 1e-9:
                    break
                if radius < 5e-8 and time.time() > deadline - 2.0:
                    break
        return bx, bv

    trial = 0
    while time.time() - t0 < LIMIT:
        remain = (t0 + LIMIT) - time.time()
        if remain < 6.0:
            break
        trial += 1
        if trial % 12 == 0:
            try:
                clusters = build_clusters(best)
            except Exception:
                pass
        if len(clusters) == 0:
            clusters = build_clusters(best)
        ci = rng.randint(len(clusters))
        cl = clusters[ci]
        if len(cl) >= n:
            # split it
            u = rng.randn(d)
            u /= (np.linalg.norm(u) + 1e-300)
            proj = best @ u
            med = np.median(proj)
            i0 = np.where(proj <= med)[0]
            if len(i0) > 0 and len(i0) < n:
                cl = i0 if rng.rand() < 0.5 else np.where(proj > med)[0]
            else:
                cl = np.arange(n // 2)
        Y = best.copy()
        r = rng.rand()
        if r < 0.6:
            a, b = rng.choice(d, 2, replace=False)
            theta = rng.uniform(0.03, 0.20) * (1 if rng.rand() < 0.5 else -1)
            c, s = np.cos(theta), np.sin(theta)
            qa = Y[cl][:, a].copy()
            qb = Y[cl][:, b].copy()
            Y[cl, a] = c * qa - s * qb
            Y[cl, b] = s * qa + c * qb
        else:
            k = rng.randint(d)
            Y[cl, k] *= -1.0
        Y = _norm(Y)
        # short annealed softmax Adam polish
        try:
            Z, vz = adam_softmax(Y, steps=300, tau0=3e-3, tau1=3e-4, lr0=0.01, lr1=0.001)
        except Exception:
            continue
        if not np.all(np.isfinite(Z)):
            continue
        if vz > 0.245:
            continue
        fpz = fingerprint(Z)
        if is_dup(fpz, archive):
            continue
        archive.append(fpz.copy())
        if len(archive) > 400:
            archive = archive[-400:]
        # promote to full polish
        deadline = min(time.time() + 9.0, t0 + LIMIT - 1.0)
        try:
            # a bit more Adam at smaller tau before LP
            Z2, vz2 = adam_softmax(Z, steps=200, tau0=8e-4, tau1=1e-4, lr0=0.005, lr1=0.0008)
            if vz2 < vz:
                Z, vz = Z2, vz2
        except Exception:
            pass
        try:
            W, vw = full_lp_polish(Z, deadline)
        except Exception:
            continue
        # also consider Z directly
        cand, cv = (W, vw) if vw < vz else (Z, vz)
        fpw = fingerprint(cand)
        if not is_dup(fpw, archive, tol=2e-5):
            archive.append(fpw.copy())
        if cv < best_val - 1e-14:
            best = _norm(cand.copy())
            best_val = float(cv)
            try:
                clusters = build_clusters(best)
            except Exception:
                pass
    best = _norm(best)
    return best
