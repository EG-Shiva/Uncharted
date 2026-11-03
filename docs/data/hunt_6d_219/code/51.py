import time
import itertools
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def solve():
    t0 = time.time()
    BUDGET = 168.0
    SYM_END = t0 + 62.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    rng = np.random.default_rng(0)

    x_best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            v = np.load(fn)
            v = np.asarray(v, dtype=float)
            if v.shape == (n, d):
                v = _norm(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    x_best = v.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(x_best)

    def consider(X):
        nonlocal x_best, best_val
        X = _norm(np.asarray(X, dtype=float))
        if X.shape != (n, d) or not np.all(np.isfinite(X)):
            return 1e9
        c = _max_cos(X)
        if c < best_val:
            best_val = c
            x_best = X.copy()
        return c

    # ---------- signed perm helpers ----------
    def compose(p1, s1, p2, s2):
        # y = g2(g1(x)): p[j]=p1[p2[j]], s[j]=s2[j]*s1[p2[j]]
        p = tuple(p1[p2[j]] for j in range(len(p1)))
        s = tuple(s2[j] * s1[p2[j]] for j in range(len(p1)))
        return p, s

    def inv_elem(p, s):
        # p,s tuples
        dd = len(p)
        q = [0]*dd
        for j in range(dd):
            q[p[j]] = j
        pinv = tuple(q)
        sinv = tuple(s[q[i]] for i in range(dd))
        return pinv, sinv

    def closure(gens):
        ident = (tuple(range(d)), tuple([1]*d))
        # add inverses
        aug = []
        for (p, s) in gens:
            aug.append((p, s))
            aug.append(inv_elem(p, s))
        seen = {ident}
        elems = [ident]
        queue = [ident]
        while queue:
            cur = queue.pop()
            cp, cs = cur
            for (gp, gs) in aug:
                np_, ns_ = compose(cp, cs, gp, gs)
                key = (np_, ns_)
                if key not in seen:
                    seen.add(key)
                    elems.append(key)
                    queue.append(key)
                    if len(elems) > 600:
                        return elems
        return elems

    def random_block_group():
        # 4+2 split, perms on A only, signed perms on B
        perm_all = rng.permutation(d)
        A = perm_all[:4].tolist()
        B = perm_all[4:].tolist()
        def rand_gen():
            p = list(range(d))
            shA = rng.permutation(A)
            for j, v in zip(A, shA):
                p[j] = int(v)
            # B perm
            if rng.random() < 0.5:
                p[B[0]], p[B[1]] = p[B[1]], p[B[0]]
            s = [1]*d
            s[B[0]] = 1 if rng.random() < 0.5 else -1
            s[B[1]] = 1 if rng.random() < 0.5 else -1
            # occasionally add even flip on A pair to get sign blocks
            if rng.random() < 0.3:
                i1, i2 = rng.choice(A, size=2, replace=False)
                # flip pair: need to keep closure small; encode as signs on those positions
                # flip signs on two A positions (even)
                s[i1] = -s[i1]
                s[i2] = -s[i2]
            return (tuple(p), tuple(s))
        ng = int(rng.integers(2, 4))
        gens = [rand_gen() for _ in range(ng)]
        elems = closure(gens)
        G = len(elems)
        return elems, G, A, B

    def elems_to_arrays(elems):
        G = len(elems)
        perms = np.zeros((G, d), dtype=int)
        signs = np.zeros((G, d), dtype=float)
        for i, (p, s) in enumerate(elems):
            perms[i] = np.array(p, dtype=int)
            signs[i] = np.array(s, dtype=float)
        return perms, signs

    # ---------- symmetric+free joint optimisation ----------
    def sym_optimize(perms, signs, m, Nf, beta, maxiter, Y0=None):
        G = perms.shape[0]
        Ns = m * G
        Ntot = Ns + Nf
        assert Ntot == n
        # build M (G,d,d)
        M = np.zeros((G, d, d), dtype=float)
        for g in range(G):
            for j in range(d):
                M[g, j, perms[g, j]] = signs[g, j]
        iu0l, iu1l = np.triu_indices(Ntot, 1)
        if Y0 is None:
            Y0 = rng.standard_normal(((m + Nf) * d))
        else:
            Y0 = np.asarray(Y0, dtype=float).ravel().copy()
        def fun(y):
            Y = y.reshape(m + Nf, d)
            Yr = Y[:m]
            Yf = Y[m:]
            nr = np.maximum(np.linalg.norm(Yr, axis=1, keepdims=True), 1e-12)
            nf = np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
            R = Yr / nr
            F = Yf / nf if Nf > 0 else np.zeros((0, d))
            # expand: S[kg] = M[g] @ R[k]
            # S shape (m,G,d): S[k,g,j]=sum_i M[g,j,i]R[k,i]
            S = np.einsum('gji,ki->kgj', M, R)
            S2 = S.reshape(Ns, d)
            X = S2 if Nf == 0 else np.vstack([S2, F])
            Gr = X @ X.T
            g = Gr[iu0l, iu1l]
            mx = float(np.max(g))
            e = np.exp(beta * (g - mx))
            Ssum = float(np.sum(e)) + 1e-300
            loss = mx + np.log(Ssum) / beta
            w = e / Ssum
            W = np.zeros((Ntot, Ntot), dtype=float)
            W[iu0l, iu1l] = w
            W = W + W.T
            GX = W @ X
            dot = np.sum(GX * X, axis=1, keepdims=True)
            # split
            GXs = GX[:Ns].reshape(m, G, d)
            # grad wrt R: GR[k,i]=sum_gj M[g,j,i]*GXs[k,g,j]
            GR = np.einsum('gji,kgj->ki', M, GXs)
            dotR = np.sum(GR * R, axis=1, keepdims=True)
            GYr = (GR - dotR * R) / nr
            if Nf > 0:
                GXf = GX[Ns:]
                dotF = np.sum(GXf * F, axis=1, keepdims=True)
                GYf = (GXf - dotF * F) / nf
                GY = np.vstack([GYr, GYf])
            else:
                GY = GYr
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, Y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            y = res.x
        except Exception:
            y = Y0
        # rebuild X
        Y = y.reshape(m + Nf, d)
        Yr = Y[:m]; Yf = Y[m:]
        nr = np.maximum(np.linalg.norm(Yr, axis=1, keepdims=True), 1e-12)
        R = Yr / nr
        S = np.einsum('gji,ki->kgj', M, R)
        S2 = _norm(S.reshape(Ns, d))
        if Nf > 0:
            F = _norm(Yf)
            X = np.vstack([S2, F])
        else:
            X = S2
        return _norm(X)

    def run_free_lbfgs(x0, beta, maxiter):
        y0 = x0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            mx = float(np.max(g))
            e = np.exp(beta * (g - mx))
            S = float(np.sum(e)) + 1e-300
            loss = mx + np.log(S) / beta
            w = e / S
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            dot = (GX * X).sum(axis=1, keepdims=True)
            GY = (GX - dot * X) / rn
            return float(loss), GY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7, "maxfun": 20000})
            Y = res.x.reshape(n, d)
            return _norm(Y)
        except Exception:
            return _norm(x0)

    def lp_polish(x, deadline, r_init=1.1e-3):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        while time.time() < deadline and time.time() - t0 < BUDGET - 1.0:
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(np.max(c))
            thr = m - max(18.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3600:
                k2 = 3600
                part = np.argpartition(-csel, k2 - 1)[:k2]
                I, J, csel = I[part], J[part], csel[part]
                p = k2
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
            obj = np.zeros(n * d + 1); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5
                continue
            if res.status != 0:
                radius *= 0.5
                continue
            y = _norm(x + res.x[:-1].reshape(n, d))
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                bx = y.copy()
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return bx, best

    # ---------- Phase 1: symmetric search ----------
    sym_cands = []  # list of (val, X)
    trial = 0
    while time.time() < SYM_END:
        trial += 1
        try:
            elems, G, A, B = random_block_group()
            if G < 16 or G > 192:
                continue
            m = n // G
            if m == 0:
                continue
            Ns = m * G
            Nf = n - Ns
            perms, signs = elems_to_arrays(elems)
            # random init for reps+free
            Y0 = rng.standard_normal(((m + Nf) * d))
            beta = float(rng.choice([120.0, 180.0, 250.0]))
            Xsym = sym_optimize(perms, signs, m, Nf, beta=beta, maxiter=55, Y0=Y0)
            c = _max_cos(Xsym)
            # short free sharpen to score basin potential (cheap)
            # only for promising ones
            if c < best_val + 0.02:
                Xf = run_free_lbfgs(Xsym, beta=350.0, maxiter=25)
                cf = _max_cos(Xf)
                consider(Xf)
                sym_cands.append((cf, Xf.copy()))
                # also keep raw sym
                sym_cands.append((c, Xsym.copy()))
                sym_cands.sort(key=lambda t: t[0])
                sym_cands = sym_cands[:8]
            else:
                if len(sym_cands) < 8:
                    sym_cands.append((c, Xsym.copy()))
        except Exception:
            continue
        if time.time() - t0 > BUDGET - 90:
            break

    # ---------- Phase 2: break symmetry gradually + release ----------
    # candidate pool: top sym + incumbent
    pool = []
    sym_cands.sort(key=lambda t: t[0])
    for _, X in sym_cands[:4]:
        pool.append(X.copy())
    pool.append(x_best.copy())
    # deduplicate-ish: add noisy variants
    end_release = t0 + BUDGET - 12.0
    # release each with increasing noise
    noises = [0.0, 1e-4, 1e-3, 1e-2]
    # iterate
    idx = 0
    while time.time() < end_release and idx < len(pool) * 2 + 6:
        try:
            base = pool[idx % len(pool)] if len(pool) > 0 else x_best
            idx += 1
            amp = noises[idx % len(noises)] if idx % 3 != 0 else float(rng.choice([1e-4, 1e-3, 1e-2]))
            X = base.copy()
            if amp > 0:
                Nz = rng.standard_normal(X.shape)
                Nz -= np.sum(Nz * X, axis=1, keepdims=True) * X
                nn = np.linalg.norm(Nz, axis=1, keepdims=True) + 1e-12
                X = _norm(X + amp * (Nz / nn))
            b2 = float(rng.choice([350.0, 550.0, 800.0]))
            X = run_free_lbfgs(X, beta=b2, maxiter=40)
            consider(X)
            if time.time() > end_release - 4:
                break
            # occasionally strong kick on worst points to change topology
            if idx % 4 == 0 and time.time() < end_release - 8:
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                pm = np.max(G, axis=1)
                worst = np.argsort(-pm)[:6]
                for vi in worst:
                    X[vi] = _norm(rng.standard_normal((1, d)))[0])
                X = run_free_lbfgs(X, beta=200.0, maxiter=30)
                consider(X)
        except Exception:
            continue

    # ---------- final squeeze ----------
    try:
        remain = BUDGET - (time.time() - t0) - 1.0
        if remain > 4:
            xb, vb = lp_polish(x_best, deadline=time.time() + min(remain * 0.55, 9.0))
            if vb < best_val:
                best_val = vb
                x_best = xb
            if time.time() - t0 < BUDGET - 4:
                Xf = run_free_lbfgs(x_best, beta=1100.0, maxiter=45)
                consider(Xf)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
