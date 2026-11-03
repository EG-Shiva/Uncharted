import time
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

def _compose(p1, s1, p2, s2):
    # f2 o f1 where f(x)_j = s_j * x_{p_j}
    # result p_j = p1[p2[j]], s_j = s2_j * s1_{p2_j}
    p = [p1[p2[j]] for j in range(len(p1))]
    s = [s2[j] * s1[p2[j]] for j in range(len(p1))]
    return tuple(p), tuple(s)

def _closure(gens, d=6):
    seen = set()
    elems = []
    ident = (tuple(range(d)), tuple([1]*d))
    stack = [ident]
    seen.add(ident)
    elems.append(ident)
    # BFS
    queue = [ident]
    while queue:
        cur = queue.pop()
        for g in gens:
            for a,b in [(cur,g),(g,cur)]:
                pass
            nxt_p, nxt_s = _compose(cur[0], cur[1], g[0], g[1])
            key = (nxt_p, nxt_s)
            if key not in seen:
                seen.add(key)
                elems.append(key)
                queue.append(key)
            nxt_p2, nxt_s2 = _compose(g[0], g[1], cur[0], cur[1])
            key2 = (nxt_p2, nxt_s2)
            if key2 not in seen:
                seen.add(key2)
                elems.append(key2)
                queue.append(key2)
    return elems

def _build_groups():
    d = 6
    # D6: cyclic shift + reversal
    perm_a = tuple([(j-1) % d for j in range(d)])  # roll by 1
    sign_a = tuple([1]*d)
    perm_b = tuple([0,5,4,3,2,1])
    sign_b = tuple([1]*d)
    D12 = _closure([(perm_a,sign_a),(perm_b,sign_b)], d=d)
    # sign flips of first three coords
    gens_sign = []
    for c in range(3):
        s = [1]*d
        s[c] = -1
        gens_sign.append((tuple(range(d)), tuple(s)))
    S8 = _closure(gens_sign, d=d)
    # 6-cycle alone
    C6 = _closure([(perm_a,sign_a)], d=d)
    def to_arrays(elems):
        perms = np.array([list(p) for p,s in elems], dtype=int)
        signs = np.array([list(s) for p,s in elems], dtype=float)
        return perms, signs
    p12, s12 = to_arrays(D12)
    p8, s8 = to_arrays(S8)
    p6, s6 = to_arrays(C6)
    return [(p12,s12),(p8,s8),(p6,s6)]

def solve():
    t0 = time.time()
    BUDGET = 168.0
    n, d = 219, 6
    iu0, iu1 = np.triu_indices(n, 1)
    rng_global = np.random.RandomState(0)

    # load incumbent
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
        x_best = _norm(rng_global.randn(n, d))
        best_val = _max_cos(x_best)

    def run_free_lbfgs(x0, beta, maxiter):
        y0 = x0.ravel().copy()
        def fun(y):
            Y = y.reshape(n, d)
            rn = np.linalg.norm(Y, axis=1, keepdims=True)
            rn = np.maximum(rn, 1e-12)
            X = Y / rn
            G = X @ X.T
            g = G[iu0, iu1]
            m = float(np.max(g))
            e = np.exp(beta * (g - m))
            S = float(np.sum(e)) + 1e-300
            loss = m + np.log(S) / beta
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

    def lp_polish(x, deadline, r_init=1.0e-3, max_steps=12):
        x = _norm(x.copy())
        best = _max_cos(x)
        bx = x.copy()
        radius = r_init
        steps = 0
        while time.time() < deadline and steps < max_steps:
            steps += 1
            if radius < 1e-14:
                break
            Gf = x @ x.T
            c = Gf[iu0, iu1]
            m = float(c.max())
            thr = m - max(18.0 * radius, 1e-9)
            sel = c > thr
            I = iu0[sel]; J = iu1[sel]; csel = c[sel]
            p = len(I)
            if p == 0:
                break
            if p > 3500:
                k2 = 3500
                part = np.argpartition(-csel, k2-1)[:k2]
                I, J, csel = I[part], J[part], csel[part]
                p = k2
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:, None]*d+np.arange(d), J[:, None]*d+np.arange(d), np.full((p,1), n*d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n*d+1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, n*d+1)).tocsr()
            obj = np.zeros(n*d+1); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
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
                radius = min(radius*1.5, 0.02)
            else:
                radius *= 0.5
        return bx, best

    groups = _build_groups()
    # configs: (group_idx, n_orbits)
    # sizes: 12->18 orbits, 8->27 orbits, 6->36 orbits; each +3 extras
    cfgs = [(0, 18, 3), (1, 27, 3), (2, 36, 3)]

    for cfg_i, (gi, k, e) in enumerate(cfgs):
        if time.time() - t0 > BUDGET - 45:
            break
        perms, signs = groups[gi]  # G x d
        Gsz = perms.shape[0]
        assert k*Gsz + e == n
        inv_perms = np.zeros_like(perms)
        for g in range(Gsz):
            inv = np.zeros(d, dtype=int)
            for j in range(d):
                inv[perms[g, j]] = j
            inv_perms[g] = inv
        Nfull = n
        fiu0, fiu1 = iu0, iu1
        rng = np.random.RandomState(1000 + gi*77)

        # init representatives random
        Y0 = rng.randn(k+e, d)

        def sym_fun_factory(beta):
            def fun(yflat):
                Y = yflat.reshape(k+e, d)
                rn = np.linalg.norm(Y, axis=1, keepdims=True)
                rn = np.maximum(rn, 1e-12)
                Zn = Y / rn  # (k+e) x d
                Zorb = Zn[:k]  # k x d
                Zext = Zn[k:]  # e x d
                # expand
                X = np.zeros((Nfull, d))
                # orbits
                for g in range(Gsz):
                    pg = perms[g]; sg = signs[g]
                    X[g*k:(g+1)*k if False else 0:0] = 0  # placeholder
                # build ordered as orbit-major: index = o*Gsz+g
                # to keep simple, order: for o in range(k), for g in range(Gsz)
                # vectorized per g using broadcasting
                # X_og[j] = sg[j]*Zorb[o, pg[j]]
                Xo = np.zeros((k, Gsz, d))
                for g in range(Gsz):
                    Xo[:, g, :] = Zorb[:, perms[g]] * signs[g]
                X[:k*Gsz] = Xo.transpose(1,0,2).reshape(k*Gsz, d)
                X[k*Gsz:] = Zext
                Gmat = X @ X.T
                gvec = Gmat[fiu0, fiu1]
                m = float(np.max(gvec))
                ee = np.exp(beta*(gvec - m))
                S = float(np.sum(ee)) + 1e-300
                loss = m + np.log(S)/beta
                w = ee / S
                W = np.zeros((Nfull, Nfull))
                W[fiu0, fiu1] = w
                W = W + W.T
                GX = W @ X  # grad wrt X
                # map back to Zn
                Gorb = GX[:k*Gsz].reshape(Gsz, k, d).transpose(1,0,2)  # k x Gsz x d
                grad_Zorb_n = np.zeros((k, d))
                for g in range(Gsz):
                    inv = inv_perms[g]
                    sg = signs[g]
                    Gg = Gorb[:, g, :]  # k x d indexed by j
                    contrib = (Gg * sg)[ :, inv]
                    grad_Zorb_n += contrib
                grad_Zext_n = GX[k*Gsz:]
                grad_Zn = np.zeros((k+e, d))
                grad_Zn[:k] = grad_Zorb_n
                grad_Zn[k:] = grad_Zext_n
                # to Y
                dot = (grad_Zn * Zn).sum(axis=1, keepdims=True)
                GY = (grad_Zn - dot * Zn) / rn
                return float(loss), GY.ravel()
            return fun

        Y = Y0.copy()
        # annealed betas
        betas = [70.0, 160.0, 350.0, 650.0] if Gsz >= 8 else [70.0, 160.0, 350.0, 650.0]
        maxiters = [60, 60, 50, 40]
        # shorten if low time
        for bi, beta in enumerate(betas):
            if time.time() - t0 > BUDGET - 40:
                break
            fun = sym_fun_factory(beta)
            try:
                res = minimize(fun, Y.ravel().copy(), jac=True, method="L-BFGS-B",
                               options={"maxiter": maxiters[bi], "ftol": 1e-12, "gtol": 1e-7, "maxfun": 10000})
                Y = res.x.reshape(k+e, d)
            except Exception:
                break
        # expand symmetric optimum to full
        Yn = _norm(Y)
        Zorb = Yn[:k]; Zext = Yn[k:]
        Xo = np.zeros((k, Gsz, d))
        for g in range(Gsz):
            Xo[:, g, :] = Zorb[:, perms[g]] * signs[g]
        Xsym = np.zeros((n, d))
        Xsym[:k*Gsz] = Xo.transpose(1,0,2).reshape(k*Gsz, d)
        Xsym[k*Gsz:] = Zext
        Xsym = _norm(Xsym)
        vsym = _max_cos(Xsym)
        # skip degenerate symmetric (duplicates)
        if vsym > 0.99:
            continue
        # release symmetry: free optimisation
        # two-stage free LBFGS
        try:
            Xr = run_free_lbfgs(Xsym, 200.0, maxiter=60)
            vr = _max_cos(Xr)
            # second sharpen
            if time.time() - t0 < BUDGET - 25:
                Xr2 = run_free_lbfgs(Xr, 500.0, maxiter=50)
                vr2 = _max_cos(Xr2)
                if vr2 < vr:
                    Xr, vr = Xr2, vr2
        except Exception:
            continue
        # LP polish if promising (within 0.01 of best to save time, else still try briefly)
        if vr < best_val + 0.015 and time.time() - t0 < BUDGET - 15:
            try:
                Xp, vp = lp_polish(Xr, deadline=time.time()+4.0)
                if vp < vr:
                    Xr, vr = Xp, vp
            except Exception:
                pass
        if vr < best_val:
            x_best = Xr.copy()
            best_val = vr

    # final squeeze on global best if time remains
    if time.time() - t0 < BUDGET - 8:
        try:
            Xf = run_free_lbfgs(x_best, 300.0, maxiter=40)
            vf = _max_cos(Xf)
            if vf < best_val:
                x_best, best_val = Xf, vf
            Xp, vp = lp_polish(x_best, deadline=t0+BUDGET-1.5, max_steps=20)
            if vp < best_val:
                x_best, best_val = Xp, vp
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
