import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def build_expand(Rn, Gorder, thetas, n_planes, D, fixed_idx):
    R = Rn.shape[0]
    N = R * Gorder
    X = np.empty((N, D))
    cs = np.cos(np.outer(np.arange(Gorder), thetas))
    sn = np.sin(np.outer(np.arange(Gorder), thetas))
    for g in range(Gorder):
        for a in range(R):
            idx = a * Gorder + g
            r = Rn[a]
            x = X[idx]
            for j in range(n_planes):
                c = cs[g, j]; s = sn[g, j]
                u0 = r[2*j]; u1 = r[2*j+1]
                x[2*j] = c*u0 - s*u1
                x[2*j+1] = s*u0 + c*u1
            for fi in fixed_idx:
                x[fi] = r[fi]
    return X

def make_lse_fun(R, Gorder, ks, D):
    n_planes = len(ks)
    thetas = 2*np.pi*np.array(ks, dtype=float)/Gorder
    pstart = 2*n_planes
    fixed_idx = list(range(pstart, D))
    cs_all = np.cos(np.outer(np.arange(Gorder), thetas))
    sn_all = np.sin(np.outer(np.arange(Gorder), thetas))
    def val_grad(Yflat, alpha):
        Y = Yflat.reshape(R, D)
        norms = np.linalg.norm(Y, axis=1)
        norms = np.maximum(norms, 1e-12)
        Rn = Y / norms[:, None]
        N = R*Gorder
        X = np.empty((N, D))
        for g in range(Gorder):
            c = cs_all[g]; s = sn_all[g]
            for a in range(R):
                idx = a*Gorder + g
                r = Rn[a]
                row = X[idx]
                for j in range(n_planes):
                    cj = c[j]; sj = s[j]
                    u0 = r[2*j]; u1 = r[2*j+1]
                    row[2*j] = cj*u0 - sj*u1
                    row[2*j+1] = sj*u0 + cj*u1
                for fi in fixed_idx:
                    row[fi] = r[fi]
        Gg = X @ X.T
        np.fill_diagonal(Gg, -10.0)
        m = float(np.max(Gg))
        W = np.exp(alpha*(Gg - m))
        np.fill_diagonal(W, 0.0)
        S = float(np.sum(W) * 0.5)
        if S <= 0:
            S = 1.0
        LSE = m + np.log(S)/alpha
        GradX = (W @ X) / S
        Gr = np.zeros_like(Rn)
        for g in range(Gorder):
            c = cs_all[g]; s = sn_all[g]
            for a in range(R):
                idx = a*Gorder + g
                gx = GradX[idx]
                acc0 = np.empty(n_planes); acc1 = np.empty(n_planes)
                for j in range(n_planes):
                    cj = c[j]; sj = s[j]
                    g0 = gx[2*j]; g1 = gx[2*j+1]
                    Gr[a, 2*j] += cj*g0 + sj*g1
                    Gr[a, 2*j+1] += -sj*g0 + cj*g1
                for fi in fixed_idx:
                    Gr[a, fi] += gx[fi]
        Gy = np.empty_like(Y)
        for a in range(R):
            r = Rn[a]
            g = Gr[a]
            dot = float(np.dot(g, r))
            Gy[a] = (g - dot*r)/norms[a]
        return LSE, Gy.ravel()
    return val_grad, thetas, fixed_idx

def sym_run(R, Gorder, ks, D, alphas, Y0, maxiter_list):
    fun, _, _ = make_lse_fun(R, Gorder, ks, D)
    Yflat = Y0.ravel().copy()
    for alpha, mi in zip(alphas, maxiter_list):
        def f(v, al=alpha):
            L, G = fun(v, al)
            return L, G
        try:
            res = minimize(f, Yflat, jac=True, method='L-BFGS-B',
                           options={'maxiter': mi, 'ftol': 1e-12, 'gtol': 1e-7, 'maxfun': 100000})
            Yflat = res.x.copy()
        except Exception:
            break
    Y = Yflat.reshape(R, D)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def sym_to_full(Y, Gorder, ks, D):
    n_planes = len(ks)
    thetas = 2*np.pi*np.array(ks, dtype=float)/Gorder
    return build_expand(Y, Gorder, thetas, n_planes, D, list(range(2*n_planes, D)))

def smooth_refine(v, alpha0, alpha1, steps, lr, beta=0.9):
    n, d = v.shape
    cur = v.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    mom = np.zeros_like(cur)
    for s in range(steps):
        frac = s/max(steps-1, 1) if steps > 1 else 1.0
        alpha = alpha0*((alpha1/max(alpha0, 1e-12))**frac)
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        W = np.exp(alpha*(G-m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad*cur, axis=1, keepdims=True)
        Gt = Grad - proj*cur
        rs = W.sum(axis=1, keepdims=True)+1e-12
        Gt = Gt/rs
        mom = beta*mom + (1.0-beta)*Gt
        bc = 1.0-beta**(s+1)
        step = mom/bc
        cur_lr = lr*(1.0-0.5*frac)
        cur = cur - cur_lr*step
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        if (s % 200 == 199) or (s == steps-1):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
    return best, best_s

def lp_polish_general(x, seconds, init_radius, min_delta, factor=20.0):
    x = x/np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time()-start < seconds and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(np.max(cmat))
        delta = max(factor*radius, min_delta)
        sel = cmat > m - delta
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 4000:
            order = np.argsort(-cmat[sel])
            take = np.sort(order[:4000])
            I = I[take]; J = J[take]
            p = len(I)
            csub = cmat[sel][take]
        else:
            csub = cmat[sel]
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:, None]*d+ar, J[:, None]*d+ar, np.full((p, 1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
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
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return best_x, best

def contact_degrees(x, tol=1e-9):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    C = (G > m - tol).sum(axis=1)
    return C, m

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.default_rng(12345)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        tmp = rng.standard_normal(size=(99, 13))
        xb = tmp
    xb = xb/np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    best_x = xb.copy()
    D = 13
    cands = []
    # C11: 9 reps, 5 planes k=1..5 +3 fixed
    # C9: 11 reps, 4 planes k=1..4 +5 fixed
    configs = [
        (9, 11, [1, 2, 3, 4, 5]),
        (11, 9, [1, 2, 3, 4]),
    ]
    sym_time_limit = t0 + 55.0
    for (R, Gorder, ks) in configs:
        n_rest = 4
        for ri in range(n_rest):
            if time.time() > sym_time_limit:
                break
            Y0 = rng.standard_normal(size=(R, D))
            Y0 /= np.linalg.norm(Y0, axis=1, keepdims=True)
            Y0 = Y0 * 1.0
            # add small noise to break exact symmetry of init
            try:
                Yopt = sym_run(R, Gorder, ks, D, [100.0, 400.0], Y0, [200, 250])
            except Exception:
                continue
            try:
                Xfull = sym_to_full(Yopt, Gorder, ks, D)
                Xfull /= np.linalg.norm(Xfull, axis=1, keepdims=True)
                v = max_cos(Xfull)
                cands.append((v, Xfull.copy()))
                if v < best_val:
                    best_val = v
                    best_x = Xfull.copy()
            except Exception:
                continue
            if time.time() > sym_time_limit:
                break
    # release symmetry: annealed LogSumExp
    cands_sorted = sorted(cands, key=lambda t: t[0])
    # keep top 3 symmetric + baseline
    release_set = []
    for i in range(min(3, len(cands_sorted))):
        release_set.append(cands_sorted[i][1])
    release_set.append(xb)
    released = []
    for X0 in release_set:
        if time.time() - t0 > 120.0:
            break
        cur = X0.copy()
        # annealed stages
        try:
            c1, s1 = smooth_refine(cur, 80.0, 200.0, 700, 0.08)
            c2, s2 = smooth_refine(c1, 200.0, 400.0, 700, 0.05)
            c3, s3 = smooth_refine(c2, 400.0, 600.0, 600, 0.03)
            cc, ss = c3, s3
            # pick best of stages
            if s1 < ss:
                cc, ss = c1, s1
            if s2 < ss:
                cc, ss = c2, s2
            released.append((ss, cc))
            if ss < best_val:
                best_val = ss
                best_x = cc.copy()
        except Exception:
            continue
    # choose best released for wide+narrow LP
    if len(released) > 0:
        released_sorted = sorted(released, key=lambda t: t[0])
        target = released_sorted[0][1]
    else:
        target = best_x
    rem = BUDGET - (time.time() - t0)
    if rem > 8.0:
        # wide-active LP first
        try:
            wide_time = min(rem*0.5, 25.0)
            Wx, Wv = lp_polish_general(target, wide_time, init_radius=5e-3, min_delta=5e-4, factor=20.0)
            if Wv < best_val + 1e-7:
                # continue refining from wide result if not too bad
                if Wv < best_val:
                    best_val = Wv
                    best_x = Wx.copy()
                target = Wx if Wv < max_cos(target)+2e-4 else target
        except Exception:
            pass
        rem2 = BUDGET - (time.time() - t0)
        if rem2 > 5.0:
            try:
                narrow_time = min(rem2-1.0, 30.0)
                Nx, Nv = lp_polish_general(target if 'Wx' not in locals() else (Wx if 'Wx' in locals() else target), narrow_time, init_radius=1e-3, min_delta=1e-9, factor=20.0)
                if Nv <= best_val:
                    best_val = Nv
                    best_x = Nx.copy()
                elif Nv < best_val + 1e-9:
                    best_x = Nx.copy()
            except Exception:
                pass
    # final safety: ensure best among all
    try:
        bv = max_cos(best_x)
    except Exception:
        best_x = xb
    best_x = best_x/np.linalg.norm(best_x, axis=1, keepdims=True)
    # tiny shrink guard against NaN/zero
    best_x = np.where(np.isnan(best_x), 0.0, best_x)
    nr = np.linalg.norm(best_x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best_x = best_x/nr
    return best_x
