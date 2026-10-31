import time
import itertools
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

# ---------- quaternion groups (for discard check) ----------
def _bt_list():
    pts = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4); v[i] = s
            pts.append(v)
    for s in itertools.product((-0.5, 0.5), repeat=4):
        pts.append(np.array(s, dtype=float))
    return np.array(pts)  # 24

def _bo_list():
    bt = _bt_list()
    extra = []
    for i, j in itertools.combinations(range(4), 2):
        for si in (-1.0, 1.0):
            for sj in (-1.0, 1.0):
                v = np.zeros(4)
                v[i] = si / np.sqrt(2.0); v[j] = sj / np.sqrt(2.0)
                extra.append(v)
    extra = np.array(extra)  # 24
    return np.vstack([bt, extra])  # 48

def _bi_list():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    phi_inv = 1.0 / phi
    base_vals = np.array([0.0, 0.5, phi / 2.0, phi_inv / 2.0])
    pts = []
    # 8 axis
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4); v[i] = s
            pts.append(v)
    # 16 half
    for s in itertools.product((-0.5, 0.5), repeat=4):
        pts.append(np.array(s, dtype=float))
    # 96 even perms
    for perm in itertools.permutations(range(4)):
        # parity
        inv = 0
        for a in range(4):
            for b in range(a+1, 4):
                if perm[a] > perm[b]:
                    inv += 1
        if inv % 2 == 1:
            continue
        # perm maps position p -> base_vals[perm[p]]? we need assignment of values to positions
        # values order (0, .5, phi/2, phiinv/2) placed at positions according to perm
        # equivalent: for this perm, value index at position p is perm[p]
        # iterate signs for the 3 nonzero values
        # find where zero is
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            v = np.zeros(4)
            si = 0
            for p in range(4):
                vi = perm[p]
                if vi == 0:
                    v[p] = 0.0
                elif vi == 1:
                    v[p] = signs[0] * 0.5
                elif vi == 2:
                    v[p] = signs[1] * phi / 2.0
                else:
                    v[p] = signs[2] * phi_inv / 2.0
            pts.append(v)
    arr = np.array(pts, dtype=float)
    # deduplicate (zero sign duplicates shouldn't exist, but be safe)
    # round and unique
    r = np.round(arr, 12)
    # use view unique
    _, idx = np.unique(r, axis=0, return_index=True)
    return arr[idx]  # expect 120

def _build_cyclic(order, p):
    G = np.zeros((order, 4, 4))
    for t in range(order):
        a1 = 2*np.pi*t/order
        a2 = 2*np.pi*p*t/order
        c1, s1 = np.cos(a1), np.sin(a1)
        c2, s2 = np.cos(a2), np.sin(a2)
        G[t] = np.array([[c1,-s1,0,0],[s1,c1,0,0],[0,0,c2,-s2],[0,0,s2,c2]])
    return G

def _build_dihedral(order, p):
    R = _build_cyclic(order, p)
    F = np.diag([1.0,-1.0,1.0,-1.0])
    G2 = R @ F  # right multiply? compose
    # group = R union R*F
    return np.concatenate([R, G2 @ np.eye(4).reshape(1,4,4) if False else G2], axis=0)

def _expand(S, G):
    # S (K,4), G (M,4,4), X[i]=G[m] @ S[k]
    K = S.shape[0]; M = G.shape[0]
    X = np.einsum('mij,kj->kmi', G, S).reshape(K*M, 4)
    return X

def _lse_opt_seeds(Y0, G, alphas=(50.0,200.0,1000.0), iters=(250,250,300)):
    from scipy.optimize import minimize
    K = Y0.shape[0]
    y = Y0.ravel().copy()
    for alpha, mx in zip(alphas, iters):
        def fun_grad(yy):
            Y = yy.reshape(K, 4)
            nr = np.sqrt((Y*Y).sum(axis=1, keepdims=True))
            nr = np.maximum(nr, 1e-12)
            S = Y / nr
            X = _expand(S, G)
            N = X.shape[0]
            D = X @ X.T
            np.fill_diagonal(D, -10.0)
            dm = float(D.max())
            E = np.exp(alpha*(D-dm))
            np.fill_diagonal(E, 0.0)
            Ssum = float(E.sum())*0.5
            if Ssum <= 0 or not np.isfinite(Ssum):
                return 1e6, np.zeros_like(yy)
            f = dm + np.log(Ssum)/alpha
            # weights normalized over ordered pairs: sum E = 2*Ssum
            W = E / (2.0*Ssum)  # sums to 1 over ordered? sum=1
            # grad_X: for each i, sum_j (W_ij+W_ji) X_j ; W symmetric
            # W symmetric so 2*W @ X but W sums to 1 (ordered sum =1?) E sum =2Ssum so W sum=1
            # d f / d D_ij (i<j) = E_ij/(alpha? no) -> E_ij/(Ssum*alpha?) wait check:
            # f = dm + log(Ssum)/alpha, dSsum/dD_ij = alpha*E_ij (for i<j, one term)
            # df = E_ij/Ssum dD_ij. So weight w_ij = E_ij/Ssum for i<j.
            # ordered W' = E/(2Ssum), then grad = 2*(W+W^T?) Let's just compute directly:
            # grad_X = (1/Ssum) * (E @ X)  (since E symmetric, each ordered contributes)
            # because sum_{i<j} w_ij (dD) with w=E/Ssum; dD_ij = dX_i.X_j + X_i.dX_j => grad_i = sum_j w_ij X_j (with w symmetric extension)
            # with ordered E/Ssum, E_sym/Ssum @ X gives same.
            W2 = E / Ssum  # ordered sum =2
            GX = W2 @ X
            # back to seeds: grad_S[k] = sum_m G_m^T? grad: X=G s => ds = G^T dX
            # row form: grad_s = grad_x @ G
            M = G.shape[0]
            GXr = GX.reshape(K, M, 4)
            GS = np.zeros((K, 4))
            for m in range(M):
                GS += GXr[:, m, :] @ G[m]
            # to Y
            GY = (GS - (GS*S).sum(axis=1, keepdims=True)*S) / nr
            return float(f), GY.ravel()
        try:
            res = minimize(fun_grad, y, jac=True, method='L-BFGS-B',
                           options={'maxiter': mx, 'ftol': 1e-14, 'gtol': 1e-8})
            y = res.x
        except Exception:
            break
    Y = y.reshape(K, 4)
    S = _normed(Y)
    X = _expand(S, G)
    return X, S

def _full_lbfgs(X0, alpha=600.0, maxiter=120):
    from scipy.optimize import minimize
    N = X0.shape[0]
    y = X0.ravel().copy()
    def fun_grad(yy):
        Y = yy.reshape(N, 4)
        nr = np.sqrt((Y*Y).sum(axis=1, keepdims=True))
        nr = np.maximum(nr, 1e-12)
        S = Y / nr
        D = S @ S.T
        np.fill_diagonal(D, -10.0)
        dm = float(D.max())
        E = np.exp(alpha*(D-dm))
        np.fill_diagonal(E, 0.0)
        Ssum = float(E.sum())*0.5
        if Ssum <= 0 or not np.isfinite(Ssum):
            return 1e6, np.zeros_like(yy)
        f = dm + np.log(Ssum)/alpha
        W2 = E / Ssum
        GS = W2 @ S
        GY = (GS - (GS*S).sum(axis=1, keepdims=True)*S) / nr
        return float(f), GY.ravel()
    try:
        res = minimize(fun_grad, y, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-7})
        y = res.x
    except Exception:
        pass
    Y = y.reshape(N, 4)
    return _normed(Y)

def _lp_polish(x, deadline):
    try:
        from scipy.optimize import linprog
        from scipy.sparse import coo_matrix
    except Exception:
        return x
    x = _normed(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu]
        mval = float(c.max())
        sel = c > mval - max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n*d+1
        try:
            rows = np.repeat(np.arange(p), 2*d+1)
            cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method='highs')
        except Exception:
            break
        if getattr(res,'status',1)!=0:
            radius*=0.5; continue
        try:
            y = x + res.x[:-1].reshape(n,d)
        except Exception:
            radius*=0.5; continue
        y = _normed(y)
        vv = _max_cos(y)
        if vv < best:
            x,best,radius = y,vv,min(radius*1.5,0.05)
        else:
            radius*=0.5
    return x

def solve():
    t0 = time.time()
    TL = 172.0
    deadline_final = t0 + TL
    rng = np.random.default_rng(292292)
    # incumbent
    cands = []
    for fn in ("best.npy","record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape==(292,4) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a,axis=1,keepdims=True)
                if nr.min()>1e-9:
                    cands.append(a/nr)
        except Exception:
            pass
    if len(cands)==0:
        Xb = _normed(rng.standard_normal((292,4)))
    else:
        vals=[_max_cos(c) for c in cands]
        Xb = cands[int(np.argmin(vals))].copy()
    best_inc = _max_cos(Xb)
    # --- binary polyhedral discard check (free left action => orbit size = |H|) ---
    try:
        bt = _bt_list(); bo = _bo_list(); bi = _bi_list()
        sizes = [len(bt), len(bo), len(bi)]
    except Exception:
        sizes = [24,48,120]
    canreach = any(292 % s == 0 for s in sizes)
    # left action free, so unions can only reach multiples; 292=4*73 not divisible -> discard
    # hyperoctahedral divisors check (kept for completeness, not searched heavily)
    # proceed to cyclic torus families which can reach 292 exactly
    sym_best = None; sym_val = 2.0
    sym_budget_end = t0 + 95.0
    # candidate list: (order, p, K)
    cand_list = []
    for p in [1,2,3,5,11,17]:
        cand_list.append((73,p,4))
    for p in [1,3,5]:
        cand_list.append((146,p,2))
    for p in [1,3]:
        cand_list.append((292,p,1))
    # small deterministic shuffle
    # prioritize 73 families
    for (order,p,K) in cand_list:
        if time.time() > sym_budget_end:
            break
        if order*K != 292:
            continue
        G = _build_cyclic(order,p)
        # init seeds
        Y0 = rng.standard_normal((K,4))
        Y0 = _normed(Y0)  # start on sphere
        try:
            X, S = _lse_opt_seeds(Y0, G, alphas=(50.0,200.0,800.0), iters=(200,200,220))
        except Exception:
            continue
        # dedup check <1e-8 (dot>1-5e-17 essentially exact)
        try:
            Gd = X @ X.T
            np.fill_diagonal(Gd,-2.0)
            mx = float(Gd.max())
            # duplicate if extremely close to 1
            if mx > 1.0-1e-12:
                # check if true duplicate (orbits collided)
                # discard this candidate
                continue
            v = mx
        except Exception:
            continue
        if v < sym_val:
            sym_val = v; sym_best = X.copy()
        # early exit if already beating incumbent by margin
        if sym_val < best_inc - 0.002 and time.time() > t0+50:
            break
    # expand best symmetric: break symmetry + polish, compare to incumbent
    X_best_overall = Xb.copy(); best_overall = best_inc
    # polish symmetric candidate if it exists and is not hopeless
    if sym_best is not None and sym_val < best_inc + 0.01:
        # tiny tangent perturbation 1e-4
        pert = rng.standard_normal(sym_best.shape)*1e-4
        # project to tangent
        pert -= (pert*sym_best).sum(axis=1,keepdims=True)*sym_best
        Xs = _normed(sym_best + pert)
        # short full-dim L-BFGS to break symmetry (budget aware)
        remain = deadline_final - time.time()
        if remain > 40:
            try:
                Xs = _full_lbfgs(Xs, alpha=500.0, maxiter=80)
            except Exception:
                pass
            try:
                Xs = _full_lbfgs(Xs, alpha=1000.0, maxiter=60)
            except Exception:
                pass
        # LP polish with limited budget
        lp_end = min(deadline_final-22.0, time.time()+12.0)
        if lp_end - time.time() > 1.0:
            try:
                Xs = _lp_polish(Xs, lp_end)
            except Exception:
                pass
        try:
            vs = _max_cos(Xs)
        except Exception:
            vs = 2.0
        if vs < best_overall:
            X_best_overall = Xs; best_overall = vs
        # also keep raw sym if its polished version worse but raw better? raw already worse than vs typically
    # final polish of overall best with remaining time
    final_end = deadline_final - 0.5
    # give at least some polish to overall
    if final_end - time.time() > 2.0:
        try:
            X_best_overall = _lp_polish(X_best_overall, final_end)
            best_overall = _max_cos(X_best_overall)
        except Exception:
            pass
    X_best_overall = _normed(X_best_overall)
    if not np.all(np.isfinite(X_best_overall)):
        X_best_overall = Xb
    return X_best_overall
