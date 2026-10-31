import time
import itertools
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    np.maximum(n, 1e-12, out=n)
    return A / n

# ---- binary polyhedral groups for discard check ----
def _bt_list():
    pts = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4); v[i] = s
            pts.append(v)
    for s in itertools.product((-0.5, 0.5), repeat=4):
        pts.append(np.array(s, dtype=float))
    return np.array(pts)

def _bo_list():
    bt = _bt_list()
    extra = []
    for i, j in itertools.combinations(range(4), 2):
        for si in (-1.0, 1.0):
            for sj in (-1.0, 1.0):
                v = np.zeros(4)
                v[i] = si / np.sqrt(2.0); v[j] = sj / np.sqrt(2.0)
                extra.append(v)
    return np.vstack([bt, np.array(extra)])

def _bi_list():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    phi_inv = 1.0 / phi
    pts = []
    for i in range(4):
        for s in (-1.0, 1.0):
            v = np.zeros(4); v[i] = s
            pts.append(v)
    for s in itertools.product((-0.5, 0.5), repeat=4):
        pts.append(np.array(s, dtype=float))
    for perm in itertools.permutations(range(4)):
        inv = 0
        for a in range(4):
            for b in range(a+1, 4):
                if perm[a] > perm[b]:
                    inv += 1
        if inv % 2 == 1:
            continue
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            v = np.zeros(4)
            for p in range(4):
                vi = perm[p]
                if vi == 0:
                    v[p] = 0.0
                elif vi == 1:
                    v[p] = signs[0]*0.5
                elif vi == 2:
                    v[p] = signs[1]*phi/2.0
                else:
                    v[p] = signs[2]*phi_inv/2.0
            pts.append(v)
    arr = np.array(pts, dtype=float)
    _, idx = np.unique(np.round(arr, 12), axis=0, return_index=True)
    return arr[idx]

def _can_tile(n, orbit_sizes):
    # unbounded knap reachable?
    reach = {0}
    for _ in range(12):
        new = set(reach)
        for r in reach:
            for o in orbit_sizes:
                if r+o == n:
                    return True
                if r+o < n:
                    new.add(r+o)
        if new == reach:
            break
        reach = new
    return n in reach

# ---- cyclic / dihedral torus subgroups of SO(4) ----
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
    Rf = np.einsum('mij,jk->mik', R, F)
    return np.concatenate([R, Rf], axis=0)

def _expand(S, G):
    K = S.shape[0]
    M = G.shape[0]
    X = np.einsum('mij,kj->kmi', G, S).reshape(K*M, 4)
    return X

def _lse_opt_seeds(Y0, G, alphas=(50.0, 200.0, 1000.0), iters=(220, 220, 260)):
    K = Y0.shape[0]
    y = Y0.ravel().copy()
    for alpha, mx in zip(alphas, iters):
        def fg(yy):
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
            ssum = float(E.sum())*0.5
            if ssum <= 0 or not np.isfinite(ssum):
                return 1e6, np.zeros_like(yy)
            f = dm + np.log(ssum)/alpha
            W = E / ssum
            GX = W @ X  # (N,4)
            G3 = GX.reshape(K, G.shape[0], 4)
            # pullback: grad_S[k,l]=sum_m sum_j G3[k,m,j]*G[m,j,l]
            GS = np.einsum('kmj,mjl->kl', G3, G)
            # chain through normalization
            dot = (GS*S).sum(axis=1, keepdims=True)
            GY = (GS - dot*S) / nr
            return f, GY.ravel()
        try:
            res = minimize(fg, y, jac=True, method='L-BFGS-B',
                           options={'maxiter': mx, 'maxfun': mx*4, 'ftol': 1e-12, 'gtol': 1e-8})
            y = res.x
        except Exception:
            break
    Y = y.reshape(K, 4)
    S = _normed(Y)
    return S

def _lp_polish(x, seconds):
    x = _normed(x.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 2e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time()-start < seconds and radius > 1e-14:
        Gf = x @ x.T
        ci = Gf[iu]
        m = ci.max()
        thr = max(20*radius, 1e-9)
        sel = ci > m - thr
        if np.count_nonzero(sel) == 0:
            radius *= 0.5
            continue
        I = iu[0][sel]; J = iu[1][sel]; c = ci[sel]
        p = len(I)
        nv = n*d+1
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2*d+1)
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method='highs')
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y = _normed(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _anneal(x, steps, b0, b1, lr0, mu):
    n, d = x.shape
    v = np.zeros_like(x)
    lb = np.log(b1/b0) if b1 > b0 else 0.0
    for t in range(steps):
        frac = t/max(1, steps-1)
        beta = b0*np.exp(lb*frac)
        lr = lr0*(1.0-0.5*frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta*(G-gmax))
        np.fill_diagonal(W, 0.0)
        S = W.sum(axis=1, keepdims=True)+1e-300
        A = (W@x)/S
        dot = (A*x).sum(axis=1, keepdims=True)
        Gt = A-dot*x
        v = mu*v+Gt
        vd = (v*x).sum(axis=1, keepdims=True)
        v = v-vd*x
        x = x-lr*v
        x = _normed(x)
    return x

def _refine_void(v, Y, steps=150, beta=200.0, lr=0.05):
    for _ in range(steps):
        g = Y@v
        gm = g.max()
        w = np.exp(beta*(g-gm)); s=w.sum()+1e-300
        a = (w@Y)/s
        grad = a-(a@v)*v
        v = v-lr*grad
        n = np.linalg.norm(v)
        if n < 1e-12:
            break
        v = v/n
    return v

def solve():
    t0 = time.time()
    BUDGET = 172.0
    FINAL = 28.0
    rng = np.random.default_rng(20261031)
    cands = []
    for fn in ("best.npy","record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape==(292,4):
                cands.append(_normed(a))
        except Exception:
            pass
    if len(cands)==0:
        cands.append(_normed(rng.standard_normal((292,4))))
    vals = [_max_cos(a) for a in cands]
    x_best = cands[int(np.argmin(vals))].copy()
    best_val = float(min(vals))

    # discard check for binary polyhedral / hyperoctahedral
    try:
        nbt, nbo, nbi = len(_bt_list()), len(_bo_list()), len(_bi_list())
    except Exception:
        nbt, nbo, nbi = 24, 48, 120
    # generic orbit sizes 24/48/120 cannot tile 292 alone; need special
    # small-orbit sets: BT {4,6,8,12}, BO {6,8,12,16,24}, BI {12,20,30,60}
    tile_bt = _can_tile(292, [24,12,8,6,4])
    tile_bo = _can_tile(292, [48,24,16,12,8,6])
    tile_bi = _can_tile(292, [120,60,30,20,12])
    # hyperoctahedral generic 384 >292, special {8,16,24,32,48,64,96,128,192}
    tile_ho = _can_tile(292, [8,16,24,32,48,64,96,128,192])
    # we keep flags but focus on cyclic which always tiles via divisors
    _ = (tile_bt, tile_bo, tile_bi, tile_ho, nbt, nbo, nbi)

    sym_deadline = t0 + 55.0
    # cyclic/dihedral configs that give exactly 292
    configs = [
        ('c',73,1),('c',73,5),('c',73,9),('c',73,15),('c',73,22),
        ('c',146,3),('c',146,7),('c',292,3),
        ('d',73,1),('d',73,7),('d',146,3),
    ]
    sym_cands = []
    rs = np.random.default_rng(777)
    for kind, m, p in configs:
        if time.time() > sym_deadline:
            break
        G = _build_cyclic(m,p) if kind=='c' else _build_dihedral(m,p)
        M = G.shape[0]
        if 292 % M != 0:
            continue
        K = 292//M
        if K* M != 292 or K>6:
            continue
        for restart in range(2):
            if time.time() > sym_deadline:
                break
            Y0 = _normed(rs.standard_normal((K,4)))
            S = _lse_opt_seeds(Y0, G)
            X = _expand(S, G)
            X = _normed(X)
            # dedup closer than ~1e-9
            Gf = X@X.T
            np.fill_diagonal(Gf, -2.0)
            if float(Gf.max()) > 1-1e-12:
                continue
            v = _max_cos(X)
            sym_cands.append((v, X.copy()))
            if v < best_val + 0.01 and time.time() < sym_deadline-8:
                # quick symmetry-breaking LP to test basin
                t_rem = max(0.5, min(4.0, sym_deadline-time.time()-1.0))
                Xp = X + 1e-4*rs.standard_normal(X.shape)
                # tangent project
                Xp = Xp - (Xp*X).sum(axis=1,keepdims=True)*X
                Xp = _normed(X+1e-4*_normed(Xp-X))
                Xp = _lp_polish(Xp, t_rem)
                vp = _max_cos(Xp)
                if vp < best_val:
                    x_best, best_val = Xp, vp
    # keep best symmetric raw for final expansion if competitive
    sym_cands.sort(key=lambda t: t[0])
    for v, X in sym_cands[:2]:
        if time.time() > sym_deadline:
            break
        if v < best_val + 0.015:
            Xp = X + 1e-4*rs.standard_normal(X.shape)
            Xp = _normed(Xp)
            Xp = _lp_polish(Xp, 4.0)
            vp = _max_cos(Xp)
            if vp < best_val:
                x_best, best_val = Xp, vp

    explore_end = t0 + BUDGET - FINAL
    sigmas = [0.005, 0.015, 0.04, 0.10, 0.25]
    it = 0
    # short lock-in polish
    if time.time() < explore_end:
        x_best = _lp_polish(x_best, min(3.0, max(0.5, explore_end-time.time()-2.0)))
        best_val = _max_cos(x_best)
    while time.time() < explore_end and it < 200:
        it += 1
        tl = explore_end - time.time()
        if tl < 2.0:
            break
        X = x_best.copy()
        if it % 3 == 0:
            Gf = X@X.T
            np.fill_diagonal(Gf, -2.0)
            mx = Gf.max()
            tight = (Gf > mx-1e-9).sum(axis=1)
            degs = (Gf > mx-1e-7).sum(axis=1)
            ratt = np.where(tight==0)[0]
            order = np.argsort(-degs)
            k = int(rng.integers(2,6))
            chosen = list(ratt[:min(len(ratt),k)])
            for o in order:
                if len(chosen) >= k:
                    break
                if o not in chosen:
                    if rng.random()<0.15 and len(chosen)>0:
                        continue
                    chosen.append(int(o))
            sigma = float(sigmas[rng.integers(len(sigmas))]*0.3)
            X = _normed(X+sigma*rng.standard_normal(X.shape))
            for idx in chosen:
                Y = np.delete(X, idx, axis=0)
                Cc = _normed(rng.standard_normal((12000 if tl>20 else 6000,4)))
                D = Cc@Y.T
                sc = D.max(axis=1)
                oi = np.argpartition(sc,4)[:4]
                pick = oi[rng.integers(len(oi))]
                vv = _refine_void(Cc[pick].copy(), Y)
                X[idx]=vv
            steps,b0,b1,lr0,mu = (800,30.0,400.0,0.06,0.85) if rng.random()<0.5 else (400,60.0,800.0,0.04,0.90)
            X = _anneal(X, steps,b0,b1,lr0,mu)
            qs = 2.0 if tl>15 else 1.0
            Xp = _lp_polish(X, min(qs, tl-0.2))
            vp = _max_cos(Xp)
            if vp < best_val:
                x_best, best_val = Xp, vp
        else:
            sigma = float(sigmas[rng.integers(len(sigmas))])
            X = _normed(X+sigma*rng.standard_normal(X.shape))
            if rng.random()<0.6:
                k = int(rng.integers(1,5))
                idx = rng.choice(292,size=k,replace=False)
                R = _normed(rng.standard_normal((k,4)))
                X[idx]=R
                X = _normed(X)
            r = rng.random()
            if r<0.4:
                steps,b0,b1,lr0,mu = 900,25.0,350.0,0.06,0.85
            elif r<0.7:
                steps,b0,b1,lr0,mu = 450,50.0,600.0,0.05,0.88
            else:
                steps,b0,b1,lr0,mu = 250,100.0,1000.0,0.03,0.90
            X = _anneal(X, steps,b0,b1,lr0,mu)
            if _max_cos(X) < best_val+0.005:
                qs = 2.0 if tl>15 else 1.0
                Xp = _lp_polish(X, min(qs, tl-0.2))
                vp = _max_cos(Xp)
                if vp < best_val:
                    x_best, best_val = Xp, vp
    remain = t0+BUDGET-time.time()
    fp = min(FINAL, max(5.0, remain))
    x_best = _lp_polish(x_best, fp)
    x_best = _normed(x_best)
    return x_best
