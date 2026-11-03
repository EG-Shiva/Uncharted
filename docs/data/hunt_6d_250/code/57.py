import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import linprog

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _tangent_bases(X):
    n, d = X.shape
    k = d - 1
    B = np.empty((n, d, k), dtype=np.float64)
    eye = np.eye(d, dtype=np.float64)
    for i in range(n):
        xi = X[i]
        jmax = int(np.argmax(np.abs(xi)))
        cols = []
        for e in range(d):
            if e == jmax:
                continue
            v = eye[e].copy()
            v = v - float(v @ xi) * xi
            cols.append(v)
        M = np.stack(cols, axis=1)  # 6x5
        Q, _ = np.linalg.qr(M)
        # ensure tangent (project residual)
        # Q already orthonormal spanning tangent
        B[i] = Q
    return B

def _adam_polish(X0, steps=300, lr=0.008):
    n, d = X0.shape
    Y = X0.copy()
    m = np.zeros_like(Y)
    v = np.zeros_like(Y)
    b1 = 0.9; b2 = 0.999; eps = 1e-8
    b1p = 1.0; b2p = 1.0
    # precompute beta schedule
    # beta 500 -> 5000 exponential
    l0 = np.log(500.0); l1 = np.log(5000.0)
    for t in range(1, steps+1):
        f = (t-1)/max(1, steps-1)
        beta = float(np.exp(l0*(1-f)+l1*f))
        nrm = np.sqrt(np.sum(Y*Y, axis=1, keepdims=True))
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        if not np.isfinite(M):
            break
        D = G - M
        D *= beta
        np.clip(D, -30.0, 0.0, out=D)
        np.exp(D, out=D)
        np.fill_diagonal(D, 0.0)
        S = float(np.sum(D))
        if not np.isfinite(S) or S <= 0:
            break
        D /= S
        GX = D @ X
        GX *= 2.0
        dot = np.sum(GX*X, axis=1, keepdims=True)
        GY = GX - dot*X
        GY /= nrm
        if not np.all(np.isfinite(GY)):
            break
        m *= b1; m += (1.0-b1)*GY
        v *= b2; v += (1.0-b2)*(GY*GY)
        b1p *= b1; b2p *= b2
        denom1 = 1.0-b1p; denom2 = 1.0-b2p
        # mhat, vhat without extra alloc
        # Y -= lr*mhat/(sqrt(vhat)+eps)
        Mh = m/denom1
        Vh = v/denom2
        Y -= lr * Mh / (np.sqrt(Vh)+eps)
        if not np.all(np.isfinite(Y)):
            break
    Xn = _norm_rows(Y)
    return Xn

def _lp_polish(x, best_v, deadline, k_top=4000, r0=2e-3):
    n, d = x.shape
    x = _norm_rows(x)
    try:
        bv = _max_cos(x)
        if bv < best_v:
            best_v = bv
    except Exception:
        return x, best_v
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    ar = np.arange(d)
    it = 0
    while time.time() < deadline - 1.0 and radius > 1e-13 and it < 60:
        it += 1
        Gall = x @ x.T
        c_all = Gall[iu0, iu1]
        m = float(np.max(c_all))
        window = max(20.0*radius, 5e-7)
        sel = c_all > m - window
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > k_top:
            idx = np.argpartition(-c_all, k_top)[:k_top]
            I = iu0[idx]; J = iu1[idx]; c_sel = c_all[idx]
        else:
            I = iu0[sel]; J = iu1[sel]; c_sel = c_all[sel]
        p = I.shape[0]
        nv = n*d+1
        try:
            cols = np.concatenate([I[:,None]*d+ar, J[:,None]*d+ar, np.full((p,1), n*d)], axis=1).ravel()
            rows = np.repeat(np.arange(p), 2*d+1)
            vals = np.concatenate([x[J], x[I], -np.ones((p,1))], axis=1).ravel()
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d), np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            bounds = [(-radius,radius)]*(n*d)+[(None,float(m))]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n), bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        dx = np.asarray(res.x[:-1], dtype=np.float64).reshape(n,d)
        y = _norm_rows(x+dx)
        try:
            vv = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if vv < best_v - 1e-12 and np.all(np.isfinite(y)):
            x = y; best_v = vv
            radius = min(radius*1.5, 0.012)
        else:
            radius *= 0.5
    return x, best_v

def solve():
    t_start = time.time()
    BUDGET = 168.0
    def remaining():
        return BUDGET - (time.time()-t_start)
    n, d = 250, 6
    k = d-1
    Nvars = n*k
    best = None; best_val = np.inf
    for fn in ("best.npy","record.npy"):
        try:
            z = np.asarray(np.load(fn), dtype=np.float64)
            if z.shape==(n,d):
                z = _norm_rows(z)
                if np.all(np.isfinite(z)):
                    vv = _max_cos(z)
                    if vv < best_val:
                        best_val = vv; best = z.copy()
        except Exception:
            continue
    if best is None:
        rng = np.random.default_rng(0)
        best = rng.standard_normal((n,d)); best=_norm_rows(best)
        best_val=_max_cos(best)
    cur = best.copy(); cur_val=float(best_val)
    # build plateau Jacobian
    try:
        G0 = cur @ cur.T
        iu = np.triu_indices(n,1)
        gall = G0[iu]
        M0 = float(np.max(gall))
        thr = M0 - 2e-6
        mask = gall >= thr
        I = iu[0][mask].astype(np.int64); J = iu[1][mask].astype(np.int64)
        # cap ~2000 closest if too many
        if I.size > 2200:
            idx = np.argpartition(-gall[mask], 2200)[:2200]
            I=I[idx]; J=J[idx]
        # if too few, widen
        if I.size < 400:
            thr2 = M0 - 1e-5
            mask2 = gall >= thr2
            I = iu[0][mask2].astype(np.int64); J = iu[1][mask2].astype(np.int64)
            if I.size > 2200:
                idx = np.argpartition(-gall[mask2], 2200)[:2200]
                I=I[idx]; J=J[idx]
        B = _tangent_bases(cur)
        p = I.size
        A = np.zeros((p, Nvars), dtype=np.float64)
        # rows: B_i^T x_j and B_j^T x_i
        for r in range(p):
            ii=int(I[r]); jj=int(J[r])
            xi = cur[ii]; xj = cur[jj]
            # B[ii] is d x k, so B^T x = k vec
            A[r, ii*k:(ii+1)*k] = B[ii].T @ xj
            A[r, jj*k:(jj+1)*k] = B[jj].T @ xi
        # Gram eig for smallest modes
        Gram = A.T @ A
        # symmetrize
        Gram = 0.5*(Gram+Gram.T)
        # add tiny diag for stability? no, want exact null
        w, V = np.linalg.eigh(Gram)
        # V columns ascending; take 30 smallest
        nmodes = min(30, Nvars)
        Vsmall = V[:, :nmodes]  # Nvars x nmodes
    except Exception:
        Vsmall=None; B=None
    cands = []  # list of (rawmax, Xstep)
    if Vsmall is not None and B is not None:
        try:
            stepsizes = [0.02,0.05,0.10,0.20]
            # precompute per-mode U reshaped norms
            for mi in range(Vsmall.shape[1]):
                if remaining() < 90:
                    break
                vec = Vsmall[:, mi].copy()  # 1250
                U = vec.reshape(n,k)  # tangent coeffs
                # per-point norms
                rnorm = np.sqrt(np.sum(U*U, axis=1))
                rmax = float(np.max(rnorm))
                if rmax < 1e-12:
                    continue
                # ambient unit dirs
                # Udir_i = B_i @ U_i / r_i
                Uhat = np.zeros((n,d), dtype=np.float64)
                for i in range(n):
                    if rnorm[i] > 1e-14:
                        Uhat[i] = B[i] @ (U[i]/rnorm[i])
                for sgn in (1.0,-1.0):
                    U2 = U if sgn>0 else -U
                    # Uhat sign
                    Uh = Uhat if sgn>0 else -Uhat
                    for s in stepsizes:
                        theta = s * rnorm / rmax  # (n,)
                        c = np.cos(theta); si = np.sin(theta)
                        Xn = c[:,None]*cur + si[:,None]*Uh
                        Xn = _norm_rows(Xn)
                        try:
                            vv = _max_cos(Xn)
                        except Exception:
                            continue
                        cands.append((vv, Xn))
            # keep best raw for polish (lowest max)
            cands.sort(key=lambda t: t[0])
            # limit polish count by time
            npol = min(len(cands), 90)
            if remaining() < 60:
                npol = min(npol, 30)
            to_polish = cands[:npol]
        except Exception:
            to_polish=[]
    else:
        to_polish=[]
    # Adam polish each
    polished = []
    inc_M = float(cur_val)
    # incumbent adjacency at 1e-4
    try:
        Ginc = cur @ cur.T
        np.fill_diagonal(Ginc, -2.0)
        inc_mask = (Ginc >= inc_M-1e-4)
        # upper triangle count
        iu0,iu1_ = np.triu_indices(n,1)
        inc_vec = inc_mask[iu0,iu1_]
        inc_cnt = int(np.count_nonzero(inc_vec))
        if inc_cnt==0:
            inc_cnt=1
    except Exception:
        inc_vec=None; inc_cnt=1
    best_overall = best.copy(); best_overall_val=float(best_val)
    # polish loop
    for idx,(rawv,Xs) in enumerate(to_polish):
        if remaining() < 35:
            break
        # short Adam 300 steps; reduce steps if low time
        steps = 300
        if remaining() < 50:
            steps = 150
        try:
            Xp = _adam_polish(Xs, steps=steps, lr=0.008)
            vv = _max_cos(Xp)
        except Exception:
            continue
        # adjacency diff
        try:
            Gp = Xp @ Xp.T
            np.fill_diagonal(Gp,-2.0)
            Mp = float(np.max(Gp))
            iu0,iu1_ = np.triu_indices(n,1)
            pv = (Gp[iu0,iu1_] >= Mp-1e-4)
            # Jaccard distance vs incumbent
            inter = int(np.count_nonzero(pv & inc_vec))
            union = int(np.count_nonzero(pv | inc_vec))
            dist = 1.0 - (inter/max(1,union))
        except Exception:
            dist=0.0; Mp=vv
        polished.append((vv, Xp, dist, Mp))
        if vv < best_overall_val and np.all(np.isfinite(Xp)):
            best_overall = Xp.copy(); best_overall_val = vv
    # filter distinct >=5%
    filt = [t for t in polished if t[2] >= 0.05]
    # improvers: vv < inc_M (strict) or < best?
    improvers = [t for t in filt if t[0] < inc_M - 1e-12]
    if len(improvers)==0:
        # take best distinct even if not improving (may improve after LP)
        filt_sorted = sorted(filt, key=lambda t: t[0])
        improvers = filt_sorted[:2]
    else:
        # two most distinct improvers
        improvers = sorted(improvers, key=lambda t: -t[2])[:2]
    # LP-polish the two in remaining time
    for vv,Xp,dist,Mp in improvers:
        if remaining() < 12:
            break
        # split remaining time
        deadline = t_start + BUDGET - 2.0
        # give each ~ half remaining
        try:
            Xl, vl = _lp_polish(Xp.copy(), min(vv,best_overall_val), deadline, k_top=4000, r0=2e-3)
            if vl < best_overall_val and np.all(np.isfinite(Xl)):
                best_overall = Xl.copy(); best_overall_val = vl
        except Exception:
            continue
    # final fallback: if no improvement, LP-polish best_overall briefly
    if remaining() > 12 and best_overall_val >= best_val - 1e-12:
        try:
            deadline = t_start + BUDGET - 1.0
            Xl, vl = _lp_polish(best_overall.copy(), best_overall_val, deadline, k_top=4000, r0=5e-4)
            if vl < best_overall_val and np.all(np.isfinite(Xl)):
                best_overall = Xl.copy(); best_overall_val = vl
        except Exception:
            pass
    out = _norm_rows(best_overall.copy())
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, 0.0)
        nrm = np.linalg.norm(out, axis=1, keepdims=True)
        bad = (nrm[:,0]<1e-12)|(~np.isfinite(nrm[:,0]))
        out[bad]=0.0; out[bad,0]=1.0
        out=_norm_rows(out)
    return out
