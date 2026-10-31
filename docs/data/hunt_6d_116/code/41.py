import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, deadline):
    X = _norm(X.copy())
    n, d = X.shape
    best = _maxcos(X)
    bx = X.copy()
    radius = 2.0e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        if time.time() > deadline - 0.3:
            break
        G = X @ X.T
        c = G[iu]
        m = float(c.max())
        thr = m - max(20.0*radius, 1e-9)
        idx = np.where(c > thr)[0]
        if idx.size == 0:
            break
        if idx.size > 1400:
            o = np.argsort(c[idx])[-1400:]
            idx = idx[o]
        sel = np.zeros_like(c, dtype=bool)
        sel[idx] = True
        I = iu[0][sel]; J = iu[1][sel]
        cs = c[sel]
        p = len(I)
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p,1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
            A_eq = coo_matrix((X.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
            obj = np.zeros(nv); obj[-1]=1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cs, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n,d)
        y = _norm(y)
        vv = _maxcos(y)
        if vv < best:
            best = vv; bx = y.copy(); X = y
            radius = min(radius*1.5, 0.05)
        else:
            if vv <= best + 1e-15:
                X = y
            radius *= 0.5
    return bx, best

def _tangent_bases(X, rng):
    n, d = X.shape
    k = d-1
    T = np.empty((n, d, k))
    for i in range(n):
        x = X[i]
        Z = rng.standard_normal((d, k))
        Z = Z - np.outer(x, x @ Z)
        Q, _ = np.linalg.qr(Z)
        # ensure tangent
        Q = Q - np.outer(x, x @ Q)
        Q, _ = np.linalg.qr(Q)
        T[i] = Q
    return T

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    rng = np.random.default_rng(12345)
    # load incumbent
    def _load(f):
        try:
            a = np.load(f)
            a = np.asarray(a, dtype=float)
            if a.shape != (116,6) or not np.all(np.isfinite(a)):
                return None
            nn = np.linalg.norm(a, axis=1, keepdims=True)
            if np.any(nn < 1e-12):
                return None
            return a / nn
        except Exception:
            return None
    Xb = _load("best.npy")
    Xr = _load("record.npy")
    cands = []
    if Xb is not None:
        cands.append(Xb)
    if Xr is not None:
        cands.append(Xr)
    if len(cands)==0:
        Xcur = _norm(rng.standard_normal((116,6)))
    else:
        vals = [_maxcos(X) for X in cands]
        Xcur = cands[int(np.argmin(vals))].copy()
    Xcur = _norm(Xcur)
    Mcur = _maxcos(Xcur)
    best_X = Xcur.copy()
    best_val = Mcur
    n, d = 116, 6
    N = n*(d-1)
    iu_all = np.triu_indices(n, 1)
    iu0 = iu_all[0]; iu1 = iu_all[1]
    # precompute skew basis indices for rotations
    skew = [(a,b) for a in range(6) for b in range(a+1,6)]
    step_sizes = [1e-2, 3e-3, 1e-3, 3e-4]
    it = 0
    # initial short LP to stabilize
    if HARD - time.time() > 8:
        Xp, vp = _lp_polish(Xcur, time.time()+3.0)
        if vp < best_val:
            best_val = vp; best_X = Xp.copy()
        Xcur = Xp.copy(); Mcur = vp if vp < Mcur else Mcur
        # re-eval
        Mcur = _maxcos(Xcur)

    while time.time() < HARD - 9.0:
        it += 1
        remain = HARD - time.time()
        if remain < 9.0:
            break
        G = Xcur @ Xcur.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        C = G[iu0, iu1]
        slack = M - C
        # active set slack < 3e-7
        act_mask = slack < 3.0e-7
        p_act = int(np.sum(act_mask))
        # adapt threshold to keep 300..900 actives
        thr_act = 3.0e-7
        if p_act < 250:
            thr_act = 1.0e-6
            act_mask = slack < thr_act
            p_act = int(np.sum(act_mask))
        if p_act < 200:
            thr_act = 5.0e-6
            act_mask = slack < thr_act
            p_act = int(np.sum(act_mask))
        if p_act > 1200:
            # keep tightest 900
            o = np.argsort(slack)[:900]
            act_mask = np.zeros_like(act_mask); act_mask[o]=True
            p_act = 900
        if p_act < 50:
            # degenerate, do LP and random move
            Xp, vp = _lp_polish(Xcur, time.time()+2.0)
            if vp < best_val:
                best_val=vp; best_X=Xp.copy()
            Xcur = Xp + rng.normal(scale=1e-3,size=(n,d))
            Xcur=_norm(Xcur)
            continue
        AI = iu0[act_mask]; AJ = iu1[act_mask]
        c_act = C[act_mask]
        # extended set for scoring: slack < 0.015 (cap 4000)
        ext_mask = slack < 0.015
        p_ext = int(np.sum(ext_mask))
        if p_ext > 4000:
            o = np.argsort(slack)[:4000]
            ext_mask = np.zeros_like(ext_mask); ext_mask[o]=True
            p_ext = 4000
        EI = iu0[ext_mask]; EJ = iu1[ext_mask]
        c_ext = C[ext_mask]
        # tangent bases
        T = _tangent_bases(Xcur, rng)  # n,6,5
        # build rigidity matrix p_act x N
        A = np.zeros((p_act, N), dtype=float)
        # Tx_j blocks: T_i^T x_j
        # vectorize
        for kk in range(p_act):
            i = AI[kk]; j = AJ[kk]
            A[kk, i*5:(i+1)*5] = T[i].T @ Xcur[j]
            A[kk, j*5:(j+1)*5] = T[j].T @ Xcur[i]
        # SVD
        try:
            U_s, s_s, Vt = np.linalg.svd(A, full_matrices=False)
        except Exception:
            break
        # rank
        tol = 1e-9
        rank = int(np.sum(s_s > tol))
        K = N - rank
        if K <= 16:
            # essentially rigid: polish and kick orthogonally
            Xp, vp = _lp_polish(Xcur, time.time()+2.0)
            if vp < best_val:
                best_val=vp; best_X=Xp.copy()
                Xcur=Xp.copy(); Mcur=vp
            else:
                Xcur = Xcur + rng.normal(scale=2e-3,size=(n,d))
                Xcur=_norm(Xcur)
            continue
        Z = Vt[rank:,:].T  # N x K
        # rotation basis N x 15
        Qrot = np.zeros((N, len(skew)))
        for qq,(a,b) in enumerate(skew):
            for i in range(n):
                xi = Xcur[i]
                # E x = e_a * x_b - e_b * x_a
                u = np.zeros(6); u[a]=xi[b]; u[b]=-xi[a]
                Qrot[i*5:(i+1)*5, qq] = T[i].T @ u
        Qr, _ = np.linalg.qr(Qrot)
        # sample null directions
        S = 1200
        if remain < 40:
            S = 600
        Coef = rng.standard_normal((S, K))
        # project: Zc = Z @ Coef.T -> N x S
        Zc = Z @ Coef.T  # N x S
        # remove rotation
        proj = Qr @ (Qr.T @ Zc)
        Zc = Zc - proj
        nnrm = np.linalg.norm(Zc, axis=0, keepdims=True)
        nnrm[nnrm<1e-15]=1.0
        Zc = Zc / nnrm
        # reshape to U_all S x n x 6 : U_i = T_i y_i
        Yall = Zc.T.reshape(S, n, 5)  # S,n,5
        # U = sum_k T[i,:,k]*Y
        # einsum
        Uall = np.einsum('n g k, s n k -> s n g', T, Yall)  # S,n,6
        # score with extended set: lin and quad
        # lin = x_i . u_j + x_j . u_i
        # gather
        try:
            Ui = Uall[:, EI, :]  # S,p_ext,6
            Uj = Uall[:, EJ, :]
        except Exception:
            # memory fallback smaller S
            Ui = Uall[:, EI, :]
            Uj = Uall[:, EJ, :]
        X_EI = Xcur[EI]  # p,6
        X_EJ = Xcur[EJ]
        lin = np.sum(Ui * X_EJ[None,:,:], axis=2) + np.sum(Uj * X_EI[None,:,:], axis=2)  # S,p
        dotUU = np.sum(Ui*Uj, axis=2)
        sq = np.sum(Uall*Uall, axis=2)  # S,n
        sqI = sq[:, EI]; sqJ = sq[:, EJ]
        q = dotUU - 0.5*c_ext[None,:]*(sqI+sqJ)
        s_probe = 3e-3
        pred = c_ext[None,:] + s_probe*lin + (s_probe**2)*q
        pred_max = np.max(pred, axis=1)
        # also consider negative probe (lin flips, q same)
        pred2 = c_ext[None,:] - s_probe*lin + (s_probe**2)*q
        pred_max2 = np.max(pred2, axis=1)
        pred_best = np.minimum(pred_max, pred_max2)
        order = np.argsort(pred_best)[:6]
        # exact trials along best few
        improved = False
        best_trial_val = M
        best_trial_X = None
        for oi in order:
            y = Zc[:, oi]  # N
            Ym = y.reshape(n,5)
            Um = np.einsum('n g k, n k -> n g', T, Ym)  # n,6
            for ss in step_sizes:
                for sgn in (1.0, -1.0):
                    Xt = _norm(Xcur + (ss*sgn)*Um)
                    v = _maxcos(Xt)
                    if v < best_trial_val:
                        best_trial_val = v
                        best_trial_X = Xt.copy()
                    if v < best_val - 1e-12:
                        pass
            # early time check
            if time.time() > HARD - 9.0:
                break
        if best_trial_X is not None and best_trial_val < M - 1e-12:
            # accept and LP polish short
            Xcur = best_trial_X.copy()
            Mcur = best_trial_val
            if Mcur < best_val:
                best_val = Mcur; best_X = Xcur.copy()
            budget = min(4.0, (HARD - time.time()) - 5.0)
            if budget > 1.0:
                Xp, vp = _lp_polish(Xcur, time.time()+budget)
                if vp < best_val:
                    best_val = vp; best_X = Xp.copy()
                Xcur = Xp.copy(); Mcur = _maxcos(Xcur)
                if Mcur < best_val:
                    best_val=Mcur; best_X=Xcur.copy()
            continue
        else:
            # no improving null step: try to move along best direction + polish to change basin
            # take the top-1 direction with smallest probe step that doesn't blow up
            # choose small step and polish
            if best_trial_X is not None:
                # if trial within small tolerance, explore from there
                if best_trial_val < M + 2e-4:
                    Xcur = best_trial_X.copy()
                    budget = min(3.0, (HARD - time.time())-4.0)
                    if budget > 0.8:
                        Xp, vp = _lp_polish(Xcur, time.time()+budget)
                        if vp < best_val:
                            best_val=vp; best_X=Xp.copy()
                        # keep exploring from polished point even if not better (basin hop)
                        # only keep if not much worse
                        if vp < M + 5e-4:
                            Xcur = Xp.copy(); Mcur = vp
                        else:
                            # revert to best and add tiny null noise
                            Xcur = best_X.copy() + rng.normal(scale=5e-4,size=(n,d))
                            Xcur=_norm(Xcur)
                    else:
                        Xcur = best_trial_X.copy()
                    continue
            # fallback random null nudge
            y = Zc[:, int(order[0])]
            Ym = y.reshape(n,5)
            Um = np.einsum('n g k, n k -> n g', T, Ym)
            Xcur = _norm(Xcur + 1e-3*Um + rng.normal(scale=2e-4,size=(n,d)))
            if HARD - time.time() > 6:
                Xp, vp = _lp_polish(Xcur, time.time()+1.5)
                if vp < best_val:
                    best_val=vp; best_X=Xp.copy()
                Xcur = Xp.copy()
    return _norm(best_X)
