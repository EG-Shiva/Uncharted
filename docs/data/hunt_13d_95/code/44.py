import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def build_X_A(R, F):
    # R (7,13) -> 91 orbit points, F (4,13) -> 4 free
    rn = np.linalg.norm(R, axis=1, keepdims=True); rn[rn==0]=1.0
    fn = np.linalg.norm(F, axis=1, keepdims=True); fn[fn==0]=1.0
    Un = R / rn
    Fn = F / fn
    orb = np.empty((91, 13))
    for j in range(7):
        u = Un[j]
        for k in range(13):
            orb[j*13+k] = np.roll(u, k)
    X = np.vstack([orb, Fn])
    return X, Un, Fn, rn[:,0], fn[:,0]

def build_X_B(R8):
    # R8 (8,13): first 7 full orbits (91) + 4 shifts of 8th rep (0,3,6,9)
    rn = np.linalg.norm(R8, axis=1, keepdims=True); rn[rn==0]=1.0
    Un = R8 / rn
    orb = np.empty((91, 13))
    for j in range(7):
        u = Un[j]
        for k in range(13):
            orb[j*13+k] = np.roll(u, k)
    u8 = Un[7]
    part = np.stack([np.roll(u8, k) for k in (0,3,6,9)], axis=0)
    X = np.vstack([orb, part])
    return X, Un, rn[:,0]

def sym_adam_A(R0, F0, steps, beta0=30.0, beta1=1000.0, lr=0.04):
    R = R0.copy(); F = F0.copy()
    mR = np.zeros_like(R); vR = np.zeros_like(R)
    mF = np.zeros_like(F); vF = np.zeros_like(F)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0, l1 = np.log(beta0), np.log(beta1)
    for t in range(1, steps+1):
        beta = float(np.exp(l0 + (l1-l0)*(t-1)/max(1,steps-1)))
        X, Un, Fn, rn, fn = build_X_A(R, F)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = G.max()
        E = np.exp(beta*(G-mx))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S==0:
            break
        Ws = E / S
        GX = Ws @ X
        Gorb = GX[:91]; Gfree = GX[91:]
        # grad wrt Un
        gU = np.zeros((7,13))
        for j in range(7):
            acc = np.zeros(13)
            for k in range(13):
                acc += np.roll(Gorb[j*13+k], -k)
            gU[j] = acc
        # chain through normalization
        dotR = (gU*Un).sum(axis=1, keepdims=True)
        gR = (gU - dotR*Un) / rn[:,None]
        dotF = (Gfree*Fn).sum(axis=1, keepdims=True)
        gF = (Gfree - dotF*Fn) / fn[:,None]
        # adam
        mR = b1*mR + (1-b1)*gR; vR = b2*vR + (1-b2)*gR*gR
        mF = b1*mF + (1-b1)*gF; vF = b2*vF + (1-b2)*gF*gF
        mRh = mR/(1-b1**t); vRh = vR/(1-b2**t)
        mFh = mF/(1-b1**t); vFh = vF/(1-b2**t)
        R -= lr * mRh/(np.sqrt(vRh)+eps)
        F -= lr * mFh/(np.sqrt(vFh)+eps)
    X,_,_,_,_ = build_X_A(R, F)
    return X, R, F

def sym_adam_B(R0, steps, beta0=30.0, beta1=1000.0, lr=0.04):
    R = R0.copy()
    mR = np.zeros_like(R); vR = np.zeros_like(R)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0, l1 = np.log(beta0), np.log(beta1)
    for t in range(1, steps+1):
        beta = float(np.exp(l0 + (l1-l0)*(t-1)/max(1,steps-1)))
        X, Un, rn = build_X_B(R)
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mx = G.max()
        E = np.exp(beta*(G-mx))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.finite(S) or S==0:
            break
        Ws = E / S
        GX = Ws @ X
        Gorb = GX[:91]; Gpart = GX[91:]
        gU = np.zeros((8,13))
        for j in range(7):
            acc = np.zeros(13)
            for k in range(13):
                acc += np.roll(Gorb[j*13+k], -k)
            gU[j] = acc
        acc8 = np.zeros(13)
        for gi, k in enumerate((0,3,6,9)):
            acc8 += np.roll(Gpart[gi], -k)
        gU[7] = acc8
        dotR = (gU*Un).sum(axis=1, keepdims=True)
        gR = (gU - dotR*Un) / rn[:,None]
        mR = b1*mR + (1-b1)*gR; vR = b2*vR + (1-b2)*gR*gR
        mRh = mR/(1-b1**t); vRh = vR/(1-b2**t)
        R -= lr * mRh/(np.sqrt(vRh)+eps)
    X,_,_ = build_X_B(R)
    return X, R

def full_adam_refine(X0, steps, beta0=60.0, beta1=800.0, lr=0.02):
    Y = X0.copy()
    m = np.zeros_like(Y); v = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    l0, l1 = np.log(beta0), np.log(beta1)
    for t in range(1, steps+1):
        nrm = np.linalg.norm(Y, axis=1, keepdims=True); nrm[nrm==0]=1.0
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        beta = float(np.exp(l0 + (l1-l0)*(t-1)/max(1,steps-1)))
        mx = G.max()
        E = np.exp(beta*(G-mx))
        np.fill_diagonal(E, 0.0)
        S = E.sum()
        if not np.isfinite(S) or S==0:
            break
        Ws = E / S
        GX = Ws @ X
        dot = (GX*X).sum(axis=1, keepdims=True)
        gY = (GX - dot*X) / nrm
        m = b1*m + (1-b1)*gY; vv = b2*v + (1-b2)*gY*gY
        # use v variable name conflict; keep separate
        v = vv
        mh = m/(1-b1**t); vh = v/(1-b2**t)
        Y -= lr * mh/(np.sqrt(vh)+eps)
    nrm = np.linalg.norm(Y, axis=1, keepdims=True); nrm[nrm==0]=1.0
    return Y / nrm

def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        sel = c > m - max(20*radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n*d+1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        A_ub = coo_matrix((vals,(rows,cols)), shape=(p,nv)).tocsr()
        A_eq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))), shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius,radius)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n,d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius*1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(13)
    # baseline
    base = None; base_val = 1e9
    for fn in ("best.npy","record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape==(95,13):
                a = a/np.linalg.norm(a,axis=1,keepdims=True)
                vv = max_cos(a)
                if vv < base_val:
                    base_val = vv; base = a.copy()
        except Exception:
            pass
    if base is None:
        base = rng.standard_normal((95,13)); base/=np.linalg.norm(base,axis=1,keepdims=True)
        base_val = max_cos(base)
    best = base.copy(); best_val = base_val
    cands = []
    # symmetry search
    n_rest = 8
    steps_sym = 1400
    for r in range(n_rest):
        if time.time() > t0+75:
            break
        if r % 2 == 0:
            R0 = rng.standard_normal((7,13)); F0 = rng.standard_normal((4,13))
            try:
                X,_,_ = sym_adam_A(R0,F0,steps_sym,30,1000,0.04)
            except Exception:
                continue
        else:
            R0 = rng.standard_normal((8,13))
            try:
                X, _ = sym_adam_B(R0,steps_sym,30,1000,0.04)
            except Exception:
                continue
        vv = max_cos(X)
        # penalize coincident orbit degeneracy: if vv>0.9 skip storing as good but keep for diversity
        cands.append((vv, X.copy()))
        if vv < best_val:
            best_val = vv; best = X.copy()
    # sort cands
    cands.sort(key=lambda z: z[0])
    # break symmetry + full refine top 3 symmetric (even if worse than record, they seed new basin)
    DEI = 0
    for idx in range(min(3, len(cands))):
        if time.time() > t0+120:
            break
        _, Xs = cands[idx]
        for pert in (2e-3, 8e-3):
            if time.time() > t0+125:
                break
            Y = Xs + pert*rng.standard_normal(Xs.shape)
            Y /= np.linalg.norm(Y,axis=1,keepdims=True)
            try:
                Y = full_adam_refine(Y, 900, 50, 600, 0.02)
            except Exception:
                pass
            vv = max_cos(Y)
            if vv < best_val:
                best_val = vv; best = Y.copy()
            # keep refined for LP stage
            cands.append((vv, Y.copy()))
            DEI += 1
            if DEI >= 4:
                break
    # LP polish: best symmetric-broken + baseline
    cands.sort(key=lambda z: z[0])
    # try polish up to 3 distinct
    tried = 0
    for _, Xc in cands[:3]:
        if time.time() > deadline-12:
            break
        try:
            Y = lp_polish(Xc, 4.0, deadline-6)
            vv = max_cos(Y)
            if vv < best_val:
                best_val = vv; best = Y.copy()
        except Exception:
            pass
        tried += 1
    # final polish of overall best
    if time.time() < deadline-8:
        try:
            remain = max(2.0, min(12.0, deadline-time.time()-2.0))
            Y = lp_polish(best, remain*0.6, deadline-1.0)
            if max_cos(Y) < best_val:
                best = Y; best_val = max_cos(Y)
        except Exception:
            pass
    # safety: if somehow worse than baseline, return baseline
    if max_cos(best) > base_val and base is not None:
        best = base
    best = best/np.linalg.norm(best,axis=1,keepdims=True)
    return best
