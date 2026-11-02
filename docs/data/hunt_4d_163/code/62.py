import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, deadline, r0=0.004, cap=1300):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                sel[np.argmax(c)] = True
                cnt = 1
            if cnt > cap:
                idx = np.argsort(-c)[:cap]
                sel2 = np.zeros_like(sel, dtype=bool)
                sel2[idx] = True
                sel = sel2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            cols[:, :d] = I[:, None] * d + ar
            cols[:, d:2 * d] = J[:, None] * d + ar
            cols[:, 2 * d] = n * d
            cols = cols.ravel()
            vals = np.empty((p, 2 * d + 1), dtype=float)
            vals[:, :d] = x[J]
            vals[:, d:2 * d] = x[I]
            vals[:, 2 * d] = -1.0
            vals = vals.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _full_gd(x0, steps, T0, T1, lr0, mu, rng, noise0=0.0, deadline=None):
    x = x0.copy()
    n, d = x.shape
    v = np.zeros_like(x)
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    bv = float(G.max())
    bx = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lr = lr0 * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S < 1e-300 or not np.isfinite(S):
            S = 1e-300
        grad = (E @ x) / S
        grad -= (grad * x).sum(axis=1, keepdims=True) * x
        v = mu * v + (1.0 - mu) * grad
        x = x - lr * v
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _lbfgs_full(x_init, deadline, taus=(0.002, 0.0008, 0.00035), iters=(70, 70, 90)):
    x = x_init / np.maximum(np.linalg.norm(x_init, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.5:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * float(np.log(s / Np))
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            GY = (GX - (GX * X).sum(axis=1, keepdims=True) * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 5),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
        except Exception:
            break
    Yf = y.reshape(n, d)
    Yf /= np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
    return Yf

def _expand_sym(th, al, be, K, m, F):
    L = len(th)
    nstd = L * K
    nf = F.shape[0]
    X = np.empty((nstd + nf, 4))
    twopi = 2 * np.pi
    dA = twopi / K
    dB = twopi * m / K
    for j in range(L):
        c = np.cos(th[j]); s = np.sin(th[j])
        a0 = al[j]; b0 = be[j]
        for kk in range(K):
            a = a0 + kk * dA
            b = b0 + kk * dB
            X[j * K + kk, 0] = c * np.cos(a)
            X[j * K + kk, 1] = c * np.sin(a)
            X[j * K + kk, 2] = s * np.cos(b)
            X[j * K + kk, 3] = s * np.sin(b)
    if nf > 0:
        Fn = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
        X[nstd:] = Fn
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X

def _sym_opt(K, L, m, rng, taus=(0.003, 0.0008, 0.0003), iters=(120, 120, 140)):
    n = L * K + 1
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    twopi = 2 * np.pi
    dA = twopi / K
    dB = twopi * m / K
    kk = np.arange(K)
    # init params
    th0 = rng.uniform(0.35, 1.22, size=L)
    al0 = rng.uniform(0, twopi, size=L)
    be0 = rng.uniform(0, twopi, size=L)
    f0 = rng.randn(4)
    p0 = np.concatenate([th0, al0, be0, f0])
    bounds = [(0.03, 1.54)] * L + [(None, None)] * (2 * L + 4)
    p = p0.copy()
    for tau, maxiter in zip(taus, iters):
        def fg(pp, _tau=tau):
            th = pp[0:L]; al = pp[L:2*L]; be = pp[2*L:3*L]; fr = pp[3*L:3*L+4]
            rn_f = max(float(np.linalg.norm(fr)), 1e-12)
            F = (fr / rn_f).reshape(1, 4)
            X = _expand_sym(th, al, be, K, m, F)
            g = (X @ X.T)[iu0, iu1]
            mx = float(g.max())
            e = np.exp((g - mx) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(mx), np.zeros_like(pp)
            Ls = mx + _tau * float(np.log(s / Np))
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            grad = np.zeros_like(pp)
            # orbit grads
            for j in range(L):
                idx = j * K + kk
                GXj = GX[idx]
                c = float(np.cos(th[j])); s_ = float(np.sin(th[j]))
                av = al[j] + kk * dA
                bv = be[j] + kk * dB
                Ca = np.cos(av); Sa = np.sin(av); Cb = np.cos(bv); Sb = np.sin(bv)
                dT = np.empty((K, 4))
                dT[:, 0] = -s_ * Ca; dT[:, 1] = -s_ * Sa; dT[:, 2] = c * Cb; dT[:, 3] = c * Sb
                dA_ = np.empty((K, 4))
                dA_[:, 0] = -c * Sa; dA_[:, 1] = c * Ca; dA_[:, 2] = 0; dA_[:, 3] = 0
                dB_ = np.empty((K, 4))
                dB_[:, 0] = 0; dB_[:, 1] = 0; dB_[:, 2] = -s_ * Sb; dB_[:, 3] = s_ * Cb
                grad[j] = float((GXj * dT).sum())
                grad[L + j] = float((GXj * dA_).sum())
                grad[2*L + j] = float((GXj * dB_).sum())
            # free point
            GXf = GX[-1]
            Fv = X[-1]
            dot = float(GXf @ Fv)
            Gfr = (GXf - dot * Fv) / rn_f
            grad[3*L:3*L+4] = Gfr
            return float(Ls), grad
        try:
            res = minimize(fg, p, method="L-BFGS-B", jac=True,
                           bounds=bounds,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter*6),
                                    "ftol": 1e-14, "gtol": 1e-9})
            p = res.x
        except Exception:
            break
    th = p[0:L]; al = p[L:2*L]; be = p[2*L:3*L]; fr = p[3*L:3*L+4]
    F = (fr / max(float(np.linalg.norm(fr)), 1e-12)).reshape(1,4) if False else (fr/np.maximum(np.linalg.norm(fr),1e-12)).reshape(1,4)
    X = _expand_sym(th, al, be, K, m, F)
    return X, _max_cos(X)

def _deep_hole(others, rng, nc=8000, refine=3):
    d = others.shape[1]
    cand = rng.randn(nc, d)
    cand /= np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-12)
    chunk = 4000
    m_all = np.empty(nc)
    for a in range(0, nc, chunk):
        b = min(nc, a+chunk)
        m_all[a:b] = (cand[a:b] @ others.T).max(axis=1)
    bi = int(np.argmin(m_all))
    best_v = float(m_all[bi]); best_c = cand[bi].copy()
    rad = 0.3
    for _ in range(refine):
        prop = rng.randn(2500, d) * (rad*0.5)
        prop -= (prop @ best_c)[:, None]*best_c[None, :]
        prop = best_c[None, :]+prop
        prop /= np.maximum(np.linalg.norm(prop, axis=1, keepdims=True), 1e-12)
        m = (prop @ others.T).max(axis=1)
        j = int(np.argmin(m))
        if float(m[j]) < best_v:
            best_v = float(m[j]); best_c = prop[j].copy()
        rad *= 0.45
    return best_c

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    LOOP_END = HARD_END - 12.0
    rng = np.random.RandomState(20261102)
    cands = []
    try:
        xb = np.load("best.npy").astype(np.float64)
        xb = xb/np.linalg.norm(xb,axis=1,keepdims=True)
        cands.append(xb)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(np.float64)
        xr = xr/np.linalg.norm(xr,axis=1,keepdims=True)
        cands.append(xr)
    except Exception:
        pass
    if len(cands)==0:
        tmp = rng.randn(163,4); tmp/=np.linalg.norm(tmp,axis=1,keepdims=True)
        cands.append(tmp)
    best = min(cands, key=lambda z: _max_cos(z)).copy()
    best_val = _max_cos(best)
    # ---- symmetric manifold search ----
    configs = [(27,6,5),(27,6,8),(27,6,11),(18,9,5),(18,9,7),(18,9,11),(54,3,5),(54,3,17),(54,3,25)]
    sym_best = None; sym_val = 1e9
    sym_deadline = t0 + 34.0
    for (K,L,mm) in configs:
        if time.time() > sym_deadline:
            break
        for rep in range(2):
            if time.time() > sym_deadline:
                break
            try:
                X, v = _sym_opt(K,L,mm,rng)
            except Exception:
                continue
            if v < sym_val:
                sym_val = v; sym_best = X.copy()
    # three-stage symmetry breaking
    cur_sym = sym_best if sym_best is not None else best.copy()
    cur_sym_val = sym_val if sym_best is not None else best_val
    amps = [0.005, 0.02, 0.05]
    for si, amp in enumerate(amps):
        if time.time() > LOOP_END - 8.0:
            break
        stage_end = LOOP_END - (2 - si)*22.0
        # start from better of cur_sym and best with noise
        base = cur_sym.copy()
        # tangent noise
        nz = rng.randn(*base.shape)*amp
        nz -= (nz*base).sum(axis=1,keepdims=True)*base
        Xp = base + nz
        Xp /= np.maximum(np.linalg.norm(Xp,axis=1,keepdims=True),1e-12)
        # anneal
        T0 = 0.004 if si==0 else 0.005
        T1 = 0.0004
        steps = 420 if si<2 else 520
        Xp,_ = _full_gd(Xp, steps, T0, T1, 0.22, 0.82, rng, noise0=0.0, deadline=min(stage_end, LOOP_END))
        if time.time() > LOOP_END - 2.0:
            v = _max_cos(Xp)
            if v < best_val:
                best= Xp; best_val=v
            cur_sym,cur_sym_val = Xp,v
            break
        Xp = _lbfgs_full(Xp, min(stage_end,LOOP_END), taus=(0.002,0.0008,0.00035), iters=(60,60,80))
        if time.time() < LOOP_END:
            Xp,_ = _lp_polish(Xp, min(LOOP_END, time.time()+1.2), r0=0.004)
        v = _max_cos(Xp)
        cur_sym, cur_sym_val = Xp.copy(), v
        if v < best_val:
            best = Xp.copy(); best_val = v
    # ---- final basin hopping from global best ----
    n,d = best.shape
    iu0,iu1 = np.triu_indices(n,1)
    it = 0
    while time.time() < LOOP_END:
        it += 1
        base = best if rng.rand()<0.7 else cur_sym
        x = base.copy()
        mode = rng.rand()
        if mode < 0.45:
            sigma = 0.002+rng.rand()*0.012
            x = x + rng.randn(n,d)*sigma
            x/=np.linalg.norm(x,axis=1,keepdims=True)
            G = x@x.T; np.fill_diagonal(G,-2.0)
            rm = G.max(axis=1); order=np.argsort(-rm)
            k = 1 if rng.rand()<0.5 else 2
            used=set()
            for _ in range(k):
                pool = order[:10]
                victim = int(pool[rng.randint(0,len(pool))]) if rng.rand()<0.8 else int(rng.randint(0,n))
                if victim in used: continue
                used.add(victim)
                mask=np.ones(n,bool); mask[victim]=False
                x[victim]=_deep_hole(x[mask],rng,nc=7000+rng.randint(0,4000),refine=3)
            T0=0.003+rng.rand()*0.005; T1=0.0003+rng.rand()*0.0006
            steps=150+rng.randint(0,150)
            lr=0.16+rng.rand()*0.14; mu=0.72+rng.rand()*0.18
            x,_=_full_gd(x,steps,T0,T1,lr*0.35,mu,rng,deadline=LOOP_END)
            if rng.rand()<0.25 and time.time()<LOOP_END-2.0:
                tau=float(rng.choice([0.002,0.001,0.0006]))
                x=_lbfgs_full(x,LOOP_END,taus=(tau,),iters=(70,))
            x,_=_lp_polish(x,min(LOOP_END,time.time()+0.8),r0=0.0035)
        else:
            Gc=base@base.T; np.fill_diagonal(Gc,-2.0)
            rowmax=Gc.max(axis=1)
            w=int(np.argmax(rowmax)) if rng.rand()<0.85 else int(rng.randint(0,n))
            msize=int(rng.choice([4,5,6,7,8]))
            neigh=np.argsort(-Gc[w])[:msize]
            x=base+rng.randn(n,d)*0.004
            common=rng.randn(d)*(0.04+rng.rand()*0.07)
            x[neigh]+=common[None,:]
            x[neigh]+=rng.randn(len(neigh),d)*(0.008+rng.rand()*0.016)
            x/=np.linalg.norm(x,axis=1,keepdims=True)
            flat=(x@x.T)[iu0,iu1]
            topk=np.argpartition(-flat,min(40,len(flat)-1))[:40]
            for _ in range(int(rng.choice([2,3]))):
                ei=int(topk[rng.randint(0,len(topk))])
                a=int(iu0[ei]); b_=int(iu1[ei])
                s=0.015+rng.rand()*0.028
                da=x[a]-x[b_]; db=-da
                da=da-(da@x[a])*x[a]; db=db-(db@x[b_])*x[b_]
                x[a]+=(da/(np.linalg.norm(da)+1e-12))*s
                x[b_]+=(db/(np.linalg.norm(db)+1e-12))*s
            x/=np.linalg.norm(x,axis=1,keepdims=True)
            Gt2=x@x.T; np.fill_diagonal(Gt2,-2.0)
            rm2=Gt2.max(axis=1); order=np.argsort(-rm2)
            k=int(rng.choice([2,3,4]))
            used=set()
            for _ in range(k):
                rr=rng.rand()
                if rr<0.5: victim=int(order[rng.randint(0,10)])
                elif rr<0.75: victim=int(neigh[rng.randint(0,len(neigh))])
                else: victim=int(rng.randint(0,n))
                if victim in used: continue
                used.add(victim)
                mask=np.ones(n,bool); mask[victim]=False
                x[victim]=_deep_hole(x[mask],rng,nc=9000+rng.randint(0,5000),refine=4)
            T0=0.003+rng.rand()*0.004; T1=0.0003+rng.rand()*0.0006
            steps=180+rng.randint(0,150)
            x,_=_full_gd(x,steps,T0,T1,0.20,0.80,rng,deadline=LOOP_END)
            if rng.rand()<0.3 and time.time()<LOOP_END-2.0:
                x=_lbfgs_full(x,LOOP_END,taus=(0.0012,0.0005),iters=(60,70))
            x,_=_lp_polish(x,min(LOOP_END,time.time()+0.9),r0=0.0035)
        v=_max_cos(x)
        if v < best_val:
            best=x.copy(); best_val=v
        if v < cur_sym_val + 0.002:
            cur_sym=x.copy(); cur_sym_val=v
        if time.time() > LOOP_END:
            break
    # final tight LP polish on best
    if time.time() < HARD_END - 2.0:
        best,_=_lp_polish(best,min(HARD_END-0.5,time.time()+4.0),r0=0.003)
        best_val=_max_cos(best)
    best/=np.linalg.norm(best,axis=1,keepdims=True)
    # tiny shrink guard
    return best
