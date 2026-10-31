import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _nm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _Lmat(q):
    w, x, y, z = q
    return np.array([[w, -x, -y, -z],
                     [x,  w, -z,  y],
                     [y,  z,  w, -x],
                     [z, -y,  x,  w]], dtype=float)

def _hurwitz_group():
    qs = []
    for i in range(4):
        for s in (-1.0, 1.0):
            q = np.zeros(4); q[i] = s
            qs.append(q)
    for b in range(16):
        q = np.ones(4) * 0.5
        for i in range(4):
            if (b >> i) & 1:
                q[i] = -0.5
        qs.append(q)
    Gs = np.stack([_Lmat(q) for q in qs], axis=0)
    return Gs  # (24,4,4)

def _cyclic_group(m, k=5):
    Gs = []
    for p in range(m):
        a1 = 2*np.pi*p/m
        a2 = 2*np.pi*k*p/m
        c1, s1 = np.cos(a1), np.sin(a1)
        c2, s2 = np.cos(a2), np.sin(a2)
        R = np.array([[c1, -s1, 0, 0],
                      [s1, c1, 0, 0],
                      [0, 0, c2, -s2],
                      [0, 0, s2, c2]], dtype=float)
        Gs.append(R)
    return np.stack(Gs, axis=0)

def _expand(seeds, Gs):
    # seeds (ns,4), Gs (ng,4,4) -> (ns*ng,4)
    ns = seeds.shape[0]
    ng = Gs.shape[0]
    X = np.einsum('gij,kj->kgi', Gs, seeds).reshape(ns*ng, 4)
    return _nm(X)

def _lp(x, deadline):
    x = _nm(x.copy())
    n, d = x.shape
    best = _mc(x)
    rad = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and rad > 1e-14:
        c = (x @ x.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20*rad, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:,None]*d+np.arange(d),
                               J[:,None]*d+np.arange(d),
                               np.full((p,1), n*d)],1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p,1))],1).ravel()
        Aub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
        Aeq = coo_matrix((x.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
        obj = np.zeros(nv); obj[-1]=1.0
        try:
            r = linprog(obj, A_ub=Aub, b_ub=-c[sel], A_eq=Aeq, b_eq=np.zeros(n),
                        bounds=[(-rad,rad)]*(n*d)+[(None,None)], method="highs")
        except Exception:
            break
        if r.status != 0:
            rad *= 0.5
            continue
        y = _nm(x + r.x[:-1].reshape(n,d))
        v = _mc(y)
        if v < best:
            x, best, rad = y, v, min(rad*1.5, 0.05)
        else:
            rad *= 0.5
    return x

def _adam_full(X, steps, b0, b1, lr0, lr1):
    n, d = X.shape
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(1, steps+1):
        f = t/steps
        beta = float(np.exp(lb0+(lb1-lb0)*f))
        lr = float(np.exp(ll0+(ll1-ll0)*f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gm = float(np.max(G))
        W = np.exp(np.clip(beta*(G-gm), -20, 0))
        np.fill_diagonal(W, 0.0)
        Gr = (W @ X)*beta
        Gt = Gr - np.sum(Gr*X, axis=1, keepdims=True)*X
        m = b1a*m+(1-b1a)*Gt
        v = b2a*v+(1-b2a)*(Gt*Gt)
        mh = m/(1-b1a**t); vh = v/(1-b2a**t)
        X = _nm(X - lr*mh/(np.sqrt(vh)+eps))
    return X

def _adam_sym(seeds, Gs, steps, b0=30., b1=500., lr0=0.02, lr1=0.002, deadline=None):
    ns = seeds.shape[0]
    ng = Gs.shape[0]
    GtR = np.stack([G.T for G in Gs], axis=0)  # (ng,4,4)
    m = np.zeros_like(seeds); v = np.zeros_like(seeds)
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    S = seeds.copy()
    for t in range(1, steps+1):
        if deadline is not None and time.time() > deadline:
            break
        f = t/steps
        beta = float(np.exp(lb0+(lb1-lb0)*f))
        lr = float(np.exp(ll0+(ll1-ll0)*f))
        X = np.einsum('gij,kj->kgi', Gs, S).reshape(ns*ng, 4)
        # X already unit (S unit, R orthogonal)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        E = np.exp(np.clip(beta*(G-cmax), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True)+1e-12
        grad = (E @ X)/rs
        Gt = grad - np.sum(grad*X, axis=1, keepdims=True)*X
        # map to seeds: gs[k] = sum_g R_g^T Gt[k*ng+g]
        Gt3 = Gt.reshape(ns, ng, 4)
        # einsum: GtR[g,a,b] * Gt3[k,g,b] -> gs[k,a]
        gs = np.einsum('gab,kgb->ka', GtR, Gt3)
        # average scale for stability (Adam handles it anyway)
        m = b1a*m+(1-b1a)*gs
        v = b2a*v+(1-b2a)*(gs*gs)
        mh = m/(1-b1a**t); vh = v/(1-b2a**t)
        S = _nm(S - lr*mh/(np.sqrt(vh)+eps))
    return S

def _void_samples(Xref, rng, nsamp=20000, k=7):
    C = _nm(rng.normal(size=(nsamp,4)))
    D = C @ Xref.T
    mx = D.max(axis=1)
    idx = np.argsort(mx)[:max(k*40, 200)]
    cand = C[idx]
    # greedy diverse pick of k deepest holes
    picked = []
    order = np.argsort(mx[idx])
    cand = cand[order]
    for c in cand:
        ok = True
        for p in picked:
            if float(np.dot(c,p)) > 0.86:
                ok = False; break
        if ok:
            picked.append(c)
        if len(picked) >= k:
            break
    while len(picked) < k:
        picked.append(cand[len(picked)])
    return np.stack(picked, axis=0)

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    SEARCH = t0 + 148.0
    rng = np.random.default_rng(7)
    cands = []
    for f in ("best.npy","record.npy"):
        try:
            c = np.load(f).astype(float)
            if c.shape==(168,4):
                c = _nm(c)
                if np.all(np.isfinite(c)):
                    cands.append(c)
        except Exception:
            pass
    if not cands:
        xb = _nm(rng.normal(size=(168,4))); bv=_mc(xb)
    else:
        vs=[_mc(c) for c in cands]
        xb=cands[int(np.argmin(vs))].copy(); bv=float(min(vs))
    best_sym_val = 10.0
    best_sym_X = None
    G24 = _hurwitz_group()
    G12 = _cyclic_group(12, k=5)
    G8 = _cyclic_group(8, k=3)
    # symmetric configs: (ns, Gs, ninit_each, steps)
    fam_list = [
        (7, G24, 8, 750),
        (14, G12, 2, 550),
        (21, G8, 2, 500),
    ]
    # precompute voids of incumbent for seeding
    try:
        hole_seeds = _void_samples(xb, rng, nsamp=20000, k=21)
    except Exception:
        hole_seeds = _nm(rng.normal(size=(21,4)))
    init_counter = 0
    for (ns, Gs, ninit, ssteps) in fam_list:
        ng = Gs.shape[0]
        for ii in range(ninit):
            if time.time() > SEARCH-8.0:
                break
            init_counter += 1
            mode = ii % 4
            if mode == 0:
                S0 = _nm(rng.normal(size=(ns,4)))
            elif mode == 1:
                # sample from incumbent points
                perm = rng.permutation(168)[:ns]
                S0 = xb[perm] + 0.05*rng.normal(size=(ns,4))
                S0 = _nm(S0)
            elif mode == 2:
                # holes of incumbent
                if ns <= len(hole_seeds):
                    sel = rng.choice(len(hole_seeds), size=ns, replace=False)
                    S0 = hole_seeds[sel] + 0.03*rng.normal(size=(ns,4))
                else:
                    base = _void_samples(xb, rng, nsamp=15000, k=ns)
                    S0 = base + 0.03*rng.normal(size=(ns,4))
                S0 = _nm(S0)
            else:
                S0 = _nm(rng.normal(size=(ns,4)))
                # mix one hole in
                try:
                    h = hole_seeds[rng.integers(0,len(hole_seeds))]
                    S0[0] = h/_np_norm(h) if False else h
                except Exception:
                    pass
            try:
                Sopt = _adam_sym(S0, Gs, steps=ssteps, b0=30., b1=500., lr0=0.02, lr1=0.002, deadline=SEARCH-6.0)
            except Exception:
                continue
            try:
                Xsym = _expand(Sopt, Gs)
            except Exception:
                continue
            if Xsym.shape != (168,4):
                continue
            vsym = _mc(Xsym)
            if vsym < best_sym_val:
                best_sym_val = vsym
                best_sym_X = Xsym.copy()
            # keep symmetric if it beats global (unlikely but keep)
            if vsym < bv:
                bv = vsym; xb = Xsym.copy()
            # symmetry breaking
            for nz in (0.005, 0.015, 0.03):
                if time.time() > SEARCH-4.0:
                    break
                try:
                    X0 = _nm(Xsym + nz*rng.normal(size=(168,4)))
                    # short full anneal
                    Xo = _adam_full(X0, steps=550, b0=30., b1=420., lr0=0.02, lr1=0.0018)
                except Exception:
                    continue
                v = _mc(Xo)
                do_lp = False; bud = 0.0
                if v < bv:
                    bv = v; xb = Xo.copy(); do_lp = True; bud = 1.5
                elif v < bv+0.002 and (init_counter % 2 == 0):
                    do_lp = True; bud = 0.7
                if do_lp and time.time() < SEARCH-1.0:
                    try:
                        Xp = _lp(Xo, min(time.time()+bud, SEARCH+2.0, HARD-6.0))
                        vp = _mc(Xp)
                        if vp < bv:
                            bv = vp; xb = Xp.copy()
                    except Exception:
                        pass
            # also try direct LP on symmetric (will jam) quickly once
            if vsym < bv+0.01 and time.time() < SEARCH-3.0:
                try:
                    Xp = _lp(Xsym, min(time.time()+0.6, SEARCH, HARD-6.0))
                    vp = _mc(Xp)
                    if vp < bv:
                        bv = vp; xb = Xp.copy()
                except Exception:
                    pass
        if time.time() > SEARCH-8.0:
            break
    # fallback basin-hopping from best if time remains (a few plain kicks)
    extra = 0
    while time.time() < SEARCH-2.0 and extra < 6:
        extra += 1
        try:
            X0 = _nm(xb + 0.04*rng.normal(size=(168,4)))
            Xo = _adam_full(X0, steps=600, b0=28., b1=420., lr0=0.02, lr1=0.002)
            v = _mc(Xo)
            if v < bv:
                bv = v; xb = Xo.copy()
                Xp = _lp(Xo, min(time.time()+1.0, HARD-4.0))
                vp = _mc(Xp)
                if vp < bv:
                    bv = vp; xb = Xp.copy()
        except Exception:
            pass
    if time.time() < HARD-2.0:
        try:
            xb = _lp(xb, min(HARD-1.0, time.time()+max(5.0, HARD-time.time()-1.0)))
        except Exception:
            pass
    return _nm(xb)
