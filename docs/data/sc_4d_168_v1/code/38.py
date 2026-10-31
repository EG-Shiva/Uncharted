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
    return Gs

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
    ns = seeds.shape[0]
    ng = Gs.shape[0]
    X = np.einsum('gij,kj->kgi', Gs, seeds).reshape(ns*ng, 4)
    return _nm(X)

def _lp(x, deadline):
    x = _nm(x.copy())
    n, d = x.shape
    try:
        best = _mc(x)
    except Exception:
        return x
    rad = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and rad > 1e-14:
        try:
            C = (x @ x.T)[iu]
        except Exception:
            break
        m = float(np.max(C))
        sel = C > m - max(20.0*rad, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        cs = C[sel]
        p = len(I)
        if p == 0:
            break
        nv = n*d+1
        rows = np.repeat(np.arange(p), 2*d+1)
        cols = np.concatenate([I[:, None]*d+np.arange(d),
                               J[:, None]*d+np.arange(d),
                               np.full((p, 1), n*d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            Aeq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            r = linprog(obj, A_ub=Aub, b_ub=-cs, A_eq=Aeq, b_eq=np.zeros(n),
                        bounds=[(-rad, rad)]*(n*d)+[(None, None)], method="highs")
        except Exception:
            break
        if r.status != 0:
            rad *= 0.5
            continue
        try:
            y = _nm(x + r.x[:-1].reshape(n, d))
            v = _mc(y)
        except Exception:
            rad *= 0.5
            continue
        if v < best:
            x, best, rad = y, v, min(rad*1.5, 0.05)
        else:
            rad *= 0.5
    return x

def _adam_full(X, steps, b0, b1, lr0, lr1, mu=0.88, deadline=None):
    n, d = X.shape
    vel = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    for s in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        f = s / max(1, steps-1)
        beta = float(np.exp(lb0+(lb1-lb0)*f))
        lr = lr0*(1.0-f)+lr1*f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        E = np.exp(np.clip(beta*(G-cmax), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True)+1e-12
        grad = (E @ X)/rs
        dot = np.sum(grad*X, axis=1, keepdims=True)
        gt = grad - dot*X
        sc = float(np.mean(np.linalg.norm(gt, axis=1)))+1e-12
        gt = gt/sc
        vel = mu*vel+(1.0-mu)*gt
        X = _nm(X - lr*vel)
    return X

def _adam_sym(S, Gs, steps, b0=30., b1=500., lr0=0.02, lr1=0.002, deadline=None):
    ns = S.shape[0]
    ng = Gs.shape[0]
    GtR = np.stack([G.T for G in Gs], axis=0)
    m = np.zeros_like(S); v = np.zeros_like(S)
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    Sc = S.copy()
    for t in range(1, steps+1):
        if deadline is not None and time.time() > deadline:
            break
        f = t/steps
        beta = float(np.exp(lb0+(lb1-lb0)*f))
        lr = float(np.exp(ll0+(ll1-ll0)*f))
        X = np.einsum('gij,kj->kgi', Gs, Sc).reshape(ns*ng, 4)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cmax = float(np.max(G))
        E = np.exp(np.clip(beta*(G-cmax), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True)+1e-12
        grad = (E @ X)/rs
        dot = np.sum(grad*X, axis=1, keepdims=True)
        Gt = grad - dot*X
        Gt3 = Gt.reshape(ns, ng, 4)
        gs = np.einsum('gab,kgb->ka', GtR, Gt3)
        # normalize per-seed scale mildly (Adam is adaptive, keep raw)
        m = b1a*m+(1-b1a)*gs
        v = b2a*v+(1-b2a)*(gs*gs)
        mh = m/(1-b1a**t); vh = v/(1-b2a**t)
        Sc = _nm(Sc - lr*mh/(np.sqrt(vh)+eps))
    return Sc

def _void_samples(Xref, rng, nsamp=20000, k=21):
    C = _nm(rng.normal(size=(nsamp, 4)))
    D = C @ Xref.T
    mx = D.max(axis=1)
    idx = np.argsort(mx)[:max(k*40, 300)]
    cand = C[idx]
    mx2 = mx[idx]
    order = np.argsort(mx2)
    cand = cand[order]
    picked = []
    for c in cand:
        ok = True
        for p in picked:
            if float(np.dot(c, p)) > 0.86:
                ok = False
                break
        if ok:
            picked.append(c)
        if len(picked) >= k:
            break
    while len(picked) < k:
        picked.append(cand[len(picked) % len(cand)])
    return np.stack(picked, axis=0)

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    SYM_END = t0 + 82.0
    SEARCH = t0 + 150.0
    rng = np.random.default_rng(12345)
    cands = []
    for f in ("best.npy", "record.npy"):
        try:
            c = np.load(f).astype(float)
            if c.shape == (168, 4):
                c = _nm(c)
                if np.all(np.isfinite(c)):
                    cands.append(c)
        except Exception:
            pass
    if not cands:
        xb = _nm(rng.normal(size=(168, 4)))
        bv = _mc(xb)
    else:
        vs = [_mc(c) for c in cands]
        xb = cands[int(np.argmin(vs))].copy()
        bv = float(min(vs))
    x_best = xb.copy()
    G24 = _hurwitz_group()
    G12 = _cyclic_group(12, k=5)
    G8 = _cyclic_group(8, k=3)
    try:
        hole_seeds = _void_samples(xb, rng, nsamp=20000, k=21)
    except Exception:
        hole_seeds = _nm(rng.normal(size=(21, 4)))
    fam_list = [(7, G24, 6, 600), (14, G12, 2, 420), (21, G8, 2, 420)]
    for (ns, Gs, ninit, ssteps) in fam_list:
        if time.time() > SYM_END:
            break
        for ii in range(ninit):
            if time.time() > SYM_END:
                break
            mode = ii % 4
            try:
                if mode == 0:
                    S0 = _nm(rng.normal(size=(ns, 4)))
                elif mode == 1:
                    perm = rng.permutation(168)[:ns]
                    S0 = _nm(xb[perm] + 0.05*rng.normal(size=(ns, 4)))
                elif mode == 2:
                    if ns <= len(hole_seeds):
                        sel = rng.choice(len(hole_seeds), size=ns, replace=False)
                        S0 = _nm(hole_seeds[sel] + 0.03*rng.normal(size=(ns, 4)))
                    else:
                        base = _void_samples(xb, rng, nsamp=15000, k=ns)
                        S0 = _nm(base + 0.03*rng.normal(size=(ns, 4)))
                    S0 = _nm(S0)
                else:
                    S0 = _nm(rng.normal(size=(ns, 4)))
                    try:
                        h = hole_seeds[rng.integers(0, len(hole_seeds))]
                        S0[0] = h / (np.linalg.norm(h)+1e-30)
                    except Exception:
                        pass
            except Exception:
                continue
            try:
                Sopt = _adam_sym(S0, Gs, steps=ssteps, b0=30., b1=500., lr0=0.02, lr1=0.002, deadline=SYM_END)
                Xsym = _expand(Sopt, Gs)
            except Exception:
                continue
            if Xsym.shape != (168, 4):
                continue
            try:
                vsym = _mc(Xsym)
            except Exception:
                continue
            if vsym < bv:
                bv = vsym; xb = Xsym.copy(); x_best = xb.copy()
            for nz in (0.005, 0.015, 0.03):
                if time.time() > SYM_END and nz > 0.005:
                    break
                if time.time() > SEARCH:
                    break
                try:
                    X0 = _nm(Xsym + nz*rng.normal(size=(168, 4)))
                    Xo = _adam_full(X0, steps=420, b0=30., b1=420., lr0=0.02, lr1=0.0018, mu=0.88, deadline=SEARCH)
                    v = _mc(Xo)
                except Exception:
                    continue
                do_lp = False; bud = 0.0
                if v < bv:
                    bv = v; xb = Xo.copy(); x_best = xb.copy(); do_lp = True; bud = 1.2
                elif v < bv + 0.002 and (ii % 2 == 0) and nz == 0.005:
                    do_lp = True; bud = 0.5
                if do_lp and time.time() < SEARCH:
                    try:
                        Xp = _lp(Xo, min(time.time()+bud, SEARCH+1.0, HARD-8.0))
                        vp = _mc(Xp)
                        if vp < bv:
                            bv = vp; xb = Xp.copy(); x_best = xb.copy()
                    except Exception:
                        pass
            if vsym < bv + 0.01 and time.time() < SYM_END - 1.0:
                try:
                    Xp = _lp(Xsym, min(time.time()+0.5, HARD-8.0))
                    vp = _mc(Xp)
                    if vp < bv:
                        bv = vp; xb = Xp.copy(); x_best = xb.copy()
                except Exception:
                    pass
    # intensive squeeze / basin hopping from global best
    try:
        # direct long refine of best without noise (tighten)
        if time.time() < SEARCH:
            Xo = _adam_full(x_best.copy(), steps=700, b0=40., b1=600., lr0=0.014, lr1=0.0007, mu=0.88, deadline=SEARCH)
            v = _mc(Xo)
            if v < bv:
                bv = v; xb = Xo.copy(); x_best = xb.copy()
            elif v < bv + 0.001:
                try:
                    Xp = _lp(Xo, min(time.time()+1.0, HARD-6.0))
                    vp = _mc(Xp)
                    if vp < bv:
                        bv = vp; xb = Xp.copy(); x_best = xb.copy()
                except Exception:
                    pass
    except Exception:
        pass
    kick_list = [0.008, 0.015, 0.03, 0.05, 0.012, 0.02]
    ki = 0
    while time.time() < SEARCH - 1.0:
        nz = kick_list[ki % len(kick_list)]; ki += 1
        try:
            X0 = _nm(x_best + nz*rng.normal(size=(168, 4)))
            Xo = _adam_full(X0, steps=550, b0=28., b1=500., lr0=0.018, lr1=0.0012, mu=0.88, deadline=SEARCH)
            v = _mc(Xo)
        except Exception:
            continue
        if v < bv:
            bv = v; xb = Xo.copy(); x_best = xb.copy()
            try:
                Xp = _lp(Xo, min(time.time()+1.2, HARD-4.0))
                vp = _mc(Xp)
                if vp < bv:
                    bv = vp; xb = Xp.copy(); x_best = xb.copy()
            except Exception:
                pass
        elif v < bv + 0.0015 and time.time() < SEARCH - 5.0 and (ki % 3 == 0):
            try:
                Xp = _lp(Xo, min(time.time()+0.6, HARD-4.0))
                vp = _mc(Xp)
                if vp < bv:
                    bv = vp; xb = Xp.copy(); x_best = xb.copy()
            except Exception:
                pass
        if ki > 14:
            break
    if time.time() < HARD - 1.0:
        try:
            xb = _lp(x_best, HARD-0.5)
        except Exception:
            xb = x_best
    try:
        xb = _nm(xb)
        _ = _mc(xb)
        if xb.shape != (168, 4) or not np.all(np.isfinite(xb)):
            xb = _nm(x_best)
    except Exception:
        xb = _nm(x_best)
    return _nm(xb)
