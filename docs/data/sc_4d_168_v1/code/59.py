import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _mc(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _nm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _Lmat(q):
    w, x, y, z = q
    return np.array([[w, -x, -y, -z],
                     [x, w, -z, y],
                     [y, z, w, -x],
                     [z, -y, x, w]], dtype=float)

def _hurwitz():
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
    return np.stack([_Lmat(q) for q in qs], axis=0)

def _cyclic(m, k):
    G = np.zeros((m, 4, 4))
    for p in range(m):
        a1 = 2 * np.pi * p / m
        a2 = 2 * np.pi * k * p / m
        c1, s1 = np.cos(a1), np.sin(a1)
        c2, s2 = np.cos(a2), np.sin(a2)
        G[p] = np.array([[c1, -s1, 0, 0],
                         [s1, c1, 0, 0],
                         [0, 0, c2, -s2],
                         [0, 0, s2, c2]], dtype=float)
    return G

def _expand(S, Gs):
    ns = S.shape[0]; ng = Gs.shape[0]
    X = np.einsum('gij,kj->kgi', Gs, S).reshape(ns * ng, 4)
    return _nm(X)

def _lp(X, deadline):
    X = _nm(X.copy())
    n, d = X.shape
    best = _mc(X)
    rad = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and rad > 1e-14:
        c = (X @ X.T)[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * rad, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        Aub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        Aeq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            r = linprog(obj, A_ub=Aub, b_ub=-c[sel], A_eq=Aeq, b_eq=np.zeros(n),
                        bounds=[(-rad, rad)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            break
        if r.status != 0:
            rad *= 0.5
            continue
        Y = _nm(X + r.x[:-1].reshape(n, d))
        v = _mc(Y)
        if v < best:
            X, best, rad = Y, v, min(rad * 1.5, 0.05)
        else:
            rad *= 0.5
    return X

def _adam_sym(S0, Gs, steps, b0, b1, lr0, lr1, deadline=None):
    S = S0.copy()
    ns = S.shape[0]
    GtR = np.stack([G.T for G in Gs], axis=0)
    m = np.zeros_like(S); v = np.zeros_like(S)
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(1, steps + 1):
        if deadline is not None and (t % 200 == 0) and time.time() > deadline:
            break
        f = t / steps
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        X = np.einsum('gij,kj->kgi', Gs, S).reshape(ns * Gs.shape[0], 4)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cm), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = (E @ X) / rs
        Gt = grad - np.sum(grad * X, axis=1, keepdims=True) * X
        Gt3 = Gt.reshape(ns, Gs.shape[0], 4)
        gs = np.einsum('gab,kgb->ka', GtR, Gt3)
        m = b1a * m + (1 - b1a) * gs
        v = b2a * v + (1 - b2a) * (gs * gs)
        mh = m / (1 - b1a ** t); vh = v / (1 - b2a ** t)
        S = _nm(S - lr * mh / (np.sqrt(vh) + eps))
    return S

def _adam_full(X0, steps, b0, b1, lr0, lr1, deadline=None):
    X = X0.copy()
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(1, steps + 1):
        if deadline is not None and (t % 200 == 0) and time.time() > deadline:
            break
        f = t / steps
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cm), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = (E @ X) / rs * 1.0
        Gt = grad - np.sum(grad * X, axis=1, keepdims=True) * X
        m = b1a * m + (1 - b1a) * Gt
        v = b2a * v + (1 - b2a) * (Gt * Gt)
        mh = m / (1 - b1a ** t); vh = v / (1 - b2a ** t)
        X = _nm(X - lr * mh / (np.sqrt(vh) + eps))
    return X

def _adam_mask(X0, mask, steps, b0, b1, lr0, lr1, deadline=None):
    X = X0.copy()
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return X
    m = np.zeros((len(idx), 4)); v = np.zeros((len(idx), 4))
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(1, steps + 1):
        if deadline is not None and (t % 200 == 0) and time.time() > deadline:
            break
        f = t / steps
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        cm = float(np.max(G))
        E = np.exp(np.clip(beta * (G - cm), -25.0, 0.0))
        np.fill_diagonal(E, 0.0)
        rs = np.sum(E, axis=1, keepdims=True) + 1e-12
        grad = (E @ X) / rs
        Gt = grad - np.sum(grad * X, axis=1, keepdims=True) * X
        g = Gt[idx]
        m = b1a * m + (1 - b1a) * g
        v = b2a * v + (1 - b2a) * (g * g)
        mh = m / (1 - b1a ** t); vh = v / (1 - b2a ** t)
        X[idx] = _nm(X[idx] - lr * mh / (np.sqrt(vh) + eps))
    return X

def _contact_mask(X, slack=1e-6):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    iu = np.triu_indices(X.shape[0], 1)
    return (G[iu] > m - slack), m

def _balanced_init(rng, ns):
    S = _nm(rng.normal(size=(ns, 4)))
    for i in range(ns):
        b1 = np.linalg.norm(S[i, :2]); b2 = np.linalg.norm(S[i, 2:])
        if min(b1, b2) < 0.25:
            for _ in range(10):
                c = _nm(rng.normal(size=(1, 4)))[0]
                if min(np.linalg.norm(c[:2]), np.linalg.norm(c[2:])) >= 0.25:
                    S[i] = c
                    break
    return S

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    SEARCH = t0 + 145.0
    SYM_END = t0 + 88.0
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
        xb = _nm(rng.normal(size=(168, 4))); bv = _mc(xb)
    else:
        vs = [_mc(c) for c in cands]
        xb = cands[int(np.argmin(vs))].copy(); bv = float(min(vs))
    xb_init = xb.copy()
    try:
        inc_mask, _ = _contact_mask(xb_init, slack=1e-6)
    except Exception:
        inc_mask = None
    G24 = _hurwitz()
    fams = [
        (7, G24, "h24"),
        (14, _cyclic(12, 1), "c12k1"),
        (14, _cyclic(12, 5), "c12k5"),
        (21, _cyclic(8, 1), "c8k1"),
        (21, _cyclic(8, 3), "c8k3"),
        (12, _cyclic(14, 1), "c14k1"),
        (8, _cyclic(21, 2), "c21k2"),
    ]
    # short screening budget per family
    sym_results = []  # (val, Xsym, ns, Gs)
    for (ns, Gs, _nm_) in fams:
        if time.time() > SYM_END:
            break
        ng = Gs.shape[0]
        if ns * ng != 168:
            continue
        nshort = 5 if ns in (7, 14, 21) else 3
        best_l = 10.0; best_X = None; best_S = None
        for ii in range(nshort):
            if time.time() > SYM_END:
                break
            mode = ii % 3
            if mode == 0:
                S0 = _balanced_init(rng, ns)
            elif mode == 1:
                perm = rng.permutation(168)[:ns * 4].reshape(ns, 4)
                # average groups of incumbent to make seeds
                S0 = np.zeros((ns, 4))
                for k in range(ns):
                    S0[k] = np.mean(xb_init[perm[k]], axis=0)
                S0 = _nm(S0 + 0.1 * rng.normal(size=(ns, 4)))
                # rebalance blocks for cyclic
                if ng != 24:
                    for k in range(ns):
                        if min(np.linalg.norm(S0[k, :2]), np.linalg.norm(S0[k, 2:])) < 0.2:
                            S0[k] = _balanced_init(rng, 1)[0]
            else:
                S0 = _balanced_init(rng, ns)
                # inject void direction
                C = _nm(rng.normal(size=(2000, 4)))
                D = C @ xb_init.T
                mx = D.max(axis=1)
                S0[0] = C[int(np.argmin(mx))]
                S0 = _nm(S0)
            try:
                So = _adam_sym(S0, Gs, steps=1200, b0=20., b1=500., lr0=0.02, lr1=0.002, deadline=SYM_END)
                Xsym = _expand(So, Gs)
            except Exception:
                continue
            vv = _mc(Xsym)
            if vv < best_l:
                best_l = vv; best_X = Xsym.copy(); best_S = So.copy()
        if best_X is None:
            continue
        # promote best of family to long run
        try:
            if time.time() < SYM_END:
                So2 = _adam_sym(best_S, Gs, steps=2200, b0=20., b1=800., lr0=0.02, lr1=0.001, deadline=SYM_END + 4.0)
                X2 = _expand(So2, Gs)
                v2 = _mc(X2)
                if v2 < best_l:
                    best_l = v2; best_X = X2
        except Exception:
            pass
        sym_results.append((best_l, best_X, ns, Gs))
        if best_l < bv:
            # symmetric itself beats incumbent (rare) -> still need novelty check
            try:
                cm, _m = _contact_mask(best_X, slack=1e-6)
                inter = float(np.sum(np.logical_and(cm, inc_mask))) / max(1, float(np.sum(np.logical_or(cm, inc_mask))))
            except Exception:
                inter = 0.0
            if inter < 0.92:
                bv = best_l; xb = best_X.copy()
    # sort symmetric candidates
    sym_results.sort(key=lambda t: t[0])
    # staged symmetry breaking on top 3
    for si, (sv, SX, ns, Gs) in enumerate(sym_results[:3]):
        if time.time() > SEARCH - 6.0:
            break
        ng = Gs.shape[0]
        # orbit-wise release: each seed = one orbit of size ng
        try:
            # order orbits by worst internal contact
            Xw = SX.copy()
            gw = Xw @ Xw.T
            np.fill_diagonal(gw, -2.0)
            orb_score = []
            for k in range(ns):
                idx = np.arange(k * ng, (k + 1) * ng)
                # max within orbit + to rest
                orb_score.append(float(np.max(gw[np.ix_(idx, np.arange(168))])))
            order = np.argsort(orb_score)[::-1]
            for k in order:
                if time.time() > SEARCH - 5.0:
                    break
                mask = np.zeros(168, dtype=bool)
                mask[k * ng:(k + 1) * ng] = True
                Xw = _adam_mask(Xw, mask, steps=350, b0=30., b1=450., lr0=0.012, lr1=0.0012, deadline=SEARCH - 4.0)
            # evaluate orbit-released
            vw = _mc(Xw)
            # full anneal from orbit-released
            for nz, st in ((0.005, 700), (0.02, 700), (0.05, 600)):
                if time.time() > SEARCH - 4.0:
                    break
                try:
                    X0 = _nm(Xw + nz * rng.normal(size=(168, 4)))
                    Xo = _adam_full(X0, steps=st, b0=25., b1=600., lr0=0.018, lr1=0.001, deadline=SEARCH - 2.0)
                except Exception:
                    continue
                v = _mc(Xo)
                # novelty gate at slack 1e-6 vs incumbent
                try:
                    cm, _ = _contact_mask(Xo, slack=1e-6)
                    inter = float(np.sum(np.logical_and(cm, inc_mask))) / max(1.0, float(np.sum(np.logical_or(cm, inc_mask))))
                except Exception:
                    inter = 0.0
                improved = False
                if v < bv and inter < 0.92:
                    bv = v; xb = Xo.copy(); improved = True
                elif v < bv - 2e-4:
                    # big win even if similar topology -> accept
                    bv = v; xb = Xo.copy(); improved = True
                if (improved or (v < bv + 0.002 and si == 0)) and time.time() < SEARCH:
                    try:
                        bud = 1.2 if improved else 0.5
                        Xp = _lp(Xo, min(time.time() + bud, HARD - 5.0))
                        vp = _mc(Xp)
                        try:
                            c2, _ = _contact_mask(Xp, slack=1e-6)
                            inter2 = float(np.sum(np.logical_and(c2, inc_mask))) / max(1.0, float(np.sum(np.logical_or(c2, inc_mask))))
                        except Exception:
                            inter2 = 0.0
                        if vp < bv and (inter2 < 0.94 or vp < bv - 1e-7):
                            bv = vp; xb = Xp.copy()
                    except Exception:
                        pass
            # also direct break from symmetric without orbit stage
            for nz in (0.01, 0.03):
                if time.time() > SEARCH - 3.0:
                    break
                try:
                    X0 = _nm(SX + 0) if False else _nm(SX + nz * rng.normal(size=(168, 4)))
                    Xo = _adam_full(X0, steps=600, b0=25., b1=550., lr0=0.016, lr1=0.001, deadline=SEARCH - 1.0)
                    v = _mc(Xo)
                    try:
                        cm, _ = _contact_mask(Xo, slack=1e-6)
                        inter = float(np.sum(np.logical_and(cm, inc_mask))) / max(1.0, float(np.sum(np.logical_or(cm, inc_mask))))
                    except Exception:
                        inter = 0.0
                    if v < bv and inter < 0.92:
                        bv = v; xb = Xo.copy()
                        Xp = _lp(Xo, min(time.time() + 1.0, HARD - 4.0))
                        vp = _mc(Xp)
                        if vp < bv:
                            bv = vp; xb = Xp.copy()
                except Exception:
                    pass
        except Exception:
            continue
    # opportunistic plain kicks from current best (keeps fallback)
    for _ in range(4):
        if time.time() > SEARCH:
            break
        try:
            X0 = _nm(xb + 0.03 * rng.normal(size=(168, 4)))
            Xo = _adam_full(X0, steps=500, b0=28., b1=450., lr0=0.015, lr1=0.001, deadline=SEARCH)
            v = _mc(Xo)
            if v < bv:
                try:
                    c2, _ = _contact_mask(Xo, slack=1e-6)
                    inter = float(np.sum(np.logical_and(c2, inc_mask))) / max(1.0, float(np.sum(np.logical_or(c2, inc_mask))))
                except Exception:
                    inter = 0.0
                if inter < 0.96:
                    bv = v; xb = Xo.copy()
        except Exception:
            pass
    if time.time() < HARD - 2.0:
        try:
            xb = _lp(xb, HARD - 1.0)
        except Exception:
            pass
    return _nm(xb)
