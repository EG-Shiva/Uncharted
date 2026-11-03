import time
import itertools
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

# ---------- fixed shells ----------
def _fixed_axes():
    X = np.zeros((12, 6))
    k = 0
    for i in range(6):
        for s in (1.0, -1.0):
            X[k, i] = s
            k += 1
    return X

def _fixed_w2():
    # (1,1,0,0,0,0)/sqrt2 signed orbit size 60
    pts = []
    for i in range(6):
        for j in range(i+1, 6):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    v = np.zeros(6)
                    v[i] = si / np.sqrt(2.0)
                    v[j] = sj / np.sqrt(2.0)
                    pts.append(v)
    return np.array(pts)

def _fixed_w3():
    pts = []
    for combo in itertools.combinations(range(6), 3):
        for bits in range(8):
            v = np.zeros(6)
            for k, idx in enumerate(combo):
                s = 1.0 if (bits >> k) & 1 else -1.0
                v[idx] = s / np.sqrt(3.0)
            pts.append(v)
    return np.array(pts)  # 160

def _fixed_hcube():
    pts = []
    for bits in range(64):
        v = np.ones(6)
        for k in range(6):
            if (bits >> k) & 1:
                v[k] = -1.0
        v = v / np.sqrt(6.0)
        pts.append(v)
    return np.array(pts)  # 64

# ---------- parametric position templates (perm-only) ----------
def _template_ABB():
    # (a,b,b,0,0,0) : pos_a (6) x choose 2 for b (10) = 60
    pa, pb1, pb2 = [], [], []
    for ia in range(6):
        rem = [x for x in range(6) if x != ia]
        for j1i in range(len(rem)):
            for j2i in range(j1i+1, len(rem)):
                pa.append(ia); pb1.append(rem[j1i]); pb2.append(rem[j2i])
    return (np.array(pa), np.array(pb1), np.array(pb2))

def _template_ABC():
    pa, pb, pc = [], [], []
    for ia in range(6):
        for ib in range(6):
            if ib == ia: continue
            for ic in range(6):
                if ic == ia or ic == ib: continue
                pa.append(ia); pb.append(ib); pc.append(ic)
    return (np.array(pa), np.array(pb), np.array(pc))  # 120

def _template_AABBC():
    # (a,a,b,b,c,0): 6!/(2!2!) = 180
    # generate via brute permutations of multiset labels
    seen = set()
    pa1, pa2, pb1, pb2, pc = [], [], [], [], []
    # enumerate: choose 2 for a, 2 for b from rest, 1 for c from rest
    for ca in itertools.combinations(range(6), 2):
        rem1 = [x for x in range(6) if x not in ca]
        for cb in itertools.combinations(rem1, 2):
            rem2 = [x for x in rem1 if x not in cb]
            for cc in rem2:
                # c position = cc, zero = remaining one (implicit)
                pa1.append(ca[0]); pa2.append(ca[1])
                pb1.append(cb[0]); pb2.append(cb[1])
                pc.append(cc)
    # this gives 15*6*5=450? overcount because zero vs c distinguished? rem2 has 2 entries, cc choice 2 => 15*6*2=180. good.
    return (np.array(pa1), np.array(pa2), np.array(pb1), np.array(pb2), np.array(pc))

def _descent_full(X, beta, steps, lr, mu):
    n, d = X.shape
    v = np.zeros_like(X)
    for it in range(steps):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        mean = (W @ X) / rs
        dot = np.sum(mean * X, axis=1, keepdims=True)
        tang = mean - dot * X
        v = mu * v + tang
        cur = lr * (1.0 - 0.5 * it / steps)
        X = X - cur * v
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X

def _lp_polish(X, deadline):
    X = _norm_rows(X)
    n, d = X.shape
    best = _max_cos(X)
    radius = 2e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline - 1.0 and radius > 1e-14:
        Gall = X @ X.T
        c_all = Gall[iu]
        m = float(np.max(c_all))
        thresh = m - max(20.0 * radius, 5e-7)
        sel = c_all > thresh
        cnt = int(np.sum(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 3800:
            K = 3800
            top = np.argpartition(-c_all, K)[:K]
            msel = np.zeros_like(sel)
            msel[top] = True
            sel = msel
        I = iu[0][sel]; J = iu[1][sel]
        c_sel = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None]*d + np.arange(d),
                               J[:, None]*d + np.arange(d),
                               np.full((p, 1), n*d)], axis=1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            bounds = [(-radius, radius)]*(n*d) + [(None, None)]
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = _max_cos(y)
        if v < best:
            X, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    HARD = t0 + 175.0
    rng = np.random.default_rng(0)
    n, d = 250, 6
    beta_sym = 2000.0

    # load incumbent
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (250, 6):
                a = _norm_rows(np.nan_to_num(a))
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        z = rng.standard_normal((n, d)); z = _norm_rows(z)
        cands.append(z)
    scores = [_max_cos(a) for a in cands]
    bi = int(np.argmin(scores))
    incumbent = cands[bi].copy()
    inc_score = float(scores[bi])

    # templates / fixed
    T_ABB = _template_ABB()  # 60
    T_ABC = _template_ABC()  # 120
    T_AABBC = _template_AABBC()  # 180
    F12 = _fixed_axes()
    F60 = _fixed_w2()
    F160 = _fixed_w3()
    F64 = _fixed_hcube()

    # define combos: list of dicts
    combos = []
    combos.append({"name": "ABC+ABC+10", "blocks": [("ABC", None), ("ABC", None)], "free": 10})
    combos.append({"name": "AABBC+F60+10", "blocks": [("AABBC", None), ("FIX", F60)], "free": 10})
    combos.append({"name": "AABBC+ABB+10", "blocks": [("AABBC", None), ("ABB", None)], "free": 10})
    combos.append({"name": "ABC+F64+F60+6", "blocks": [("ABC", None), ("FIX", F64), ("FIX", F60)], "free": 6})
    combos.append({"name": "ABC+F64+26", "blocks": [("ABC", None), ("FIX", F64)], "free": 66 - 0})  # 120+64=184 rem66
    combos[-1]["free"] = 66
    combos.append({"name": "ABC+F60+70", "blocks": [("ABC", None), ("FIX", F60)], "free": 70})
    combos.append({"name": "ABC+ABB+70", "blocks": [("ABC", None), ("ABB", None)], "free": 70})
    combos.append({"name": "AABBC+F64+6", "blocks": [("AABBC", None), ("FIX", F64)], "free": 6})
    combos.append({"name": "ABB+F64+F60+6dup", "blocks": [("ABB", None), ("FIX", F64), ("FIX", F60)], "free": 64})  # 60+64+60=184 rem66
    combos[-1]["free"] = 66
    combos.append({"name": "F160+F64+26", "blocks": [("FIX", F160), ("FIX", F64)], "free": 26})
    combos.append({"name": "F160+F60+30", "blocks": [("FIX", F160), ("FIX", F60)], "free": 30})
    combos.append({"name": "F160+F60+F12+18", "blocks": [("FIX", F160), ("FIX", F60), ("FIX", F12)], "free": 18})

    best_sym_X = None
    best_sym_val = np.inf

    sym_deadline = t0 + 82.0

    for ci, combo in enumerate(combos):
        if time.time() > sym_deadline:
            break
        blocks = combo["blocks"]
        m_free = int(combo["free"])
        # count params
        nparam = 0
        for typ, _ in blocks:
            if typ == "ABB": nparam += 1
            elif typ == "ABC": nparam += 2
            elif typ == "AABBC": nparam += 2
        if m_free < 0 or m_free > 70:
            continue
        # total check
        tot = m_free
        for typ, arr in blocks:
            if typ == "FIX": tot += arr.shape[0]
            elif typ == "ABB": tot += 60
            elif typ == "ABC": tot += 120
            elif typ == "AABBC": tot += len(T_AABBC[0])
        if tot != 250:
            continue
        # initial vars
        # thetas: ABB t~1.0, ABC (0.9,0.8), AABBC (0.9,0.8)
        th0 = []
        bounds = []
        for typ, _ in blocks:
            if typ == "ABB":
                th0.append(0.9); bounds.append((0.05, 1.52))
            elif typ == "ABC":
                th0 += [0.9, 0.785]; bounds += [(0.05, 1.52), (0.05, 1.52)]
            elif typ == "AABBC":
                th0 += [0.9, 0.785]; bounds += [(0.05, 1.52), (0.05, 1.52)]
        th0 = np.array(th0, dtype=float)
        Yf0 = rng.standard_normal((m_free, d)) if m_free > 0 else np.zeros((0, d))
        x0 = np.concatenate([th0, Yf0.ravel()]) if len(th0) + Yf0.size > 0 else np.zeros(1)
        # need bounds for free vars (None)
        bnds = bounds + [(None, None)] * (Yf0.size)

        # closure to build X and grad
        # precompute block order sizes for assembly
        def build_X(th, Yf):
            parts = []
            ti = 0
            for typ, arr in blocks:
                if typ == "FIX":
                    parts.append(arr)
                elif typ == "ABB":
                    t = float(th[ti]); ti += 1
                    a = np.cos(t); b = np.sin(t) / np.sqrt(2.0)
                    m = 60
                    Xb = np.zeros((m, 6))
                    pa, pb1, pb2 = T_ABB
                    Xb[np.arange(m), pa] = a
                    Xb[np.arange(m), pb1] = b
                    Xb[np.arange(m), pb2] = b
                    parts.append(Xb)
                elif typ == "ABC":
                    t1 = float(th[ti]); t2 = float(th[ti+1]); ti += 2
                    a = np.cos(t1); b = np.sin(t1)*np.cos(t2); c = np.sin(t1)*np.sin(t2)
                    nn = np.sqrt(a*a+b*b+c*c)
                    a, b, c = a/nn, b/nn, c/nn
                    m = 120
                    Xb = np.zeros((m, 6))
                    pa, pb_, pc_ = T_ABC
                    Xb[np.arange(m), pa] = a
                    Xb[np.arange(m), pb_] = b
                    Xb[np.arange(m), pc_] = c
                    parts.append(Xb)
                elif typ == "AABBC":
                    t1 = float(th[ti]); t2 = float(th[ti+1]); ti += 2
                    c_ = np.cos(t1); s = np.sin(t1)
                    a = s*np.cos(t2)/np.sqrt(2.0); b = s*np.sin(t2)/np.sqrt(2.0)
                    m = len(T_AABBC[0])
                    Xb = np.zeros((m, 6))
                    pa1, pa2, pb1, pb2, pc_ = T_AABBC
                    Xb[np.arange(m), pa1] = a; Xb[np.arange(m), pa2] = a
                    Xb[np.arange(m), pb1] = b; Xb[np.arange(m), pb2] = b
                    Xb[np.arange(m), pc_] = c_
                    # already unit? 2a^2+2b^2+c^2 = s^2(cos^2+sin^2)+c^2=1 yes
                    parts.append(Xb)
            if Yf.shape[0] > 0:
                Xn = Yf / np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
                parts.append(Xn)
            return np.vstack(parts) if parts else np.zeros((0, 6))

        def fg(flat):
            nth = len(th0)
            th = flat[:nth] if nth > 0 else np.zeros(0)
            yf = flat[nth:].reshape(m_free, d) if m_free > 0 else np.zeros((0, d))
            # build blocks with derivative info
            # first build X
            X = build_X(th, yf)
            # smooth max
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            M = float(np.max(G))
            D = beta_sym * (G - M)
            np.clip(D, -30.0, 0.0, out=D)
            E = np.exp(D, out=np.empty_like(D))
            np.fill_diagonal(E, 0.0)
            S = float(np.sum(E))
            if S <= 0 or not np.isfinite(S):
                S = 1e-300
            loss = M + np.log(S) / beta_sym
            W = E / S
            GX = 2.0 * (W @ X)
            # grads
            gth = np.zeros_like(th)
            # walk blocks to compute param grads; need offsets in X
            off = 0
            ti = 0
            for typ, arr in blocks:
                if typ == "FIX":
                    off += arr.shape[0]
                elif typ == "ABB":
                    t = float(th[ti])
                    da = -np.sin(t); db = np.cos(t)/np.sqrt(2.0)
                    m = 60
                    Gb = GX[off:off+m]
                    pa, pb1, pb2 = T_ABB
                    ar = np.arange(m)
                    g = da*np.sum(Gb[ar, pa]) + db*(np.sum(Gb[ar, pb1])+np.sum(Gb[ar, pb2]))
                    gth[ti] = g
                    ti += 1; off += m
                elif typ == "ABC":
                    t1 = float(th[ti]); t2 = float(th[ti+1])
                    # recompute a,b,c and derivatives (with renorm correction ignored: nn=1 since spherical)
                    # a=cos t1 etc exactly norm 1
                    da1 = -np.sin(t1); db1 = np.cos(t1)*np.cos(t2); dc1 = np.cos(t1)*np.sin(t2)
                    db2 = -np.sin(t1)*np.sin(t2); dc2 = np.sin(t1)*np.cos(t2)
                    m = 120
                    Gb = GX[off:off+m]
                    pa, pb_, pc_ = T_ABC
                    ar = np.arange(m)
                    g1 = da1*np.sum(Gb[ar, pa]) + db1*np.sum(Gb[ar, pb_]) + dc1*np.sum(Gb[ar, pc_])
                    g2 = db2*np.sum(Gb[ar, pb_]) + dc2*np.sum(Gb[ar, pc_])
                    gth[ti] = g1; gth[ti+1] = g2
                    ti += 2; off += m
                elif typ == "AABBC":
                    t1 = float(th[ti]); t2 = float(th[ti+1])
                    s = np.sin(t1); c_ = np.cos(t1)
                    # a = s cos2/sqrt2 etc
                    da1 = np.cos(t1)*np.cos(t2)/np.sqrt(2.0)
                    db1 = np.cos(t1)*np.sin(t2)/np.sqrt(2.0)
                    dc1 = -np.sin(t1)
                    da2 = -s*np.sin(t2)/np.sqrt(2.0)
                    db2 = s*np.cos(t2)/np.sqrt(2.0)
                    m = len(T_AABBC[0])
                    Gb = GX[off:off+m]
                    pa1, pa2, pb1, pb2, pc_ = T_AABBC
                    ar = np.arange(m)
                    sumA = np.sum(Gb[ar, pa1])+np.sum(Gb[ar, pa2])
                    sumB = np.sum(Gb[ar, pb1])+np.sum(Gb[ar, pb2])
                    sumC = np.sum(Gb[ar, pc_])
                    gth[ti] = da1*sumA + db1*sumB + dc1*sumC
                    gth[ti+1] = da2*sumA + db2*sumB
                    ti += 2; off += m
            if m_free > 0:
                Gf = GX[off:off+m_free]
                Xf = X[off:off+m_free]
                nrm = np.maximum(np.linalg.norm(yf, axis=1, keepdims=True), 1e-12)
                dot = np.sum(Gf*Xf, axis=1, keepdims=True)
                Gy = (Gf - dot*Xf) / nrm
                gflat = np.concatenate([gth, Gy.ravel()])
            else:
                gflat = gth
            return loss, gflat

        try:
            res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bnds,
                           options={"maxiter": 120, "ftol": 1e-20, "gtol": 1e-12, "maxcor": 10})
            flat = res.x
        except Exception:
            continue
        nth = len(th0)
        thb = flat[:nth] if nth > 0 else np.zeros(0)
        yfb = flat[nth:].reshape(m_free, d) if m_free > 0 else np.zeros((0, d))
        try:
            Xb = build_X(thb, yfb)
        except Exception:
            continue
        Xb = _norm_rows(Xb)
        # dedup check: if duplicates (max~1) skip? still keep but likely bad
        val = _max_cos(Xb)
        if val < best_sym_val:
            best_sym_val = val
            best_sym_X = Xb.copy()
        if time.time() > sym_deadline:
            break

    # choose start for full polish
    if best_sym_X is not None and best_sym_val < inc_score:
        X = best_sym_X.copy()
        cur_best = best_sym_val
    else:
        X = incumbent.copy()
        cur_best = inc_score

    # break symmetry: full polish with remaining time
    # stage A: short smooth descents
    try:
        t_rem = HARD - time.time()
        if t_rem > 30:
            # two descents with different betas
            X = _descent_full(X, 90.0, 350, 0.008, 0.85)
            if time.time() < HARD - 40:
                X = _descent_full(X, 220.0, 250, 0.005, 0.88)
    except Exception:
        pass
    # small random basin kicks keeping best
    try:
        if time.time() < HARD - 35:
            v = _max_cos(X)
            if v < cur_best:
                cur_best = v
            else:
                # keep incumbent if descent worsened slightly? keep better
                if v > cur_best + 5e-4:
                    X = incumbent.copy()
                    cur_best = inc_score
                else:
                    cur_best = min(cur_best, v)
    except Exception:
        pass
    # stage B: LP polish
    try:
        if HARD - time.time() > 5:
            X = _lp_polish(X, HARD - 1.0)
    except Exception:
        pass

    # final safety: return best among X and incumbent
    try:
        vx = _max_cos(_norm_rows(X))
    except Exception:
        vx = np.inf
    if not np.isfinite(vx) or vx > inc_score + 1e-3:
        # if our polish failed badly, return incumbent (still polished?)
        # try quick LP on incumbent with tiny remaining time
        try:
            if HARD - time.time() > 4:
                incumbent = _lp_polish(incumbent, HARD - 0.5)
        except Exception:
            pass
        X = incumbent
    X = _norm_rows(np.nan_to_num(X))
    return X
