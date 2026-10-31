import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return X / n

def _smooth_full(X, iters, T0, T1, lr0, lr1):
    n, d = X.shape
    for k in range(iters):
        f = k / max(iters - 1, 1)
        T = T0 * (T1 / T0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp((G - m) / T)
        np.fill_diagonal(W, 0.0)
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        gt = grad - dot * X
        rs = np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
        step = gt / rs
        sn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step / np.maximum(sn, 1.0)
        X = X - lr * step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(X, seconds, cap=1500):
    X = _norm(X)
    n, d = X.shape
    iu0, iu1 = np.triu_indices(n, 1)
    best = _max_cos(X)
    radius = 2e-3
    start = time.time()
    cd = np.arange(d)
    while time.time() - start < seconds and radius > 1e-13:
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        thr = m - max(20.0 * radius, 1e-9)
        sel = np.where(c_all > thr)[0]
        if sel.size == 0:
            break
        if sel.size > cap:
            v = c_all[sel]
            kth = np.sort(v)[-cap]
            sel = np.where(c_all >= kth)[0]
            if sel.size > cap + 300:
                sel = sel[::2][:cap]
        I = iu0[sel]; J = iu1[sel]
        c = c_all[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + cd, J[:, None] * d + cd,
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d),
                                       np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        Y = _norm(X + res.x[:-1].reshape(n, d))
        v = _max_cos(Y)
        if v < best:
            X, best = Y, v
            radius = min(radius * 1.6, 0.03)
        else:
            radius *= 0.5
    return X

# ---- torus ansatz helpers ----
def _make_struct(nks):
    M = len(nks)
    lk = []
    adiv = []
    bth = []
    bph = []
    off = 0
    for k, nk in enumerate(nks):
        r = int(max(2, round(float(np.sqrt(nk)))))
        # adjust r to keep c reasonable
        c = int(max(2, (nk + r - 1) // r))
        # recompute r as ceil(nk/c) for spacing consistency? keep r
        for idx in range(nk):
            a = idx // c
            b = idx % c
            lk.append(k)
            adiv.append(a / float(r))
            bth.append(2.0 * np.pi * a / float(r))
            bph.append(2.0 * np.pi * b / float(c))
    return (np.array(lk, dtype=int), np.array(adiv), np.array(bth), np.array(bph))

def _torus_eval(params, M, lk, adiv, bth, bph, T, Npair):
    raw = params[0:M]
    th0 = params[M:2*M]
    ph0 = params[2*M:3*M]
    tw = params[3*M:4*M]
    # sigmoid -> chi in (0, pi/2)
    # stable sigmoid
    sig = np.empty(M)
    for k in range(M):
        r = raw[k]
        if r >= 0:
            e = np.exp(-r)
            sig[k] = 1.0 / (1.0 + e)
        else:
            e = np.exp(r)
            sig[k] = e / (1.0 + e)
    chi = 0.5 * np.pi * sig
    s = np.sin(chi[lk])
    c = np.cos(chi[lk])
    th = bth + th0[lk]
    ph = bph + ph0[lk] + tw[lk] * adiv
    ct = np.cos(th); st = np.sin(th); cp = np.cos(ph); sp = np.sin(ph)
    X = np.stack([s*ct, s*st, c*cp, c*sp], axis=1)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    E = np.exp((G - m) / T)
    np.fill_diagonal(E, 0.0)
    S = float(np.sum(E))
    if S <= 0:
        S = 1e-300
    f = m + T * (np.log(S) - np.log(Npair))
    W = E / S
    GX = W @ X  # n x 4
    # per-point derivatives
    # dX/dchi
    dchi_x1 = c * ct; dchi_x2 = c * st; dchi_x3 = -s * cp; dchi_x4 = -s * sp
    g_chi_pt = GX[:,0]*dchi_x1 + GX[:,1]*dchi_x2 + GX[:,2]*dchi_x3 + GX[:,3]*dchi_x4
    g_chi_pt *= 2.0
    # dX/dtheta0
    dth_x1 = -s * st; dth_x2 = s * ct
    g_th_pt = 2.0 * (GX[:,0]*dth_x1 + GX[:,1]*dth_x2)
    # dX/dphi0
    dph_x3 = -c * sp; dph_x4 = c * cp
    g_ph_pt = 2.0 * (GX[:,2]*dph_x3 + GX[:,3]*dph_x4)
    g_tw_pt = g_ph_pt * adiv
    g = np.zeros_like(params)
    # aggregate per layer
    for k in range(M):
        mask = (lk == k)
        d = 0.5*np.pi*sig[k]*(1.0-sig[k])
        g[k] = float(np.sum(g_chi_pt[mask])) * d
        g[M+k] = float(np.sum(g_th_pt[mask]))
        g[2*M+k] = float(np.sum(g_ph_pt[mask]))
        g[3*M+k] = float(np.sum(g_tw_pt[mask]))
    return f, g, X, m

def solve():
    t0 = time.time()
    HARD = 168.0
    rng = np.random.default_rng(163163)
    n, d = 163, 4
    # incumbent fallback
    inc = None; inc_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            cc = np.asarray(np.load(fn), dtype=float)
            if cc.shape == (n, d):
                cc = np.nan_to_num(cc, nan=0.0, posinf=0.0, neginf=0.0)
                cc = _norm(cc)
                v = _max_cos(cc)
                if v < inc_val:
                    inc_val = v; inc = cc.copy()
        except Exception:
            pass
    if inc is None:
        inc = _norm(rng.normal(size=(n, d)))
        inc_val = _max_cos(inc)
    Npair = float(n*(n-1))  # ordered off-diag count
    cands = []  # (val, X_ansatz)
    t_screen_end = t0 + 82.0
    trial = 0
    # bounds for L-BFGS-B
    while time.time() < t_screen_end and trial < 90:
        trial += 1
        M = int(rng.integers(5, 12))  # 5..11
        w = rng.random(M) + 0.3
        w /= w.sum()
        nks = np.floor(w * n).astype(int)
        # fix sum
        diff = n - int(nks.sum())
        # distribute diff
        order = rng.permutation(M)
        ii = 0
        while diff != 0:
            k = int(order[ii % M]); ii += 1
            if diff > 0:
                nks[k] += 1; diff -= 1
            else:
                if nks[k] > 6:
                    nks[k] -= 1; diff += 1
                if ii > 500:
                    break
        # enforce min 5
        for k in range(M):
            if nks[k] < 5:
                need = 5 - int(nks[k])
                # steal from largest
                j = int(np.argmax(nks))
                take = min(need, int(nks[j]) - 5)
                if take > 0:
                    nks[j] -= take; nks[k] += take
        if int(nks.sum()) != n or np.any(nks < 4):
            continue
        lk, adiv, bth, bph = _make_struct(nks)
        # init params
        raw0 = rng.normal(scale=0.7, size=M)
        th00 = rng.uniform(-np.pi, np.pi, size=M)
        ph00 = rng.uniform(-np.pi, np.pi, size=M)
        tw00 = rng.uniform(-0.6, 0.6, size=M)
        p0 = np.concatenate([raw0, th00, ph00, tw00])
        bounds = [(-4.0, 4.0)]*M + [(None, None)]*(2*M) + [(-3.0, 3.0)]*M
        cur = p0.copy()
        try:
            for stage, (T, mx) in enumerate([(1e-2, 60), (3e-3, 60), (8e-4, 40)]):
                if time.time() > t_screen_end:
                    break
                def fun(p, T=T):
                    f, g, _, _ = _torus_eval(p, M, lk, adiv, bth, bph, T, Npair)
                    return f, g
                res = minimize(fun, cur, jac=True, method="L-BFGS-B",
                               bounds=bounds, options={"maxiter": mx, "ftol": 1e-12, "gtol": 1e-8})
                cur = res.x
                if time.time() > t_screen_end - 1.0:
                    break
            _, _, Xb, truev = _torus_eval(cur, M, lk, adiv, bth, bph, 1e-4, Npair)
            # true max
            tv = _max_cos(Xb)
            cands.append((tv, Xb.copy(), int(M)))
            # keep sorted small
            cands.sort(key=lambda z: z[0])
            cands = cands[:4]
        except Exception:
            continue
    # release best 2-3 to full descent
    best_overall = inc.copy()
    best_val = inc_val
    # also consider raw ansatz cands without descent
    full_cands = []
    if cands:
        for tv, Xb, Mm in cands[:3]:
            full_cands.append(Xb)
    else:
        full_cands.append(inc.copy())
    # full smooth descent on each (limited time)
    t_full_end = t0 + HARD - 24.0
    refined = []
    for Xb in full_cands:
        if time.time() > t_full_end:
            break
        try:
            X = Xb.copy()
            # annealed descent: two blocks
            remain = t_full_end - time.time()
            # scale iters by remain
            it1 = 700 if remain > 30 else 300
            X = _smooth_full(X, iters=it1, T0=1.5e-2, T1=2e-3, lr0=0.15, lr1=0.03)
            if time.time() > t_full_end:
                refined.append(X); continue
            X = _smooth_full(X, iters=500, T0=2e-3, T1=3e-4, lr0=0.03, lr1=0.008)
            refined.append(X)
            v = _max_cos(X)
            if v < best_val:
                best_val = v; best_overall = X.copy()
        except Exception:
            continue
    # pick best refined for LP polish, also try incumbent polish path if torus far
    # choose top 1 refined + incumbent as candidates for final polish
    finals = []
    if refined:
        vals = [( _max_cos(X), X) for X in refined]
        vals.sort(key=lambda z: z[0])
        finals.append(vals[0][1])
    finals.append(inc.copy())
    # dedupe: polish the current best first
    # decide which to polish with remaining time
    rem = HARD - (time.time() - t0) - 1.5
    if rem < 2.0:
        return _norm(best_overall)
    # polish best_overall primarily
    try:
        sec_each = rem / max(1, len(finals))
        sec_each = min(sec_each, 14.0)
        pf_best = best_overall; pf_val = best_val
        for Xf in finals:
            if time.time() - t0 > HARD - 2.0:
                break
            s = max(2.0, sec_each)
            # cap seconds to remaining
            s = min(s, HARD - (time.time()-t0) - 1.0)
            if s < 1.0:
                break
            Y = _lp_polish(Xf, seconds=s)
            v = _max_cos(Y)
            if v < pf_val:
                pf_val = v; pf_best = Y.copy()
        best_overall = pf_best; best_val = pf_val
    except Exception:
        pass
    if best_val > inc_val:
        return _norm(inc)
    return _norm(best_overall)
