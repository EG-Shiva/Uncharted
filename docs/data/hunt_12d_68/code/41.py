import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

N = 68
D = 12
NORB = 4
KORB = 17
NPL = 6

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rot_tables(mults):
    mults = np.asarray(mults, dtype=float)
    C = np.zeros((NPL, KORB))
    S = np.zeros((NPL, KORB))
    for j in range(NPL):
        for k in range(KORB):
            a = 2.0 * np.pi * float(mults[j]) * float(k) / 17.0
            C[j, k] = np.cos(a)
            S[j, k] = np.sin(a)
    return C, S

def _expand(Sseeds, C, S_):
    X = np.empty((N, D))
    for a in range(NORB):
        s = Sseeds[a]
        for k in range(KORB):
            r = a * KORB + k
            for j in range(NPL):
                px = s[2 * j]; py = s[2 * j + 1]
                c = C[j, k]; sn = S_[j, k]
                X[r, 2 * j] = c * px - sn * py
                X[r, 2 * j + 1] = sn * px + c * py
    return X

def _apply_RT(vec, k, C, S_):
    out = np.empty(D)
    for j in range(NPL):
        gx = vec[2 * j]; gy = vec[2 * j + 1]
        c = C[j, k]; sn = S_[j, k]
        out[2 * j] = c * gx + sn * gy
        out[2 * j + 1] = -sn * gx + c * gy
    return out

def _make_sym_fg(C, S_, beta, IU):
    def fg(yf):
        Y = yf.reshape(NORB, D)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        Ss = Y / nr
        X = _expand(Ss, C, S_)
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        Sm = float(e.sum()) + 1e-300
        F = m + np.log(Sm) / beta
        w = e / Sm
        P = np.zeros((N, N))
        P[IU] = w
        P = P + P.T
        GX = P @ X  # 68x12
        Gseed = np.zeros((NORB, D))
        for a in range(NORB):
            acc = np.zeros(D)
            base = a * KORB
            for k in range(KORB):
                gx = GX[base + k]
                # R_k^T gx
                for j in range(NPL):
                    c = C[j, k]; sn = S_[j, k]
                    acc[2 * j] += c * gx[2 * j] + sn * gx[2 * j + 1]
                    acc[2 * j + 1] += -sn * gx[2 * j] + c * gx[2 * j + 1]
            Gseed[a] = acc
        dot = np.sum(Gseed * Ss, axis=1, keepdims=True)
        GY = (Gseed - dot * Ss) / nr
        return F, GY.ravel()
    return fg

def _run_sym(S0, C, S_, betas_iters):
    y0 = (S0 / np.linalg.norm(S0, axis=1, keepdims=True)).ravel().copy()
    IU = np.triu_indices(N, 1)
    y = y0
    for beta, it in betas_iters:
        fg = _make_sym_fg(C, S_, float(beta), IU)
        try:
            r = minimize(fg, y, jac=True, method='L-BFGS-B',
                         options={'maxiter': int(it), 'ftol': 1e-16, 'gtol': 1e-8, 'maxfun': int(it) * 3})
            y = r.x
        except Exception:
            break
    Y = y.reshape(NORB, D)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _sym_lp(Ss, C, S_, deadline, r0=4e-3):
    Ss = Ss / np.linalg.norm(Ss, axis=1, keepdims=True)
    X = _expand(Ss, C, S_)
    best = _max_cos(X)
    radius = r0
    IU = np.triu_indices(N, 1)
    nv = NORB * D + 1
    while time.time() < deadline - 0.3 and radius > 1e-14:
        cfull = (X @ X.T)[IU]
        m = float(cfull.max())
        sel = cfull > m - max(20.0 * radius, 2e-9)
        I = IU[0][sel]; J = IU[1][sel]
        csel = cfull[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > 2500:
            od = np.argsort(-csel)[:2500]
            I, J, csel = I[od], J[od], csel[od]
            p = 2500
        # map global idx -> (a,k)
        Aa = I // KORB; Ka = I % KORB
        Ab = J // KORB; Kb = J % KORB
        rows_l = []
        cols_l = []
        vals_l = []
        for t in range(p):
            a = int(Aa[t]); ka = int(Ka[t]); b = int(Ab[t]); kb = int(Kb[t])
            Xj = X[J[t]]; Xi = X[I[t]]
            u = _apply_RT(Xj, ka, C, S_)
            v = _apply_RT(Xi, kb, C, S_)
            if a == b:
                wv = u + v
                for dd in range(D):
                    rows_l.append(t); cols_l.append(a * D + dd); vals_l.append(wv[dd])
            else:
                for dd in range(D):
                    rows_l.append(t); cols_l.append(a * D + dd); vals_l.append(u[dd])
                    rows_l.append(t); cols_l.append(b * D + dd); vals_l.append(v[dd])
            rows_l.append(t); cols_l.append(NORB * D); vals_l.append(-1.0)
        try:
            A_ub = coo_matrix((np.array(vals_l), (np.array(rows_l), np.array(cols_l))), shape=(p, nv)).tocsr()
            # equality: Ss[a].ds_a = 0
            er = np.repeat(np.arange(NORB), D)
            ec = np.arange(NORB * D)
            ev = Ss.ravel()
            A_eq = coo_matrix((ev, (er, ec)), shape=(NORB, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(NORB),
                          bounds=[(-radius, radius)] * (NORB * D) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        ds = res.x[:-1].reshape(NORB, D)
        Y2 = Ss + ds
        Y2 /= np.linalg.norm(Y2, axis=1, keepdims=True)
        X2 = _expand(Y2, C, S_)
        v = _max_cos(X2)
        if v < best - 1e-15:
            Ss, X, best = Y2, X2, v
            radius = min(radius * 1.5, 0.02)
        else:
            improved = False
            for shrink in (0.3, 0.1):
                Ys = Ss + ds * shrink
                Ys /= np.linalg.norm(Ys, axis=1, keepdims=True)
                Xs = _expand(Ys, C, S_)
                vv = _max_cos(Xs)
                if vv < best:
                    Ss, X, best = Ys, Xs, vv
                    improved = True
                    break
            if not improved:
                radius *= 0.5
    return Ss

def _full_fg_factory(beta):
    IU = np.triu_indices(N, 1)
    def fg(yf):
        Y = yf.reshape(N, D)
        nr = np.linalg.norm(Y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        X = Y / nr
        g = (X @ X.T)[IU]
        m = float(g.max())
        e = np.exp(beta * (g - m))
        Sm = float(e.sum()) + 1e-300
        F = m + np.log(Sm) / beta
        w = e / Sm
        P = np.zeros((N, N))
        P[IU] = w
        P = P + P.T
        GX = P @ X
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nr
        return F, GY.ravel()
    return fg

def _run_full_lbfgs(x0, beta, maxiter):
    fg = _full_fg_factory(float(beta))
    y0 = (x0 / np.linalg.norm(x0, axis=1, keepdims=True)).ravel().copy()
    try:
        r = minimize(fg, y0, jac=True, method='L-BFGS-B',
                     options={'maxiter': int(maxiter), 'ftol': 1e-16, 'gtol': 1e-8, 'maxfun': int(maxiter) * 3})
        Y = r.x.reshape(N, D)
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        return Y
    except Exception:
        return x0

def _full_lp(x, deadline, r0=6e-4, cap=2200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = N, D
    best = _max_cos(x)
    radius = r0
    iu = np.triu_indices(n, 1)
    while time.time() < deadline - 0.25 and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            od = np.argsort(-csel)[:cap]
            I, J, csel = I[od], J[od], csel[od]
            p = cap
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best - 1e-15:
            x, best = y, v
            radius = min(radius * 1.5, 0.02)
        else:
            ok = False
            for sh in (0.3, 0.1):
                yy = x + res.x[:-1].reshape(n, d) * sh
                yy /= np.linalg.norm(yy, axis=1, keepdims=True)
                vv = _max_cos(yy)
                if vv < best:
                    x, best = yy, vv
                    ok = True
                    break
            if not ok:
                radius *= 0.5
    return x

def solve():
    t0 = time.time()
    TLIM = 172.0
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    inc = None
    inc_val = 1e9
    for c in cands:
        try:
            c = np.asarray(c, dtype=float)
            if c.shape != (68, 12):
                continue
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = _max_cos(c)
            if v < inc_val:
                inc_val = v
                inc = c.copy()
        except Exception:
            continue
    if inc is None:
        rng0 = np.random.default_rng(0)
        inc = rng0.standard_normal((68, 12))
        inc /= np.linalg.norm(inc, axis=1, keepdims=True)
        inc_val = _max_cos(inc)
    x_best = inc.copy()
    best_val = float(inc_val)
    rng = np.random.default_rng(41717)
    # build ~20 multiplier sets: distinct values in 1..8
    mult_list = []
    mult_list.append([1, 2, 3, 4, 5, 6])
    mult_list.append([1, 2, 4, 5, 7, 8])
    mult_list.append([1, 3, 4, 6, 7, 8])
    while len(mult_list) < 22:
        perm = rng.choice(np.arange(1, 9), size=6, replace=False)
        # random signs -> map to 1..16
        cand = []
        for vv in perm:
            if rng.random() < 0.5:
                cand.append(int(vv))
            else:
                cand.append(int(17 - int(vv)))
        # avoid duplicate sets
        key = tuple(sorted([min(v, 17 - v) for v in cand]))
        seen = set(tuple(sorted([min(v, 17 - v) for v in m])) for m in mult_list)
        if key in seen and rng.random() < 0.8:
            continue
        mult_list.append(cand)
    sym_results = []  # (score, mults, seeds)
    scan_deadline = t0 + 115.0
    # quick chain for scan
    quick_chain = [(100.0, 120), (300.0, 120), (1000.0, 120), (3000.0, 80)]
    for mi, mults in enumerate(mult_list):
        if time.time() > scan_deadline:
            break
        if time.time() > t0 + TLIM - 30:
            break
        C, S_ = _rot_tables(mults)
        S0 = rng.standard_normal((NORB, D))
        S0 /= np.linalg.norm(S0, axis=1, keepdims=True)
        try:
            Sopt = _run_sym(S0, C, S_, quick_chain)
        except Exception:
            continue
        X = _expand(Sopt, C, S_)
        v = _max_cos(X)
        sym_results.append((v, mults, Sopt))
        # also try sign-flipped seed variant if time allows? keep to one per mult for speed
    if len(sym_results) == 0:
        return x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    sym_results.sort(key=lambda t: t[0])
    # polish best two
    polished = []
    for idx in range(min(2, len(sym_results))):
        if time.time() > t0 + TLIM - 35:
            break
        v, mults, Sopt = sym_results[idx]
        C, S_ = _rot_tables(mults)
        try:
            S2 = _run_sym(Sopt, C, S_, [(1000.0, 200), (3000.0, 250)])
        except Exception:
            S2 = Sopt
        # symmetry-preserving LP with small budget
        try:
            dl = min(time.time() + 4.0, t0 + TLIM - 30)
            if dl > time.time() + 0.8:
                S2 = _sym_lp(S2, C, S_, dl, r0=3e-3)
        except Exception:
            pass
        X2 = _expand(S2 / np.linalg.norm(S2, axis=1, keepdims=True), C, S_)
        v2 = _max_cos(X2)
        polished.append((v2, mults, S2))
    if len(polished) == 0:
        polished = sym_results[:1]
    polished.sort(key=lambda t: t[0])
    sym_best_val, sym_best_mults, sym_best_seeds = polished[0]
    Cb, Sb = _rot_tables(sym_best_mults)
    sym_best_seeds = sym_best_seeds / np.linalg.norm(sym_best_seeds, axis=1, keepdims=True)
    Xsym = _expand(sym_best_seeds, Cb, Sb)
    # decide release: within 5e-4 of incumbent (or better)
    if sym_best_val <= best_val + 5e-4:
        # release symmetry and polish fully
        Xf = Xsym.copy()
        # a couple of full smooth steps then LP
        remain = TLIM - (time.time() - t0)
        if remain > 12:
            try:
                Xf = _run_full_lbfgs(Xf, 300.0, 120)
            except Exception:
                pass
        if time.time() - t0 < TLIM - 10:
            try:
                Xf = _run_full_lbfgs(Xf, 1200.0, 120)
            except Exception:
                pass
        dl = t0 + TLIM - 1.2
        if time.time() < dl - 1.0:
            try:
                Xf = _full_lp(Xf, dl, r0=5e-4)
            except Exception:
                pass
        vf = _max_cos(Xf)
        if vf < best_val:
            x_best = Xf
            best_val = vf
        # keep symmetric if full somehow worse but symmetric beats inc
        if sym_best_val < best_val and sym_best_val < inc_val:
            # prefer the better
            x_best = Xsym
            best_val = sym_best_val
    # else: no release, return incumbent
    # final choice: best among x_best and incumbent
    if best_val >= inc_val and not np.array_equal(x_best, inc):
        # ensure we return at least incumbent if nothing better
        if _max_cos(x_best) > inc_val:
            x_best = inc.copy()
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
