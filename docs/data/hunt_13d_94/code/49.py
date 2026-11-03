import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _tangent_perturb(X, sigma, rng):
    N = rng.standard_normal(size=X.shape)
    dot = (N * X).sum(axis=1, keepdims=True)
    Nt = N - dot * X
    Y = X + sigma * Nt
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y

def _expand_forward(params, template):
    n = 94; d = 13
    X = np.empty((n, d), dtype=np.float64)
    pos = 0; ppos = 0
    ulist = []
    nlist = []
    for b in template:
        if b == 26:
            v = params[ppos:ppos+13].astype(np.float64)
            ppos += 13
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                v = np.ones(13); nv = float(np.linalg.norm(v))
            u = v / nv
            ulist.append(u); nlist.append(nv)
            for k in range(13):
                X[pos+k] = np.roll(u, k)
                X[pos+13+k] = -np.roll(u, k)
            pos += 26
        elif b == 13:
            v = params[ppos:ppos+13].astype(np.float64)
            ppos += 13
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                v = np.ones(13); nv = float(np.linalg.norm(v))
            u = v / nv
            ulist.append(u); nlist.append(nv)
            for k in range(13):
                X[pos+k] = np.roll(u, k)
            pos += 13
        elif b == 16:
            V = params[ppos:ppos+16*13].reshape(16,13).astype(np.float64)
            ppos += 16*13
            nr = np.linalg.norm(V, axis=1, keepdims=True)
            nr[nr < 1e-12] = 1.0
            U = V / nr
            ulist.append(U); nlist.append(nr.ravel().copy())
            X[pos:pos+16] = U
            pos += 16
    return X, ulist, nlist

def _sym_fg_factory(template, kk):
    def fg(params):
        X, ulist, nlist = _expand_forward(params, template)
        n = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp(kk * (G - m))
        np.fill_diagonal(E, 0.0)
        Sall = float(E.sum())
        if Sall <= 0 or not np.isfinite(Sall):
            return 1e6, np.zeros_like(params)
        Sprime = Sall * 0.5
        f = m + np.log(Sprime) / kk
        # grad w.r.t X (normalized rows)
        GX = (E @ X) / Sprime
        # backprop to params
        grad = np.empty_like(params)
        pos = 0; ppos = 0; bi = 0
        for b in template:
            if b == 26:
                u = ulist[bi]; nv = nlist[bi]; bi += 1
                gu = np.zeros(13)
                for k in range(13):
                    gu += np.roll(GX[pos+k], -k)
                    gu += -np.roll(GX[pos+13+k], -k)
                dot = float(gu @ u)
                gv = (gu - dot * u) / nv
                grad[ppos:ppos+13] = gv
                pos += 26; ppos += 13
            elif b == 13:
                u = ulist[bi]; nv = nlist[bi]; bi += 1
                gu = np.zeros(13)
                for k in range(13):
                    gu += np.roll(GX[pos+k], -k)
                dot = float(gu @ u)
                gv = (gu - dot * u) / nv
                grad[ppos:ppos+13] = gv
                pos += 13; ppos += 13
            elif b == 16:
                U = ulist[bi]; nr = nlist[bi]; bi += 1
                Gblk = GX[pos:pos+16]
                dot = (Gblk * U).sum(axis=1)
                GV = (Gblk - dot[:, None] * U) / nr[:, None]
                grad[ppos:ppos+16*13] = GV.ravel()
                pos += 16; ppos += 16*13
        return float(f), grad
    return fg

def _global_adam(x, steps, k0, k1, lr0, lr1):
    n, d = x.shape
    m1 = np.zeros_like(x); v1 = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        frac = s / max(1, steps - 1)
        kk = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        mval = float(np.max(G - np.eye(n) * 2.0))
        W = np.exp(kk * (G - mval))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum()) * 0.5
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (W @ x) / S
        dot = (Grad * x).sum(axis=1, keepdims=True)
        Gt = Grad - dot * x
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1); bc2 = 1.0 - b2 ** (s + 1)
        upd = lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
        x = x - upd
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if not np.all(np.isfinite(x)):
            break
    return x

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20.0 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    T0 = time.time()
    BUDGET = 176.0
    SYM_END = T0 + 62.0
    ADAM_END = T0 + 138.0
    FINAL_END = T0 + BUDGET - 2.0
    rng = np.random.default_rng(0)
    rec = None; bst = None
    for fn in ["record.npy", "best.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                if fn == "record.npy": rec = a.copy()
                else: bst = a.copy()
        except Exception:
            pass
    if rec is None and bst is None:
        r = rng.standard_normal(size=(94, 13))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        rec = r.copy(); bst = r.copy()
    if rec is None: rec = bst.copy()
    if bst is None: bst = rec.copy()
    rv = _max_cos(rec); bv = _max_cos(bst)
    if bv < rv:
        global_best = bst.copy(); global_val = bv
    else:
        global_best = rec.copy(); global_val = rv
    # templates summing to 94
    templates = [
        [26, 26, 26, 16],
        [26, 26, 13, 13, 16],
        [26, 13, 13, 13, 13, 16],
        [13, 13, 13, 13, 13, 13, 16],
        [26, 26, 26, 13, 3-3+0],  # placeholder replaced below
    ]
    templates = [
        [26, 26, 26, 16],
        [26, 26, 13, 13, 16],
        [26, 13, 13, 13, 13, 16],
        [13, 13, 13, 13, 13, 13, 16],
    ]
    # legendre pattern length 13
    leg = np.ones(13)
    qr = {1, 3, 4, 9, 10, 12}
    for i in range(13):
        if i == 0:
            leg[i] = -1.0
        else:
            leg[i] = 1.0 if (i % 13) in qr else -1.0
    leg = leg / np.linalg.norm(leg)
    sym_results = []
    ntrials = 100
    kk = 90.0
    for trial in range(ntrials):
        if time.time() > SYM_END:
            break
        tmpl = templates[trial % len(templates)]
        # init params
        parts = []
        use_leg = (trial < 8)
        for b in tmpl:
            if b == 26 or b == 13:
                if use_leg:
                    nz = rng.standard_normal(13) * 0.3
                    v0 = leg + nz * 0.3
                else:
                    v0 = rng.standard_normal(13)
                parts.append(v0)
            elif b == 16:
                V0 = rng.standard_normal((16, 13))
                parts.append(V0.ravel())
        p0 = np.concatenate(parts)
        fg = _sym_fg_factory(tmpl, kk)
        try:
            res = minimize(fg, p0, method='L-BFGS-B', jac=True,
                           options={'maxiter': 60, 'maxfun': 4000, 'ftol': 1e-12, 'gtol': 1e-5, 'maxcor': 8})
            popt = res.x
        except Exception:
            popt = p0
        try:
            Xexp, _, _ = _expand_forward(popt, tmpl)
            Xexp = Xexp / np.linalg.norm(Xexp, axis=1, keepdims=True)
            val = _max_cos(Xexp)
        except Exception:
            continue
        sym_results.append((val, Xexp.copy(), tmpl))
        if val < global_val and np.all(np.isfinite(Xexp)):
            global_val = val
            global_best = Xexp.copy()
    sym_results.sort(key=lambda t: t[0])
    keep5 = sym_results[:5] if len(sym_results) >= 5 else sym_results
    # break symmetry + unrestricted Adam
    sigmas = [0.005, 0.02, 0.05]
    adam_cands = []
    for (sv, SX, tmpl) in keep5:
        for sg in sigmas:
            if time.time() > ADAM_END:
                break
            try:
                Y = _tangent_perturb(SX, sg, rng)
                Y2 = _global_adam(Y, steps=350, k0=40.0, k1=450.0, lr0=0.015, lr1=0.0015)
                v2 = _max_cos(Y2)
            except Exception:
                continue
            adam_cands.append((v2, Y2.copy()))
            if v2 < global_val:
                global_val = v2
                global_best = Y2.copy()
        if time.time() > ADAM_END:
            break
    # also perturb global best directly
    if time.time() < ADAM_END - 5:
        for sg in [0.01, 0.03]:
            if time.time() > ADAM_END:
                break
            try:
                Y = _tangent_perturb(global_best, sg, rng)
                Y2 = _global_adam(Y, steps=400, k0=50.0, k1=500.0, lr0=0.012, lr1=0.001)
                v2 = _max_cos(Y2)
                adam_cands.append((v2, Y2.copy()))
                if v2 < global_val:
                    global_val = v2; global_best = Y2.copy()
            except Exception:
                pass
    adam_cands.sort(key=lambda t: t[0])
    # final LP polish on few best
    cands = []
    cands.append((global_val, global_best.copy()))
    for vv, XX in adam_cands[:4]:
        cands.append((vv, XX.copy()))
    for vv, XX, _ in keep5[:2]:
        cands.append((vv, XX.copy()))
    cands.sort(key=lambda t: t[0])
    uniq = []
    for vv, XX in cands:
        dup = False
        for _, YY in uniq:
            try:
                if np.linalg.norm(XX - YY) < 1e-6:
                    dup = True; break
            except Exception:
                pass
        if not dup:
            uniq.append((vv, XX))
        if len(uniq) >= 5:
            break
    remaining = FINAL_END - time.time()
    if remaining > 4 and len(uniq) > 0:
        per = min(max(remaining / max(1, len(uniq)), 4.0), 22.0)
        final_best = global_best.copy(); final_val = global_val
        for vv, XX in uniq:
            if time.time() > FINAL_END - 1.5:
                break
            dl = min(time.time() + per, FINAL_END)
            try:
                P, pv = _lp_polish(XX.copy(), dl)
            except Exception:
                continue
            if pv < final_val:
                final_val = pv; final_best = P.copy()
        global_best = final_best; global_val = final_val
    global_best = global_best / np.linalg.norm(global_best, axis=1, keepdims=True)
    return global_best
