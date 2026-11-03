import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _spectrum(x, iuI=None, iuJ=None):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    if iuI is None:
        c = G[np.triu_indices(x.shape[0], 1)]
    else:
        c = G[iuI, iuJ]
    s = np.sort(c)[::-1]
    c0 = float(s[0]); c100 = float(s[99]); c1000 = float(s[999])
    m1500 = float(s[:1500].mean())
    return c0, c100, c1000, m1500, s

def _tangent_bases(x):
    n, d = x.shape
    dd = d - 1
    Q = np.empty((n, d, dd))
    amax = np.argmax(np.abs(x), axis=1)
    E = np.eye(d)
    for i in range(n):
        k = int(amax[i])
        xi = x[i]
        cols = [l for l in range(d) if l != k]
        F = np.empty((d, dd))
        for ci, l in enumerate(cols):
            e = E[l]
            F[:, ci] = e - xi[l] * xi
        Qi, _ = np.linalg.qr(F)
        Q[i] = Qi
    return Q

def _is_better(new_spec, best_spec, allow=5e-5, tol=1e-12):
    n0, n1, n2, n3 = new_spec
    b0, b1, b2, b3 = best_spec
    if n0 < b0 - tol:
        return True
    if n0 > b0 + allow + tol:
        return False
    # within allowance: compare plateau
    if n1 < b1 - tol:
        return True
    if n1 > b1 + tol:
        return False
    if n2 < b2 - tol:
        return True
    if n2 > b2 + tol:
        return False
    if n3 < b3 - tol:
        return True
    if n3 > b3 + tol:
        return False
    # all plateau equal, then smaller c0 wins (already handled large win)
    if n0 < b0 - tol:
        return True
    return False

def _adam_cvar(x0, steps, K0, K1, lr, plat_spec, deadline, check_every=50):
    n, d = x0.shape
    x = x0.copy()
    m = np.zeros_like(x); v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    NUP = n * (n - 1) // 2
    iuI, iuJ = np.triu_indices(n, 1)
    best_lex = None
    best_lex_spec = None
    best_lex_x = None
    # track best c0 too
    best_c0 = 1e9
    best_c0_x = None
    w1 = 1.0 / 200.0
    w2 = 0.3 / 1300.0
    for t in range(1, steps + 1):
        if time.time() > deadline and t > 20:
            break
        frac = t / steps
        K = float(np.exp(la0 + (la1 - la0) * frac))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        c_all = G[iuI, iuJ]
        # ranking
        # top 1500 indices
        if NUP > 1500:
            part = np.argpartition(-c_all, 1500)[:1500]
            # need order among them to split 200/1300
            vals = c_all[part]
            o = np.argsort(-vals)
            top = part[o[:200]]
            nxt = part[o[200:1500]]
        else:
            o = np.argsort(-c_all)
            top = o[:200]
            nxt = o[200:]
        W = np.exp(np.clip(K * (G - gmax), -50.0, 0.0))
        np.fill_diagonal(W, 0.0)
        ssum = float(W.sum())
        if ssum > 0 and np.isfinite(ssum):
            W /= ssum
        else:
            W[:] = 0.0
        # add rank weights (directed, symmetric)
        # W currently directed softmax sums to 1; rank adds ~2.6
        It = iuI[top]; Jt = iuJ[top]
        W[It, Jt] += w1
        W[Jt, It] += w1
        In = iuI[nxt]; Jn = iuJ[nxt]
        if len(In) > 0:
            W[In, Jn] += w2
            W[Jn, In] += w2
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1; b2p *= b2
        m *= b1; m += (1 - b1) * gt
        v *= b2; v += (1 - b2) * (gt * gt)
        mh = m / (1 - b1p); vh = v / (1 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
        if t % check_every == 0 or t == steps:
            try:
                c0, c100, c1000, mm, _ = _spectrum(x, iuI, iuJ)
            except Exception:
                continue
            if c0 < best_c0:
                best_c0 = c0
                best_c0_x = x.copy()
            spec = (c0, c100, c1000, mm)
            if _is_better(spec, plat_spec):
                # update local best lex
                if best_lex_spec is None or _is_better(spec, best_lex_spec, allow=1e9):
                    # allow=large to compare among candidates: pure lex with c0 first
                    best_lex_spec = spec
                    best_lex_x = x.copy()
            # also keep track of best plateau regardless? we use plat_spec as ref
    return x, best_c0, best_c0_x, best_lex_spec, best_lex_x

def _lp_pair(x, radius, margin, cap, time_lim):
    n, d = x.shape
    dd = d - 1
    NV = n * dd
    iuI, iuJ = np.triu_indices(n, 1)
    G = x @ x.T
    c_all = G[iuI, iuJ]
    m = float(c_all.max())
    sel = c_all > (m - margin)
    I = iuI[sel]; J = iuJ[sel]; csel = c_all[sel]
    p = len(I)
    if p == 0:
        return []
    if p > cap:
        part = np.argpartition(-csel, cap)[:cap]
        I = I[part]; J = J[part]; csel = csel[part]
        p = cap
    try:
        Q = _tangent_bases(x)
    except Exception:
        return []
    xI = x[I]; xJ = x[J]
    QI = Q[I]; QJ = Q[J]
    AI = np.sum(QI * xJ[:, :, None], axis=1)
    AJ = np.sum(QJ * xI[:, :, None], axis=1)
    ar = np.arange(dd)
    cols_block = np.empty((p, 2 * dd + 1), dtype=int)
    cols_block[:, :dd] = I[:, None] * dd + ar[None, :]
    cols_block[:, dd:2 * dd] = J[:, None] * dd + ar[None, :]
    cols_block[:, -1] = NV
    vals_block = np.empty((p, 2 * dd + 1), dtype=float)
    vals_block[:, :dd] = AI
    vals_block[:, dd:2 * dd] = AJ
    vals_block[:, -1] = -1.0
    rows = np.repeat(np.arange(p), 2 * dd + 1)
    cols = cols_block.ravel(); vals = vals_block.ravel()
    try:
        Aub = coo_matrix((vals, (rows, cols)), shape=(p, NV + 1)).tocsr()
    except Exception:
        return []
    bub = -csel
    obj1 = np.zeros(NV + 1); obj1[-1] = 1.0
    bnds1 = [(-radius, radius)] * NV + [(None, None)]
    cands = []
    t_opt = None
    y1 = None
    try:
        res = linprog(obj1, A_ub=Aub, b_ub=bub, bounds=bnds1, method="highs",
                      options={"time_limit": time_lim, "presolve": True})
    except Exception:
        res = None
    if res is not None and getattr(res, "status", 1) == 0 and res.x is not None:
        try:
            y1 = np.asarray(res.x[:-1], dtype=float).reshape(n, dd)
            t_opt = float(res.x[-1])
            if np.all(np.isfinite(y1)) and np.isfinite(t_opt):
                dx = np.einsum("ndk,nk->nd", Q, y1)
                tr = x + dx
                tr /= np.linalg.norm(tr, axis=1, keepdims=True)
                cands.append((tr, t_opt, Q, AI, AJ, I, J, csel))
            else:
                y1 = None; t_opt = None
        except Exception:
            y1 = None; t_opt = None
    # sum-load LP2 at fixed t_opt
    if t_opt is not None and y1 is not None:
        try:
            p2n = min(p, 1500)
            o = np.argsort(-csel)[:p2n]
            I2 = I[o]; J2 = J[o]; cs2 = csel[o]
            AI2 = AI[o]; AJ2 = AJ[o]
            coeff = np.zeros((n, dd))
            np.add.at(coeff, I2, AI2)
            np.add.at(coeff, J2, AJ2)
            c2 = coeff.ravel()
            cols2b = np.empty((p2n, 2 * dd), dtype=int)
            cols2b[:, :dd] = I2[:, None] * dd + ar[None, :]
            cols2b[:, dd:] = J2[:, None] * dd + ar[None, :]
            vals2b = np.empty((p2n, 2 * dd), dtype=float)
            vals2b[:, :dd] = AI2
            vals2b[:, dd:] = AJ2
            rows2 = np.repeat(np.arange(p2n), 2 * dd)
            Ay = coo_matrix((vals2b.ravel(), (rows2, cols2b.ravel())), shape=(p2n, NV)).tocsr()
            bub2 = -cs2 + t_opt + 1e-12
            bnds2 = [(-radius, radius)] * NV
            res2 = linprog(c2, A_ub=Ay, b_ub=bub2, bounds=bnds2, method="highs",
                           options={"time_limit": time_lim, "presolve": True})
            if res2 is not None and getattr(res2, "status", 1) == 0 and res2.x is not None:
                y2 = np.asarray(res2.x, dtype=float).reshape(n, dd)
                if np.all(np.isfinite(y2)):
                    dx2 = np.einsum("ndk,nk->nd", Q, y2)
                    tr2 = x + dx2
                    tr2 /= np.linalg.norm(tr2, axis=1, keepdims=True)
                    cands.append((tr2, t_opt, None, None, None, None, None, None))
        except Exception:
            pass
    # return trial arrays
    return [c[0] for c in cands]

def _strict_polish(x0, deadline, cap=3000):
    x = x0 / np.linalg.norm(x0, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    iuI, iuJ = np.triu_indices(n, 1)
    radii = [3e-4, 1e-4, 3e-5, 1e-5, 3e-6]
    ri = 0
    while time.time() < deadline:
        r = radii[ri % len(radii)]; ri += 1
        G = x @ x.T
        c = G[iuI, iuJ]
        mval = float(c.max())
        margin = max(20.0 * r, 3e-5)
        sel = c > (mval - margin)
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        pnum = len(I)
        if pnum == 0:
            continue
        if pnum > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]; pnum = cap
        # quick exit if time low
        rem = deadline - time.time()
        if rem < 1.5:
            break
        try:
            Q = _tangent_bases(x)
        except Exception:
            break
        dd = d - 1; NV = n * dd
        xI = x[I]; xJ = x[J]
        QI = Q[I]; QJ = Q[J]
        AI = np.sum(QI * xJ[:, :, None], axis=1)
        AJ = np.sum(QJ * xI[:, :, None], axis=1)
        ar = np.arange(dd)
        cols_block = np.empty((pnum, 2 * dd + 1), dtype=int)
        cols_block[:, :dd] = I[:, None] * dd + ar[None, :]
        cols_block[:, dd:2 * dd] = J[:, None] * dd + ar[None, :]
        cols_block[:, -1] = NV
        vals_block = np.empty((pnum, 2 * dd + 1), dtype=float)
        vals_block[:, :dd] = AI
        vals_block[:, dd:2 * dd] = AJ
        vals_block[:, -1] = -1.0
        rows = np.repeat(np.arange(pnum), 2 * dd + 1)
        try:
            Aub = coo_matrix((vals_block.ravel(), (rows, cols_block.ravel())), shape=(pnum, NV + 1)).tocsr()
        except Exception:
            break
        bub = -csel
        obj1 = np.zeros(NV + 1); obj1[-1] = 1.0
        bnds1 = [(-r, r)] * NV + [(None, None)]
        try:
            tl = min(5.0, max(0.8, rem - 0.5))
            res = linprog(obj1, A_ub=Aub, b_ub=bub, bounds=bnds1, method="highs",
                          options={"time_limit": tl, "presolve": True})
        except Exception:
            continue
        if res is None or getattr(res, "status", 1) != 0 or res.x is None:
            continue
        try:
            y1 = np.asarray(res.x[:-1], dtype=float).reshape(n, dd)
            if not np.all(np.isfinite(y1)):
                continue
            dx = np.einsum("ndk,nk->nd", Q, y1)
            trial = x + dx
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            nm = _max_cos(trial)
            if nm < best - 1e-12:
                best = nm; best_x = trial.copy()
                x = trial.copy()
            elif nm <= _max_cos(x) + 1e-12:
                # accept neutral to keep walking
                x = trial.copy()
        except Exception:
            continue
    return best_x, best

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    CYC_END = t0 + 128.0
    rng = np.random.default_rng(0)
    n, d = 256, 6
    iuI, iuJ = np.triu_indices(n, 1)
    # load incumbent
    best_x = None; best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                nr[nr == 0] = 1.0
                a = a / nr
                vv = _max_cos(a)
                if vv < best:
                    best = vv
                    best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        best_x = rng.normal(size=(n, d))
        best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
        best = _max_cos(best_x)
    c0, c100, c1000, m1500, _ = _spectrum(best_x, iuI, iuJ)
    plat_x = best_x.copy()
    plat_spec = (c0, c100, c1000, m1500)
    c0_best_x = best_x.copy()
    c0_best = c0
    cur = best_x.copy()
    radii_cycle = [1e-3, 3e-4, 1e-4, 3e-5]
    lrs = [3e-3, 2e-3, 2e-3, 1.2e-3, 1e-3, 2.5e-3, 1.5e-3, 1e-3, 8e-4, 2e-3]
    noises = [0.0, 6e-4, 1e-3, 4e-4, 8e-4, 5e-4, 1.2e-3, 3e-4, 7e-4, 9e-4]
    cyc = 0
    while time.time() < CYC_END and cyc < 10:
        if T_END - time.time() < 30:
            break
        lr = lrs[cyc % len(lrs)]
        ns = noises[cyc % len(noises)]
        # starting point: perturb plat
        if ns > 0:
            pert = rng.normal(size=(n, d)) * ns
            # project to tangent of plat
            dotp = np.sum(pert * plat_x, axis=1, keepdims=True)
            pert = pert - dotp * plat_x
            start = plat_x + pert
            start /= np.linalg.norm(start, axis=1, keepdims=True)
        else:
            start = cur.copy() if cyc == 0 else plat_x.copy()
        steps = 650 if cyc < 6 else 500
        deadline_a = min(CYC_END - 4.0, time.time() + 11.0)
        # anneal K 500->8000
        try:
            final_x, ac0, ac0x, alex_spec, alex_x = _adam_cvar(start, steps, 500.0, 8000.0, lr, plat_spec, deadline_a, check_every=50)
        except Exception:
            final_x = None; ac0x = None; alex_x = None
            ac0 = 1e9; alex_spec = None
        # evaluate adam final
        cand_list = []
        if final_x is not None:
            cand_list.append(final_x)
        if ac0x is not None:
            cand_list.append(ac0x)
        if alex_x is not None:
            cand_list.append(alex_x)
        for cand in cand_list:
            try:
                s = _spectrum(cand, iuI, iuJ)
                spec = (s[0], s[1], s[2], s[3])
            except Exception:
                continue
            if s[0] < c0_best - 1e-15:
                c0_best = s[0]; c0_best_x = cand.copy()
            if _is_better(spec, plat_spec):
                plat_spec = spec
                plat_x = cand.copy()
        # LP phase from plat_x with wide margin
        r = radii_cycle[cyc % len(radii_cycle)]
        rem = CYC_END - time.time()
        if rem < 3.0:
            cyc += 1
            continue
        try:
            trials = _lp_pair(plat_x, r, margin=1e-4, cap=2500, time_lim=min(5.0, max(1.0, rem - 1.0)))
        except Exception:
            trials = []
        for tr in trials:
            try:
                s = _spectrum(tr, iuI, iuJ)
                spec = (s[0], s[1], s[2], s[3])
            except Exception:
                continue
            if s[0] < c0_best - 1e-15:
                c0_best = s[0]; c0_best_x = tr.copy()
            if _is_better(spec, plat_spec):
                plat_spec = spec
                plat_x = tr.copy()
        cur = plat_x.copy()
        cyc += 1
    # strict polish both plat and c0 best
    time_left = T_END - time.time()
    if time_left > 8:
        half = time_left / 2.0
        try:
            px, pv = _strict_polish(plat_x, time.time() + half - 1.0)
            if pv < c0_best:
                c0_best = pv; c0_best_x = px.copy()
        except Exception:
            pass
        try:
            qx, qv = _strict_polish(c0_best_x, T_END - 1.0)
            if qv < c0_best:
                c0_best = qv; c0_best_x = qx.copy()
        except Exception:
            pass
    else:
        pass
    out = c0_best_x / np.linalg.norm(c0_best_x, axis=1, keepdims=True)
    # fallback
    if not np.all(np.isfinite(out)) or _max_cos(out) > best + 1e-9:
        out = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return np.asarray(out, dtype=float)
