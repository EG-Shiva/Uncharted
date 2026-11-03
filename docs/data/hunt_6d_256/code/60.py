import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _spec_from_sorted(c):
    # c sorted descending
    c0 = float(c[0])
    c100 = float(c[99])
    c1000 = float(c[999])
    m1500 = float(c[:1500].mean())
    return c0, c100, c1000, m1500

def _spec(x, iuI=None, iuJ=None):
    n = x.shape[0]
    if iuI is None:
        iuI, iuJ = np.triu_indices(n, 1)
    G = x @ x.T
    c = G[iuI, iuJ]
    cs = np.sort(c)[::-1]
    return _spec_from_sorted(cs), cs

def _lex_plateau_better(nspec, cspec):
    # compare (c100, c1000, mean1500, c0) lexicographically
    # nspec, cspec are (c0,c100,c1000,m)
    tol = 1e-12
    # c100
    if nspec[1] < cspec[1] - tol:
        return True
    if nspec[1] > cspec[1] + tol:
        return False
    if nspec[2] < cspec[2] - tol:
        return True
    if nspec[2] > cspec[2] + tol:
        return False
    if nspec[3] < cspec[3] - 5e-13:
        return True
    if nspec[3] > cspec[3] + 5e-13:
        return False
    if nspec[0] < cspec[0] - tol:
        return True
    return False

def _adam_cvar(x0, steps, K0, K1, lr, iuI, iuJ, ceiling, cur_spec, best_c0, deadline):
    n, d = x0.shape
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    # track best lex seen in this run (init = start)
    run_best_x = x.copy()
    run_best_spec = cur_spec  # reference for improvement inside run
    # we keep run_best updated when lex improves (and within ceiling)
    # also track best c0 for global
    M = iuI.size
    for t in range(1, steps + 1):
        if time.time() > deadline:
            break
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G.max())
        cflat = G[iuI, iuJ]
        # top sets via argpartition
        # 1500
        try:
            idx1500 = np.argpartition(-cflat, 1499)[:1500]
        except Exception:
            idx1500 = np.argsort(-cflat)[:1500]
        v1500 = cflat[idx1500]
        try:
            sub = np.argpartition(-v1500, 199)[:200]
        except Exception:
            sub = np.argsort(-v1500)[:200]
        idx200 = idx1500[sub]
        mask200 = np.zeros(1500, dtype=bool)
        mask200[sub] = True
        idxmid = idx1500[~mask200]
        # LSE weights
        # K*(c-cmax) <=0
        E = K * (cflat - cmax)
        # clip very negative to avoid underflow cost (exp(-50)->0)
        np.clip(E, -50.0, 0.0, out=E)
        Wf = np.exp(E)
        s = float(Wf.sum())
        if not np.isfinite(s) or s <= 0:
            s = 1.0
            Wf[:] = 0.0
        f = Wf / s  # alpha=1
        f[idx200] += 1.0 / 200.0
        f[idxmid] += 0.3 / 1300.0
        # build symmetric W
        W = np.zeros((n, n))
        W[iuI, iuJ] = f
        W += W.T
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 50 == 0 or t == steps:
            try:
                G2 = x @ x.T
                c2 = G2[iuI, iuJ]
                cs2 = np.sort(c2)[::-1]
                nspec = _spec_from_sorted(cs2)
            except Exception:
                continue
            if nspec[0] <= ceiling + 1e-12 and _lex_plateau_better(nspec, run_best_spec):
                run_best_spec = nspec
                run_best_x = x.copy()
    return run_best_x, run_best_spec

def _lp_attempt(x, radius, margin, cap, iuI, iuJ):
    n, d = x.shape
    G = x @ x.T
    c = G[iuI, iuJ]
    mv = float(c.max())
    sel = c > (mv - margin)
    I = iuI[sel]; J = iuJ[sel]
    csel = c[sel]
    p = len(I)
    if p == 0:
        return None
    if p > cap:
        part = np.argpartition(-csel, cap)[:cap]
        I = I[part]; J = J[part]; csel = csel[part]
        p = cap
    rows = np.repeat(np.arange(p), 2 * d + 1)
    cols = np.concatenate([I[:, None] * d + np.arange(d),
                           J[:, None] * d + np.arange(d),
                           np.full((p, 1), n * d)], 1).ravel()
    vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
    try:
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
    except Exception:
        return None
    obj = np.zeros(n * d + 1)
    obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
    except Exception:
        return None
    if res.status != 0:
        return None
    y = x + res.x[:-1].reshape(n, d)
    nrm = np.linalg.norm(y, axis=1, keepdims=True)
    if not np.all(np.isfinite(nrm)) or np.any(nrm < 1e-12):
        return None
    y /= nrm
    return y

def _lp_polish_strict(x, deadline, r0=2.0e-3, cap=3000, iuI=None, iuJ=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = r0
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        mv = float(c.max())
        margin = max(20.0 * radius, 5e-5)
        sel = c > (mv - margin)
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            p = cap
        y = _lp_attempt(x, radius, margin, cap, iuI, iuJ)
        # _lp_attempt recomputes selection; to keep consistent we call generic with same radius/margin
        # Instead we already have selection; build directly (reuse y if built with same margin)
        # For simplicity use y from _lp_attempt (it used same margin? no, it uses margin param)
        # We called with margin variable, so ok.
        if y is None:
            radius *= 0.5
            continue
        val = _max_cos(y)
        if val < best - 1e-13:
            x, best = y, val
            best_x = y.copy()
            radius = min(radius * 1.4, 0.02)
        else:
            # also update x if equal? keep x for further shrinking
            radius *= 0.5
    return best_x

def solve():
    t0 = time.time()
    DEADLINE = t0 + 178.0
    n, d = 256, 6
    iuI, iuJ = np.triu_indices(n, 1)
    base = None
    base_v = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    vv = _max_cos(a)
                    if vv < base_v:
                        base_v = vv
                        base = a.copy()
        except Exception:
            pass
    if base is None:
        rs = np.random.default_rng(0)
        base = rs.normal(size=(n, d))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        base_v = _max_cos(base)
    orig = base.copy()
    orig_c0 = base_v
    cur_x = base.copy()
    cur_spec, _ = _spec(cur_x, iuI, iuJ)
    best_x = base.copy()
    best_c0 = base_v
    best_spec = cur_spec
    ceiling = orig_c0 + 5e-5

    radii_cycle = [1e-3, 3e-4, 1e-4, 3e-5]
    n_cycles = 7
    # allocate time: leave 35s for final polish
    cycle_deadline = DEADLINE - 35.0
    for cyc in range(n_cycles):
        if time.time() > cycle_deadline - 5:
            break
        # (a) Adam CVaR, K annealed 500->8000
        steps = 500 if cyc < 5 else 400
        lr = 0.007 if cyc % 2 == 0 else 0.005
        try:
            ax, aspec = _adam_cvar(cur_x, steps, 500.0, 8000.0, lr, iuI, iuJ, ceiling + 1e-12, cur_spec, best_c0, min(cycle_deadline, DEADLINE - 30))
        except Exception:
            ax, aspec = None, None
        if ax is not None:
            # check global best
            if aspec[0] < best_c0 - 1e-12:
                best_c0 = aspec[0]
                best_x = ax.copy()
                best_spec = aspec
                ceiling = min(ceiling, best_c0 + 5e-5)
            # check cur update: accept if within ceiling and lex better than cur
            if aspec[0] <= ceiling + 1e-12 and _lex_plateau_better(aspec, cur_spec):
                cur_x = ax.copy()
                cur_spec = aspec
            elif aspec[0] < cur_spec[0] - 1e-12 and aspec[0] <= ceiling + 1e-12:
                # c0 gain even if plateau tie logic missed
                # require plateau not badly degraded
                if aspec[1] <= cur_spec[1] + 5e-7 and aspec[2] <= cur_spec[2] + 5e-7:
                    cur_x = ax.copy()
                    cur_spec = aspec
        if time.time() > cycle_deadline - 3:
            break
        # (b) tangent LP with widened margin 1e-4, cycling radii
        # use two radii per cycle to save time: current and next
        rlist = [radii_cycle[cyc % 4], radii_cycle[(cyc + 1) % 4]]
        # alternate single radius per cycle plus extra if time rich? use rlist
        for r in rlist:
            if time.time() > cycle_deadline:
                break
            try:
                y = _lp_attempt(cur_x, r, 1e-4, 2500, iuI, iuJ)
            except Exception:
                y = None
            if y is None:
                continue
            try:
                G2 = y @ y.T
                c2 = G2[iuI, iuJ]
                cs2 = np.sort(c2)[::-1]
                nspec = _spec_from_sorted(cs2)
            except Exception:
                continue
            if nspec[0] < best_c0 - 1e-12:
                # strict improvement -> update both
                best_c0 = nspec[0]
                best_x = y.copy()
                best_spec = nspec
                ceiling = min(ceiling, best_c0 + 5e-5)
            if nspec[0] <= ceiling + 1e-12 and _lex_plateau_better(nspec, cur_spec):
                cur_x = y.copy()
                cur_spec = nspec
            elif nspec[0] < cur_spec[0] - 5e-9 and nspec[0] <= ceiling + 1e-12:
                if nspec[1] <= cur_spec[1] + 5e-7:
                    cur_x = y.copy()
                    cur_spec = nspec
            # if LP made c0 much worse, do not keep it; continue from cur_x
    # final: strict LP polish from cur_x and best_x
    # choose starts: cur_x (plateau-pushed) and best_x
    candidates = []
    try:
        candidates.append((best_c0, best_x.copy()))
    except Exception:
        pass
    try:
        candidates.append((cur_spec[0], cur_x.copy()))
    except Exception:
        pass
    # also original fallback
    try:
        candidates.append((orig_c0, orig.copy()))
    except Exception:
        pass
    final_best_x = best_x.copy()
    final_best_c0 = best_c0
    # polish each candidate briefly, pick lowest c0
    # give most time to cur_x
    remaining = DEADLINE - time.time()
    if remaining > 5:
        # polish cur_x first with strict LP
        try:
            px = _lp_polish_strict(cur_x, DEADLINE - 8.0, r0=2.0e-3, cap=3000, iuI=iuI, iuJ=iuJ)
            pv = _max_cos(px)
            if pv < final_best_c0 - 1e-13:
                final_best_c0 = pv
                final_best_x = px.copy()
            # if cur polish improved over best, keep it
            # also try polishing best_x if distinct and time remains
            if time.time() < DEADLINE - 10:
                # check distinct
                qx = _lp_polish_strict(best_x, DEADLINE - 2.0, r0=2.0e-3, cap=3000, iuI=iuI, iuJ=iuJ)
                qv = _max_cos(qx)
                if qv < final_best_c0 - 1e-13:
                    final_best_c0 = qv
                    final_best_x = qx.copy()
        except Exception:
            pass
    # no-regression fallback to loaded best.npy
    try:
        b = np.load("best.npy")
        b = np.asarray(b, dtype=float)
        if b.shape == (256, 6):
            b = b / np.linalg.norm(b, axis=1, keepdims=True)
            bv = _max_cos(b)
            if bv < final_best_c0:
                final_best_x = b
                final_best_c0 = bv
    except Exception:
        pass
    final_best_x = final_best_x / np.linalg.norm(final_best_x, axis=1, keepdims=True)
    return final_best_x
