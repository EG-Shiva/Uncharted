import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish(x, seconds, init_radius=3e-3, max_iters=40):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = float(init_radius)
    start = time.time()
    iuI, iuJ = np.triu_indices(n, 1)
    it = 0
    while time.time() - start < seconds and radius > 1e-14 and it < max_iters:
        it += 1
        if time.time() - start >= seconds:
            break
        G = x @ x.T
        c = G[iuI, iuJ]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        cnt = int(sel.sum())
        if cnt == 0:
            break
        if cnt > 1500:
            idx = np.where(sel)[0]
            cidx = c[idx]
            part = np.argpartition(cidx, -1500)[-1500:]
            sel = np.zeros_like(sel, dtype=bool)
            sel[idx[part]] = True
            cnt = 1500
        I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty(p * (2 * d + 1), dtype=np.int64)
        vals = np.empty(p * (2 * d + 1), dtype=np.float64)
        Cb = cols.reshape(p, 2 * d + 1)
        Vb = vals.reshape(p, 2 * d + 1)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None] * d + ar
        Cb[:, d:2*d] = J[:, None] * d + ar
        Cb[:, 2*d] = n * d
        Vb[:, :d] = x[J]
        Vb[:, d:2*d] = x[I]
        Vb[:, 2*d] = -1.0
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            x = y; best = v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def _forced_step(x, FI, FJ, delta, radius=0.03, cap=1500):
    n, d = x.shape
    iuI, iuJ = np.triu_indices(n, 1)
    G = x @ x.T
    c = G[iuI, iuJ]
    m = float(c.max())
    # normal set: near-max, excluding forced
    thresh = m - 0.005
    sel = c > thresh
    # remove forced pairs from normal set
    if len(FI) > 0:
        fset = set(zip([int(a) for a in FI], [int(a) for a in FJ]))
        # build mask quickly: mark pairs that are forced
        # since sizes small, loop in python is fine (<=52000)
        # use dictionary lookup via pair index
        # create boolean for selected that overlaps
        # faster: for each forced, find its linear index via search only among sel
        # brute: iterate over sel indices
        selI = iuI[sel]; selJ = iuJ[sel]
        # build set for O(1)
        keep = np.ones(int(sel.sum()), dtype=bool)
        # map pair->pos using dict of selected
        pos = { (int(a),int(b)): k for k,(a,b) in enumerate(zip(selI.tolist(), selJ.tolist())) }
        for a,b in fset:
            if (a,b) in pos:
                keep[pos[(a,b)]] = False
        # rebuild sel
        tmp_idx = np.where(sel)[0][keep]
        sel = np.zeros_like(sel, dtype=bool)
        sel[tmp_idx] = True
    cnt = int(sel.sum())
    if cnt > cap:
        idx = np.where(sel)[0]
        cidx = c[idx]
        part = np.argpartition(cidx, -cap)[-cap:]
        sel = np.zeros_like(sel, dtype=bool)
        sel[idx[part]] = True
        cnt = cap
    I = iuI[sel]; J = iuJ[sel]; csel = c[sel]
    # forced cosines
    if len(FI) > 0:
        cf = np.sum(x[FI] * x[FJ], axis=1)
    else:
        cf = np.zeros(0)
    p = cnt + len(FI)
    if p == 0:
        return None
    nv = n * d + 1
    rows_list = []
    cols_list = []
    vals_list = []
    b_ub = np.empty(p)
    # normal rows
    # build via vectorized blocks like polish
    nn = cnt
    nf = len(FI)
    total_entries = nn * (2*d+1) + nf * (2*d)
    rows = np.empty(total_entries, dtype=np.int64)
    cols = np.empty(total_entries, dtype=np.int64)
    vals = np.empty(total_entries, dtype=np.float64)
    off = 0
    if nn > 0:
        r = np.repeat(np.arange(nn), 2*d+1)
        rows[off:off+nn*(2*d+1)] = r
        Cb = np.empty((nn, 2*d+1), dtype=np.int64)
        Vb = np.empty((nn, 2*d+1), dtype=np.float64)
        ar = np.arange(d)
        Cb[:, :d] = I[:, None]*d + ar
        Cb[:, d:2*d] = J[:, None]*d + ar
        Cb[:, 2*d] = n*d
        Vb[:, :d] = x[J]
        Vb[:, d:2*d] = x[I]
        Vb[:, 2*d] = -1.0
        cols[off:off+nn*(2*d+1)] = Cb.ravel()
        vals[off:off+nn*(2*d+1)] = Vb.ravel()
        off += nn*(2*d+1)
        b_ub[:nn] = -csel
    if nf > 0:
        FIa = np.asarray(FI, dtype=np.int64); FJa = np.asarray(FJ, dtype=np.int64)
        r2 = np.repeat(nn + np.arange(nf), 2*d)
        rows[off:off+nf*2*d] = r2
        Cb2 = np.empty((nf, 2*d), dtype=np.int64)
        Vb2 = np.empty((nf, 2*d), dtype=np.float64)
        ar = np.arange(d)
        Cb2[:, :d] = FIa[:, None]*d + ar
        Cb2[:, d:2*d] = FJa[:, None]*d + ar
        Vb2[:, :d] = x[FJa]
        Vb2[:, d:2*d] = x[FIa]
        cols[off:off+nf*2*d] = Cb2.ravel()
        vals[off:off+nf*2*d] = Vb2.ravel()
        off += nf*2*d
        b_ub[nn:] = (m - delta) - cf
    try:
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
    except Exception:
        return None
    obj = np.zeros(nv); obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
    except Exception:
        return None
    if res.status != 0:
        return None
    y = x + res.x[:-1].reshape(n, d)
    yn = np.linalg.norm(y, axis=1, keepdims=True)
    yn[yn == 0] = 1.0
    y = y / yn
    return y

def _gd_relax(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    for s in range(steps):
        f = s / max(1, steps - 1)
        tau = tau0 * (tau1 / tau0) ** f
        lr = lr0 * (lr1 / lr0) ** f
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = float(G.max())
        E = np.exp((G - m) / tau)
        np.fill_diagonal(E, 0.0)
        Gr = E @ x
        proj = np.sum(Gr * x, axis=1, keepdims=True)
        Gt = Gr - proj * x
        nr = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        Gt = Gt / nr
        w = np.sum(E, axis=1, keepdims=True)
        mw = float(np.mean(w)) + 1e-12
        w = np.clip(w / mw, 0.3, 3.0)
        x = x - lr * w * Gt
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def solve():
    t0 = time.time()
    BUDGET = 168.0
    rng = np.random.default_rng(7)
    best_x = None
    best_v = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (323, 7):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.any(~np.isfinite(a)):
                    continue
                v = _max_cos(a)
                if v < best_v:
                    best_v = v
                    best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        a = rng.standard_normal(size=(323, 7))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        best_x = a
        best_v = _max_cos(a)
    x0 = best_x.copy()
    deltas = [3e-4, 1e-3, 3e-3]
    n_trials = 40
    iuI_all, iuJ_all = np.triu_indices(323, 1)
    for trial in range(n_trials):
        if time.time() - t0 > BUDGET - 16.0:
            break
        # analyze current best topology
        G0 = best_x @ best_x.T
        np.fill_diagonal(G0, -2.0)
        m0 = float(G0.max())
        cc = np.sum(G0 > m0 - 1e-9, axis=1)
        hub_order = np.argsort(-cc, kind="stable")
        # tight pairs for forcing (slack < 3e-3)
        c_all = G0[iuI_all, iuJ_all]
        tight_mask = c_all > m0 - 0.003
        tI = iuI_all[tight_mask]; tJ = iuJ_all[tight_mask]
        tc = c_all[tight_mask]
        # sort tight by cosine desc, then rotate
        o = np.argsort(-tc, kind="stable")
        tI = tI[o]; tJ = tJ[o]; tc = tc[o]
        delta = deltas[trial % 3]
        FI = []; FJ = []
        if trial % 2 == 0:
            # matching mode: greedy matching over rotated order
            want = 36
            shift = (trial * 131) % max(1, len(tI))
            used = np.zeros(323, dtype=bool)
            # iterate cyclically
            for k in range(len(tI)):
                idx = (shift + k) % len(tI)
                a = int(tI[idx]); b = int(tJ[idx])
                if not used[a] and not used[b]:
                    FI.append(a); FJ.append(b)
                    used[a] = True; used[b] = True
                    if len(FI) >= want:
                        break
        else:
            # hub-incident mode: pick 2-3 overloaded hubs rotating
            nh = 2 + (trial % 2)
            start_h = ((trial // 2) * 3) % 323
            hubs = []
            for k in range(nh):
                hubs.append(int(hub_order[(start_h + k) % 323]))
            hset = set(hubs)
            want = 40
            shift = (trial * 57) % max(1, len(tI))
            for k in range(len(tI)):
                idx = (shift + k) % len(tI)
                a = int(tI[idx]); b = int(tJ[idx])
                if a in hset or b in hset:
                    FI.append(a); FJ.append(b)
                    if len(FI) >= want:
                        break
            if len(FI) == 0:
                # fallback to top pairs matching
                used = np.zeros(323, dtype=bool)
                for k in range(len(tI)):
                    a = int(tI[k]); b = int(tJ[k])
                    if not used[a] and not used[b]:
                        FI.append(a); FJ.append(b)
                        used[a] = True; used[b] = True
                        if len(FI) >= 30:
                            break
        FI = np.array(FI, dtype=np.int64); FJ = np.array(FJ, dtype=np.int64)
        if len(FI) == 0:
            continue
        rad = 0.03 if delta >= 1e-3 else 0.02
        y = _forced_step(best_x, FI, FJ, delta, radius=rad)
        if y is None:
            # infeasible: tangential kick on forced vertices
            y = best_x.copy()
            uniq = np.unique(np.concatenate([FI, FJ]))
            for k in uniq[:40]:
                nz = rng.standard_normal(size=(7,)) * rad * 0.5
                nz = nz - np.dot(nz, y[k]) * y[k]
                y[k] = y[k] + nz
                y[k] /= np.linalg.norm(y[k])
        # fast vectorised LogSumExp tangential descent (GX matrix products)
        # alternate annealing slightly per trial
        if trial % 2 == 0:
            y = _gd_relax(y, steps=220, tau0=0.02, tau1=0.003, lr0=0.02, lr1=0.002)
        else:
            y = _gd_relax(y, steps=220, tau0=0.03, tau1=0.005, lr0=0.025, lr1=0.003)
        # re-polish without forcing
        remain = BUDGET - (time.time() - t0)
        if remain < 6:
            break
        y = _lp_polish(y, min(2.5, remain - 4.0), init_radius=3e-3, max_iters=25)
        v = _max_cos(y)
        if v < best_v - 1e-12:
            best_v = v
            best_x = y.copy()
        # else discard (systematic rotation continues from best)
        # small deterministic jitter of order for next matching comes from rng-free shifts above
    remain = BUDGET - (time.time() - t0)
    if remain > 3.0:
        best_x = _lp_polish(best_x, min(remain - 1.0, 9.0), init_radius=3e-3, max_iters=60)
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
