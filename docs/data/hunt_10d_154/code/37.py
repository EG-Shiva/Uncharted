import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _lp_polish_capped(x, t_end, radius0=1e-3):
    x = _norm(np.asarray(x, dtype=float))
    n, d = x.shape
    best = _max_cos(x)
    cur_best = x.copy()
    radius = radius0
    iu = np.triu_indices(n, 1)
    while time.time() < t_end - 0.4 and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(),
                               (np.repeat(np.arange(n), d), np.arange(n * d))),
                              shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v
            cur_best = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return cur_best, best

def _dual_masses(X):
    n, d = X.shape
    try:
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        iu = np.triu_indices(n, 1)
        c = G[iu]
        sel = c > m - 1e-7
        I = iu[0][sel]; J = iu[1][sel]
        p = len(I)
        if p == 0:
            return (G > m - 1e-7).sum(axis=1).astype(float)
        radius = 1e-3
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(),
                           (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
        res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                      b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                      method="highs")
        if res.status != 0:
            raise RuntimeError("lp fail")
        marg = None
        try:
            il = getattr(res, "ineqlin", None)
            if il is not None:
                if hasattr(il, "marginals"):
                    marg = np.asarray(il.marginals, dtype=float)
                elif isinstance(il, dict) and "marginals" in il:
                    marg = np.asarray(il["marginals"], dtype=float)
            if marg is None and hasattr(res, "marginals"):
                marg = np.asarray(res.marginals, dtype=float)
            if marg is None:
                try:
                    dres = dict(res)
                    if "ineqlin" in dres:
                        v = dres["ineqlin"]
                        if isinstance(v, dict) and "marginals" in v:
                            marg = np.asarray(v["marginals"], dtype=float)
                        elif hasattr(v, "marginals"):
                            marg = np.asarray(v.marginals, dtype=float)
                except Exception:
                    pass
        except Exception:
            marg = None
        if marg is None or marg.shape[0] != p:
            raise RuntimeError("no marginals")
        dual = np.zeros(n)
        am = np.abs(marg)
        # aggregate by point
        for k in range(p):
            a = am[k]
            dual[I[k]] += a
            dual[J[k]] += a
        if float(dual.sum()) <= 0 or not np.all(np.isfinite(dual)):
            raise RuntimeError("zero dual")
        return dual
    except Exception:
        try:
            G2 = X @ X.T
            np.fill_diagonal(G2, -2.0)
            m2 = float(G2.max())
            deg = (G2 > m2 - 1e-7).sum(axis=1).astype(float)
            return deg
        except Exception:
            return np.ones(X.shape[0]))

def _select_clusters(X, dual, n_out=6):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    adj = [set() for _ in range(n)]
    iu = np.triu_indices(n, 1)
    c = G[iu]
    sel = c > m - 1e-7
    I = iu[0][sel]; J = iu[1][sel]
    for a, b in zip(I.tolist(), J.tolist()):
        adj[a].add(b); adj[b].add(a)
    seeds = np.argsort(-dual, kind="stable")[:24].tolist()
    cand = []
    for si, s in enumerate(seeds):
        target = 12 + (si * 5 + 3) % 7  # 12..18
        cl = set([int(s)])
        # grow greedily by dual
        while len(cl) < target:
            frontier = set()
            for u in cl:
                for v in adj[u]:
                    if v not in cl:
                        frontier.add(v)
            if not frontier:
                break
            bv = max(frontier, key=lambda v: dual[v])
            cl.add(int(bv))
        if len(cl) < 12:
            continue
        tot = float(sum(dual[list(cl)]))
        cand.append((tot, tuple(sorted(cl))))
    # dedup + sort
    cand.sort(key=lambda t: -t[0])
    picked = []
    for tot, tup in cand:
        s = set(tup)
        ok = True
        for _, p2 in picked:
            inter = len(s.intersection(set(p2)))
            if inter > 8:
                ok = False
                break
        if ok:
            picked.append((tot, tup))
        if len(picked) >= n_out:
            break
    return [np.array(t, dtype=int) for _, t in picked]

def _null_basis(X, cl_idx, m_val, thr=1e-7):
    C = len(cl_idx)
    d = X.shape[1]
    pos = {int(g): k for k, g in enumerate(cl_idx.tolist())}
    # norm rows
    rows = []
    for k, g in enumerate(cl_idx.tolist()):
        r = np.zeros(C * d)
        r[k * d:(k + 1) * d] = X[g]
        rows.append(r)
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    sel = c > m_val - thr
    I = iu[0][sel]; J = iu[1][sel]
    for a, b in zip(I.tolist(), J.tolist()):
        ain = int(a) in pos
        bin_ = int(b) in pos
        if not ain and not bin_:
            continue
        r = np.zeros(C * d)
        if ain and bin_:
            la = pos[int(a)]; lb = pos[int(b)]
            r[la * d:(la + 1) * d] = X[b]
            r[lb * d:(lb + 1) * d] = X[a]
            rows.append(r)
        else:
            if ain:
                la = pos[int(a)]
                r[la * d:(la + 1) * d] = X[b]
            else:
                lb = pos[int(b)]
                r[lb * d:(lb + 1) * d] = X[a]
            rows.append(r)
    J = np.stack(rows, axis=0) if len(rows) > 0 else np.zeros((0, C * d))
    try:
        if J.shape[0] == 0:
            raise RuntimeError("empty")
        _, svals, vh = np.linalg.svd(J, full_matrices=True)
        rank = int((svals > 1e-6).sum())
        null = vh[rank:, :]
        if null.shape[0] == 0:
            raise RuntimeError("no null")
        return null
    except Exception:
        return None

def _cluster_opt(Xc_init, Xf, old_int_mask, old_cross_mask, Nold, steps=650, lr=0.05, T0=0.02, T1=0.003, lam=8.0):
    C, d = Xc_init.shape
    Xc = Xc_init.copy()
    m_ad = np.zeros_like(Xc); v_ad = np.zeros_like(Xc)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p = 1.0; b2p = 1.0
    if steps > 1:
        logr = np.log(max(T1, 1e-12) / max(T0, 1e-12)) / (steps - 1)
    else:
        logr = 0.0
    if Nold is None or Nold <= 0:
        Nold = 1
        lam_use = 0.0
    else:
        lam_use = lam
    for t in range(1, steps + 1):
        T = T0 * np.exp(logr * (t - 1))
        if T < 1e-12:
            T = 1e-12
        Gc = Xc @ Xc.T
        Gx = Xc @ Xf.T
        # max for stability (ignore diag of Gc)
        np.fill_diagonal(Gc, -2.0)
        mv = Gc.max()
        if Gx.size > 0:
            mv = max(float(mv), float(Gx.max()))
        Ei = np.exp((Gc - mv) / T)
        np.fill_diagonal(Ei, 0.0)
        Ex = np.exp((Gx - mv) / T) if Gx.size > 0 else np.zeros((C, 0))
        S = float(Ei.sum() * 0.5 + Ex.sum())
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (Ei @ Xc) / S
        if Gx.size > 0:
            Grad = Grad + (Ex @ Xf) / S
        # penalty grad preventing old contacts returning above 0.38
        if lam_use != 0.0:
            # recompute Gc with diag restored? need dots; Gc diag is -2 now, fix
            # Gc_full for penalty: use Xc@Xc.T again? reuse with diag=1
            # instead compute penalty from stored G (recompute quickly)
            Gcf = Xc @ Xc.T
            Gxf = Xc @ Xf.T
            Ri = np.zeros_like(Gcf)
            Rc = np.zeros_like(Gxf)
            if old_int_mask is not None:
                D = Gcf - 0.38
                D[D < 0] = 0.0
                Ri = D * old_int_mask.astype(float)
                np.fill_diagonal(Ri, 0.0)
            if old_cross_mask is not None and Gxf.size > 0:
                Dc = Gxf - 0.38
                Dc[Dc < 0] = 0.0
                Rc = Dc * old_cross_mask.astype(float)
            Gpen = (Ri @ Xc) / Nold
            if Gxf.size > 0:
                Gpen = Gpen + (Rc @ Xf) / Nold
            Grad = Grad + (2.0 * lam_use) * Gpen
        dot = (Grad * Xc).sum(axis=1, keepdims=True)
        Gt = Grad - dot * Xc
        b1p *= b1; b2p *= b2
        m_ad = b1 * m_ad + (1.0 - b1) * Gt
        v_ad = b2 * v_ad + (1.0 - b2) * (Gt * Gt)
        mhat = m_ad / (1.0 - b1p)
        vhat = v_ad / (1.0 - b2p)
        step = mhat / (np.sqrt(vhat) + eps)
        Xc = Xc - lr * step
        Xc = _norm(Xc)
    return Xc

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    rng = np.random.default_rng(7)
    n, d = 154, 10
    bestX = None; bestVal = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape == (n, d):
                a = _norm(a.astype(float))
                v = _max_cos(a)
                if v < bestVal:
                    bestVal = v; bestX = a.copy()
        except Exception:
            pass
    if bestX is None:
        bestX = _norm(rng.normal(size=(n, d)))
        bestVal = _max_cos(bestX)
    # safety: ensure finite
    bestX = _norm(bestX)
    bestVal = _max_cos(bestX)
    tried = set()
    # main cycling
    while time.time() < DEAD - 12.0:
        dual = _dual_masses(bestX)
        clusters = _select_clusters(bestX, dual, n_out=6)
        if len(clusters) == 0:
            break
        improved_outer = False
        for cl in clusters:
            if time.time() > DEAD - 11.0:
                break
            key = tuple(sorted(cl.tolist()))
            if key in tried:
                continue
            tried.add(key)
            C = len(cl)
            # old contacts for penalty
            G0 = bestX @ bestX.T
            np.fill_diagonal(G0, -2.0)
            m0 = float(G0.max())
            old_pairs = []
            # map frozen
            all_idx = np.arange(n)
            mask = np.ones(n, dtype=bool); mask[cl] = False
            f_idx = all_idx[mask]
            pos = {int(g): k for k, g in enumerate(cl.tolist())}
            fpos = {int(g): k for k, g in enumerate(f_idx.tolist())}
            old_int = np.zeros((C, C), dtype=bool)
            old_cross = np.zeros((C, len(f_idx)), dtype=bool)
            iu2 = np.triu_indices(n, 1)
            c2 = G0[iu2]
            sel2 = c2 > m0 - 1e-7
            I2 = iu2[0][sel2]; J2 = iu2[1][sel2]
            for a, b in zip(I2.tolist(), J2.tolist()):
                ain = int(a) in pos; bin_ = int(b) in pos
                if ain and bin_:
                    la = pos[int(a)]; lb = pos[int(b)]
                    old_int[la, lb] = True; old_int[lb, la] = True
                    old_pairs.append((int(a), int(b)))
                elif ain:
                    la = pos[int(a)]; fb = fpos[int(b)]
                    old_cross[la, fb] = True
                    old_pairs.append((int(a), int(b)))
                elif bin_:
                    lb = pos[int(b)]; fa = fpos[int(a)]
                    old_cross[lb, fa] = True
                    old_pairs.append((int(a), int(b)))
            Nold = int(old_int.sum() // 2 + old_cross.sum())
            if Nold == 0:
                Nold = 1
            # null basis
            null = _null_basis(bestX, cl, m0, thr=1e-7)
            Xc0 = bestX[cl].copy()
            Xf = bestX[f_idx].copy()
            # sample several displacements
            n_samp = 4
            angs = rng.uniform(0.05, 0.15, size=n_samp)
            best_pre = None; best_pre_val = 1e9
            for s in range(n_samp):
                ang = float(angs[s])
                if null is not None and null.shape[0] > 0:
                    coeff = rng.normal(size=(null.shape[0],))
                    nv = float(np.linalg.norm(coeff))
                    if nv == 0 or not np.isfinite(nv):
                        coeff = np.ones(null.shape[0],) / np.sqrt(null.shape[0])
                    else:
                        coeff = coeff / nv
                    vec = coeff @ null  # (C*d,)
                    V = vec.reshape(C, d)
                    # scale to RMS ang
                    fn_ = float(np.linalg.norm(V))
                    if fn_ == 0 or not np.isfinite(fn_):
                        V = rng.normal(size=(C, d))
                        V = V - (V * Xc0).sum(axis=1, keepdims=True) * Xc0
                        fn_ = float(np.linalg.norm(V))
                    target_norm = ang * np.sqrt(C)
                    V = V * (target_norm / max(fn_, 1e-12))
                else:
                    V = rng.normal(size=(C, d))
                    V = V - (V * Xc0).sum(axis=1, keepdims=True) * Xc0
                    fn_ = float(np.linalg.norm(V))
                    target_norm = ang * np.sqrt(C)
                    V = V * (target_norm / max(fn_, 1e-12))
                Xc_init = _norm(Xc0 + V)
                # reoptimise only cluster with softmax + penalty
                try:
                    Xc_opt = _cluster_opt(Xc_init, Xf, old_int, old_cross, Nold,
                                          steps=650, lr=0.05, T0=0.02, T1=0.003, lam=8.0)
                except Exception:
                    continue
                # evaluate cluster-involved max (pre-LP) to pick best sample
                try:
                    Xfull_pre = bestX.copy(); Xfull_pre[cl] = Xc_opt
                    Gp = Xfull_pre @ Xfull_pre.T
                    np.fill_diagonal(Gp, -2.0)
                    # max over pairs involving cluster (fast proxy)
                    # just use global max
                    vp = float(Gp.max())
                except Exception:
                    continue
                if vp < best_pre_val:
                    best_pre_val = vp
                    best_pre = Xfull_pre.copy()
                if time.time() > DEAD - 10.0:
                    break
            if best_pre is None:
                continue
            # unfreeze: strictly time-capped LP polish ~6s
            lp_budget_end = min(time.time() + 6.0, DEAD - 4.0)
            if lp_budget_end - time.time() < 1.0:
                continue
            try:
                polished, pval = _lp_polish_capped(best_pre, lp_budget_end)
            except Exception:
                continue
            if pval < bestVal - 5e-9:
                bestVal = pval
                bestX = polished.copy()
                improved_outer = True
                break  # recompute duals/clusters from new best
            # else keep best fallback (do nothing)
        if not improved_outer:
            # if no improvement in this round, try new random dual jitter or break if out of time
            # clear tried partially to allow different sizes? keep going if time left but force new seeds
            # if we exhausted all clusters without improvement, perturb tried to allow re-try with different angs (limit)
            if len(tried) > 24 or time.time() > DEAD - 12.0:
                break
            # continue loop will recompute same clusters -> would skip due to tried; so clear oldest? just break to avoid waste
            # instead do one small random restart of worst cluster? To use time, clear tried and continue with new rng state
            # but to avoid infinite loop, only allow one more round
            if time.time() < DEAD - 30.0:
                tried = set()
                continue
            else:
                break
    # final squeeze if time remains
    if time.time() < DEAD - 3.0:
        try:
            t_end = min(DEAD - 1.0, time.time() + 4.0)
            bestX, bestVal = _lp_polish_capped(bestX, t_end)
        except Exception:
            pass
    bestX = _norm(bestX)
    return bestX
