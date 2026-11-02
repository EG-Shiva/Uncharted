import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

def _anneal(x0, steps, T0, T1, lr, mu, rng, deadline=None, noise0=0.0):
    x = x0.copy()
    n, d = x.shape
    vel = np.zeros_like(x)
    G0 = x @ x.T
    np.fill_diagonal(G0, -2.0)
    bv = float(G0.max())
    bx = x.copy()
    lratio = np.log(T1 / T0) if T0 > 0 and T1 > 0 else 0.0
    for k in range(steps):
        if deadline is not None and (k & 127) == 0 and time.time() > deadline:
            break
        frac = k / max(1, steps - 1)
        T = T0 * np.exp(lratio * frac)
        if T < 5e-5:
            T = 5e-5
        lrk = lr * (1.0 - 0.55 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S < 1e-300:
            S = 1e-300
        Grad = (E @ x) / S
        Grad -= (Grad * x).sum(axis=1, keepdims=True) * x
        vel = mu * vel + (1.0 - mu) * Grad
        x = x - lrk * vel
        if noise0 > 0:
            amp = noise0 * (1.0 - frac)
            if amp > 0:
                nz = rng.randn(n, d) * amp
                nz -= (nz * x).sum(axis=1, keepdims=True) * x
                x = x + nz
        x = _norm(x)
        if m < bv:
            bv = m
            bx = x.copy()
    return bx, bv

def _lbfgs(x_init, deadline, taus=(0.0025, 0.0007), iters=(60, 70)):
    x = _norm(x_init.copy())
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.4:
            break
        def fg(yflat, _tau=tau):
            Y = yflat.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m) / _tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yflat)
            L = m + _tau * np.log(s / Np)
            w = e / s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            GY = (GX - (GX * X).sum(axis=1, keepdims=True) * X) / rn
            return float(L), GY.ravel()
        try:
            res = minimize(fg, y, method="L-BFGS-B", jac=True,
                           options={"maxiter": int(maxiter), "maxfun": int(maxiter * 4),
                                    "ftol": 1e-14, "gtol": 1e-9})
            y = res.x
        except Exception:
            break
    Yf = _norm(y.reshape(n, d))
    return Yf, _max_cos(Yf)

def _lp_polish(x, deadline, r0=0.002, cap=1400):
    x = _norm(x.copy())
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x, 1.0
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if int(sel.sum()) == 0:
                sel[int(np.argmax(c))] = True
            if int(sel.sum()) > cap:
                idx = np.argsort(-c)[:cap]
                s2 = np.zeros_like(sel, dtype=bool)
                s2[idx] = True
                sel = s2
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            tmpc = np.empty((p, 2 * d + 1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None] * d + ar
            tmpc[:, d:2 * d] = J[:, None] * d + ar
            tmpc[:, 2 * d] = n * d
            cols = tmpc.ravel()
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2 * d] = x[I]
            cc[:, 2 * d] = -1.0
            vals = cc.ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x, best

def _tangent_bases(x):
    n, d = x.shape
    T = np.empty((n, d, 3))
    for i in range(n):
        a = x[i]
        k = int(np.argmin(np.abs(a)))
        M = np.zeros((d, 3))
        col = 0
        for j in range(d):
            if j == k:
                continue
            e = np.zeros(d); e[j] = 1.0
            e = e - np.dot(e, a) * a
            for c2 in range(col):
                e = e - np.dot(e, M[:, c2]) * M[:, c2]
                e = e - np.dot(e, a) * a
            ne = np.linalg.norm(e)
            if ne < 1e-12:
                e = np.random.randn(d); e = e - np.dot(e, a) * a
                for c2 in range(col):
                    e = e - np.dot(e, M[:, c2]) * M[:, c2]
                ne = np.linalg.norm(e)
                if ne < 1e-12:
                    continue
            e /= ne
            M[:, col] = e
            col += 1
            if col >= 3:
                break
        # orthonormalize once more
        for c2 in range(3):
            for c3 in range(c2):
                M[:, c2] -= np.dot(M[:, c2], M[:, c3]) * M[:, c3]
            M[:, c2] /= max(np.linalg.norm(M[:, c2]), 1e-12)
        T[i] = M
    return T

def _rigidity(x, Ic, Jc, T):
    p = len(Ic)
    n = x.shape[0]
    R = np.zeros((p, n * 3))
    for k in range(p):
        i = int(Ic[k]); j = int(Jc[k])
        vi = T[i].T @ x[j]
        vj = T[j].T @ x[i]
        R[k, i * 3:(i + 1) * 3] = vi
        R[k, j * 3:(j + 1) * 3] = vj
    return R

def _lp_break(x, I_all, J_all, c_all, drop_mask, radius, delta, deadline=None):
    n, d = x.shape
    # drop_mask True => relaxed by delta
    p = len(I_all)
    nv = n * d + 1
    rows = np.repeat(np.arange(p), 2 * d + 1)
    tmpc = np.empty((p, 2 * d + 1), dtype=int)
    ar = np.arange(d)
    tmpc[:, :d] = I_all[:, None] * d + ar
    tmpc[:, d:2 * d] = J_all[:, None] * d + ar
    tmpc[:, 2 * d] = n * d
    cols = tmpc.ravel()
    cc = np.empty((p, 2 * d + 1), dtype=float)
    cc[:, :d] = x[J_all]
    cc[:, d:2 * d] = x[I_all]
    cc[:, 2 * d] = -1.0
    vals = cc.ravel()
    try:
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
    except Exception:
        return None
    b_ub = -c_all.copy()
    b_ub[drop_mask] += float(delta)
    A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
    obj = np.zeros(nv); obj[-1] = 1.0
    try:
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                      bounds=[(-float(radius), float(radius))] * (n * d) + [(None, None)], method="highs")
    except Exception:
        return None
    if res.status != 0:
        return None
    y = x + res.x[:-1].reshape(n, d)
    y = _norm(y)
    return y

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 132.0
    rng = np.random.RandomState(1234567)
    nT = 163
    try:
        bx = np.load("best.npy")
    except Exception:
        try:
            bx = np.load("record.npy")
        except Exception:
            bx = rng.randn(nT, 4)
    bx = np.asarray(bx, dtype=float)
    if bx.shape != (nT, 4) or not np.all(np.isfinite(bx)):
        bx = rng.randn(nT, 4)
    bx = _norm(bx)
    best_x = bx.copy()
    best_val = _max_cos(best_x)
    # quick standard polish of starting point to get true local value
    # analyse base configuration
    n, d = nT, 4
    iu0, iu1 = np.triu_indices(n, 1)
    def contacts_of(x, tol):
        G = x @ x.T
        c = G[iu0, iu1]
        m = float(c.max())
        sel = c > m - tol
        return m, c, sel, iu0[sel], iu1[sel]
    m0, c0, sel_c0, Ic0, Jc0 = contacts_of(best_x, 1e-9)
    # rigidity + flop modes
    try:
        Tb = _tangent_bases(best_x)
        Rb = _rigidity(best_x, Ic0, Jc0, Tb)
        # smallest singular vectors
        try:
            U, S, Vt = np.linalg.svd(Rb, full_matrices=False)
            flop_vec = Vt[-1].copy()
            flop2 = Vt[-2].copy() if Vt.shape[0] >= 2 else flop_vec.copy()
            smin = float(S[-1])
        except Exception:
            flop_vec = rng.randn(n * 3); flop2 = rng.randn(n * 3); smin = 0.0
    except Exception:
        flop_vec = rng.randn(n * 3); flop2 = rng.randn(n * 3)
    flop_score = np.zeros(n)
    flop2_score = np.zeros(n)
    for i in range(n):
        flop_score[i] = float(np.linalg.norm(flop_vec[i * 3:(i + 1) * 3]))
        flop2_score[i] = float(np.linalg.norm(flop2[i * 3:(i + 1) * 3]))
    # second tier
    _, c_all0, _, _, _ = contacts_of(best_x, 2e-4)
    m_all = m0
    sel_all = c_all0 > m_all - 2e-4
    I_all0 = iu0[sel_all]; J_all0 = iu1[sel_all]; c_all_vals = c_all0[sel_all]
    # degrees on tight contacts
    deg = np.zeros(n, dtype=int)
    for a, b in zip(Ic0, Jc0):
        deg[a] += 1; deg[b] += 1
    # pair importance proxy
    pair_imp = {}
    # map pair index in I_all to importance
    imp_all = np.zeros(len(I_all0))
    for k in range(len(I_all0)):
        imp_all[k] = float(deg[int(I_all0[k])] + deg[int(J_all0[k])])
    # contact index set for fast lookup of tight contacts
    tight_set = set(zip(Ic0.tolist(), Jc0.tolist()))
    # build adjacency for community expansion
    adj = [[] for _ in range(n)]
    for a, b in zip(Ic0.tolist(), Jc0.tolist()):
        adj[a].append(b); adj[b].append(a)
    # generate drop sets (indices into I_all/J_all)
    pair_to_all = {(int(I_all0[k]), int(J_all0[k])): k for k in range(len(I_all0))}
    drop_sets = []
    hub_pts = np.argsort(-deg)[:10]
    # 1) clustered on one point + neighbours
    for h in hub_pts[:8]:
        cand = []
        neigh = set([int(h)] + adj[int(h)])
        for k in range(len(I_all0)):
            a = int(I_all0[k]); b = int(J_all0[k])
            if a == int(h) or b == int(h):
                cand.append(k)
        # expand to neighbours
        extra = []
        for k in range(len(I_all0)):
            a = int(I_all0[k]); b = int(J_all0[k])
            if k in cand:
                continue
            if a in neigh or b in neigh:
                extra.append(k)
        # sort extra by imp
        extra.sort(key=lambda k: -imp_all[k])
        need = 24 - len(cand)
        if need > 0:
            cand = cand + extra[:need]
        if len(cand) > 40:
            # keep highest imp
            cand.sort(key=lambda k: -imp_all[k])
            cand = cand[:32]
        if len(cand) >= 12:
            drop_sets.append(sorted(cand))
    # 2) community (2-hop internal)
    for h in hub_pts[:4]:
        # BFS 2 hops
        seen = set([int(h)])
        frontier = set([int(h)])
        for _ in range(2):
            nf = set()
            for u in frontier:
                for w in adj[u]:
                    if w not in seen:
                        seen.add(w); nf.add(w)
            frontier = nf
        cand = [k for k in range(len(I_all0)) if int(I_all0[k]) in seen and int(J_all0[k]) in seen]
        cand.sort(key=lambda k: -imp_all[k])
        if len(cand) > 40:
            cand = cand[:32]
        if len(cand) >= 12:
            # ensure 15-40: pad if small
            if len(cand) < 15:
                # pad with incident
                add = [k for k in range(len(I_all0)) if k not in cand and (int(I_all0[k]) in seen or int(J_all0[k]) in seen)]
                add.sort(key=lambda k: -imp_all[k])
                cand = cand + add[:15 - len(cand)]
            drop_sets.append(sorted(cand))
    # 3) flop support
    for topk in (12, 18, 24):
        order = np.argsort(-(flop_score + 0.5 * flop2_score))[:topk]
        sset = set(order.tolist())
        cand_both = [k for k in range(len(I_all0)) if int(I_all0[k]) in sset and int(J_all0[k]) in sset]
        cand_one = [k for k in range(len(I_all0)) if (int(I_all0[k]) in sset) ^ (int(J_all0[k]) in sset)]
        cand_one.sort(key=lambda k: -imp_all[k])
        cand = cand_both + cand_one
        if len(cand) > 40:
            cand = cand[:32]
        if len(cand) < 15:
            # pad with top imp
            o = np.argsort(-imp_all)
            for k in o:
                if k not in cand:
                    cand.append(int(k))
                if len(cand) >= 20:
                    break
        drop_sets.append(sorted(cand[:32]))
    # deduplicate
    uniq = []
    seen_ds = set()
    for ds in drop_sets:
        t = tuple(ds)
        if t not in seen_ds:
            seen_ds.add(t)
            uniq.append(ds)
    drop_sets = uniq[:20]

    # archive retaining by score and changed count
    base_tight = set(zip(Ic0.tolist(), Jc0.tolist()))
    def changed_count(x_new):
        m, _, _, I2, J2 = contacts_of(x_new, 1e-7)
        s2 = set(zip(I2.tolist(), J2.tolist()))
        inter = len(base_tight.intersection(s2))
        return int(len(base_tight) + len(s2) - 2 * inter), m
    cands = []  # (score, changed, x)
    radii = [0.01, 0.03, 0.06]
    deltas = [0.008, 0.012]
    # limit total LP-break attempts by time
    att = 0
    for ds in drop_sets:
        if time.time() > SEARCH_END - 25:
            break
        for radius in radii:
            if time.time() > SEARCH_END - 25:
                break
            delta = deltas[0] if radius < 0.02 else deltas[1]
            drop_mask = np.zeros(len(I_all0), dtype=bool)
            drop_mask[np.array(ds, dtype=int)] = True
            y0 = _lp_break(best_x, I_all0, J_all0, c_all_vals, drop_mask, radius, delta)
            if y0 is None:
                continue
            # quick repair: short anneal + lbfgs
            dl = min(time.time() + 3.0, SEARCH_END)
            try:
                yr, vr = _anneal(y0, steps=900, T0=0.006, T1=0.0004, lr=0.035, mu=0.86, rng=rng, deadline=dl, noise0=0.001)
            except Exception:
                continue
            try:
                yl, vl = _lbfgs(yr, deadline=dl)
                if vl < vr:
                    yr, vr = yl, vl
            except Exception:
                pass
            try:
                ch, _ = changed_count(yr)
            except Exception:
                ch = 0
            cands.append((float(vr), int(ch), yr.copy()))
            att += 1
            if vr < best_val - 1e-12:
                # polish immediately
                try:
                    yp, vp = _lp_polish(yr, deadline=min(time.time() + 4.0, DEADLINE - 5))
                    ch2, _ = changed_count(yp)
                    cands.append((float(vp), int(ch2), yp.copy()))
                    if vp < best_val:
                        best_val = float(vp); best_x = yp.copy()
                except Exception:
                    best_val = float(vr); best_x = yr.copy()
            if att >= 45:
                break
        if att >= 45:
            break
    # sort cands: prefer low score, but keep bifurcated
    cands.sort(key=lambda t: t[0])
    # select diverse bifurcated set: high changed even if slightly worse
    bifur = [c for c in cands if c[1] >= 25]
    bifur.sort(key=lambda t: t[0])
    # extended polish on top few bifurcated + top overall
    to_polish = []
    seen_ids = set()
    for c in cands[:4]:
        to_polish.append(c)
    for c in bifur[:4]:
        # avoid dup
        if all(np.linalg.norm(c[2] - q[2]) > 1e-6 for q in to_polish):
            to_polish.append(c)
    for (v0, ch0, x0) in to_polish:
        if time.time() > DEADLINE - 22:
            break
        if v0 > best_val + 0.004 and ch0 < 40:
            continue
        try:
            dl = min(time.time() + 6.0, DEADLINE - 8)
            xa, va = _anneal(x0, steps=1600, T0=0.005, T1=0.00025, lr=0.034, mu=0.88, rng=rng, deadline=dl, noise0=0.0006)
            xb, vb = _lbfgs(xa, deadline=dl, taus=(0.002, 0.0006, 0.0003), iters=(60, 70, 80))
            if vb < va:
                xa, va = xb, vb
            xp, vp = _lp_polish(xa, deadline=dl, r0=0.002)
            if vp < best_val - 1e-12:
                best_val = float(vp); best_x = xp.copy()
            elif va < best_val - 1e-12:
                best_val = float(va); best_x = xa.copy()
        except Exception:
            continue
    # final standard polish of best
    try:
        if time.time() < DEADLINE - 10:
            best_x, best_val = _lp_polish(best_x, deadline=DEADLINE - 2, r0=0.003)
    except Exception:
        pass
    # fallback: if we never improved, still return polished best
    best_x = _norm(best_x)
    return np.asarray(best_x, dtype=float)
