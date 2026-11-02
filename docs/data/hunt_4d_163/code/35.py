import time
import numpy as np
from scipy.optimize import linprog, minimize, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _per_point_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), float(G.max())

def _lp_polish(x, deadline, r0=1e-3):
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iu0, iu1 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        try:
            G = x @ x.T
            c = G[iu0, iu1]
            m = float(c.max())
            sel = c > m - max(20.0 * radius, 1e-9)
            if np.count_nonzero(sel) == 0:
                sel[int(np.argmax(c))] = True
            I = iu0[sel]; J = iu1[sel]
            csel = c[sel]
            p = len(I)
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cc = np.empty((p, 2 * d + 1), dtype=float)
            cc[:, :d] = x[J]
            cc[:, d:2*d] = x[I]
            cc[:, 2*d] = -1.0
            tmpc = np.empty((p, 2*d+1), dtype=int)
            ar = np.arange(d)
            tmpc[:, :d] = I[:, None]*d + ar
            tmpc[:, d:2*d] = J[:, None]*d + ar
            tmpc[:, 2*d] = n*d
            A_ub = coo_matrix((cc.ravel(), (rows, tmpc.ravel())), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n*d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)]*(n*d)+[(None, None)], method="highs")
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius*1.5, 0.05)
            else:
                radius *= 0.5
        except Exception:
            radius *= 0.5
            continue
    return x

def _smooth(Xi, deadline, taus=(0.004, 0.0012, 0.0005), iters=(80, 80, 100)):
    x = Xi / np.maximum(np.linalg.norm(Xi, axis=1, keepdims=True), 1e-12)
    n, d = x.shape
    iu0, iu1 = np.triu_indices(n, 1)
    Np = len(iu0)
    y = x.ravel().copy()
    for tau, maxiter in zip(taus, iters):
        if time.time() > deadline - 0.2:
            break
        def fg(yf, _tau=tau):
            Y = yf.reshape(n, d)
            rn = np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
            X = Y / rn
            g = (X @ X.T)[iu0, iu1]
            m = float(g.max())
            e = np.exp((g - m)/_tau)
            s = float(e.sum())
            if s <= 0 or not np.isfinite(s):
                return float(m), np.zeros_like(yf)
            L = m + _tau*np.log(s/Np)
            w = e/s
            W = np.zeros((n, n))
            W[iu0, iu1] = w
            W = W + W.T
            GX = W @ X
            GY = (GX - (GX*X).sum(axis=1, keepdims=True)*X)/rn
            return float(L), GY.ravel()
        try:
            r = minimize(fg, y, method="L-BFGS-B", jac=True,
                         options={"maxiter": int(maxiter), "maxfun": int(maxiter*4), "ftol":1e-14, "gtol":1e-8})
            y = r.x
        except Exception:
            break
    Yf = y.reshape(n, d)
    Yf /= np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
    return Yf

def _polish_short(X, deadline):
    # split deadline between smooth and LP
    now = time.time()
    rem = deadline - now
    if rem <= 0.05:
        return X
    s_dead = now + rem*0.6
    try:
        X = _smooth(X, s_dead)
    except Exception:
        pass
    try:
        X = _lp_polish(X, deadline, r0=1e-3)
    except Exception:
        pass
    return X

def _hole(Xf, rng, ns=20000):
    Y = rng.standard_normal((ns, 4))
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    D = Y @ Xf.T
    vals = D.max(axis=1)
    i = int(np.argmin(vals))
    return Y[i], float(vals[i])

def _graph_hash(X, mval=None):
    try:
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        if mval is None:
            mval = float(G.max())
        thr = mval - 1e-9
        deg = (G > thr).sum(axis=1)
        deg = tuple(sorted(map(int, deg)))
        nc = int((G > thr).sum()//2)
        return (X.shape[0], nc, deg)
    except Exception:
        return None

def _align_B_to_A(A, B):
    # ICP + Kabsch, A,B: Nx4 normalized
    n = A.shape[0]
    Qtot = np.eye(4)
    Bq = B.copy()
    for _ in range(6):
        try:
            S = A @ Bq.T
            _, col = linear_sum_assignment(-S)
            Bm = Bq[col]
            M = Bm.T @ A
            U, _, Vt = np.linalg.svd(M)
            Qs = U @ Vt
            if np.linalg.det(Qs) < 0:
                # allow reflection (orthogonal) keep as is; optionally fix
                pass
            Qtot = Qtot @ Qs
            Bq = Bq @ Qs
        except Exception:
            break
    return Bq, Qtot

def _union_child(A, Bal, rng):
    N = A.shape[0]
    U = np.vstack([A, Bal])
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    M = U.shape[0]
    # greedy maximin starting from random seed
    start = int(rng.integers(M))
    sel = [start]
    selmask = np.zeros(M, dtype=bool)
    selmask[start] = True
    best_dot = U @ U[start]
    best_dot[start] = 2.0
    for _ in range(N-1):
        # pick unselected with minimal best_dot
        cand = np.where(~selmask)[0]
        # masked best_dot
        j = cand[int(np.argmin(best_dot[cand]))]
        sel.append(int(j))
        selmask[j] = True
        best_dot[j] = 2.0
        d = U @ U[j]
        best_dot = np.minimum(best_dot, d + selmask.astype(float)*10.0)
        # keep selected at large value
        for s in sel:
            best_dot[s] = 2.0
    sel = np.array(sel, dtype=int)
    Xc = U[sel]
    # one local-swap pass: try replace worst with best unused
    try:
        G = Xc @ Xc.T
        np.fill_diagonal(G, -2.0)
        ppm = G.max(axis=1)
        order = np.argsort(ppm)[::-1]
        unused = np.where(~np.isin(np.arange(M), sel))[0]
        # precompute dots unused vs Xc
        if len(unused) > 0:
            D = np.abs(U[unused] @ Xc.T).max(axis=1) if False else (U[unused] @ Xc.T).max(axis=1)
            # try up to 4 swaps
            for k in range(min(4, N)):
                wi = int(order[k])
                # best unused minimal max-dot to Xc without wi
                Xrest = np.delete(Xc, wi, axis=0)
                du = (U[unused] @ Xrest.T).max(axis=1)
                bi = int(np.argmin(du))
                # evaluate swap
                Xt = Xc.copy()
                Xt[wi] = U[unused[bi]]
                if _max_cos(Xt) + 1e-12 < _max_cos(Xc):
                    Xc = Xt
    except Exception:
        pass
    return Xc

def solve():
    t0 = time.time()
    DEADLINE = t0 + 174.0
    SEARCH_END = t0 + 150.0
    rng = np.random.default_rng(16351)
    best_x = None; best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape != (163, 4):
                continue
            a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
            if not np.all(np.isfinite(a)):
                continue
            v = _max_cos(a)
            if v < best_v:
                best_v = v; best_x = a.copy()
        except Exception:
            continue
    if best_x is None:
        z = rng.standard_normal((163, 4))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        best_x = z; best_v = _max_cos(z)
    # elite archive of 163-point optima
    elites = [(best_x.copy(), float(best_v))]
    hashes = set()
    try:
        hashes.add(_graph_hash(best_x, best_v))
    except Exception:
        pass
    use_count = [0]

    def try_add_163(X):
        nonlocal best_x, best_v
        try:
            v = _max_cos(X)
        except Exception:
            return False
        if not np.isfinite(v):
            return False
        h = _graph_hash(X, v)
        is_new = (h not in hashes)
        # keep elites sorted, max 10
        elites.append((X.copy(), float(v)))
        use_count.append(0)
        # sort by value
        idx = np.argsort([e[1] for e in elites])
        elites[:] = [elites[i] for i in idx]
        use_count[:] = [use_count[i] for i in idx]
        while len(elites) > 10:
            elites.pop(); use_count.pop()
        if is_new:
            hashes.add(h)
        if v < best_v - 1e-15:
            best_x = X.copy(); best_v = float(v)
        return is_new

    # ---- N-continuation: decreasing 163->155 ----
    Nmap = {}
    Nmap[163] = best_x.copy()
    cur = best_x.copy()
    for N in range(162, 154, -1):
        if time.time() > SEARCH_END - 30:
            break
        try:
            ppm, mv = _per_point_max(cur)
            worst = int(np.argmax(ppm))
            mask = np.ones(cur.shape[0], dtype=bool); mask[worst] = False
            nxt = cur[mask]
            dl = min(SEARCH_END, time.time()+1.8)
            nxt = _polish_short(nxt, dl)
            Nmap[N] = nxt.copy()
            cur = nxt
        except Exception:
            break
    # increasing 163->172
    cur = best_x.copy()
    for N in range(164, 173):
        if time.time() > SEARCH_END - 30:
            break
        try:
            h, _ = _hole(cur, rng, ns=18000)
            nxt = np.vstack([cur, h[None, :]])
            dl = min(SEARCH_END, time.time()+1.8)
            nxt = _polish_short(nxt, dl)
            Nmap[N] = nxt.copy()
            cur = nxt
        except Exception:
            break
    # path back up 155->163 (inherits loose topologies)
    if 155 in Nmap:
        cur = Nmap[155].copy()
        for N in range(156, 164):
            if time.time() > SEARCH_END - 12:
                break
            try:
                h, _ = _hole(cur, rng, ns=18000)
                # small jitter to escape
                cur = cur + rng.standard_normal(cur.shape)*0.004
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
                nxt = np.vstack([cur, h[None, :]])
                dl = min(SEARCH_END, time.time()+2.0)
                nxt = _polish_short(nxt, dl)
                cur = nxt
                if N == 163:
                    try_add_163(cur)
                else:
                    Nmap[N] = cur.copy() if N not in Nmap else Nmap[N]
            except Exception:
                break
    # path back down 172->163 (inherits crowded topologies)
    if 172 in Nmap:
        cur = Nmap[172].copy()
        for N in range(171, 162, -1):
            if time.time() > SEARCH_END - 8:
                break
            try:
                ppm, mv = _per_point_max(cur)
                # delete worst-contact point (most constrained)
                worst = int(np.argmax(ppm))
                mask = np.ones(cur.shape[0], dtype=bool); mask[worst] = False
                nxt = cur[mask]
                # slight shake
                nxt = nxt + rng.standard_normal(nxt.shape)*0.004
                nxt /= np.maximum(np.linalg.norm(nxt, axis=1, keepdims=True), 1e-12)
                dl = min(SEARCH_END, time.time()+2.0)
                nxt = _polish_short(nxt, dl)
                cur = nxt
                if N == 163:
                    try_add_163(cur)
            except Exception:
                break
    # add a couple of random-restart 163 optima for diversity if time
    for _ in range(2):
        if time.time() > SEARCH_END - 8:
            break
        try:
            z = rng.standard_normal((163, 4))
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            # mix with best: replace 30 random points by holes
            dl = min(SEARCH_END, time.time()+2.0)
            zp = _polish_short(z, dl)
            try_add_163(zp)
        except Exception:
            pass
    # ---- orthogonal crossover loop ----
    child_id = 0
    while time.time() < SEARCH_END:
        if len(elites) < 2:
            # mutate best
            try:
                cur = best_x + rng.standard_normal(best_x.shape)*0.03
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
                dl = min(SEARCH_END, time.time()+2.8)
                cur = _polish_short(cur, dl)
                try_add_163(cur)
            except Exception:
                pass
            continue
        # pick pair: least used + best, or round robin favouring graph distance
        try:
            # choose two with maximal degree-seq distance among least-used
            order_use = np.argsort(use_count)
            cands = order_use[:min(5, len(elites))]
            bi = int(cands[0])
            # find partner maximizing hash distance
            h0 = elites[bi][0]
            G0 = h0 @ h0.T; np.fill_diagonal(G0, -2.0)
            m0 = float(G0.max()); deg0 = np.sort((G0 > m0-1e-9).sum(axis=1))
            best_j = -1; best_d = -1
            for j in range(len(elites)):
                if j == bi:
                    continue
                Hj = elites[j][0]
                Gj = Hj @ Hj.T; np.fill_diagonal(Gj, -2.0)
                mj = float(Gj.max()); degj = np.sort((Gj > mj-1e-9).sum(axis=1))
                d = float(np.abs(deg0-degj).sum()) + abs(m0-mj)*1e6*0.001
                # bonus for never-paired? use child_id
                if d > best_d:
                    best_d = d; best_j = j
            if best_j < 0:
                best_j = int(order_use[1]) if len(order_use) > 1 else 1
            use_count[bi] += 1; use_count[best_j] += 1
            A = elites[bi][0].copy(); B = elites[best_j][0].copy()
            # random orthogonal pre-rotation for diversity every other child
            if (child_id % 3) == 2:
                Qr, _ = np.linalg.qr(rng.standard_normal((4, 4)))
                B = B @ Qr
            Bal, _ = _align_B_to_A(A, B)
            Xc = _union_child(A, Bal, rng)
            # anneal: small noise then polish
            Xc = Xc + rng.standard_normal(Xc.shape)*0.008
            Xc /= np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)
            dl = min(SEARCH_END, time.time()+3.0)
            # smooth with slightly hotter taus for topology jump then LP
            try:
                Xc = _smooth(Xc, dl-1.0, taus=(0.006, 0.002, 0.0007), iters=(70, 70, 90))
            except Exception:
                pass
            try:
                Xc = _lp_polish(Xc, dl, r0=1.2e-3)
            except Exception:
                pass
            try_add_163(Xc)
            child_id += 1
            # also try reciprocal greedy with different seed occasionally
            if time.time() < SEARCH_END - 1 and (child_id % 4 == 0):
                try:
                    Bal2, _ = _align_B_to_A(B, A)
                    Xc2 = _union_child(B, Bal2, rng)
                    Xc2 = Xc2 + rng.standard_normal(Xc2.shape)*0.01
                    Xc2 /= np.maximum(np.linalg.norm(Xc2, axis=1, keepdims=True), 1e-12)
                    dl2 = min(SEARCH_END, time.time()+2.5)
                    Xc2 = _polish_short(Xc2, dl2)
                    try_add_163(Xc2)
                except Exception:
                    pass
        except Exception:
            # fallback mutation
            try:
                cur = best_x + rng.standard_normal(best_x.shape)*0.05
                cur /= np.maximum(np.linalg.norm(cur, axis=1, keepdims=True), 1e-12)
                dl = min(SEARCH_END, time.time()+2.0)
                cur = _polish_short(cur, dl)
                try_add_163(cur)
            except Exception:
                pass
    # final: pick best elite and long polish
    try:
        elites_sorted = sorted(elites, key=lambda e: e[1])
        best_x = elites_sorted[0][0].copy(); best_v = float(elites_sorted[0][1])
    except Exception:
        pass
    try:
        rem = DEADLINE - time.time()
        if rem > 6:
            sd = time.time() + min(10.0, rem-7.0)
            best_x = _smooth(best_x, sd, taus=(0.004, 0.0015, 0.0006, 0.0003), iters=(90, 90, 90, 70))
    except Exception:
        pass
    try:
        best_x = _lp_polish(best_x, DEADLINE-0.4, r0=2e-3)
    except Exception:
        pass
    best_x = best_x / np.maximum(np.linalg.norm(best_x, axis=1, keepdims=True), 1e-12)
    return best_x
