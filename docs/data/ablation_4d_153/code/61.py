import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _deg_hash(x, tol=5e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    B = G > (m - tol)
    deg = np.sum(B, axis=1)
    return (tuple(sorted(deg.tolist())), round(float(m) * 1e7))

def _lp_polish(x, deadline, r0=1e-3, cap=2500, maxit=300):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < maxit:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            radius = min(radius * 1.5, 0.05)
            continue
        if idx.size > cap:
            o = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[o]
        I = iu0[0][idx]; J = iu0[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            x = y; best = v; best_x = y.copy()
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return best_x, best

def _smooth_relax(x, steps, tau, lr, rng=None):
    n, d = x.shape
    for t in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G[np.isfinite(G)]))
        # global softmax weights, tempered
        W = np.exp(np.clip((G - m) / tau, -50, 0.0))
        np.fill_diagonal(W, 0.0)
        # per-row normalize to avoid hub dominance
        rs = np.sum(W, axis=1, keepdims=True)
        rs[rs < 1e-12] = 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True) + 1e-12
        # normalize step per point
        grad = grad / np.maximum(gn, 1.0)
        x = x - lr * grad
        x = _norm(x)
    return x

def _random_SO4_block(rng, th1, th2):
    # random double rotation with angles th1,th2
    A = rng.standard_normal((4, 4))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    c1, s1 = np.cos(th1), np.sin(th1)
    c2, s2 = np.cos(th2), np.sin(th2)
    R0 = np.array([[c1, -s1, 0, 0],
                   [s1, c1, 0, 0],
                   [0, 0, c2, -s2],
                   [0, 0, s2, c2]])
    return Q.T @ R0 @ Q

def _contact_edges(x, tol=1e-9):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    iu = np.triu_indices(n, 1)
    c = G[iu]
    mask = c > (m - tol)
    return (iu[0][mask], iu[1][mask], m, G)

def _fiedler_cut(x, rng, edges, k_eig=3):
    n = x.shape[0]
    I, J = edges[0], edges[1]
    # Laplacian
    L = np.zeros((n, n))
    for a, b in zip(I.tolist(), J.tolist()):
        L[a, a] += 1.0; L[b, b] += 1.0
        L[a, b] -= 1.0; L[b, a] -= 1.0
    try:
        vals, vecs = np.linalg.eigh(L)
    except Exception:
        return None
    # combine low eigenvectors with random coeffs for diversity
    # skip eigenvalue 0 (vec 0)
    m_use = min(k_eig, n - 1)
    coeff = rng.standard_normal(m_use)
    coeff = coeff / (np.linalg.norm(coeff) + 1e-12)
    f = np.zeros(n)
    for k in range(m_use):
        f = f + coeff[k] * vecs[:, 1 + k]
    # split by median to get balanced-ish but varied size
    # random quantile threshold
    q = rng.uniform(0.3, 0.7)
    thr = np.quantile(f, q)
    S = np.where(f > thr)[0]
    if S.size < 8 or S.size > n - 8:
        # fallback to sign split
        S = np.where(f > 0)[0]
    if S.size < 8 or S.size > n - 8:
        return None
    # randomly pick side
    if rng.random() < 0.5:
        pass
    else:
        S = np.setdiff1d(np.arange(n), S)
    return S

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    EXPLORE_END = t0 + 122.0
    rng = np.random.default_rng(0)
    n, d = 153, 4
    best_x = None; best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal((n, d)))
        best_v = _max_cos(best_x)
    global_best_x = best_x.copy()
    global_best_v = best_v
    cur_x = best_x.copy()
    cur_v = best_v
    seen = set()
    try:
        seen.add(_deg_hash(global_best_x))
    except Exception:
        pass

    # precompute once contact edges for cuts; refresh when global improves
    trial = 0
    thetas = [0.02, 0.04, 0.07, 0.10, 0.14]
    while time.time() < EXPLORE_END:
        trial += 1
        if time.time() > EXPLORE_END - 2.0:
            break
        # choose base: mostly global best, sometimes cur
        if trial % 4 == 0:
            base = cur_x.copy()
            base_v = cur_v
        else:
            base = global_best_x.copy()
            base_v = global_best_v
        # choose subset mode
        mode = trial % 5
        S = None
        try:
            if mode in (0, 1, 2):
                I, J, m0, G0 = _contact_edges(base, tol=5e-7)
                S = _fiedler_cut(base, rng, (I, J), k_eig=3 if mode != 2 else 5)
                if S is None:
                    raise ValueError("no cut")
            elif mode == 3:
                # geometric cap
                u = rng.standard_normal(4); u /= np.linalg.norm(u)
                proj = base @ u
                k = int(rng.integers(10, 42))
                S = np.argsort(-proj)[:k]
            else:
                # BFS cluster on contact graph
                I, J, m0, G0 = _contact_edges(base, tol=5e-7)
                adj = [[] for _ in range(n)]
                for a, b in zip(I.tolist(), J.tolist()):
                    adj[a].append(b); adj[b].append(a)
                start = int(rng.integers(0, n))
                k = int(rng.integers(10, 36))
                seen_n = {start}
                frontier = [start]
                order = []
                while frontier and len(order) < k:
                    v = frontier.pop(0)
                    order.append(v)
                    for w in adj[v]:
                        if w not in seen_n:
                            seen_n.add(w)
                            frontier.append(w)
                # if disconnected, fill with nearest by dot
                if len(order) < k:
                    Gc = base @ base[start]
                    extra = np.argsort(-Gc)
                    for w in extra.tolist():
                        if w not in seen_n:
                            order.append(w); seen_n.add(w)
                        if len(order) >= k:
                            break
                S = np.array(order, dtype=int)
        except Exception:
            # fallback random subset
            k = int(rng.integers(10, 30))
            S = rng.choice(n, size=k, replace=False)
        # random rotation angles
        th_scale = thetas[rng.integers(0, len(thetas))]
        th1 = float(rng.uniform(0.5, 1.0) * th_scale * (1 if rng.random() < 0.5 else -1))
        th2 = float(rng.uniform(0.0, 1.0) * th_scale * (1 if rng.random() < 0.5 else -1))
        try:
            R = _random_SO4_block(rng, th1, th2)
        except Exception:
            continue
        y = base.copy()
        y[S] = (R @ y[S].T).T
        y = _norm(y)
        # optional small tangent jitter on S to break exact symmetry
        if rng.random() < 0.5:
            jit = rng.standard_normal((S.size, d)) * 0.004
            dot = np.sum(jit * y[S], axis=1, keepdims=True)
            jit = jit - dot * y[S]
            y[S] = _norm(y[S] + jit)
        v0 = _max_cos(y)
        # skip if explosion too large
        if v0 > base_v + 0.06:
            continue
        # quick smooth relax to reduce spike before LP (cheap)
        try:
            tau = float(rng.choice([0.005, 0.01, 0.02]))
            lr = 0.03
            y = _smooth_relax(y, steps=80, tau=tau, lr=lr)
        except Exception:
            pass
        # LP polish with short deadline
        dl = min(time.time() + 1.6, EXPLORE_END)
        if dl <= time.time() + 0.15:
            break
        try:
            yp, vp = _lp_polish(y, dl, r0=1.2e-3, cap=2200, maxit=120)
        except Exception:
            continue
        try:
            hk = _deg_hash(yp)
        except Exception:
            hk = None
        # update global
        if vp < global_best_v - 1e-12:
            global_best_v = vp
            global_best_x = yp.copy()
            cur_x = yp.copy(); cur_v = vp
            if hk is not None:
                seen.add(hk)
            # refresh quickly: continue exploiting new best
            continue
        # basin-hopping acceptance for cur
        if hk is not None and hk in seen:
            # same basin: only accept if better than cur
            if vp < cur_v:
                cur_x = yp.copy(); cur_v = vp
        else:
            if hk is not None:
                seen.add(hk)
            # accept diverse basin if not much worse (temperature)
            temp = 0.002 if (trial % 7 != 0) else 0.008
            if vp < cur_v + temp and vp < base_v + 0.01:
                cur_x = yp.copy(); cur_v = vp
            # occasional restart cur to global best if cur drifted bad
            if cur_v > global_best_v + 0.005 and rng.random() < 0.3:
                cur_x = global_best_x.copy(); cur_v = global_best_v

    # final thorough polish of global best
    try:
        dl = HARD_END - 1.0
        if dl > time.time():
            global_best_x, global_best_v = _lp_polish(global_best_x, dl, r0=1e-3, cap=3000, maxit=300)
    except Exception:
        pass
    global_best_x = _norm(global_best_x)
    return global_best_x
