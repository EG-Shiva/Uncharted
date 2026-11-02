import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _quench(X, steps, b0, b1, lr0, lr1, mu=0.86, clip=0.03):
    n, d = X.shape
    V = np.zeros_like(X)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        Gr = (W @ X) / sw
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        V = mu * V + Gt
        step = lr * V
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        over = sn > clip
        step = np.where(over, step * (clip / sn), step)
        X = X - step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X


def _lp_finish(X, seconds, deadline):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n, d = X.shape
    best = _max_cos(X)
    radius = 6e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        it += 1
        if it > 30:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 3e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 600:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 600:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y
            best = v
            radius = min(radius * 1.5, 0.015)
        else:
            radius *= 0.5
    return X


def _tangent_bases(X):
    n, d = X.shape
    dm = d - 1
    B = np.empty((n, d, dm))
    e1 = np.zeros(d)
    e1[0] = 1.0
    e2 = np.zeros(d)
    e2[1] = 1.0
    I = np.eye(d)
    for i in range(n):
        x = X[i]
        e = e1 if abs(float(x[0])) < 0.99 else e2
        u = e - x
        nu = float(np.linalg.norm(u))
        if nu < 1e-12:
            # already aligned, use Householder on e2
            e = e2 if abs(float(x[0])) < 0.99 else e1
            u = e - x
            nu = float(np.linalg.norm(u))
        u = u / nu
        H = I - 2.0 * np.outer(u, u)
        # first column should be x (up to sign); pick complement
        # ensure H[:,0] ~ x, else just orthogonalize via QR fallback
        B[i] = H[:, 1:]
        # re-orthogonalize against x numerically
        # B should satisfy x^T B = 0
    return B


def _contact_list(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    c = G[iu]
    mask = c > m - tol
    I = iu[0][mask]
    J = iu[1][mask]
    cc = c[mask]
    S = set(zip(I.tolist(), J.tolist()))
    return m, I, J, cc, S, G


def _build_A_sparse(X, B, I, J, cc):
    n, d = X.shape
    dm = d - 1
    N = n * dm
    p = len(I)
    rows = []
    cols = []
    data = []
    for t in range(p):
        i = int(I[t]); j = int(J[t]); c = float(cc[t])
        # row vectors
        ri = X[j] - c * X[i]  # (d,)
        rj = X[i] - c * X[j]
        vi = ri @ B[i]  # (dm,)
        vj = rj @ B[j]
        base_i = i * dm
        base_j = j * dm
        for k in range(dm):
            a = float(vi[k])
            if a != 0.0:
                rows.append(t); cols.append(base_i + k); data.append(a)
            b = float(vj[k])
            if b != 0.0:
                rows.append(t); cols.append(base_j + k); data.append(b)
    A = coo_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=(p, N)).tocsr()
    return A


def _null_modes(A, kwant=15):
    from scipy.sparse.linalg import svds
    p, N = A.shape
    k = int(min(kwant, min(p, N) - 1))
    if k < 1:
        return None
    try:
        u, s, vt = svds(A, k=k, which='SM', maxiter=20000, tol=1e-8)
        # svds returns ascending order; smallest first already
        order = np.argsort(s)
        vt = vt[order]
        return vt  # (k, N)
    except Exception:
        return None


def _mode_direction(B, vt, rng):
    n, dm = B.shape[0], B.shape[2]
    N = n * dm
    k = vt.shape[0]
    coeff = rng.standard_normal(k)
    cn = float(np.linalg.norm(coeff))
    if cn == 0:
        coeff[0] = 1.0
        cn = 1.0
    coeff = coeff / cn
    v = coeff @ vt  # (N,)
    nv = float(np.linalg.norm(v)) + 1e-12
    v = v / nv
    Vb = v.reshape(n, dm)
    D = np.einsum('ndk,nk->nd', B, Vb)
    return D


def _random_tangent(X, rng):
    R = rng.standard_normal(X.shape)
    R = R - np.sum(R * X, axis=1, keepdims=True) * X
    f = float(np.linalg.norm(R)) + 1e-12
    return R / f


def _walk_until_topology(X0, D, m0, S0, alpha_max, rise_limit):
    n = X0.shape[0]
    # 40 steps
    nsteps = 36
    alphas = np.linspace(0, alpha_max, nsteps + 1)[1:]
    prevX = X0
    for a in alphas:
        Xa = X0 + float(a) * D
        Xa = Xa / (np.linalg.norm(Xa, axis=1, keepdims=True) + 1e-12)
        G = Xa @ Xa.T
        np.fill_diagonal(G, -2.0)
        ma = float(G.max())
        if ma - m0 > rise_limit:
            return prevX, ma, False
        # check topology: argmax not in S0
        flat = int(np.argmax(G))
        ii = flat // n
        jj = flat % n
        if ii > jj:
            ii, jj = jj, ii
        if (ii, jj) not in S0:
            return Xa, ma, True
        prevX = Xa
    return prevX, _max_cos(prevX), False


def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    rng = np.random.default_rng(20261102)
    n, d = 58, 22

    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (n, d):
                continue
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        best = z
        best_val = _max_cos(best)
    bestX = best.copy()

    # initial light LP to ensure jam (short)
    try:
        if time.time() < DEADLINE - 60:
            Xp = _lp_finish(bestX.copy(), 4.0, DEADLINE - 12.0)
            vp = _max_cos(Xp)
            if vp < best_val:
                best_val = vp
                bestX = Xp
    except Exception:
        pass

    # build null modes for current best
    def refresh_modes(Xcur):
        try:
            m0, I0, J0, cc0, S0, _ = _contact_list(Xcur, tol=1e-7)
            # if too few contacts, widen tol to get Jacobian
            if len(I0) < 30:
                m0b, I0b, J0b, cc0b, S0b, _ = _contact_list(Xcur, tol=2e-4)
                # keep strict set for topology, wide set for Jacobian
                B = _tangent_bases(Xcur)
                A = _build_A_sparse(Xcur, B, I0b, J0b, cc0b)
                vt = _null_modes(A, k want if False else 15) if A.shape[0] >= 16 else None
                return B, vt, m0, S0
            B = _tangent_bases(Xcur)
            A = _build_A_sparse(Xcur, B, I0, J0, cc0)
            vt = _null_modes(A, k=15) if A.shape[0] >= 16 else None
            return B, vt, m0, S0
        except Exception:
            return None, None, _max_cos(Xcur), set()

    # need fix for kwarg typo above: redo cleanly
    def refresh_modes2(Xcur):
        try:
            m0, I0, J0, cc0, S0, _ = _contact_list(Xcur, tol=1e-7)
            if len(I0) < 30:
                _, Iw, Jw, ccw, _, _ = _contact_list(Xcur, tol=2e-4)
                B = _tangent_bases(Xcur)
                if len(Iw) >= 16:
                    from scipy.sparse.linalg import svds as _sv
                    A = _build_A_sparse(Xcur, B, Iw, Jw, ccw)
                    vt = _null_modes(A, k=15)
                else:
                    vt = None
                return B, vt, m0, S0
            B = _tangent_bases(Xcur)
            A = _build_A_sparse(Xcur, B, I0, J0, cc0)
            vt = _null_modes(A, k=15)
            return B, vt, m0, S0
        except Exception:
            return None, None, _max_cos(Xcur), set()

    Bcur, vtcur, mcur, Scur = refresh_modes2(bestX)

    trial = 0
    since_refresh = 0
    while time.time() < DEADLINE - 14.0:
        trial += 1
        since_refresh += 1
        if since_refresh > 40 or Bcur is None:
            Bcur, vtcur, mcur, Scur = refresh_modes2(bestX)
            since_refresh = 0
            if time.time() >= DEADLINE - 14.0:
                break
        # pick escape params
        alpha_max = float(rng.uniform(0.05, 0.32))
        if trial % 17 == 0:
            alpha_max = float(rng.uniform(0.30, 0.55))
        rise_limit = float(rng.uniform(0.002, 0.007))
        try:
            if vtcur is not None and Bcur is not None:
                D = _mode_direction(Bcur, vtcur, rng)
                # occasionally mix random tangent to avoid cycling
                if trial % 9 == 0:
                    D = 0.8 * D + 0.2 * _random_tangent(bestX, rng)
                    f = float(np.linalg.norm(D)) + 1e-12
                    D = D / f
            else:
                D = _random_tangent(bestX, rng)
        except Exception:
            D = _random_tangent(bestX, rng)

        try:
            m0 = _max_cos(bestX)
            # Scur may be stale; recompute strict set cheaply every trial
            _, _, _, _, S0, _ = _contact_list(bestX, tol=1e-7)
            Xesc, _, _changed = _walk_until_topology(bestX, D, m0, S0, alpha_max, rise_limit)
        except Exception:
            continue

        # quench escape point
        try:
            b0 = 80 if (trial % 2 == 0) else 120
            b1 = 2000 if (trial % 3 != 0) else 3500
            Xq = _quench(Xesc.copy(), 550, b0, b1, 0.09, 0.006, 0.86, 0.032)
        except Exception:
            continue
        vq = _max_cos(Xq)
        improved = False
        if vq < best_val - 1e-12:
            # polish immediately
            try:
                secs = 2.2
                Xp = _lp_finish(Xq.copy(), secs, DEADLINE - 12.0)
                vp = _max_cos(Xp)
            except Exception:
                Xp, vp = Xq, vq
            if vp < best_val - 1e-12:
                best_val = vp
                bestX = Xp.copy()
                Bcur, vtcur, mcur, Scur = refresh_modes2(bestX)
                since_refresh = 0
                improved = True
            else:
                best_val = vq
                bestX = Xq.copy()
                Bcur, vtcur, mcur, Scur = refresh_modes2(bestX)
                since_refresh = 0
                improved = True
        elif vq < best_val + 0.0015 and (trial % 4 == 0):
            # promising new basin: short LP to see true value
            try:
                Xp = _lp_finish(Xq.copy(), 1.6, DEADLINE - 12.0)
                vp = _max_cos(Xp)
                if vp < best_val - 1e-12:
                    best_val = vp
                    bestX = Xp.copy()
                    Bcur, vtcur, mcur, Scur = refresh_modes2(bestX)
                    since_refresh = 0
                    improved = True
            except Exception:
                pass
        # occasional random fresh quench to keep diversity (cheap)
        if not improved and trial % 25 == 0 and time.time() < DEADLINE - 30:
            try:
                Z = bestX + 0.12 * rng.standard_normal(bestX.shape)
                Z /= (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
                Z = _quench(Z, 400, 60, 1500, 0.10, 0.008, 0.86, 0.035)
                vz = _max_cos(Z)
                if vz < best_val - 1e-12:
                    best_val = vz
                    bestX = Z.copy()
                    Bcur, vtcur, mcur, Scur = refresh_modes2(bestX)
                    since_refresh = 0
            except Exception:
                pass

    # final polish
    try:
        if time.time() < DEADLINE - 11.0:
            bestX = _quench(bestX.copy(), 700, 200, 4000, 0.04, 0.003, 0.85, 0.02)
            v = _max_cos(bestX)
            if v < best_val:
                best_val = v
    except Exception:
        pass
    try:
        if time.time() < DEADLINE - 2.0:
            secs = max(1.0, min(9.0, DEADLINE - 1.5 - time.time()))
            Xlp = _lp_finish(bestX.copy(), secs, DEADLINE - 0.8)
            v = _max_cos(Xlp)
            if v < best_val:
                bestX = Xlp
                best_val = v
    except Exception:
        pass

    bestX = bestX / (np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-12)
    bestX = np.where(np.isfinite(bestX), bestX, rng.standard_normal((n, d)))
    bestX = bestX / (np.linalg.norm(bestX, axis=1, keepdims=True) + 1e-12)
    return bestX
