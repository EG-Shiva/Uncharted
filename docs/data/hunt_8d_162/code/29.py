import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator
try:
    from scipy.sparse.linalg import cg as _cg
except Exception:
    _cg = None

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def lp_polish(x, seconds, r0=1e-3, rmax=0.05):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 3000:
            idx = np.argpartition(-c[sel], 3000)[:3000]
            I, J = I[idx], J[idx]
            p = 3000
        cc = np.array([float(x[a] @ x[b]) for a, b in zip(I, J)])
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cc, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, rmax)
        else:
            radius *= 0.5
    return x

def tangent_bases(x):
    n, d = x.shape
    Qs = np.empty((n, d, d - 1))
    Eye = np.eye(d)
    for i in range(n):
        xi = x[i]
        k = int(np.argmax(np.abs(xi)))
        cols = [j for j in range(d) if j != k]
        V = Eye[:, cols] - xi[:, None] * xi[None, cols]
        Q, _ = np.linalg.qr(V)
        Qs[i] = Q
    return Qs

def build_active(x, thresh=1e-6, cap=2500):
    n = x.shape[0]
    G = x @ x.T
    iu0 = np.triu_indices(n, 1)
    c = G[iu0]
    m = float(c.max())
    slack = m - c
    order = np.argsort(slack)
    sel = slack < thresh
    cnt_sel = int(sel.sum())
    if cnt_sel > cap:
        keep = order[:cap]
        I = iu0[0][keep]
        J = iu0[1][keep]
        cc = c[keep]
    else:
        I = iu0[0][sel]
        J = iu0[1][sel]
        cc = c[sel]
    # per-point tight counts for bias (slack<1e-7)
    tight = slack < 1e-7
    cnt = np.zeros(n, dtype=float)
    np.add.at(cnt, iu0[0][tight], 1)
    np.add.at(cnt, iu0[1][tight], 1)
    return m, I, J, cc, cnt

def build_Jsparse(x, Qs, I, J):
    n, d = x.shape
    dm = d - 1
    p = len(I)
    Np = n * dm
    if p == 0:
        return csr_matrix((0, Np))
    # QsT_x : n x n? compute projections: Pi_j = Qs[i].T @ x[j]
    # each row has 2*dm nonzeros
    rows = np.repeat(np.arange(p), 2 * dm)
    cols = np.empty(p * 2 * dm, dtype=np.int64)
    vals = np.empty(p * 2 * dm, dtype=float)
    for r in range(p):
        a = int(I[r]); b = int(J[r])
        pa = Qs[a].T @ x[b]  # dm
        pb = Qs[b].T @ x[a]
        base = r * 2 * dm
        cols[base:base+dm] = a * dm + np.arange(dm)
        vals[base:base+dm] = pa
        cols[base+dm:base+2*dm] = b * dm + np.arange(dm)
        vals[base+dm:base+2*dm] = pb
    return coo_matrix((vals, (rows, cols)), shape=(p, Np)).tocsr()

def project_null(J, y0, reg=1e-10):
    p, Np = J.shape
    if p == 0:
        return y0.copy()
    r = J @ y0
    nr = float(np.dot(r, r) ** 0.5)
    if nr < 1e-14:
        return y0.copy()
    # solve (J J^T + reg I) lam = r via CG
    JT = J.T.tocsr()
    def matvec(v):
        w = JT @ v
        return (J @ w) + reg * v
    try:
        Aop = LinearOperator((p, p), matvec=matvec, dtype=float)
        if _cg is None:
            raise RuntimeError("no cg")
        try:
            lam, info = _cg(Aop, r, tol=1e-7, maxiter=500)
        except TypeError:
            lam, info = _cg(Aop, r, rtol=1e-7, maxiter=500)
        if not np.all(np.isfinite(lam)) or info not in (0,):
            # still use lam if finite
            if not np.all(np.isfinite(lam)):
                raise RuntimeError("nonfinite")
        y = y0 - (JT @ lam)
        # check residual
        res = J @ y
        # if not reduced, fallback richardson refinement
        if float(np.dot(res, res) ** 0.5) > 0.5 * nr:
            raise RuntimeError("bad proj")
        return y
    except Exception:
        # Richardson iteration toward nullspace: y <- y - eta J^T J y
        # estimate step
        eta = 0.5 / max(1.0, float((J.data ** 2).sum() / max(1, p) * 8.0))
        eta = min(eta, 1e-3)
        y = y0.copy()
        for _ in range(600):
            Jy = J @ y
            y = y - eta * (J.T @ Jy)
            # re-anchor null component? keep as is (null preserved)
        # remove drift: re-add null part? Richardson already gives projection
        # rescale to keep component along y0 direction: project formula needs y close to y0
        # blend: y = y0 - (y0 - y) is already y; keep
        return y

def retract(x, dx, alpha):
    y = x + alpha * dx
    nrm = np.linalg.norm(y, axis=1, keepdims=True)
    nrm = np.maximum(nrm, 1e-12)
    return y / nrm

def largest_feasible(x, dx, m0, allow=1e-7, amax=0.06):
    def f(a):
        return max_cos(retract(x, dx, a))
    # quick check tiny step
    if f(2e-5) > m0 + allow:
        return 0.0
    hi = 2e-4
    while hi <= amax + 1e-12:
        v = f(hi)
        if v > m0 + allow or not np.isfinite(v):
            break
        if hi >= amax:
            return amax
        hi *= 2.0
        if hi > amax:
            hi = amax
    lo = hi / 2.0
    # binary refine 22 iters
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        v = f(mid)
        if np.isfinite(v) and v <= m0 + allow:
            lo = mid
        else:
            hi = mid
    return float(lo)

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.RandomState(1234567)
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = rng.randn(162, 8)
    xb = np.asarray(xb, dtype=float).reshape(162, 8)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True) + 1e-300
    best_val = max_cos(xb)
    x_best = xb.copy()
    n, d = 162, 8
    dm = d - 1
    # initial active info for bias
    nwalks = 0
    # run nullspace crawls
    while time.time() < DEADLINE - 16.0:
        nwalks += 1
        x_cur = x_best.copy()
        m_init = max_cos(x_cur)
        # randomize walk hyperparams deterministically
        inner_steps = int(rng.randint(8, 18))
        allow = float(10 ** rng.uniform(-7.5, -6.5))  # ~3e-8..3e-7, centered 1e-7
        allow = min(max(allow, 3e-8), 3e-7)
        amax_global = float(rng.uniform(0.02, 0.07))
        # record active set signatures to detect topology change
        _, I0, J0, _, _ = build_active(x_cur, thresh=1e-6)
        set0 = set(zip(I0.tolist(), J0.tolist()))
        crawled = False
        for it in range(inner_steps):
            if time.time() > DEADLINE - 16.0:
                break
            m_cur, I, J, cc, cnt = build_active(x_cur, thresh=1e-6, cap=2500)
            # tangent bases
            Qs = tangent_bases(x_cur)
            Jsp = build_Jsparse(x_cur, Qs, I, J)
            # bias weights away from overloaded points (LP-dual proxy = contact counts)
            # w_i = 1/sqrt(1+cnt/3), rattlers get boost
            w = 1.0 / np.sqrt(1.0 + cnt / 3.0)
            # occasional boost to a random subset to diversify
            if rng.rand() < 0.4:
                k = int(rng.randint(4, 14))
                # prefer low-contact points to lead
                order = np.argsort(cnt)
                lead = order[:max(10, 2*k)]
                chosen = rng.choice(lead, size=min(k, len(lead)), replace=False)
                w = w.copy()
                w[chosen] *= float(rng.uniform(1.5, 3.0))
            y0 = rng.randn(n * dm)
            # apply per-point weights
            for i in range(n):
                y0[i*dm:(i+1)*dm] *= w[i]
            y = project_null(Jsp, y0, reg=1e-9)
            ny = float(np.linalg.norm(y))
            if not np.isfinite(ny) or ny < 1e-9:
                continue
            y /= ny
            # form dx
            dx = np.empty((n, d))
            for i in range(n):
                dx[i] = Qs[i] @ y[i*dm:(i+1)*dm]
            # try both signs, pick larger feasible step
            a_pos = largest_feasible(x_cur, dx, m_cur, allow=allow, amax=amax_global)
            a_neg = largest_feasible(x_cur, -dx, m_cur, allow=allow, amax=amax_global)
            if a_pos >= a_neg:
                a, s = a_pos, 1.0
            else:
                a, s = a_neg, -1.0
            if a < 5e-5:
                # direction blocked; try new random field next iter
                continue
            x_cur = retract(x_cur, s * dx, a * 0.99)
            crawled = True
            # check topology change: new entrant or separation
            m_new = max_cos(x_cur)
            _, I1, J1, _, _ = build_active(x_cur, thresh=1e-6, cap=5000)
            set1 = set(zip(I1.tolist(), J1.tolist()))
            new_in = len(set1 - set0)
            gone = len(set0 - set1)
            # if material change, stop crawl and polish
            if new_in > 0 or gone > 0:
                # update reference set to allow continued crawl a bit more?
                # spec: repeat until inactive enters or active separates, then polish
                # break to polish
                break
            set0 = set1
        if not crawled:
            # fallback tiny random kick so LP gets a different basin occasionally
            x_cur = x_best + 0.002 * rng.randn(n, d)
            x_cur /= np.linalg.norm(x_cur, axis=1, keepdims=True)
        # LP polish the crawled point
        time_left = DEADLINE - time.time()
        pol_sec = min(4.0, max(1.2, time_left - 10.0))
        if pol_sec < 0.8:
            break
        y_pol = lp_polish(x_cur, seconds=pol_sec, r0=1e-3, rmax=0.05)
        v = max_cos(y_pol)
        if v < best_val - 1e-12:
            best_val = v
            x_best = y_pol.copy()
        # also try polishing from slightly perturbed version to escape snap-back
        if nwalks % 3 == 0 and time.time() < DEADLINE - 20.0:
            x_alt = x_cur + 0.001 * rng.randn(n, d)
            x_alt /= np.linalg.norm(x_alt, axis=1, keepdims=True)
            y2 = lp_polish(x_alt, seconds=1.0, r0=5e-4, rmax=0.02)
            v2 = max_cos(y2)
            if v2 < best_val - 1e-12:
                best_val = v2
                x_best = y2.copy()
    # final snap
    if time.time() < DEADLINE - 2.0:
        x_best = lp_polish(x_best, seconds=max(0.5, DEADLINE - time.time() - 0.5), r0=5e-4, rmax=0.03)
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
