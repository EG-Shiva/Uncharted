import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _norm(x):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n < 1e-12] = 1.0
    x = x / n
    x[~np.isfinite(x)] = 0.0
    n2 = np.linalg.norm(x, axis=1, keepdims=True)
    bad = (n2[:, 0] < 1e-12)
    if np.any(bad):
        x[bad] = np.array([1.0, 0.0, 0.0, 0.0])
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _lp_polish(x, seconds, deadline=1e18):
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < deadline and radius > 1e-14:
        c = (x @ x.T)[iu0]
        mcur = float(c.max())
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu0[0][sel], iu0[1][sel]
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
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
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
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _adam_full(x, steps, tau0, tau1, lr0, lr1):
    n, d = x.shape
    m = np.zeros_like(x); v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    iu = np.triu_indices(n, 1)
    for k in range(steps):
        frac = k / (steps - 1) if steps > 1 else 0.0
        tau = tau0 * (tau1 / tau0) ** frac if tau1 != tau0 else tau0
        lr = lr0 + (lr1 - lr0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gm = float(G.max())
        E = G - gm
        E /= tau
        # clip to avoid overflow (max 0)
        np.clip(E, -60.0, 0.0, out=E)
        np.exp(E, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum())
        if not np.isfinite(S) or S <= 0:
            break
        E /= S
        grad = E @ x
        dot = (grad * x).sum(axis=1, keepdims=True)
        g = grad - dot * x
        m *= b1; m += (1.0 - b1) * g
        v *= b2; v += (1.0 - b2) * (g * g)
        bc1 = 1.0 - b1 ** (k + 1); bc2 = 1.0 - b2 ** (k + 1)
        mh = m / bc1; vh = v / bc2
        step = mh / (np.sqrt(vh) + eps)
        np.clip(step, -2.0, 2.0, out=step)
        x = x - lr * step
        x = _norm(x)
    return x

# ---- symmetric orbit machinery ----
def _build_X(pflat, R, K, twist):
    p = pflat.reshape(R, 3)
    a = p[:, 0]; b = p[:, 1]; c = p[:, 2]
    a = np.clip(a, -6.0, 6.0)
    s = 1.0 / (1.0 + np.exp(-a))
    r1 = np.sqrt(np.clip(s, 1e-12, 1 - 1e-12))
    r2 = np.sqrt(np.clip(1.0 - s, 1e-12, 1 - 1e-12))
    alpha = 2.0 * np.pi / K
    beta = 2.0 * np.pi * twist / K
    kk = np.arange(K)
    phi1 = b[:, None] + kk[None, :] * alpha
    phi2 = c[:, None] + kk[None, :] * beta
    co1 = np.cos(phi1); si1 = np.sin(phi1)
    co2 = np.cos(phi2); si2 = np.sin(phi2)
    X = np.empty((R * K, 4))
    X[:, 0] = (r1[:, None] * co1).ravel()
    X[:, 1] = (r1[:, None] * si1).ravel()
    X[:, 2] = (r2[:, None] * co2).ravel()
    X[:, 3] = (r2[:, None] * si2).ravel()
    # renorm (should already be unit)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    aux = (s, r1, r2, phi1, phi2, co1, si1, co2, si2)
    return X, aux

def _sym_obj(pflat, R, K, twist, tau, iu):
    X, aux = _build_X(pflat, R, K, twist)
    s, r1, r2, phi1, phi2, co1, si1, co2, si2 = aux
    N = R * K
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    gv = G[iu]
    M = float(gv.max())
    z = (gv - M) / tau
    np.clip(z, -60.0, 0.0, out=z)
    E = np.exp(z)
    S = float(E.sum())
    if S <= 0 or not np.isfinite(S):
        return 1e6, np.zeros_like(pflat)
    L = M + tau * np.log(S)
    w = E / S
    # full symmetric weight matrix via accumulation without dense build
    # grad_X = W @ X where W triu = w
    # compute via: grad = zeros, use vectorized scatter
    I = iu[0]; J = iu[1]
    # build dense W rows? N=272 so dense ok
    W = np.zeros((N, N))
    W[I, J] = w
    W[J, I] = w
    gradX = W @ X  # (N,4)
    gX = gradX.reshape(R, K, 4)
    dsda = s * (1.0 - s)
    # avoid div by zero
    r1s = np.maximum(r1, 1e-9); r2s = np.maximum(r2, 1e-9)
    dr1 = dsda / (2.0 * r1s)
    dr2 = -dsda / (2.0 * r2s)
    gx1 = gX[:, :, 0]; gx2 = gX[:, :, 1]; gx3 = gX[:, :, 2]; gx4 = gX[:, :, 3]
    ga = np.sum(gx1 * (dr1[:, None] * co1) + gx2 * (dr1[:, None] * si1) +
                gx3 * (dr2[:, None] * co2) + gx4 * (dr2[:, None] * si2), axis=1)
    gb = np.sum(gx1 * (-r1[:, None] * si1) + gx2 * (r1[:, None] * co1), axis=1)
    gc = np.sum(gx3 * (-r2[:, None] * si2) + gx4 * (r2[:, None] * co2), axis=1)
    grad = np.stack([ga, gb, gc], axis=1).ravel()
    # clip huge grads
    grad = np.clip(grad, -1e4, 1e4)
    return float(L), grad

def _optim_sym(R, K, twist, rng, taus, maxiter_each, deadline):
    N = R * K
    iu = np.triu_indices(N, 1)
    a0 = rng.standard_normal(R) * 1.0
    b0 = rng.uniform(0, 2 * np.pi, size=R)
    c0 = rng.uniform(0, 2 * np.pi, size=R)
    p = np.stack([a0, b0, c0], axis=1).ravel()
    bounds = [(-6.0, 6.0)] * R + [(None, None)] * (2 * R)
    # order bounds to match ravel: (a0,b0,c0,a1,b1,c1...) so pattern repeats
    bnds = []
    for r in range(R):
        bnds += [(-6.0, 6.0), (None, None), (None, None)]
    for tau in taus:
        if time.time() > deadline:
            break
        def fun(pf):
            L, g = _sym_obj(pf, R, K, twist, tau, iu)
            return L, g
        try:
            res = minimize(fun, p, jac=True, method="L-BFGS-B", bounds=bnds,
                           options={"maxiter": maxiter_each, "ftol": 1e-12, "gtol": 1e-8})
            p = res.x
        except Exception:
            break
    X, _ = _build_X(p, R, K, twist)
    return _norm(X), _max_cos(X)

def solve():
    rng = np.random.default_rng(7)
    t0 = time.time()
    deadline = t0 + 174.0
    try:
        x0 = np.load("best.npy")
    except Exception:
        x0 = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    if x0 is not None:
        try:
            x0 = _norm(np.asarray(x0, dtype=float))
            if x0.shape == (272, 4):
                cands.append(x0)
        except Exception:
            pass
    if xr is not None:
        try:
            xr = _norm(np.asarray(xr, dtype=float))
            if xr.shape == (272, 4):
                cands.append(xr)
        except Exception:
            pass
    if not cands:
        cands.append(_norm(rng.standard_normal((272, 4))))
    best = min(cands, key=_max_cos).copy()
    best_val = _max_cos(best)
    sym_best_list = []
    # ---- Phase 1: symmetric orbit search ----
    sym_deadline = t0 + 62.0
    configs = [(17, 16, 3), (17, 16, 1), (17, 16, 5), (34, 8, 3), (34, 8, 1), (17, 16, 7)]
    taus_sched = [0.02, 0.01, 0.005, 0.002, 0.001]
    for ci, (R, K, tw) in enumerate(configs):
        if time.time() > sym_deadline - 2.0:
            break
        # fewer iters for larger R to save time
        mi = 120 if R > 20 else 180
        try:
            Xs, vs = _optim_sym(R, K, tw, rng, taus_sched, mi, sym_deadline - 1.0)
        except Exception:
            continue
        sym_best_list.append((vs, Xs.copy()))
        if vs < best_val:
            best, best_val = Xs.copy(), vs
        if time.time() > sym_deadline - 2.0:
            break
        # second seed for first two configs if time allows
        if ci < 2 and time.time() < sym_deadline - 12.0:
            try:
                Xs2, vs2 = _optim_sym(R, K, tw, rng, [0.015, 0.006, 0.0015], mi, sym_deadline - 1.0)
            except Exception:
                continue
            sym_best_list.append((vs2, Xs2.copy()))
            if vs2 < best_val:
                best, best_val = Xs2.copy(), vs2
    sym_best_list.sort(key=lambda t: t[0])
    # keep top symmetric bases + current best
    bases = []
    for vs, Xs in sym_best_list[:3]:
        bases.append(Xs)
    bases.append(best.copy())
    # deduplicate roughly: keep at most 4
    bases = bases[:4]
    # ---- Phase 2: break symmetry ----
    sigmas = [1e-3, 1e-2, 3e-2]
    branch_count = 0
    bi = 0
    while time.time() < deadline - 38.0 and bi < 14:
        base = bases[bi % len(bases)]
        sigma = sigmas[bi % len(sigmas)]
        bi += 1
        branch_count += 1
        y = base + sigma * rng.standard_normal((272, 4))
        y = _norm(y)
        # anneal params cycle
        if branch_count % 3 == 0:
            y = _adam_full(y, 350, 0.014, 0.0012, 0.009, 0.0009)
        elif branch_count % 3 == 1:
            y = _adam_full(y, 420, 0.022, 0.0022, 0.013, 0.0011)
        else:
            y = _adam_full(y, 350, 0.008, 0.0007, 0.006, 0.0005)
        v = _max_cos(y)
        if v < best_val:
            best, best_val = y.copy(), v
            # snap immediately
            if time.time() < deadline - 34.0:
                ys = _lp_polish(y, seconds=2.5, deadline=deadline - 32.0)
                vs = _max_cos(ys)
                if vs < best_val:
                    best, best_val = ys, vs
        elif v < best_val + 0.003 and (branch_count % 3 == 0):
            if time.time() < deadline - 34.0:
                ys = _lp_polish(y, seconds=1.5, deadline=deadline - 32.0)
                vs = _max_cos(ys)
                if vs < best_val:
                    best, best_val = ys, vs
        # also try void reinsertion of worst point occasionally
        if branch_count % 4 == 0 and time.time() < deadline - 40.0:
            G = y @ y.T
            np.fill_diagonal(G, -2.0)
            pw = G.max(axis=1)
            wi = int(np.argmax(pw))
            others = np.delete(y, wi, axis=0)
            cand = rng.standard_normal((6000, 4))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            sc = (cand @ others.T).max(axis=1)
            y2 = y.copy()
            y2[wi] = cand[int(np.argmin(sc))]
            y2 = _adam_full(y2, 250, 0.01, 0.001, 0.008, 0.0008)
            v2 = _max_cos(y2)
            if v2 < best_val:
                best, best_val = y2.copy(), v2
    remain = deadline - time.time() - 1.5
    if remain > 3.0:
        best = _lp_polish(best, seconds=min(remain, 32.0), deadline=deadline - 1.0)
    return _norm(best)
