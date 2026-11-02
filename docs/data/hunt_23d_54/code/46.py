import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

D = 23
N_ORB = 23
N = 54
_BUILD_IDX = (np.arange(D)[None, :] - np.arange(N_ORB)[:, None]) % D
_ACC_IDX = (np.arange(D)[None, :] + np.arange(N_ORB)[:, None]) % D
_RK = np.arange(N_ORB)[:, None]

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _build_all(a, b, c):
    A = a[_BUILD_IDX]
    B = b[_BUILD_IDX]
    Call = c[_BUILD_IDX]
    return A, B, Call

def _greedy_select(A, B, Call):
    base = np.concatenate([A, B], axis=0)  # 46x23
    # dots
    Dcb = Call @ base.T  # 23x46
    Dcc = Call @ Call.T  # 23x23
    base_max = float((base @ base.T + np.eye(46) * -10).max())
    selected = []
    sel_mask = np.zeros(23, dtype=bool)
    cur_max_to_sel = np.full(23, -2.0)
    for _ in range(8):
        best_i = -1
        best_v = 1e9
        for cand in range(23):
            if sel_mask[cand]:
                continue
            m1 = float(Dcb[cand].max())
            if selected:
                m2 = float(cur_max_to_sel[cand] if False else 0)
                # max to already selected
                m2 = -2.0
                for s in selected:
                    v = Dcc[cand, s]
                    if v > m2:
                        m2 = v
                m = m1 if m1 > m2 else m2
            else:
                m = m1
            if m < best_v:
                best_v = m
                best_i = cand
        selected.append(best_i)
        sel_mask[best_i] = True
    # one swap pass
    def sel_max(sel):
        if len(sel) == 0:
            return -2.0
        S = Call[sel]  # kx23
        m = base_max
        if len(sel) > 1:
            G = S @ S.T
            np.fill_diagonal(G, -2.0)
            m = max(m, float(G.max()))
        Dm = S @ base.T
        m = max(m, float(Dm.max()))
        return m
    cur = sel_max(selected)
    for _ in range(2):
        improved = False
        for pos in range(len(selected)):
            for cand in range(23):
                if sel_mask[cand]:
                    continue
                trial = selected.copy()
                trial[pos] = cand
                v = sel_max(trial)
                if v < cur - 1e-12:
                    sel_mask[selected[pos]] = False
                    sel_mask[cand] = True
                    selected[pos] = cand
                    cur = v
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return selected

def _config_from_seeds(a, b, c):
    A, B, Call = _build_all(a, b, c)
    sel = _greedy_select(A, B, Call)
    X = np.concatenate([A, B, Call[sel]], axis=0)
    return X, sel, (A, B, Call)

def _trial_optimize(a0, b0, c0, steps=2000, lr=0.02):
    a = a0.copy(); b = b0.copy(); c = c0.copy()
    ma = np.zeros_like(a); va = np.zeros_like(a)
    mb = np.zeros_like(b); vb = np.zeros_like(b)
    mc = np.zeros_like(c); vc = np.zeros_like(c)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lB0, lB1 = np.log(50.0), np.log(800.0)
    for t in range(steps):
        frac = t / max(steps - 1, 1)
        beta = float(np.exp(lB0 + (lB1 - lB0) * frac))
        A, Bm, Call = _build_all(a, b, c)
        sel = _greedy_select(A, Bm, Call)
        Cs = Call[sel]
        X = np.concatenate([A, Bm, Cs], axis=0)
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        s = float(E.sum() / 2.0) + 1e-300
        Ws = E / s  # symmetric, sum=2? E/s sum=2, but Ws should sum to 2? check: W_ij=E_ij/s, symmetric sum=2. grad uses Ws@X /2? derive: S=sum_{i<j}E_ij=s. W_ij=E_ij/s. Ws_ij=W_ij for i!=j. E/s has that. E.sum()=2s so E/s sum=2 correct (counts both triangles). grad_i=sum_j Ws_ij x_j = (Ws@X)_i. So use Ws=E/s.
        # Actually Ws = E/s is correct (not /2)
        Gr = Ws @ X
        # split
        GA = Gr[0:23]; GB = Gr[23:46]; GCs = Gr[46:54]
        # accumulate to seeds
        # da[q]=sum_k GA[k,(q+k)%d]
        MA = GA[_RK, _ACC_IDX]  # 23x23? _RK is 23x1, _ACC_IDX 23x23 -> broadcast? need proper
        # _RK shape (23,1), _ACC_IDX (23,23): GA[_RK, _ACC_IDX] gives (23,23)
        da = MA.sum(axis=0)
        MB = GB[_RK, _ACC_IDX]
        db = MB.sum(axis=0)
        dc = np.zeros_like(c)
        for ii, k in enumerate(sel):
            g = GCs[ii]
            dc += np.roll(g, -int(k))
        # riemannian project
        da_t = da - float(da @ a) * a
        db_t = db - float(db @ b) * b
        dc_t = dc - float(dc @ c) * c
        # adam
        bc1 = 1 - b1 ** (t + 1)
        bc2 = 1 - b2 ** (t + 1)
        ma = b1 * ma + (1 - b1) * da_t
        va = b2 * va + (1 - b2) * (da_t * da_t)
        mb = b1 * mb + (1 - b1) * db_t
        vb = b2 * vb + (1 - b2) * (db_t * db_t)
        mc = b1 * mc + (1 - b1) * dc_t
        vc = b2 * vc + (1 - b2) * (dc_t * dc_t)
        sa = (ma / bc1) / (np.sqrt(va / bc2) + eps)
        sb = (mb / bc1) / (np.sqrt(vb / bc2) + eps)
        sc = (mc / bc1) / (np.sqrt(vc / bc2) + eps)
        a = a - lr * sa
        b = b - lr * sb
        c = c - lr * sc
        a /= np.linalg.norm(a); b /= np.linalg.norm(b); c /= np.linalg.norm(c)
    Xf, self_, _ = _config_from_seeds(a, b, c)
    return Xf, _max_cos(Xf), (a, b, c)

def _smooth_full(Xinit, steps, beta0, beta1, lr0, lr1):
    X = Xinit.copy()
    n, d = X.shape
    m = np.zeros_like(X); v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for t in range(steps):
        f = t / max(steps - 1, 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        mm = float(G.max())
        E = np.exp(beta * (G - mm))
        np.fill_diagonal(E, 0.0)
        s = float(E.sum() / 2.0) + 1e-300
        Ws = E / s
        Gr = Ws @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        m = b1 * m + (1 - b1) * Gt
        v = b2 * v + (1 - b2) * (Gt * Gt)
        bc1 = 1 - b1 ** (t + 1); bc2 = 1 - b2 ** (t + 1)
        step = (m / bc1) / (np.sqrt(v / bc2) + eps)
        X = X - lr * step
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _lp_polish(X, seconds, t0=None, budget=1e9):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and (time.time() - t0) > budget:
            break
        if radius < 1e-14:
            break
        G = X @ X.T
        c_all = G[iu]
        m = float(c_all.max())
        sel = c_all > m - max(20 * radius, 1e-9)
        if np.count_nonzero(sel) > 900:
            idx = np.where(sel)[0]
            order = np.argsort(c_all[idx])[::-1][:900]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu[0][sel], iu[1][sel]
        c = c_all[sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            X, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return X

def solve():
    t0 = time.time()
    BUDGET = 172.0
    rng = np.random.RandomState(12345)
    # fallback best
    try:
        xb = np.load("best.npy").astype(float)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    except Exception:
        xb = rng.randn(54, 23); xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    if xb.shape != (54, 23):
        tmp = rng.randn(54, 23); tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
        xb = tmp
    try:
        xr = np.load("record.npy").astype(float)
        xr /= np.linalg.norm(xr, axis=1, keepdims=True)
        if xr.shape == (54, 23) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass
    best_val = _max_cos(xb)
    x_best = xb.copy()

    # Phase 1: symmetry-restricted search, strict 120s timer, >=30 triples
    sym_cands = []
    n_target = 32
    # estimate steps: 2000 as required
    for trial in range(n_target):
        elapsed = time.time() - t0
        if elapsed > 118.0:
            break
        # allow fewer steps if running out of time but still reach 30 trials
        remaining = 118.0 - elapsed
        trials_left = n_target - trial
        # if not enough time for 2000-step trials, reduce steps
        steps = 2000
        if remaining < trials_left * 1.0:
            steps = max(400, int(remaining / max(trials_left, 1) * 1800))
        a0 = rng.randn(23); a0 /= np.linalg.norm(a0)
        b0 = rng.randn(23); b0 /= np.linalg.norm(b0)
        c0 = rng.randn(23); c0 /= np.linalg.norm(c0)
        try:
            Xf, vf, _ = _trial_optimize(a0, b0, c0, steps=steps, lr=0.02)
        except Exception:
            continue
        sym_cands.append((vf, Xf))
        if vf < best_val:
            best_val = vf; x_best = Xf.copy()
        # also if time exceeded 120s stop
        if time.time() - t0 > 120.0:
            break
    # ensure at least 30 attempts even if broke early due to speed: quick short trials
    if len(sym_cands) < 30 and (time.time() - t0) < 120.0:
        need = 30 - len(sym_cands)
        for _ in range(need):
            if time.time() - t0 > 119.0:
                break
            a0 = rng.randn(23); a0 /= np.linalg.norm(a0)
            b0 = rng.randn(23); b0 /= np.linalg.norm(b0)
            c0 = rng.randn(23); c0 /= np.linalg.norm(c0)
            try:
                Xf, vf, _ = _trial_optimize(a0, b0, c0, steps=400, lr=0.02)
            except Exception:
                continue
            sym_cands.append((vf, Xf))
            if vf < best_val:
                best_val = vf; x_best = Xf.copy()

    # Phase 2: gradual symmetry release
    sym_cands.sort(key=lambda z: z[0])
    keep = [x for _, x in sym_cands[:3]]
    # always include current best
    if len(keep) == 0:
        keep = [x_best.copy()]
    sigmas = [1e-3, 3e-3, 1e-2]
    for Xsrc in list(keep):
        for sig in sigmas:
            if time.time() - t0 > BUDGET - 22.0:
                break
            Xn = Xsrc.copy()
            Nse = rng.randn(*Xn.shape)
            Nse -= np.sum(Nse * Xn, axis=1, keepdims=True) * Xn
            Xn = Xn + sig * Nse
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            try:
                Xs = _smooth_full(Xn, steps=350, beta0=100, beta1=1200, lr0=0.01, lr1=0.001)
            except Exception:
                continue
            vs = _max_cos(Xs)
            sym_cands.append((vs, Xs))
            if vs < best_val:
                best_val = vs; x_best = Xs.copy()
        if time.time() - t0 > BUDGET - 22.0:
            break
    # extra cheap smooth on overall best
    if time.time() - t0 < BUDGET - 21.0:
        try:
            Xs = _smooth_full(x_best.copy(), steps=250, beta0=200, beta1=2000, lr0=0.008, lr1=0.0008)
            vs = _max_cos(Xs)
            if vs < best_val:
                best_val = vs; x_best = Xs.copy()
        except Exception:
            pass

    # Phase 3: reserve final ~20s for one trust-region LP polish on best
    rem = BUDGET - (time.time() - t0)
    lp_sec = min(20.0, max(rem - 1.0, 2.0))
    if lp_sec > 0.5:
        try:
            Xp = _lp_polish(x_best.copy(), seconds=lp_sec, t0=t0, budget=BUDGET)
            vp = _max_cos(Xp)
            if vp < best_val:
                best_val = vp; x_best = Xp.copy()
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
