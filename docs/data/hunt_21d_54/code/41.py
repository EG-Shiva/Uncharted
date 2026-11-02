import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.linalg import null_space

def _proj(x):
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    x = x / nrm
    return np.nan_to_num(x)

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _contact_graph_flat(x, tol=1e-9):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    iu = np.triu_indices(n, 1)
    c = G[iu]
    b = c > (m - tol)
    return b, m, c

def _lp_polish(x, seconds, cap=2500):
    x = _proj(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.2:
            break
        G = x @ x.T
        c = G[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > cap:
            idx = np.argpartition(-c[sel], cap)[:cap]
            I, J = I[idx], J[idx]
            csv = c[sel][idx]
        else:
            csv = c[sel]
            p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                          shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs", options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y = _proj(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def _tangent_bases(x):
    n, d = x.shape
    Tb = np.zeros((n, d, d - 1))
    for i in range(n):
        ns = null_space(x[i:i+1, :])  # d x (d-1)
        # null_space returns (d, d-1) orthonormal
        if ns.shape != (d, d - 1):
            # fallback: QR
            q, _ = np.linalg.qr(np.random.randn(d, d))
            # project out x
            q = q - np.outer(x[i], (x[i] @ q))
            q, _ = np.linalg.qr(q[:, :d-1])
            ns = q
        Tb[i] = ns
    return Tb

def _null_modes(x, thresh=1e-3, nmode=20, cap=3000):
    n, d = x.shape
    dm = d - 1
    N = n * dm
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    m = float(c.max())
    sel = c > (m - thresh)
    I0, J0 = iu[0][sel], iu[1][sel]
    p0 = len(I0)
    if p0 > cap:
        idx = np.argpartition(-c[sel], cap)[:cap]
        I0, J0 = I0[idx], J0[idx]
    p = len(I0)
    if p == 0:
        return None, m, p
    Tb = _tangent_bases(x)  # n x d x dm
    # Build dense A (p x N)
    A = np.zeros((p, N), dtype=float)
    # For speed: vectorize per pair loop (p up to 3000, N=1080 -> loop ok)
    for k in range(p):
        i, j = int(I0[k]), int(J0[k])
        # row block: Tb[i].T @ x[j] at position i*dm, Tb[j].T @ x[i] at j*dm
        A[k, i*dm:(i+1)*dm] = Tb[i].T @ x[j]
        A[k, j*dm:(j+1)*dm] = Tb[j].T @ x[i]
    # Gram
    Gram = A.T @ A  # N x N
    # eigh smallest
    try:
        w, V = np.linalg.eigh(Gram)
    except Exception:
        return None, m, p
    # w ascending; take nmode smallest vectors (columns)
    K = min(nmode, N)
    modes = V[:, :K]  # N x K
    return (modes, Tb, I0, J0), m, p

def solve():
    T0 = time.time()
    LIMIT = 168.0
    EXPLORE_END = T0 + 148.0  # 20s reserved for final polish
    rng = np.random.default_rng(7)
    try:
        b0 = np.load("best.npy")
    except Exception:
        b0 = np.load("record.npy")
    b0 = _proj(np.nan_to_num(np.asarray(b0, dtype=float)))
    if b0.shape != (54, 21):
        b0 = _proj(rng.normal(size=(54, 21)))
    try:
        r0 = _proj(np.asarray(np.load("record.npy"), dtype=float))
        if r0.shape == (54, 21):
            if _max_cos(r0) < _max_cos(b0):
                b0 = r0
    except Exception:
        pass
    best = b0.copy()
    best_val = _max_cos(best)
    n, d = 54, 21
    dm = d - 1
    # brief ensure at local opt (2s)
    if LIMIT - (time.time() - T0) > 60:
        best = _lp_polish(best, 2.0)
        best_val = _max_cos(best)
    # incumbent contact graph for Hamming
    inc_flat, inc_m, _ = _contact_graph_flat(best, tol=1e-9)
    thetas = [0.05, 0.1, 0.2, 0.35, 0.5]
    alphas = [float(np.tan(t)) for t in thetas]
    NMODE = 20
    # outer basin loop: recompute modes whenever best changes or periodically
    mode_data = None
    recompute = True
    it_outer = 0
    while time.time() - T0 < EXPLORE_END - 4.0:
        it_outer += 1
        if recompute:
            try:
                mode_data, m_cur, p_cur = _null_modes(best, thresh=1e-3, nmode=NMODE, cap=2500)
            except Exception:
                mode_data = None
            recompute = False
            if mode_data is None:
                # fallback: random tangent kick
                mode_data = None
        if mode_data is None:
            # simple random tangent trial
            cand_dirs = []
            Tb_fallback = _tangent_bases(best)
            for _ in range(5):
                y = rng.normal(size=(n*dm,))
                dx = np.zeros((n, d))
                for i in range(n):
                    yi = y[i*dm:(i+1)*dm]
                    dx[i] = Tb_fallback[i] @ yi
                mx = np.linalg.norm(dx, axis=1).max() + 1e-12
                dx /= mx
                cand_dirs.append(dx)
        else:
            modes, Tb, I0, J0 = mode_data
            K = modes.shape[1]
            cand_dirs = []
            # make ~6 random combos per outer iter
            for _ in range(6):
                kpick = int(rng.integers(2, 6))  # 2-5 modes
                cols = rng.choice(K, size=kpick, replace=False)
                coeff = rng.normal(size=kpick)
                coeff /= (np.linalg.norm(coeff) + 1e-12)
                y = modes[:, cols] @ coeff  # N
                dx = np.zeros((n, d))
                for i in range(n):
                    yi = y[i*dm:(i+1)*dm]
                    dx[i] = Tb[i] @ yi
                mx = np.linalg.norm(dx, axis=1).max() + 1e-12
                dx /= mx
                cand_dirs.append(dx)
        # line search over thetas for each dir
        trials = []
        for dx in cand_dirs:
            for a in alphas:
                if time.time() - T0 > EXPLORE_END - 4.0:
                    break
                y = best + a * dx
                y = _proj(y)
                g = y @ y.T
                np.fill_diagonal(g, -2.0)
                mv = float(g.max())
                if mv >= 0.12:
                    continue
                # Hamming vs incumbent
                iu = np.triu_indices(n, 1)
                # thresholded at own max
                bt = g[iu] > (mv - 1e-9)
                ham = int(np.sum(bt != inc_flat))
                trials.append((mv, ham, y))
        if not trials:
            recompute = True
            continue
        # score: first by max-cosine, second by Hamming (prefer different)
        # keep trials with materially different topology: ham > 60
        filt = [t for t in trials if t[1] > 60]
        if not filt:
            filt = sorted(trials, key=lambda t: -t[1])[:4]
        # sort filt by mv, but boost high-ham slightly: pick diverse set
        filt_sorted = sorted(filt, key=lambda t: t[0])
        # take up to 2 best to polish (each 3-4s)
        npol = 2 if (EXPLORE_END - (time.time() - T0)) > 30 else 1
        polished_any = False
        for k in range(min(npol, len(filt_sorted))):
            if time.time() - T0 > EXPLORE_END - 3.5:
                break
            _, ham, y0 = filt_sorted[k]
            # also skip near-duplicate low-ham unless very good max
            budget = 3.0
            if time.time() - T0 > EXPLORE_END - 10:
                budget = 2.0
            yc = _lp_polish(y0, budget)
            vc = _max_cos(yc)
            if vc < best_val - 1e-12:
                best, best_val = yc, vc
                inc_flat, inc_m, _ = _contact_graph_flat(best, tol=1e-9)
                recompute = True
                polished_any = True
                break
        if not polished_any:
            # if no improvement, periodically refresh modes from a mildly
            # perturbed diverse trial to explore further (without committing)
            # pick a mid trial and make it the mode base occasionally
            if it_outer % 3 == 0:
                # choose trial with decent max but large ham as new exploration center?
                # do not replace best, but recompute modes around it for next iter
                # implement by temporarily shifting mode base
                div = sorted(filt, key=lambda t: -t[1])[0]
                if div[0] < 0.10:
                    try:
                        mode_data2, _, _ = _null_modes(div[2], thresh=1e-3, nmode=NMODE, cap=2500)
                        if mode_data2 is not None:
                            mode_data = mode_data2
                    except Exception:
                        pass
            if it_outer % 5 == 0:
                recompute = True
        # safety: if best improved a lot, continue immediately
    # final LP polish with reserved time
    time_left = LIMIT - (time.time() - T0) - 1.0
    if time_left > 1.0:
        best = _lp_polish(best, min(time_left, 20.0))
    best = _proj(best)
    return best
