import time
import numpy as np

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _adam_lse(x0, steps, K0, K1, lr, b1=0.9, b2=0.999, eps=1e-8):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1p = 1.0
    b2p = 1.0
    la0 = np.log(K0); la1 = np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        E = K * (G - gmax)
        # clip for stability
        np.clip(E, -60.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1; b2p *= b2
        m *= b1; m += (1.0 - b1) * gt
        v *= b2; v += (1.0 - b2) * (gt * gt)
        mh = m / (1.0 - b1p)
        vh = v / (1.0 - b2p)
        step = lr * mh / (np.sqrt(vh) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x = _norm(x)
        if t % 100 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _rowmax(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def _void_dirs(x, k, rng, nsamp=20000):
    # find k diverse void directions minimizing max dot
    C = _norm(rng.normal(size=(nsamp, 6)))
    S = C @ x.T  # nsamp x 256
    m = S.max(axis=1)
    idx = np.argpartition(m, k)[:k]
    return C[idx]

def solve():
    t0 = time.time()
    T_END = t0 + 170.0
    rng = np.random.default_rng(0)
    # load incumbent
    x_best = None; best = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (256, 6) and np.all(np.isfinite(a)):
                a = _norm(a)
                vv = _max_cos(a)
                if vv < best:
                    best = vv; x_best = a.copy()
        except Exception:
            pass
    if x_best is None:
        x_best = _norm(rng.normal(size=(256, 6)))
        best = _max_cos(x_best)
    fallback = x_best.copy(); fallback_val = best

    cur_x = x_best.copy(); cur_v = best

    # Phase A: de novo diverse basins with moderate-K annealing
    try:
        n_trials = 6
        for ti in range(n_trials):
            if time.time() > T_END - 120:
                break
            if ti == 0:
                x0 = fallback.copy()
                # small jitter to leave single-pair basin
                Z = rng.normal(size=x0.shape)
                Z -= np.sum(Z * x0, axis=1, keepdims=True) * x0
                zn = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-16
                x0 = _norm(x0 + 0.01 * Z / zn)
            else:
                x0 = _norm(rng.normal(size=(256, 6)))
            # two stage: spread then focus
            xa, va = _adam_lse(x0, 700, 80.0, 3500.0, 0.010)
            if time.time() > T_END - 100:
                if va < cur_v:
                    cur_x, cur_v = xa, va
                break
            xb, vb = _adam_lse(xa, 400, 500.0, 3000.0, 0.005)
            if vb < cur_v:
                cur_x, cur_v = xb, vb
            # also keep global best
            if cur_v < best:
                x_best, best = cur_x.copy(), cur_v
    except Exception:
        pass

    # ensure we start hopping from global best
    if best < cur_v:
        cur_x, cur_v = x_best.copy(), best
    else:
        x_best, best = cur_x.copy(), cur_v
    if fallback_val < best:
        x_best, best = fallback.copy(), fallback_val
        cur_x, cur_v = x_best.copy(), best

    # Phase B: hub-kick / void-refill hopping with short moderate-K repair
    try:
        it = 0
        while time.time() < T_END - 25:
            it += 1
            mode = it % 4
            x0 = cur_x.copy()
            rm, Gfull = _rowmax(cur_x)
            order = np.argsort(-rm)  # worst first
            if mode == 0:
                # perturb worst 8 with tangent noise amp 0.02
                k = 8
                idx = order[:k]
                Z = rng.normal(size=(k, 6))
                xi = x0[idx]
                Z -= np.sum(Z * xi, axis=1, keepdims=True) * xi
                zn = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-16
                amp = 0.025
                x0[idx] = _norm(xi + amp * Z / zn)
            elif mode == 1:
                # replace worst 2 by voids
                k = 2
                idx = order[:k]
                V = _void_dirs(cur_x, k, rng, nsamp=15000)
                x0[idx] = V
            elif mode == 2:
                # perturb worst 24 lightly + global tiny shake
                k = 24
                idx = order[:k]
                Z = rng.normal(size=(k, 6))
                xi = x0[idx]
                Z -= np.sum(Z * xi, axis=1, keepdims=True) * xi
                zn = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-16
                x0[idx] = _norm(xi + 0.012 * Z / zn)
            else:
                # global jitter
                Z = rng.normal(size=x0.shape)
                Z -= np.sum(Z * x0, axis=1, keepdims=True) * x0
                zn = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-16
                x0 = _norm(x0 + 0.006 * Z / zn)
            # short repair with plateau-friendly K
            xr, vr = _adam_lse(x0, 350, 400.0, 3000.0, 0.006)
            if vr < cur_v - 1e-12:
                cur_x, cur_v = xr, vr
                if vr < best:
                    x_best, best = xr.copy(), vr
            elif vr < cur_v:
                cur_x, cur_v = xr, vr
            # occasional acceptance of equal to move across plateau
            elif it % 10 == 0 and vr < cur_v + 5e-6:
                # random walk on plateau to change topology
                cur_x, cur_v = xr, vr
    except Exception:
        pass

    # Phase C: final equalization polish with constant moderate K, small lr
    try:
        while time.time() < T_END - 6:
            # longer fine run
            xf, vf = _adam_lse(cur_x, 600, 2500.0, 2500.0, 0.0025)
            if vf < best:
                x_best, best = xf.copy(), vf
            cur_x, cur_v = xf, vf
            if time.time() < T_END - 14:
                xg, vg = _adam_lse(cur_x, 600, 1200.0, 1800.0, 0.002)
                if vg < best:
                    x_best, best = xg.copy(), vg
                cur_x, cur_v = xg, vg
            else:
                break
    except Exception:
        pass

    if best > fallback_val:
        return fallback
    # tiny shrink not needed; ensure normalized
    x_best = _norm(x_best)
    return x_best
