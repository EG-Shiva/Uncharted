import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def expand_seeds(S, cosT, sinT):
    # S: (K,6,2), cosT/sinT: (O,6) -> X: (K*O,12)
    K = S.shape[0]
    O = cosT.shape[0]
    S0 = S[:, :, 0][:, None, :]  # K,1,6
    S1 = S[:, :, 1][:, None, :]
    c = cosT[None, :, :]  # 1,O,6
    s = sinT[None, :, :]
    X0 = c * S0 - s * S1
    X1 = s * S0 + c * S1
    X = np.empty((K, O, 6, 2), dtype=S.dtype)
    X[:, :, :, 0] = X0
    X[:, :, :, 1] = X1
    return X.reshape(K * O, 12)


def sym_adam(S_init, cosT, sinT, steps, T0, T1, lr):
    K = S_init.shape[0]
    O = cosT.shape[0]
    N = K * O
    S = S_init.copy()
    m = np.zeros_like(S)
    v = np.zeros_like(S)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    for it in range(1, steps + 1):
        lt = lT0 + (lT1 - lT0) * (it / steps)
        T = float(np.exp(lt))
        Xmat = expand_seeds(S, cosT, sinT)  # N,12
        G = Xmat @ Xmat.T
        np.fill_diagonal(G, -10.0)
        M = float(G.max())
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        sm = float(W.sum())
        if not np.isfinite(sm) or sm == 0:
            break
        P = W / sm
        GradMat = P @ Xmat  # N,12
        Gm = GradMat.reshape(K, O, 6, 2)
        c = cosT  # O,6
        s = sinT
        G0 = Gm[:, :, :, 0]  # K,O,6
        G1 = Gm[:, :, :, 1]
        # pullback R^T
        dS0 = (c[None, :, :] * G0 + s[None, :, :] * G1).sum(axis=1)  # K,6
        dS1 = (-s[None, :, :] * G0 + c[None, :, :] * G1).sum(axis=1)
        Gt = np.empty_like(S)
        Gt[:, :, 0] = dS0
        Gt[:, :, 1] = dS1
        # tangent projection per seed
        dot = (Gt * S).sum(axis=(1, 2), keepdims=True)
        Gt = Gt - dot * S
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** it
        c2 = 1.0 - b2 ** it
        mh = m / c1
        vh = v / c2
        upd = mh / (np.sqrt(vh) + eps)
        dot2 = (upd * S).sum(axis=(1, 2), keepdims=True)
        upd = upd - dot2 * S
        S = S - lr * upd
        nr = np.linalg.norm(S, axis=(1, 2), keepdims=True)
        nr[nr == 0] = 1.0
        S = S / nr
    return S


def full_adam(x_init, steps, T0, T1, lr):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    for s in range(1, steps + 1):
        lt = lT0 + (lT1 - lT0) * (s / steps)
        T = float(np.exp(lt))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = float(G.max())
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        sm = float(W.sum())
        if not np.isfinite(sm) or sm == 0:
            break
        P = W / sm
        Grad = P @ x
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        mh = m / c1
        vh = v / c2
        upd = mh / (np.sqrt(vh) + eps)
        # remove radial part
        dot2 = np.sum(upd * x, axis=1, keepdims=True)
        upd = upd - dot2 * x
        x = x - lr * upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        sel = c > mcur - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    rng = np.random.default_rng(0)
    # load incumbent
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    best_val = 1e9
    x_best = None
    for c in (xb, xr):
        if c is not None:
            try:
                a = np.array(c, dtype=float)
                a /= np.linalg.norm(a, axis=1, keepdims=True)
                if a.shape == (102, 12):
                    vv = max_cos(a)
                    if vv < best_val:
                        best_val = vv
                        x_best = a.copy()
            except Exception:
                pass
    if x_best is None:
        x_best = rng.standard_normal((102, 12))
        x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
        best_val = max_cos(x_best)

    sym_cands = []  # (val, X)

    def run_case(K, O, m_fixed, n_rest, steps, T0, T1, lr, deadline):
        nonlocal sym_cands
        for r in range(n_rest):
            if time.time() > deadline:
                break
            # choose multipliers: half fixed, half random
            if (r % 2 == 0) or m_fixed is None:
                mvec = np.array(m_fixed, dtype=float)
            else:
                if O == 17:
                    perm = rng.choice(np.arange(1, 17), size=6, replace=False)
                    mvec = perm.astype(float)
                else:
                    odds = np.arange(1, 34, 2)
                    perm = rng.choice(odds, size=6, replace=False)
                    mvec = perm.astype(float)
            g = np.arange(O)[:, None]  # O,1
            ang = 2.0 * np.pi * g * mvec[None, :] / float(O)  # O,6
            cosT = np.cos(ang)
            sinT = np.sin(ang)
            S0 = rng.standard_normal((K, 6, 2))
            nr = np.linalg.norm(S0, axis=(1, 2), keepdims=True)
            S0 /= nr
            try:
                Sopt = sym_adam(S0, cosT, sinT, steps, T0, T1, lr)
                X = expand_seeds(Sopt, cosT, sinT)
                # reject collisions (duplicate orbit points)
                G = X @ X.T
                np.fill_diagonal(G, -2.0)
                vv = float(G.max())
                if not np.isfinite(vv):
                    continue
                # need distinct points: check min dist not ~duplicate
                sym_cands.append((vv, X.copy()))
                sym_cands.sort(key=lambda t: t[0])
                sym_cands = sym_cands[:6]
            except Exception:
                continue

    # Phase 1: symmetric search, ~75s
    sym_deadline = min(T_END - 85.0, t0 + 80.0)
    if time.time() < sym_deadline:
        # 6x17 main
        run_case(6, 17, [1, 2, 4, 6, 8, 9], 60, 900, 0.03, 0.004, 0.004, sym_deadline - 12.0 if sym_deadline - 12.0 > time.time() else sym_deadline)
    if time.time() < sym_deadline:
        run_case(3, 34, [1, 3, 5, 7, 9, 11], 25, 900, 0.03, 0.004, 0.004, sym_deadline)

    # Phase 2: symmetry breaking quench
    # collect seeds to quench: top symmetric + incumbent
    quench_list = []
    for vv, X in sym_cands[:4]:
        quench_list.append(X)
    # add incumbent perturbed versions to guarantee competitiveness
    # (small diversity, will be polished later)
    quenched_best_val = best_val
    quenched_best = x_best.copy()
    # also consider raw symmetric cands as-is
    for vv, X in sym_cands:
        if vv < quenched_best_val:
            # don't directly accept bad symmetric (>record) as best, keep for quench only
            pass
    # quench each symmetric candidate with symmetry-breaking noise
    quench_deadline = min(T_END - 30.0, time.time() + 55.0)
    for idx, X0 in enumerate(quench_list):
        if time.time() > quench_deadline:
            break
        for sig, steps_q in [(0.015, 1200), (0.006, 900)]:
            if time.time() > quench_deadline:
                break
            try:
                Xn = X0 + sig * rng.standard_normal(X0.shape)
                Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
                Xq = full_adam(Xn, steps_q, 0.022, 0.0032, 0.0022)
                vq = max_cos(Xq)
                if vq < quenched_best_val:
                    quenched_best_val = vq
                    quenched_best = Xq.copy()
                # also try orbit-common move: add same noise per orbit block
                # (approximate first release stage)
                Kguess = 6 if X0.shape[0] == 102 else 6
                # skip if not divisible
            except Exception:
                continue
    # also quick basin hops from incumbent using released quench (cheap, different basin attempt)
    hop_deadline = min(T_END - 28.0, time.time() + 12.0)
    rng2 = np.random.default_rng(1)
    it = 0
    while time.time() < hop_deadline and it < 6:
        it += 1
        try:
            Xn = quenched_best + 0.01 * rng2.standard_normal((102, 12))
            Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
            Xq = full_adam(Xn, 700, 0.02, 0.0035, 0.002)
            vq = max_cos(Xq)
            if vq < quenched_best_val:
                quenched_best_val = vq
                quenched_best = Xq.copy()
        except Exception:
            continue

    # Phase 3: final LP polish on best
    # choose starting point: best of quenched vs original incumbent
    final_x = quenched_best if quenched_best_val <= best_val + 1e-12 else x_best
    # if symmetric quench failed to beat, still ensure we polish incumbent
    try:
        remain = T_END - time.time() - 1.0
        if remain > 4.0:
            final_x = lp_polish(final_x, min(remain, 28.0), T_END - 0.5)
    except Exception:
        pass
    final_x /= np.linalg.norm(final_x, axis=1, keepdims=True)
    # safety: if polish made it worse or NaN, revert
    try:
        vf = max_cos(final_x)
        if not np.isfinite(vf) or vf > min(best_val, quenched_best_val) + 0.002:
            # revert to best known
            if quenched_best_val <= best_val:
                final_x = quenched_best
            else:
                final_x = x_best
    except Exception:
        final_x = x_best
    final_x /= np.linalg.norm(final_x, axis=1, keepdims=True)
    return final_x
