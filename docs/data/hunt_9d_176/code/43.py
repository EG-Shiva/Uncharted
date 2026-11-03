import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def smooth_quench(X, steps, a0, a1, lr0, rng=None, deadline=None):
    n, d = X.shape
    x = X.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        if deadline is not None and (t % 200 == 0) and time.time() > deadline:
            break
        frac = t / max(1, steps - 1)
        alpha = a0 * ((a1 / a0) ** frac)
        lr = lr0 * (1.0 - 0.5 * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E) * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1.0 - b1) * Gt
        v = b2 * v + (1.0 - b2) * (Gt * Gt)
        mhat = m / (1.0 - b1 ** (t + 1))
        vhat = v / (1.0 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        x = x - lr * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        x /= nr
    return x


def per_point_worst(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G


def greedy_hole(X, idx, rng, ntrials=256):
    n, d = X.shape
    keep = np.ones(n, dtype=bool)
    keep[idx] = False
    cur = X[keep].copy()
    Y = X.copy()
    for j in idx:
        T = rng.standard_normal((ntrials, d))
        T /= np.linalg.norm(T, axis=1, keepdims=True)
        D = T @ cur.T
        mm = D.max(axis=1)
        pick = int(np.argmin(mm))
        Y[j] = T[pick]
        cur = np.vstack([cur, T[pick][None, :]])
    return Y


def lp_polish_deadline(X, deadline, r_init=0.004, cap=2500):
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gf = x @ x.T
        c_all = Gf[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idxa = np.where(sel)[0]
            part = np.argpartition(-c_all[idxa], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idxa[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]; J = iu0[1][sel]
        c = c_all[sel]
        p = cnt
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(nr)) or np.any(nr < 1e-12):
            radius *= 0.5
            continue
        y /= nr
        vv = max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    LOOP_END = t0 + 150.0
    FINAL_END = t0 + 175.0
    rng = np.random.default_rng(0)
    n, d = 176, 9
    # load start
    best_X = None
    best_v = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    a = a / nr
                    vv = max_cos(a)
                    if vv < best_v:
                        best_v = vv
                        best_X = a.copy()
        except Exception:
            pass
    if best_X is None:
        best_X = rng.standard_normal((n, d))
        best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
        best_v = max_cos(best_X)
    base = best_X.copy()

    it = 0
    # pre-defined schedule to diversify basins
    while time.time() < LOOP_END:
        it += 1
        mode = it % 5
        Y = base.copy()
        if mode == 0:
            sig = [0.02, 0.05, 0.10, 0.18][(it // 5) % 4]
            Y = Y + sig * rng.standard_normal((n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps, a0, a1, lr0 = 700, 30.0, 250.0, 0.004
        elif mode == 1:
            k = [3, 5, 8, 12][(it // 5) % 4]
            w, _ = per_point_worst(Y)
            idx = np.argpartition(-w, k)[:k]
            Y = greedy_hole(Y, idx, rng, ntrials=256)
            steps, a0, a1, lr0 = 800, 40.0, 300.0, 0.0035
        elif mode == 2:
            # random knockout (not just worst) for topology change
            k = [4, 7, 12][(it // 5) % 3]
            idx = rng.choice(n, size=k, replace=False)
            Y = greedy_hole(Y, idx, rng, ntrials=256)
            steps, a0, a1, lr0 = 800, 25.0, 220.0, 0.0045
        elif mode == 3:
            # collective rotation-ish: jitter + small global random rotation
            sig = 0.03 + 0.02 * ((it // 5) % 3)
            Y = Y + sig * rng.standard_normal((n, d))
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
            steps, a0, a1, lr0 = 600, 50.0, 350.0, 0.003
        else:
            # heavy shake of a random subset
            k = 20 + ((it // 5) % 3) * 10
            idx = rng.choice(n, size=k, replace=False)
            Y[idx] = rng.standard_normal((k, d))
            Y[idx] /= np.linalg.norm(Y[idx], axis=1, keepdims=True)
            # reinsert greedily? just quench
            steps, a0, a1, lr0 = 900, 20.0, 200.0, 0.005

        Yq = smooth_quench(Y, steps=steps, a0=a0, a1=a1, lr0=lr0, rng=rng, deadline=LOOP_END)
        vv = max_cos(Yq)
        if vv < best_v and np.all(np.isfinite(Yq)):
            best_v = vv
            best_X = Yq.copy()
            base = Yq.copy()
        # also try a quick second quench with higher alpha from same Yq if promising
        if vv < best_v + 0.002 and time.time() < LOOP_END - 5:
            Yq2 = smooth_quench(Yq, steps=300, a0=120.0, a1=500.0, lr0=0.0015, deadline=LOOP_END)
            vv2 = max_cos(Yq2)
            if vv2 < best_v and np.all(np.isfinite(Yq2)):
                best_v = vv2
                best_X = Yq2.copy()
                base = Yq2.copy()
        if best_v < 0.458:
            break

    # single final LP squeeze on the best found (cheap, capped)
    if time.time() < FINAL_END - 2:
        try:
            best_X = lp_polish_deadline(best_X, deadline=FINAL_END, r_init=0.004, cap=2500)
        except Exception:
            pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
