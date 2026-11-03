import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))


def point_stress(X, alpha=120.0):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    M = float(np.max(G))
    E = np.exp(alpha * (G - M))
    np.fill_diagonal(E, 0.0)
    return E.sum(axis=1), M


def greedy_reinsert(Y, drop_idx, rng, ntrials=2048):
    n, d = Y.shape
    order = np.array(drop_idx, dtype=int).copy()
    rng.shuffle(order)
    mask = np.ones(n, dtype=bool)
    mask[order] = False
    cur_list = [Y[mask].copy()]
    # build as array incrementally
    cur = cur_list[0]
    out = Y.copy()
    for j in order:
        trials = rng.standard_normal((ntrials, d))
        tn = np.linalg.norm(trials, axis=1, keepdims=True)
        tn[tn == 0] = 1.0
        trials /= tn
        dots = trials @ cur.T
        m = dots.max(axis=1)
        pick = int(np.argmin(m))
        out[j] = trials[pick]
        cur = np.vstack([cur, out[j][None, :]])
    return out


def smooth_quench(x, steps=600, a0=30.0, a1=400.0, lr=0.004):
    n, d = x.shape
    m = np.zeros_like(x)
    vv = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        frac = t / max(1, steps - 1)
        alpha = a0 * ((a1 / a0) ** frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(alpha * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if S <= 0 or not np.isfinite(S):
            break
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        if not np.all(np.isfinite(Gt)):
            break
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        lrt = lr * (1.0 - 0.5 * frac)
        x = x - lrt * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        x /= nr
    nr = np.linalg.norm(x, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    return x / nr


def lp_polish_deadline(x, deadline, r_init=0.004, cap=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        mval = float(c_all.max())
        thresh = mval - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx_all = np.where(sel)[0]
            # keep largest cap
            part = np.argpartition(-c_all[idx_all], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx_all[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]
        J = iu0[1][sel]
        c_sel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        cols[:, :d] = I[:, None] * d + np.arange(d)
        cols[:, d:2 * d] = J[:, None] * d + np.arange(d)
        cols[:, 2 * d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x


def solve():
    t0 = time.time()
    LOOP_END = t0 + 138.0
    FINAL_END = t0 + 172.0
    rng = np.random.default_rng(42)
    n, d = 176, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    cands.append(a / nr)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])

    it = 0
    # cheap elitist basin hopping, no LP inside
    while time.time() < LOOP_END:
        it += 1
        u = rng.random()
        cand = None
        try:
            if u < 0.60:
                # hub knockout -> void reinsert
                k = int(rng.integers(2, 9))  # 2..8
                stress, _ = point_stress(best_X, alpha=120.0)
                # bias to high stress: take top 2k then sample k
                kk2 = min(n, 2 * k + 4)
                top = np.argpartition(-stress, kk2)[:kk2]
                # sample k among top with prob proportional to stress
                w = stress[top] - stress[top].min() + 1e-9
                w = w / w.sum()
                chosen = rng.choice(top, size=k, replace=False, p=w)
                tmp = best_X.copy()
                # small jitter before reinsert to diversify cur set slightly?
                tmp = greedy_reinsert(tmp, chosen, rng, ntrials=2048)
                cand = tmp
                steps = 550 if (it % 3) else 800
            elif u < 0.85:
                # global / subset jitter
                sigma = float(rng.choice([0.02, 0.05, 0.09, 0.15, 0.25]))
                tmp = best_X + sigma * rng.standard_normal((n, d))
                tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
                cand = tmp
                steps = 550
            else:
                # large void-seeking: pick worst point and move to void + jitter rest
                stress, _ = point_stress(best_X, alpha=120.0)
                j = int(np.argmax(stress))
                tmp = best_X + 0.03 * rng.standard_normal((n, d))
                tmp /= np.linalg.norm(tmp, axis=1, keepdims=True)
                # reinsert j to farthest among trials
                trials = rng.standard_normal((4096, d))
                trials /= np.linalg.norm(trials, axis=1, keepdims=True)
                dots = trials @ tmp.T
                # ignore self column j (set low)
                dots[:, j] = -10.0
                m = dots.max(axis=1)
                pick = int(np.argmin(m))
                tmp[j] = trials[pick]
                cand = tmp
                steps = 650
        except Exception:
            continue
        # anneal
        try:
            # vary schedule slightly for diversity
            if it % 5 == 0:
                cand_q = smooth_quench(cand, steps=steps, a0=40.0, a1=500.0, lr=0.005)
            else:
                cand_q = smooth_quench(cand, steps=steps, a0=30.0, a1=400.0, lr=0.004)
        except Exception:
            continue
        try:
            v = max_cos(cand_q)
        except Exception:
            continue
        if v < best_v and np.isfinite(v):
            best_X = cand_q
            best_v = v
        # safety: stop if very close to time
        if time.time() >= LOOP_END:
            break

    # single final LP squeeze with remaining budget
    try:
        if time.time() < FINAL_END - 3.0:
            best_X = lp_polish_deadline(best_X, FINAL_END - 1.0, r_init=0.004, cap=3000)
    except Exception:
        pass
    # normalize, sanitize
    best_X = np.asarray(best_X, dtype=float)
    nr = np.linalg.norm(best_X, axis=1, keepdims=True)
    nr[~np.isfinite(nr)] = 1.0
    nr[nr < 1e-12] = 1.0
    best_X = best_X / nr
    best_X[~np.isfinite(best_X)] = 0.0
    # fix any non-finite rows (should not happen)
    for i in range(n):
        if not np.all(np.isfinite(best_X[i])) or np.linalg.norm(best_X[i]) < 0.5:
            v = rng.standard_normal(d)
            v /= np.linalg.norm(v)
            best_X[i] = v
    nr = np.linalg.norm(best_X, axis=1, keepdims=True)
    best_X = best_X / nr
    return best_X
