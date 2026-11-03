import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def anneal_run(X, steps, beta0, beta1, lr0, lr1, deadline=1e30):
    n, d = X.shape
    for s in range(steps):
        if s % 50 == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        # exp weights
        W = np.exp((G - m) * beta)
        np.fill_diagonal(W, 0.0)
        Grad = W @ X
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - proj * X
        gn = float(np.linalg.norm(Gt) / np.sqrt(n))
        if not np.isfinite(gn) or gn < 1e-16:
            continue
        X = X - (lr / gn) * Gt
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        X = X / nr
    return X

def lp_polish_deadline(x, deadline, r_init=0.004, cap=2500):
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
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > cap:
            idx = np.where(sel)[0]
            # keep largest cap
            part = np.argpartition(-c_all[idx], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        ar = np.arange(d)
        cols[:, :d] = I[:, None] * d + ar
        cols[:, d:2*d] = J[:, None] * d + ar
        cols[:, 2*d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
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
    LOOP_END = t0 + 148.0
    FINAL_END = t0 + 176.0
    rng = np.random.default_rng(0)
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
    if not cands:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [max_cos(c) for c in cands]
    bi = int(np.argmin(vals))
    best_X = cands[bi].copy()
    best_v = float(vals[bi])

    # light initial squeeze is skipped because best is already polished;
    # go straight to basin hopping
    it = 0
    # preallocate triu for contact counts
    while time.time() < LOOP_END:
        it += 1
        # choose k: cycle 1..5, bias small
        if it % 7 == 0:
            k = 5
        elif it % 3 == 0:
            k = 3
        elif it % 2 == 0:
            k = 2
        else:
            k = 1
        # find worst points by per-point max
        G = best_X @ best_X.T
        np.fill_diagonal(G, -2.0)
        pmax = G.max(axis=1)
        # add tiny random tie-break to diversify
        jitter = rng.standard_normal(n) * 1e-12
        order = np.argsort(-(pmax + jitter))
        # sometimes target high-contact hubs: compute contacts near max
        if it % 4 == 0:
            # count contacts within 1e-3 of global max
            thr = best_v - 1e-3
            cntc = np.count_nonzero(G > thr, axis=1)
            order2 = np.argsort(-(cntc.astype(float) + rng.standard_normal(n) * 0.1))
            # mix: take half from each
            idx_set = list(order2[:k])
        else:
            idx_set = list(order[:k])
        # occasionally add a random index
        if it % 5 == 0:
            idx_set[-1] = int(rng.integers(0, n))
        idx_set = sorted(set(int(v) for v in idx_set))
        k = len(idx_set)
        cand = best_X.copy()
        mask = np.ones(n, dtype=bool)
        mask[idx_set] = False
        cur = cand[mask].copy()  # (n-k, d)
        # greedy hole reinsertion with modest trials
        ntrials = 350 if k > 2 else 500
        for j in idx_set:
            trials = rng.standard_normal((ntrials, d))
            trials /= np.linalg.norm(trials, axis=1, keepdims=True)
            dots = trials @ cur.T
            mm = dots.max(axis=1)
            pick = int(np.argmin(mm))
            cand[j] = trials[pick]
            cur = np.vstack([cur, cand[j][None, :]])
        # small global tangential jitter to help topology change
        if it % 2 == 0:
            cand = cand + rng.standard_normal((n, d)) * 0.008
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        # anneal params vary
        beta0 = 25.0 + (it % 3) * 10.0
        beta1 = 150.0 + (it % 4) * 30.0
        lr0 = 0.02
        lr1 = 0.002
        steps = 320
        if k >= 3:
            steps = 420
        cand = anneal_run(cand, steps=steps, beta0=beta0, beta1=beta1, lr0=lr0, lr1=lr1, deadline=LOOP_END)
        v = max_cos(cand)
        # accept if promising -> short LP
        if v < best_v:
            # quick squeeze 2-3 sec
            dl = min(time.time() + 2.5, LOOP_END, FINAL_END - 20.0)
            if dl > time.time() + 0.3:
                cand2 = lp_polish_deadline(cand, dl, r_init=0.004)
                v2 = max_cos(cand2)
            else:
                cand2, v2 = cand, v
            if v2 < best_v:
                best_X, best_v = cand2, v2
        elif v < best_v + 0.0015 and (it % 12 == 0) and time.time() + 5 < LOOP_END:
            # occasional gamble on near-miss
            dl = min(time.time() + 1.5, LOOP_END)
            cand2 = lp_polish_deadline(cand, dl, r_init=0.003)
            v2 = max_cos(cand2)
            if v2 < best_v:
                best_X, best_v = cand2, v2
        # loop continues from best_X (elitist)
    # final squeeze with remaining time
    if time.time() < FINAL_END - 0.5:
        best_X = lp_polish_deadline(best_X, FINAL_END - 0.2, r_init=0.005)
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
