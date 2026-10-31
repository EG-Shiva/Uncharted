import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _contacts(x, thr=1e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(np.max(g))
    C = g > (m - thr)
    cnt = np.sum(C, axis=1)
    return m, cnt, g

def _smooth_descent(x, steps, mu=0.85, K0=25.0, K1=300.0, delta0=0.05, delta1=0.004, lr0=0.02, lr1=0.0015):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(steps):
        p = t / max(steps - 1, 1)
        K = K0 + (K1 - K0) * (p * p)
        delta = delta0 * (1 - p) + delta1 * p
        lr = lr0 * (1 - p) + lr1 * p
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(K * (G - m))
        W[G < (m - delta)] = 0.0
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        nrm = float(np.mean(np.sqrt(np.sum(gt * gt, axis=1))) + 1e-12)
        if nrm < 1e-15:
            continue
        gt = gt / nrm
        v = mu * v + (1.0 - mu) * gt
        x = x - lr * v
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-16)
    return x

def _refine_patch(remaining, new_pts, steps=250, mu=0.85):
    k = new_pts.shape[0]
    if k == 0:
        return new_pts
    n_rem = remaining.shape[0]
    v = np.zeros_like(new_pts)
    for t in range(steps):
        p = t / max(steps - 1, 1)
        K = 25.0 + (220.0 - 25.0) * (p * p)
        lr = 0.012 * (1 - p) + 0.001 * p
        all_pts = np.vstack([remaining, new_pts])
        G = new_pts @ all_pts.T
        for i in range(k):
            G[i, n_rem + i] = -2.0
        mi = np.max(G, axis=1, keepdims=True)
        W = np.exp(K * (G - mi))
        W[W < 1e-4] = 0.0
        grad = W @ all_pts
        dot = np.sum(grad * new_pts, axis=1, keepdims=True)
        gt = grad - dot * new_pts
        nrm = float(np.mean(np.sqrt(np.sum(gt * gt, axis=1))) + 1e-12)
        if nrm < 1e-15:
            continue
        gt = gt / nrm
        v = mu * v + (1.0 - mu) * gt
        new_pts = new_pts - lr * v
        new_pts = new_pts / (np.linalg.norm(new_pts, axis=1, keepdims=True) + 1e-16)
    return new_pts

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        sel_thr = m - max(20.0 * radius, 1e-9)
        sel = c > sel_thr
        if np.count_nonzero(sel) == 0:
            radius *= 0.5
            continue
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
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
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-16)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _greedy_fill(remaining, k, rng, S=15000):
    # sample candidates, greedily minimize max-dot to remaining+chosen
    cand = rng.normal(size=(S, 4))
    cand = cand / (np.linalg.norm(cand, axis=1, keepdims=True) + 1e-16)
    s_rem = np.max(cand @ remaining.T, axis=1)
    # order for diversity: add tiny noise to break ties deterministically
    chosen_idx = []
    cur_max = s_rem.copy()
    for _ in range(k):
        # pick best (smallest cur_max), but with tabu for near-duplicates
        order = np.argsort(cur_max)
        picked = -1
        for oi in order[:400]:
            ok = True
            for cj in chosen_idx:
                if float(cand[oi] @ cand[cj]) > 0.90:
                    ok = False
                    break
            if ok:
                picked = int(oi)
                break
        if picked < 0:
            picked = int(order[0])
        chosen_idx.append(picked)
        # update cur_max with dot to picked
        d = cand @ cand[picked]
        cur_max = np.maximum(cur_max, d)
        # mask picked so not picked again
        cur_max[picked] = 2.0
    new_pts = cand[np.array(chosen_idx, dtype=int)].copy()
    return new_pts

def solve():
    t0 = time.time()
    deadline = t0 + 178.0
    search_end = t0 + 138.0
    rng = np.random.default_rng(7)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(float)
            if a.shape == (153, 4):
                a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-16)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        best = rng.normal(size=(153, 4))
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)

    # precompute rattler info for first trials
    def get_rattler_nei(x, k):
        m, cnt, G = _contacts(x, thr=1e-7)
        ratt = int(np.argmin(cnt))
        # nearest neighbors by dot (largest dot)
        dots = G[ratt].copy()
        # G diag already -2, but ratt row diag is -2; ignore
        nei = np.argsort(dots)[-k:]
        return ratt, nei, cnt, dots

    trial_idx = 0
    sigmas = [0.02, 0.05, 0.10, 0.18]
    while time.time() < search_end and trial_idx < 300:
        trial_idx += 1
        remaining = None
        k_cap = 0
        do_patch = (trial_idx % 3 != 0)  # 2/3 patch, 1/3 gaussian
        if do_patch:
            if trial_idx <= 6:
                # targeted rattler cap removal
                k_cap = 10 + (trial_idx % 4)  # 10..13
                try:
                    ratt, nei, cnt, dots = get_rattler_nei(best, k_cap - 1)
                    rm = np.zeros(153, dtype=bool)
                    rm[ratt] = True
                    # take k_cap-1 nearest (largest dot)
                    G2 = best @ best.T
                    np.fill_diagonal(G2, -2.0)
                    order = np.argsort(G2[ratt])[-(k_cap - 1):]
                    rm[order] = True
                    idx_rm = np.where(rm)[0]
                except Exception:
                    idx_rm = rng.choice(153, size=k_cap, replace=False)
                mask = np.ones(153, dtype=bool)
                mask[idx_rm] = False
                remaining = best[mask]
                new_pts = _greedy_fill(remaining, k_cap, rng, S=15000)
                new_pts = _refine_patch(remaining, new_pts, steps=250)
                trial = np.vstack([remaining, new_pts])
                trial = trial + 1e-3 * rng.normal(size=trial.shape)
                trial = trial / (np.linalg.norm(trial, axis=1, keepdims=True) + 1e-16)
                steps = 650
            else:
                # random cap ruin
                k_cap = int(rng.integers(6, 17))  # 6..16
                # random center: either existing point + noise or uniform
                if rng.random() < 0.7:
                    c0 = best[int(rng.integers(0, 153))] + 0.3 * rng.normal(size=(4,))
                else:
                    c0 = rng.normal(size=(4,))
                c0 = c0 / (np.linalg.norm(c0) + 1e-16)
                dots_c = best @ c0
                idx_rm = np.argsort(dots_c)[-k_cap:]
                mask = np.ones(153, dtype=bool)
                mask[idx_rm] = False
                remaining = best[mask]
                new_pts = _greedy_fill(remaining, k_cap, rng, S=12000)
                new_pts = _refine_patch(remaining, new_pts, steps=220)
                trial = np.vstack([remaining, new_pts])
                trial = trial + 8e-4 * rng.normal(size=trial.shape)
                trial = trial / (np.linalg.norm(trial, axis=1, keepdims=True) + 1e-16)
                steps = 600
            trial_opt = _smooth_descent(trial, steps=steps)
            v = _max_cos(trial_opt)
            # selective LP for promising
            if v < best_val + 3e-4 and time.time() < search_end - 3.0:
                try:
                    trial_opt = _lp_polish(trial_opt, min(time.time() + 2.5, search_end))
                    v = _max_cos(trial_opt)
                except Exception:
                    pass
            if v < best_val - 1e-13:
                best = trial_opt
                best_val = v
        else:
            sigma = sigmas[(trial_idx // 3) % len(sigmas)]
            trial = best + sigma * rng.normal(size=best.shape)
            trial = trial / (np.linalg.norm(trial, axis=1, keepdims=True) + 1e-16)
            steps = 550 if sigma > 0.06 else 650
            # vary annealing slightly
            trial_opt = _smooth_descent(trial, steps=steps, K0=20.0, K1=350.0)
            v = _max_cos(trial_opt)
            if v < best_val + 1e-4 and time.time() < search_end - 3.0:
                try:
                    trial_opt = _lp_polish(trial_opt, min(time.time() + 2.0, search_end))
                    v = _max_cos(trial_opt)
                except Exception:
                    pass
            if v < best_val - 1e-13:
                best = trial_opt
                best_val = v
        if time.time() >= search_end:
            break

    try:
        if time.time() < deadline - 2.0:
            best = _lp_polish(best, deadline - 1.0)
    except Exception:
        pass
    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-16)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True) + 1e-16
    for i in range(best.shape[0]):
        if not np.isfinite(nr[i, 0]) or nr[i, 0] < 1e-12:
            vv = rng.normal(size=(4,))
            vv = vv / (np.linalg.norm(vv) + 1e-16)
            best[i] = vv
            nr[i, 0] = 1.0
    best = best / nr
    return best
