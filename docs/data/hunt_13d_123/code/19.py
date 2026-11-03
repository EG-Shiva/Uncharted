import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, seconds, p_cap=2000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    start = time.time()
    iu0, iu1 = np.triu_indices(n, 1)
    while True:
        if time.time() - start >= seconds:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        window = max(20.0 * radius, 1e-9)
        sel = c_all > (m - window)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > p_cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], p_cap - 1)[:p_cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = p_cap
        I = iu0[sel]
        J = iu1[sel]
        csel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            erows = np.repeat(np.arange(n), d)
            ecols = np.arange(n * d)
            A_eq = coo_matrix((x.ravel(), (erows, ecols)), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        v = _max_cos(y)
        if v < best:
            best = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
            if radius < 1e-7:
                x = best_x.copy()
        if time.time() - start >= seconds:
            break
    return best_x


def _smooth_run(y, steps, T0, T1, lr0, lr1):
    n, d = y.shape
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = T0 + (T1 - T0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        E = np.exp((G - m) / T)
        np.fill_diagonal(E, 0.0)
        row = E.sum(axis=1, keepdims=True) + 1e-20
        Avg = (E @ y) / row
        dot = (Avg * y).sum(axis=1, keepdims=True)
        Gt = Avg - dot * y
        y = y - lr * Gt
        y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y


def _best_hole(Xf, rng, n_cand=25000, batch=10000, refine=200):
    d = Xf.shape[1]
    best_v = None
    best_md = 1e9
    done = 0
    while done < n_cand:
        b = min(batch, n_cand - done)
        Y = rng.normal(size=(b, d))
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
        dots = Y @ Xf.T
        md = dots.max(axis=1)
        j = int(np.argmin(md))
        if float(md[j]) < best_md:
            best_md = float(md[j])
            best_v = Y[j].copy()
        done += b
    v = best_v.copy()
    md = best_md
    for s in range(refine):
        frac = s / max(refine - 1, 1)
        sigma = 0.02 * (1.0 - frac) + 0.001
        prop = v + sigma * rng.normal(size=(d,))
        nrm = np.linalg.norm(prop)
        if nrm == 0:
            continue
        prop /= nrm
        mdp = float((prop @ Xf.T).max())
        if mdp < md:
            v = prop
            md = mdp
    return v


def solve():
    t0 = time.time()
    BUDGET = 175.0
    FINAL_SEC = 11.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 123, 13

    X0 = None
    best_val0 = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.any(~np.isfinite(a)):
                    continue
                v = _max_cos(a)
                if v < best_val0:
                    best_val0 = v
                    X0 = a.copy()
        except Exception:
            pass
    if X0 is None:
        X0 = rng.normal(size=(n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        best_val0 = _max_cos(X0)

    best = X0.copy()
    best_val = float(best_val0)
    cur = X0.copy()
    cur_val = float(best_val0)

    deadline = t0 + BUDGET - FINAL_SEC - 0.5
    it = 0
    # small initial snap if time allows
    if time.time() < deadline - 5:
        try:
            cand = _lp_polish(best, 1.0, p_cap=1500)
            v = _max_cos(cand)
            if v < best_val:
                best_val = v
                best = cand.copy()
                cur = cand.copy()
                cur_val = v
        except Exception:
            pass

    while time.time() < deadline:
        it += 1
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        Gc = cur @ cur.T
        np.fill_diagonal(Gc, -2.0)
        wmax = Gc.max(axis=1)
        worst_idx = np.argsort(-wmax)
        tight_thr = cur_val - 1e-6
        counts_tight = (Gc > tight_thr).sum(axis=1)
        ratt_idx = np.argsort(counts_tight)
        mode = it % 6
        trial = cur.copy()
        try:
            if mode == 0:
                trial = trial + 0.012 * rng.normal(size=trial.shape)
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                trial = _smooth_run(trial, steps=180, T0=0.02, T1=0.004, lr0=0.4, lr1=0.05)
            elif mode == 1:
                targets = set()
                for k in range(min(3, n)):
                    targets.add(int(worst_idx[k]))
                targets.add(int(ratt_idx[0]))
                targets.add(int(rng.integers(0, n)))
                trial += 0.003 * rng.normal(size=trial.shape)
                for idx in list(targets):
                    trial[idx] += 0.18 * rng.normal(size=(d,))
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                trial = _smooth_run(trial, steps=180, T0=0.018, T1=0.003, lr0=0.35, lr1=0.04)
            elif mode == 2:
                n_replace = 2 if (it % 12 == 0) else 1
                repl = [int(worst_idx[k]) for k in range(n_replace)]
                for r in repl:
                    mask = np.ones(n, dtype=bool)
                    mask[r] = False
                    Xf = trial[mask]
                    v = _best_hole(Xf, rng, n_cand=25000, batch=10000, refine=200)
                    trial[r] = v
                trial += 0.002 * rng.normal(size=trial.shape)
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                trial = _smooth_run(trial, steps=140, T0=0.015, T1=0.003, lr0=0.3, lr1=0.03)
            elif mode == 3:
                w = int(worst_idx[0])
                K = 8 + (it % 7)
                dots_w = (trial @ trial[w])
                dots_w[w] = -2.0
                neigh = np.argsort(-dots_w)[:K - 1]
                S = [w] + [int(q) for q in neigh]
                for sidx in S:
                    v = rng.normal(size=(d,))
                    v /= np.linalg.norm(v)
                    trial[sidx] = v
                trial = _smooth_run(trial, steps=250, T0=0.025, T1=0.004, lr0=0.45, lr1=0.05)
            elif mode == 4:
                repl = [int(ratt_idx[0]), int(ratt_idx[1])] if n > 1 else [int(ratt_idx[0])]
                # deduplicate
                repl = list(dict.fromkeys(repl))[:2]
                for r in repl:
                    mask = np.ones(n, dtype=bool)
                    mask[r] = False
                    Xf = trial[mask]
                    v = _best_hole(Xf, rng, n_cand=20000, batch=10000, refine=150)
                    trial[r] = v
                trial += 0.002 * rng.normal(size=trial.shape)
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                trial = _smooth_run(trial, steps=150, T0=0.02, T1=0.004, lr0=0.35, lr1=0.04)
            else:
                trial = trial + 0.03 * rng.normal(size=trial.shape)
                trial /= np.linalg.norm(trial, axis=1, keepdims=True)
                trial = _smooth_run(trial, steps=250, T0=0.03, T1=0.006, lr0=0.5, lr1=0.06)
        except Exception:
            continue

        try:
            v_sm = _max_cos(trial)
        except Exception:
            continue

        # LP snap only if promising to save time
        did_lp = False
        if v_sm < best_val + 0.0008 and remaining > 4.0:
            lp_sec = 0.7 if remaining > 25 else 0.4
            try:
                trial2 = _lp_polish(trial, lp_sec, p_cap=1500)
                v2 = _max_cos(trial2)
                trial = trial2
                v_sm = v2
                did_lp = True
            except Exception:
                pass

        v = float(v_sm)
        if v < best_val - 1e-14:
            best_val = v
            best = trial.copy()
            cur = trial.copy()
            cur_val = v
        else:
            tol = 0.003 * np.exp(-it / 30.0) + 0.0002
            if v < cur_val + tol:
                cur = trial.copy()
                cur_val = v
            if it % 25 == 0:
                cur = best + 0.008 * rng.normal(size=best.shape)
                cur /= np.linalg.norm(cur, axis=1, keepdims=True)
                try:
                    cur_val = _max_cos(cur)
                except Exception:
                    cur = best.copy()
                    cur_val = best_val
        if time.time() >= deadline:
            break

    final_sec = max(1.0, min(FINAL_SEC, (t0 + BUDGET) - time.time() - 0.5))
    try:
        polished = _lp_polish(best, final_sec, p_cap=3000)
        pv = _max_cos(polished)
        if pv < best_val:
            best = polished
            best_val = pv
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
