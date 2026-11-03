import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _soft_anneal(x, steps, a0, a1, lr0, lr1, mu=0.85):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        alpha = a0 + (a1 - a0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        # weights, m is max so exponent <=0
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True)
        rs = np.maximum(rs, 1e-16)
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        step = lr * v
        # trust region per row
        nm = np.linalg.norm(step, axis=1, keepdims=True)
        sc = np.minimum(1.0, 0.04 / (nm + 1e-16))
        step = step * sc
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _best_hole(others, rng, kcand=4000):
    d = others.shape[1]
    cand = rng.standard_normal((kcand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    dots = cand @ others.T
    worst = dots.max(axis=1)
    q = cand[int(np.argmin(worst))].copy()
    # refine single point against soft-max
    for _ in range(150):
        dots = others @ q
        m = dots.max()
        w = np.exp(50.0 * (dots - m))
        ws = w.sum()
        if ws == 0 or not np.isfinite(ws):
            break
        g = (w @ others) / ws
        gt = g - float(g @ q) * q
        q = q - 0.2 * gt
        nq = np.linalg.norm(q)
        if nq < 1e-12 or not np.isfinite(nq):
            break
        q /= nq
    return q


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best_val = _max_cos(x)
    best_x = x.copy()
    radius = 1e-3
    iu_r, iu_c = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-13:
        G = x @ x.T
        c_full = G[iu_r, iu_c]
        m = c_full.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c_full > thr
        p = int(sel.sum())
        if p == 0:
            radius *= 0.5
            continue
        if p > 3000:
            idx = np.argpartition(-c_full, 3000)[:3000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
            p = 3000
        I = iu_r[sel]
        J = iu_c[sel]
        c_sel = c_full[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(),
                 (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel,
                          A_eq=A_eq, b_eq=np.zeros(n),
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
        if np.any(~np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        vv = _max_cos(y)
        if vv < best_val:
            best_val = vv
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
            if time.time() >= deadline:
                break
        else:
            radius *= 0.5
    return best_x


def solve():
    t0 = time.time()
    HARD_END = t0 + 172.0
    EXPLORE_END = t0 + 132.0
    rng = np.random.default_rng(7)
    n, d = 102, 12
    try:
        best = np.load("best.npy").astype(float)
    except Exception:
        try:
            best = np.load("record.npy").astype(float)
        except Exception:
            best = rng.standard_normal((n, d))
    if best.shape != (n, d):
        tmp = rng.standard_normal((n, d))
        try:
            m0 = min(best.shape[0], n)
            m1 = min(best.shape[1], d)
            tmp[:m0, :m1] = best[:m0, :m1]
        except Exception:
            pass
        best = tmp
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    best_val = _max_cos(best)

    cur = best.copy()
    cur_val = best_val
    # keep few best soft candidates
    cands = [(best_val, best.copy())]

    it = 0
    while time.time() < EXPLORE_END:
        it += 1
        # choose base: mostly wander from cur, sometimes from best
        if rng.random() < 0.75:
            base = cur
            base_val = cur_val
        else:
            base = best
            base_val = best_val
        # per-point max of base for targeting
        try:
            Gb = base @ base.T
            np.fill_diagonal(Gb, -2.0)
            pm = Gb.max(axis=1)
            order = np.argsort(-pm)
        except Exception:
            order = np.arange(n)
            pm = np.zeros(n)

        trial = base.copy()
        mode = int(rng.integers(0, 7))
        if mode == 0:
            trial = base + 0.010 * rng.standard_normal((n, d))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            steps = 700
            a0, a1, lr0, lr1 = 12.0, 110.0, 0.05, 0.005
        elif mode == 1:
            trial = base + 0.030 * rng.standard_normal((n, d))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            steps = 800
            a0, a1, lr0, lr1 = 15.0, 130.0, 0.05, 0.004
        elif mode == 2:
            c = int(order[int(rng.integers(0, 5))])
            Gc = base @ base[c]
            Gc[c] = -10
            nn = np.argsort(-Gc)[:12]
            region = list(set([c] + nn.tolist()))
            trial[region] += 0.16 * rng.standard_normal((len(region), d))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            steps = 900
            a0, a1, lr0, lr1 = 10.0, 120.0, 0.05, 0.004
        elif mode == 3:
            c = int(order[int(rng.integers(0, 8))])
            mask = np.ones(n, dtype=bool)
            mask[c] = False
            others = trial[mask]
            trial[c] = _best_hole(others, rng, 4000)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            steps = 900
            a0, a1, lr0, lr1 = 12.0, 140.0, 0.045, 0.003
        elif mode == 4:
            k1 = int(order[int(rng.integers(0, 10))])
            k2 = int(order[int(rng.integers(0, 10))])
            if k2 == k1:
                k2 = (k2 + 11) % n
            for c in (k1, k2):
                mask = np.ones(n, dtype=bool)
                mask[c] = False
                others = trial[mask]
                trial[c] = _best_hole(others, rng, 2500)
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            steps = 900
            a0, a1, lr0, lr1 = 10.0, 120.0, 0.045, 0.004
        elif mode == 5:
            A = np.eye(d) + 0.10 * rng.standard_normal((d, d))
            trial = base @ A
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            steps = 700
            a0, a1, lr0, lr1 = 14.0, 110.0, 0.05, 0.005
        else:
            k = 10
            idx = rng.choice(n, size=k, replace=False)
            trial[idx] += 0.24 * rng.standard_normal((k, d))
            trial /= np.linalg.norm(trial, axis=1, keepdims=True)
            steps = 700
            a0, a1, lr0, lr1 = 12.0, 100.0, 0.055, 0.005

        # jitter params a bit
        a1 = a1 * (0.9 + 0.2 * rng.random())
        trial = _soft_anneal(trial, steps, a0, a1, lr0, lr1, 0.85)
        v = _max_cos(trial)
        if v < best_val - 1e-12:
            best = trial.copy()
            best_val = v
            cur = trial.copy()
            cur_val = v
            cands.append((v, trial.copy()))
            cands = sorted(cands, key=lambda z: z[0])[:3]
        elif v < cur_val + 0.004 and rng.random() < 0.75:
            cur = trial.copy()
            cur_val = v
        # keep near-best for final polish diversity
        elif v < best_val + 0.001 and len(cands) < 3:
            cands.append((v, trial.copy()))
            cands = sorted(cands, key=lambda z: z[0])[:3]
        # pull wanderer back if too far
        if cur_val > best_val + 0.006:
            # reset to best with small kick
            cur = best + 0.008 * rng.standard_normal((n, d))
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            cur_val = _max_cos(cur)
        if time.time() >= HARD_END - 30.0:
            break

    # final LP polish: best first, then second if time
    cands = sorted(cands, key=lambda z: z[0])[:3]
    # ensure best is first
    remain = HARD_END - time.time()
    if remain > 5.0:
        # polish top candidate
        dl = time.time() + remain - 2.0 - (8.0 if len(cands) > 1 else 0.0)
        dl = min(dl, HARD_END - 2.0)
        if dl - time.time() > 3.0:
            polished = _lp_polish(cands[0][1], dl)
            pv = _max_cos(polished)
            if pv < best_val:
                best = polished
                best_val = pv
            else:
                # keep polished if equal (more jammed)
                if pv <= cands[0][0] + 1e-12:
                    best = polished
                    best_val = pv
        # second candidate
        if len(cands) > 1 and (HARD_END - time.time()) > 8.0:
            dl2 = time.time() + (HARD_END - time.time()) - 1.5
            p2 = _lp_polish(cands[1][1], dl2)
            v2 = _max_cos(p2)
            if v2 < best_val:
                best = p2
                best_val = v2
        else:
            # ensure best is polished even if single
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
