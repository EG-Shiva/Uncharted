import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def contact_data(x, tol=1e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(g.max())
    cm = g > m - tol
    cnt = cm.sum(axis=1)
    iu = np.triu_indices(x.shape[0], 1)
    mask = g[iu] > m - tol
    return cnt, m, cm, mask, g

def smooth_global(x, steps, T0, T1, s0, s1):
    x = x.copy()
    for it in range(steps):
        frac = it / max(1, steps - 1)
        T = T0 * (T1 / T0) ** frac
        s = s0 + (s1 - s0) * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        cmax = G.max()
        W = np.exp((G - cmax) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x) / T
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        x = x - s * (Gt / nrm)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def anneal_subset(y, mov, iters, T0, T1, s0, s1, rng, noise=0.0):
    y = y.copy()
    mov = np.asarray(mov, dtype=int)
    for it in range(iters):
        frac = it / max(1, iters - 1)
        T = T0 * (T1 / T0) ** frac
        s = s0 + (s1 - s0) * frac
        G = y @ y.T
        np.fill_diagonal(G, -2.0)
        cmax = float(G[mov].max())
        W = np.exp((G - cmax) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W[mov] @ y) / T
        Xm = y[mov]
        dot = np.sum(Grad * Xm, axis=1, keepdims=True)
        Gt = Grad - dot * Xm
        nrm = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        step = Gt / nrm
        if noise > 0 and frac < 0.5:
            amp = noise * (1.0 - frac / 0.5)
            pert = rng.standard_normal(step.shape)
            pert -= np.sum(pert * Xm, axis=1, keepdims=True) * Xm
            step = step + pert
            step /= np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        y[mov] = Xm - s * step
        y[mov] /= np.linalg.norm(y[mov], axis=1, keepdims=True)
    return y

def lp_polish(x, budget):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < budget and radius > 1e-14:
        cmat = x @ x.T
        c = cmat[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def greedy_holes(scaffold, k, rng, M=20000):
    # scaffold (ns,4), returns (k,4)
    out = []
    for j in range(k):
        cand = rng.standard_normal((M, 4))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        d1 = cand @ scaffold.T
        w1 = d1.max(axis=1)
        if len(out) > 0:
            prev = np.stack(out, axis=0)
            d2 = cand @ prev.T
            w2 = d2.max(axis=1)
            worst = np.maximum(w1, w2)
        else:
            worst = w1
        # add tiny jitter to break ties deterministically varied
        best_i = int(np.argmin(worst))
        # consider top few best, pick randomly among top 5 for diversity
        part = np.argpartition(worst, 5)[:5]
        # choose the absolute best most of time, sometimes 2nd best
        if rng.random() < 0.75:
            chosen = best_i
        else:
            chosen = int(rng.choice(part))
        out.append(cand[chosen])
    return np.stack(out, axis=0)

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 148.0
    rng = np.random.default_rng(0)
    n, d = 153, 4
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy")
    except Exception:
        xr = None
    cands = []
    for a in (xb, xr):
        if a is not None:
            try:
                a = np.asarray(a, dtype=float)
                if a.shape == (n, d):
                    a = a / np.linalg.norm(a, axis=1, keepdims=True)
                    if np.all(np.isfinite(a)):
                        cands.append((max_cos(a), a.copy()))
            except Exception:
                pass
    if len(cands) == 0:
        x0 = rng.standard_normal((n, d))
        x0 /= np.linalg.norm(x0, axis=1, keepdims=True)
    else:
        cands.sort(key=lambda t: t[0])
        x0 = cands[0][1].copy()
    incumbent = x0.copy()
    inc_val = max_cos(incumbent)
    best = incumbent.copy()
    best_val = inc_val
    _, _, _, best_mask, _ = contact_data(best, tol=1e-7)
    pool = []  # list of (val, X, mask)
    trial = 0
    # precompute hub order refresh every trial from base
    while time.time() < SEARCH_END:
        trial += 1
        remain = DEADLINE - time.time()
        if remain < 8:
            break
        # choose base: mostly best, sometimes pool diverse
        if len(pool) > 0 and rng.random() < 0.35:
            bi = int(rng.integers(len(pool)))
            base = pool[bi][1]
        else:
            base = incumbent if rng.random() < 0.7 else best
        base = base / np.linalg.norm(base, axis=1, keepdims=True)
        try:
            cnt, m0, cm, bmask, Gfull = contact_data(base, tol=1e-7)
        except Exception:
            cnt = np.ones(n) * 5
            Gfull = base @ base.T
            m0 = float(Gfull.max())
        # select hub and cluster
        mode = trial % 4
        mov = None
        if mode == 0 or mode == 1:
            # overloaded hub-centred
            order = np.argsort(-cnt, kind="stable")
            # pick among top 12 with rotation for distinct hubs
            top = order[:12]
            hub = int(top[(trial // 4 + int(rng.integers(12))) % 12]) if len(top) >= 12 else int(order[0])
            # neighbours by contact then by closeness
            neigh = np.where(cm[hub])[0].tolist()
            if len(neigh) == 0:
                # fallback to closest dots
                dots = Gfull[hub].copy()
                dots[hub] = -2
                neigh = list(np.argsort(-dots)[:8])
            # contact counts of neighbours for stress ordering
            neigh.sort(key=lambda j: (-cnt[j], -Gfull[hub, j]))
            kpick = int(rng.integers(4, 7))  # 4..6 neighbours
            kpick = min(kpick, len(neigh))
            # if hub has >kpick contacts, take top kpick (most stressed); else take all + closest non-contacts
            chosen = neigh[:kpick]
            if len(chosen) < kpick:
                dots = Gfull[hub].copy()
                dots[hub] = -2
                for j in np.argsort(-dots):
                    if j != hub and j not in chosen:
                        chosen.append(int(j))
                    if len(chosen) >= kpick:
                        break
            mov = [hub] + [int(c) for c in chosen]
        elif mode == 2:
            # low-contact / rattler centred
            order_low = np.argsort(cnt, kind="stable")
            # pick rattler or 2nd lowest
            cand_low = order_low[:8]
            li = int(cand_low[int(rng.integers(min(8, len(cand_low))))])
            dots = Gfull[li].copy()
            dots[li] = -2
            closest = list(np.argsort(-dots)[:7])
            kpick = int(rng.integers(4, 7))
            mov = [int(li)] + [int(c) for c in closest[:kpick]]
        else:
            # random mid hub to diversify
            # pick hub with 6-8 contacts at random
            cand = np.where(cnt >= 6)[0]
            if len(cand) == 0:
                cand = np.arange(n)
            hub = int(cand[int(rng.integers(len(cand)))])
            neigh = np.where(cm[hub])[0].tolist()
            if len(neigh) == 0:
                dots = Gfull[hub].copy(); dots[hub] = -2
                neigh = list(np.argsort(-dots)[:8])
            neigh.sort(key=lambda j: -Gfull[hub, j])
            kpick = int(rng.integers(4, 7))
            kpick = min(kpick, len(neigh))
            chosen = neigh[:kpick]
            mov = [hub] + [int(c) for c in chosen]
        mov = sorted(set(int(v) for v in mov))
        k = len(mov)
        if k < 4 or k > 9:
            continue
        fixed_idx = np.array([i for i in range(n) if i not in set(mov)], dtype=int)
        scaffold = base[fixed_idx]
        # reinsert one at a time by minimax hole optimisation (greedy sequential)
        try:
            re = greedy_holes(scaffold, k, rng, M=16000)
        except Exception:
            continue
        y = base.copy()
        y[np.array(mov)] = re
        # joint tangent-space optimisation of only reinserted subset
        # vary schedule slightly per trial
        T0 = 0.022 if (trial % 2 == 0) else 0.03
        try:
            y = anneal_subset(y, mov, iters=350, T0=T0, T1=0.0028, s0=0.016, s1=0.0012, rng=rng, noise=0.15 if (trial % 3 == 0) else 0.0)
        except Exception:
            continue
        # short global exponential smoothing (unfrozen)
        try:
            y = smooth_global(y, steps=220, T0=0.012, T1=0.0022, s0=0.012, s1=0.0012)
        except Exception:
            continue
        v_mid = max_cos(y)
        # quick reject if far worse (save LP time) unless graph novel check needed
        # compute contact graph diff at 1e-7
        try:
            _, _, _, ymask, _ = contact_data(y, tol=1e-7)
            diff = int(np.sum(ymask != best_mask))
        except Exception:
            diff = 0
        novel = diff > 30
        # score first by novelty then value: allow LP if novel or promising value
        if (not novel) and v_mid > best_val + 0.002:
            # not novel and not promising: maybe keep as cur diversity occasionally
            if v_mid < best_val + 0.006 and len(pool) < 6 and rng.random() < 0.1:
                pool.append((v_mid, y.copy(), ymask.copy()))
            continue
        # unfreeze LP polishing short
        lp_budget = 1.4 if novel else 1.0
        # reduce budget if time short
        if DEADLINE - time.time() < 15:
            lp_budget = 0.6
        try:
            z = lp_polish(y, budget=lp_budget)
        except Exception:
            z = y
        vz = max_cos(z)
        try:
            _, _, _, zmask, _ = contact_data(z, tol=1e-7)
            zdiff = int(np.sum(zmask != best_mask))
        except Exception:
            zdiff = diff
            zmask = ymask
        znovel = zdiff > 30
        # update best by value
        if vz < best_val - 1e-12:
            best = z.copy()
            best_val = vz
            best_mask = zmask.copy()
            incumbent = z.copy()
            inc_val = vz
            # prune pool worse than best+0.004
            pool = [(v, X, mm) for (v, X, mm) in pool if v < best_val + 0.004]
        elif znovel:
            # keep novel basins even if slightly worse (topology change priority)
            if vz < best_val + 0.0025:
                # add to pool, cap size
                pool.append((vz, z.copy(), zmask.copy()))
                if len(pool) > 8:
                    pool.sort(key=lambda t: t[0])
                    pool = pool[:8]
                # move incumbent to novel to continue surgery from new topology sometimes
                if rng.random() < 0.5 or vz < inc_val:
                    incumbent = z.copy()
                    inc_val = vz
        else:
            # non-novel but improved over incumbent?
            if vz < inc_val - 5e-7:
                incumbent = z.copy()
                inc_val = vz
        # occasional reset incumbent to best if drifted too far
        if inc_val > best_val + 0.006:
            incumbent = best.copy()
            inc_val = best_val
    # final polish best with remaining time
    rem = DEADLINE - time.time() - 1.5
    if rem > 2:
        try:
            best = lp_polish(best, budget=min(rem, 20.0))
        except Exception:
            pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
