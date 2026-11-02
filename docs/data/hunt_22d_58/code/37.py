import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _lp_polish(X, seconds, deadline):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n, d = X.shape
    best = _max_cos(X)
    radius = 5e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        it += 1
        if it > 30:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 5e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 650:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 650:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y
            best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def _hole_replace(X, victim, rng, n_cand=2500):
    n, d = X.shape
    mask = np.ones(n, dtype=bool)
    mask[victim] = False
    Y = X[mask]
    C = rng.standard_normal((n_cand, d))
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    dots = C @ Y.T
    worst = dots.max(axis=1)
    bi = int(np.argmin(worst))
    X2 = X.copy()
    X2[victim] = C[bi]
    return X2

def _pump(X0, t, stress, beta0, beta1, lr0, lr1, steps, deadline, rng=None, noise=0.0):
    X = X0.copy()
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lb0, lb1 = np.log(beta0), np.log(beta1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    iu = np.triu_indices(n, 1)
    for s in range(steps):
        if s % 100 == 0 and time.time() >= deadline:
            break
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        D = G - t
        # restrict to pairs above t-0.01
        active = D > -0.01
        Z = beta * D
        # stable softplus_beta and sigmoid
        sig = np.empty_like(D)
        sp = np.empty_like(D)
        # large positive
        m_pos = Z > 20.0
        m_neg = Z < -20.0
        m_mid = ~(m_pos | m_neg)
        sig[m_pos] = 1.0
        sp[m_pos] = D[m_pos]
        e = np.exp(Z[m_neg])
        sig[m_neg] = e
        sp[m_neg] = e / beta
        if np.any(m_mid):
            Zm = Z[m_mid]
            eZ = np.exp(Zm)
            sig[m_mid] = eZ / (1.0 + eZ)
            sp[m_mid] = np.log1p(eZ) / beta
        dL = 2.0 * sp * sig
        dL[~active] = 0.0
        # stress weighting (dual multipliers), drop unfavourable contacts
        if stress is not None:
            dL = dL * stress
            # hard drop tiny stress among near contacts
            # (stress already small there)
        np.fill_diagonal(dL, 0.0)
        Gr = dL @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        # normalize grad scale to avoid explosion as beta grows
        gn = float(np.linalg.norm(Gt) + 1e-12)
        # scale to unit-ish: divide by mean weight
        # keep Adam stable
        m1 = b1 * m1 + (1 - b1) * Gt
        m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** (s + 1)
        c2 = 1.0 - b2 ** (s + 1)
        m1h = m1 / c1
        m2h = m2 / c2
        step = lr * m1h / (np.sqrt(m2h) + eps)
        # project to tangent
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        over = sn > 0.05
        step = np.where(over, step * (0.05 / sn), step)
        X = X - step
        if noise > 0 and rng is not None:
            X = X + noise * (1.0 - f) * rng.standard_normal(X.shape)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def solve():
    t0 = time.time()
    DEADLINE = t0 + 165.0
    rng = np.random.default_rng(12345)
    n, d = 58, 22
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        z /= (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        best = z
        best_val = _max_cos(best)
    # incumbent contact stats
    G0 = best @ best.T
    np.fill_diagonal(G0, -2.0)
    base_min_deg = int(((G0 > best_val - 0.004).sum(axis=1)).min())
    cur = best.copy()
    cur_val = best_val
    # symmetric stress weights (start uniform)
    stress = np.ones((n, n))
    iu = np.triu_indices(n, 1)
    # short initial squeeze only if time allows
    # skip heavy initial LP; do 1s
    try:
        if time.time() < DEADLINE - 20:
            b2 = _lp_polish(best, 1.0, DEADLINE - 10)
            v2 = _max_cos(b2)
            if v2 < best_val:
                best = b2
                best_val = v2
                cur = best.copy()
                cur_val = best_val
    except Exception:
        pass
    # lexicographic targets: slightly below incumbent, lowered when feasible
    deltas = [0.00025, 0.00035, 0.0005, 0.0008]
    rnd = 0
    while time.time() < DEADLINE - 12 and rnd < 22:
        rnd += 1
        # pick target below current best
        delta = deltas[min(rnd // 4, len(deltas) - 1)]
        # also try staged lower targets 0.0663/0.0660/0.0655 relative to best
        t_target = best_val - delta
        # perturb cur to new basin every few rounds
        start = cur.copy()
        mode = rnd % 4
        if mode == 1:
            # small global kick
            start = start + 0.006 * rng.standard_normal(start.shape)
            start /= (np.linalg.norm(start, axis=1, keepdims=True) + 1e-12)
        elif mode == 2:
            # kick worst points
            Gc = start @ start.T
            np.fill_diagonal(Gc, -2.0)
            mx = int(np.argmax(Gc))
            i0, j0 = mx // n, mx % n
            # worst-point by max contact
            rowmax = Gc.max(axis=1)
            victims = np.argsort(-rowmax)[:2]
            for vv in victims:
                start = _hole_replace(start, int(vv), rng, n_cand=2500)
            # small noise on the pair
            start[i0] = start[i0] + 0.01 * rng.standard_normal(d)
            start[j0] = start[j0] + 0.01 * rng.standard_normal(d)
            start /= (np.linalg.norm(start, axis=1, keepdims=True) + 1e-12)
        elif mode == 3:
            # reinsert one overloaded hub into largest hole
            Gc = start @ start.T
            np.fill_diagonal(Gc, -2.0)
            degs = (Gc > float(Gc.max()) - 0.004).sum(axis=1)
            hub = int(np.argmax(degs))
            start = _hole_replace(start, hub, rng, n_cand=3000)
        else:
            start = start + 0.012 * rng.standard_normal(start.shape)
            start /= (np.linalg.norm(start, axis=1, keepdims=True) + 1e-12)
        # anneal beta upward within round
        try:
            out = _pump(start, t_target, stress, beta0=180.0, beta1=1600.0,
                        lr0=0.004, lr1=0.0004, steps=700, deadline=DEADLINE - 11,
                        rng=rng, noise=0.0)
        except Exception:
            continue
        v = _max_cos(out)
        G = out @ out.T
        np.fill_diagonal(G, -2.0)
        # violations above target
        nviol = int((G[iu] > t_target).sum())
        degs = (G > float(G.max()) - 0.004).sum(axis=1)
        min_deg = int(degs.min())
        # update stress from current violations (dual-like weights)
        try:
            Z = 800.0 * (G - t_target)
            Z = np.clip(Z, -20, 20)
            eZ = np.exp(Z)
            sig = eZ / (1.0 + eZ)
            # stress ~ sigmoid, favour hard contacts, drop easy ones
            ns = sig / (float(sig[iu].max()) + 1e-12)
            ns[ns < 0.02] = 0.0  # drop unfavourable contacts
            np.fill_diagonal(ns, 0.0)
            # symmetrize (already sym) and keep
            stress = 0.5 * stress + 0.5 * (ns + 1e-3)
            np.fill_diagonal(stress, 0.0)
        except Exception:
            pass
        # de-jamming test: markedly fewer contacts than incumbent
        dejammed = (min_deg <= base_min_deg - 6) or (min_deg <= 15)
        if v < best_val - 1e-12:
            # feasible improvement at this t (or at least new record): polish
            try:
                pol = _lp_polish(out, 1.2, DEADLINE - 10)
                vp = _max_cos(pol)
                if vp < v:
                    out, v = pol, vp
            except Exception:
                pass
            best = out.copy()
            best_val = v
            cur = out.copy()
            cur_val = v
            # recompute base degree
            G0 = best @ best.T
            np.fill_diagonal(G0, -2.0)
            base_min_deg = int(((G0 > best_val - 0.004).sum(axis=1)).min())
            # lower target next round automatically via best_val
        elif dejammed and v < best_val + 0.002:
            # keep topology-changed state as new basin even if slightly worse
            cur = out.copy()
            cur_val = v
        elif v < cur_val:
            cur = out.copy()
            cur_val = v
        # early exit if t-feasible found: t_target achieved -> lower t immediately
        # (handled by best update since t_target < best_val old)
        if nviol == 0 and v < t_target + 1e-12 and time.time() < DEADLINE - 12:
            # found feasible well below record, polish and continue lowering
            try:
                pol = _lp_polish(out, 1.0, DEADLINE - 10)
                vp = _max_cos(pol)
                if vp < best_val:
                    best = pol.copy()
                    best_val = vp
                    cur = pol.copy()
                    cur_val = vp
            except Exception:
                pass
    # final sharpen on best
    remain = DEADLINE - time.time() - 1.0
    if remain > 3 and best is not None:
        try:
            pol = _lp_polish(best, min(remain - 0.5, 12.0), DEADLINE - 0.5)
            vp = _max_cos(pol)
            if vp < best_val:
                best = pol
                best_val = vp
        except Exception:
            pass
    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-12)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best
