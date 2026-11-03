import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _try_duals(res, p):
    try:
        if hasattr(res, "ineqlin"):
            iq = res.ineqlin
            marg = None
            if isinstance(iq, dict):
                marg = iq.get("marginals", None)
                if marg is None:
                    marg = iq.get("dual", None)
            else:
                marg = getattr(iq, "marginals", None)
                if marg is None:
                    marg = getattr(iq, "dual", None)
            if marg is not None:
                a = np.asarray(marg, dtype=float).ravel()
                if a.size == p:
                    return np.abs(a)
        # older scipy: res['ineqlin']
        try:
            d = dict(res)
            if "ineqlin" in d:
                iq = d["ineqlin"]
                if isinstance(iq, dict) and "marginals" in iq:
                    a = np.asarray(iq["marginals"], dtype=float).ravel()
                    if a.size == p:
                        return np.abs(a)
        except Exception:
            pass
    except Exception:
        pass
    return None


def _lp_polish(x, seconds, t0=None, deadline=None, init_radius=0.008):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = init_radius
    start = time.time()
    iu = np.triu_indices(n, 1)
    last_duals = None
    last_I = None
    last_J = None
    while True:
        now = time.time()
        if now - start >= seconds:
            break
        if t0 is not None and deadline is not None and (now - t0) >= deadline:
            break
        if seconds - (now - start) < 0.10:
            break
        if radius < 1e-14:
            break
        G = x @ x.T
        c_all = G[iu]
        m = c_all.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel_idx = np.where(c_all > thr)[0]
        if len(sel_idx) > 2500:
            part = np.argpartition(c_all, -2500)[-2500:]
            sel_idx = part
        I = iu[0][sel_idx]
        J = iu[1][sel_idx]
        csel = c_all[sel_idx]
        p = len(sel_idx)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate(
            [I[:, None] * d + np.arange(d),
             J[:, None] * d + np.arange(d),
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix(
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
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
        duals = _try_duals(res, p)
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
            if duals is not None:
                last_duals = duals
                last_I = I.copy()
                last_J = J.copy()
        else:
            radius *= 0.5
    return x, best, (last_I, last_J, last_duals)


def _single_lp_duals(x):
    n, d = x.shape
    iu = np.triu_indices(n, 1)
    G = x @ x.T
    c_all = G[iu]
    m = float(c_all.max())
    thr = m - 1e-9
    sel = np.where(c_all > thr)[0]
    if len(sel) == 0:
        sel = np.array([int(np.argmax(c_all))])
    if len(sel) > 3000:
        part = np.argpartition(c_all, -3000)[-3000:]
        sel = part
    I = iu[0][sel]
    J = iu[1][sel]
    csel = c_all[sel]
    p = len(sel)
    radius = 0.005
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
            (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
            shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                      b_eq=np.zeros(n),
                      bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                      method="highs")
        if res.status == 0:
            du = _try_duals(res, p)
            if du is not None:
                return I, J, du, m
    except Exception:
        pass
    return I, J, np.ones(p), m


def _exp_steps(x, steps, alpha, lr, max_step=0.02):
    n, d = x.shape
    for _ in range(steps):
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        m = G.max()
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        grad = W @ x
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        step = lr * gt
        nm = np.linalg.norm(step, axis=1).max()
        if nm > max_step and nm > 0:
            step *= (max_step / nm)
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _refine_single(F, S, alpha1=25.0, steps1=60, alpha2=80.0, steps2=60, lr=0.5):
    Y = S.copy()
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    for alpha, steps in ((alpha1, steps1), (alpha2, steps2)):
        for _ in range(steps):
            D = Y @ F.T
            mr = D.max(axis=1, keepdims=True)
            E = np.exp(alpha * (D - mr))
            s = E.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            W = E / s
            G = W @ F
            dot = np.sum(G * Y, axis=1, keepdims=True)
            Gt = G - dot * Y
            Y = Y - lr * Gt
            Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y


def solve():
    t0 = time.time()
    DEADLINE = 155.0
    LIMIT = 171.0
    rng = np.random.default_rng(0)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = np.load("record.npy").astype(float)
    xb = np.asarray(xb, dtype=float)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = xb.shape
    incumbent = xb.copy()
    inc_val = _max_cos(incumbent)

    def time_up():
        return (time.time() - t0) >= DEADLINE

    # 1) one LP polish
    if not time_up():
        remain = min(10.0, max(0.0, DEADLINE - (time.time() - t0) - 40.0))
        # ensure at least a short polish even late
        if remain < 3.0 and (DEADLINE - (time.time() - t0)) > 4.0:
            remain = min(4.0, DEADLINE - (time.time() - t0) - 2.0)
        if remain > 1.0:
            xp, vp, _ = _lp_polish(incumbent, seconds=remain, t0=t0, deadline=DEADLINE, init_radius=0.008)
            if vp < inc_val:
                incumbent, inc_val = xp, vp
            else:
                # keep polished even if equal within tol (vertex)
                if vp <= inc_val + 1e-12:
                    incumbent = xp
                    inc_val = vp
    if time_up():
        return incumbent

    x = incumbent.copy()
    # 2) dual stress
    try:
        I, J, pi, mcur = _single_lp_duals(x)
    except Exception:
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mcur = float(G.max())
        iu = np.triu_indices(n, 1)
        c = G[iu]
        sel = np.where(c > mcur - 1e-9)[0]
        I, J = iu[0][sel], iu[1][sel]
        pi = np.ones(len(sel))
    stress = np.zeros(n)
    np.add.at(stress, I, pi)
    np.add.at(stress, J, pi)
    # contacts per point for rattlers
    Gc = x @ x.T
    np.fill_diagonal(Gc, -2.0)
    m2 = float(Gc.max())
    cnt = (Gc > m2 - 1e-9).sum(axis=1)
    ratt = np.where(cnt == 0)[0]
    if len(ratt) == 0:
        ratt = np.argsort(cnt)[:2]
    elif len(ratt) > 2:
        ratt = ratt[:2]
    # highest-stress connected cluster
    K = 14
    # adjacency
    adj = [[] for _ in range(n)]
    for a, b in zip(I.tolist(), J.tolist()):
        adj[a].append(b)
        adj[b].append(a)
    seed = int(np.argmax(stress))
    # avoid seed being rattler
    order = np.argsort(-stress)
    for o in order:
        if o not in set(ratt.tolist()):
            seed = int(o)
            break
    cluster = set([seed])
    while len(cluster) < K:
        frontier = set()
        for c_ in cluster:
            for nb in adj[c_]:
                if nb not in cluster and nb not in set(ratt.tolist()):
                    frontier.add(nb)
        if not frontier:
            # add highest stress outside
            for o in order:
                if o not in cluster and o not in set(ratt.tolist()):
                    frontier.add(int(o))
                    break
            if not frontier:
                break
        # pick highest stress frontier
        best_f = max(list(frontier), key=lambda v: stress[v])
        cluster.add(int(best_f))
        if len(cluster) >= K:
            break
    cluster = list(cluster)[:K]
    removed = list(dict.fromkeys(list(ratt.tolist()) + cluster))
    # ensure 12..20 removed; adjust
    if len(removed) < 12:
        for o in order:
            if o not in set(removed):
                removed.append(int(o))
            if len(removed) >= 12:
                break
    if len(removed) > 20:
        removed = removed[:20]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[np.array(removed, dtype=int)] = False
    F = x[keep_mask].copy()

    # 3) reinsert one at a time, frozen
    y_full = x.copy()
    # we will rebuild F incrementally
    curF = F.copy()
    # insertion order: rattlers first then cluster by ascending stress
    cl_sorted = sorted(cluster, key=lambda v: stress[v])
    ins_order = list(ratt.tolist()) + cl_sorted
    # dedup preserve order
    seen = set()
    uniq_order = []
    for v in ins_order:
        if v not in seen:
            seen.add(v)
            uniq_order.append(v)
    # if removed larger, add rest
    for v in removed:
        if v not in seen:
            seen.add(v)
            uniq_order.append(v)

    for idx_pos, orig_idx in enumerate(uniq_order):
        if time_up() or (time.time() - t0) > DEADLINE - 25.0:
            break
        m_cur = curF.shape[0]
        # 50000 random sphere samples for deepest holes
        try:
            cand = rng.standard_normal(size=(50000, d))
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            dots = cand @ curF.T
            worst = dots.max(axis=1)
            best5 = np.argpartition(worst, 5)[:5]
            holes = cand[best5]
        except Exception:
            holes = rng.standard_normal(size=(5, d))
            holes /= np.linalg.norm(holes, axis=1, keepdims=True)
        starts = rng.standard_normal(size=(200, d))
        starts /= np.linalg.norm(starts, axis=1, keepdims=True)
        S = np.vstack([starts, holes])
        Sref = _refine_single(curF, S, alpha1=25.0, steps1=50, alpha2=80.0, steps2=50, lr=0.5)
        # pick best by true max
        Df = Sref @ curF.T
        wv = Df.max(axis=1)
        bi = int(np.argmin(wv))
        hbest = Sref[bi] / np.linalg.norm(Sref[bi])
        curF = np.vstack([curF, hbest.reshape(1, -1)])
        y_full[orig_idx] = hbest
    if time_up():
        return incumbent

    # 4) unfreeze, short exponential smoothing
    y = y_full.copy()
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    try:
        y = _exp_steps(y, steps=150, alpha=30.0, lr=0.12 / 30.0)
        y = _exp_steps(y, steps=120, alpha=80.0, lr=0.12 / 80.0)
    except Exception:
        pass
    v_new = _max_cos(y)
    if time_up():
        # return best between incumbent and y
        if v_new < inc_val:
            return y
        return incumbent

    # 5) final LP polish under strict deadline
    remain = DEADLINE - (time.time() - t0) - 0.8
    if remain > 1.5:
        yp, vp, _ = _lp_polish(y, seconds=remain, t0=t0, deadline=DEADLINE, init_radius=0.006)
        y, v_new = yp, vp
    # choose best
    if v_new < inc_val - 1e-15:
        incumbent, inc_val = y, v_new
    # safety: if time exceeded limit, still return
    incumbent /= np.linalg.norm(incumbent, axis=1, keepdims=True)
    return incumbent
