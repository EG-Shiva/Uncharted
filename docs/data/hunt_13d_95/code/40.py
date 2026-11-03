import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def lp_polish(x, seconds, t_deadline=None):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if t_deadline is not None and time.time() > t_deadline:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
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
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
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
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def refine_hole_vec(v0, Y, beta=300.0, steps=150, step0=0.08):
    v = v0.copy()
    v /= np.linalg.norm(v)
    for s in range(steps):
        dots = Y @ v
        m = float(np.max(dots))
        w = np.exp(beta * (dots - m))
        grad = w @ Y
        gd = float(np.dot(grad, v))
        grad = grad - gd * v
        gn = float(np.linalg.norm(grad))
        if gn > 1e-14:
            grad /= gn
        step = step0 * (1.0 - 0.7 * s / max(1, steps))
        v = v - step * grad
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-14:
            break
        v /= nrm
    return v


def refine_inserted(Xfull, ins_idx, steps=350):
    n, d = Xfull.shape
    k = len(ins_idx)
    Xw = Xfull.copy()
    V = np.zeros((k, d))
    mu = 0.85
    best_X = Xw.copy()
    best_v = max_cos(Xw)
    ins_idx = np.asarray(ins_idx, dtype=int)
    for s in range(steps):
        frac = s / float(max(1, steps))
        beta = 50.0 * ((800.0 / 50.0) ** frac)
        step = 0.03 * (1.0 - 0.6 * frac)
        Xi = Xw[ins_idx]
        G = Xi @ Xw.T
        # kill self interaction
        G[np.arange(k), ins_idx] = -np.inf
        m = float(np.max(G))
        E = np.exp(beta * (G - m))
        E[np.arange(k), ins_idx] = 0.0
        grad = E @ Xw
        gdot = np.sum(grad * Xi, axis=1, keepdims=True)
        grad = grad - gdot * Xi
        gn = float(np.linalg.norm(grad, axis=1).max())
        if gn > 1e-14:
            grad /= gn
        V = mu * V + (1.0 - mu) * grad
        Xi2 = Xi - step * V
        Xi2 /= np.linalg.norm(Xi2, axis=1, keepdims=True)
        Xw[ins_idx] = Xi2
        if (s + 1) % 50 == 0 or s == steps - 1:
            v = max_cos(Xw)
            if v < best_v:
                best_v = v
                best_X = Xw.copy()
    return best_X, best_v


def full_soft(Xc, steps=200):
    n, d = Xc.shape
    Xw = Xc.copy()
    V = np.zeros_like(Xw)
    mu = 0.9
    best_X = Xw.copy()
    best_v = max_cos(Xw)
    for s in range(steps):
        frac = s / float(max(1, steps))
        beta = 30.0 * ((400.0 / 30.0) ** frac)
        step = 0.02 * (1.0 - 0.7 * frac)
        G = Xw @ Xw.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        E = np.exp(beta * (G - m))
        np.fill_diagonal(E, 0.0)
        grad = E @ Xw
        gdot = np.sum(grad * Xw, axis=1, keepdims=True)
        grad -= gdot * Xw
        gn = float(np.linalg.norm(grad, axis=1).max())
        if gn > 1e-12:
            grad /= gn
        V = mu * V + (1.0 - mu) * grad
        Xw = Xw - step * V
        Xw /= np.linalg.norm(Xw, axis=1, keepdims=True)
        if (s + 1) % 100 == 0 or s == steps - 1:
            v = max_cos(Xw)
            if v < best_v:
                best_v = v
                best_X = Xw.copy()
    return best_X, best_v


def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    SEARCH_END = t0 + 150.0

    X0 = None
    for fn in ("best.npy", "record.npy"):
        try:
            X0 = np.load(fn)
            break
        except Exception:
            continue
    if X0 is None:
        rng0 = np.random.default_rng(0)
        X0 = rng0.standard_normal((95, 13))
    X0 = np.asarray(X0, dtype=np.float64)
    if X0.shape != (95, 13):
        rng0 = np.random.default_rng(1)
        tmp = rng0.standard_normal((95, 13))
        r = min(95, X0.shape[0])
        c = min(13, X0.shape[1] if X0.ndim > 1 else 13)
        tmp[:r, :c] = X0[:r, :c]
        X0 = tmp
    X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
    best = X0.copy()
    best_val = max_cos(best)
    n, d = 95, 13
    rng = np.random.default_rng(777)

    targets = [18, 20, 16, 22, 15, 19, 21, 17]
    attempt = 0
    while time.time() < SEARCH_END:
        if time.time() > DEADLINE - 25:
            break
        attempt += 1
        # contact analysis of current best
        G0 = best @ best.T
        np.fill_diagonal(G0, -10.0)
        m0 = float(G0.max())
        thr = 1e-7
        contacts = np.sum(G0 > m0 - thr, axis=1)
        order_asc = np.argsort(contacts, kind="stable")
        order_desc = order_asc[::-1]
        rattlers = np.where(contacts == 0)[0].tolist()
        low = np.where(contacts < 8)[0].tolist()
        base_set = []
        seen = set()
        for i in rattlers + low:
            if i not in seen:
                seen.add(i)
                base_set.append(int(i))
        K = targets[(attempt - 1) % len(targets)]
        # choose hub: cycle over top-4 hubs, bias to index 44 if high
        top_hubs = list(order_desc[:4])
        # ensure 44 is tried often if it is heavily contacted
        if 44 < n and contacts[44] >= np.sort(contacts)[-5]:
            top_hubs[(attempt - 1) % 4] = 44
        hub = int(top_hubs[(attempt - 1) % len(top_hubs)])
        # truncate base if too large
        if len(base_set) > K:
            # keep lowest-contact ones up to K
            base_set = [int(i) for i in order_asc[:K]]
            remove = sorted(base_set)
        else:
            # BFS cluster around hub to fill to K
            adj = G0 > m0 - thr
            in_remove = set(base_set)
            cluster = []
            visited = set()
            queue = [hub]
            visited.add(hub)
            # if hub already in base, start from next hub neighbour
            qi = 0
            while qi < len(queue) and len(base_set) + len(cluster) < K:
                u = queue[qi]
                qi += 1
                if u not in in_remove and u not in cluster:
                    cluster.append(int(u))
                    if len(base_set) + len(cluster) >= K:
                        break
                # neighbours sorted by tightness
                nbrs = np.where(adj[u])[0]
                if len(nbrs) > 0:
                    nbrs = nbrs[np.argsort(-G0[u, nbrs])]
                    for w in nbrs:
                        w = int(w)
                        if w not in visited:
                            visited.add(w)
                            queue.append(w)
                # safety: if queue exhausted, add lowest contact not yet included
                if qi >= len(queue) and len(base_set) + len(cluster) < K:
                    for cand in order_asc:
                        cand = int(cand)
                        if cand not in in_remove and cand not in cluster and cand not in visited:
                            queue.append(cand)
                            visited.add(cand)
                            break
            remove = sorted(base_set + cluster)
            # enforce 15-22 size
            if len(remove) < 15:
                for cand in order_asc:
                    cand = int(cand)
                    if cand not in remove:
                        remove.append(cand)
                    if len(remove) >= 15:
                        break
                remove = sorted(remove)
            if len(remove) > 22:
                remove = sorted(remove)[:22]
        k = len(remove)
        if k < 10:
            continue
        keep = [i for i in range(n) if i not in set(remove)]
        core = best[np.array(keep)]
        # reinsert one at a time into largest holes
        cur_list = [row for row in core]
        Y = np.stack(cur_list, axis=0)
        # time box per attempt
        for pos in range(k):
            if time.time() > SEARCH_END and pos > 0:
                break
            ncand = 20000
            C = rng.standard_normal((ncand, d))
            C /= np.linalg.norm(C, axis=1, keepdims=True)
            dots = C @ Y.T
            vals = dots.max(axis=1)
            bi = int(np.argmin(vals))
            # take top-3 distinct candidates
            part = np.argpartition(vals, 3)[:3]
            # order them
            part = part[np.argsort(vals[part])]
            best_v_cand = np.inf
            best_c_cand = None
            for pi in part:
                v0 = C[int(pi)]
                # local minimisation of max dot
                vref = refine_hole_vec(v0, Y, beta=300.0, steps=120, step0=0.08)
                md = float(np.max(Y @ vref))
                if md < best_v_cand:
                    best_v_cand = md
                    best_c_cand = vref
            # fallback
            if best_c_cand is None:
                best_c_cand = C[bi]
            Y = np.concatenate([Y, best_c_cand[None, :]], axis=0)
        # rebuild full matrix: core rows stay, removed positions filled with last k rows of Y
        Xnew = np.zeros((n, d))
        Xnew[np.array(keep)] = core
        new_pts = Y[len(core):]
        Xnew[np.array(remove)] = new_pts
        Xnew /= np.linalg.norm(Xnew, axis=1, keepdims=True)
        # jointly refine only inserted points, frozen core
        Xins, vins = refine_inserted(Xnew, np.array(remove, dtype=int), steps=350)
        # quick check: if far worse than best, still try short full polish occasionally
        # unfreeze: short full soft + LP polish if promising
        if vins < best_val + 0.004:
            Xf, vf = full_soft(Xins, steps=150)
            # short LP polish with strict budget
            remain_lp = min(2.0, max(0.5, DEADLINE - 12.0 - time.time()))
            if remain_lp > 0.6:
                try:
                    Xp = lp_polish(Xf, remain_lp, t_deadline=DEADLINE - 8.0)
                except Exception:
                    Xp = Xf
                vp = max_cos(Xp)
            else:
                Xp, vp = Xf, vf
            if vp < best_val - 1e-12:
                best, best_val = Xp, vp
            elif vf < best_val - 1e-12:
                best, best_val = Xf, vf
        # hard time check
        if time.time() > SEARCH_END:
            break

    # final squeeze
    if time.time() < DEADLINE - 3:
        try:
            budget = min(max(2.0, DEADLINE - 2.0 - time.time()), 10.0)
            if budget > 0.5:
                best = lp_polish(best, budget, t_deadline=DEADLINE - 1.0)
        except Exception:
            pass
    best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
