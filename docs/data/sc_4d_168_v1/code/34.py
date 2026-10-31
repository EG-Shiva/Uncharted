import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _rand_rot(rng):
    A = rng.randn(4, 4)
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    s = np.sign(d)
    s[s == 0] = 1.0
    Q = Q * s[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q

def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    SEARCH_END = t0 + 132.0
    rng = np.random.RandomState(20261031)
    n, d = 168, 4
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr)
        xb = _norm(xb)
        if xb.shape == (n, d) and xr.shape == (n, d):
            if _max_cos(xr) < _max_cos(xb):
                xb = xr.copy()
        elif xr.shape == (n, d):
            xb = xr.copy()
    except Exception:
        pass
    if xb.shape != (n, d):
        xb = rng.randn(n, d)
    xb = _norm(xb)
    best_val0 = _max_cos(xb)

    iu0 = np.triu_indices(n, 1)

    def lp_polish(x, deadline):
        x = _norm(x)
        best = _max_cos(x)
        radius = 1e-3
        while time.time() < deadline and radius > 1e-14:
            G = x @ x.T
            c = G[iu0]
            m = float(np.max(c))
            sel = c > m - max(20.0 * radius, 1e-9)
            I = iu0[0][sel]; J = iu0[1][sel]
            csel = c[sel]
            p = len(I)
            if p == 0:
                break
            nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d),
                                   J[:, None] * d + np.arange(d),
                                   np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
            except Exception:
                break
            if res.status != 0:
                radius *= 0.5
                continue
            y = x + res.x[:-1].reshape(n, d)
            y = _norm(y)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.05)
            else:
                radius *= 0.5
        return x, best

    def refine(x_init, steps, beta0=30.0, beta1=600.0, lr0=0.016, lr1=0.0008, mu=0.88):
        x = x_init.copy()
        vel = np.zeros_like(x)
        lb0 = np.log(beta0); lb1 = np.log(beta1)
        for s in range(steps):
            if time.time() > SEARCH_END:
                break
            f = s / max(1, steps - 1)
            beta = float(np.exp(lb0 + (lb1 - lb0) * f))
            lr = lr0 * (1.0 - f) + lr1 * f
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp(np.clip(beta * (G - cmax), -25.0, 0.0))
            np.fill_diagonal(E, 0.0)
            rs = np.sum(E, axis=1, keepdims=True) + 1e-12
            grad = (E @ x) / rs
            dot = np.sum(grad * x, axis=1, keepdims=True)
            gt = grad - dot * x
            sc = np.mean(np.linalg.norm(gt, axis=1)) + 1e-12
            gt = gt / sc
            vel = mu * vel + (1.0 - mu) * gt
            x = x - lr * vel
            x = _norm(x)
        return x

    def refine_single(x0, Xref, steps=100, beta0=45.0, beta1=280.0, lr0=0.09, lr1=0.006):
        s = x0 / (np.linalg.norm(x0) + 1e-30)
        lb0 = np.log(beta0); lb1 = np.log(beta1)
        for it in range(steps):
            f = it / max(1, steps - 1)
            beta = float(np.exp(lb0 + (lb1 - lb0) * f))
            lr = lr0 * (1.0 - f) + lr1 * f
            dots = Xref @ s
            cm = float(np.max(dots))
            w = np.exp(np.clip(beta * (dots - cm), -25.0, 0.0))
            w /= (np.sum(w) + 1e-30)
            g = w @ Xref
            gt = g - np.dot(g, s) * s
            gn = np.linalg.norm(gt) + 1e-12
            gt = gt / gn
            s = s - lr * gt
            ns = np.linalg.norm(s)
            if ns < 1e-12:
                s = rng.randn(4); ns = np.linalg.norm(s)
            s = s / ns
        return s

    def deepest_hole(Xref, n_sample=8000):
        C = rng.randn(n_sample, 4)
        C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-30)
        D = C @ Xref.T
        m = D.max(axis=1)
        return C[int(np.argmin(m))]

    def contact_hash(x):
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        slack = m - G[iu0]
        p1 = int(np.sum(slack < 1e-4))
        c2 = int(np.sum(slack < 1e-3))
        c3 = int(np.sum(slack < 5e-3))
        deg = np.sum(G > m - 1e-4, axis=1)
        sdeg = tuple(sorted(deg.astype(int).tolist()))
        return (p1, sdeg, c2, c3)

    def align_B_to_A(A, B):
        best_R = np.eye(4)
        best_score = -1e100
        inits = [np.eye(4), _rand_rot(rng), _rand_rot(rng)]
        for R0 in inits:
            R = R0.copy()
            for _ in range(5):
                Br = B @ R
                D = A @ Br.T
                j = np.argmax(D, axis=1)
                score = float(np.mean(D[np.arange(n), j]))
                H = B[j].T @ A
                try:
                    U, S, Vt = np.linalg.svd(H)
                except Exception:
                    break
                Rn = U @ Vt
                if np.linalg.det(Rn) < 0:
                    U[:, -1] *= -1
                    Rn = U @ Vt
                R = Rn
            Br = B @ R
            D = A @ Br.T
            score = float(np.mean(np.max(D, axis=1)))
            if score > best_score:
                best_score = score
                best_R = R.copy()
        return B @ best_R, best_R

    # archive: list of [x, val, h]
    archive = []
    hash_to_best = {}
    global_best_val = float('inf')
    global_best_x = xb.copy()

    def archive_try_add(x, v):
        nonlocal global_best_val, global_best_x
        h = contact_hash(x)
        improved_global = v < global_best_val - 1e-15
        if improved_global:
            global_best_val = v
            global_best_x = x.copy()
        if h in hash_to_best:
            # novel histogram already seen: only accept if improves that hash or global
            if v < hash_to_best[h] - 1e-12:
                hash_to_best[h] = v
                # replace the entry with same hash if better
                for e in archive:
                    if e[2] == h and v < e[1]:
                        e[0][:] = x
                        e[1] = v
                        break
                return True
            else:
                if improved_global:
                    # allow duplicate hash if global best
                    hash_to_best[h] = v
                    for e in archive:
                        if e[2] == h and v < e[1]:
                            e[0][:] = x
                            e[1] = v
                            break
                    return True
                return False
        else:
            # novel hash
            # acceptance threshold: must be within +2e-4 of global best (before insertion)
            # use updated global best
            if v > global_best_val + 2e-4 + 1e-12:
                return False
            hash_to_best[h] = v
            if len(archive) < 10:
                archive.append([x.copy(), v, h])
            else:
                # replace worst
                idx = int(np.argmax([e[1] for e in archive]))
                if v < archive[idx][1]:
                    oldh = archive[idx][2]
                    # remove old hash if no other entry has it
                    if sum(1 for e in archive if e[2] == oldh) <= 1:
                        hash_to_best.pop(oldh, None)
                        # but keep if it was re-added? already handled
                        if oldh not in [e[2] for e in archive if e is not archive[idx]]:
                            pass
                    archive[idx] = [x.copy(), v, h]
                else:
                    # novel but worse than all: to preserve diversity, replace worst
                    # only if not too worse (within 2e-4 already checked)
                    # replace worst to keep walk diverse
                    oldh = archive[idx][2]
                    if sum(1 for e in archive if e[2] == oldh) <= 1:
                        hash_to_best.pop(oldh, None)
                    archive[idx] = [x.copy(), v, h]
            return True

    # init archive with record/best
    archive_try_add(_norm(xb), _max_cos(_norm(xb)))
    # strongly kicked + hole variants to fill archive
    n_init = 0
    while len(archive) < 6 and time.time() < SEARCH_END - 5.0 and n_init < 12:
        n_init += 1
        base = _norm(xb)
        if n_init % 2 == 0:
            y = base + 0.28 * rng.randn(n, d)
            y = _norm(y)
            y = refine(y, 350, 20.0, 500.0, 0.018, 0.0008)
        else:
            y = base.copy()
            k = int(rng.choice([3, 4, 5, 6]))
            victims = rng.choice(n, size=k, replace=False)
            mask = np.ones(n, dtype=bool); mask[victims] = False
            Xk = y[mask]
            newp = []
            for ki in range(k):
                Xref = Xk if ki == 0 else np.vstack([Xk, np.array(newp)])
                c0 = deepest_hole(Xref, n_sample=6000)
                xn = refine_single(c0, Xref, steps=80)
                newp.append(xn)
            y[victims] = np.array(newp)
            y = _norm(y + 0.005 * rng.randn(n, d))
            y = refine(y, 350, 30.0, 600.0, 0.015, 0.0007)
        v = _max_cos(y)
        # light LP for first two to get polished optima
        if n_init <= 2 and v < global_best_val + 0.001 and time.time() < SEARCH_END - 4.0:
            try:
                y2, v2 = lp_polish(y, min(time.time() + 0.9, SEARCH_END))
                y, v = y2, v2
            except Exception:
                pass
        archive_try_add(y, v)

    # ensure best polished once at start
    if time.time() < SEARCH_END - 3.0:
        try:
            yp, vp = lp_polish(global_best_x, min(time.time() + 1.2, SEARCH_END))
            archive_try_add(yp, vp)
        except Exception:
            pass

    def make_kick_offspring(parent):
        x = parent.copy()
        typ = rng.randint(0, 6)
        if typ == 0:
            x = x + 0.03 * rng.randn(n, d)
            x = _norm(x)
            x = refine(x, 450, 30.0, 700.0, 0.015, 0.0007)
        elif typ == 1:
            x = x + 0.10 * rng.randn(n, d)
            x = _norm(x)
            x = refine(x, 550, 22.0, 550.0, 0.017, 0.0008)
        elif typ == 2:
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            m = float(np.max(G))
            deg = np.sum(G > m - 0.004, axis=1)
            k_move = int(rng.choice([2, 3, 4, 5]))
            order = np.argsort(-deg, kind="stable")
            victims = order[:k_move].copy()
            mask = np.ones(n, dtype=bool); mask[victims] = False
            Xk = x[mask]
            newp = []
            for ki in range(k_move):
                Xref = Xk if ki == 0 else np.vstack([Xk, np.array(newp)])
                c0 = deepest_hole(Xref, n_sample=7000)
                xn = refine_single(c0, Xref, steps=100)
                newp.append(xn)
            x[victims] = np.array(newp)
            x = _norm(x + 0.005 * rng.randn(n, d))
            x = refine(x, 500, 30.0, 700.0, 0.015, 0.0007)
        elif typ == 3:
            c = rng.randint(n)
            G = x @ x.T
            nbrs = np.argsort(-G[c])[:int(rng.choice([4, 6, 8]))]
            x[nbrs] = x[nbrs] + 0.14 * rng.randn(len(nbrs), d)
            x = _norm(x)
            x = refine(x, 500, 25.0, 600.0, 0.016, 0.0007)
        elif typ == 4:
            k_move = int(rng.choice([1, 2, 3]))
            victims = rng.choice(n, size=k_move, replace=False)
            mask = np.ones(n, dtype=bool); mask[victims] = False
            Xk = x[mask]
            newp = []
            for ki in range(k_move):
                Xref = Xk if ki == 0 else np.vstack([Xk, np.array(newp)])
                c0 = deepest_hole(Xref, n_sample=8000)
                xn = refine_single(c0, Xref, steps=110)
                newp.append(xn)
            x[victims] = np.array(newp)
            x = _norm(x + 0.004 * rng.randn(n, d))
            x = refine(x, 500, 30.0, 700.0, 0.014, 0.0006)
        else:
            x = x + 0.20 * rng.randn(n, d)
            x = _norm(x)
            x = refine(x, 600, 18.0, 450.0, 0.018, 0.0008)
        return x

    def make_crossover_offspring(A, B):
        try:
            Bal, _ = align_B_to_A(A, B)
        except Exception:
            Bal = B.copy()
        u = rng.randn(4); u /= (np.linalg.norm(u) + 1e-30)
        pa = A @ u
        pb = Bal @ u
        # cap size 1/3 to 1/2
        frac = rng.uniform(0.33, 0.5)
        k = int(round(n * frac))
        # donor cap: top-k of Bal along u (random sign)
        if rng.rand() < 0.5:
            pa = -pa; pb = -pb
        idxB = np.argsort(-pb)[:k]
        idxA = np.argsort(-pa)[:k]
        donors = Bal[idxB].copy()
        C = A.copy()
        C[idxA] = donors
        # repair near-duplicates: if any pair > 0.97 (or > current max+0.05) replace
        G = C @ C.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        # find duplicate pairs with very high cos
        if m > 0.96:
            I, J = np.where(G > 0.96)
            # collect second indices to replace
            seen = set()
            reps = []
            for a, b in zip(I.tolist(), J.tolist()):
                if a >= b:
                    continue
                if b not in seen:
                    seen.add(b)
                    reps.append(b)
                if len(reps) >= 4:
                    break
            if len(reps) > 0:
                mask = np.ones(n, dtype=bool); mask[reps] = False
                Xk = C[mask]
                newp = []
                for ki, ridx in enumerate(reps):
                    Xref = Xk if ki == 0 else np.vstack([Xk, np.array(newp)])
                    c0 = deepest_hole(Xref, n_sample=7000)
                    xn = refine_single(c0, Xref, steps=90)
                    newp.append(xn)
                for ridx, xn in zip(reps, newp):
                    C[ridx] = xn
        C = _norm(C + 0.004 * rng.randn(n, d))
        C = refine(C, 550, 30.0, 700.0, 0.014, 0.0007)
        return C

    trial = 0
    while time.time() < SEARCH_END:
        trial += 1
        if time.time() > SEARCH_END - 1.0:
            break
        # select parent (not always best)
        vals = np.array([e[1] for e in archive])
        order = np.argsort(vals)
        rsel = rng.rand()
        if rsel < 0.45:
            pidx = order[0]
        elif rsel < 0.75:
            # rank-weighted among top half
            m2 = max(2, len(archive) // 2 + 1)
            pidx = order[rng.randint(0, min(m2, len(archive)))]
        else:
            pidx = order[rng.randint(0, len(archive))]
        parent = archive[pidx][0]
        # choose offspring type
        if rng.rand() < 0.62 or len(archive) < 2:
            child = make_kick_offspring(parent)
        else:
            # pick second parent distinct
            j = rng.randint(0, len(archive))
            if archive[j][0] is parent and len(archive) > 1:
                j = (j + 1) % len(archive)
            A = parent
            B = archive[j][0]
            # randomize direction A/B
            if rng.rand() < 0.5:
                A, B = B, A
            child = make_crossover_offspring(A, B)
        v = _max_cos(child)
        # worse-accepting threshold +2e-4
        if v > global_best_val + 2e-4 + 1e-12:
            continue
        # opportunistic LP polish if promising
        do_lp = False
        if v < global_best_val + 0.0008 and time.time() < SEARCH_END - 2.5:
            if v < global_best_val or (v < global_best_val + 0.00015 and (trial % 2 == 0)):
                do_lp = True
        if do_lp:
            try:
                child2, v2 = lp_polish(child, min(time.time() + 1.0, SEARCH_END))
                child, v = child2, v2
            except Exception:
                pass
        archive_try_add(child, v)

    # final squeeze on global best
    try:
        gx, gv = lp_polish(global_best_x, HARD_END - 0.5)
        global_best_x, global_best_val = gx, gv
    except Exception:
        pass
    global_best_x = _norm(global_best_x)
    return global_best_x
