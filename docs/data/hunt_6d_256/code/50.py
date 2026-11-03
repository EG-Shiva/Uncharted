import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _top_vec(x, k=2000):
    n = x.shape[0]
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    c = G[iu[0], iu[1]]
    c = np.sort(c)[::-1]
    if k is not None and k < c.size:
        c = c[:k]
    return c

def _gram_dist(a, b):
    return float(np.linalg.norm(a - b))

def _adam_lse(x0, steps, K0, K1, lr, deadline=1e18):
    x = x0.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    la0, la1 = np.log(K0), np.log(K1)
    best_val = _max_cos(x)
    best_x = x.copy()
    for t in range(1, steps + 1):
        if t % 100 == 0 and time.time() > deadline:
            break
        K = float(np.exp(la0 + (la1 - la0) * t / steps))
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        W = np.exp(K * (G - gmax))
        np.fill_diagonal(W, 0.0)
        S = float(W.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            continue
        grad = (W @ x) / S
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        b1p *= b1
        b2p *= b2
        m *= b1
        m += (1 - b1) * gt
        v *= b2
        v += (1 - b2) * (gt * gt)
        mhat = m / (1 - b1p)
        vhat = v / (1 - b2p)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        sdot = np.sum(step * x, axis=1, keepdims=True)
        step = step - sdot * x
        x = x - step
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        if t % 100 == 0 or t == steps:
            val = _max_cos(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
    return best_x, best_val

def _lp_polish(x, deadline, r0=2.0e-3, cap=3000):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    iuI, iuJ = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c = G[iuI, iuJ]
        mv = float(c.max())
        margin = max(20.0 * radius, 8e-5)
        sel = c > (mv - margin)
        I = iuI[sel]; J = iuJ[sel]
        csel = c[sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        if p > cap:
            part = np.argpartition(-csel, cap)[:cap]
            I = I[part]; J = J[part]; csel = csel[part]
            p = cap
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, n * d + 1)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, n * d + 1)).tocsr()
        except Exception:
            break
        obj = np.zeros(n * d + 1)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-csel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        try:
            val = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if val < best:
            x, best = y, val
            radius = min(radius * 1.4, 0.02)
        else:
            radius *= 0.5
    return x

def _perturb_tangent(x, sigma, rng):
    N = rng.normal(size=x.shape)
    N = N - np.sum(N * x, axis=1, keepdims=True) * x
    y = x + sigma * N
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    return y

def _procrustes_to_target(donor, target):
    M = donor.T @ target
    try:
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        Q = U @ Vt
    except Exception:
        Q = np.eye(donor.shape[1])
    return donor @ Q

def _block_indices(X, bsize, rng, method):
    n = X.shape[0]
    bsize = int(min(max(8, bsize), n))
    if method == "pca":
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            vv = Vt[0]
        except Exception:
            vv = rng.normal(size=(X.shape[1],))
            vv = vv / np.linalg.norm(vv)
        proj = X @ vv
        order = np.argsort(proj, kind="stable")
        start = int(rng.integers(0, n - bsize + 1))
        S = order[start:start + bsize]
        return S
    else:
        seed = int(rng.integers(0, n))
        dots = X @ X[seed]
        order = np.argsort(-dots, kind="stable")
        S = order[:bsize]
        return S

def solve():
    t0 = time.time()
    DEADLINE = t0 + 178.0
    GEN_DEADLINE = t0 + 130.0
    rng = np.random.default_rng(20261103)
    n, d = 256, 6
    inc_x = None
    inc_v = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (256, 6):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    vv = _max_cos(a)
                    if vv < inc_v:
                        inc_v = vv
                        inc_x = a.copy()
        except Exception:
            pass
    if inc_x is None:
        rs = np.random.default_rng(0)
        inc_x = rs.normal(size=(n, d))
        inc_x /= np.linalg.norm(inc_x, axis=1, keepdims=True)
        inc_v = _max_cos(inc_x)
    # candidate pool for void refill
    NCAND_REFILL = 20000
    Rc = rng.normal(size=(NCAND_REFILL, d))
    Rc /= np.linalg.norm(Rc, axis=1, keepdims=True)

    pool = []
    pool_scores = []
    pool_vecs = []
    # 0: incumbent itself (lightly polished? keep as is for diversity anchor)
    pool.append(inc_x.copy())
    pool_scores.append(inc_v)
    try:
        pool_vecs.append(_top_vec(inc_x))
    except Exception:
        pool_vecs.append(None)

    # generate 10-12 shallow-polished optima: tangent-noise scales + random restarts
    scales = [0.03]*3 + [0.08]*3 + [0.15]*3
    n_rand = 2
    total_wanted = 1 + len(scales) + n_rand  # 12
    # shallow polish each perturbed start
    for si, sig in enumerate(scales):
        if time.time() > GEN_DEADLINE - 60:
            break
        try:
            y0 = _perturb_tangent(inc_x, float(sig), rng)
            yp, vp = _adam_lse(y0, 350, 300.0, 5000.0, 0.008, deadline=GEN_DEADLINE)
            pool.append(yp.copy())
            pool_scores.append(float(vp))
            pool_vecs.append(_top_vec(yp))
        except Exception:
            continue
    for ri in range(n_rand):
        if time.time() > GEN_DEADLINE - 55:
            break
        try:
            y0 = rng.normal(size=(n, d))
            y0 /= np.linalg.norm(y0, axis=1, keepdims=True)
            yp, vp = _adam_lse(y0, 400, 300.0, 5000.0, 0.01, deadline=GEN_DEADLINE)
            pool.append(yp.copy())
            pool_scores.append(float(vp))
            pool_vecs.append(_top_vec(yp))
        except Exception:
            continue
    # pairwise gram distances for diversity
    P = len(pool)
    # select mating pairs: prefer diverse (large gram dist) with good avg score
    pair_list = []
    try:
        Dmat = np.zeros((P, P))
        for i in range(P):
            for j in range(i+1, P):
                if pool_vecs[i] is None or pool_vecs[j] is None:
                    dd = 0.0
                else:
                    dd = _gram_dist(pool_vecs[i], pool_vecs[j])
                Dmat[i, j] = dd
                Dmat[j, i] = dd
        med = float(np.median(Dmat[np.triu_indices(P, 1)])) if P > 2 else 0.0
        for i in range(P):
            for j in range(i+1, P):
                avg = 0.5*(pool_scores[i]+pool_scores[j])
                dd = float(Dmat[i, j])
                pair_list.append((avg - 0.02*dd, i, j, dd))
        pair_list.sort(key=lambda t: t[0])
        # keep diverse first: filter dd >= med*0.5, take up to 24
        filt = [t for t in pair_list if t[3] >= med*0.5]
        if len(filt) < 8:
            filt = pair_list
        pair_list = filt[:24]
    except Exception:
        pair_list = []
        for i in range(min(P, 6)):
            for j in range(i+1, min(P, 6)):
                pair_list.append((0, i, j, 0))
        pair_list = pair_list[:24]

    archive_X = [p.copy() for p in pool]
    archive_scores = list(pool_scores)
    archive_vecs = list(pool_vecs)
    children = []
    children_scores = []
    children_vecs = []

    block_cycle = [32, 64, 128]
    meth_cycle = ["pca", "nn"]
    DUP_THRESH = 0.003

    ci = 0
    for (_, pi, pj, _) in pair_list:
        if time.time() > GEN_DEADLINE:
            break
        # choose base = better scoring parent
        if pool_scores[pi] <= pool_scores[pj]:
            ia, ib = pi, pj
        else:
            ia, ib = pj, pi
        A = pool[ia]
        B = pool[ib]
        sa, sb = float(pool_scores[ia]), float(pool_scores[ib])
        thresh = float(max(sa, sb))
        bsize = block_cycle[ci % len(block_cycle)]
        method = meth_cycle[(ci // len(block_cycle)) % len(meth_cycle)]
        # alternate base direction every other child
        if ci % 2 == 1:
            A, B = B, A
            # thresh unchanged
        ci += 1
        try:
            Brot = _procrustes_to_target(B, A)
            Csim = A @ Brot.T
            _, col = linear_sum_assignment(-Csim)
            Bord = Brot[col]
            S = _block_indices(A, bsize, rng, method)
            Sset = set(int(s) for s in S)
            child = A.copy()
            child[S] = Bord[S]
            # collision deletion: any transplanted closer than thresh to retained
            G = child @ child.T
            np.fill_diagonal(G, -2.0)
            # retained indices
            mask_ret = np.ones(n, dtype=bool)
            mask_ret[S] = False
            to_del = []
            # for each s in S compute max to retained
            for s in S:
                s = int(s)
                mv = float(G[s, mask_ret].max()) if np.any(mask_ret) else -2.0
                if mv > thresh:
                    to_del.append((mv, s))
            # internal transplant-transplant (should be rare)
            if len(S) > 1:
                Ss = np.array([int(s) for s in S], dtype=int)
                Gss = G[np.ix_(Ss, Ss)]
                # find pairs > thresh
                # mark worse one
                for a_ii in range(len(Ss)):
                    for b_ii in range(a_ii+1, len(Ss)):
                        if float(Gss[a_ii, b_ii]) > thresh:
                            # decide which has larger cross max
                            ma = float(G[Ss[a_ii], mask_ret].max()) if np.any(mask_ret) else -2
                            mb = float(G[Ss[b_ii], mask_ret].max()) if np.any(mask_ret) else -2
                            victim = Ss[a_ii] if ma >= mb else Ss[b_ii]
                            if victim not in [t[1] for t in to_del]:
                                to_del.append((float(Gss[a_ii, b_ii]), int(victim)))
            del_idx = sorted(set(t[1] for t in to_del))
            keep_mask = np.ones(n, dtype=bool)
            if len(del_idx) > 0:
                keep_mask[np.array(del_idx, dtype=int)] = False
            F = child[keep_mask]
            need = n - F.shape[0]
            if need > 0:
                # greedy refill from Rc void candidates
                curF = F.copy()
                used = set()
                newpts = []
                # precompute? iterative
                for _ in range(need):
                    if time.time() > GEN_DEADLINE:
                        break
                    # compute max dot of each candidate to curF
                    # chunk to limit memory
                    CH = 5000
                    best_v = 1e9
                    best_c = None
                    best_i = -1
                    for a0 in range(0, NCAND_REFILL, CH):
                        Zb = Rc[a0:a0+CH]
                        Sm = Zb @ curF.T
                        cm = Sm.max(axis=1)
                        # penalize used
                        for k in range(cm.size):
                            if (a0+k) in used:
                                cm[k] = 1e9
                        bi = int(np.argmin(cm))
                        bv = float(cm[bi])
                        if bv < best_v:
                            best_v = bv
                            best_c = Zb[bi].copy()
                            best_i = a0+bi
                    if best_c is None:
                        break
                    used.add(int(best_i))
                    newpts.append(best_c)
                    curF = np.vstack([curF, best_c[None, :]])
                if len(newpts) != need:
                    # if time broke, skip this child
                    continue
                child_new = curF.copy()
            else:
                child_new = child.copy()
            # score raw child
            raw = _max_cos(child_new)
            # shallow polish child
            child_pol, vpol = _adam_lse(child_new, 250, 300.0, 5000.0, 0.008, deadline=GEN_DEADLINE+5)
            # archive duplicate check via gram distance
            try:
                vv = _top_vec(child_pol)
            except Exception:
                continue
            # compute min dist to archive
            mind = 1e9
            for av in archive_vecs + children_vecs:
                if av is None:
                    continue
                try:
                    dd = _gram_dist(vv, av)
                    if dd < mind:
                        mind = dd
                except Exception:
                    pass
            if mind < DUP_THRESH:
                # near-duplicate: keep only if strictly better than all
                if len(children_scores) == 0 or vpol < min(children_scores + archive_scores) - 1e-12:
                    pass
                else:
                    continue
            children.append(child_pol.copy())
            children_scores.append(float(vpol))
            children_vecs.append(vv)
            archive_scores.append(float(vpol))
            # do not grow archive_vecs unbounded for speed: keep it
        except Exception:
            continue

    # pick best two children for deep LP polish
    best_x = inc_x.copy()
    best_v = float(inc_v)
    # consider pool best
    for p_, s_ in zip(pool, pool_scores):
        if float(s_) < best_v:
            best_v = float(s_)
            best_x = p_.copy()
    for c_, s_ in zip(children, children_scores):
        if float(s_) < best_v:
            best_v = float(s_)
            best_x = c_.copy()
    # order children by score
    if len(children) > 0:
        order = np.argsort(np.array(children_scores, kind="stable") if False else np.array(children_scores))
        # fix argsort call
        order = np.argsort(np.array(children_scores))
        top2 = [children[int(o)] for o in order[:2]]
    else:
        top2 = []
    for ti, tch in enumerate(top2):
        if time.time() > DEADLINE - 8:
            break
        try:
            t_rem = DEADLINE - 2.0
            # joint Adam deeper
            steps_deep = 500 if ti == 0 else 350
            x1, v1 = _adam_lse(tch, steps_deep, 500.0, 8000.0, 0.006, deadline=t_rem - 6)
            # LP polish with remaining time split
            lp_dead = min(time.time() + (8.0 if ti == 0 else 5.0), DEADLINE - 1.0)
            x2 = _lp_polish(x1, lp_dead, r0=2.0e-3, cap=3000)
            v2 = _max_cos(x2)
            if v2 < best_v:
                best_v = float(v2)
                best_x = x2.copy()
            elif v1 < best_v:
                best_v = float(v1)
                best_x = x1.copy()
        except Exception:
            continue
    # fallback to loaded incumbent if none improves
    try:
        b = np.load("best.npy")
        b = np.asarray(b, dtype=float)
        if b.shape == (256, 6):
            b = b / np.linalg.norm(b, axis=1, keepdims=True)
            bv = _max_cos(b)
            if bv < best_v:
                best_v = bv
                best_x = b.copy()
    except Exception:
        pass
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
