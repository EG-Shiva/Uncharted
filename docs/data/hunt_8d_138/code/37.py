import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def _roll_mat(d, shift):
    M = np.zeros((d, d))
    for i in range(d):
        M[i, (i - shift) % d] = 1.0
    return M

def _rev_mat(d):
    M = np.zeros((d, d))
    for i in range(d):
        M[i, (-i) % d] = 1.0
    return M

def _block_roll_mat(d, block, shift):
    # block 0: first 4, block 1: last 4 (d=8)
    M = np.zeros((d, d))
    for i in range(d):
        M[i, i] = 1.0
    if block == 0:
        idx = list(range(4))
    else:
        idx = list(range(4, 8))
    b = len(idx)
    P = np.zeros((d, d))
    # start identity then permute block
    for i in range(d):
        if i in idx:
            pos = idx.index(i)
            src = idx[(pos - shift) % b]
            P[i, src] = 1.0
        else:
            P[i, i] = 1.0
    return P

def _closure(gens, limit=256):
    d = gens[0].shape[0]
    grp = [np.eye(d)]
    seen = {tuple(np.eye(d).ravel().round(6))}
    stack = [np.eye(d)]
    while stack:
        cur = stack.pop()
        for g in gens:
            for nxt in (cur @ g, g @ cur):
                key = tuple(nxt.ravel().round(6))
                if key not in seen:
                    seen.add(key)
                    grp.append(nxt)
                    stack.append(nxt)
                    if len(grp) >= limit:
                        return grp
    return grp

def _build_groups(d=8):
    R = _roll_mat(d, 1)
    Rev = _rev_mat(d)
    Neg = -np.eye(d)
    R1 = _block_roll_mat(d, 0, 1)
    R2 = _block_roll_mat(d, 1, 1)
    groups = {}
    groups['C8'] = _closure([R])
    groups['C8xNeg'] = _closure([R, Neg])
    groups['D8'] = _closure([R, Rev])
    groups['Block44'] = _closure([R1, R2])
    groups['D8xNeg'] = _closure([R, Rev, Neg])
    return groups

def _expand(Snorm, Gmats):
    # Snorm (k,d), Gmats list of (d,d), action row: s @ G.T
    k = Snorm.shape[0]
    G = len(Gmats)
    d = Snorm.shape[1]
    X = np.empty((k * G, d))
    for gi, Gm in enumerate(Gmats):
        X[gi * k:(gi + 1) * k] = Snorm @ Gm.T
    return X

def _prune_to_n(X, n):
    X = X.copy()
    while X.shape[0] > n:
        m = X.shape[0]
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        # find max pair
        flat = int(np.argmax(G))
        ii, jj = flat // m, flat % m
        # second max excluding each other
        Gi = G[ii].copy(); Gi[jj] = -2.0
        Gj = G[jj].copy(); Gj[ii] = -2.0
        mi = Gi.max(); mj = Gj.max()
        # also consider mean crowding as tie breaker
        if mi > mj:
            victim = ii
        elif mj > mi:
            victim = jj
        else:
            # remove one with larger sum
            si = Gi.sum(); sj = Gj.sum()
            victim = ii if si >= sj else jj
        X = np.delete(X, victim, axis=0)
    return X

def _lse_grad_X(X, beta):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    E = np.exp(beta * (G - m))
    np.fill_diagonal(E, 0.0)
    S = float(E.sum() / 2.0) + 1e-300
    loss = m + np.log(S) / beta
    W = E / S  # (E/S_triu) @ X gives correct grad
    Grad = W @ X
    return loss, Grad, m

def _adam_sym(Gmats, Ysym0, Yfree0, betas, iters_list, lrs, deadline, rng=None):
    k, d = Ysym0.shape
    Ysym = Ysym0.copy()
    has_free = Yfree0 is not None and Yfree0.shape[0] > 0
    if has_free:
        Yfree = Yfree0.copy()
    else:
        Yfree = None
    ms = np.zeros_like(Ysym); vs = np.zeros_like(Ysym)
    if has_free:
        mf = np.zeros_like(Yfree); vf = np.zeros_like(Yfree)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    Gnum = len(Gmats)
    # stack Gmats for fast expand: (G,d,d)
    GST = np.stack([g.T for g in Gmats], axis=0)  # G,d,d transposed already? s@G.T = einsum
    for beta, iters, lr in zip(betas, iters_list, lrs):
        for t in range(iters):
            if time.time() > deadline:
                break
            step += 1
            ns = np.sqrt(np.sum(Ysym * Ysym, axis=1, keepdims=True)) + 1e-30
            Sn = Ysym / ns
            # expand: Xsym (G*k,d)
            # Sn (k,d), GST (G,d,d): X[g*k:(g+1)*k]= Sn @ GST[g]
            Xsym = np.empty((k * Gnum, d))
            for gi in range(Gnum):
                Xsym[gi * k:(gi + 1) * k] = Sn @ GST[gi]
            if has_free:
                nf = np.sqrt(np.sum(Yfree * Yfree, axis=1, keepdims=True)) + 1e-30
                Xfree = Yfree / nf
                X = np.vstack([Xsym, Xfree])
            else:
                X = Xsym
                nf = None; Xfree = None
            loss, GradX, m = _lse_grad_X(X, beta)
            # split grad
            GradSym_exp = GradX[:k * Gnum]
            # accumulate to Sn: dL/dSn[i] = sum_g Grad[g,i] @ G_g  (since X= Sn @ G.T => dL/dSn = Grad @ G)
            GradSn = np.zeros_like(Sn)
            for gi in range(Gnum):
                Gb = Gmats[gi]  # d,d
                GradSn += GradSym_exp[gi * k:(gi + 1) * k] @ Gb
            # chain through normalization
            dot_s = np.sum(GradSn * Sn, axis=1, keepdims=True)
            GradYsym = (GradSn - dot_s * Sn) / ns
            # adam update sym
            ms = b1 * ms + (1 - b1) * GradYsym
            vs = b2 * vs + (1 - b2) * GradYsym * GradYsym
            mhat = ms / (1 - b1 ** step)
            vhat = vs / (1 - b2 ** step)
            Ysym -= lr * mhat / (np.sqrt(vhat) + eps)
            if has_free:
                GradXf = GradX[k * Gnum:]
                dot_f = np.sum(GradXf * Xfree, axis=1, keepdims=True)
                GradYf = (GradXf - dot_f * Xfree) / nf
                mf = b1 * mf + (1 - b1) * GradYf
                vf = b2 * vf + (1 - b2) * GradYf * GradYf
                mhatf = mf / (1 - b1 ** step)
                vhatf = vf / (1 - b2 ** step)
                Yfree -= lr * mhatf / (np.sqrt(vhatf) + eps)
        if time.time() > deadline:
            break
    # final X
    ns = np.sqrt(np.sum(Ysym * Ysym, axis=1, keepdims=True)) + 1e-30
    Sn = Ysym / ns
    Xsym = np.empty((k * Gnum, d))
    for gi in range(Gnum):
        Xsym[gi * k:(gi + 1) * k] = Sn @ Gmats[gi].T
    if has_free:
        nf = np.sqrt(np.sum(Yfree * Yfree, axis=1, keepdims=True)) + 1e-30
        Xfree = Yfree / nf
        X = np.vstack([Xsym, Xfree])
    else:
        X = Xsym
    return _norm(X)

def _adam_free(X0, betas, iters_list, lrs, deadline):
    n, d = X0.shape
    Y = X0.copy() * 1.0
    # ensure nonzero
    m = np.zeros_like(Y); v = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    for beta, iters, lr in zip(betas, iters_list, lrs):
        for t in range(iters):
            if time.time() > deadline:
                break
            step += 1
            nn = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True)) + 1e-30
            X = Y / nn
            _, GradX, _ = _lse_grad_X(X, beta)
            dot = np.sum(GradX * X, axis=1, keepdims=True)
            GradY = (GradX - dot * X) / nn
            m = b1 * m + (1 - b1) * GradY
            v = b2 * v + (1 - b2) * GradY * GradY
            mhat = m / (1 - b1 ** step)
            vhat = v / (1 - b2 ** step)
            Y -= lr * mhat / (np.sqrt(vhat) + eps)
        if time.time() > deadline:
            break
    return _norm(Y)

def _tangent_noise(X, sigma, rng):
    N = rng.standard_normal(X.shape)
    dot = np.sum(N * X, axis=1, keepdims=True)
    Nt = N - dot * X
    Y = X + sigma * Nt
    return _norm(Y)

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
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
        y = y / np.linalg.norm(y, axis=1, keepdims=True)
        vv = _max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 168.0
    n, d = 138, 8
    rng = np.random.default_rng(0)
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            vv = np.load(fn).astype(float)
            if vv.shape == (n, d):
                vv = _norm(vv)
                c = _max_cos(vv)
                if c < best_val:
                    best_val = c; best = vv.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((n, d)))
        best_val = _max_cos(best)

    groups = _build_groups(d)
    # (group_name, k seeds)
    configs = [
        ('C8xNeg', 9),
        ('D8', 9),
        ('Block44', 8),
        ('C8', 12),
        ('D8xNeg', 6),
        ('C8xNeg', 7),
    ]
    cands = []
    # symmetric phase budget: leave ~70s for breaking+polish
    sym_deadline = t0 + 95.0
    betas = [30.0, 80.0, 180.0, 350.0]
    iters_list = [350, 350, 450, 450]
    lrs = [0.015, 0.01, 0.008, 0.005]
    for ci, (gname, k) in enumerate(configs):
        if time.time() > sym_deadline:
            break
        Gmats = groups[gname]
        Gnum = len(Gmats)
        M = k * Gnum
        free_n = (n - M) if M < n else 0
        # deterministic seed per config
        r2 = np.random.default_rng(1000 + ci)
        Ysym0 = r2.standard_normal((k, d))
        Yfree0 = r2.standard_normal((free_n, d)) if free_n > 0 else np.zeros((0, d))
        try:
            Xfull = _adam_sym(Gmats, Ysym0, Yfree0, betas, iters_list, lrs, min(sym_deadline, deadline - 75.0), r2)
        except Exception:
            continue
        if Xfull.shape[0] > n:
            X138 = _prune_to_n(Xfull, n)
        elif Xfull.shape[0] < n:
            # should not happen (free pads exactly), but pad randomly if needed
            need = n - Xfull.shape[0]
            extra = _norm(r2.standard_normal((need, d)))
            X138 = np.vstack([Xfull, extra])
        else:
            X138 = Xfull
        X138 = _norm(X138)
        val = _max_cos(X138)
        cands.append((val, X138.copy()))
        # keep sorted, keep top 3
        cands.sort(key=lambda t: t[0])
        cands = cands[:3]
        if val < best_val - 1e-12:
            best_val = val; best = X138.copy()

    # symmetry breaking on top symmetric candidates
    # choose up to 2 best symmetric cands
    cands.sort(key=lambda t: t[0])
    targets = []
    for _, Xc in cands[:2]:
        targets.append(Xc.copy())
    # also include incumbent to allow escape? include best with noise later
    if len(targets) == 0:
        targets = [best.copy()]

    # successive anneals with increasing noise as prescribed
    sigmas = [1e-3, 1e-2, 3e-2]
    # beta schedules for breaking: moderate then high
    for ti, Xt in enumerate(targets):
        if time.time() > deadline - 25.0:
            break
        cur = Xt.copy()
        # evaluate; if much worse than best, still try but limited
        for si, sigma in enumerate(sigmas):
            if time.time() > deadline - 22.0:
                break
            r3 = np.random.default_rng(2000 + ti * 10 + si)
            cur_n = _tangent_noise(cur, sigma, r3)
            # anneal free: beta increasing
            if si == 0:
                b2 = [60.0, 150.0, 300.0]; it2 = [400, 400, 350]; lr2 = [0.01, 0.008, 0.005]
            elif si == 1:
                b2 = [80.0, 200.0, 350.0]; it2 = [400, 450, 350]; lr2 = [0.01, 0.007, 0.004]
            else:
                b2 = [50.0, 150.0, 400.0]; it2 = [450, 450, 400]; lr2 = [0.012, 0.007, 0.004]
            try:
                cur_n = _adam_free(cur_n, b2, it2, lr2, min(deadline - 18.0, time.time() + 12.0))
            except Exception:
                pass
            v = _max_cos(cur_n)
            if v < best_val - 1e-12:
                best_val = v; best = cur_n.copy()
            # greedy accept if improves current trajectory or close
            cur_v = _max_cos(cur)
            if v < cur_v:
                cur = cur_n
            else:
                # keep noisy version anyway for next sigma to diversify once
                if si == 1:
                    cur = cur_n  # allow jump
                # else keep cur
                pass
        # end sigmas

    # final LP polish of incumbent with remaining time
    try:
        remain = deadline - time.time()
        if remain > 4.0:
            polished = _lp_polish(best, deadline - 1.0)
            pv = _max_cos(polished)
            if pv < best_val:
                best_val = pv; best = polished
    except Exception:
        pass
    best = _norm(best)
    return best
