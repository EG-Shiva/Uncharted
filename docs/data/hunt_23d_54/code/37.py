import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())


def _norm(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _quick_anneal(X, steps, k0, k1, lr0, lr1, w0, w1):
    n, d = X.shape
    x = X.copy()
    lk0, lk1 = np.log(k0), np.log(k1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        f = s / max(steps - 1, 1)
        k = float(np.exp(lk0 + (lk1 - lk0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        w = w0 + (w1 - w0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        sel = G > (m - w)
        W = np.zeros_like(G)
        # stable exp: G-m in [-w,0]
        W[sel] = np.exp(k * (G[sel] - m))
        np.fill_diagonal(W, 0.0)
        F = W @ x
        dot = np.sum(F * x, axis=1, keepdims=True)
        Ft = F - dot * x
        deg = np.sum(W, axis=1, keepdims=True) + 1.0
        x = x - lr * Ft / deg
        x = _norm(x)
    return x


def _lp_polish(X, seconds, t0=None, hard_deadline=1e18):
    x = _norm(X.copy())
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t0 is not None and time.time() > hard_deadline:
            break
        if radius < 1e-14:
            break
        if time.time() - start > seconds:
            break
        Gf = x @ x.T
        call = Gf[iu0]
        m = float(call.max())
        sel = call > m - max(20 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 700:
            idx = np.where(sel)[0]
            order = np.argsort(call[idx])[::-1][:700]
            nsel = np.zeros_like(sel, dtype=bool)
            nsel[idx[order]] = True
            sel = nsel
        I, J = iu0[0][sel], iu0[1][sel]
        c = call[sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
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
    return x


def _infer_blocks(X, delta=0.02):
    n = X.shape[0]
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = G > (m - delta)
    np.fill_diagonal(C, False)
    deg = C.sum(axis=1)
    order = np.argsort(-deg)
    color = np.full(n, -1, dtype=int)
    ncols = 0
    for v in order:
        nbr = np.where(C[v])[0]
        used = set()
        for u in nbr:
            if color[u] != -1:
                used.add(int(color[u]))
        c = 0
        while c in used:
            c += 1
        color[v] = c
        if c + 1 > ncols:
            ncols = c + 1
    blocks = []
    for c in range(ncols):
        blocks.append(np.where(color == c)[0])
    return blocks


def _contact_sig(X, delta=0.01):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    C = G > (m - delta)
    np.fill_diagonal(C, False)
    cnt = C.sum(axis=1)
    return tuple(sorted(cnt.tolist()))


def _sig_dist(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def _is_diverse(sig, sigs, thresh=8):
    if not sigs:
        return True
    return min(_sig_dist(sig, s) for s in sigs) > thresh


def _procrustes_align(B, A):
    # find R minimising ||A - B R||
    M = B.T @ A
    try:
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        R = U @ Vt
    except Exception:
        R = np.eye(B.shape[1])
    return B @ R


def _greedy_select(U, k, rng):
    # U: (N,d) normalized, select k minimising max cosine greedily
    N = U.shape[0]
    D = U @ U.T
    # start from minimal pair
    # mask diagonal
    np.fill_diagonal(D, 2.0)
    flat = int(np.argmin(D))
    i0, j0 = flat // N, flat % N
    selected = [int(i0), int(j0)]
    selmask = np.zeros(N, dtype=bool)
    selmask[i0] = True
    selmask[j0] = True
    # iteratively add best hole
    # keep D with diag 2 so not chosen weirdly; recompute max to selected
    # use original dots without diag hack
    D2 = U @ U.T
    for _ in range(k - 2):
        # max dot to selected set
        md = D2[:, selected].max(axis=1)
        md[selmask] = 2.0
        nxt = int(np.argmin(md))
        selected.append(nxt)
        selmask[nxt] = True
        if len(selected) >= k:
            break
    # if k <2 edge
    selected = selected[:k]
    return U[np.array(selected)].copy()


def _random_plane(d, rng):
    a = rng.randn(d)
    b = rng.randn(d)
    a /= (np.linalg.norm(a) + 1e-18)
    b = b - float(b @ a) * a
    n = float(np.linalg.norm(b))
    if n < 1e-12:
        b = rng.randn(d)
        b = b - float(b @ a) * a
        n = float(np.linalg.norm(b)) + 1e-18
    b /= n
    return a, b


def _rotate_indices(X, idx, u, v, theta):
    if len(idx) == 0:
        return X
    c, s = float(np.cos(theta)), float(np.sin(theta))
    B = X[idx]
    a = B @ u
    b = B @ v
    Bperp = B - np.outer(a, u) - np.outer(b, v)
    an = a * c - b * s
    bn = a * s + b * c
    Bn = Bperp + np.outer(an, u) + np.outer(bn, v)
    # rotation preserves norm; renormalize for safety
    Bn /= (np.linalg.norm(Bn, axis=1, keepdims=True) + 1e-18)
    X = X.copy()
    X[idx] = Bn
    return X


def _best_void(kept, rng, n_try=1500):
    d = kept.shape[1]
    C = rng.randn(n_try, d)
    C /= (np.linalg.norm(C, axis=1, keepdims=True) + 1e-18)
    dots = C @ kept.T
    m = dots.max(axis=1)
    return C[int(np.argmin(m))]


def solve():
    t0 = time.time()
    HARD = t0 + 168.0
    STOP = t0 + 130.0
    rng = np.random.RandomState(0)
    n, d = 54, 23

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = rng.randn(n, d)
    xb = _norm(xb)
    if xb.shape != (n, d):
        tmp = rng.randn(n, d)
        xb = _norm(tmp)
    try:
        xr = np.load("record.npy").astype(float)
        xr = _norm(xr)
        if xr.shape == (n, d) and _max_cos(xr) < _max_cos(xb):
            xb = xr.copy()
    except Exception:
        pass

    best = _max_cos(xb)
    x_best = xb.copy()

    pop = [x_best.copy()]
    sc = [best]
    sigs = [_contact_sig(x_best)]

    # small diverse init with cheap anneal
    for _ in range(3):
        if time.time() > STOP - 5:
            break
        try:
            y = x_best + rng.randn(n, d) * 0.012
            y = _norm(y)
            y = _quick_anneal(y, steps=150, k0=80, k1=2000, lr0=0.012, lr1=0.0008, w0=0.05, w1=0.005)
            v = _max_cos(y)
            if v < best + 0.01:
                pop.append(y)
                sc.append(v)
                sigs.append(_contact_sig(y))
                if v < best:
                    best, x_best = v, y.copy()
        except Exception:
            continue

    # keep pop <=5 best-diverse
    def _trim():
        global pop, sc, sigs
        if len(pop) <= 5:
            return
        order = np.argsort(sc)
        keep = order[:5].tolist()
        pop = [pop[i] for i in keep]
        sc = [sc[i] for i in keep]
        sigs = [sigs[i] for i in keep]

    # trim init
    if len(pop) > 5:
        _trim()

    it = 0
    while time.time() < STOP:
        it += 1
        if HARD - time.time() < 25:
            break
        # pick parents: best + random diverse
        if len(pop) == 1:
            a, b = 0, 0
        else:
            # best index
            bi = int(np.argmin(sc))
            a = bi
            # second parent random among rest
            others = [i for i in range(len(pop)) if i != bi]
            b = others[rng.randint(len(others))]
            # occasionally two random
            if rng.rand() < 0.3:
                a = rng.randint(len(pop))
                b = rng.randint(len(pop))
        p1 = pop[a]
        p2 = pop[b]
        try:
            p2a = _procrustes_align(_norm(p2), _norm(p1))
            p2a = _norm(p2a)
        except Exception:
            p2a = p2.copy()

        children = []
        # make 3 children
        for ci in range(3):
            if time.time() > STOP:
                break
            try:
                U = np.vstack([p1, p2a])
                U = _norm(U)
                ch = _greedy_select(U, n, rng)
                ch = _norm(ch)
                blocks = _infer_blocks(ch, delta=0.02)
                # choose mutation
                r = rng.rand()
                if r < 0.4:
                    # single block rigid rotation 0.05-0.5 rad
                    bi2 = rng.randint(len(blocks))
                    idx = blocks[bi2]
                    if len(idx) <= 1 or len(idx) >= n - 1:
                        # fallback: random subset of 9
                        idx = rng.choice(n, size=9, replace=False)
                    u, v = _random_plane(d, rng)
                    th = float(rng.uniform(0.05, 0.5)) * (1 if rng.rand() < 0.5 else -1)
                    ch = _rotate_indices(ch, np.asarray(idx), u, v, th)
                elif r < 0.7:
                    # two blocks opposite
                    if len(blocks) >= 2:
                        i1, i2 = rng.choice(len(blocks), size=2, replace=False)
                        u, v = _random_plane(d, rng)
                        th = float(rng.uniform(0.05, 0.4))
                        ch = _rotate_indices(ch, np.asarray(blocks[i1]), u, v, th)
                        ch = _rotate_indices(ch, np.asarray(blocks[i2]), u, v, -th)
                    else:
                        u, v = _random_plane(d, rng)
                        idx = rng.choice(n, size=9, replace=False)
                        th = float(rng.uniform(0.05, 0.5))
                        ch = _rotate_indices(ch, np.asarray(idx), u, v, th)
                else:
                    # delete + greedily reinsert whole block into holes
                    bi2 = rng.randint(len(blocks))
                    idx = np.asarray(blocks[bi2])
                    if len(idx) == 0 or len(idx) > 18:
                        idx = rng.choice(n, size=9, replace=False)
                        idx = np.asarray(idx)
                    keep_mask = np.ones(n, dtype=bool)
                    keep_mask[idx] = False
                    kept = ch[keep_mask]
                    newpts = []
                    cur = kept.copy()
                    for _k in range(len(idx)):
                        nv = _best_void(cur, rng, n_try=1200)
                        # small jitter
                        nv = nv + rng.randn(d) * 0.01
                        nv /= (np.linalg.norm(nv) + 1e-18)
                        newpts.append(nv)
                        cur = np.vstack([cur, nv[None, :]])
                    ch2 = ch.copy()
                    ch2[idx] = np.vstack(newpts)
                    ch = _norm(ch2)
                # prescreen cheap softmax/Adam (few hundred steps)
                ch = _quick_anneal(ch, steps=200, k0=70, k1=2200, lr0=0.013, lr1=0.0007, w0=0.055, w1=0.005)
                children.append(ch)
            except Exception:
                continue
        if not children:
            continue
        try:
            vals = [_max_cos(c) for c in children]
        except Exception:
            continue
        order = np.argsort(vals)
        top = order[0]
        vtop = float(vals[top])
        ctop = children[top]
        sig_top = _contact_sig(ctop)
        diverse = _is_diverse(sig_top, sigs, thresh=8)
        # LP-polish only best one if promising: comparable or better
        should_lp = (vtop < best + 0.004 and (diverse or vtop < min(sc) + 0.0005))
        if should_lp:
            if time.time() > STOP - 3:
                # no time for LP, just consider prescreen
                if vtop < best - 1e-12:
                    best, x_best = vtop, _norm(ctop).copy()
                    pop.append(x_best.copy()); sc.append(best); sigs.append(sig_top)
                    _trim()
                elif vtop < best + 0.002 and diverse:
                    pop.append(_norm(ctop).copy()); sc.append(vtop); sigs.append(sig_top)
                    _trim()
            else:
                try:
                    lp_sec = min(2.5, max(0.5, STOP - (time.time() - t0) if False else 2.5))
                    # respect hard deadline
                    lp_sec = min(lp_sec, max(0.5, HARD - time.time() - 22.0))
                    if lp_sec < 0.4:
                        continue
                    polished = _lp_polish(ctop, seconds=lp_sec, t0=t0, hard_deadline=HARD - 20.0)
                    vp = _max_cos(polished)
                    sig_p = _contact_sig(polished)
                    div_p = _is_diverse(sig_p, sigs, thresh=8)
                    if vp < best - 1e-12:
                        best, x_best = vp, _norm(polished).copy()
                        pop.append(x_best.copy()); sc.append(best); sigs.append(sig_p)
                        _trim()
                    elif vp < best + 0.002 and div_p:
                        pop.append(_norm(polished).copy()); sc.append(vp); sigs.append(sig_p)
                        _trim()
                    else:
                        # even if not diverse, keep if strictly among best
                        if vp < min(sc) - 1e-12 and len(pop) < 6:
                            pop.append(_norm(polished).copy()); sc.append(vp); sigs.append(sig_p)
                            _trim()
                except Exception:
                    continue
        else:
            # without LP, still keep if it improves incumbent prescreen
            if vtop < best - 1e-9:
                best, x_best = vtop, _norm(ctop).copy()
                pop.append(x_best.copy()); sc.append(best); sigs.append(sig_top)
                _trim()
        # occasional second child diverse keep without LP (cheap diversity)
        if len(order) > 1 and time.time() < STOP - 2:
            try:
                s2 = order[1]
                v2 = float(vals[s2])
                sig2 = _contact_sig(children[s2])
                if v2 < best + 0.002 and _is_diverse(sig2, sigs, thresh=12) and len(pop) < 5:
                    pop.append(_norm(children[s2]).copy()); sc.append(v2); sigs.append(sig2)
            except Exception:
                pass

    # final LP polish of incumbent with remaining time
    try:
        remain = HARD - time.time() - 2.0
        if remain > 1.0:
            fsec = min(remain, 20.0)
            x_best = _lp_polish(x_best, seconds=fsec, t0=t0, hard_deadline=HARD - 1.0)
            best = _max_cos(x_best)
    except Exception:
        pass
    x_best = _norm(x_best)
    return x_best
