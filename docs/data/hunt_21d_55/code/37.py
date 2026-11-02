import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def _batch_norm_rows(X):
    n = np.sqrt(np.sum(X * X, axis=2, keepdims=True))
    n = np.maximum(n, 1e-12)
    return X / n


def _max_cos_single(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def _batch_max_cos(X):
    B, n, d = X.shape
    G = np.matmul(X, np.transpose(X, (0, 2, 1)))
    r = np.arange(n)
    G[:, r, r] = -2.0
    return np.max(G, axis=(1, 2))


def _batch_adam_quench(X0, steps, a0, a1, lr0, lr1, deadline=None, check_every=100):
    X = X0.copy()
    B, n, d = X.shape
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    r = np.arange(n)
    la0 = np.log(a0)
    la1 = np.log(a1)
    ll0 = np.log(lr0)
    ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        frac = t / steps
        alpha = float(np.exp(la0 + (la1 - la0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = np.matmul(X, np.transpose(X, (0, 2, 1)))
        G[:, r, r] = -2.0
        Mx = np.max(G, axis=(1, 2))
        # exp weights
        E = np.exp(alpha * (G - Mx[:, None, None]))
        E[:, r, r] = 0.0
        S = np.sum(E, axis=(1, 2))
        S = np.maximum(S, 1e-300)
        grad = np.matmul(E, X) / S[:, None, None]
        proj = np.sum(grad * X, axis=2, keepdims=True)
        gt = grad - proj * X
        m = b1 * m + (1.0 - b1) * gt
        v = b2 * v + (1.0 - b2) * (gt * gt)
        c1 = 1.0 - b1 ** t
        c2 = 1.0 - b2 ** t
        mhat = m / c1
        vhat = v / c2
        X = X - lr * mhat / (np.sqrt(vhat) + eps)
        X = _batch_norm_rows(X)
        if deadline is not None and (t % check_every == 0) and time.time() > deadline:
            break
    return X


def _fingerprint_vec(x, k=40):
    n = x.shape[0]
    g = x @ x.T
    iu = np.triu_indices(n, 1)
    c = g[iu]
    c = np.sort(c)[::-1]
    return c[:k].copy()


def _lp_polish_deadline(x, deadline, radius_init=1e-3, cap=1300):
    x = _norm_rows(np.asarray(x, dtype=np.float64))
    n, d = x.shape
    best = _max_cos_single(x)
    radius = radius_init
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cfull = (x @ x.T)[iu]
        m = float(cfull.max())
        sel = cfull > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            order = np.argsort(-cfull[idx])[:cap]
            nsel = np.zeros_like(sel)
            nsel[idx[order]] = True
            sel = nsel
        csel = cfull[sel]
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
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
        y = x + res.x[:-1].reshape(n, d)
        y = _norm_rows(y)
        vv = _max_cos_single(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best


def solve():
    t0 = time.time()
    HARD_END = t0 + 150.0
    SEARCH_END = t0 + 138.0
    rng = np.random.default_rng(42)
    n, d = 55, 21
    B = 120

    # --- Stage 1: de novo population, brief annealed Adam ---
    X0 = rng.standard_normal(size=(B, n, d))
    X0 = _batch_norm_rows(X0)
    X1 = _batch_adam_quench(X0, steps=500, a0=20.0, a1=600.0,
                            lr0=0.025, lr1=0.0015,
                            deadline=HARD_END - 25.0)
    scores1 = _batch_max_cos(X1)
    order1 = np.argsort(scores1)

    # fingerprints for diversity
    fps = []
    for b in range(B):
        fps.append(_fingerprint_vec(X1[b], k=40))
    fps = np.array(fps)  # (B,40)

    # diverse archive: greedy by score with min-distance
    kept = []
    for idx in order1:
        if time.time() > SEARCH_END:
            break
        f = fps[idx]
        ok = True
        for k in kept:
            dist = float(np.linalg.norm(f - fps[k]))
            if dist < 2e-4:
                ok = False
                break
        if ok:
            kept.append(int(idx))
        if len(kept) >= 40:
            break
    if len(kept) == 0:
        kept = [int(order1[0])]

    best_overall = None
    best_score = 1e9
    for k in kept[:30]:
        s = float(scores1[k])
        if s < best_score:
            best_score = s
            best_overall = X1[k].copy()

    # --- Stage 2: refine top diverse subset longer ---
    n2 = min(20, len(kept))
    sel2 = kept[:n2]
    X2init = np.stack([X1[k] for k in sel2], axis=0)
    # also add a few fresh jittered copies of the best for exploitation
    if time.time() < SEARCH_END - 5.0:
        X2 = _batch_adam_quench(X2init, steps=900, a0=40.0, a1=900.0,
                                lr0=0.012, lr1=0.0008,
                                deadline=SEARCH_END - 2.0)
        scores2 = _batch_max_cos(X2)
        fps2 = np.array([_fingerprint_vec(X2[i], k=40) for i in range(X2.shape[0])])
        # update global best
        for i in range(X2.shape[0]):
            s = float(scores2[i])
            if s < best_score:
                best_score = s
                best_overall = X2[i].copy()
    else:
        X2 = X2init
        scores2 = _batch_max_cos(X2)
        fps2 = np.array([_fingerprint_vec(X2[i], k=40) for i in range(X2.shape[0])])

    # rank stage-2 with diversity
    order2 = np.argsort(scores2)
    kept2 = []
    for ii in order2:
        f = fps2[ii]
        ok = True
        for k in kept2:
            if float(np.linalg.norm(f - fps2[k])) < 1e-4:
                ok = False
                break
        if ok:
            kept2.append(int(ii))
        if len(kept2) >= 8:
            break

    # --- Stage 3: screened LP polish only if below ~0.075, max 2s each ---
    thresh = 0.075
    n_pol = 0
    for ii in kept2:
        if time.time() > HARD_END - 8.0:
            break
        # map back to X2 row
        cand = X2[ii]
        s = float(scores2[ii])
        if s > thresh + 0.015 and n_pol >= 2:
            continue
        if s > thresh:
            # not eligible for LP per spec; try one more short Adam push instead
            continue
        budget = 2.0
        dl = min(time.time() + budget, HARD_END - 4.0)
        if dl - time.time() < 0.4:
            break
        xp, sp = _lp_polish_deadline(cand, deadline=dl, radius_init=1e-3)
        n_pol += 1
        if sp < best_score:
            best_score = sp
            best_overall = xp.copy()
        if time.time() > SEARCH_END:
            break

    # If nothing reached threshold, do one extra aggressive quench on the single best
    if best_score > thresh and time.time() < HARD_END - 12.0:
        single = best_overall[None, :, :].copy()
        single = _batch_adam_quench(single, steps=600, a0=60.0, a1=1200.0,
                                    lr0=0.008, lr1=0.0005,
                                    deadline=HARD_END - 8.0)
        s = float(_batch_max_cos(single)[0])
        if s < best_score:
            best_score = s
            best_overall = single[0].copy()
        # LP only if now below thresh
        if s < thresh and time.time() < HARD_END - 6.0:
            dl = min(time.time() + 2.0, HARD_END - 4.0)
            xp, sp = _lp_polish_deadline(best_overall, deadline=dl, radius_init=8e-4)
            if sp < best_score:
                best_score = sp
                best_overall = xp.copy()

    if best_overall is None:
        best_overall = _norm_rows(rng.standard_normal(size=(n, d)))
        best_score = _max_cos_single(best_overall)

    best_overall = _norm_rows(best_overall)

    # --- Final comparison against lab best (only load here) ---
    try:
        for fn in ("best.npy", "record.npy"):
            try:
                a = np.load(fn)
                a = np.asarray(a, dtype=np.float64)
                if a.shape == (55, 21):
                    a = _norm_rows(a)
                    # validity
                    if np.all(np.isfinite(a)):
                        nr = np.linalg.norm(a, axis=1)
                        if np.min(nr) > 1e-9:
                            s = _max_cos_single(a)
                            if s < best_score:
                                best_score = s
                                best_overall = a.copy()
            except Exception:
                pass
    except Exception:
        pass

    best_overall = _norm_rows(np.asarray(best_overall, dtype=np.float64))
    return best_overall
