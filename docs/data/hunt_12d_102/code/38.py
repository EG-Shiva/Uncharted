import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos_single(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best_val = _max_cos_single(x)
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
        if p > 2500:
            idx = np.argpartition(-c_full, 2500)[:2500]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
            p = 2500
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
        v = _max_cos_single(y)
        if v < best_val:
            best_val = v
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
            if time.time() >= deadline:
                break
        else:
            radius *= 0.5
    return best_x


def _whiten_iso(X, iters=6):
    n = X.shape[0]
    d = X.shape[1]
    target = n / d
    Y = X.copy()
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    for _ in range(iters):
        C = Y.T @ Y
        # eigh
        try:
            w, U = np.linalg.eigh(C)
        except Exception:
            break
        w = np.maximum(w, 1e-8)
        sc = np.sqrt(target / w)
        M = (U * sc) @ U.T
        Y = Y @ M
        Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return Y


def _greedy_farthest(rng, n, d, pool=2500):
    cand = rng.standard_normal((pool, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    first = int(rng.integers(0, pool))
    chosen_idx = [first]
    # maintain max dot to chosen set
    # dots_max[i] = max_{k chosen} cand[i]@cand[chosen_k]
    dots_max = cand @ cand[first]
    dots_max[first] = np.inf  # avoid repick
    for _ in range(1, n):
        nxt = int(np.argmin(dots_max))
        chosen_idx.append(nxt)
        dots_max[nxt] = np.inf
        # update
        nd = cand @ cand[nxt]
        # only for not-yet-chosen improve max
        dots_max = np.maximum(dots_max, np.where(np.isinf(dots_max), -np.inf, nd) if False else nd)
        # fix inf entries stay inf
        for c in chosen_idx:
            dots_max[c] = np.inf
    X = cand[np.array(chosen_idx)]
    return X


def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    ADAM_END = t0 + 105.0
    rng = np.random.default_rng(0)
    n, d = 102, 12
    B = 20

    # load record / best
    try:
        rec = np.load("record.npy").astype(float)
    except Exception:
        rec = None
    try:
        bst = np.load("best.npy").astype(float)
    except Exception:
        bst = rec
    if rec is None and bst is None:
        rec = rng.standard_normal((n, d))
        rec /= np.linalg.norm(rec, axis=1, keepdims=True)
    if rec is None:
        rec = bst
    if bst is None:
        bst = rec
    # ensure shapes
    def _fix(X):
        X = np.asarray(X, dtype=float)
        if X.shape != (n, d):
            tmp = rng.standard_normal((n, d))
            try:
                m0 = min(X.shape[0], n)
                m1 = min(X.shape[1], d)
                tmp[:m0, :m1] = X[:m0, :m1]
            except Exception:
                pass
            X = tmp
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        return X
    rec = _fix(rec)
    bst = _fix(bst)
    base = bst if _max_cos_single(bst) < _max_cos_single(rec) else rec

    # build diverse population (B, n, d)
    pop = np.empty((B, n, d))
    # counts: 5 random, 4 greedy, 5 perturbed, 6 isotropised+perturb
    iso = _whiten_iso(base, iters=8)
    # also isotropised random? whiten a random config
    rand_for_iso = rng.standard_normal((n, d))
    rand_for_iso /= np.linalg.norm(rand_for_iso, axis=1, keepdims=True)
    iso_rand = _whiten_iso(rand_for_iso, iters=6)

    sigmas = [0.01, 0.025, 0.05, 0.09, 0.15]
    for b in range(B):
        if b < 5:
            X = rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            pop[b] = X
        elif b < 8:
            # greedy farthest (different pool sizes / seeds via rng state)
            X = _greedy_farthest(rng, n, d, pool=2500)
            # tiny jitter to break exact pool discreteness
            X = X + 0.005 * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            pop[b] = X
        elif b < 13:
            s = sigmas[(b - 8) % len(sigmas)]
            X = base + s * rng.standard_normal((n, d))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            pop[b] = X
        else:
            # isotropised variants
            if (b % 2) == 0:
                src = iso
            else:
                src = iso_rand
            # perturb isotropised
            amp = [0.0, 0.01, 0.03, 0.06][(b - 13) % 4]
            # random global rotation to diversify (does not change score but changes Adam trajectory? actually rotation invariant, so add perturb instead)
            X = src + amp * rng.standard_normal((n, d))
            # occasional stretch then renormalise
            if b == B - 1:
                A = np.eye(d) + 0.10 * rng.standard_normal((d, d))
                X = src @ A
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            pop[b] = X

    Xp = pop  # (B,n,d)
    # batched Riemannian Adam on softmax
    STEPS = 6000
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m_ad = np.zeros_like(Xp)
    v_ad = np.zeros_like(Xp)
    Eye = np.eye(n)
    best_snap = Xp.copy()
    # initial scores
    def batch_max_cos(Xb):
        G = np.matmul(Xb, Xb.transpose(0, 2, 1))
        # kill diag
        G[:, np.arange(n), np.arange(n)] = -2.0
        return G.max(axis=(1, 2))
    cur_best_scores = batch_max_cos(Xp)

    lr0 = 0.008
    # anneal alpha 20 -> 250 exponential
    for s in range(STEPS):
        if time.time() > ADAM_END:
            break
        frac = s / max(STEPS - 1, 1)
        alpha = 20.0 * ((250.0 / 20.0) ** frac)
        lrs = lr0 * (1.0 - 0.65 * frac)
        # Gram
        G = np.matmul(Xp, Xp.transpose(0, 2, 1))
        # off-diag max for stability: subtract 3*I
        Gm = G - 3.0 * Eye[None, :, :]
        mval = Gm.max(axis=(1, 2))  # (B,)
        # weights
        # may be large; compute in place
        W = np.exp(alpha * (G - mval[:, None, None]))
        W[:, np.arange(n), np.arange(n)] = 0.0
        S = W.sum(axis=(1, 2))
        S = np.maximum(S, 1e-30)
        Grad = np.matmul(W, Xp) / S[:, None, None]
        dot = np.sum(Grad * Xp, axis=2, keepdims=True)
        Gt = Grad - dot * Xp
        m_ad = beta1 * m_ad + (1.0 - beta1) * Gt
        v_ad = beta2 * v_ad + (1.0 - beta2) * (Gt * Gt)
        bc1 = 1.0 - beta1 ** (s + 1)
        bc2 = 1.0 - beta2 ** (s + 1)
        mhat = m_ad / bc1
        vhat = v_ad / bc2
        step = lrs * mhat / (np.sqrt(vhat) + eps)
        # tangent projection of step
        sdot = np.sum(step * Xp, axis=2, keepdims=True)
        step = step - sdot * Xp
        Xp = Xp - step
        Xp /= np.linalg.norm(Xp, axis=2, keepdims=True)
        # periodic elite tracking
        if (s % 250 == 249) or (s == STEPS - 1):
            sc = batch_max_cos(Xp)
            improved = sc < cur_best_scores
            if np.any(improved):
                best_snap[improved] = Xp[improved]
                cur_best_scores[improved] = sc[improved]

    # final scores: combine final Xp and best_snap
    cand_list = []
    fin_scores = batch_max_cos(Xp)
    for b in range(B):
        cand_list.append((float(fin_scores[b]), Xp[b].copy()))
        cand_list.append((float(cur_best_scores[b]), best_snap[b].copy()))
    # also add base record
    cand_list.append((_max_cos_single(base), base.copy()))
    # sort by score
    cand_list.sort(key=lambda t: t[0])
    # permutation-invariant descriptor: sorted upper-tri cosines
    iu_r, iu_c = np.triu_indices(n, 1)
    descs = []
    for sc, Xc in cand_list:
        G = Xc @ Xc.T
        c = G[iu_r, iu_c]
        c_sorted = np.sort(c)[::-1]  # descending
        # downsample to 512 for speed? keep full but stride
        descs.append(c_sorted[::8])  # ~644 length
    # greedy distinct selection
    elites = []
    elites_desc = []
    thresh = 5e-4
    for idx, (sc, Xc) in enumerate(cand_list):
        if len(elites) >= 6:
            break
        dd = descs[idx]
        ok = True
        for ed in elites_desc:
            dist = float(np.mean(np.abs(dd - ed)))
            if dist < thresh:
                ok = False
                break
        if ok or len(elites) == 0:
            elites.append((sc, Xc))
            elites_desc.append(dd)
    # if too few, relax
    if len(elites) < 3:
        elites = [(sc, Xc) for sc, Xc in cand_list[:3]]
    # take top 3 distinct for LP polish
    top3 = elites[:3]
    # LP polish each up to 8 sec, respecting hard end
    polished = []
    for k, (sc, Xc) in enumerate(top3):
        remain = HARD_END - time.time()
        if remain < 3.0:
            polished.append((sc, Xc))
            continue
        budget = min(8.0, remain - 1.0)
        if budget < 1.0:
            polished.append((sc, Xc))
            continue
        Xp_pol = _lp_polish(Xc, time.time() + budget)
        v = _max_cos_single(Xp_pol)
        polished.append((v, Xp_pol))
    # also consider unpolished candidates in case LP hurt (it shouldn't)
    all_cands = polished + cand_list[:5]
    all_cands.sort(key=lambda t: t[0])
    bestX = all_cands[0][1]
    bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
