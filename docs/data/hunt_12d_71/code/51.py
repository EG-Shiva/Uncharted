import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm_rows(X):
    X = np.asarray(X, dtype=float)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _row_max_cnt(X, mglob=None, slack=1e-6):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    rm = G.max(axis=1)
    if mglob is None:
        mglob = float(rm.max())
    cnt = (G > (mglob - slack)).sum(axis=1)
    return rm, cnt, G

def _fingerprint(X, k=100, scale=1e6):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    kk = min(k, c.size)
    idx = np.argpartition(-c, kk - 1)[:kk]
    top = np.sort(c[idx])
    return tuple((np.round(top * scale)).astype(int).tolist())

def _adam_subset_quench(X_init, movable, steps, T0, T1, lr, mu_note=None, rng=None, noise=0.0):
    X = np.array(X_init, dtype=float, copy=True)
    n, d = X.shape
    mov = np.asarray(movable, dtype=int)
    frozen = np.ones(n, dtype=bool)
    frozen[mov] = False
    M = np.zeros_like(X)
    V = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lt0, lt1 = np.log(T0), np.log(T1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(steps):
        frac = s / max(1, steps - 1)
        T = float(np.exp(lt0 + (lt1 - lt0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -1e100)
        m = float(G.max())
        Q = (G - m) / T
        np.clip(Q, -50, 0, out=Q)
        E = np.exp(Q)
        np.fill_diagonal(E, 0.0)
        rs = E.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Grad = (E @ X) / rs
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        if len(mov) < n:
            Gt[frozen] = 0.0
        # clip extreme rows
        gn = np.linalg.norm(Gt, axis=1, keepdims=True) + 1e-12
        bad = gn[:, 0] > 2.0
        if np.any(bad):
            Gt[bad] *= (2.0 / gn[bad])
        if rng is not None and noise > 0:
            N = rng.standard_normal((n, d)) * noise
            N -= X * np.sum(N * X, axis=1, keepdims=True)
            if len(mov) < n:
                N[frozen] = 0.0
            Gt = Gt + N
        M = b1 * M + (1 - b1) * Gt
        V = b2 * V + (1 - b2) * (Gt * Gt)
        bc1 = 1 - b1 ** (s + 1)
        bc2 = 1 - b2 ** (s + 1)
        Mh = M / bc1
        Vh = V / bc2
        Upd = Mh / (np.sqrt(Vh) + eps)
        # limit step per row
        un = np.linalg.norm(Upd, axis=1, keepdims=True) + 1e-12
        lim = 3.0
        over = un[:, 0] > lim
        if np.any(over):
            Upd[over] *= (lim / un[over])
        if len(mov) < n:
            Upd[frozen] = 0.0
        X[mov] = X[mov] - lr * Upd[mov]
        X[mov] = _norm_rows(X[mov])
        if (s % 80 == 0) or (s == steps - 1):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_m:
                best_m = vc
                best_X = X.copy()
    return best_X, best_m

def _single_refine(s, R, steps=150, T=0.004, lr=0.12, mu=0.8):
    s = np.asarray(s, dtype=float).copy()
    ss = float(np.linalg.norm(s))
    if not np.isfinite(ss) or ss == 0:
        s = np.zeros_like(s); s[0] = 1.0
    else:
        s = s / ss
    v = np.zeros_like(s)
    for _ in range(steps):
        g = R @ s
        m = float(g.max())
        e = np.exp(np.clip((g - m) / T, -50, 0))
        den = float(e.sum()) + 1e-30
        Grad = (R.T @ e) / den
        Gt = Grad - float(Grad @ s) * s
        v = mu * v + Gt
        s = s - lr * v
        nn = float(np.linalg.norm(s))
        if not np.isfinite(nn) or nn == 0:
            break
        s = s / nn
    return s

def _find_hole(R, rng, n_sample=40000, n_cand=3):
    d = R.shape[1]
    Z = rng.standard_normal(size=(n_sample, d))
    Z = _norm_rows(Z)
    S = Z @ R.T
    scores = S.max(axis=1)
    idx = np.argpartition(scores, n_cand)[:n_cand]
    best_s = None
    best_v = 1e100
    for k in idx:
        s1 = _single_refine(Z[k], R, steps=140, T=0.005, lr=0.12)
        s1 = _single_refine(s1, R, steps=140, T=0.0015, lr=0.07)
        v = float((R @ s1).max())
        if v < best_v:
            best_v = v
            best_s = s1
    nrm = float(np.linalg.norm(best_s))
    if not np.isfinite(nrm) or nrm == 0:
        best_s = Z[idx[0]]
        nrm = float(np.linalg.norm(best_s))
    return best_s / (nrm + 1e-12), best_v

def _lp_polish_wide(x, seconds, deadline=None, r0=3e-3):
    x = _norm_rows(np.array(x, dtype=float, copy=True))
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = r0
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if deadline is not None and time.time() > deadline:
            break
        if radius < 1e-14:
            break
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        band = max(30.0 * radius, 3e-4)
        sel = cmat > m - band
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        if p > 2000:
            order = np.argsort(-cmat[sel])[:2000]
            I, J = I[order], J[order]
            csel = cmat[sel][order]
            p = 2000
        else:
            csel = cmat[sel]
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.concatenate([I[:, None] * d + np.arange(d),
                             J[:, None] * d + np.arange(d),
                             np.full((p, 1), n * d)], 1).ravel()
        vv = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr() if False else coo_matrix((vv, (rows, cc)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv); obj[-1] = 1.0
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
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD_END = t0 + 173.0
    FINAL_POLISH = 15.0
    HOP_END = HARD_END - FINAL_POLISH - 1.5
    rng = np.random.default_rng(20261101)
    n, d = 71, 12
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (71, 12) and np.all(np.isfinite(a)):
                a = _norm_rows(a)
                cands.append((_max_cos(a), a))
        except Exception:
            pass
    if not cands:
        a = _norm_rows(rng.standard_normal((71, 12)))
        cands.append((_max_cos(a), a))
    cands.sort(key=lambda t: t[0])
    best_score, best_X = cands[0][0], cands[0][1].copy()
    cur_X = best_X.copy()
    cur_score = best_score
    tabu = set()
    tabu.add(_fingerprint(best_X))
    iu_all = np.triu_indices(n, 1)
    it = 0
    while time.time() < HOP_END:
        it += 1
        if time.time() > HOP_END - 2.0:
            break
        # base choice: mostly current, sometimes best
        base = cur_X if rng.random() < 0.65 else best_X
        base = _norm_rows(base)
        Gfull = base @ base.T
        np.fill_diagonal(Gfull, -2.0)
        mglob = float(Gfull.max())
        rm, cnt, G = _row_max_cnt(base, mglob=mglob, slack=1e-6)
        flat = int(np.argmax(Gfull[iu_all]))
        wi, wj = int(iu_all[0][flat]), int(iu_all[1][flat])
        # coordinated deletion size 8-18
        K = int(rng.integers(8, 19))
        # one endpoint of max pair
        endpoint = wi if rng.random() < 0.5 else wj
        # tie-break: prefer endpoint with larger row-max among the two occasionally
        if rng.random() < 0.4:
            endpoint = wi if rm[wi] >= rm[wj] else wj
        # worst score: high row-max, low degree -> priority
        # score = rowmax - 1e-7*cnt (higher worse)
        score_w = rm - 1e-7 * cnt.astype(float)
        order_worst = np.argsort(-score_w)
        # pick cluster seed among top-15 worst excluding endpoint
        pool = [i for i in order_worst if i != endpoint][:15]
        seed = int(pool[int(rng.integers(0, len(pool)))])
        # spatial cluster: nearest neighbours of seed by cosine (rigidity graph)
        neigh_order = np.argsort(-G[seed])
        neigh_order = [int(v) for v in neigh_order if v != endpoint and v != seed]
        Dset = [endpoint, seed]
        for v in neigh_order:
            if len(Dset) >= K:
                break
            Dset.append(v)
        # if still short (should not), fill with worst
        for v in order_worst:
            if len(Dset) >= K:
                break
            v = int(v)
            if v not in Dset:
                Dset.append(v)
        Dset = sorted(list(set(Dset)))[:K]
        keep_mask = np.ones(n, dtype=bool)
        keep_mask[Dset] = False
        keep_idx = np.where(keep_mask)[0]
        # Y starts as base copy
        Y = base.copy()
        # core matrix for hole search
        # greedy reinsert one at a time
        inserted_sofar = []
        # adaptive sample size
        time_left = HOP_END - time.time()
        if time_left < 20:
            nsamp = 15000
        elif time_left < 60:
            nsamp = 30000
        else:
            nsamp = 50000
        success = True
        for tpos, didx in enumerate(Dset):
            # current core rows: keep + already inserted
            cur_rows = []
            cur_rows.append(Y[keep_mask])
            if inserted_sofar:
                cur_rows.append(Y[np.array(inserted_sofar, dtype=int)])
            Rcur = np.vstack(cur_rows)
            try:
                hole, _hv = _find_hole(Rcur, rng, n_sample=nsamp, n_cand=3)
            except Exception:
                success = False
                break
            Y[didx] = hole
            inserted_sofar.append(didx)
            # movable = inserted + 10 nearest neighbours of new hole among keep
            dots = Y[keep_idx] @ hole
            nn_order = np.argsort(-dots)[:10]
            neigh_keep = keep_idx[nn_order]
            movable = np.unique(np.concatenate([np.array(inserted_sofar, dtype=int), neigh_keep]))
            T0_ = float(rng.uniform(0.004, 0.010))
            T1_ = float(rng.uniform(0.0001, 0.0008))
            lr_ = float(rng.uniform(0.004, 0.014))
            steps_ = 220 if tpos < len(Dset) - 1 else 320
            try:
                Yq, _sq = _adam_subset_quench(Y, movable, steps=steps_, T0=T0_, T1=T1_, lr=lr_)
                Y = Yq
            except Exception:
                pass
            if time.time() > HOP_END - 1.0:
                success = False
                break
        if not success:
            continue
        # unfreeze short quench (full)
        try:
            T0_ = float(rng.uniform(0.003, 0.007))
            T1_ = float(rng.uniform(0.00008, 0.0004))
            lr_ = float(rng.uniform(0.004, 0.010))
            Y, _ = _adam_subset_quench(Y, np.arange(n), steps=320, T0=T0_, T1=T1_, lr=lr_)
        except Exception:
            pass
        # wide-band LP
        try:
            Y = _lp_polish_wide(Y, seconds=1.4, deadline=HOP_END - 0.5, r0=3e-3)
        except Exception:
            pass
        try:
            sc = _max_cos(Y)
        except Exception:
            continue
        fp = _fingerprint(Y)
        if fp in tabu:
            # force topology change: still allow if strictly better than best
            if sc >= best_score:
                # sideways tabu move rejected, jitter cur slightly to escape?
                continue
        tabu.add(fp)
        if len(tabu) > 800:
            # keep bounded: drop arbitrary (convert to list pop) - just clear oldest by rebuilding
            # sets unordered; clear half
            tabu = set(list(tabu)[-400:])
        # accept sideways within 2e-7
        if sc < cur_score + 2e-7:
            cur_X = Y.copy()
            cur_score = sc
        if sc < best_score - 1e-12:
            best_score = sc
            best_X = Y.copy()
        elif sc < best_score and fp not in tabu:
            best_score = sc
            best_X = Y.copy()
    # final polish of best
    try:
        Y = best_X.copy()
        Y, _ = _adam_subset_quench(Y, np.arange(n), steps=500, T0=0.005, T1=0.0001, lr=0.008)
        Y = _lp_polish_wide(Y, seconds=min(9.0, max(2.0, HARD_END - time.time() - 1.0)), deadline=HARD_END - 0.8, r0=2e-3)
        sc = _max_cos(Y)
        if sc < best_score:
            best_score = sc
            best_X = Y
    except Exception:
        pass
    best_X = _norm_rows(best_X)
    return best_X
