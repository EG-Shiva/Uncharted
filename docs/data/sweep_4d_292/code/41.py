import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _contact_mask(x, tol=1e-7):
    n = x.shape[0]
    G = x @ x.T
    iu = np.triu_indices(n, 1)
    c = G[iu]
    m = c.max()
    return (c > m - tol), m, iu

def _degrees(x, tol=1e-7):
    msk, m, iu = _contact_mask(x, tol)
    n = x.shape[0]
    deg = np.zeros(n, dtype=int)
    I = iu[0][msk]; J = iu[1][msk]
    np.add.at(deg, I, 1)
    np.add.at(deg, J, 1)
    return deg, msk, m

def _anneal_full(x, steps, beta0, beta1, lr0, lr1, mu):
    n, d = x.shape
    x = _norm(x)
    vel = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    ll = np.log(lr1 / lr0) if abs(lr1 - lr0) > 0 else 0.0
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * np.exp(ll * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        r = W.sum(axis=1, keepdims=True)
        np.maximum(r, 1e-30, out=r)
        A = (W @ x) / r
        adot = np.sum(A * x, axis=1, keepdims=True)
        At = A - adot * x
        vel = mu * vel + (1.0 - mu) * At
        bias = 1.0 - mu ** (s + 1)
        vh = vel / max(bias, 1e-12)
        x = x - lr * vh
        x = _norm(x)
    return x

def _anneal_mask(x, active, steps, beta0, beta1, lr0, lr1, mu):
    x = _norm(x)
    n, d = x.shape
    act = np.asarray(active)
    if len(act) == 0:
        return x
    vel = np.zeros_like(x)
    lb = np.log(beta1 / beta0) if beta1 > beta0 else 0.0
    ll = np.log(lr1 / lr0) if abs(lr1 - lr0) > 0 else 0.0
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = beta0 * np.exp(lb * frac)
        lr = lr0 * np.exp(ll * frac)
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        gmax = G.max()
        W = np.exp(beta * (G - gmax))
        np.fill_diagonal(W, 0.0)
        r = W.sum(axis=1, keepdims=True)
        np.maximum(r, 1e-30, out=r)
        A = (W @ x) / r
        adot = np.sum(A * x, axis=1, keepdims=True)
        At = A - adot * x
        vel[act] = mu * vel[act] + (1.0 - mu) * At[act]
        bias = 1.0 - mu ** (s + 1)
        x[act] = x[act] - lr * (vel[act] / max(bias, 1e-12))
        xa = x[act] / np.linalg.norm(x[act], axis=1, keepdims=True)
        x[act] = xa
    return x

def _lp_polish(x, seconds, r0=8e-4):
    x = _norm(np.asarray(x, dtype=float).copy())
    n, d = x.shape
    best = _max_cos(x)
    xb = x.copy()
    if seconds <= 0:
        return xb
    start = time.time()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu0]
        m = c.max()
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if not np.any(sel):
            break
        I = iu0[0][sel]; J = iu0[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        ci = I[:, None] * d + np.arange(d)
        cj = J[:, None] * d + np.arange(d)
        cols = np.concatenate([ci, cj, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
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
            best = v; xb = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return xb

def _sample_holes(Xkeep, k, rng, n_cand=65536):
    cand = rng.standard_normal((n_cand, 4))
    cand = cand / np.linalg.norm(cand, axis=1, keepdims=True)
    best = np.empty(n_cand, dtype=float)
    cs = 8192
    for a in range(0, n_cand, cs):
        b = min(a + cs, n_cand)
        D = cand[a:b] @ Xkeep.T
        best[a:b] = D.max(axis=1)
    order = np.argsort(best)
    M = min(400, n_cand)
    top_idx = order[:M]
    C = cand[top_idx].copy()
    # refine best few hundred holes: minimise smooth max vs Xkeep
    beta = 90.0
    lr = 0.035
    for it in range(120):
        D = C @ Xkeep.T  # M x Nk
        dm = D.max(axis=1, keepdims=True)
        W = np.exp(beta * (D - dm))
        W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-30)
        Gdir = W @ Xkeep  # M x 4
        dot = np.sum(Gdir * C, axis=1, keepdims=True)
        Gt = Gdir - dot * C
        step = lr * (1.0 - it / 120.0) + 0.004
        C = C - step * Gt
        C = C / np.linalg.norm(C, axis=1, keepdims=True)
    # recompute keep-max for refined
    keep_max = np.empty(M, dtype=float)
    cs2 = 400
    for a in range(0, M, cs2):
        b = min(a + cs2, M)
        D = C[a:b] @ Xkeep.T
        keep_max[a:b] = D.max(axis=1)
    order2 = np.argsort(keep_max)
    C = C[order2]
    keep_max = keep_max[order2]
    # greedy insertion under separation target below 0.90347
    # minimise max(keep_max, mutual dots)
    selected = []
    sel_idx = []
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    remaining = list(range(M))
    # first pick best keep
    first = 0
    selected.append(Cn[first])
    sel_idx.append(first)
    used = {first}
    for _ in range(1, k):
        best_score = 1e9
        best_j = -1
        for j in range(M):
            if j in used:
                continue
            s = keep_max[j]
            cj = Cn[j]
            for si in selected:
                d = float(cj @ si)
                if d > s:
                    s = d
            if s < best_score:
                best_score = s
                best_j = j
        if best_j < 0:
            break
        selected.append(Cn[best_j])
        used.add(best_j)
    sel = np.array(selected)
    # temporary separation target check (below 0.90347): if worst > 0.90347 try to still return best found
    return sel

def solve():
    t0 = time.time()
    rng = np.random.default_rng(292292)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            z = rng.standard_normal((292, 4))
            x0 = z / np.linalg.norm(z, axis=1, keepdims=True)
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (292, 4) or not np.all(np.isfinite(x0)):
        z = rng.standard_normal((292, 4))
        x0 = z / np.linalg.norm(z, axis=1, keepdims=True)
    x0 = _norm(x0)
    try:
        r0 = np.load("record.npy")
        r0 = np.asarray(r0, dtype=float)
        if r0.shape == (292, 4) and np.all(np.isfinite(r0)):
            r0 = _norm(r0)
            if _max_cos(r0) < _max_cos(x0):
                x0 = r0
    except Exception:
        pass
    best = x0.copy()
    best_val = _max_cos(best)
    _, best_mask, _ = _contact_mask(best)
    pool = [best.copy()]
    pool_vals = [best_val]
    pool_masks = [best_mask.copy()]
    KLIST = [4, 8, 12, 16, 20]
    it = 0
    EXPLORE = 148.0
    while time.time() - t0 < EXPLORE:
        it += 1
        if time.time() - t0 > 165:
            break
        # choose base: round-robin over pool with random pick biased to best
        if len(pool) == 1 or rng.random() < 0.55:
            base = best.copy()
        else:
            bi = int(rng.integers(0, len(pool)))
            base = pool[bi].copy()
        base_val = _max_cos(base)
        deg, _, _ = _degrees(base)
        idx_all = np.arange(292)
        ratt = idx_all[deg == 0].tolist()
        low = idx_all[(deg >= 1) & (deg <= 4)].tolist()
        high = idx_all[deg >= 8].tolist()
        mid = idx_all[(deg >= 5) & (deg <= 7)].tolist()
        k = KLIST[int(rng.integers(0, len(KLIST)))]
        # build delete set mixing classes
        delete = []
        # always include rattlers first
        rng.shuffle(ratt); rng.shuffle(low); rng.shuffle(high); rng.shuffle(mid)
        if len(ratt) > 0:
            take = min(len(ratt), max(1, k // 4))
            delete += ratt[:take]
        need = k - len(delete)
        # allocate ~40% low, ~40% high, rest mid/random
        n_low = min(len(low), int(round(0.4 * k)) if need > 0 else 0)
        # ensure at least 1 low/high if available and k>=8
        if k >= 8:
            if len(low) > 0 and n_low == 0:
                n_low = 1
        delete += low[:n_low]
        need = k - len(delete)
        n_high = min(len(high), int(round(0.4 * k)) if need > 0 else 0)
        if k >= 8 and len(high) > 0 and n_high == 0 and need > 0:
            n_high = 1
        delete += high[:n_high]
        need = k - len(delete)
        if need > 0 and len(mid) > 0:
            delete += mid[:min(len(mid), need)]
            need = k - len(delete)
        if need > 0:
            remaining = [i for i in idx_all.tolist() if i not in delete]
            rng.shuffle(remaining)
            delete += remaining[:need]
        delete = np.array(delete[:k], dtype=int)
        mask = np.ones(292, dtype=bool)
        mask[delete] = False
        Xkeep = base[mask]
        # find replacements with >=65536 samples
        try:
            holes = _sample_holes(Xkeep, k, rng, n_cand=65536)
        except Exception:
            continue
        if holes.shape[0] < k:
            continue
        xt = base.copy()
        for s_i, h in zip(delete, holes):
            xt[int(s_i)] = h + 0.004 * rng.standard_normal(4)
        xt = _norm(xt)
        # initially optimise only inserted + neighbours within cosine>0.75
        try:
            Dins = xt[delete] @ xt.T  # k x 292
            neigh = set(int(d) for d in delete.tolist())
            for r in range(len(delete)):
                row = Dins[r]
                close = np.where(row > 0.75)[0].tolist()
                for c in close:
                    neigh.add(int(c))
            active = np.array(sorted(list(neigh)), dtype=int)
        except Exception:
            active = delete
        # local anneal
        xt = _anneal_mask(xt, active, steps=320, beta0=20.0, beta1=420.0, lr0=0.040, lr1=0.003, mu=0.86)
        # global annealed softmax repulsion
        xt = _anneal_full(xt, steps=420, beta0=22.0, beta1=520.0, lr0=0.038, lr1=0.0028, mu=0.87)
        xt = _anneal_full(xt, steps=140, beta0=200.0, beta1=1200.0, lr0=0.009, lr1=0.0011, mu=0.88)
        try:
            vt = _max_cos(xt)
        except Exception:
            continue
        # LP polish if promising or diverse check needed (polish cheap candidates occasionally)
        do_lp = False
        if vt < best_val + 0.002:
            do_lp = True
        if do_lp:
            remain = EXPLORE - (time.time() - t0)
            if remain < 1.0:
                do_lp = False
        if do_lp:
            xp = _lp_polish(xt, seconds=min(0.9, max(0.35, (EXPLORE - (time.time() - t0)) * 0.05)), r0=8e-4)
            try:
                vp = _max_cos(xp)
            except Exception:
                vp = vt
            if vp < vt + 1e-12:
                xt, vt = xp, vp
        # contact diff vs best
        try:
            msk_t, _, _ = _contact_mask(xt)
            diff_best = int(np.sum(msk_t != best_mask))
        except Exception:
            diff_best = 0
        if vt < best_val - 1e-12:
            best = xt.copy(); best_val = vt; best_mask = msk_t.copy()
            # add to pool, keep pool bounded
            pool.append(xt.copy()); pool_vals.append(vt); pool_masks.append(msk_t.copy())
            if len(pool) > 6:
                # evict worst non-best
                worst = int(np.argmax(pool_vals))
                # never evict best (index of min)
                if pool_vals[worst] != min(pool_vals):
                    pool.pop(worst); pool_vals.pop(worst); pool_masks.pop(worst)
                else:
                    # evict second worst
                    order = np.argsort(pool_vals)
                    pool.pop(int(order[-1])); pool_vals.pop(int(order[-1])); pool_masks.pop(int(order[-1]))
        elif diff_best >= 20 and vt < best_val + 0.004:
            # genuinely new basin: retain even without record
            # check novelty vs whole pool
            novel = True
            for pm in pool_masks:
                ncmp = min(len(pm), len(msk_t))
                if int(np.sum(pm[:ncmp] != msk_t[:ncmp])) < 20:
                    novel = False
                    break
            if novel:
                pool.append(xt.copy()); pool_vals.append(vt); pool_masks.append(msk_t.copy())
                if len(pool) > 6:
                    order = np.argsort(pool_vals)
                    # keep best always, evict worst
                    pool = [pool[i] for i in order[:6]]
                    pool_vals = [pool_vals[i] for i in order[:6]]
                    pool_masks = [pool_masks[i] for i in order[:6]]
        # occasional small shake of best to escape
        if it % 12 == 0 and len(pool) > 1:
            # intensify best slightly
            cand = best + 0.012 * rng.standard_normal((292, 4))
            cand = _norm(cand)
            cand = _anneal_full(cand, steps=300, beta0=30.0, beta1=600.0, lr0=0.028, lr1=0.002, mu=0.87)
            try:
                vc = _max_cos(cand)
            except Exception:
                vc = 1e9
            if vc < best_val - 1e-12:
                mskc, _, _ = _contact_mask(cand)
                best = cand.copy(); best_val = vc; best_mask = mskc.copy()
                pool.append(cand.copy()); pool_vals.append(vc); pool_masks.append(mskc.copy())
    # final intensify around best
    while time.time() - t0 < 162.0:
        sigma = float(rng.choice(np.array([0.008, 0.016, 0.030])))
        xt = best + sigma * rng.standard_normal((292, 4))
        xt = _norm(xt)
        xt = _anneal_full(xt, steps=380, beta0=30.0, beta1=650.0, lr0=0.028, lr1=0.002, mu=0.87)
        try:
            vt = _max_cos(xt)
        except Exception:
            break
        if vt < best_val - 1e-12:
            best = xt.copy(); best_val = vt
            try:
                bm, _, _ = _contact_mask(best)
                best_mask = bm
            except Exception:
                pass
        if time.time() - t0 > 162:
            break
    remain = min(14.0, max(0.0, 174.0 - (time.time() - t0) - 0.5))
    if remain > 0.8:
        best = _lp_polish(best, seconds=remain, r0=7e-4)
    best = _norm(best)
    return best
