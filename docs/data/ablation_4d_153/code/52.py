import time
import numpy as np

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n

def _max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _degrees_rowmax(x, tol=1e-6):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    deg = np.sum(G > (m - tol), axis=1)
    rowmax = np.max(G, axis=1)
    return deg, rowmax, m, G

def _hash_key(x, tol=5e-7):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    deg = np.sum(G > (m - tol), axis=1)
    return (tuple(sorted(deg.tolist())), round(float(m) * 1e7))

def _smooth_opt_free(x, free_idx, steps, tau0, tau1, lr0, lr1, mom=0.9):
    n, d = x.shape
    vel = np.zeros_like(x)
    lt0, lt1 = np.log(tau0), np.log(tau1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    fmask = np.zeros(n, dtype=bool)
    fmask[np.asarray(free_idx)] = True
    for t in range(steps):
        f = t / max(1, steps - 1)
        tau = float(np.exp(lt0 + (lt1 - lt0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = x @ x.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        W = np.exp((G - m) / tau)
        np.fill_diagonal(W, 0.0)
        row = np.sum(W, axis=1, keepdims=True)
        row[row < 1e-12] = 1e-12
        grad = (W @ x) / row
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad = grad - dot * x
        gn = np.linalg.norm(grad, axis=1, keepdims=True)
        big = gn > 2.0
        if np.any(big):
            grad = grad * np.where(big, 2.0 / np.maximum(gn, 1e-12), 1.0)
        grad[~fmask] = 0.0
        vel = mom * vel + (1.0 - mom) * grad
        bc = 1.0 - mom ** (t + 1)
        step = vel / bc
        x = x - lr * step
        x = _norm(x)
    return x

def _lp_polish(x, deadline, cap=2500, r0=1e-3, maxit=200):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    x = _norm(x)
    n, d = x.shape
    best = _max_cos(x)
    best_x = x.copy()
    radius = r0
    iu0 = np.triu_indices(n, 1)
    it = 0
    while time.time() < deadline and radius > 1e-14 and it < maxit:
        it += 1
        G = x @ x.T
        c = G[iu0]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        idx = np.where(c > thresh)[0]
        if idx.size == 0:
            radius = min(radius * 1.5, 0.05)
            continue
        if idx.size > cap:
            order = np.argpartition(c[idx], -cap)[-cap:]
            idx = idx[order]
        I = iu0[0][idx]; J = iu0[1][idx]
        csel = c[idx]
        p = idx.size
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cI = I[:, None] * d + np.arange(d)
        cJ = J[:, None] * d + np.arange(d)
        cols = np.concatenate([cI, cJ, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
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
        y = _norm(y)
        v = _max_cos(y)
        if v < best:
            best = v; best_x = y.copy(); x = y
            radius = min(radius * 1.5, 0.05)
        else:
            # keep x at best? stay at x but shrink
            radius *= 0.5
            # if we improved best earlier, x is best; else keep x
            # ensure x tracks best
            if _max_cos(x) > best + 1e-15:
                x = best_x.copy()
    return best_x, best

def _sequential_hole(Yfrozen, S, rng, n_cand=100000, batch=10000):
    Yg = Yfrozen.copy()
    d = Yg.shape[1]
    nb = max(1, n_cand // batch)
    for _ in range(S):
        best_c = None
        best_s = np.inf
        m = Yg.shape[0]
        for _b in range(nb):
            C = rng.standard_normal(size=(batch, d))
            C = _norm(C)
            # batched dot: (batch, m)
            Sm = C @ Yg.T
            sm = Sm.max(axis=1)
            i = int(np.argmin(sm))
            vi = float(sm[i])
            if vi < best_s:
                best_s = vi
                best_c = C[i].copy()
        Yg = np.vstack([Yg, best_c.reshape(1, -1)])
    return Yg

def solve():
    t0 = time.time()
    HARD_END = t0 + 174.0
    FINAL_RESERVE = 26.0
    EXPLORE_END = HARD_END - FINAL_RESERVE
    rng = np.random.default_rng(20261031)
    n, d = 153, 4
    best_x = None; best_v = 1e9
    for fname in ("best.npy", "record.npy"):
        try:
            a = np.load(fname)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _norm(a)
                if np.all(np.isfinite(a)):
                    v = _max_cos(a)
                    if v < best_v:
                        best_v = v; best_x = a.copy()
        except Exception:
            pass
    if best_x is None:
        best_x = _norm(rng.standard_normal(size=(n, d)))
        best_v = _max_cos(best_x)
    global_best_x = best_x.copy()
    global_best_v = best_v
    seen = set()
    seen.add(_hash_key(global_best_x))
    # diverse pool: list of (v, x)
    pool = [(global_best_v, global_best_x.copy())]

    def refresh_centres(bx):
        deg, rowmax, m, G = _degrees_rowmax(bx, tol=1e-6)
        # candidates with at most 4 contacts
        idx4 = np.where(deg <= 4)[0]
        if idx4.size < 4:
            # take lowest degrees
            order = np.lexsort((rowmax, deg))
            idx4 = order[:max(6, idx4.size if idx4.size>0 else 6)]
        else:
            # sort by rowmax descending? prefer high rowmax (tight) among low-degree? use descending rowmax to attack worst
            o = idx4[np.argsort(-rowmax[idx4])]
            idx4 = o[:min(8, o.size)]
        # ensure distinct and at least 4
        if idx4.size < 4:
            order = np.argsort(deg)
            idx4 = order[:6]
        return np.asarray(idx4, dtype=int), deg, rowmax, m

    centres, _, _, _ = refresh_centres(global_best_x)
    patch_sizes = [6, 7, 8, 9, 10]
    trial = 0
    # build infinite trial generator: cycle centres and sizes
    while time.time() < EXPLORE_END:
        # refresh centres periodically or after improvement
        if trial % 25 == 0:
            centres, _, _, _ = refresh_centres(global_best_x)
        # pick base: mostly global best, sometimes diverse pool
        if len(pool) > 1 and rng.random() < 0.30:
            # pick random among pool within tolerance
            k = int(rng.integers(len(pool)))
            base = pool[k][1]
        else:
            base = global_best_x
        # pick centre: round-robin + random
        ci = int(centres[int(rng.integers(len(centres)))]) if len(centres) > 0 else int(rng.integers(n))
        # need to ensure ci valid for base that may be permuted: centres refer to global_best ordering,
        # for diverse base use low-degree of that base occasionally
        # With prob 0.3 recompute centre from base itself
        if rng.random() < 0.4:
            try:
                degb, rowmaxb, _, _ = _degrees_rowmax(base, tol=1e-6)
                low = np.where(degb <= 4)[0]
                if low.size > 0:
                    # bias to higher rowmax
                    w = rowmaxb[low] - rowmaxb[low].min() + 1e-9
                    w = w / w.sum()
                    # sample
                    ci = int(low[int(np.searchsorted(np.cumsum(w), rng.random()))] if low.size>1 else low[0])
                else:
                    ci = int(np.argmin(degb))
            except Exception:
                pass
        S = patch_sizes[int(rng.integers(len(patch_sizes)))]
        # also sometimes force 6-10 per instruction; S already in range
        trial += 1
        if time.time() > EXPLORE_END - 2.0:
            break
        try:
            # build delete set: S nearest to centre (including centre) -> 6..10 deleted, 143..147 frozen
            sims = base @ base[ci % n]
            idx_sorted = np.argsort(-sims)
            del_idx = idx_sorted[:S]
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[del_idx] = False
            Yfrozen = base[keep_mask]
            # sequential maximin hole insertion with >=100k candidates
            Yg = _sequential_hole(Yfrozen, S, rng, n_cand=100000, batch=10000)
            if Yg.shape[0] != n:
                continue
            if time.time() > EXPLORE_END:
                break
            free_idx = list(range(n - S, n))
            # annealed softmax on free only, frozen as blockers
            Yg = _smooth_opt_free(Yg, free_idx, steps=320, tau0=0.014, tau1=0.0015, lr0=0.12, lr1=0.007, mom=0.9)
            if time.time() > EXPLORE_END:
                # still LP polish quickly?
                pass
            # full LP polish short
            budget = 1.1
            dl = min(time.time() + budget, EXPLORE_END + 1.0)
            if dl <= time.time():
                continue
            Yp, vp = _lp_polish(Yg, dl, cap=2200, r0=8e-4, maxit=120)
            hk = _hash_key(Yp)
            is_new = hk not in seen
            seen.add(hk)
            # keep by exact max cosine and diversity
            if vp < global_best_v - 1e-12:
                global_best_v = vp
                global_best_x = Yp.copy()
                pool.append((vp, Yp.copy()))
                # prune pool to best 12
                pool = sorted(pool, key=lambda z: z[0])[:12]
                centres, _, _, _ = refresh_centres(global_best_x)
            else:
                # add to pool if diverse and competitive (within 2e-3) and new hash
                if is_new and vp < global_best_v + 2e-3:
                    # avoid pool blowup: keep best diverse
                    # check hash not already in pool (approx via key)
                    pool.append((vp, Yp.copy()))
                    pool = sorted(pool, key=lambda z: z[0])[:12]
                    # keep seen bounded
                    if len(seen) > 5000:
                        # keep recent: reset to pool hashes
                        seen = set([_hash_key(p[1]) for p in pool])
        except Exception:
            continue
    # final full polish
    try:
        dl = HARD_END - 1.0
        if time.time() < dl:
            global_best_x, global_best_v = _lp_polish(global_best_x, dl, cap=3000, r0=1e-3, maxit=300)
    except Exception:
        pass
    global_best_x = _norm(global_best_x)
    return global_best_x
