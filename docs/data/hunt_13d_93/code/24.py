import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _top_set(X, K=100):
    n = X.shape[0]
    G = X @ X.T
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    K = min(K, c.shape[0])
    idx = np.argpartition(-c, K - 1)[:K]
    s = set()
    for t in idx:
        i = int(iu0[t]); j = int(iu1[t])
        s.add(i * n + j)
    return s

def _jaccard(a, b):
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 1.0
    return inter / union

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    HOP_END = 163.0
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 93, 13

    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape != (93, 13):
                continue
            a = _norm(np.asarray(a, dtype=np.float64))
            v = _maxcos(a)
            if v < best_val:
                best_val = v
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((93, 13)))
        best_val = _maxcos(best)

    # ---------- local helpers (closure over rng, t0) ----------
    def refine_hole(v, Y, steps=200):
        # minimize max_j <v, Y_j> on sphere
        lr = 0.05
        tau = 0.008
        v = v / (np.linalg.norm(v) + 1e-18)
        for _ in range(steps):
            dots = Y @ v
            m0 = float(dots.max())
            w = np.exp((dots - m0) / tau)
            s = float(w.sum())
            if s <= 0 or not np.isfinite(s):
                break
            w /= s
            g = w @ Y
            gt = g - float(g @ v) * v
            nr = float(np.linalg.norm(gt))
            if nr < 1e-12 or not np.isfinite(nr):
                break
            gt /= nr
            v2 = v - lr * gt
            n2 = float(np.linalg.norm(v2))
            if n2 < 1e-12:
                lr *= 0.5
                continue
            v2 /= n2
            m2 = float((Y @ v2).max())
            if m2 < m0:
                v = v2
                lr = min(lr * 1.05, 0.08)
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break
        return v / (np.linalg.norm(v) + 1e-18)

    def find_hole(Y):
        # 50k random candidates, pick largest hole, then 200 steps refine
        M = 50000
        # chunk to bound memory but keep 50k total
        best_v = None
        best_w = 1e9
        CH = 10000
        for _ in range(M // CH):
            C = rng.standard_normal((CH, d))
            C = _norm(C)
            D = C @ Y.T
            worst = D.max(axis=1)
            idx = int(np.argmin(worst))
            if float(worst[idx]) < best_w:
                best_w = float(worst[idx])
                best_v = C[idx].copy()
        return refine_hole(best_v, Y, steps=200)

    def cluster_opt(X, idx, steps, tau0, tau1, lr0, lr1, mu=0.9):
        idx = np.asarray(idx, dtype=int)
        V = np.zeros((len(idx), d))
        ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
        llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
        for k in range(steps):
            frac = k / max(1, steps - 1)
            tau = tau0 * np.exp(ltau * frac)
            lr = lr0 * np.exp(llr * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            E = np.exp((G - cmax) / max(tau, 1e-12))
            np.fill_diagonal(E, 0.0)
            Gr = E @ X
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Ft = Gr - dot * X
            # only update idx rows
            g = Ft[idx]
            V = mu * V + (1.0 - mu) * g
            # re-project momentum tangent
            dotv = (V * X[idx]).sum(axis=1, keepdims=True)
            Vt = V - dotv * X[idx]
            X[idx] = X[idx] - lr * Vt
            X[idx] = _norm(X[idx])
        return X

    def anneal_full(X, steps, tau0, tau1, s0, s1, mu=0.9):
        V = np.zeros_like(X)
        ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
        ls = np.log(s1 / s0) if s0 != s1 else 0.0
        for k in range(steps):
            frac = k / max(1, steps - 1)
            tau = tau0 * np.exp(ltau * frac)
            step = s0 * np.exp(ls * frac)
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            # threshold to sparsify far pairs a bit
            E = np.exp((G - cmax) / max(tau, 1e-12))
            np.fill_diagonal(E, 0.0)
            S = E.sum(axis=1, keepdims=True) + 1e-12
            Gr = (E @ X) / S
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Ft = Gr - dot * X
            V = mu * V + (1.0 - mu) * Ft
            dotv = (V * X).sum(axis=1, keepdims=True)
            Vt = V - dotv * X
            X = X - step * Vt
            X = _norm(X)
        return X

    def greedy(X, cur_val, step_init, max_iters, tau=0.0012):
        X = X.copy()
        bestX = X.copy()
        bv = cur_val
        step = step_init
        for _ in range(max_iters):
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            cmax = float(np.max(G))
            thresh = cmax - 0.006
            E = np.exp((G - cmax) / tau)
            E[G <= thresh] = 0.0
            np.fill_diagonal(E, 0.0)
            if E.sum() == 0:
                break
            S = E.sum(axis=1, keepdims=True) + 1e-12
            Gr = (E @ X) / S
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Ft = Gr - dot * X
            Xp = _norm(X - step * Ft)
            Gp = Xp @ Xp.T
            np.fill_diagonal(Gp, -2.0)
            vp = float(np.max(Gp))
            if vp < cur_val:
                X = Xp
                cur_val = vp
                if vp < bv:
                    bv = vp
                    bestX = Xp.copy()
                step = min(step * 1.2, 0.005)
            else:
                step *= 0.5
                if step < 1e-13:
                    break
        return bestX, bv

    def short_polish(X):
        v = _maxcos(X)
        # short annealed softmax then greedy (Direction-1 style)
        X = anneal_full(X, steps=600, tau0=0.008, tau1=0.0012, s0=0.008, s1=0.0008, mu=0.9)
        v2 = _maxcos(X)
        # greedy squeeze
        bX, bv = greedy(X, v2, 8e-4, 500, tau=0.0012)
        return bX, bv

    # quick squeeze of starting point
    try:
        bX, bv = greedy(best, best_val, 8e-4, 600, tau=0.0012)
        if bv < best_val:
            best, best_val = bX, bv
    except Exception:
        pass

    best_set = _top_set(best, 100)
    pool = [(best_val, best.copy(), best_set)]
    # keep diverse pool max 8
    def pool_add(X, v):
        nonlocal pool
        s = _top_set(X, 100)
        pool.append((v, X.copy(), s))
        # keep best + at most 7 others, drop worst
        pool = sorted(pool, key=lambda t: t[0])
        # always keep best first; limit size
        if len(pool) > 8:
            pool = pool[:8]

    ks = [3, 5, 8, 12]
    trial = 0
    # run many surgeries
    while time.time() - t0 < HOP_END:
        # time guard per trial
        if time.time() - t0 > DEADLINE - 6:
            break
        # round-robin base from pool to exploit diverse basins
        bidx = trial % len(pool)
        base_val, base_X = pool[bidx][0], pool[bidx][1].copy()
        # identify bottleneck pair
        try:
            G = base_X @ base_X.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            a = flat // n
            b = flat % n
            # union of 10-15 largest neighbours per endpoint
            nn = 12
            # neighbours excluding self and other endpoint
            row_a = G[a].copy()
            row_a[a] = -2.0
            row_a[b] = -2.0
            row_b = G[b].copy()
            row_b[b] = -2.0
            row_b[a] = -2.0
            top_a = np.argsort(-row_a)[:nn]
            top_b = np.argsort(-row_b)[:nn]
            cluster = set([int(a), int(b)] + [int(x) for x in top_a] + [int(x) for x in top_b])
            cluster = sorted(cluster)
        except Exception:
            trial += 1
            continue
        k = ks[trial % len(ks)]
        k = min(k, len(cluster))
        mode = (trial // len(ks)) % 4  # 0: neither endpoint, 1: include a, 2: include b, 3: both/random
        cands = [c for c in cluster]
        del_set = None
        try:
            if mode == 0:
                avail = [c for c in cluster if c != a and c != b]
                if len(avail) < k:
                    avail = [c for c in cluster]
                    del_set = list(rng.choice(avail, size=k, replace=False))
                else:
                    del_set = list(rng.choice(avail, size=k, replace=False))
            elif mode == 1:
                avail = [c for c in cluster if c != b]
                # force include a
                rest = [c for c in avail if c != a]
                need = k - 1
                if need <= 0:
                    del_set = [int(a)]
                else:
                    need = min(need, len(rest))
                    picks = list(rng.choice(rest, size=need, replace=False)) if need > 0 else []
                    del_set = [int(a)] + [int(x) for x in picks]
                    # if still short pad randomly
                    while len(del_set) < k:
                        r = int(rng.integers(0, n))
                        if r not in del_set:
                            del_set.append(r)
            elif mode == 2:
                avail = [c for c in cluster if c != a]
                rest = [c for c in avail if c != b]
                need = k - 1
                if need <= 0:
                    del_set = [int(b)]
                else:
                    need = min(need, len(rest))
                    picks = list(rng.choice(rest, size=need, replace=False)) if need > 0 else []
                    del_set = [int(b)] + [int(x) for x in picks]
                    while len(del_set) < k:
                        r = int(rng.integers(0, n))
                        if r not in del_set:
                            del_set.append(r)
            else:
                # random from cluster (may include endpoints or not)
                del_set = list(rng.choice(cluster, size=k, replace=False))
            del_set = [int(x) for x in del_set]
        except Exception:
            trial += 1
            continue

        # greedy reinsertion one at a time
        try:
            keep = [i for i in range(n) if i not in set(del_set)]
            Y = base_X[keep].copy()  # (n-k) x d
            order = list(del_set)
            rng.shuffle(order)
            new_vecs = {}
            for di in order:
                if time.time() - t0 > DEADLINE - 6:
                    break
                v = find_hole(Y)
                new_vecs[di] = v
                Y = np.vstack([Y, v.reshape(1, -1)])
            Xc = base_X.copy()
            for di, v in new_vecs.items():
                Xc[di] = v
            # perturb reinserted cluster by 1e-3..1e-2
            sig = float(np.exp(rng.uniform(np.log(1e-3), np.log(1e-2))))
            Xc[del_set] = Xc[del_set] + sig * rng.standard_normal((len(del_set), d))
            Xc = _norm(Xc)
            # optimize cluster with remainder frozen
            Xc = cluster_opt(Xc, del_set, steps=450, tau0=0.01, tau1=0.0015, lr0=0.01, lr1=0.001, mu=0.9)
            # unfreeze: short annealed softmax polish + greedy
            Xp, vp = short_polish(Xc)
        except Exception:
            trial += 1
            continue

        # acceptance: improve incumbent or diverse topology
        try:
            if vp < best_val - 1e-12:
                best, best_val = Xp.copy(), vp
                best_set = _top_set(best, 100)
                pool_add(Xp, vp)
            else:
                s = _top_set(Xp, 100)
                jac = _jaccard(s, best_set)
                if jac < 0.85:
                    pool_add(Xp, vp)
                # also occasionally keep slightly worse but very good to allow hopping:
                # if within 0.002 and pool small, keep is already covered by jaccard rule
        except Exception:
            pass
        trial += 1

    # final intensive polish on best with remaining time
    try:
        while time.time() - t0 < DEADLINE - 1.5:
            bX, bv = greedy(best, best_val, 4e-4, 800, tau=0.0010)
            if bv < best_val:
                best, best_val = bX, bv
                best_set = _top_set(best, 100)
            else:
                break
            if time.time() - t0 > DEADLINE - 3:
                break
        # one more anneal if time left
        if time.time() - t0 < DEADLINE - 6:
            try:
                Xa = anneal_full(best.copy(), steps=500, tau0=0.004, tau1=0.0006, s0=0.004, s1=0.0004)
                bX, bv = greedy(Xa, _maxcos(Xa), 2e-4, 800, tau=0.0007)
                if bv < best_val:
                    best, best_val = bX, bv
            except Exception:
                pass
    except Exception:
        pass

    best = _norm(best)
    return best
