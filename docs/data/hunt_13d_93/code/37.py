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

def _adam_smooth(X, steps, tau0, tau1, lr0, lr1):
    n, d = X.shape
    x = X.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    tt = 0
    ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    for s in range(steps):
        f = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * f)
        lr = lr0 * np.exp(llr * f)
        if tau < 1e-12:
            tau = 1e-12
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mg = float(G.max())
        W = np.exp((G - mg) / tau)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        tt += 1
        m = b1 * m + (1.0 - b1) * gt
        vv = b2 * v + (1.0 - b2) * (gt * gt)
        v = vv
        mh = m / (1.0 - b1 ** tt)
        vh = vv / (1.0 - b2 ** tt)
        step = lr * mh / (np.sqrt(vh) + eps)
        dot2 = (step * x).sum(axis=1, keepdims=True)
        step_t = step - dot2 * x
        x = x - step_t
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def _adam_subset(X, idx, steps, tau0, tau1, lr0, lr1):
    n, d = X.shape
    x = X.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    tt = 0
    ltau = np.log(tau1 / tau0) if tau0 != tau1 else 0.0
    llr = np.log(lr1 / lr0) if lr0 != lr1 else 0.0
    idx = np.asarray(idx, dtype=int)
    for s in range(steps):
        f = s / max(1, steps - 1)
        tau = tau0 * np.exp(ltau * f)
        lr = lr0 * np.exp(llr * f)
        if tau < 1e-12:
            tau = 1e-12
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        mg = float(G.max())
        W = np.exp((G - mg) / tau)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = (grad * x).sum(axis=1, keepdims=True)
        gt = grad - dot * x
        tt += 1
        m = b1 * m + (1.0 - b1) * gt
        vv = b2 * v + (1.0 - b2) * (gt * gt)
        v = vv
        mh = m / (1.0 - b1 ** tt)
        vh = vv / (1.0 - b2 ** tt)
        step = lr * mh / (np.sqrt(vh) + eps)
        dot2 = (step * x).sum(axis=1, keepdims=True)
        step_t = step - dot2 * x
        x[idx] = x[idx] - step_t[idx]
        x[idx] /= np.linalg.norm(x[idx], axis=1, keepdims=True)
    return x

def _hardK(X, iters, step0, K, iu0, iu1):
    best = _maxcos(X)
    step = step0
    mtot = len(iu0)
    K = int(min(K, mtot))
    grad = np.empty_like(X)
    for _ in range(iters):
        G = X @ X.T
        c = G[iu0, iu1]
        if K < mtot:
            idx = np.argpartition(-c, K)[:K]
        else:
            idx = np.arange(mtot)
        I = iu0[idx]
        J = iu1[idx]
        grad.fill(0.0)
        np.add.at(grad, I, X[J])
        np.add.at(grad, J, X[I])
        grad /= float(len(idx))
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        ng = float(np.linalg.norm(gt))
        if ng == 0 or not np.isfinite(ng):
            step *= 0.5
            if step < 1e-13:
                break
            continue
        Xt = X - step * gt
        Xt /= np.linalg.norm(Xt, axis=1, keepdims=True)
        G2 = Xt @ Xt.T
        np.fill_diagonal(G2, -2.0)
        vv = float(G2.max())
        if vv < best:
            X = Xt
            best = vv
            step = min(step * 1.25, 0.02)
        else:
            step *= 0.5
            if step < 1e-13:
                break
    return X, best

def _top100_set(X, K=100):
    n = X.shape[0]
    G = X @ X.T
    iu0, iu1 = np.triu_indices(n, 1)
    c = G[iu0, iu1]
    if K >= len(c):
        idx = np.arange(len(c))
    else:
        idx = np.argpartition(-c, K)[:K]
    s = set()
    for k in idx:
        i = int(iu0[k]); j = int(iu1[k])
        s.add(i * 256 + j)
    return s

def _jaccard(a, b):
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / max(1, union)

def _find_hole(others, rng, n_cand=50000, steps=200):
    d = others.shape[1]
    C = rng.standard_normal((n_cand, d))
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    D = C @ others.T
    worst = D.max(axis=1)
    vv = C[int(np.argmin(worst))].copy()
    lr = 0.05
    tau = 0.005
    for it in range(steps):
        dots = others @ vv
        m0 = float(dots.max())
        w = np.exp((dots - m0) / tau)
        s = float(w.sum())
        if s <= 0 or not np.isfinite(s):
            break
        w /= s
        g = w @ others
        gt = g - float(g @ vv) * vv
        nrm = float(np.linalg.norm(gt))
        if nrm < 1e-12 or not np.isfinite(nrm):
            break
        gt /= nrm
        v2 = vv - lr * gt
        n2 = float(np.linalg.norm(v2))
        if n2 < 1e-12:
            lr *= 0.5
            continue
        v2 /= n2
        m2 = float((others @ v2).max())
        if m2 < m0:
            vv = v2
            lr = min(lr * 1.05, 0.08)
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
            if it % 50 == 49:
                tau *= 0.7
                if tau < 8e-4:
                    tau = 8e-4
    vv /= np.linalg.norm(vv)
    return vv

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    HOP_END = t0 + 138.0
    rng = np.random.default_rng(7)
    n, d = 93, 13
    iu0, iu1 = np.triu_indices(n, 1)
    best = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            if a.shape != (93, 13):
                continue
            a = _norm(np.asarray(a, dtype=np.float64))
            vv = _maxcos(a)
            if vv < best_val:
                best_val = vv
                best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _norm(rng.standard_normal((93, 13)))
        best_val = _maxcos(best)
    incumbent = best.copy()
    inc_val = best_val
    inc_top = _top100_set(incumbent, 100)
    pool = [(incumbent.copy(), inc_val, inc_top)]
    pool_tops = [inc_top]

    k_cycle = [3, 5, 8, 12]
    trial = 0
    while time.time() < HOP_END:
        remain = (t0 + DEADLINE) - time.time()
        if remain < 8:
            break
        # choose base: mostly incumbent, sometimes diverse pool member
        if len(pool) > 1 and (trial % 5 == 4):
            # pick random non-best pool member to diversify
            pi = 1 + int(rng.integers(0, len(pool) - 1))
            # guard
            pi = pi % len(pool)
            base = pool[pi][0]
        else:
            base = incumbent
        k = k_cycle[trial % len(k_cycle)]
        mode = (trial // len(k_cycle)) % 3  # 0: incl a, 1: incl b, 2: neither
        try:
            G = base @ base.T
            np.fill_diagonal(G, -2.0)
            flat = int(np.argmax(G))
            a = flat // n; b = flat % n
            # neighbours
            ra = np.argsort(-G[a])
            rb = np.argsort(-G[b])
            # take top 12-15 (vary slightly by trial)
            tn = 10 + (trial % 6)  # 10..15
            na = [int(x) for x in ra[:tn + 1] if int(x) != a][:tn]
            nb = [int(x) for x in rb[:tn + 1] if int(x) != b][:tn]
            cluster = sorted(set(na) | set(nb) | {a, b})
            # ensure size 15-25: if too small expand
            if len(cluster) < 15:
                # add global worst points
                rm = G.max(axis=1)
                order = np.argsort(-rm)
                for c in order:
                    c = int(c)
                    if c not in cluster:
                        cluster.append(c)
                    if len(cluster) >= 20:
                        break
                cluster = sorted(set(cluster))
            # choose k deletions
            cluster_arr = np.array(cluster, dtype=int)
            if mode == 0:
                # include a
                rest = [c for c in cluster if c != a]
                rng.shuffle(rest)
                chosen = [int(a)] + [int(x) for x in rest[:k - 1]]
                # if k==1 edge (not in cycle) fine
                if len(chosen) < k:
                    # fill randomly
                    extra = rng.choice(n, size=k - len(chosen), replace=False)
                    for e in extra:
                        e = int(e)
                        if e not in chosen:
                            chosen.append(e)
                    chosen = chosen[:k]
            elif mode == 1:
                rest = [c for c in cluster if c != b]
                rng.shuffle(rest)
                chosen = [int(b)] + [int(x) for x in rest[:k - 1]]
                if len(chosen) < k:
                    extra = rng.choice(n, size=k - len(chosen), replace=False)
                    for e in extra:
                        e = int(e)
                        if e not in chosen:
                            chosen.append(e)
                    chosen = chosen[:k]
            else:
                rest = [c for c in cluster if c != a and c != b]
                rng.shuffle(rest)
                chosen = [int(x) for x in rest[:k]]
                # if not enough, fill
                if len(chosen) < k:
                    cand = [i for i in range(n) if i != a and i != b and i not in chosen]
                    rng.shuffle(cand)
                    chosen += [int(x) for x in cand[:k - len(chosen)]]
            chosen = [int(x) for x in chosen[:k]]
            order_ins = chosen.copy()
            rng.shuffle(order_ins)
            X = base.copy()
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[np.array(chosen, dtype=int)] = False
            stack = X[keep_mask].copy()
            for c in order_ins:
                vv = _find_hole(stack, rng, n_cand=50000, steps=200)
                X[c] = vv
                stack = np.vstack([stack, vv[None, :]])
                if time.time() > t0 + DEADLINE - 6:
                    break
            # perturb reinserted cluster
            sig = float(10 ** rng.uniform(-3.0, -2.0))  # 1e-3..1e-2
            X[chosen] += sig * rng.standard_normal((len(chosen), d))
            X[chosen] /= np.linalg.norm(X[chosen], axis=1, keepdims=True)
            # optimize cluster with remainder frozen
            # shorter for large k to save time
            if k <= 5:
                X = _adam_subset(X, chosen, steps=350, tau0=0.008, tau1=0.0006, lr0=0.015, lr1=0.002)
            else:
                X = _adam_subset(X, chosen, steps=250, tau0=0.010, tau1=0.0008, lr0=0.015, lr1=0.002)
            # unfreeze: short full polish
            # vary polish slightly
            if trial % 2 == 0:
                X = _adam_smooth(X, steps=500, tau0=0.008, tau1=0.0005, lr0=0.012, lr1=0.0012)
            else:
                X = _adam_smooth(X, steps=350, tau0=0.012, tau1=0.0007, lr0=0.015, lr1=0.0015)
            Xb, vb = _hardK(X, 220, 0.0012, 300, iu0, iu1)
            Xc, vc = _hardK(Xb, 180, 0.0007, 80, iu0, iu1)
            if vc < vb:
                Xb, vb = Xc, vc
            vv = vb
            Xf = Xb
        except Exception:
            trial += 1
            continue
        try:
            top = _top100_set(Xf, 100)
            jac = _jaccard(top, inc_top)
            if vv < inc_val - 1e-12:
                incumbent = Xf.copy()
                inc_val = vv
                inc_top = top
                pool.append((Xf.copy(), vv, top))
                pool_tops.append(top)
                # keep pool bounded
                if len(pool) > 12:
                    # keep best + most diverse: sort by val
                    pool_sorted = sorted(pool, key=lambda z: z[1])
                    # keep best 6 + last diverse ones
                    pool = pool_sorted[:6] + pool[-3:]
            elif jac < 0.85:
                # diverse topology, keep as pool member (valuable)
                pool.append((Xf.copy(), vv, top))
                pool_tops.append(top)
                if len(pool) > 12:
                    pool_sorted = sorted(pool, key=lambda z: z[1])
                    pool = pool_sorted[:6] + pool[-3:]
            else:
                # same plateau, discard (do not add)
                pass
        except Exception:
            pass
        trial += 1
        # opportunistic: if time low, break
        if trial > 200:
            break
    # final polish of incumbent with remaining time
    try:
        remain = (t0 + DEADLINE) - time.time()
        if remain > 4:
            # multiscale hard equalize
            for K, s0, it in [(600, 0.0009, 400), (200, 0.0006, 400), (80, 0.00035, 300)]:
                if time.time() > t0 + DEADLINE - 3:
                    break
                b, bv = _hardK(incumbent, it, s0, K, iu0, iu1)
                if bv < inc_val:
                    incumbent, inc_val = b, bv
            # small smooth touch
            if time.time() < t0 + DEADLINE - 4:
                Xt = _adam_smooth(incumbent, steps=300, tau0=0.004, tau1=0.0003, lr0=0.006, lr1=0.0008)
                Xb, vb = _hardK(Xt, 250, 0.0006, 120, iu0, iu1)
                if vb < inc_val:
                    incumbent, inc_val = Xb, vb
    except Exception:
        pass
    incumbent = _norm(incumbent)
    # safety shrink not needed but ensure finite
    return incumbent
