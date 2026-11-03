import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _refine_single(c, Y, iters=60, beta=3000.0):
    c = c / (np.linalg.norm(c) + 1e-16)
    for t in range(iters):
        d = Y @ c
        M = float(np.max(d))
        e = beta * (d - M)
        np.clip(e, -30.0, 0.0, out=e)
        np.exp(e, out=e)
        S = float(np.sum(e))
        if S < 1e-100:
            break
        e /= S
        g = Y.T @ e
        gd = float(g @ c)
        g = g - gd * c
        # decaying lr
        frac = t / max(1, iters - 1)
        lr = 0.04 * (1.0 - frac) + 0.004 * frac
        c = c - lr * g
        n = float(np.linalg.norm(c))
        if n == 0 or not np.isfinite(n):
            break
        c /= n
    return c

def _joint_block(B0, O, steps=120, beta=5000.0, lr=0.004):
    B = B0.copy()
    b = B.shape[0]
    m = np.zeros_like(B)
    v = np.zeros_like(B)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    t = 0
    for s in range(steps):
        Dbo = B @ O.T
        Dbb = B @ B.T
        np.fill_diagonal(Dbb, -10.0)
        M = float(Dbo.max()) if Dbo.size else -2.0
        Mb = float(Dbb.max()) if Dbb.size else -2.0
        if Mb > M:
            M = Mb
        Ebo = beta * (Dbo - M)
        np.clip(Ebo, -30.0, 0.0, out=Ebo)
        np.exp(Ebo, out=Ebo)
        Ebb = beta * (Dbb - M)
        np.clip(Ebb, -30.0, 0.0, out=Ebb)
        np.exp(Ebb, out=Ebb)
        np.fill_diagonal(Ebb, 0.0)
        S = float(np.sum(Ebo) + np.sum(Ebb))
        if S < 1e-100 or not np.isfinite(S):
            continue
        Wbo = Ebo / S
        Wbb = Ebb / S
        G = Wbo @ O + Wbb @ B
        G *= 2.0
        dot = np.sum(G * B, axis=1, keepdims=True)
        G -= dot * B
        t += 1
        m *= b1
        m += (1.0 - b1) * G
        v *= b2
        v += (1.0 - b2) * (G * G)
        c1 = 1.0 - (b1 ** t)
        c2 = 1.0 - (b2 ** t)
        mh = m / c1
        vh = v / c2
        step = mh / (np.sqrt(vh) + eps) * lr
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 5.0 * lr + 1e-12
        sc = np.minimum(1.0, cap / sn)
        step *= sc
        B -= step
        nrm = np.linalg.norm(B, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        B /= nrm
        md = np.sum(m * B, axis=1, keepdims=True)
        m -= md * B
    return B

def _adam_all(X0, steps, b0, b1, lr0, lr1):
    X = X0.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1m = 0.9
    b2m = 0.999
    eps = 1e-8
    t = 0
    lb0 = np.log(b0)
    lb1 = np.log(b1)
    for s in range(steps):
        frac = s / (steps - 1) if steps > 1 else 1.0
        beta = np.exp(lb0 + (lb1 - lb0) * frac)
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        cm = float(np.max(G))
        G -= cm
        G *= beta
        np.clip(G, -30.0, 0.0, out=G)
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G))
        if S < 1e-100:
            continue
        GX = G @ X
        GX *= (2.0 / S)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GX -= dot * X
        t += 1
        m *= b1m
        m += (1.0 - b1m) * GX
        v *= b2m
        v += (1.0 - b2m) * (GX * GX)
        c1 = 1.0 - (b1m ** t)
        c2 = 1.0 - (b2m ** t)
        denom = np.sqrt(v / c2) + eps
        step = (lr / c1) * m / denom
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-16
        cap = 5.0 * lr + 1e-12
        sc = np.minimum(1.0, cap / sn)
        step *= sc
        X -= step
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
        md = np.sum(m * X, axis=1, keepdims=True)
        m -= md * X
    return X

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    TRIAL_END = t0 + 140.0
    n, d = 250, 6
    best = None
    best_val = np.inf
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=float)
            if a.shape == (n, d):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    vv = _max_cos(a)
                    if vv < best_val:
                        best_val = vv
                        best = a.copy()
        except Exception:
            pass
    if best is None:
        rng0 = np.random.default_rng(0)
        best = rng0.standard_normal((n, d))
        best /= np.linalg.norm(best, axis=1, keepdims=True)
        best_val = _max_cos(best)
    base_fallback = best.copy()
    base_fallback_val = best_val

    sizes = [15, 25, 40]
    # (size, seed) trials
    trials = []
    for sz in sizes:
        for sd in (101, 202):
            trials.append((sz, sd))
    # limit to fit time: will early break anyway
    for ti, (K, rseed) in enumerate(trials):
        if time.time() > TRIAL_END - 12.0:
            break
        try:
            cur = best.copy()
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            cur_max = float(np.max(G))
            flat = int(np.argmax(G))
            i_star = flat // n
            j_star = flat % n
            thr = cur_max - 2e-5
            Adj = (cur @ cur.T) > thr
            np.fill_diagonal(Adj, False)
            rng = np.random.default_rng(10000 + ti * 1000 + rseed)
            # BFS component from i_star with randomized neighbor order
            visited = np.zeros(n, dtype=bool)
            order = []
            queue = [int(i_star)]
            visited[int(i_star)] = True
            qpos = 0
            while qpos < len(queue):
                u = queue[qpos]
                qpos += 1
                order.append(u)
                nb = np.where(Adj[u])[0]
                if nb.size > 1:
                    # shuffle for diversity
                    perm = rng.permutation(nb.shape[0])
                    nb = nb[perm]
                for w in nb:
                    w = int(w)
                    if not visited[w]:
                        visited[w] = True
                        queue.append(w)
            comp = order
            if len(comp) >= K:
                sel = np.array(comp[:K], dtype=int)
                # ensure both bottleneck points included
                if int(j_star) not in sel:
                    sel[-1] = int(j_star)
                    if int(i_star) not in sel:
                        sel[-2] = int(i_star)
                else:
                    if int(i_star) not in sel:
                        sel[-1] = int(i_star)
            else:
                need = K - len(comp)
                # add worst rowmax points outside
                np.fill_diagonal(G, -2.0)
                rowmax = np.max(G, axis=1)
                idx_sorted = np.argsort(-rowmax)
                extra = []
                inset = set(int(x) for x in comp)
                inset.add(int(i_star)); inset.add(int(j_star))
                for cnd in idx_sorted:
                    cnd = int(cnd)
                    if cnd not in inset:
                        extra.append(cnd)
                        inset.add(cnd)
                        if len(extra) >= need:
                            break
                sel = np.array(comp + extra, dtype=int)
                # force include bottleneck pair
                sset = set(int(x) for x in sel)
                if int(i_star) not in sset or int(j_star) not in sset:
                    # replace last ones
                    sel = sel.tolist()
                    if int(i_star) not in sset:
                        sel[-1] = int(i_star)
                    if int(j_star) not in set(int(x) for x in sel):
                        sel[-2] = int(j_star)
                    sel = np.array(sel, dtype=int)
                # dedup just in case
                # ensure length K unique
                uniq = []
                seen = set()
                for x in sel:
                    x = int(x)
                    if x not in seen:
                        seen.add(x)
                        uniq.append(x)
                # fill if dedup shortened
                if len(uniq) < K:
                    for cnd in idx_sorted:
                        cnd = int(cnd)
                        if cnd not in seen:
                            uniq.append(cnd)
                            seen.add(cnd)
                            if len(uniq) >= K:
                                break
                sel = np.array(uniq[:K], dtype=int)

            mask = np.ones(n, dtype=bool)
            mask[sel] = False
            frozen = cur[mask]  # (n-K) x d
            # joint regrowth
            rrg = np.random.default_rng(50000 + ti * 977 + rseed)
            picked = []
            Bsofar = None
            aborted = False
            for k in range(K):
                if time.time() > TRIAL_END - 8.0:
                    aborted = True
                    break
                R = rrg.standard_normal((8192, d))
                R /= (np.linalg.norm(R, axis=1, keepdims=True) + 1e-16)
                Df = R @ frozen.T
                mf = np.max(Df, axis=1)
                if len(picked) > 0:
                    Bs = np.stack(picked)
                    Db = R @ Bs.T
                    mb = np.max(Db, axis=1)
                    m = np.maximum(mf, mb)
                else:
                    m = mf
                bi = int(np.argmin(m))
                c = R[bi].copy()
                if len(picked) > 0:
                    Ys = np.vstack([frozen, np.stack(picked)])
                else:
                    Ys = frozen
                c = _refine_single(c, Ys, iters=60, beta=3000.0)
                picked.append(c)
                if ((k + 1) % 5 == 0) or (k == K - 1):
                    try:
                        Bl = np.stack(picked)
                        Bl = _joint_block(Bl, frozen, steps=120, beta=5000.0, lr=0.004)
                        picked = [Bl[i].copy() for i in range(Bl.shape[0])]
                    except Exception:
                        pass
            if aborted:
                continue
            Ynew = cur.copy()
            Bl = np.stack(picked)
            # normalize
            Bl /= (np.linalg.norm(Bl, axis=1, keepdims=True) + 1e-16)
            Ynew[sel] = Bl
            # short collective polish
            try:
                if time.time() < TRIAL_END - 6.0:
                    Ynew = _adam_all(Ynew, steps=220, b0=800.0, b1=12000.0, lr0=0.0025, lr1=0.0006)
            except Exception:
                pass
            Ynew /= (np.linalg.norm(Ynew, axis=1, keepdims=True) + 1e-16)
            vv = _max_cos(Ynew)
            if vv < best_val - 1e-12 and np.all(np.isfinite(Ynew)):
                best = Ynew.copy()
                best_val = vv
                # continue from improved basin
            # also if close but not better, ignore
        except Exception:
            continue
        if time.time() > HARD - 12.0:
            break

    # final safety: if no improvement, return fallback; tiny shrink not needed as already unit
    try:
        if best is None:
            best = base_fallback
            best_val = base_fallback_val
        # keep best of best vs fallback (numerical)
        if base_fallback_val < _max_cos(best):
            best = base_fallback
        best = best / np.linalg.norm(best, axis=1, keepdims=True)
        # guard zero/nan
        if not np.all(np.isfinite(best)):
            best = base_fallback / np.linalg.norm(base_fallback, axis=1, keepdims=True)
    except Exception:
        try:
            best = np.asarray(np.load("best.npy"), dtype=float)
            best = best / np.linalg.norm(best, axis=1, keepdims=True)
        except Exception:
            rng0 = np.random.default_rng(0)
            best = rng0.standard_normal((n, d))
            best /= np.linalg.norm(best, axis=1, keepdims=True)
    return best
