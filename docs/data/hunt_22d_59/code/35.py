import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normed(A):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return A / n

def _polish_masked(X0, active_idx, steps, beta0, beta1, lr0, lr1, mu=0.9):
    X = X0.copy()
    n = X.shape[0]
    active = np.zeros(n, dtype=bool)
    active[np.asarray(active_idx, dtype=int)] = True
    pair_mask = active[:, None] | active[None, :]
    np.fill_diagonal(pair_mask, False)
    full_mask = ~np.eye(n, dtype=bool)
    vel = np.zeros_like(X)
    best_val = _max_cos(X)
    best_X = X.copy()
    lb0 = np.log(beta0); lb1 = np.log(beta1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    denom = max(1, steps - 1)
    for s in range(steps):
        frac = s / float(denom)
        beta = float(np.exp(lb0 + (lb1 - lb0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        # active max
        # masked max without copy of large array (59x59 tiny)
        m_act = float(np.max(np.where(pair_mask, G, -2.0)))
        # weights
        W = np.exp(beta * (G - m_act))
        W[~pair_mask] = 0.0
        S = float(W.sum() * 0.5)
        if S < 1e-30 or not np.isfinite(S):
            continue
        Grad = (W @ X) / S
        dot = (Grad * X).sum(axis=1, keepdims=True)
        Gt = Grad - dot * X
        Gt[~active] = 0.0
        vel = mu * vel + (1.0 - mu) * Gt
        X[active] = X[active] - lr * vel[active]
        # renorm active
        an = np.linalg.norm(X[active], axis=1, keepdims=True)
        an = np.maximum(an, 1e-12)
        X[active] = X[active] / an
        if (s % 50 == 49) or (s == steps - 1):
            # global max
            G2 = X @ X.T
            np.fill_diagonal(G2, -2.0)
            v = float(G2.max())
            if v < best_val:
                best_val = v
                best_X = X.copy()
    return best_X, best_val

def solve():
    import time
    import numpy as np
    t0 = time.time()
    np.random.seed(0)
    n, d = 59, 22
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = _normed(a)
                cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        best_X = _normed(np.random.randn(n, d))
    else:
        vals = [_max_cos(c) for c in cands]
        best_X = cands[int(np.argmin(vals))].copy()
    best_val = _max_cos(best_X)

    all_idx = np.arange(n)

    def elapsed():
        return time.time() - t0

    # Stage 0: brief direct polish
    try:
        Xp, vp = _polish_masked(best_X, all_idx, steps=1200, beta0=30.0, beta1=300.0, lr0=0.015, lr1=0.0008, mu=0.9)
        if vp < best_val:
            best_val = vp; best_X = Xp
    except Exception:
        pass

    # cooperative deletion-reinsertion trials
    B = 20000
    trial = 0
    k_cycle = [4, 5, 6, 7, 4, 6, 5, 7]
    while True:
        if elapsed() > 145.0:
            break
        if trial > 300:
            break
        k = int(k_cycle[trial % len(k_cycle)])
        trial += 1
        try:
            # contact counts for removal choice
            G = best_X @ best_X.T
            np.fill_diagonal(G, -2.0)
            m0 = float(G.max())
            marg_c = 0.004
            contacts = (G >= (m0 - marg_c)).sum(axis=1)
            if (trial % 2) == 1:
                # highest-contact removal
                order = np.argsort(contacts)  # ascending
                remove = order[-k:].copy()
                np.random.shuffle(remove)
            else:
                remove = np.random.choice(n, size=k, replace=False)
            frozen_mask = np.ones(n, dtype=bool)
            frozen_mask[remove] = False
            frozen_idx = all_idx[frozen_mask]
            F = best_X[frozen_idx]  # (nf,d)

            # batched random search: one batch of B=20000
            P = np.random.randn(B, d)
            P = _normed(P)
            # frozen max: B x nf product
            # chunk to bound memory (B=20000, nf~55 -> 1.1M floats fine in one go)
            Df = P @ F.T
            cur = Df.max(axis=1)  # (B,)
            inserted = []
            for _ in range(k):
                bi = int(np.argmin(cur))
                c = P[bi].copy()
                inserted.append(c)
                dd = P @ c
                # elementwise max
                np.maximum(cur, dd, out=cur)
                cur[bi] = 2.0  # never pick again
            inserted = np.stack(inserted, axis=0)  # (k,d)
            Xt = best_X.copy()
            Xt[remove] = inserted

            # active set: removed + 12 nearest neighbours each
            Gt = Xt @ Xt.T
            # for each removed r, nearest = largest dot among others
            active_set = set([int(r) for r in remove])
            for r in remove:
                row = Gt[int(r)].copy()
                row[int(r)] = -2.0
                # also exclude other removed? keep them, they are active anyway
                nn = int(12)
                top = np.argsort(row)[-nn:]
                for t_ in top:
                    active_set.add(int(t_))
            active_idx = np.array(sorted(active_set), dtype=int)

            # optimize only inserted + neighbours with masked objective
            Xa, va = _polish_masked(Xt, active_idx, steps=900, beta0=25.0, beta1=250.0, lr0=0.018, lr1=0.0009, mu=0.9)
            # brief full polish
            Xf, vf = _polish_masked(Xa, all_idx, steps=600, beta0=60.0, beta1=350.0, lr0=0.006, lr1=0.0003, mu=0.88)
            if vf < best_val:
                best_val = vf
                best_X = Xf.copy()
            elif va < best_val:
                # va is global best tracked inside masked run (global max), can still improve
                best_val = va
                best_X = Xa.copy()
        except Exception:
            continue
        # time check every batch (every trial)
        if elapsed() > 145.0:
            break

    # final polish with remaining time
    if elapsed() < 165.0:
        try:
            rem = 165.0 - elapsed()
            steps = 1500 if rem > 12 else 600
            Xp, vp = _polish_masked(best_X, all_idx, steps=steps, beta0=80.0, beta1=500.0, lr0=0.003, lr1=0.00008, mu=0.9)
            if vp < best_val:
                best_val = vp; best_X = Xp
        except Exception:
            pass
    if elapsed() < 172.0:
        try:
            Xp, vp = _polish_masked(best_X, all_idx, steps=800, beta0=40.0, beta1=250.0, lr0=0.0012, lr1=0.00006, mu=0.85)
            if vp < best_val:
                best_val = vp; best_X = Xp
        except Exception:
            pass

    best_X = _normed(best_X)
    return best_X
