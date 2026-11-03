import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _adam_polish(X, steps, a0, a1, lr0, lr1, t0, BUDGET):
    X = X.copy()
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    cur_best = X.copy()
    cur_val = _max_cos(X)
    la0 = np.log(a0); la1 = np.log(a1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for t in range(1, steps + 1):
        if (t & 127) == 0 and (time.time() - t0) > BUDGET:
            break
        frac = t / steps
        alpha = float(np.exp(la0 + (la1 - la0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        G -= M
        G *= alpha
        np.exp(G, out=G)
        np.fill_diagonal(G, 0.0)
        S = float(np.sum(G))
        if not np.isfinite(S) or S <= 0:
            continue
        Grad = (G @ X) / S
        dot = np.sum(Grad * X, axis=1, keepdims=True)
        Gt = Grad - dot * X
        m *= b1; m += (1.0 - b1) * Gt
        v *= b2; v += (1.0 - b2) * (Gt * Gt)
        c1 = 1.0 - (b1 ** t); c2 = 1.0 - (b2 ** t)
        if c1 < 1e-12: c1 = 1.0
        if c2 < 1e-12: c2 = 1.0
        upd = (m / c1) / (np.sqrt(v / c2) + eps)
        X -= lr * upd
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        X /= nrm
        if (t % 500 == 0) or (t == steps):
            cv = _max_cos(X)
            if cv < cur_val:
                cur_val = cv
                cur_best = X.copy()
    fv = _max_cos(X)
    if fv < cur_val:
        cur_val = fv; cur_best = X.copy()
    return cur_best, cur_val

def _gram_alternate(G_init, target, iters):
    G = G_init.copy()
    n = G.shape[0]
    for _ in range(iters):
        # (i) unit diagonal, cap off-diagonal at target, floor at -1
        # cap everything then fix diagonal (diagonal would be capped otherwise)
        # in-place threshold
        np.minimum(G, target, out=G)
        # floor
        np.maximum(G, -1.0, out=G)
        np.fill_diagonal(G, 1.0)
        # enforce symmetry (minimum/maximum preserve it)
        # (ii)+(iii) PSD + rank<=22 via top-22 eigen clipping
        w, V = np.linalg.eigh(G)
        # last 22 are largest (eigh ascending)
        Vt = V[:, -22:]
        wt = w[-22:]
        # clip negatives
        wt = np.maximum(wt, 0.0)
        # reconstruct: (Vt * wt) @ Vt.T
        G = (Vt * wt) @ Vt.T
        # symmetrize numerically
        G = (G + G.T) * 0.5
    return G

def _factor(G, d=22):
    w, V = np.linalg.eigh(G)
    idx = np.argsort(w)[-d:]
    wt = np.maximum(w[idx], 1e-12)
    Vt = V[:, idx]
    X = Vt * np.sqrt(wt)[None, :]
    # Vt is (59,22), X rows in R22
    X = _normalize(X)
    return X

def solve():
    t0 = time.time()
    BUDGET = 150.0
    HARD = 165.0
    rng = np.random.default_rng(7)
    n, d = 59, 22

    # load incumbent
    best = None; best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn).astype(np.float64)
            if a.shape == (59, 22) and np.all(np.isfinite(a)):
                a = _normalize(a)
                v = _max_cos(a)
                if v < best_val:
                    best_val = v; best = a.copy()
        except Exception:
            pass
    if best is None:
        best = _normalize(rng.standard_normal((n, d)))
        best_val = _max_cos(best)
    incumbent = best.copy(); inc_val = best_val

    G_inc = incumbent @ incumbent.T

    targets = [0.064, 0.060, 0.055]
    iters_list = [350, 300, 250]
    cands = []  # (val, X)

    # generate trials: for each target, one from incumbent + 3 random
    for ti, t_targ in enumerate(targets):
        iters = iters_list[ti % len(iters_list)]
        inits = []
        inits.append(G_inc)
        # random-points Gram
        Xr = _normalize(rng.standard_normal((n, d)))
        inits.append(Xr @ Xr.T)
        # symmetrized uniform Gram
        for _ in range(2):
            A = rng.uniform(-0.25, 0.25, size=(n, n))
            A = (A + A.T) * 0.5
            np.fill_diagonal(A, 1.0)
            inits.append(A)
        # extra pure random symmetric wider
        A = rng.standard_normal((n, n))
        A = (A + A.T) * 0.5
        # scale off-diag down
        A *= 0.15 / (np.mean(np.abs(A)) + 1e-12) * 0.15
        np.fill_diagonal(A, 1.0)
        # clip to [-1,1]
        np.clip(A, -1.0, 1.0, out=A)
        inits.append(A)

        for G0 in inits:
            if (time.time() - t0) > BUDGET - 40.0:
                break
            Gf = _gram_alternate(G0, t_targ, iters)
            Xf = _factor(Gf, d)
            vf = _max_cos(Xf)
            cands.append((vf, Xf))
            # also try a second factoring with slight jitter on eigenvalues?
            # cheap diversity: add small noise to Xf and renormalize
            if len(cands) % 3 == 0:
                Xn = _normalize(Xf + 0.01 * rng.standard_normal((n, d)))
                cands.append((_max_cos(Xn), Xn))
        if (time.time() - t0) > BUDGET - 40.0:
            break

    # rank candidates, keep best few for polish
    cands.sort(key=lambda z: z[0])
    # keep top 3 distinct + incumbent fallback already tracked
    K = 3
    to_polish = []
    for vf, Xf in cands[:8]:
        # avoid duplicates that are essentially incumbent basin? keep all top
        to_polish.append((vf, Xf))
        if len(to_polish) >= K:
            break

    # short 2000-step annealed LSE-Adam polish on best few
    polished = []
    for vf, Xf in to_polish:
        if (time.time() - t0) > BUDGET - 20.0:
            break
        # skip hopelessly bad factors to save time
        if vf > best_val + 0.03 and len(polished) >= 1:
            continue
        Xp, vp = _adam_polish(Xf, steps=2000, a0=50.0, a1=700.0,
                              lr0=0.005, lr1=0.0004, t0=t0, BUDGET=HARD)
        polished.append((vp, Xp))
        if vp < best_val:
            best_val = vp; best = Xp.copy()

    # also consider unpolished factors in case polish diverged
    for vf, Xf in cands:
        if vf < best_val:
            best_val = vf; best = Xf.copy()
            break  # cands sorted, first is best; but polished likely better

    # re-check polished best
    for vp, Xp in polished:
        if vp < best_val:
            best_val = vp; best = Xp.copy()

    # fallback: ensure at least incumbent
    if best_val > inc_val:
        best_val = inc_val; best = incumbent.copy()

    # use remaining time for final squeeze of global best
    if (time.time() - t0) < HARD - 8.0:
        remain = HARD - (time.time() - t0)
        # longer final polish, anneal to high alpha
        steps = 6000 if remain > 30 else 2500
        Xp, vp = _adam_polish(best, steps=steps, a0=200.0, a1=1500.0,
                              lr0=0.003, lr1=0.0002, t0=t0, BUDGET=HARD)
        if vp < best_val:
            best_val = vp; best = Xp.copy()

    best = _normalize(best)
    return best
