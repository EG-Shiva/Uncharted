import time
import numpy as np

def _norm_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[~np.isfinite(n)] = 1.0
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _top20(X):
    n = X.shape[0]
    iu = np.triu_indices(n, 1)
    c = (X @ X.T)[iu]
    # partition top 20
    if c.size >= 20:
        idx = np.argpartition(-c, 20)[:20]
        v = c[idx]
        v = np.sort(v)[::-1]
    else:
        v = np.sort(c)[::-1]
    return v

_W20 = np.exp(-np.arange(20) / 4.0)
_W20 = _W20 / _W20[0]

def _energy_from_top(v):
    # weighted lexicographic surrogate, top dominates
    k = min(len(v), 20)
    return float(np.dot(_W20[:k], v[:k]) / 20.0)

def _quench(X_init, steps, T0, T1, lr0, lr1, deadline=None):
    X = _norm_rows(np.array(X_init, dtype=float, copy=True))
    n, d = X.shape
    m1 = np.zeros_like(X)
    v1 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    lT0, lT1 = np.log(T0), np.log(T1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    G0 = X @ X.T
    np.fill_diagonal(G0, -2.0)
    best_m = float(G0.max())
    best_X = X.copy()
    for s in range(1, steps + 1):
        if deadline is not None and (s % 120 == 0) and time.time() > deadline:
            break
        frac = (s - 1) / max(steps - 1, 1)
        T = float(np.exp(lT0 * (1 - frac) + lT1 * frac))
        lr = float(np.exp(ll0 * (1 - frac) + ll1 * frac))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        if m < best_m:
            best_m = m
            best_X = X.copy()
        D = (G - m) / T
        np.clip(D, -50, 0, out=D)
        E = np.exp(D)
        np.fill_diagonal(E, 0.0)
        S = float(E.sum() * 0.5)
        if not np.isfinite(S) or S <= 0:
            S = 1.0
        Gr = (E @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        if not np.all(np.isfinite(Gt)):
            break
        m1 = b1 * m1 + (1 - b1) * Gt
        v1 = b2 * v1 + (1 - b2) * (Gt * Gt)
        c1 = 1.0 - b1 ** s
        c2 = 1.0 - b2 ** s
        step = (m1 / c1) / (np.sqrt(v1 / c2) + eps)
        sn = float(np.linalg.norm(step))
        mxn = 8.0 * np.sqrt(n * d)
        if sn > mxn and sn > 0:
            step *= (mxn / sn)
        X = X - lr * step
        X = _norm_rows(X)
        # cheap intermediate check
        if (s % 200 == 0) or (s == steps):
            Gc = X @ X.T
            np.fill_diagonal(Gc, -2.0)
            vc = float(Gc.max())
            if vc < best_m:
                best_m = vc
                best_X = X.copy()
    return best_X, best_m

def _perturb_loosest(X, rng):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    pmx = G.max(axis=1)
    k = int(rng.integers(5, 21))
    # loosest = smallest per-point max
    idx = np.argsort(pmx)[:k]
    if rng.random() < 0.3:
        # mix in some worst to diversify
        widx = np.argsort(-pmx)[:2]
        idx[:min(2, k)] = widx[:min(2, k)]
    X0 = X.copy()
    X0[idx] = _norm_rows(rng.standard_normal((len(idx), d)))
    # tiny jitter to rest
    X0 += rng.standard_normal((n, d)) * 0.005
    return _norm_rows(X0)

def _perturb_rotate_cluster(X, rng):
    n, d = X.shape
    k = int(rng.integers(5, 16))
    seed = int(rng.integers(0, n))
    G = X @ X.T
    sims = G[seed].copy()
    sims[seed] = 2.0
    idx = np.argsort(-sims)[:k]
    # random coordinate 2-plane
    a = int(rng.integers(0, d))
    b = int(rng.integers(0, d - 1))
    if b >= a:
        b += 1
    th = float(rng.uniform(0.05, 0.3)) * (1.0 if rng.random() < 0.5 else -1.0)
    c, s = np.cos(th), np.sin(th)
    X0 = X.copy()
    # same ambient Givens rotation applied to cluster
    xa = X0[idx, a].copy()
    xb = X0[idx, b].copy()
    X0[idx, a] = c * xa - s * xb
    X0[idx, b] = s * xa + c * xb
    X0[idx] += rng.standard_normal((len(idx), d)) * 0.02
    # jitter few random others
    jr = rng.choice(n, size=min(4, n), replace=False)
    X0[jr] += rng.standard_normal((len(jr), d)) * 0.02
    return _norm_rows(X0)

def _perturb_pca_gap(X, rng):
    n, d = X.shape
    # largest PCA gap approximated by biggest spherical hole
    # find hole via random candidates
    nc = 4000
    Y = _norm_rows(rng.standard_normal((nc, d)))
    D = Y @ X.T
    scores = D.max(axis=1)
    h = Y[int(np.argmin(scores))]
    # crowded cluster to move
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    pmx = G.max(axis=1)
    k = int(rng.integers(5, 16))
    worst = np.argsort(-pmx)[:k]
    X0 = X.copy()
    alpha = float(rng.uniform(0.5, 0.9))
    for ii in worst:
        noise = rng.standard_normal(d) * 0.35
        v = (1 - alpha) * X0[ii] + alpha * h + noise * 0.3
        # normalize later
        X0[ii] = v
    X0 = _norm_rows(X0)
    return X0

def _fingerprint(X):
    v = _top20(X)
    # pad to 20
    return v.copy()

def solve():
    t0 = time.time()
    HARD_END = t0 + 173.0
    HOP_END = HARD_END - 14.0
    rng = np.random.default_rng(20261101)
    n, d = 71, 12
    best_X = None
    try:
        best_X = np.load("best.npy").astype(float)
    except Exception:
        best_X = None
    if best_X is None or best_X.shape != (71, 12) or not np.all(np.isfinite(best_X)):
        try:
            best_X = np.load("record.npy").astype(float)
        except Exception:
            best_X = rng.standard_normal((71, 12))
    best_X = _norm_rows(np.asarray(best_X, dtype=float))
    try:
        best_score = _max_cos(best_X)
    except Exception:
        best_score = 1.0
    # also consider record.npy if better
    try:
        rec = _norm_rows(np.load("record.npy").astype(float))
        rs = _max_cos(rec)
        if rs < best_score:
            best_X, best_score = rec, rs
    except Exception:
        pass
    best_top = _top20(best_X)
    best_en = _energy_from_top(best_top)
    cur_X = best_X.copy()
    cur_score = best_score
    cur_en = best_en
    archive = [_fingerprint(best_X)]
    it = 0
    # quench length schedule: short 500-900 steps ~ well under 5-10s
    while time.time() < HOP_END:
        it += 1
        if time.time() > HOP_END - 1.5:
            break
        frac_t = (time.time() - t0) / max(HOP_END - t0, 1e-9)
        frac_t = min(max(frac_t, 0.0), 1.0)
        # SA temperature on energy: allow ~1e-4 worsening early
        Tmet = 6e-6 * (1 - frac_t) + 8e-7 * frac_t
        # Tmet on energy scale (energy ~ max/20?) rescale: energy diff ~ max diff * sum(w)/20 ~0.3*diff
        # so use Tmet_e = Tmet*0.3 ; just use Tmet directly small
        r = rng.random()
        base = cur_X if rng.random() < 0.7 else best_X
        if r < 0.38:
            X0 = _perturb_loosest(base, rng)
            steps = int(rng.integers(550, 900))
            T0 = float(rng.uniform(0.010, 0.020))
            T1 = 0.00035
            lr0, lr1 = 0.025, 0.0025
        elif r < 0.70:
            X0 = _perturb_rotate_cluster(base, rng)
            steps = int(rng.integers(600, 950))
            T0 = float(rng.uniform(0.012, 0.022))
            T1 = 0.0003
            lr0, lr1 = 0.03, 0.002
        else:
            X0 = _perturb_pca_gap(base, rng)
            steps = int(rng.integers(600, 950))
            T0 = float(rng.uniform(0.012, 0.022))
            T1 = 0.00035
            lr0, lr1 = 0.03, 0.0022
        # p-norm style alternative: occasionally use colder T0
        if it % 7 == 0:
            T0 *= 0.5
        try:
            Xq, _ = _quench(X0, steps, T0, T1, lr0, lr1, deadline=HOP_END)
            qscore = _max_cos(Xq)
            qtop = _top20(Xq)
            qen = _energy_from_top(qtop)
        except Exception:
            continue
        # archive distinctness
        try:
            fp = qtop
            dmin = min(float(np.linalg.norm(fp - a)) for a in archive) if archive else 1e9
            if dmin > 5e-5:
                archive.append(fp.copy())
                if len(archive) > 300:
                    archive.pop(0)
            # if duplicate, slightly penalize acceptance by cooling? skip
        except Exception:
            pass
        # best tracking on true max (primary) then energy
        improved = False
        if qscore < best_score - 1e-12 or (abs(qscore - best_score) <= 1e-12 and qen < best_en - 1e-14):
            best_X, best_score, best_en = Xq, qscore, qen
            cur_X, cur_score, cur_en = Xq, qscore, qen
            improved = True
        else:
            # SA on energy allowing uphill up to ~1e-4 in max (~3e-5 in energy)
            if qen < cur_en or rng.random() < np.exp((cur_en - qen) / max(Tmet, 1e-12)):
                cur_X, cur_score, cur_en = Xq, qscore, qen
        # occasional intensification on best
        if (it % 10 == 0) and time.time() < HOP_END - 4:
            try:
                Xf, _ = _quench(best_X, 350, 0.002, 0.00015, 0.008, 0.001, deadline=HOP_END)
                sf = _max_cos(Xf)
                ef = _energy_from_top(_top20(Xf))
                if sf < best_score - 1e-12 or (abs(sf - best_score) <= 1e-12 and ef < best_en):
                    best_X, best_score, best_en = Xf, sf, ef
                    cur_X, cur_score, cur_en = Xf, sf, ef
            except Exception:
                pass
    # final tightening quenches on best (cheap, pure numpy)
    try:
        if time.time() < HARD_END - 6:
            Xf, _ = _quench(best_X, 900, 0.003, 0.00012, 0.010, 0.0008, deadline=HARD_END - 2.0)
            sf = _max_cos(Xf)
            if sf < best_score:
                best_X, best_score = Xf, sf
        if time.time() < HARD_END - 3:
            Xf2, _ = _quench(best_X, 600, 0.0012, 0.00008, 0.006, 0.0006, deadline=HARD_END - 1.0)
            sf2 = _max_cos(Xf2)
            if sf2 < best_score:
                best_X, best_score = Xf2, sf2
    except Exception:
        pass
    best_X = _norm_rows(np.asarray(best_X, dtype=float))
    # strict feasibility: no zero rows, no NaN (already normalized)
    return best_X
