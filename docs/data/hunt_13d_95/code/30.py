import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
try:
    from scipy.special import ndtri
    _HAS_NDTRI = True
except Exception:
    _HAS_NDTRI = False

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def per_point_max(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return g.max(axis=1)

def contact_counts_thr(x, mc, tol):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return (g > mc - tol).sum(axis=1)

def lp_polish(x, seconds, radius_init=1e-3, cap=1200):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c > thresh
        if int(sel.sum()) > cap:
            idx = np.argpartition(c, -cap)[-cap:]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def halton_seq(n, dim, seed_offset=0):
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41]
    out = np.empty((n, dim))
    for j in range(dim):
        b = primes[j % len(primes)]
        for i in range(n):
            idx = i + 1 + seed_offset
            f = 1.0
            r = 0.0
            while idx > 0:
                f /= b
                r += f * (idx % b)
                idx //= b
            # avoid 0/1
            if r <= 0:
                r = 0.5 / n
            if r >= 1:
                r = 1 - 0.5 / n
            out[i, j] = r
    return out

def solve():
    t0 = time.time()
    DEADLINE = t0 + 172.0
    rng = np.random.default_rng(20261103)
    n, d, B = 95, 13, 24
    betas = [20.0, 80.0, 250.0, 700.0]
    lrs = [0.01, 0.007, 0.004, 0.002]
    stage_time = [17.0, 17.0, 17.0, 17.0]  # ~68s total

    # init population (B, n, d) fresh start, never seed from record
    X = np.empty((B, n, d))
    # half random Gaussian
    for b in range(12):
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        X[b] = z
    # half low-discrepancy via Halton -> normal
    try:
        for b in range(12, 24):
            u = halton_seq(n * d, 1, seed_offset=(b-12)*7919).reshape(n, d)
            # need d dims independent: generate n*d with dim=d
            uh = halton_seq(n, d, seed_offset=(b-12)*104729 + 13)
            uh = np.clip(uh, 1e-6, 1-1e-6)
            if _HAS_NDTRI:
                z = ndtri(uh)
            else:
                # Box-Muller-ish approx: uniform -> normal via sum
                z = (uh - 0.5) * 4.0
            # random rotation to avoid axis alignment
            A = rng.standard_normal((d, d))
            Q, _ = np.linalg.qr(A)
            z = z @ Q.T
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            X[b] = z
    except Exception:
        for b in range(12, 24):
            z = rng.standard_normal((n, d))
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            X[b] = z
    # extra orthogonalization for diversity on some members
    for b in range(0, B, 6):
        A = rng.standard_normal((d, d))
        Q, _ = np.linalg.qr(A)
        X[b] = (X[b] @ Q.T)
        X[b] /= np.linalg.norm(X[b], axis=1, keepdims=True)

    M = np.zeros_like(X)
    Vc = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    step_global = 0

    def true_maxima(Xb):
        out = np.empty(Xb.shape[0])
        for b in range(Xb.shape[0]):
            out[b] = max_cos(Xb[b])
        return out

    def fingerprints(Xb):
        F = []
        for b in range(Xb.shape[0]):
            xb = Xb[b]
            g = xb @ xb.T
            mc = float(np.max(g - 2.0*np.eye(n)))
            # contact counts at mc-0.002
            gg = g.copy()
            np.fill_diagonal(gg, -2.0)
            cc = (gg > mc - 0.002).sum(axis=1)
            cc_s = np.sort(cc)
            # eigenvalues of true Gram
            ev = np.linalg.eigvalsh(g)
            ev = np.sort(ev)
            F.append((mc, cc_s, ev))
        return F

    # batched Adam over stages
    for stage in range(4):
        if time.time() > DEADLINE - 55:
            break
        beta = betas[stage]
        lr = lrs[stage]
        t_end = time.time() + stage_time[stage]
        # decay lr slightly within stage to lr_next
        lr_end = lrs[stage+1] if stage+1 < len(lrs) else lr*0.7
        it = 0
        while time.time() < t_end and time.time() < DEADLINE - 55:
            it += 1
            step_global += 1
            f = min(1.0, it / 800.0)
            cur_lr = lr * (1-f) + lr_end * f
            # batched Gram (B,n,n)
            G = np.matmul(X, np.transpose(X, (0, 2, 1)))
            # mask diag for softmax: set to very low
            # compute per-member max for stability
            # need to exclude diag: set diag -2 temporarily for max
            # efficient: get max over off-diag
            Gmax = np.empty(B)
            for b in range(B):
                # quick max: use masked
                gb = G[b]
                # fill diag -2 in copy? avoid copy: compute max with diag correction
                m = float(np.max(gb - 2.0*np.eye(n)))
                Gmax[b] = m
            # weights
            # W = exp(beta*(G - Gmax)); zero diag
            # to save memory do in place per batch? do vectorized
            W = np.exp(beta * (G - Gmax[:, None, None]))
            # zero diag
            for b in range(B):
                np.fill_diagonal(W[b], 0.0)
            ssum = W.sum(axis=(1, 2), keepdims=True) + 1e-300
            W /= ssum
            Grad = np.matmul(W, X)  # (B,n,d)
            dot = np.sum(Grad * X, axis=2, keepdims=True)
            Gt = Grad - dot * X
            # Adam
            M = b1 * M + (1 - b1) * Gt
            Vc = b2 * Vc + (1 - b2) * (Gt * Gt)
            # bias correction
            bc1 = 1 - b1 ** step_global
            bc2 = 1 - b2 ** step_global
            Mh = M / bc1
            Vh = Vc / bc2
            X = X - cur_lr * Mh / (np.sqrt(Vh) + eps)
            X /= np.linalg.norm(X, axis=2, keepdims=True)
            # occasional NaN guard
            if step_global % 200 == 0:
                if not np.all(np.finite(X)):
                    X[~np.isfinite(X)] = 0.01
                    X /= np.linalg.norm(X, axis=2, keepdims=True)
        # checkpoint every ~10s+ (here every stage ~17s): fingerprint + crossover
        if stage < 3:
            if time.time() > DEADLINE - 55:
                break
            try:
                tmc = true_maxima(X)
            except Exception:
                continue
            order = np.argsort(tmc)
            try:
                F = fingerprints(X)
            except Exception:
                F = None
            # find duplicates among all
            keep = []
            dup_idx = set()
            if F is not None:
                seen = []
                for bi in order:
                    mc_b, cc_b, ev_b = F[bi]
                    is_dup = False
                    for (mc2, cc2, ev2) in seen:
                        if np.array_equal(cc_b, cc2) and np.linalg.norm(ev_b - ev2) < 1e-3:
                            is_dup = True
                            break
                    if is_dup:
                        dup_idx.add(int(bi))
                    else:
                        seen.append((mc_b, cc_b, ev_b))
                        keep.append(int(bi))
            else:
                keep = list(order)
            # crossover: replace worst + duplicates with children of elites
            elites = [int(o) for o in order[:8]]
            # victims: worst 8 + duplicates (union)
            victims = set([int(o) for o in order[-8:]] + list(dup_idx))
            # don't kill top 3
            top3 = set([int(o) for o in order[:3]])
            victims = [v for v in victims if v not in top3]
            for v in victims:
                if time.time() > DEADLINE - 55:
                    break
                # pick two distinct elites
                a = elites[int(rng.integers(0, len(elites)))]
                b_ = elites[int(rng.integers(0, len(elites)))]
                if b_ == a:
                    b_ = elites[int(rng.integers(0, len(elites)))]
                Xa = X[a]
                Xb_ = X[b_]
                # contact info
                try:
                    mc_a = max_cos(Xa)
                    cc_a = contact_counts_thr(Xa, mc_a, 0.002)
                    mc_b = max_cos(Xb_)
                    cc_b = contact_counts_thr(Xb_, mc_b, 0.002)
                    ppm_b = per_point_max(Xb_)
                except Exception:
                    continue
                m = int(rng.integers(15, 26))  # 15-25
                donor_low = np.argsort(cc_a)[:m]
                recip_worst = np.argsort(-ppm_b)[:m]
                child = Xb_.copy()
                child[recip_worst] = Xa[donor_low]
                # small jitter to avoid exact clash
                child[recip_worst] += 0.005 * rng.standard_normal((m, d))
                child /= np.linalg.norm(child, axis=1, keepdims=True)
                X[v] = child
                M[v] = 0
                Vc[v] = 0
            # mild mutation for middle members to keep diversity
            for bi in order[8:16]:
                if rng.random() < 0.4:
                    sig = 0.02
                    X[bi] += sig * rng.standard_normal((n, d))
                    X[bi] /= np.linalg.norm(X[bi], axis=1, keepdims=True)
                    M[bi] = 0
                    Vc[bi] = 0

    # evaluate final population
    try:
        finals = true_maxima(X)
    except Exception:
        finals = np.full((B,), 1e9)
    order = np.argsort(finals)
    # fingerprints for distinctness
    try:
        F = fingerprints(X)
    except Exception:
        F = None
    distinct = []
    for bi in order:
        ok = True
        if F is not None:
            for dj in distinct:
                if np.array_equal(F[bi][1], F[dj][1]) and np.linalg.norm(F[bi][2]-F[dj][2]) < 1e-3:
                    ok = False
                    break
        if ok:
            distinct.append(int(bi))
        if len(distinct) >= 6:
            break
    if len(distinct) == 0:
        distinct = [int(o) for o in order[:3]]
    top3 = distinct[:3]

    # LP polish top three mutually distinct elites ~15s each (adaptive to time)
    polished = []
    for k, bi in enumerate(top3):
        remain = DEADLINE - time.time() - 8.0
        if remain < 3:
            polished.append((X[bi].copy(), float(finals[bi])))
            continue
        sec = min(15.0, max(4.0, remain / (len(top3)-k+1)))
        # use smaller radius for best, larger for others to allow basin move
        try:
            xp = lp_polish(X[bi].copy(), sec, radius_init=0.002 if k == 0 else 0.003)
            v = max_cos(xp)
        except Exception:
            xp = X[bi].copy()
            v = float(finals[bi])
        polished.append((xp, float(v)))
    # winner
    polished.sort(key=lambda t: t[1])
    winner, wval = polished[0]
    # long polish winner with remaining time
    rem = DEADLINE - time.time() - 1.5
    if rem > 3:
        try:
            s1 = min(10.0, rem*0.5)
            winner = lp_polish(winner, s1, radius_init=0.002)
            rem2 = DEADLINE - time.time() - 1.0
            if rem2 > 2:
                winner = lp_polish(winner, rem2, radius_init=6e-4)
            wval = max_cos(winner)
        except Exception:
            pass
    winner = winner / np.linalg.norm(winner, axis=1, keepdims=True)
    # fallback: only return if beats best.npy
    try:
        xb = np.load("best.npy").astype(float)
        xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
        bv = max_cos(xb)
    except Exception:
        xb = None
        bv = 1e9
    if wval < bv - 1e-12:
        return winner
    else:
        # return best (polished copy) to guarantee no regression
        if xb is not None and xb.shape == (95, 13):
            return xb
        return winner
