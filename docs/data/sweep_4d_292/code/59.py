import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def contact_info(x, tol=1e-7):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    m = float(np.max(g))
    contacts = np.sum(g > m - tol, axis=1)
    # per-point worst neighbour
    worst = np.max(g, axis=1)
    # stress ~ sum of near-max excess
    thr = m - 2e-3
    stress = np.sum(np.where(g > thr, g - thr, 0.0), axis=1)
    return m, contacts, worst, stress, g


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        xxT = x @ x.T
        c = xxT[iu]
        m = float(np.max(c))
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cc = np.arange(d)
        cols = np.concatenate(
            [I[:, None] * d + cc, J[:, None] * d + cc,
             np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix(
                (x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))),
                shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            tl = max(0.3, seconds - (time.time() - start))
            try:
                res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs", options={"time_limit": tl})
            except Exception:
                res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                              b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                              method="highs")
        except Exception:
            radius *= 0.5
            continue
        if getattr(res, "status", 1) != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
        except Exception:
            radius *= 0.5
            continue
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(y)):
            radius *= 0.5
            continue
        v = max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def smooth_adam(Y0, steps, tau0, tau1, lr):
    Y = Y0 / np.linalg.norm(Y0, axis=1, keepdims=True)
    m = np.zeros_like(Y)
    vv = np.zeros_like(Y)
    b1, b2, eps = 0.9, 0.999, 1e-8
    b1p, b2p = 1.0, 1.0
    for t in range(1, steps + 1):
        frac = t / steps
        tau = tau0 * (tau1 / tau0) ** frac
        lrt = lr * (1.0 - 0.7 * frac)
        G = Y @ Y.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = (G - M) / tau
        mask = D > -15.0
        W = np.zeros_like(G)
        W[mask] = np.exp(D[mask])
        s = float(np.sum(W))
        if s <= 0 or not np.isfinite(s):
            break
        W /= s
        grad = 2.0 * (W @ Y)
        r = np.sum(grad * Y, axis=1, keepdims=True)
        g = grad - r * Y
        b1p *= b1
        b2p *= b2
        m = b1 * m + (1.0 - b1) * g
        vv = b2 * vv + (1.0 - b2) * (g * g)
        mh = m / (1.0 - b1p)
        vh = vv / (1.0 - b2p)
        upd = mh / (np.sqrt(vh) + eps)
        un = np.linalg.norm(upd, axis=1, keepdims=True) + 1e-12
        clip = np.minimum(1.0, 3.0 / un)
        upd = upd * clip
        Y = Y - lrt * upd
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Y /= nrm
    return Y


def find_holes(fixed, k, rng, ns=16000, sep=0.90):
    d = fixed.shape[1]
    cand = rng.randn(ns, d)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    best_dot = np.empty(ns)
    chunk = 4000
    for s in range(0, ns, chunk):
        e = min(ns, s + chunk)
        best_dot[s:e] = np.max(cand[s:e] @ fixed.T, axis=1)
    order = np.argsort(best_dot)
    chosen = []
    for idx in order:
        c = cand[idx]
        ok = True
        for j in chosen:
            if float(c @ cand[j]) > sep:
                ok = False
                break
        if ok:
            chosen.append(idx)
            if len(chosen) >= k:
                break
    if len(chosen) < k:
        # fill remaining greedily ignoring sep
        for idx in order:
            if idx not in chosen:
                chosen.append(idx)
                if len(chosen) >= k:
                    break
    return cand[np.array(chosen)]


def slack_signature(x, k=1200):
    g = x @ x.T
    iu = np.triu_indices(x.shape[0], 1)
    c = (g[iu]).copy()
    c.sort()
    # largest k values -> slacks
    m = c[-1]
    tail = m - c[-k:]
    # round for hashing
    return tail


def solve():
    np.random.seed(7)
    rng = np.random.RandomState(7)
    t0 = time.time()
    BUDGET = 175.0
    try:
        xb = np.load("best.npy")
    except Exception:
        xb = np.load("record.npy")
    xb = np.asarray(xb, dtype=np.float64).reshape(292, 4)
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    best_val = max_cos(xb)
    x_best = xb.copy()

    seen_sigs = set()

    def sig_key(x):
        s = slack_signature(x, 800)
        # coarse quantisation
        q = tuple(np.round(s * 1e7).astype(np.int64).tolist()[::40])
        return q

    # ladder schedule: (direction, k)
    # direction 'down' = delete k then regrow; 'up' = insert k then delete k (different point)
    schedule = []
    for k in [1, 2, 3, 4, 6, 8, 12]:
        schedule.append(("down", k))
    for k in [1, 2, 4, 6, 8]:
        schedule.append(("up", k))
    # repeat with different variants
    cyc = 0
    # quick initial polish signature
    try:
        seen_sigs.add(sig_key(x_best))
    except Exception:
        pass

    while time.time() - t0 < 148.0:
        mode, k = schedule[cyc % len(schedule)]
        variant = (cyc // len(schedule)) % 4
        cyc += 1
        remain = BUDGET - (time.time() - t0)
        if remain < 12:
            break
        try:
            if mode == "down":
                # --- choose k points to delete from champion ---
                m0, contacts, worst, stress, _ = contact_info(x_best)
                # several deletion orders
                if variant == 0:
                    score = -contacts.astype(float) * 10.0 - worst + 1e-6 * rng.randn(292)
                    # highest contacts first -> smallest score? we want delete hubs
                    order = np.argsort(score)  # most negative = most contacts
                    dell = order[:k]
                elif variant == 1:
                    score = -(stress) + 1e-9 * rng.randn(292)
                    order = np.argsort(score)
                    dell = order[:k]
                elif variant == 2:
                    # mix: hubs + one random rattler/low
                    o1 = np.argsort(-contacts + 1e-6 * rng.randn(292))
                    # force include a low-contact point sometimes
                    lows = np.where(contacts <= 2)[0]
                    dell = list(o1[:max(0, k - 1)])
                    if len(lows) > 0 and k >= 2:
                        dell.append(lows[rng.randint(len(lows))])
                    else:
                        dell.append(o1[k - 1])
                    dell = np.array(dell, dtype=int)
                else:
                    # random biased to hubs
                    o = np.argsort(-contacts)
                    pool = list(o[:24])
                    rng.shuffle(pool)
                    dell = np.array(pool[:k], dtype=int)
                dell = np.unique(dell)
                if len(dell) != k:
                    # pad
                    extra = [i for i in range(292) if i not in set(dell.tolist())]
                    rng.shuffle(extra)
                    while len(dell) < k:
                        dell = np.append(dell, extra.pop())
                mask = np.ones(292, dtype=bool)
                mask[dell] = False
                x_small = x_best[mask].copy()
                # small random shake to escape basin
                x_small = x_small + (0.002 + 0.001 * variant) * rng.randn(*x_small.shape)
                x_small /= np.linalg.norm(x_small, axis=1, keepdims=True)
                # relax smaller code
                x_small = smooth_adam(x_small, steps=450, tau0=0.016, tau1=0.0012, lr=0.0022)
                # quick LP polish small
                if remain > 8:
                    x_small = lp_polish(x_small, min(1.2, remain - 6))
                # rebuild to 292 via largest holes (different hole choices per variant)
                ns = 16000 if variant < 2 else 22000
                holes = find_holes(x_small, k, rng, ns=ns, sep=0.88 if variant == 3 else 0.90)
                x_re = np.vstack([x_small, holes])
                # shake slightly
                x_re = x_re + 0.0015 * rng.randn(*x_re.shape)
                x_re /= np.linalg.norm(x_re, axis=1, keepdims=True)
                x_re = smooth_adam(x_re, steps=550, tau0=0.014, tau1=0.0009, lr=0.0022)
                v = max_cos(x_re)
                # signature diversity: skip LP if same basin seen and not promising
                key = sig_key(x_re)
                if key in seen_sigs and v > best_val + 0.001:
                    continue
                seen_sigs.add(key)
                if v < best_val + 0.004:
                    r2 = BUDGET - (time.time() - t0)
                    if r2 < 4:
                        if v < best_val:
                            x_best, best_val = x_re, v
                        continue
                    x_re2 = lp_polish(x_re, min(2.5, r2 - 2.0))
                    v2 = max_cos(x_re2)
                    if v2 < best_val:
                        x_best, best_val = x_re2, v2
                else:
                    if v < best_val:
                        x_best, best_val = x_re, v
            else:  # up: overshoot then delete different point
                m0, contacts, worst, stress, _ = contact_info(x_best)
                ns = 16000 if variant < 2 else 22000
                holes = find_holes(x_best, k, rng, ns=ns, sep=0.88 if variant == 3 else 0.90)
                x_big = np.vstack([x_best, holes])
                x_big = x_big + 0.002 * rng.randn(*x_big.shape)
                x_big /= np.linalg.norm(x_big, axis=1, keepdims=True)
                x_big = smooth_adam(x_big, steps=450, tau0=0.016, tau1=0.0012, lr=0.0022)
                if remain > 8:
                    x_big = lp_polish(x_big, min(1.2, remain - 6))
                # delete k points DIFFERENT from just inserted: choose hubs among first 292
                mb, cb, wb, sb, _ = contact_info(x_big)
                nbig = x_big.shape[0]
                old_idx = np.arange(292)
                # rank old points by hub-ness with variant tie-break
                if variant == 0:
                    order = np.argsort(cb[:292] + 0.5 * wb[:292] + 1e-6 * rng.randn(292))
                    # delete highest -> last
                    dell_old = order[-k:]
                elif variant == 1:
                    order = np.argsort(sb[:292] + 1e-12 * rng.randn(292))
                    dell_old = order[-k:]
                elif variant == 2:
                    # delete one rattler-adjacent + hubs to force nontrivial cycle
                    lows = np.where(cb[:292] <= 2)[0]
                    o = np.argsort(-cb[:292] + 1e-6 * rng.randn(292))
                    dell_old = list(o[:max(0, k - 1)])
                    if len(lows) > 0 and k >= 1:
                        # delete a low point instead of a hub for one slot
                        cand_low = lows[rng.randint(len(lows))]
                        if cand_low not in dell_old:
                            dell_old.append(cand_low)
                        else:
                            dell_old.append(o[k - 1])
                    dell_old = np.array(dell_old[-k:], dtype=int)
                else:
                    pool = np.argsort(-cb[:292])[:24].tolist()
                    rng.shuffle(pool)
                    dell_old = np.array(pool[:k], dtype=int)
                mask = np.ones(nbig, dtype=bool)
                mask[dell_old] = False
                x_re = x_big[mask].copy()
                assert x_re.shape[0] == 292
                x_re = x_re + 0.0015 * rng.randn(*x_re.shape)
                x_re /= np.linalg.norm(x_re, axis=1, keepdims=True)
                x_re = smooth_adam(x_re, steps=550, tau0=0.014, tau1=0.0009, lr=0.0022)
                v = max_cos(x_re)
                key = sig_key(x_re)
                if key in seen_sigs and v > best_val + 0.001:
                    continue
                seen_sigs.add(key)
                if v < best_val + 0.004:
                    r2 = BUDGET - (time.time() - t0)
                    if r2 < 4:
                        if v < best_val:
                            x_best, best_val = x_re, v
                        continue
                    x_re2 = lp_polish(x_re, min(2.5, r2 - 2.0))
                    v2 = max_cos(x_re2)
                    if v2 < best_val:
                        x_best, best_val = x_re2, v2
                else:
                    if v < best_val:
                        x_best, best_val = x_re, v
        except Exception:
            continue

    remain = BUDGET - (time.time() - t0)
    if remain > 4:
        try:
            x_best = lp_polish(x_best, min(remain - 1.0, 18.0))
        except Exception:
            pass
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
