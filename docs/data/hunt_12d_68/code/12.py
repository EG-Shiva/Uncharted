import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))


def soft_descent(x, steps, beta0, beta1, lr0, lr1, mu=0.85, noise=0.0, rng=None, power_mode=False, p0=30, p1=120):
    n, d = x.shape
    v = np.zeros_like(x)
    for s in range(steps):
        t = s / max(steps - 1, 1)
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = np.max(G)
        if (not power_mode):
            beta = beta0 * (beta1 / max(beta0, 1e-12)) ** t
            # stabilize
            W = np.exp(beta * (G - m))
            np.fill_diagonal(W, 0.0)
        else:
            p = p0 + (p1 - p0) * t
            W = np.clip(G, 0.0, 2.0) ** p
            np.fill_diagonal(W, 0.0)
            # mix a bit of exp to keep negative pairs weakly repulsive early
            if t < 0.3:
                beta = beta0
                W = W + 0.05 * np.exp(beta * (G - m)) * (G > -1)
                np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        Tg = Tg / rms
        v = mu * v + (1.0 - mu) * Tg if s > 0 else Tg
        x = x - lr * v
        if noise > 0 and rng is not None:
            amp = noise * (1.0 - t)
            x = x + amp * rng.standard_normal(x.shape)
        nrm = np.linalg.norm(x, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        x = x / nrm
    return x


def hard_descent(x, steps, lr0, lr1, topk=250, rng=None, noise=0.0):
    n, d = x.shape
    v = np.zeros_like(x)
    mu = 0.7
    for s in range(steps):
        t = s / max(steps - 1, 1)
        lr = lr0 + (lr1 - lr0) * t
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        # threshold adapts: keep topk pairs
        flat = G[np.triu_indices(n, 1)]
        kth = np.sort(flat)[-min(topk, flat.size)]
        m = flat.max()
        # active band shrinks
        eps = 0.005 * (1 - t) + 0.0002
        thr = max(kth, m - eps)
        W = np.where(G >= thr, 1.0, 0.0)
        np.fill_diagonal(W, 0.0)
        Grad = W @ x
        proj = np.sum(Grad * x, axis=1, keepdims=True)
        Tg = Grad - proj * x
        rms = np.sqrt(np.mean(Tg * Tg)) + 1e-12
        Tg = Tg / rms
        v = mu * v + (1 - mu) * Tg if s > 0 else Tg
        x = x - lr * v
        if noise > 0 and rng is not None:
            amp = noise * (1.0 - t)
            x = x + amp * rng.standard_normal(x.shape)
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x


def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        if time.time() - start > seconds - 0.25:
            break
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        if p == 0:
            radius *= 0.5
            continue
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
        vv = max_cos(y)
        if vv < best:
            x, best = y, vv
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def best_hole(x_rest, rng, ncand=8000):
    d = x_rest.shape[1]
    cand = rng.standard_normal((ncand, d))
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    scores = (cand @ x_rest.T).max(axis=1)
    return cand[int(np.argmin(scores))]


def solve():
    t0 = time.time()
    TIME_LIMIT = 172.0
    rng = np.random.default_rng(987654321)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    cands = []
    if xb is not None:
        cands.append(xb)
    if xr is not None:
        cands.append(xr)
    if len(cands) == 0:
        xb0 = rng.standard_normal((68, 12))
        xb0 /= np.linalg.norm(xb0, axis=1, keepdims=True)
    else:
        bestv = 1e9
        xb0 = cands[0]
        for c in cands:
            c = c / np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
            vv = max_cos(c)
            if vv < bestv:
                bestv = vv
                xb0 = c.copy()
    xb0 = xb0 / np.linalg.norm(xb0, axis=1, keepdims=True)
    best_val = max_cos(xb0)
    x_best = xb0.copy()

    n, d = x_best.shape
    # archive of good crude codes
    archive = [(best_val, x_best.copy())]

    def push_archive(v, x):
        # keep up to 6, distinct-ish
        archive.append((v, x.copy()))
        archive.sort(key=lambda z: z[0])
        # prune to 6
        del archive[6:]

    explore_end = t0 + TIME_LIMIT - 30.0
    trial = 0
    # quick crude polish of start to rank fairly (already polished)
    while time.time() < explore_end:
        trial += 1
        u = rng.random()
        # choose base: mostly best, sometimes 2nd best
        if len(archive) > 1 and rng.random() < 0.2:
            base = archive[1][1]
        else:
            base = x_best
        x = base.copy()

        if u < 0.12:
            # cold random restart (far basin)
            x = rng.standard_normal((n, d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps1 = int(rng.integers(500, 900))
            beta1 = float(rng.uniform(300, 550))
            x = soft_descent(x, steps=steps1, beta0=15.0, beta1=beta1,
                             lr0=0.06, lr1=0.004, mu=float(rng.uniform(0.5, 0.9)),
                             noise=0.002, rng=rng, power_mode=(rng.random() < 0.3))
        elif u < 0.30:
            # global shake various sigma
            sigma = float(rng.choice([0.005, 0.015, 0.04, 0.09]))
            x = x + sigma * rng.standard_normal(x.shape)
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            steps1 = int(rng.integers(500, 850))
            pw = rng.random() < 0.35
            x = soft_descent(x, steps=steps1, beta0=float(rng.uniform(20, 80)),
                             beta1=float(rng.uniform(320, 600)),
                             lr0=float(rng.uniform(0.03, 0.06)), lr1=float(rng.uniform(0.002, 0.006)),
                             mu=float(rng.uniform(0.6, 0.92)), noise=0.001 if rng.random() < 0.5 else 0.0,
                             rng=rng, power_mode=pw,
                             p0=int(rng.integers(20, 50)), p1=int(rng.integers(80, 160)))
        elif u < 0.52:
            # worst-cluster kick: kick k points around worst pair
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            # worst pair
            ij = int(np.argmax(G.ravel()))
            wi, wj = ij // n, ij % n
            # scores: max dot to wi/wj or overload
            per = G.max(axis=1)
            # neighborhood: points close to wi or wj
            prox = np.maximum(G[wi], G[wj])
            order = np.argsort(-prox)
            k = int(rng.integers(3, 9))
            idx = order[:k]
            # also add some of global worst with prob
            if rng.random() < 0.5:
                worder = np.argsort(-per)
                for q in worder[:2]:
                    if q not in idx:
                        idx = np.append(idx, q)
                        break
            if rng.random() < 0.55:
                for q in idx:
                    vv = rng.standard_normal(d)
                    vv /= np.linalg.norm(vv)
                    x[int(q)] = vv
            else:
                x[idx] += float(rng.uniform(0.3, 0.8)) * rng.standard_normal((len(idx), d))
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = soft_descent(x, steps=int(rng.integers(600, 900)), beta0=40.0,
                             beta1=float(rng.uniform(350, 600)),
                             lr0=0.04, lr1=0.003, mu=0.85, noise=0.001, rng=rng,
                             power_mode=(rng.random() < 0.3))
        elif u < 0.72:
            # cap surgery: pick random direction, take cap members
            ud = rng.standard_normal(d)
            ud /= np.linalg.norm(ud)
            dots = x @ ud
            k = int(rng.integers(4, 11))
            idx = np.argsort(-dots)[:k]
            # re-randomize cap: uniform in cap around ud with varying spread
            if rng.random() < 0.5:
                for q in idx:
                    vv = rng.standard_normal(d)
                    vv /= np.linalg.norm(vv)
                    # bias toward cap centre partially
                    vv = vv + float(rng.uniform(0.0, 1.5)) * ud
                    vv /= np.linalg.norm(vv)
                    x[int(q)] = vv
            else:
                # rotate / reflect cap: apply random orthogonal-ish jitter + shuffle
                sub = x[idx].copy() + 0.4 * rng.standard_normal((len(idx), d))
                sub /= np.linalg.norm(sub, axis=1, keepdims=True)
                # shuffle assignment
                perm = rng.permutation(len(idx))
                x[idx] = sub[perm]
                x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = soft_descent(x, steps=int(rng.integers(600, 900)), beta0=30.0,
                             beta1=float(rng.uniform(350, 600)),
                             lr0=0.045, lr1=0.003, mu=0.85, noise=0.001, rng=rng,
                             power_mode=(rng.random() < 0.3))
        else:
            # delete worst 1-2, reinsert at best holes
            mdel = 1 if rng.random() < 0.7 else 2
            G = x @ x.T
            np.fill_diagonal(G, -2.0)
            per = G.max(axis=1)
            order = np.argsort(-per)
            for tdel in range(mdel):
                wi = int(order[tdel]) if tdel < len(order) else int(rng.integers(0, n))
                mask = np.ones(n, dtype=bool)
                mask[wi] = False
                hole = best_hole(x[mask], rng, ncand=8000)
                x[wi] = hole
            # shake neighbours of holes
            G2 = x @ x.T
            for tdel in range(mdel):
                wi = int(order[tdel])
                nbrs = np.argsort(-G2[wi])[:5]
                x[nbrs] += 0.03 * rng.standard_normal((len(nbrs), d))
            x /= np.linalg.norm(x, axis=1, keepdims=True)
            x = soft_descent(x, steps=int(rng.integers(600, 900)), beta0=40.0,
                             beta1=float(rng.uniform(350, 600)),
                             lr0=0.04, lr1=0.0025, mu=0.85, noise=0.0, rng=rng,
                             power_mode=(rng.random() < 0.25))

        # hard fine tune (short)
        if rng.random() < 0.7:
            x = hard_descent(x, steps=int(rng.integers(120, 300)),
                             lr0=float(rng.uniform(0.008, 0.02)),
                             lr1=float(rng.uniform(0.001, 0.003)),
                             topk=int(rng.integers(150, 400)), rng=rng, noise=0.0)
        else:
            x = soft_descent(x, steps=200, beta0=400.0, beta1=650.0,
                             lr0=0.008, lr1=0.001, mu=0.8, rng=rng)

        vv = max_cos(x)
        if vv < best_val - 1e-12:
            x_best = x.copy()
            best_val = vv
            push_archive(vv, x)
        elif vv < best_val + 0.004:
            # keep diverse near-best in archive
            # avoid duplicates: only if better than worst archive or archive small
            if len(archive) < 6 or vv < archive[-1][0]:
                # check distinctness: min assignment-free distance (sorted gram) to avoid clones
                dup = False
                for (av, ax) in archive:
                    if abs(av - vv) < 1e-9:
                        # compare point sets roughly: mean abs diff after sorting? cheap: fro of sorted dots
                        # just skip if very close value and high correlation
                        pass
                if not dup:
                    push_archive(vv, x)
        if trial > 3000:
            break
        if time.time() >= explore_end:
            break

    # final stage: LP polish top archive members, pick best
    archive.sort(key=lambda z: z[0])
    # ensure x_best first
    cand_list = []
    seen = set()
    cand_list.append(x_best)
    for (_, ax) in archive:
        if len(cand_list) >= 4:
            break
        # skip near-duplicate of already listed (check max diff after alignment-free? use value diff)
        # just add if not identical object
        is_same = False
        for c in cand_list:
            if np.linalg.norm(c - ax) < 1e-9:
                is_same = True
                break
        if not is_same:
            cand_list.append(ax.copy())
    remain = TIME_LIMIT - (time.time() - t0)
    # allocate: most to best, some to others
    if remain < 4:
        x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
        return x_best
    per_cand = max(3.0, (remain - 1.5) / len(cand_list))
    # polish best last with largest budget? polish all then pick
    polished = []
    for i, c in enumerate(cand_list):
        budget = per_cand
        # give best extra: last gets remainder
        if i == 0:
            budget = min(per_cand * 1.5, remain - 3 * (len(cand_list) - 1) - 1.0)
            budget = max(budget, 3.0)
        y = lp_polish(c, budget)
        polished.append((max_cos(y), y))
        remain = TIME_LIMIT - (time.time() - t0)
        if remain < 3.0:
            break
    polished.sort(key=lambda z: z[0])
    x_best = polished[0][1]
    # extra squeeze on winner if time left
    remain = TIME_LIMIT - (time.time() - t0)
    if remain > 4.0:
        x_best = lp_polish(x_best, remain - 1.0)
    x_best = x_best / np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
