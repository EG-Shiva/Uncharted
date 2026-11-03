import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _lp_polish(X, budget):
    t0 = time.time()
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = _max_cos(x)
    except Exception:
        return x
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() - t0 < budget and radius > 1e-14:
        if time.time() - t0 >= budget:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        sel = c > m - max(20.0 * radius, 1e-9)
        I = iu[0][sel]
        J = iu[1][sel]
        p = len(I)
        if p == 0:
            radius *= 0.5
            continue
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d),
                                           np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv)
            obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq,
                          b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)],
                          method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        try:
            y = x + res.x[:-1].reshape(n, d)
            nrm = np.linalg.norm(y, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            y = y / nrm
            v = _max_cos(y)
        except Exception:
            radius *= 0.5
            continue
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def _smooth_polish(Y, steps, T0, T1, lr0, lr1, mu=0.65):
    n, d = Y.shape
    V = np.zeros_like(Y)
    lT0 = np.log(T0); lT1 = np.log(T1)
    ll0 = np.log(lr0); ll1 = np.log(lr1)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        T = float(np.exp(lT0 + (lT1 - lT0) * frac))
        lr = float(np.exp(ll0 + (ll1 - ll0) * frac))
        G = Y @ Y.T
        np.fill_diagonal(G, -1.0)
        m = float(np.max(G))
        W = (G - m) / T
        np.clip(W, -50.0, 0.0, out=W)
        np.exp(W, out=W)
        np.fill_diagonal(W, 0.0)
        rs = np.sum(W, axis=1)
        WY = W @ Y
        r = np.sum(W * G, axis=1)
        inv = np.zeros_like(rs)
        mk = rs > 1e-12
        inv[mk] = 1.0 / rs[mk]
        avgWY = WY * inv[:, None]
        avgR = r * inv
        Dir = avgWY - avgR[:, None] * Y
        Dir[~mk] = 0.0
        V *= mu
        V += Dir
        Y -= lr * V
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Y /= nrm
    return Y

def _best_void(Xkeep, rng, n_cand=3500):
    d = Xkeep.shape[1]
    cand = rng.standard_normal((n_cand, d))
    cn = np.linalg.norm(cand, axis=1, keepdims=True)
    cn[cn == 0] = 1.0
    cand /= cn
    dots = cand @ Xkeep.T
    vals = np.max(dots, axis=1)
    bi = int(np.argmin(vals))
    v = cand[bi].copy()
    # local refine: push away from nearest neighbours
    for _ in range(40):
        dv = Xkeep @ v  # (m,)
        m = float(np.max(dv))
        w = np.exp(np.clip((dv - m) / 0.02, -50, 0))
        g = w @ Xkeep  # (d,)
        g = g - np.dot(g, v) * v
        ng = np.linalg.norm(g)
        if ng < 1e-12:
            break
        g /= ng
        step = 0.02
        v = v - step * g
        nv = np.linalg.norm(v)
        if nv == 0:
            break
        v /= nv
    return v / np.linalg.norm(v)

def solve():
    t_start = time.time()
    rng = np.random.default_rng(0)
    np.random.seed(0)
    n, d = 181, 9
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                nrm = np.linalg.norm(a, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                a = a / nrm
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal((n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    best = None
    best_val = np.inf
    for c in cands:
        v = _max_cos(c)
        if v < best_val:
            best_val = v
            best = c.copy()
    # elite archive: list of [val, X]
    elite = [[best_val, best.copy()]]
    # also add perturbed record if available for diversity
    for c in cands[1:]:
        v = _max_cos(c)
        if v < elite[-1][0] + 0.01 and len(elite) < 5:
            # avoid duplicate
            if abs(v - elite[0][0]) > 1e-9:
                elite.append([v, c.copy()])
    elite.sort(key=lambda e: e[0])

    EXPLORE_END = 136.0
    cycle = 0
    while time.time() - t_start < EXPLORE_END and cycle < 1000:
        cycle += 1
        remain = EXPLORE_END - (time.time() - t_start)
        if remain < 1.0:
            break
        # pick base from elite
        rpick = float(rng.random())
        if rpick < 0.65 or len(elite) == 1:
            base = elite[0][1]
        else:
            bi = int(rng.integers(len(elite)))
            base = elite[bi][1]
        cur = base.copy()
        # diagnostics on base
        try:
            Gt = cur @ cur.T
            np.fill_diagonal(Gt, -2.0)
            vmax = float(np.max(Gt))
            row_max = np.max(Gt, axis=1)
            cnt = np.sum(Gt > vmax - 0.01, axis=1)
            order_worst = np.argsort(-row_max)
            order_cnt = np.argsort(-cnt)
        except Exception:
            order_worst = np.arange(n)
            order_cnt = np.arange(n)
            cnt = np.zeros(n)
        u = float(rng.random())
        # default anneal params
        if u < 0.22:
            # small global exploit
            sigma = float(rng.uniform(0.006, 0.02))
            noise = rng.standard_normal((n, d)) * sigma
            proj = np.sum(noise * cur, axis=1, keepdims=True)
            noise -= proj * cur
            cur += noise
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            T0 = float(rng.uniform(0.012, 0.025)); T1 = float(rng.uniform(0.0008, 0.003))
            lr0 = float(rng.uniform(0.03, 0.07)); lr1 = float(rng.uniform(0.003, 0.01))
            steps = int(rng.integers(450, 700)); mu = float(rng.uniform(0.5, 0.75))
        elif u < 0.42:
            # worst cluster shake
            K = int(rng.integers(6, 17))
            idx = order_worst[:K]
            sigma = float(rng.uniform(0.05, 0.13))
            noise = rng.standard_normal((K, d)) * sigma
            proj = np.sum(noise * cur[idx], axis=1, keepdims=True)
            noise -= proj * cur[idx]
            cur[idx] += noise
            cur[idx] /= np.linalg.norm(cur[idx], axis=1, keepdims=True)
            T0 = float(rng.uniform(0.014, 0.03)); T1 = float(rng.uniform(0.0008, 0.0025))
            lr0 = float(rng.uniform(0.04, 0.08)); lr1 = float(rng.uniform(0.003, 0.008))
            steps = int(rng.integers(500, 800)); mu = float(rng.uniform(0.55, 0.8))
        elif u < 0.58:
            # random subset shake
            K = int(rng.integers(10, 30))
            idx = rng.choice(n, size=K, replace=False)
            sigma = float(rng.uniform(0.03, 0.08))
            noise = rng.standard_normal((K, d)) * sigma
            proj = np.sum(noise * cur[idx], axis=1, keepdims=True)
            noise -= proj * cur[idx]
            cur[idx] += noise
            cur[idx] /= np.linalg.norm(cur[idx], axis=1, keepdims=True)
            T0 = float(rng.uniform(0.01, 0.022)); T1 = float(rng.uniform(0.001, 0.0035))
            lr0 = float(rng.uniform(0.03, 0.065)); lr1 = float(rng.uniform(0.003, 0.01))
            steps = int(rng.integers(450, 700)); mu = float(rng.uniform(0.5, 0.75))
        elif u < 0.78:
            # void teleport surgery: move 1-3 loose/overloaded points to largest holes
            nrep = int(rng.integers(1, 4))
            # candidates to move: mix of worst and low-contact
            # low contact = smallest cnt
            try:
                order_loose = np.argsort(cnt + rng.random(n) * 0.5)
            except Exception:
                order_loose = rng.choice(n, size=n, replace=False)
            cand_move = []
            for k in range(nrep):
                if rng.random() < 0.6:
                    cand_move.append(int(order_worst[k % n]))
                else:
                    cand_move.append(int(order_loose[k % n]))
            # avoid duplicates
            cand_move = list(dict.fromkeys(cand_move))
            # sequential void fill
            tmp = cur.copy()
            for j in cand_move:
                mask = np.ones(n, dtype=bool); mask[j] = False
                try:
                    v = _best_void(tmp[mask], rng, n_cand=3000)
                    tmp[j] = v
                except Exception:
                    vv = rng.standard_normal((d,)); vv /= np.linalg.norm(vv)
                    tmp[j] = vv
            # small jitter of neighbours to unlock
            try:
                Gj = tmp @ tmp.T
                np.fill_diagonal(Gj, -2.0)
                for j in cand_move:
                    # neighbours close to j
                    nbr = np.argsort(-Gj[j])[:6]
                    nj = rng.standard_normal((len(nbr), d)) * 0.015
                    pr = np.sum(nj * tmp[nbr], axis=1, keepdims=True)
                    nj -= pr * tmp[nbr]
                    tmp[nbr] += nj
                    tmp[nbr] /= np.linalg.norm(tmp[nbr], axis=1, keepdims=True)
            except Exception:
                pass
            cur = tmp / np.linalg.norm(tmp, axis=1, keepdims=True)
            T0 = float(rng.uniform(0.014, 0.03)); T1 = float(rng.uniform(0.0008, 0.0028))
            lr0 = float(rng.uniform(0.04, 0.075)); lr1 = float(rng.uniform(0.003, 0.008))
            steps = int(rng.integers(550, 850)); mu = float(rng.uniform(0.55, 0.8))
        elif u < 0.90:
            # overloaded hub removal: take top-contact points, reinsert far
            K = int(rng.integers(2, 5))
            idx = order_cnt[:K]
            # pick 2 to teleport
            nrep = min(K, int(rng.integers(2, 4)))
            mv = list(idx[:nrep])
            tmp = cur.copy()
            for j in mv:
                mask = np.ones(n, dtype=bool); mask[j] = False
                try:
                    v = _best_void(tmp[mask], rng, n_cand=3500)
                    tmp[j] = v
                except Exception:
                    vv = rng.standard_normal((d,)); vv /= np.linalg.norm(vv)
                    tmp[j] = vv
            # shake remaining hub members
            rest = [int(x) for x in idx if x not in mv]
            if rest:
                rr = np.array(rest)
                nj = rng.standard_normal((len(rr), d)) * float(rng.uniform(0.03, 0.07))
                pr = np.sum(nj * tmp[rr], axis=1, keepdims=True)
                nj -= pr * tmp[rr]
                tmp[rr] += nj
                tmp[rr] /= np.linalg.norm(tmp[rr], axis=1, keepdims=True)
            cur = tmp / np.linalg.norm(tmp, axis=1, keepdims=True)
            T0 = float(rng.uniform(0.015, 0.032)); T1 = float(rng.uniform(0.0008, 0.003))
            lr0 = float(rng.uniform(0.04, 0.08)); lr1 = float(rng.uniform(0.003, 0.009))
            steps = int(rng.integers(550, 850)); mu = float(rng.uniform(0.55, 0.85))
        else:
            # big global jump
            sigma = float(rng.uniform(0.09, 0.18))
            noise = rng.standard_normal((n, d)) * sigma
            proj = np.sum(noise * cur, axis=1, keepdims=True)
            noise -= proj * cur
            cur += noise
            cur /= np.linalg.norm(cur, axis=1, keepdims=True)
            T0 = float(rng.uniform(0.02, 0.035)); T1 = float(rng.uniform(0.001, 0.004))
            lr0 = float(rng.uniform(0.05, 0.09)); lr1 = float(rng.uniform(0.004, 0.012))
            steps = int(rng.integers(550, 850)); mu = 0.6
        try:
            polished = _smooth_polish(cur, steps, T0, T1, lr0, lr1, mu=mu)
            v = _max_cos(polished)
        except Exception:
            continue
        if v < elite[0][0] - 1e-12:
            # true improvement: short LP squeeze
            tnow = time.time() - t_start
            if tnow < EXPLORE_END - 2.0:
                lb = min(2.5, max(0.7, (EXPLORE_END - tnow) * 0.12))
                try:
                    refined = _lp_polish(polished, lb)
                    vr = _max_cos(refined)
                except Exception:
                    refined, vr = polished, v
            else:
                refined, vr = polished, v
            if vr < elite[0][0]:
                elite[0] = [vr, refined.copy()]
                best_val = vr
                best = refined.copy()
                elite.sort(key=lambda e: e[0])
            elif v < elite[0][0]:
                elite[0] = [v, polished.copy()]
                best_val = v
                best = polished.copy()
                elite.sort(key=lambda e: e[0])
            # keep elite size small; also add pre-LP if diverse?
        elif v < elite[-1][0] - 1e-9 and len(elite) < 5:
            # add diverse near-best to archive
            elite.append([v, polished.copy()])
            elite.sort(key=lambda e: e[0])
            if len(elite) > 5:
                elite = elite[:5]
        elif v < elite[0][0] + 0.0006 and (cycle % 9 == 0):
            if time.time() - t_start > EXPLORE_END - 5:
                continue
            try:
                refined = _lp_polish(polished, 1.0)
                vr = _max_cos(refined)
                if vr < elite[0][0] - 1e-12:
                    elite[0] = [vr, refined.copy()]
                    best_val = vr
                    best = refined.copy()
                    elite.sort(key=lambda e: e[0])
                elif vr < elite[-1][0] - 1e-9 and len(elite) < 5:
                    elite.append([vr, refined.copy()])
                    elite.sort(key=lambda e: e[0])
            except Exception:
                pass
    # final squeeze with remaining time
    try:
        elapsed = time.time() - t_start
        remain = 175.0 - elapsed
        if remain > 3.0:
            # quick low-T smooth on best before LP
            try:
                if remain > 12:
                    best = _smooth_polish(best.copy(), 300, 0.008, 0.0006, 0.02, 0.003, mu=0.6)
            except Exception:
                pass
            elapsed2 = time.time() - t_start
            remain2 = 175.0 - elapsed2
            if remain2 > 2.0:
                budget = min(30.0, remain2 - 1.5)
                best = _lp_polish(best, budget)
                bv = _max_cos(best)
                # try second elite as alternative
                if len(elite) > 1 and remain2 > 18:
                    try:
                        alt = _lp_polish(elite[1][1], min(6.0, remain2 - budget - 1.0))
                        av = _max_cos(alt)
                        if av < bv:
                            best = alt
                    except Exception:
                        pass
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
