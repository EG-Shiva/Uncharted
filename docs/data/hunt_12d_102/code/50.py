import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def adam_quench(x_init, steps, T0, T1, lr):
    x = x_init.copy()
    n, d = x.shape
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    lT0 = np.log(T0)
    lT1 = np.log(T1)
    for s in range(1, steps + 1):
        lt = lT0 + (lT1 - lT0) * (s / steps)
        T = float(np.exp(lt))
        G = x @ x.T
        np.fill_diagonal(G, -10.0)
        M = float(G.max())
        W = np.exp((G - M) / T)
        np.fill_diagonal(W, 0.0)
        Grad = (W @ x) / T
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = 0.9 * m + 0.1 * Gt
        v = 0.999 * v + 0.001 * (Gt * Gt)
        c1 = 1.0 - 0.9 ** s
        c2 = 1.0 - 0.999 ** s
        mh = m / c1
        vh = v / c2
        upd = mh / (np.sqrt(vh) + 1e-8)
        x = x - lr * upd
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        x /= nr
    return x


def lp_polish(x, seconds, t_end):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 8e-4
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and time.time() < t_end:
        if radius < 1e-14:
            break
        c = (x @ x.T)[iu]
        mcur = float(c.max())
        thr = max(20.0 * radius, 2e-4)
        sel = c > mcur - thr
        # cap problem size for speed
        if int(sel.sum()) > 2500:
            # keep tightest 2500
            idx = np.argpartition(c, -2500)[-2500:]
            mask = np.zeros_like(sel)
            mask[idx] = True
            sel = sel & mask
        I, J = iu[0][sel], iu[1][sel]
        p, nv = len(I), n * d + 1
        if p == 0:
            break
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
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        y /= nr
        vv = max_cos(y)
        if vv < best:
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x


def refine_single(q, Xcur, iters=120, T=0.005, lr=0.06):
    q = q.copy()
    nq = float(np.linalg.norm(q))
    if nq == 0 or not np.isfinite(nq):
        q = Xcur[0] * -1.0
        nq = float(np.linalg.norm(q))
    q /= nq
    for _ in range(iters):
        dots = Xcur @ q
        mm = float(dots.max())
        w = np.exp((dots - mm) / T)
        ws = float(w.sum())
        if ws == 0 or not np.isfinite(ws):
            break
        w /= ws
        g = w @ Xcur
        g = g - float(g @ q) * q
        gn = float(np.linalg.norm(g))
        if gn == 0 or not np.isfinite(gn):
            break
        q = q - lr * g
        nq = float(np.linalg.norm(q))
        if nq == 0 or not np.isfinite(nq):
            break
        q /= nq
    return q


def solve():
    t0 = time.time()
    T_LIMIT = 172.0
    t_end = t0 + T_LIMIT
    rng = np.random.default_rng(7)
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.array(np.load(fn), dtype=float)
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (102, 12):
                cands.append((max_cos(a), a))
        except Exception:
            pass
    if len(cands) == 0:
        a = rng.standard_normal((102, 12))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        x_best = a
        best = max_cos(a)
    else:
        cands.sort(key=lambda t: t[0])
        best, x_best = cands[0][0], cands[0][1].copy()
    n, d = 102, 12
    K_choices = [4, 6, 8, 12]
    trial = 0
    # main multi-point cavity reconstruction loop
    while time.time() < t_end - 12.0 and trial < 300:
        trial += 1
        if time.time() > t_end - 12.0:
            break
        # contact ranking from incumbent
        G = x_best @ x_best.T
        np.fill_diagonal(G, -2.0)
        mval = float(G.max())
        # contact count for ranking (moderate tol to spread counts)
        cnt = (G > mval - 0.002).sum(axis=1)
        rowmax = G.max(axis=1)
        k = K_choices[int(rng.integers(len(K_choices)))]
        mode = trial % 3
        del_idx = None
        if mode == 0:
            # top-k by contact count with randomisation among top-2k
            order = np.argsort(-cnt - 0.1 * rng.random(n))
            # ensure at least k
            pool = order[:min(n, max(k * 3, k + 6))]
            # pick k from pool favouring high cnt: take top-k with small random swap
            del_idx = pool[:k].copy()
            # random swap one element to diversify
            if rng.random() < 0.7 and len(pool) > k:
                del_idx[int(rng.integers(k))] = pool[int(rng.integers(k, len(pool)))]
        elif mode == 1:
            # mutually contacting cluster: seed among most overloaded
            top_seed = np.argsort(-cnt - 0.1 * rng.random(n))[:5]
            seed = int(top_seed[int(rng.integers(len(top_seed)))])
            # neighbours by largest cosine to seed
            nbr = np.argsort(-G[seed])
            nbr = nbr[nbr != seed]
            del_idx = np.concatenate([[seed], nbr[:k - 1]]).astype(int)
        else:
            # mix rowmax and contact count
            score = rowmax + 0.0005 * cnt.astype(float) + 0.0002 * rng.random(n)
            order = np.argsort(-score)
            # take cluster around worst rowmax point plus top scores
            worst = int(order[0])
            nbr = np.argsort(-G[worst])
            nbr = nbr[nbr != worst]
            # half neighbours, half global worst
            h1 = min(k - 1, max(1, k // 2))
            set1 = nbr[:h1]
            rest = [o for o in order if o != worst and o not in set(set1.tolist())]
            set2 = np.array(rest[:k - 1 - h1], dtype=int) if (k - 1 - h1) > 0 else np.array([], dtype=int)
            del_idx = np.concatenate([[worst], set1, set2]).astype(int)
            if len(del_idx) < k:
                # pad with top order
                extra = [o for o in order if o not in set(del_idx.tolist())]
                del_idx = np.concatenate([del_idx, np.array(extra[:k - len(del_idx)], dtype=int)])
            del_idx = del_idx[:k]
        del_idx = np.unique(del_idx.astype(int))
        # if duplicates collapsed, pad
        if len(del_idx) < k:
            order = np.argsort(-cnt)
            for o in order:
                if o not in set(del_idx.tolist()):
                    del_idx = np.append(del_idx, o)
                    if len(del_idx) >= k:
                        break
        k = len(del_idx)
        mask = np.ones(n, dtype=bool)
        mask[del_idx] = False
        Xrem = x_best[mask].copy()
        # reoptimise remaining so they spread into vacated region
        steps = 450 if k <= 6 else 550
        # keep steps modest to save time
        steps = min(steps, 550)
        Xrem_q = adam_quench(Xrem, steps, 0.020, 0.0025, 0.0022)
        Xrem_q /= np.linalg.norm(Xrem_q, axis=1, keepdims=True)
        if time.time() < t_end - 11.0:
            # short LP polish on remainder
            Xrem_q = lp_polish(Xrem_q, 0.30, t_end - 10.0)
        Xcur = Xrem_q.copy()
        # reinsert one at a time: best of >=20000 candidates lexicographically
        feasible = True
        for _ in range(k):
            if time.time() > t_end - 11.0:
                feasible = False
                break
            S = rng.standard_normal((20000, d))
            S /= np.linalg.norm(S, axis=1, keepdims=True)
            D = S @ Xcur.T
            sc = D.max(axis=1)
            # take 300 best primaries for secondary criterion
            part = np.argpartition(sc, 300)[:300]
            Dtop = D[part]
            sctop = sc[part]
            # number of near-max contacts within 0.003 of its own max
            cnttop = (Dtop > sctop[:, None] - 0.003).sum(axis=1)
            # lexicographic: primary sc, secondary cnt
            lex = np.lexsort((cnttop, sctop))
            bi = part[lex[0]]
            q = S[bi].copy()
            q = refine_single(q, Xcur, iters=120, T=0.005, lr=0.06)
            Xcur = np.vstack([Xcur, q[None, :]])
        if not feasible:
            continue
        Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
        # only a short LP polish after all insertions
        if time.time() < t_end - 10.5:
            Xpol = lp_polish(Xcur, 0.80, t_end - 9.0)
        else:
            Xpol = Xcur
        vp = max_cos(Xpol)
        if vp < best:
            best = vp
            x_best = Xpol.copy()
    # final snap of incumbent with remaining time (fallback retained if no gain)
    remain = t_end - time.time() - 1.0
    if remain > 2.0:
        Xf = lp_polish(x_best, min(remain, 8.0), t_end)
        vf = max_cos(Xf)
        if vf < best:
            best = vf
            x_best = Xf.copy()
    x_best /= np.linalg.norm(x_best, axis=1, keepdims=True)
    return x_best
