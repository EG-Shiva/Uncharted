import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def max_cos(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def rowmaxs(x):
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1)

def smooth_refine(v, alpha0, alpha1, steps, lr, beta=0.9, track_best=True):
    n, d = v.shape
    cur = v.copy()
    best = cur.copy()
    best_s = max_cos(cur)
    mom = np.zeros_like(cur)
    for s in range(steps):
        frac = s / max(steps - 1, 1)
        alpha = alpha0 * ((alpha1 / max(alpha0, 1e-12)) ** frac)
        G = cur @ cur.T
        np.fill_diagonal(G, -10.0)
        m = float(np.max(G))
        W = np.exp(alpha * (G - m))
        np.fill_diagonal(W, 0.0)
        Grad = W @ cur
        proj = np.sum(Grad * cur, axis=1, keepdims=True)
        Gt = Grad - proj * cur
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        Gt = Gt / rs
        mom = beta * mom + (1.0 - beta) * Gt
        bc = 1.0 - beta ** (s + 1)
        step = mom / bc
        cur_lr = lr * (1.0 - 0.5 * frac)
        cur = cur - cur_lr * step
        nr = np.linalg.norm(cur, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        cur /= nr
        if track_best and ((s % 200 == 199) or (s == steps - 1)):
            sc = max_cos(cur)
            if sc < best_s:
                best_s = sc
                best = cur.copy()
    if track_best:
        return best, best_s
    return cur, max_cos(cur)

def fingerprint(x, tol=1e-6):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    Adj = (G > m - tol)
    np.fill_diagonal(Adj, False)
    deg = Adj.sum(axis=1).astype(float)
    deg_s = np.sort(deg)[::-1]
    try:
        GG = x @ x.T
        ev = np.linalg.eigvalsh(GG)
        ev = np.sort(ev)[::-1][:10]
    except Exception:
        ev = np.zeros(10)
    nc = int(Adj.sum() // 2)
    return deg_s, ev, nc, m

def fp_dist(deg1, ev1, deg2, ev2):
    d1 = float(np.sum(np.abs(deg1 - deg2)))
    d2 = float(np.linalg.norm(ev1 - ev2))
    return d1, d2

def get_cluster(x, m, rng, kmin=8, kmax=15):
    n = x.shape[0]
    G = x @ x.T
    np.fill_diagonal(G, -2.0)
    Adj = (G > m - 1e-6)
    np.fill_diagonal(Adj, False)
    deg = Adj.sum(axis=1)
    rm = G.max(axis=1)
    # candidate hubs: top by degree, then by rowmax
    order_deg = np.argsort(-deg * 1000.0 - rm)
    nhub = 5
    hub = int(order_deg[int(rng.integers(0, min(nhub, n)))])
    # neighbours sorted by closeness
    nbrs = np.where(Adj[hub])[0]
    if len(nbrs) > 0:
        vals = G[hub, nbrs]
        nbrs = nbrs[np.argsort(-vals)]
    cluster = [hub] + nbrs.tolist()
    # expand to second neighbours if needed
    if len(cluster) < kmin:
        # add worst points / nearest to hub
        dots_hub = G[hub]
        order = np.argsort(-dots_hub)
        for o in order:
            if o not in cluster and o != hub:
                cluster.append(int(o))
            if len(cluster) >= kmin:
                break
    # target size
    tgt = int(rng.integers(kmin, kmax + 1))
    tgt = min(tgt, n)
    cluster = cluster[:tgt]
    # if cluster still small, fill random
    if len(cluster) < tgt:
        extra = rng.choice(n, size=tgt - len(cluster), replace=False)
        for e in extra:
            if int(e) not in cluster:
                cluster.append(int(e))
    # BFS expansion to keep connectivity when possible: ensure size tgt
    # already have tgt
    # order cluster by worst rowmax first for reinsertion
    cl = np.array(cluster, dtype=int)
    w = rm[cl]
    cl = cl[np.argsort(-w)]
    return cl

def hole_reinsert(cur, cluster, rng, n_cand=20000):
    n, d = cur.shape
    # reinsert one at a time
    for k in cluster:
        kk = int(k)
        others_idx = [i for i in range(n) if i != kk]
        others = cur[others_idx]
        cand = rng.standard_normal(size=(n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        dots = cand @ others.T
        md = dots.max(axis=1)
        bi = int(np.argmin(md))
        p = cand[bi].copy()
        # 100 steps per-point minimax descent against frozen others
        alpha = 160.0
        lr = 0.035
        for s in range(100):
            dts = others @ p
            mm = float(np.max(dts))
            w = np.exp(alpha * (dts - mm))
            w /= (w.sum() + 1e-16)
            grad = w @ others
            gt = grad - float(grad @ p) * p
            cur_lr = lr * (1.0 - 0.5 * s / 100.0)
            p = p - cur_lr * gt
            nn = np.linalg.norm(p)
            if nn == 0:
                p = cand[bi]
            else:
                p /= nn
        cur[kk] = p
    return cur

def lp_wide(x, seconds, kmin=900, wmin=3e-07):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    best_x = x.copy()
    radius, start = 5e-4, time.time()
    iu = np.triu_indices(n, 1)
    ar = np.arange(d)
    while time.time() - start < seconds and radius > 1e-14:
        G = x @ x.T
        c = G[iu]
        m = float(np.max(c))
        thresh = m - max(30.0 * radius, wmin, 1e-9)
        sel = c > thresh
        if np.sum(sel) < kmin:
            order = np.argsort(-c)
            sel2 = np.zeros_like(sel, dtype=bool)
            sel2[order[:kmin]] = True
            sel = sel | sel2
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
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
            best = vv
            best_x = y.copy()
            x = y
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
        if best < max_cos(x) - 1e-12:
            x = best_x.copy()
    return best_x

def solve():
    t0 = time.time()
    BUDGET = 172.0
    EXPLORE_UNTIL = 128.0
    rng = np.random.default_rng(20261103)
    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        try:
            xb = np.load("record.npy").astype(float)
        except Exception:
            xb = rng.standard_normal(size=(99, 13))
    xb = np.asarray(xb, dtype=float)
    if xb.shape != (99, 13):
        tmp = rng.standard_normal(size=(99, 13))
        xb = tmp
    xb = xb / np.linalg.norm(xb, axis=1, keepdims=True)
    n, d = 99, 13
    best_val = max_cos(xb)
    best_x = xb.copy()
    # record fingerprint (anti-return reference)
    rdeg, rev, rnc, rm0 = fingerprint(best_x)
    record_deg = rdeg.copy()
    record_ev = rev.copy()
    # archive list
    archive = []
    deg0, ev0, nc0, m0 = fingerprint(best_x)
    archive.append({'x': best_x.copy(), 'score': best_val, 'deg': deg0, 'ev': ev0, 'nc': nc0})
    cur_x = best_x.copy()
    cur_val = best_val
    it = 0
    while time.time() - t0 < EXPLORE_UNTIL and it < 1000:
        it += 1
        elapsed = time.time() - t0
        # temperature decaying 5e-5 -> 0 over ~100s
        frac = elapsed / 100.0
        if frac >= 1.0:
            Temp = 1e-12
        else:
            Temp = 5e-5 * (1.0 - frac) + 1e-12
        # select starter: most contacts among those within 5e-5 of best
        cands = [a for a in archive if a['score'] <= best_val + 5e-5]
        if len(cands) == 0:
            cands = archive
        # sort by nc descending
        cands = sorted(cands, key=lambda a: a['nc'], reverse=True)
        topk = min(3, len(cands))
        pick = int(rng.integers(0, topk)) if topk > 1 else 0
        base_entry = cands[pick]
        base = base_entry['x'].copy()
        base_m = float(max_cos(base))
        # build cluster 8-15
        try:
            cluster = get_cluster(base, base_m, rng, 8, 15)
        except Exception:
            continue
        # surgery: delete + reinsert
        try:
            cur = base.copy()
            cur = hole_reinsert(cur, cluster, rng, n_cand=20000)
            # LogSumExp refine short
            a0 = float(rng.choice([20.0, 30.0, 45.0]))
            a1 = a0 * float(rng.choice([4.0, 6.0, 8.0]))
            steps = int(rng.integers(400, 800))
            lr = float(rng.uniform(0.025, 0.055))
            cur2, _ = smooth_refine(cur, a0, a1, steps, lr, track_best=True)
        except Exception:
            continue
        s1 = max_cos(cur2)
        # fingerprint
        try:
            deg1, ev1, nc1, m1 = fingerprint(cur2)
        except Exception:
            continue
        # duplicate check vs archive
        is_dup = False
        for a in archive:
            dd, de = fp_dist(deg1, ev1, a['deg'], a['ev'])
            if dd < 0.5 and de < 1e-9:
                is_dup = True
                break
        # anti-return: reject if close to record fingerprint and not improving
        dd_rec, de_rec = fp_dist(deg1, ev1, record_deg, record_ev)
        if dd_rec < 8 and de_rec < 2e-4:
            # within small distance of record topology -> require strict improvement
            if s1 >= best_val - 1e-12:
                continue
        # best update (always archive best)
        if s1 < best_val - 1e-12:
            best_val = s1
            best_x = cur2.copy()
            cur_x = cur2.copy()
            cur_val = s1
            if not is_dup:
                archive.append({'x': cur2.copy(), 'score': s1, 'deg': deg1, 'ev': ev1, 'nc': nc1})
            else:
                # update score of dup
                pass
            # keep archive bounded
            if len(archive) > 60:
                # drop worst score
                archive = sorted(archive, key=lambda a: a['score'])[:60]
            # quick LP snap on new best if time allows
            if time.time() - t0 < EXPLORE_UNTIL - 6:
                try:
                    snapped = lp_wide(best_x, 0.8, kmin=900, wmin=3e-07)
                    sv = max_cos(snapped)
                    if sv < best_val:
                        best_val = sv
                        best_x = snapped.copy()
                        cur_x = snapped.copy()
                        cur_val = sv
                        d2, e2, n2, _ = fingerprint(snapped)
                        archive.append({'x': snapped.copy(), 'score': sv, 'deg': d2, 'ev': e2, 'nc': n2})
                except Exception:
                    pass
        else:
            # SA acceptance for archive insertion / cur move
            accept = False
            if s1 <= cur_val:
                accept = True
            else:
                try:
                    prob = float(np.exp(-(s1 - cur_val) / max(Temp, 1e-12)))
                except Exception:
                    prob = 0.0
                if rng.random() < prob:
                    accept = True
            if accept:
                cur_x = cur2.copy()
                cur_val = s1
                if not is_dup:
                    archive.append({'x': cur2.copy(), 'score': s1, 'deg': deg1, 'ev': ev1, 'nc': nc1})
                    if len(archive) > 60:
                        archive = sorted(archive, key=lambda a: a['score'])[:60]
            # occasional return to best region via archive (already via starter selection)
            if rng.random() < 0.03:
                cur_x = best_x.copy()
                cur_val = best_val
    # final refine
    remain = BUDGET - (time.time() - t0) - 1.5
    if remain > 5:
        bx = best_x.copy()
        bv = best_val
        try:
            tx, _ = smooth_refine(bx, 25.0, 160.0, 800, 0.03)
            sv = max_cos(tx)
            if sv < bv:
                bx = tx.copy(); bv = sv
        except Exception:
            pass
        try:
            tx2, _ = smooth_refine(bx, 120.0, 650.0, 700, 0.012)
            sv2 = max_cos(tx2)
            if sv2 < bv:
                bx = tx2.copy(); bv = sv2
        except Exception:
            pass
        try:
            polished = lp_wide(bx, remain * 0.8, kmin=1100, wmin=2e-07)
            pv = max_cos(polished)
            if pv < bv:
                bx = polished.copy(); bv = pv
        except Exception:
            pass
        if bv < best_val:
            best_x = bx.copy(); best_val = bv
    best_x = best_x / np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x
