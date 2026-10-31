import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import coo_matrix

def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def smooth_relax(x, iters, k0, k1, lr0, lr1, mu, noise, rng):
    n, d = x.shape
    v = np.zeros_like(x)
    for t in range(iters):
        frac = t / max(1, iters - 1)
        k = k0 * (k1 / k0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        moff = G.max()
        E = k * (G - moff)
        np.clip(E, -50.0, 0.0, out=E)
        W = np.exp(E)
        np.fill_diagonal(W, 0.0)
        rs = W.sum(axis=1, keepdims=True) + 1e-12
        grad = (W @ x) / rs
        dot = np.sum(grad * x, axis=1, keepdims=True)
        gt = grad - dot * x
        v = mu * v + gt
        if mu > 0:
            vc = v / (1.0 - mu ** (t + 1))
        else:
            vc = v
        x = x - lr * vc
        if noise > 0 and frac < 0.6:
            amp = noise * (1.0 - frac / 0.6)
            pert = rng.standard_normal(x.shape)
            pd = np.sum(pert * x, axis=1, keepdims=True)
            pert -= pd * x
            x = x + amp * pert
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def lp_polish(x, seconds):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = 1e-3
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds and radius > 1e-14:
        c = (x @ x.T)[iu]
        m = c.max()
        sel = c > m - max(20 * radius, 1e-9)
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        if p == 0:
            break
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
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
            x, best, radius = y, vv, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def signature(X, tol=1e-7):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(G.max())
    mask = G > m - tol
    # count upper triangle contacts
    iu = np.triu_indices(X.shape[0], 1)
    contacts = int(np.count_nonzero(G[iu] > m - tol))
    deg = np.count_nonzero(mask, axis=1).astype(int)
    sdeg = np.sort(deg)[::-1]
    return m, contacts, sdeg

def farthest_repair(X, n_replace, rng, n_cand=2000):
    n, d = X.shape
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    maxper = G.max(axis=1)
    idx = np.argsort(-maxper)[:n_replace]
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[idx] = False
    Xkeep = X[keep_mask]
    chosen = []
    for _ in range(n_replace):
        cand = rng.standard_normal((n_cand, d))
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        md = (cand @ Xkeep.T).max(axis=1)
        if len(chosen) > 0:
            A = np.stack(chosen)
            md2 = (cand @ A.T).max(axis=1)
            md = np.maximum(md, md2)
        bi = int(np.argmin(md))
        chosen.append(cand[bi])
        # grow Xkeep for next pick to avoid clustering among new points
        Xkeep = np.vstack([Xkeep, cand[bi:bi+1]])
    # Xkeep currently contains (n-n_replace)+n_replace = n points (old keep + new)
    return Xkeep

def align_match(P1, P2, rng, n_starts=3, icp_iters=4):
    n, d = P1.shape
    best_Q = np.eye(d)
    best_perm = np.arange(n)
    best_score = -1e100
    for s in range(n_starts):
        if s == 0:
            Q = np.eye(d)
        else:
            A = rng.standard_normal((d, d))
            Q, _ = np.linalg.qr(A)
        for _ in range(icp_iters):
            P2a = P2 @ Q
            D = P1 @ P2a.T
            r, c = linear_sum_assignment(-D)
            Amat = P2[c]
            Bmat = P1[r]
            M = Amat.T @ Bmat
            U, _, Vt = np.linalg.svd(M)
            Q = U @ Vt
        P2a = P2 @ Q
        D = P1 @ P2a.T
        r, c = linear_sum_assignment(-D)
        sc = float(D[r, c].sum())
        if sc > best_score:
            best_score = sc
            best_Q = Q.copy()
            pb = np.empty(n, dtype=int)
            pb[r] = c
            best_perm = pb
    return best_Q, best_perm

def make_child(P1, P2, Q, perm, rng):
    n = P1.shape[0]
    P2a = P2 @ Q
    if rng.random() < 0.5:
        u = rng.standard_normal(4)
        u /= np.linalg.norm(u)
        proj = P1 @ u
        k = int(rng.integers(45, 108))
        idx = np.argsort(-proj)
        S = np.zeros(n, dtype=bool)
        S[idx[:k]] = True
    else:
        ci = int(rng.integers(n))
        cc = P1[ci] + 0.3 * rng.standard_normal(4)
        cc /= np.linalg.norm(cc)
        dots = P1 @ cc
        k = int(rng.integers(45, 108))
        idx = np.argsort(-dots)
        S = np.zeros(n, dtype=bool)
        S[idx[:k]] = True
    C = np.empty_like(P1)
    for i in range(n):
        if S[i]:
            C[i] = P1[i]
        else:
            C[i] = P2a[perm[i]]
    C = C + 1e-4 * rng.standard_normal(C.shape)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    return C

def solve():
    t0 = time.time()
    rng = np.random.default_rng(20261031)
    n, d = 153, 4
    try:
        rec = np.load("best.npy")
    except Exception:
        try:
            rec = np.load("record.npy")
        except Exception:
            rec = rng.standard_normal((n, d))
    rec = np.asarray(rec, dtype=float)
    if rec.shape != (n, d):
        rec = rng.standard_normal((n, d))
    rec = rec / np.linalg.norm(rec, axis=1, keepdims=True)
    best_val = max_cos(rec)
    best = rec.copy()
    # population: list of dicts
    pop = []
    m0, c0, s0 = signature(best)
    pop.append((best_val, best.copy(), c0, s0))
    # generate from-scratch diverse optima
    k0_list = [20, 50, 100, 30, 80, 150, 40]
    k1_list = [500, 1000, 2000, 800, 1500, 2500, 1200]
    n_gen = 7
    for gi in range(n_gen):
        if time.time() - t0 > 55:
            break
        grng = np.random.default_rng(1000 + gi)
        X = grng.standard_normal((n, d))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        k0 = float(k0_list[gi % len(k0_list)])
        k1 = float(k1_list[gi % len(k1_list)])
        X = smooth_relax(X, iters=1000, k0=k0, k1=k1, lr0=0.04, lr1=0.004, mu=0.75, noise=0.015, rng=grng)
        X = lp_polish(X, seconds=0.9)
        v = max_cos(X)
        m_, c_, s_ = signature(X)
        # discard near-duplicates of record basin
        dup = False
        for (pv, px, pc, ps) in pop:
            if abs(v - pv) < 5e-6 and abs(c_ - pc) < 15 and np.abs(s_.astype(int) - ps.astype(int)).sum() < 15:
                dup = True
                break
        if not dup:
            pop.append((v, X.copy(), c_, s_))
        # keep pop sorted, cap 12
        pop = sorted(pop, key=lambda e: e[0])[:12]
    # update best from pop (record retained as fallback, but track global best)
    pop = sorted(pop, key=lambda e: e[0])
    if pop[0][0] < best_val:
        best_val, best = pop[0][0], pop[0][1].copy()
    else:
        # ensure record stays
        pass
    # crossover / mutation cycles
    loop = 0
    while time.time() - t0 < 164:
        loop += 1
        if len(pop) < 2:
            break
        # pick two distinct parents, bias to better but ensure diversity
        # choose a,b randomly
        ia = int(rng.integers(len(pop)))
        ib = int(rng.integers(len(pop)))
        if ia == ib:
            continue
        # skip if near-duplicate pair
        _, _, ca, sa = pop[ia]
        _, _, cb, sb = pop[ib]
        if abs(pop[ia][0] - pop[ib][0]) < 1e-9 and abs(ca - cb) <= 5 and np.abs(sa.astype(int) - sb.astype(int)).sum() < 10:
            continue
        P1 = pop[ia][1]
        P2 = pop[ib][1]
        # random order
        if rng.random() < 0.5:
            P1, P2 = P2, P1
        try:
            Q, perm = align_match(P1, P2, rng, n_starts=2, icp_iters=3)
        except Exception:
            continue
        try:
            C = make_child(P1, P2, Q, perm, rng)
        except Exception:
            continue
        # repair worst few
        try:
            nr = int(rng.integers(2, 6))
            C = farthest_repair(C, nr, rng, n_cand=1500)
        except Exception:
            pass
        # annealed smooth
        try:
            C = smooth_relax(C, iters=600, k0=50.0, k1=1500.0, lr0=0.03, lr1=0.003, mu=0.7, noise=0.008, rng=rng)
        except Exception:
            continue
        # LP polish short
        remain = 164 - (time.time() - t0)
        if remain < 1.2:
            break
        try:
            C = lp_polish(C, seconds=min(1.1, remain * 0.4))
        except Exception:
            continue
        v = max_cos(C)
        try:
            _, cc, sc = signature(C)
        except Exception:
            continue
        # diversity check vs parents and pop
        _, cpa, spa = pop[ia][2], pop[ia][3]
        _, cpb, spb = pop[ib][2], pop[ib][3]
        # need materially different from both parents
        def diff(c1, s1, c2, s2):
            return abs(c1 - c2), int(np.abs(s1.astype(int) - s2.astype(int)).sum())
        dca, dla = diff(cc, sc, pop[ia][2], pop[ia][3])
        dcb, dlb = diff(cc, sc, pop[ib][2], pop[ib][3])
        materially = (dca > 10 or dla > 20) and (dcb > 10 or dlb > 20)
        is_dup_pop = False
        for (pv, px, pc, ps) in pop:
            if abs(v - pv) < 2e-7 and abs(cc - pc) <= 5 and np.abs(sc.astype(int) - ps.astype(int)).sum() < 10:
                is_dup_pop = True
                break
        if v < best_val - 1e-12:
            best_val = v
            best = C.copy()
            pop.append((v, C.copy(), cc, sc))
            pop = sorted(pop, key=lambda e: e[0])[:12]
        elif (materially and not is_dup_pop):
            # accept if competitive (within 0.02 of best) to maintain diversity
            worst = max(e[0] for e in pop)
            if v < worst + 0.005 or len(pop) < 10:
                pop.append((v, C.copy(), cc, sc))
                pop = sorted(pop, key=lambda e: e[0])[:12]
        # occasional mutation of worst to inject novelty if time allows
        if loop % 15 == 0 and time.time() - t0 < 150:
            # mutate a random pop member slightly
            try:
                mi = int(rng.integers(len(pop)))
                Xm = pop[mi][1] + 0.02 * rng.standard_normal((n, d))
                Xm /= np.linalg.norm(Xm, axis=1, keepdims=True)
                Xm = smooth_relax(Xm, iters=300, k0=80.0, k1=1200.0, lr0=0.02, lr1=0.003, mu=0.6, noise=0.0, rng=rng)
                Xm = lp_polish(Xm, seconds=0.6)
                vm = max_cos(Xm)
                _, cm, sm = signature(Xm)
                dup2 = False
                for (pv, px, pc, ps) in pop:
                    if abs(vm - pv) < 2e-7 and abs(cm - pc) <= 5 and np.abs(sm.astype(int) - ps.astype(int)).sum() < 10:
                        dup2 = True
                        break
                if not dup2 and vm < max(e[0] for e in pop):
                    pop.append((vm, Xm.copy(), cm, sm))
                    pop = sorted(pop, key=lambda e: e[0])[:12]
                    if vm < best_val:
                        best_val = vm
                        best = Xm.copy()
            except Exception:
                pass
    # final polish of best with remaining time
    try:
        remain = (t0 + 178) - time.time()
        if remain > 1.0:
            best = lp_polish(best, seconds=min(remain - 0.5, 8.0))
            bv = max_cos(best)
            # retain fallback: if polish worsened, revert to record
            if bv < best_val + 1e-12:
                best_val = bv
            else:
                # keep previous best if worse (re-evaluate)
                pass
    except Exception:
        pass
    # safety: ensure best is at least as good as record, else return record
    try:
        rv = max_cos(rec)
        if max_cos(best) > rv + 1e-12:
            best = rec.copy()
    except Exception:
        pass
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    return best
