import time
import itertools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())


def lp_polish(x, seconds, t_deadline=None, radius_init=1e-3):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = max_cos(x)
    radius = float(radius_init)
    start = time.time()
    iu = np.triu_indices(n, 1)
    while time.time() - start < seconds:
        if t_deadline is not None and time.time() > t_deadline:
            break
        if radius < 1e-14:
            break
        if t_deadline is not None and time.time() > t_deadline - 0.05:
            break
        c = (x @ x.T)[iu]
        m = float(c.max())
        sel = c > m - max(20 * radius, 1e-9)
        if int(sel.sum()) > 1500:
            idx = np.argpartition(c, -1500)[-1500:]
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


def build_candidates():
    C = []
    for combo in itertools.combinations(range(13), 3):
        for mask in range(8):
            v = np.zeros(13, dtype=float)
            for k, pos in enumerate(combo):
                v[pos] = 1.0 if ((mask >> k) & 1) else -1.0
            C.append(v / np.sqrt(3.0))
    for combo in itertools.combinations(range(13), 4):
        for mask in range(16):
            v = np.zeros(13, dtype=float)
            for k, pos in enumerate(combo):
                v[pos] = 1.0 if ((mask >> k) & 1) else -1.0
            C.append(v / 2.0)
    return np.array(C, dtype=float)


def greedy_build(C):
    N = C.shape[0]
    mask = np.zeros(N, dtype=bool)
    maxv = np.full(N, -2.0, dtype=float)
    cnt = np.zeros(N, dtype=int)
    sumexp = np.zeros(N, dtype=float)
    chosen = []
    for k in range(95):
        if k == 0:
            idx = 0
        else:
            valid = np.where(~mask)[0]
            m = maxv[valid].min()
            tie = valid[maxv[valid] <= m + 1e-12]
            if len(tie) > 1:
                cmin = cnt[tie].min()
                tie2 = tie[cnt[tie] == cmin]
                if len(tie2) > 1:
                    idx = tie2[np.argmin(sumexp[tie2])]
                else:
                    idx = tie2[0]
            else:
                idx = tie[0]
        chosen.append(int(idx))
        mask[idx] = True
        if k == 94:
            break
        v = C[idx]
        dots = C @ v
        valid = np.where(~mask)[0]
        d = dots[valid]
        sumexp[valid] += np.exp(20.0 * d)
        mv = maxv[valid]
        greater = d > mv + 1e-12
        equal = (~greater) & (np.abs(d - mv) <= 1e-12)
        maxv[valid[greater]] = d[greater]
        cnt[valid[greater]] = 1
        cnt[valid[equal]] += 1
    return np.array(chosen, dtype=int)


def top50_of_G(G):
    n = G.shape[0]
    iu = np.triu_indices(n, 1)
    pairs = G[iu]
    k = min(50, pairs.size)
    part = np.partition(pairs, -k)[-k:]
    return np.sort(part)[::-1]


def is_better(new_top, best_top, tol=1e-12):
    for a, b in zip(new_top, best_top):
        if a < b - tol:
            return True
        if a > b + tol:
            return False
    return False


def swap_improve(C, chosen_idx, rng, deadline):
    n_pool = 200
    Xd = C[chosen_idx].copy()
    n = Xd.shape[0]
    G = Xd @ Xd.T
    best_top = top50_of_G(G)
    # main sweeps
    while time.time() < deadline:
        # build pool: 200 unused with smallest max-dot
        # compute max dot of all C to Xd
        Dall = C @ Xd.T  # (N,95)
        # for chosen, set inf
        maxall = Dall.max(axis=1)
        maxall[chosen_idx] = np.inf
        # argpartition 200
        if np.sum(np.isfinite(maxall)) < n_pool:
            break
        pool = np.argpartition(maxall, n_pool)[:n_pool]
        # order pool by maxall for determinism
        pool = pool[np.argsort(maxall[pool])]
        Cpool = C[pool]
        Dpool = Cpool @ Xd.T  # 200x95
        order = rng.permutation(n)
        improved_sweep = False
        for i in order:
            if time.time() > deadline:
                break
            keep = np.ones(n, dtype=bool)
            keep[i] = False
            # other pairs among keep
            idx_keep = np.where(keep)[0]
            # extract sub-gram
            Gsub = G[np.ix_(idx_keep, idx_keep)]
            iu_s = np.triu_indices(len(idx_keep), 1)
            other = Gsub[iu_s]
            # search best replacement for position i
            cur_best_top = best_top
            best_j = -1
            best_new94 = None
            # loop over pool candidates (skip if already selected)
            for j in range(len(pool)):
                if time.time() > deadline:
                    break
                pj = pool[j]
                if pj in set(chosen_idx.tolist()):
                    continue
                new94 = Dpool[j, keep]
                # quick prune: max of new94 vs cur best max
                # if max(new94) > cur_best_top[0]+1e-12 and max(other)>? actually combined max
                combined_max = new94.max() if new94.max() > other.max() else other.max()
                if combined_max > cur_best_top[0] + 1e-12:
                    continue
                combined = np.concatenate([other, new94])
                # top50
                part = np.partition(combined, -50)[-50:]
                nt = np.sort(part)[::-1]
                if is_better(nt, cur_best_top):
                    cur_best_top = nt
                    best_j = j
                    best_new94 = new94.copy()
            if best_j >= 0:
                # apply swap
                pj = int(pool[best_j])
                # update chosen_idx
                pos = np.where(chosen_idx == chosen_idx[i])[0]  # placeholder
                # find position i in chosen_idx array: chosen_idx[i] is old
                chosen_idx[i] = pj
                newvec = C[pj].copy()
                Xd[i] = newvec
                # update G row/col i
                Gi = Xd @ newvec
                G[i, :] = Gi
                G[:, i] = Gi
                G[i, i] = 1.0
                # update Dpool column i
                Dpool[:, i] = Cpool @ newvec
                # Dall max for pool refresh later; update best_top
                best_top = cur_best_top
                improved_sweep = True
        if not improved_sweep:
            break
    return Xd, chosen_idx


def gd_relax(X0, betas, steps_list, lr_list, rng, deadline, anchor=None, lam_list=None, mu=0.9):
    X = X0.copy()
    # small symmetry break
    X = X + 1e-3 * rng.standard_normal(X.shape)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    V = np.zeros_like(X)
    si = 0
    for bi, beta in enumerate(betas):
        steps = steps_list[bi]
        lr = lr_list[bi]
        lam = 0.0 if (lam_list is None or anchor is None) else lam_list[bi]
        for s in range(steps):
            if time.time() > deadline:
                return X
            G = X @ X.T
            np.fill_diagonal(G, -10.0)
            mx = G.max()
            W = np.exp((G - mx) * beta)
            np.fill_diagonal(W, 0.0)
            sw = W.sum()
            if not np.isfinite(sw) or sw == 0:
                break
            W /= sw
            Grad = W @ X
            dot = np.sum(Grad * X, axis=1, keepdims=True)
            Gt = Grad - dot * X
            if lam != 0.0 and anchor is not None:
                # pull toward anchor (original lattice point)
                d2 = np.sum(anchor * X, axis=1, keepdims=True)
                pull = anchor - d2 * X
                Gt = Gt - lam * pull
            V = mu * V + Gt
            X = X - lr * V
            X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def solve():
    t0 = time.time()
    BUDGET = 172.0
    deadline_all = t0 + BUDGET
    rng = np.random.default_rng(12345)

    try:
        xb = np.load("best.npy").astype(float)
    except Exception:
        xb = None
    try:
        xr = np.load("record.npy").astype(float)
    except Exception:
        xr = None
    base = None
    base_val = np.inf
    for c in (xb, xr):
        if c is not None and c.shape == (95, 13):
            cn = c / np.linalg.norm(c, axis=1, keepdims=True)
            v = max_cos(cn)
            if v < base_val:
                base_val = v
                base = cn.copy()
    if base is None:
        base = rng.standard_normal((95, 13))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        base_val = max_cos(base)

    # 1) lattice enumeration + greedy
    C = build_candidates()
    chosen_idx = greedy_build(C)
    # 2) swaps ~ up to 30s
    swap_deadline = min(t0 + 32.0, deadline_all - 130.0)
    if swap_deadline > time.time() + 2.0:
        try:
            Xd, chosen_idx = swap_improve(C, chosen_idx.copy(), rng, swap_deadline)
        except Exception:
            Xd = C[chosen_idx].copy()
    else:
        Xd = C[chosen_idx].copy()
    Xd = Xd / np.linalg.norm(Xd, axis=1, keepdims=True)

    # 3) smooth relaxation two branches
    betas = [30, 60, 100, 180, 300, 500]
    steps_list = [350, 350, 350, 350, 350, 450]
    # lr scaling ~ 24/beta
    lr_list = [0.8, 0.4, 0.24, 0.133, 0.08, 0.048]
    lam_list = [0.6, 0.4, 0.2, 0.0, 0.0, 0.0]

    relax_deadline_A = min(deadline_all - 70.0, time.time() + 28.0)
    XA = gd_relax(Xd, betas, steps_list, lr_list, rng, relax_deadline_A, anchor=None, lam_list=None)

    relax_deadline_B = min(deadline_all - 45.0, time.time() + 28.0)
    # fresh rng for B determinism (continue same rng)
    XB = gd_relax(Xd, betas, steps_list, lr_list, rng, relax_deadline_B, anchor=Xd, lam_list=lam_list)

    # 4) LP polish branches + base
    cands = []
    cands.append((XA, "A"))
    cands.append((XB, "B"))
    cands.append((Xd.copy(), "D"))
    cands.append((base.copy(), "base"))

    polished = []
    # allocate polish time
    # reserve ~10s final? distribute
    remain = deadline_all - time.time()
    # give each ~ remain/4 but cap
    per = max(2.0, min(9.0, (remain - 5.0) / 4.0))
    best_X = base.copy()
    best_v = max_cos(best_X)
    for Xc, _ in cands:
        try:
            Xp = lp_polish(Xc, seconds=per, t_deadline=deadline_all - 1.0)
            v = max_cos(Xp)
            if v < best_v:
                best_v = v
                best_X = Xp.copy()
        except Exception:
            continue
        if time.time() > deadline_all - 3.0:
            break
    # final tiny extra polish on current best if time left
    if time.time() < deadline_all - 4.0:
        try:
            Xf = lp_polish(best_X, seconds=min(6.0, deadline_all - time.time() - 1.0), t_deadline=deadline_all - 0.5)
            vf = max_cos(Xf)
            if vf < best_v:
                best_X = Xf
        except Exception:
            pass
    best_X = best_X / np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
