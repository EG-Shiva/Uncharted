import time
import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.linalg import orthogonal_procrustes
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _scores_pop(Pop):
    # Pop: (P,N,D) normalized
    G = np.matmul(Pop, Pop.transpose(0, 2, 1))
    P, N, _ = G.shape
    for p in range(P):
        np.fill_diagonal(G[p], -2.0)
    return G.max(axis=(1, 2))

def _greedy_init(rng, n, d, cand=400):
    X = np.empty((n, d))
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v) + 1e-12
    X[0] = v
    for k in range(1, n):
        C = rng.standard_normal((cand, d))
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        dots = C @ X[:k].T
        worst = dots.max(axis=1)
        bi = int(np.argmin(worst))
        X[k] = C[bi]
    return X

def _batch_quench(Pop, steps, b0, b1, lr0, lr1, mu=0.85, scales=None, deadline=None):
    P, N, D = Pop.shape
    V = np.zeros_like(Pop)
    if scales is None:
        scales = np.ones(P)
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        if deadline is not None and time.time() >= deadline:
            break
        f = s / max(1, steps - 1)
        beta0 = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        # batched Gram
        G = np.matmul(Pop, Pop.transpose(0, 2, 1))
        # per-member max
        m = G.max(axis=(1, 2))
        # subtract max for stability, apply per-member beta
        # G shape (P,N,N)
        betas = beta0 * scales  # (P,)
        # broadcast: exp(beta_p*(G_p - m_p))
        Gm = G - m[:, None, None]
        Gm = Gm * betas[:, None, None]
        # clip exponent to avoid overflow
        np.clip(Gm, -50, 0, out=Gm)
        W = np.exp(Gm)
        # zero diagonal
        for p in range(P):
            np.fill_diagonal(W[p], 0.0)
        sw = W.sum(axis=(1, 2))
        sw[sw <= 0] = 1.0
        W = W / sw[:, None, None]
        Gr = np.matmul(W, Pop)
        dot = (Gr * Pop).sum(axis=2, keepdims=True)
        Gt = Gr - dot * Pop
        V = mu * V + Gt
        step = lr * V
        sdot = (step * Pop).sum(axis=2, keepdims=True)
        step = step - sdot * Pop
        sn = np.linalg.norm(step, axis=2, keepdims=True) + 1e-12
        over = sn > 0.05
        step = np.where(over, step * (0.05 / sn), step)
        Pop = Pop - step
        Pop = Pop / (np.linalg.norm(Pop, axis=2, keepdims=True) + 1e-12)
    return Pop

def _single_quench(X, steps, b0, b1, lr0, lr1, mu=0.85, deadline=None):
    V = np.zeros_like(X)
    n = X.shape[0]
    lb0, lb1 = np.log(b0), np.log(b1)
    ll0, ll1 = np.log(lr0), np.log(lr1)
    for s in range(steps):
        if deadline is not None and time.time() >= deadline:
            break
        f = s / max(1, steps - 1)
        beta = float(np.exp(lb0 + (lb1 - lb0) * f))
        lr = float(np.exp(ll0 + (ll1 - ll0) * f))
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        W = np.exp(np.clip(beta * (G - m), -50, 0))
        np.fill_diagonal(W, 0.0)
        sw = float(W.sum())
        if not np.isfinite(sw) or sw <= 0:
            break
        W /= sw
        Gr = W @ X
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        V = mu * V + Gt
        step = lr * V
        sdot = np.sum(step * X, axis=1, keepdims=True)
        step = step - sdot * X
        sn = np.linalg.norm(step, axis=1, keepdims=True) + 1e-12
        step = np.where(sn > 0.05, step * (0.05 / sn), step)
        X = X - step
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X

def _signature(X, topk=150, gap=0.004):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    iu = np.triu_indices(X.shape[0], 1)
    c = G[iu]
    m = float(c.max())
    srt = np.sort(c)[::-1][:topk]
    deg = (G > m - gap).sum(axis=1)
    deg_s = np.sort(deg)
    return srt, deg_s, m

def _sig_dist(sigA, sigB):
    return float(np.linalg.norm(sigA[0] - sigB[0])) + 1e-4 * float(np.linalg.norm(sigA[1].astype(float) - sigB[1].astype(float)))

def _lp_polish(X, seconds, deadline):
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n, d = X.shape
    best = _max_cos(X)
    radius = 6e-4
    start = time.time()
    iu0 = np.triu_indices(n, 1)
    it = 0
    while True:
        if time.time() - start >= seconds:
            break
        if time.time() >= deadline:
            break
        if radius < 1e-13:
            break
        it += 1
        if it > 30:
            break
        G = X @ X.T
        c = G[iu0]
        m = float(c.max())
        window = max(20.0 * radius, 4e-5)
        order = np.argsort(-c)
        sel_list = []
        for idx in order:
            if c[idx] > m - window and len(sel_list) < 600:
                sel_list.append(int(idx))
            elif c[idx] <= m - window:
                break
            if len(sel_list) >= 600:
                break
        if len(sel_list) == 0:
            break
        sel = np.array(sel_list)
        I = iu0[0][sel]
        J = iu0[1][sel]
        p = len(sel)
        nv = n * d + 1
        try:
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], 1).ravel()
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 2.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        y = y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-12)
        v = _max_cos(y)
        if v < best - 1e-15:
            X = y
            best = v
            radius = min(radius * 1.5, 0.02)
        else:
            radius *= 0.5
    return X

def _recombine(A, B, k, rng):
    n, d = A.shape
    try:
        R, _ = orthogonal_procrustes(B, A)
    except Exception:
        R = np.eye(d)
    Bal = B @ R
    # renormalize (rotation preserves norm, but numerical)
    Bal = Bal / (np.linalg.norm(Bal, axis=1, keepdims=True) + 1e-12)
    M = A @ Bal.T
    try:
        row, col = linear_sum_assignment(-M)
    except Exception:
        col = np.arange(n)
        row = np.arange(n)
    Bre = np.empty_like(Bal)
    Bre[row] = Bal[col]
    S = rng.choice(n, size=k, replace=False)
    C = Bre.copy()
    C[S] = A[S]
    # tiny jitter
    C = C + 5e-4 * rng.standard_normal(C.shape)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    return C

def solve():
    t0 = time.time()
    INTERNAL = t0 + 150.0
    HARD = t0 + 165.0
    rng = np.random.default_rng(20261102)
    n, d = 58, 22
    P = 24
    best = None
    best_val = 1e9
    for f in ("best.npy", "record.npy"):
        try:
            v = np.load(f).astype(float)
            if v.shape != (58, 22):
                continue
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            c = _max_cos(v)
            if c < best_val:
                best_val = c
                best = v.copy()
        except Exception:
            continue
    if best is None:
        z = rng.standard_normal((n, d))
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
        best = z
        best_val = _max_cos(best)
    best_init = best.copy()
    # build population
    Pop = np.empty((P, n, d))
    Pop[0] = best_init
    for p in range(1, P):
        if p % 2 == 1:
            try:
                Pop[p] = _greedy_init(rng, n, d, cand=350)
            except Exception:
                z = rng.standard_normal((n, d))
                Pop[p] = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        else:
            z = rng.standard_normal((n, d))
            Pop[p] = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
        if time.time() >= INTERNAL - 100:
            # fill rest randomly fast
            for q in range(p + 1, P):
                z = rng.standard_normal((n, d))
                Pop[q] = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
            break
    # different inverse temperatures per member
    scales = np.exp(rng.uniform(-0.7, 0.7, size=P))
    # simultaneous vectorized quench
    try:
        Pop = _batch_quench(Pop, 320, 60, 600, 0.004, 0.0005, mu=0.85, scales=scales, deadline=INTERNAL - 40)
    except Exception:
        pass
    try:
        sc = _scores_pop(Pop)
    except Exception:
        sc = np.array([_max_cos(Pop[p]) for p in range(P)])
    # update best from population
    for p in range(P):
        if sc[p] < best_val:
            best_val = float(sc[p])
            best = Pop[p].copy()
    # evolutionary recombination of diverse elites
    splits = [15, 29, 43]
    rnd = 0
    # precompute signatures for elite selection each round
    while time.time() < INTERNAL - 30:
        if time.time() >= HARD - 25:
            break
        rnd += 1
        # refresh scores
        try:
            sc = _scores_pop(Pop)
        except Exception:
            sc = np.array([_max_cos(Pop[p]) for p in range(P)])
        order = np.argsort(sc)
        top_idx = order[:6]
        # signatures of top
        sigs = []
        for idx in top_idx:
            try:
                sigs.append(_signature(Pop[idx]))
            except Exception:
                sigs.append((np.zeros(150), np.zeros(n), 0.0))
        # elite1 = best of top
        e1_pos = 0
        e1_idx = int(top_idx[e1_pos])
        # elite2 = most diverse among remaining top, requiring quality
        best_d = -1
        e2_pos = 1
        for j in range(1, len(top_idx)):
            dd = _sig_dist(sigs[0], sigs[j])
            # slight quality penalty to keep high-quality
            qual = max(0.0, sc[top_idx[j]] - sc[e1_idx]) * 50.0
            score_d = dd - qual * 1e-4
            if score_d > best_d:
                best_d = score_d
                e2_pos = j
        e2_idx = int(top_idx[e2_pos])
        A = Pop[e1_idx].copy()
        Bm = Pop[e2_idx].copy()
        # occasionally swap to test both directions
        if rnd % 2 == 0:
            A, Bm = Bm, A
        for k in splits:
            if time.time() >= INTERNAL - 28:
                break
            if time.time() >= HARD - 22:
                break
            try:
                child = _recombine(A, Bm, k, rng)
            except Exception:
                continue
            # quench child with member-specific temperature
            b0c = float(rng.uniform(50, 120))
            b1c = float(rng.uniform(500, 1200))
            try:
                child = _single_quench(child, 260, b0c, b1c, 0.0035, 0.0004, mu=0.85, deadline=INTERNAL - 25)
            except Exception:
                pass
            try:
                v = _max_cos(child)
            except Exception:
                continue
            # diversity check: reject nearly identical contact signature
            try:
                csig = _signature(child)
                d1 = _sig_dist(csig, _signature(A))
                d2 = _sig_dist(csig, _signature(Bm))
                if min(d1, d2) < 1e-7:
                    continue
            except Exception:
                pass
            # insert into population if better than worst and diverse
            try:
                worst = int(np.argmax(sc))
                if v < sc[worst] - 1e-12:
                    Pop[worst] = child.copy()
                    sc[worst] = v
            except Exception:
                pass
            # LP-polish each child (short) if promising
            if v < best_val + 0.003:
                try:
                    child_p = _lp_polish(child, 0.7, min(INTERNAL - 20, HARD - 15))
                    vp = _max_cos(child_p)
                    if vp < v:
                        v = vp
                        child = child_p
                        # update population worst again
                        try:
                            worst2 = int(np.argmax(sc))
                            if v < sc[worst2]:
                                Pop[worst2] = child.copy()
                                sc[worst2] = v
                        except Exception:
                            pass
                except Exception:
                    pass
            if v < best_val - 1e-12:
                best_val = v
                best = child.copy()
        # inject fresh greedy every 6 rounds to keep diversity
        if rnd % 6 == 0 and time.time() < INTERNAL - 45:
            try:
                fresh = _greedy_init(rng, n, d, cand=250)
                fresh = _single_quench(fresh, 150, 80, 600, 0.003, 0.0005, deadline=INTERNAL - 30)
                vf = _max_cos(fresh)
                worst = int(np.argmax(sc))
                if vf < sc[worst]:
                    Pop[worst] = fresh
            except Exception:
                pass
        if rnd > 60:
            break
    # final full polish of best
    remain = min(INTERNAL, HARD) - time.time() - 1.0
    if remain > 3:
        try:
            lp_sec = min(remain - 0.5, 14.0)
            polished = _lp_polish(best, lp_sec, min(INTERNAL, HARD) - 0.5)
            vp = _max_cos(polished)
            if vp < best_val:
                best = polished
                best_val = vp
        except Exception:
            pass
    best = best / (np.linalg.norm(best, axis=1, keepdims=True) + 1e-12)
    best = np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0)
    nr = np.linalg.norm(best, axis=1, keepdims=True)
    nr[nr == 0] = 1.0
    best = best / nr
    return best
