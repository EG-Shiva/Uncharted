import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(g.max())

def _norm_rows(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

# ---------- groups ----------
def _cyclic8():
    Ms = []
    for s in range(8):
        P = np.zeros((8, 8))
        for i in range(8):
            P[(i + s) % 8, i] = 1.0
        Ms.append(P.T)  # right action
    return Ms

def _cyclic8_pm():
    base = _cyclic8()
    out = []
    for P in base:
        out.append(P)
        out.append(-P)
    return out

def _block44():
    Ms = []
    for a in range(4):
        for b in range(4):
            P = np.zeros((8, 8))
            for i in range(4):
                P[(i + a) % 4, i] = 1.0
            for i in range(4, 8):
                P[4 + ((i - 4 + b) % 4), i] = 1.0
            Ms.append(P.T)
    return Ms

def _negacyclic():
    # generator T: (x0..x7)->(-x7,x0..x6)
    T = np.zeros((8, 8))
    for i in range(7):
        T[i + 1, i] = 1.0
    T[0, 7] = -1.0
    Ms = [np.eye(8)]
    cur = np.eye(8)
    for _ in range(15):
        cur = T @ cur if False else cur @ T  # right action powers of T.T? use T.T
        # Actually build powers of T.T for right multiplication
        Ms.append(cur.copy()) if False else None
    # rebuild correctly with right action matrix R = T.T? T acts on column; row action is T.T
    R = T.T
    Ms = [np.linalg.matrix_power(R, k) for k in range(16)]
    return Ms

def _cyclic7():
    Ms = []
    for s in range(7):
        P = np.eye(8)
        Q = np.zeros((8, 8))
        Q[7, 7] = 1.0
        for i in range(7):
            Q[(i + s) % 7, i] = 1.0
        Ms.append(Q.T)
    return Ms

def _prune_to_138(Xexp):
    M = Xexp.shape[0]
    if M <= 138:
        return np.arange(M)
    keep = list(range(M))
    X = Xexp.copy()
    # greedy removal of closest pairs
    while len(keep) > 138:
        K = np.array(keep)
        G = X[K] @ X[K].T
        np.fill_diagonal(G, -2.0)
        # find max pair
        ij = int(np.argmax(G))
        ni = G.shape[0]
        ii, jj = ij // ni, ij % ni
        # decide which to drop: drop one with larger max to others
        ri = G[ii].max()
        rj = G[jj].max()
        # also consider sums
        if rj >= ri:
            drop = jj
        else:
            drop = ii
        keep.pop(drop)
    return np.array(keep, dtype=int)

def _seed_adam(S0, Gs, keep_seed, keep_G, beta_sched, steps_sched, lr=0.02, rng=None):
    # S0 (k,d), Gs list of (d,d), keep arrays length N=138 mapping to (seed, gidx)
    k, d = S0.shape
    S = S0.copy()
    S = _norm_rows(S)
    m1 = np.zeros_like(S)
    v1 = np.zeros_like(S)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    Gstack = np.stack(Gs)  # (ng,d,d)
    Gt = Gstack.transpose(0, 2, 1)  # transpose for backward
    ng = len(Gs)
    # precompute for keep
    ks = np.asarray(keep_seed)
    ga = np.asarray(keep_G)
    N = len(ks)
    for beta, steps in zip(beta_sched, steps_sched):
        for _ in range(int(steps)):
            t += 1
            # expand kept rows: X[i] = S[ks[i]] @ Gs[ga[i]]
            # vectorized: gather S rows then batched matvec
            Sg = S[ks]  # (N,d)
            Gsel = Gstack[ga]  # (N,d,d)
            X = np.einsum('ni,nij->nj', Sg, Gsel)
            # normalize (should be unit)
            Xn = np.linalg.norm(X, axis=1, keepdims=True) + 1e-30
            X = X / Xn
            Gmat = X @ X.T
            np.fill_diagonal(Gmat, -2.0)
            mv = Gmat.max()
            E = np.exp(beta * (Gmat - mv))
            np.fill_diagonal(E, 0.0)
            Ssum = E.sum() + 1e-300
            Gr = (E @ X) / Ssum  # grad wrt X (up to factor)
            # tangent at X
            dot = (Gr * X).sum(axis=1, keepdims=True)
            Gtang = Gr - dot * X
            # backprop to seeds: grad_seed[s] += sum_{i: ks[i]=s} Gtang[i] @ Gs[ga[i]].T
            # Gtang[i] row @ Gsel[i].T
            back = np.einsum('ni,nij->nj', Gtang, Gsel.transpose(0, 2, 1))
            GS = np.zeros_like(S)
            np.add.at(GS, ks, back)
            # project to tangent of S
            dotS = (GS * S).sum(axis=1, keepdims=True)
            GS = GS - dotS * S
            # Adam
            m1 = b1 * m1 + (1 - b1) * GS
            v1 = b2 * v1 + (1 - b2) * (GS * GS)
            m1h = m1 / (1 - b1 ** t)
            v1h = v1 / (1 - b2 ** t)
            S = S - lr * m1h / (np.sqrt(v1h) + eps)
            S = _norm_rows(S)
    return S

def _expand_full(S, Gs):
    k = S.shape[0]
    ng = len(Gs)
    Gstack = np.stack(Gs)
    Sg = np.repeat(S, ng, axis=0)
    Gall = np.tile(Gstack, (k, 1, 1))
    X = np.einsum('ni,nij->nj', Sg, Gall)
    return _norm_rows(X)

def _anneal_unstruct(x, steps, beta0, beta1, lr0, lr1):
    n = x.shape[0]
    for s in range(steps):
        f = s / max(steps - 1, 1)
        beta = beta0 + (beta1 - beta0) * f
        lr = lr0 + (lr1 - lr0) * f
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        m = G.max()
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        Ssum = W.sum() + 1e-300
        Gr = (W @ x) / Ssum
        dot = (Gr * x).sum(axis=1, keepdims=True)
        Tan = Gr - dot * x
        x = x - lr * Tan
        x = _norm_rows(x)
    return x

def _lp_polish(x, deadline):
    x = _norm_rows(x)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        Gfull = x @ x.T
        c = Gfull[iu]
        m = c.max()
        thr = m - max(20.0 * radius, 1e-9)
        sel = c > thr
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I, J = iu[0][sel], iu[1][sel]
        p = len(I)
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
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
        y = _norm_rows(y)
        v = _max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    deadline = t0 + 172.0
    rng = np.random.default_rng(7)
    n, d = 138, 8
    # incumbent
    best_overall = None
    best_val = 1e9
    for fn in ("best.npy", "record.npy"):
        try:
            v = np.load(fn).astype(float)
            if v.shape == (n, d):
                v = _norm_rows(v)
                c = _max_cos(v)
                if c < best_val:
                    best_val = c
                    best_overall = v.copy()
        except Exception:
            pass
    if best_overall is None:
        best_overall = _norm_rows(rng.normal(size=(n, d)))
        best_val = _max_cos(best_overall)

    groups = []
    groups.append(("C8", _cyclic8()))
    groups.append(("C8pm", _cyclic8_pm()))
    groups.append(("B44", _block44()))
    groups.append(("NEG", _negacyclic()))
    groups.append(("C7", _cyclic7()))

    sym_cands = []
    # seed counts: aim ng*k ~ 144
    for gname, Gs in groups:
        if time.time() > deadline - 60:
            break
        ng = len(Gs)
        # choose k
        k = (138 + ng - 1) // ng + 1
        if k < 6:
            k = 6
        if k > 22:
            k = 22
        # try two k values? just one to save time, vary seed init per group
        for trial in range(2):
            if time.time() > deadline - 60:
                break
            S0 = _norm_rows(rng.normal(size=(k, d)))
            Xexp = _expand_full(S0, Gs)
            M = Xexp.shape[0]
            if M < 138:
                # pad with random free rows (handled as extra seeds? just pad later)
                extra = _norm_rows(rng.normal(size=(138 - M, d)))
                Xexp_full = np.vstack([Xexp, extra])
                # for seed opt, use only symmetric part; pad later
                keep = _prune_to_138(Xexp) if M >= 138 else np.arange(M)
            else:
                keep = _prune_to_138(Xexp)
            # map keep to (seed,g)
            # Xexp order: seed major, g minor: idx = s*ng+g
            ks = (keep // ng) % k if M >= 138 else (keep // ng) % k
            # careful when M != k*ng? M==k*ng always here
            ga = keep % ng
            # quick beta schedule (short for trial)
            if trial == 0:
                betas = [30.0, 80.0, 200.0, 400.0]
                stepss = [200, 200, 200, 250]
            else:
                betas = [50.0, 150.0, 400.0]
                stepss = [200, 200, 250]
            try:
                Sopt = _seed_adam(S0, Gs, ks, ga, betas, stepss, lr=0.02)
            except Exception:
                continue
            Xopt_exp = _expand_full(Sopt, Gs)
            if Xopt_exp.shape[0] >= 138:
                keep2 = _prune_to_138(Xopt_exp)
                X138 = Xopt_exp[keep2]
            else:
                need = 138 - Xopt_exp.shape[0]
                X138 = np.vstack([Xopt_exp, _norm_rows(rng.normal(size=(need, d)))])
            # if collapsed duplicates cause cos~1, skip
            v = _max_cos(X138)
            sym_cands.append((v, X138.copy()))
            if v < best_val - 1e-9:
                best_val = v
                best_overall = X138.copy()
            # early exit if time low
            if time.time() > deadline - 55:
                break

    # sort symmetric candidates
    sym_cands.sort(key=lambda t: t[0])
    # symmetry breaking: take top 2 symmetric + incumbent
    break_seeds = []
    for _, Xc in sym_cands[:2]:
        break_seeds.append(Xc)
    break_seeds.append(best_overall.copy())

    # gradual noise + anneal
    amps = [1e-3, 1e-2, 3e-2]
    for Xinit in list(break_seeds):
        if time.time() > deadline - 30:
            break
        X = Xinit.copy()
        for ai, amp in enumerate(amps):
            if time.time() > deadline - 28:
                break
            # tangent noise
            Nn = rng.normal(size=X.shape)
            dot = (Nn * X).sum(axis=1, keepdims=True)
            Nt = Nn - dot * X
            # scale so mean row norm == amp
            nr = np.linalg.norm(Nt, axis=1, keepdims=True) + 1e-30
            Nt = Nt / nr * amp * np.sqrt(7.0) * 0.5
            # for first amp keep close, for larger amp bigger kick
            X = _norm_rows(X + Nt)
            # anneal
            if ai == 0:
                X = _anneal_unstruct(X, steps=500, beta0=30.0, beta1=250.0, lr0=0.02, lr1=0.003)
            elif ai == 1:
                X = _anneal_unstruct(X, steps=700, beta0=40.0, beta1=350.0, lr0=0.015, lr1=0.002)
            else:
                X = _anneal_unstruct(X, steps=800, beta0=60.0, beta1=400.0, lr0=0.012, lr1=0.0015)
            v = _max_cos(X)
            if v < best_val:
                best_val = v
                best_overall = X.copy()
        # extra random-restart anneal from this X with small kick
        if time.time() < deadline - 25:
            try:
                Y = _lp_polish(X, min(deadline - 5, time.time() + 2.5))
                v = _max_cos(Y)
                if v < best_val:
                    best_val = v
                    best_overall = Y.copy()
            except Exception:
                pass

    # final polish of overall best
    try:
        remain = deadline - time.time() - 1.0
        if remain > 3:
            best_overall = _lp_polish(best_overall, time.time() + min(remain, 12.0))
    except Exception:
        pass
    best_overall = _norm_rows(best_overall)
    return best_overall
