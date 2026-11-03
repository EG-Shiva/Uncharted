import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds, eigsh

def max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def adam_quench(X, steps=500, beta0=40.0, beta1=300.0, lr0=0.004, lr1=0.0015, seed=0):
    n, d = X.shape
    x = X.copy()
    m = np.zeros_like(x)
    vv = np.zeros_like(x)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(steps):
        frac = t / max(1, steps - 1)
        beta = beta0 * (beta1 / beta0) ** frac
        lrt = lr0 * (1.0 - frac) + lr1 * frac
        G = x @ x.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        E = np.exp(beta * (G - M))
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E) * 0.5)
        if not np.isfinite(S) or S <= 0:
            break
        Grad = (E @ x) / S
        dot = np.sum(Grad * x, axis=1, keepdims=True)
        Gt = Grad - dot * x
        m = b1 * m + (1 - b1) * Gt
        vv = b2 * vv + (1 - b2) * (Gt * Gt)
        mhat = m / (1 - b1 ** (t + 1))
        vhat = vv / (1 - b2 ** (t + 1))
        step = mhat / (np.sqrt(vhat) + eps)
        np.clip(step, -3.0, 3.0, out=step)
        x = x - lrt * step
        nr = np.linalg.norm(x, axis=1, keepdims=True)
        nr[~np.isfinite(nr)] = 1.0
        nr[nr < 1e-12] = 1.0
        x /= nr
        if not np.all(np.isfinite(x)):
            break
    nr = np.linalg.norm(x, axis=1, keepdims=True)
    nr[nr < 1e-12] = 1.0
    return x / nr

def lp_polish(X, deadline, r_init=0.004, cap=2500):
    x = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = x.shape
    try:
        best = max_cos(x)
    except Exception:
        return x
    radius = r_init
    iu0 = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        G = x @ x.T
        c_all = G[iu0]
        m = float(c_all.max())
        thresh = m - max(20.0 * radius, 1e-9)
        sel = c_all > thresh
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            break
        if cnt > cap:
            idx = np.where(sel)[0]
            part = np.argpartition(-c_all[idx], cap)[:cap]
            keep = np.zeros_like(sel)
            keep[idx[part]] = True
            sel = keep
            cnt = cap
        I = iu0[0][sel]; J = iu0[1][sel]
        c_sel = c_all[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.empty((p, 2 * d + 1), dtype=int)
        ar = np.arange(d)
        cols[:, :d] = I[:, None] * d + ar
        cols[:, d:2*d] = J[:, None] * d + ar
        cols[:, 2*d] = n * d
        cols = cols.ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        except Exception:
            radius *= 0.5
            continue
        obj = np.zeros(nv); obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs",
                          options={"time_limit": 5.0})
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.isfinite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y /= yn
        v = max_cos(y)
        if v < best:
            x, best = y, v
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return x

def solve():
    t0 = time.time()
    HARD = t0 + 172.0
    rng = np.random.default_rng(7)
    n, d = 176, 9
    dm = d - 1
    X0 = None
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d) and np.all(np.isfinite(a)):
                nr = np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(nr > 1e-12):
                    X0 = a / nr
                    break
        except Exception:
            continue
    if X0 is None:
        X0 = rng.standard_normal((n, d))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
    best_X = X0.copy()
    best_v = max_cos(best_X)
    X = best_X.copy()

    # --- active contact graph slack<1e-3 ---
    try:
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m0 = float(np.max(G))
        iu = np.triu_indices(n, 1)
        c_all = G[iu]
        sel_active = c_all > (m0 - 1e-3)
        IA = iu[0][sel_active]; JA = iu[1][sel_active]
        cA = c_all[sel_active]
        p = len(IA)
        # degrees / hubs
        deg = np.zeros(n, dtype=int)
        np.add.at(deg, IA, 1); np.add.at(deg, JA, 1)
        order_hub = np.argsort(-deg)
        # adjacency + laplacian for spectral partition
        fied = None
        try:
            adj = np.zeros((n, n), dtype=float)
            adj[IA, JA] = 1.0; adj[JA, IA] = 1.0
            Dv = adj.sum(axis=1)
            L = np.diag(Dv) - adj
            Lc = csr_matrix(L)
            k_eig = 2
            wv, Vv = eigsh(Lc, k=k_eig, which='SM', tol=1e-4, maxiter=2000)
            # fiedler = vector with smallest non-zero
            idxs = np.argsort(wv)
            Vv = Vv[:, idxs]
            # first should be ~0 constant; take second
            fied = Vv[:, 1] if Vv.shape[1] > 1 else Vv[:, 0]
            if not np.all(np.isfinite(fied)):
                fied = None
        except Exception:
            fied = None
        # --- tangent bases ---
        Rfix = rng.standard_normal((d, dm))
        Tb = np.zeros((n, d, dm))
        for i in range(n):
            xi = X[i]
            B = Rfix - xi[:, None] * (xi @ Rfix)
            Q, _ = np.linalg.qr(B)
            # Q is d x dm (qr of d x dm gives d x dm)
            Tb[i] = Q
        # --- rigidity Jacobian in tangent coords ---
        Modes = None
        if p >= 10:
            Nvar = n * dm
            kk = int(min(24, min(p, Nvar) - 2))
            if kk < 6:
                kk = int(min(p, Nvar) - 1)
            kk = max(2, kk)
            kk = min(kk, 24)
            try:
                At = np.zeros((p, Nvar), dtype=float)
                # loop
                for r in range(p):
                    i = int(IA[r]); j = int(JA[r]); c = float(cA[r])
                    ai = X[j] - c * X[i]
                    aj = X[i] - c * X[j]
                    At[r, i*dm:(i+1)*dm] = ai @ Tb[i]
                    At[r, j*dm:(j+1)*dm] = aj @ Tb[j]
                # scale rows to unit for stability?
                rn = np.linalg.norm(At, axis=1, keepdims=True)
                rn[rn < 1e-12] = 1.0
                At /= rn
                Acs = csr_matrix(At)
                # smallest singular modes
                uu, ss, vtt = svds(Acs, k=kk, which='SM', tol=1e-4, maxiter=1500)
                # vtt is kk x Nvar, sorted ascending? svds returns descending? sort by ss
                o = np.argsort(ss)
                ss = ss[o]; vtt = vtt[o]
                Modes = vtt  # smallest first
            except Exception:
                Modes = None
        if Modes is None:
            # random tangent modes fallback
            kk = 16
            Modes = rng.standard_normal((kk, n * dm)) * 0.1
        # --- BFS distances for community truncation ---
        # build neighbor list
        nbrs = [[] for _ in range(n)]
        for a, b in zip(IA.tolist(), JA.tolist()):
            nbrs[a].append(b); nbrs[b].append(a)
        from collections import deque
        def bfs_dist(src):
            dist = np.full(n, 999, dtype=int)
            dist[src] = 0
            dq = deque([src])
            while dq:
                u = dq.popleft()
                for w in nbrs[u]:
                    if dist[w] > dist[u] + 1:
                        dist[w] = dist[u] + 1
                        dq.append(w)
            return dist
        # precompute hub list
        hubs = order_hub[:10].tolist()
        # communities per hub
        comms = []
        for h in hubs:
            dist = bfs_dist(h)
            if fied is not None:
                same = (fied * fied[h] > 0)
                cand = np.where(same)[0]
                # restrict to within 4 hops
                cand = cand[dist[cand] <= 4]
                if len(cand) < 8:
                    cand = np.where(dist <= 2)[0]
                if len(cand) > 48:
                    # keep closest: sort by dist then |fied diff|
                    score = dist[cand].astype(float) * 10.0 + np.abs(fied[cand] - fied[h])
                    keep = np.argsort(score)[:48]
                    cand = cand[keep]
            else:
                cand = np.where(dist <= 2)[0]
                if len(cand) > 48:
                    cand = cand[:48]
            comms.append(np.asarray(cand, dtype=int))
        # --- propose collective moves ---
        amps = [0.01, 0.03, 0.06, 0.1]
        nmodes = Modes.shape[0]
        props = []  # (score, Xcand)
        nprop = 90
        for t in range(nprop):
            if time.time() > HARD - 60.0:
                break
            hi = int(t % len(comms))
            C = comms[hi]
            # mode mixture: 1-3 modes
            nmix = int(rng.integers(1, 4))
            midx = rng.choice(nmodes, size=nmix, replace=False)
            coeff = rng.standard_normal(nmix)
            y = np.zeros(n * dm)
            for ci_, mi_ in zip(coeff, midx):
                y += ci_ * Modes[int(mi_)]
            # occasionally add random component
            if rng.random() < 0.25:
                y += 0.5 * rng.standard_normal(y.shape)
            Y = y.reshape(n, dm)
            Dfull = np.einsum('ijk,ik->ij', Tb, Y)  # tangent displacements
            # scale community part to unit max
            dn = np.linalg.norm(Dfull[C], axis=1)
            mx = float(np.max(dn)) if len(C) > 0 else 0.0
            if not np.isfinite(mx) or mx < 1e-12:
                continue
            amp = amps[int(rng.integers(len(amps)))]
            # randomised sign already in coeff; also random flip
            if rng.random() < 0.5:
                amp = -amp
            D = np.zeros((n, d))
            D[C] = (amp / mx) * Dfull[C]
            # variant: move complement opposite slightly
            if rng.random() < 0.4:
                comp = np.ones(n, dtype=bool); comp[C] = False
                dn2 = np.linalg.norm(Dfull[comp], axis=1)
                mx2 = float(np.max(dn2)) if np.any(comp) else 0.0
                if np.isfinite(mx2) and mx2 > 1e-12:
                    D[comp] = -0.3 * (amp / mx2) * Dfull[comp]
            Y2 = X + D
            yn = np.linalg.norm(Y2, axis=1, keepdims=True)
            yn[~np.isfinite(yn)] = 1.0
            yn[yn < 1e-12] = 1.0
            Y2 /= yn
            v = max_cos(Y2)
            props.append((v, Y2))
            # also try pure random community kick occasionally
            if t % 12 == 5:
                D2 = np.zeros((n, d))
                R = rng.standard_normal((len(C), d))
                # project to tangent
                for k2, idx_ in enumerate(C):
                    R[k2] -= np.dot(R[k2], X[idx_]) * X[idx_]
                rn2 = np.linalg.norm(R, axis=1, keepdims=True)
                rn2[rn2 < 1e-12] = 1.0
                R /= rn2
                amp2 = amps[int(rng.integers(len(amps)))] * 0.7
                D2[C] = amp2 * R
                Y3 = X + D2
                Y3 /= np.linalg.norm(Y3, axis=1, keepdims=True)
                props.append((max_cos(Y3), Y3))
        # rank props
        if len(props) > 0:
            props.sort(key=lambda z: z[0])
            # keep top survivors for Adam screening (up to 8, must be reasonable)
            K = min(8, len(props))
            surv = [props[k][1] for k in range(K)]
        else:
            surv = []
    except Exception:
        surv = []
        best_v = max_cos(best_X)

    # --- screen survivors with 500 Adam steps ---
    screened = []
    try:
        for s in surv:
            if time.time() > HARD - 45.0:
                break
            q = adam_quench(s, steps=500, beta0=40.0, beta1=300.0, lr0=0.004, lr1=0.0015)
            vq = max_cos(q)
            screened.append((vq, q))
        # also screen current best lightly? no
        if len(screened) > 0:
            screened.sort(key=lambda z: z[0])
    except Exception:
        screened = []

    # pick single best screened candidate for LP polish
    lp_seed = None
    if len(screened) > 0:
        lp_seed = screened[0][1]
        # update best if screened already better
        if screened[0][0] < best_v and np.all(np.isfinite(screened[0][1])):
            best_X = screened[0][1].copy(); best_v = screened[0][0]
        # keep second as fallback? only polish best per spec
    else:
        # fallback: small jittered best
        try:
            y = best_X + 0.003 * rng.standard_normal((n, d))
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            y = adam_quench(y, steps=300)
            v = max_cos(y)
            if v < best_v:
                best_X, best_v = y, v
            lp_seed = y
        except Exception:
            lp_seed = best_X

    # --- LP-polish only single best under hard deadline ---
    try:
        if lp_seed is not None and time.time() < HARD - 8.0:
            dl = HARD - 3.0
            # choose seed: best screened if it is at least close to best
            seed = lp_seed
            # if lp_seed much worse than best, polish best instead
            try:
                vs = max_cos(seed)
                if vs > best_v + 0.005:
                    seed = best_X
            except Exception:
                seed = best_X
            p = lp_polish(seed, dl, r_init=0.004)
            v = max_cos(p)
            if v < best_v and np.all(np.isfinite(p)):
                best_X, best_v = p, v
        # final tiny polish of overall best if time remains
        if time.time() < HARD - 6.0:
            dl = HARD - 2.0
            p2 = lp_polish(best_X, dl, r_init=0.0015)
            v2 = max_cos(p2)
            if v2 < best_v and np.all(np.isfinite(p2)):
                best_X, best_v = p2, v2
    except Exception:
        pass

    best_X /= np.linalg.norm(best_X, axis=1, keepdims=True)
    return best_X
