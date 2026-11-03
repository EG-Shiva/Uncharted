import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _smooth_opt(X0, beta, maxiter, maxcor=10):
    n, d = X0.shape
    Y0 = X0.copy()
    def fg(Yflat):
        Y = Yflat.reshape(n, d)
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-12)
        X = Y / nrm
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        M = float(np.max(G))
        D = beta * (G - M)
        np.clip(D, -30.0, 0.0, out=D)
        E = np.empty_like(D)
        np.exp(D, out=E)
        np.fill_diagonal(E, 0.0)
        S = float(np.sum(E))
        if S <= 0 or not np.isfinite(S):
            S = 1e-300
        loss = M + np.log(S) / beta
        W = E / S
        GX = 2.0 * (W @ X)
        dot = np.sum(GX * X, axis=1, keepdims=True)
        GY = (GX - dot * X) / nrm
        return loss, GY.ravel()
    try:
        res = minimize(fg, Y0.ravel(), jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "ftol": 1e-20, "gtol": 1e-12,
                                "maxfun": maxiter*5, "maxcor": maxcor})
        Y = res.x.reshape(n, d)
    except Exception:
        Y = Y0
    Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return Y

def _refine_one(x0, Others, steps=60, beta0=3000.0, beta1=20000.0, lr0=0.03, lr1=0.005):
    x = x0 / max(np.linalg.norm(x0), 1e-12)
    lb0, lb1 = np.log(beta0), np.log(beta1)
    for s in range(steps):
        f = s / max(steps-1, 1)
        beta = float(np.exp(lb0 + (lb1-lb0)*f))
        lr = float(lr0 + (lr1-lr0)*f)
        d = Others @ x
        M = float(np.max(d))
        E = np.exp(np.clip((d-M)*beta, -30.0, 0.0))
        S = float(np.sum(E)) + 1e-300
        W = E / S
        g = Others.T @ W
        g = g - float(np.dot(g, x))*x
        x = x - lr*g
        n = float(np.linalg.norm(x))
        if n < 1e-12 or not np.isfinite(n):
            x = x0 / max(np.linalg.norm(x0),1e-12)
            continue
        x /= n
    return x

def _joint_block_adam(Block0, Frozen, steps=120, beta=10000.0, lr=0.008):
    B = Block0.copy()
    B /= np.maximum(np.linalg.norm(B,axis=1,keepdims=True),1e-12)
    K = B.shape[0]
    m = np.zeros_like(B); v = np.zeros_like(B)
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for s in range(steps):
        Dfb = B @ Frozen.T
        Dbb = B @ B.T
        # max over relevant
        M = float(max(np.max(Dfb), np.max(Dbb - np.eye(K)*10.0)))
        Efb = np.exp(np.clip((Dfb-M)*beta, -30.0, 0.0))
        Ebb = np.exp(np.clip((Dbb-M)*beta, -30.0, 0.0))
        np.fill_diagonal(Ebb, 0.0)
        S = float(np.sum(Efb) + 0.5*np.sum(Ebb)) + 1e-300
        Wfb = Efb / S
        Wbb = Ebb / S  # single-count: grad = Wfb@F + Wbb@B
        G = Wfb @ Frozen + Wbb @ B
        # tangential
        G = G - np.sum(G*B, axis=1, keepdims=True)*B
        t += 1
        m = b1*m + (1.0-b1)*G
        v = b2*v + (1.0-b2)*(G*G)
        c1 = 1.0 - b1**t; c2 = 1.0 - b2**t
        step = (lr/c1)*m/(np.sqrt(v/c2)+eps)
        # cap
        sn = np.linalg.norm(step, axis=1, keepdims=True)+1e-16
        cap = 5.0*lr + 1e-12
        step = step*np.minimum(1.0, cap/sn)
        B = B - step
        B /= np.maximum(np.linalg.norm(B,axis=1,keepdims=True),1e-12)
    return B

def solve():
    t_start = time.time()
    REGROW_END = t_start + 140.0
    FINAL_END = t_start + 168.0
    n, d = 250, 6
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape == (n, d):
                a = a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
                cands.append(a)
        except Exception:
            pass
    rng0 = np.random.default_rng(0)
    if not cands:
        z = rng0.standard_normal((n,d)); z/=np.linalg.norm(z,axis=1,keepdims=True)
        cands.append(z)
    def mc(X):
        G=X@X.T; np.fill_diagonal(G,-2.0); return float(np.max(G))
    best = min(cands, key=mc).copy()
    best_val = mc(best)

    sizes_cycle = [25, 15, 40, 25, 40, 15, 25, 30, 20, 40]
    trial = 0
    # quick settle of starting point
    try:
        if time.time() < t_start+20:
            cur = _smooth_opt(best, 8000.0, 60, maxcor=8)
            v = mc(cur)
            if v < best_val:
                best, best_val = cur, v
    except Exception:
        pass

    while time.time() < REGROW_END and trial < 40:
        if time.time() > REGROW_END - 4:
            break
        Kwant = sizes_cycle[trial % len(sizes_cycle)]
        seed = 1000 + trial*17
        rng = np.random.default_rng(seed)
        trial += 1
        try:
            G = best @ best.T
            np.fill_diagonal(G, -2.0)
            # worst pair
            k = int(np.argmax(G))
            a = k // n; b = k % n
            maxv = float(G[a,b])
            thresh = maxv - 2e-5
            # adjacency
            # build component containing a via BFS
            # pairs above thresh
            I, J = np.where(np.triu(G,1) > thresh)
            adj = [[] for _ in range(n)]
            for u,v_ in zip(I,J):
                adj[int(u)].append(int(v_)); adj[int(v_)].append(int(u))
            seen = np.zeros(n, dtype=bool)
            stack = [int(a)]
            seen[int(a)] = True
            comp = []
            while stack:
                u = stack.pop(); comp.append(u)
                for w in adj[u]:
                    if not seen[w]:
                        seen[w]=True; stack.append(w)
            comp = list(comp)
            rowmax = G.max(axis=1)
            # order comp by rowmax desc
            comp_sorted = sorted(comp, key=lambda u: -rowmax[u])
            victims = []
            victims.append(int(a)); victims.append(int(b))
            for u in comp_sorted:
                if len(victims) >= Kwant: break
                if u not in victims:
                    victims.append(int(u))
            # if comp too small, fill with global worst
            if len(victims) < Kwant:
                order = np.argsort(-rowmax)
                for u in order:
                    u=int(u)
                    if u not in victims:
                        victims.append(u)
                    if len(victims)>=Kwant: break
            victims = victims[:Kwant]
            K = len(victims)
            vset = set(victims)
            keep_idx = [i for i in range(n) if i not in vset]
            Frozen = best[keep_idx].copy()
            # regrow block jointly
            Block_list = []
            # pre-stack for speed: maintain current combined
            for ii in range(K):
                if time.time() > REGROW_END - 2:
                    break
                if Block_list:
                    Bp = np.stack(Block_list, axis=0)
                    C = np.vstack([Frozen, Bp])
                else:
                    C = Frozen
                cand = rng.standard_normal((8192, d))
                cand /= np.maximum(np.linalg.norm(cand,axis=1,keepdims=True),1e-12)
                D = cand @ C.T
                mx = D.max(axis=1)
                bi = int(np.argmin(mx))
                x0 = cand[bi]
                xr = _refine_one(x0, C, steps=60)
                Block_list.append(xr)
                if len(Block_list)%5==0 and len(Block_list)>=5:
                    try:
                        Bp2 = np.stack(Block_list, axis=0)
                        Bp2 = _joint_block_adam(Bp2, Frozen, steps=80, beta=9000.0, lr=0.008)
                        Block_list = [Bp2[i] for i in range(Bp2.shape[0])]
                    except Exception:
                        pass
            if len(Block_list) != K:
                continue
            Bp = np.stack(Block_list, axis=0)
            try:
                Bp = _joint_block_adam(Bp, Frozen, steps=120, beta=12000.0, lr=0.006)
            except Exception:
                pass
            Xnew = best.copy()
            for vi, xv in zip(victims, Bp):
                Xnew[vi] = xv/np.maximum(np.linalg.norm(xv),1e-12)
            # short collective polish (smooth, small iters)
            try:
                Xnew = _smooth_opt(Xnew, 9000.0, 50, maxcor=8)
            except Exception:
                pass
            vnew = mc(Xnew)
            if vnew < best_val:
                best = Xnew; best_val = vnew
        except Exception:
            continue

    # final short polish
    if time.time() < FINAL_END - 6:
        try:
            cur = _smooth_opt(best, 30000.0, 50, maxcor=8)
            v = mc(cur)
            if v < best_val:
                best, best_val = cur, v
        except Exception:
            pass
    best = best/np.maximum(np.linalg.norm(best,axis=1,keepdims=True),1e-12)
    best = np.where(np.isfinite(best), best, 0.0)
    best = best/np.maximum(np.linalg.norm(best,axis=1,keepdims=True),1e-12)
    return best
