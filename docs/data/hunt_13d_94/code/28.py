import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _adam_relax(X, steps, a0, a1, lr0, lr1, deadline=None):
    n, d = X.shape
    m1 = np.zeros_like(X)
    m2 = np.zeros_like(X)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(steps):
        if deadline is not None and (s & 127) == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        G = X @ X.T
        np.fill_diagonal(G, -10.0)
        mv = float(G.max())
        W = np.exp(alpha * (G - mv))
        np.fill_diagonal(W, 0.0)
        S = W.sum()
        if not np.isfinite(S) or S == 0:
            break
        W = W / S
        grad = W @ X
        dot = (grad * X).sum(axis=1, keepdims=True)
        gt = grad - dot * X
        m1 = b1 * m1 + (1 - b1) * gt
        m2 = b2 * m2 + (1 - b2) * (gt * gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        X = X - lr * (m1 / bc1) / (np.sqrt(m2 / bc2) + eps)
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
        if not np.all(np.isfinite(X)):
            break
    return X

def _batch_cavity_opt(Rinit, F, steps, a0, a1, lr0, lr1, deadline=None):
    # Rinit: (B,k,d), F: (m,d) frozen
    B, k, d = Rinit.shape
    R = Rinit.copy()
    m1 = np.zeros_like(R)
    m2 = np.zeros_like(R)
    b1, b2, eps = 0.9, 0.999, 1e-8
    idx = np.arange(k)
    for s in range(steps):
        if deadline is not None and (s & 127) == 0 and time.time() > deadline:
            break
        frac = s / max(1, steps - 1)
        alpha = a0 * (a1 / a0) ** frac
        lr = lr0 * (lr1 / lr0) ** frac
        # dots
        RF = np.matmul(R, F.T)  # (B,k,m)
        RR = np.matmul(R, np.transpose(R, (0, 2, 1)))  # (B,k,k)
        RR[:, idx, idx] = -10.0
        maxRF = RF.max(axis=(1, 2))
        maxRR = RR.max(axis=(1, 2))
        mv = np.maximum(maxRF, maxRR)  # (B,)
        mvE = mv[:, None, None]
        WRF = np.exp(alpha * (RF - mvE[:, :, 0:1].reshape(B, 1, 1) if False else mvE))
        # above: use mvE
        WRF = np.exp(alpha * (RF - mvE))
        WRR = np.exp(alpha * (RR - mvE))
        WRR[:, idx, idx] = 0.0
        S = WRF.sum(axis=(1, 2)) + WRR.sum(axis=(1, 2))
        S[~np.isfinite(S)] = 1.0
        S[S == 0] = 1.0
        Grf = np.matmul(WRF, F)  # (B,k,d)
        Wt = WRR + np.transpose(WRR, (0, 2, 1))
        Grr = np.matmul(Wt, R)  # (B,k,k)@(B,k,d)
        Grad = (Grf + Grr) / S[:, None, None]
        dot = (Grad * R).sum(axis=2, keepdims=True)
        Gt = Grad - dot * R
        m1 = b1 * m1 + (1 - b1) * Gt
        m2 = b2 * m2 + (1 - b2) * (Gt * Gt)
        bc1 = 1.0 - b1 ** (s + 1)
        bc2 = 1.0 - b2 ** (s + 1)
        R = R - lr * (m1 / bc1) / (np.sqrt(m2 / bc2) + eps)
        nr = np.linalg.norm(R, axis=2, keepdims=True)
        nr[nr == 0] = 1.0
        R = R / nr
        if not np.all(np.isfinite(R)):
            break
    return R

def _lp_polish(x, deadline):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n, d = x.shape
    best = _max_cos(x)
    radius = 1e-3
    iu = np.triu_indices(n, 1)
    while time.time() < deadline and radius > 1e-14:
        cmat = (x @ x.T)[iu]
        m = float(cmat.max())
        sel = cmat > m - max(20.0 * radius, 1e-9)
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
            break
        obj = np.zeros(nv)
        obj[-1] = 1.0
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-cmat[sel], A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = x + res.x[:-1].reshape(n, d)
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        v = _max_cos(y)
        if v < best:
            x, best, radius = y, v, min(radius * 1.5, 0.05)
        else:
            radius *= 0.5
    return x, best

def solve():
    T0 = time.time()
    BUDGET = 176.0
    HOP_END = T0 + 148.0
    rng_global = np.random.default_rng(7)
    best = None
    best_val = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.load(fn).astype(float)
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            if a.shape == (94, 13) and np.all(np.isfinite(a)):
                v = _max_cos(a)
                if v < best_val:
                    best_val = v
                    best = a.copy()
        except Exception:
            pass
    if best is None:
        r = np.random.default_rng(0).standard_normal(size=(94, 13))
        r /= np.linalg.norm(r, axis=1, keepdims=True)
        best = r
        best_val = _max_cos(best)
    n, d = best.shape
    global_best = best.copy()
    global_val = best_val
    elites = [(global_val, global_best.copy())]

    k_cycle = [12, 16, 8, 12, 16, 8]
    round_idx = 0
    B = 32
    while time.time() < HOP_END:
        if time.time() > HOP_END - 5:
            break
        cur_base = global_best
        G = cur_base @ cur_base.T
        np.fill_diagonal(G, -2.0)
        m = float(G.max())
        cnt = (G > m - 1e-4).sum(axis=1)
        worst = G.max(axis=1)
        # seed: high contact degree, random among top 6
        order_deg = np.argsort(-cnt, kind="stable")
        # tie-break by worst
        top = order_deg[:6]
        # deterministic pick cycling + random
        seed = int(top[rng_global.integers(0, len(top))])
        k = int(k_cycle[round_idx % len(k_cycle)])
        round_idx += 1
        # connected cluster: seed + closest angular neighbours
        dots_seed = (cur_base @ cur_base[seed])
        dots_seed[seed] = 2.0
        order_close = np.argsort(-dots_seed)
        cluster = np.empty(k, dtype=int)
        cluster[0] = seed
        c = 1
        for j in order_close:
            if j == seed:
                continue
            cluster[c] = int(j)
            c += 1
            if c >= k:
                break
        cluster = np.sort(cluster)
        mask = np.ones(n, dtype=bool)
        mask[cluster] = False
        F = cur_base[mask]  # (n-k,d)
        Rorig = cur_base[cluster]  # (k,d)
        c0 = Rorig.mean(axis=0)
        nn = np.linalg.norm(c0)
        if nn == 0 or not np.isfinite(nn):
            c0 = cur_base[seed]
        else:
            c0 = c0 / nn
        rng = np.random.default_rng(1000 + round_idx * 137)
        Rinit = np.empty((B, k, d))
        sigmas_pert = [0.05, 0.08, 0.12, 0.18, 0.25, 0.30, 0.40, 0.50]
        for b in range(B):
            if b < 8:
                sig = sigmas_pert[b]
                v = Rorig + sig * rng.standard_normal(size=(k, d))
            elif b < 24:
                sig = 0.4 + 0.6 * rng.random()
                # cavity-biased: around c0 with spread
                v = c0[None, :] + sig * rng.standard_normal(size=(k, d))
                # add small repulsion diversity: random rotation not needed
            else:
                v = rng.standard_normal(size=(k, d))
            nr = np.linalg.norm(v, axis=1, keepdims=True)
            nr[nr == 0] = 1.0
            Rinit[b] = v / nr
        # joint re-optimization with frozen background
        try:
            Ropt = _batch_cavity_opt(Rinit, F, steps=1000, a0=20.0, a1=320.0, lr0=0.02, lr1=0.0025, deadline=HOP_END)
        except Exception:
            continue
        # rank completions by exact max cosine
        vals = np.empty(B)
        dists = np.empty(B)
        for b in range(B):
            Xb = np.empty_like(cur_base)
            Xb[mask] = F
            Xb[cluster] = Ropt[b] / np.linalg.norm(Ropt[b], axis=1, keepdims=True)
            vals[b] = _max_cos(Xb)
            # distance from original cluster: 1 - mean best-match dot
            D = Ropt[b] @ Rorig.T
            best_match = D.max(axis=1).mean()
            dists[b] = 1.0 - float(best_match)
        order = np.argsort(vals, kind="stable")
        # pick best 4 with diversity: take top by val, ensure spread
        cand_idx = list(order[:8])
        # rank primarily by val; prefer diverse for 4th slot
        chosen = list(order[:3])
        # 4th: among order[3:8] pick one with dist>0.02 and smallest val if exists
        fourth = int(order[3])
        for oi in order[3:8]:
            if dists[int(oi)] > 0.02 and vals[int(oi)] < vals[fourth] + 5e-4:
                fourth = int(oi)
                break
        if fourth not in chosen:
            chosen.append(fourth)
        if len(chosen) < 4:
            chosen.append(int(order[3]))
        # unfreeze and refine only best 3-5
        for bi in chosen:
            if time.time() > HOP_END - 2:
                break
            Xb = np.empty_like(cur_base)
            Xb[mask] = F
            Xb[cluster] = Ropt[bi] / np.linalg.norm(Ropt[bi], axis=1, keepdims=True)
            # short unfreeze refine
            try:
                Xr = _adam_relax(Xb.copy(), steps=700, a0=30.0, a1=300.0, lr0=0.008, lr1=0.0012, deadline=HOP_END)
            except Exception:
                continue
            v = _max_cos(Xr)
            if v < global_val - 1e-12:
                global_val = v
                global_best = Xr.copy()
                elites.append((v, Xr.copy()))
                elites.sort(key=lambda t: t[0])
                del elites[5:]
            elif v < elites[-1][0] + 2e-4 and len(elites) < 5:
                # keep near-elite for final polish diversity (only occasionally)
                if rng_global.random() < 0.3:
                    elites.append((v, Xr.copy()))
                    elites.sort(key=lambda t: t[0])
                    del elites[5:]
        # optional: if no improvement, continue (next seed explores different cavity)
    # final short LP polish on elites
    elites.sort(key=lambda t: t[0])
    keep = elites[:3]
    total_end = T0 + BUDGET - 2.0
    remaining = max(0.0, total_end - time.time())
    per = remaining / max(1, len(keep)) if keep else 0
    per = min(max(per, 4.0), 18.0)
    final_best = global_best
    final_val = global_val
    for vv, XX in keep:
        if time.time() > total_end - 2.0:
            break
        dl = min(time.time() + per, total_end)
        try:
            P, pv = _lp_polish(XX.copy(), dl)
        except Exception:
            continue
        if pv < final_val:
            final_val = pv
            final_best = P.copy()
    final_best = final_best / np.linalg.norm(final_best, axis=1, keepdims=True)
    return final_best
