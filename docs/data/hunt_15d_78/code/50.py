import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _topk_sorted(X, k=200):
    G = X @ X.T
    iu = np.triu_indices(X.shape[0], 1)
    v = G[iu]
    v = np.sort(v)[::-1]
    return v[:k]

def _lp_polish_deadline(X, deadline, r_init=0.006):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    best = _max_cos(X)
    best_x = X.copy()
    radius = r_init
    iu0, iu1 = np.triu_indices(n, 1)
    nv = n * d + 1
    obj = np.zeros(nv); obj[-1] = 1.0
    while time.time() + 0.5 < deadline and radius > 1e-13:
        G = X @ X.T
        c_all = G[iu0, iu1]
        m = float(c_all.max())
        sel = c_all > m - max(20.0 * radius, 1e-9)
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius *= 0.5
            continue
        if cnt > 2500:
            idx = np.where(sel)[0]
            order = np.argsort(-c_all[idx])[:2500]
            mask = np.zeros_like(sel, dtype=bool)
            mask[idx[order]] = True
            sel = mask
        I = iu0[sel]; J = iu1[sel]
        p = len(I)
        c_sel = c_all[sel]
        cols = np.concatenate([I[:, None] * d + np.arange(d),
                               J[:, None] * d + np.arange(d),
                               np.full((p, 1), n * d)], axis=1).ravel()
        rows = np.repeat(np.arange(p), 2 * d + 1)
        vals = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
        A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
        bounds = [(-radius, radius)] * (n * d) + [(None, None)]
        try:
            res = linprog(obj, A_ub=A_ub, b_ub=-c_sel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=bounds, method="highs")
        except Exception:
            radius *= 0.5
            continue
        if res.status != 0:
            radius *= 0.5
            continue
        y = X + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        if not np.all(np.finite(yn)) or np.any(yn < 1e-12):
            radius *= 0.5
            continue
        y = y / yn
        v = _max_cos(y)
        if v < best:
            X = y; best = v; best_x = y.copy()
            radius = min(radius * 1.5, 0.03)
        else:
            radius *= 0.5
    return best_x, best


def solve():
    t0 = time.time()
    HARD_END = t0 + 170.0
    # total cap 160s for search+polish, leave 10s margin
    POOL_END = t0 + 160.0
    rng = np.random.default_rng(0)
    n, d = 78, 15
    B = 32

    # load fallback / record fingerprint
    rec_top = None
    bestX = None
    bestV = 1e9
    for fn in ["best.npy", "record.npy"]:
        try:
            a = np.asarray(np.load(fn), dtype=np.float64)
            if a.shape != (78, 15):
                continue
            a = a / np.linalg.norm(a, axis=1, keepdims=True)
            v = _max_cos(a)
            if fn == "record.npy":
                try:
                    rec_top = _topk_sorted(a, 200)
                except Exception:
                    rec_top = None
            if v < bestV:
                bestV = v
                bestX = a.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    champion = bestX.copy()
    champV = float(bestV)

    # ---- batched fresh multi-start ----
    Xb = rng.standard_normal((B, n, d))
    Xb /= np.linalg.norm(Xb, axis=2, keepdims=True)
    Mb = np.zeros_like(Xb)
    Vb = np.zeros_like(Xb)
    iu0, iu1 = np.triu_indices(n, 1)
    NP = len(iu0)
    Itgt = np.eye(d) / float(d)
    b1, b2, eps = 0.9, 0.999, 1e-8
    S = 750
    lam1_base, lam2_base = 5.0, 15.0
    # beta schedule exponential 20 -> 300
    b_start, b_end = 20.0, 300.0
    lr0 = 0.007

    for step in range(S):
        if time.time() > POOL_END - 40.0:
            break  # need time for LP
        frac = step / max(1, S - 1)
        beta = b_start * (b_end / b_start) ** frac
        lr = lr0 * (1.0 - 0.6 * frac)
        lam_scale = 1.0 if frac < 0.7 else 0.3
        lam1 = lam1_base * lam_scale
        lam2 = lam2_base * lam_scale
        # Gram batched
        Gt = np.matmul(Xb, Xb.transpose(0, 2, 1))  # B,N,N
        Gtri = Gt[:, iu0, iu1]  # B,NP
        m = Gtri.max(axis=1)  # B
        E = np.exp(beta * (Gtri - m[:, None]))
        Ssum = E.sum(axis=1, keepdims=True)
        Ssum = np.maximum(Ssum, 1e-300)
        Wtri = E / Ssum  # B,NP
        # build dense W
        W = np.zeros((B, n, n), dtype=np.float64)
        W[:, iu0, iu1] = Wtri
        W = W + W.transpose(0, 2, 1)
        Grad = np.matmul(W, Xb)  # B,N,D
        # covariance penalty
        # C = X^T X / n
        Ct = np.matmul(Xb.transpose(0, 2, 1), Xb) / float(n)  # B,D,D
        Diff = Ct - Itgt[None, :, :]
        Grad_cov = (4.0 / float(n)) * np.matmul(Xb, Diff)
        Grad = Grad + lam1 * Grad_cov
        # Var_i(max) penalty (term1 only for speed)
        # mask diag for personal max
        # use Gt with diag set to -2
        # set diag efficiently
        # Gt currently has diag 1; create personal max via max over axis2 after masking
        # avoid copy of full B,N,N? reuse via Gtri? need argmax per row: use masked Gt
        # do in-place on a copy view? make copy only for max: use tri trick is messy, just compute:
        # we can get p via blocked max: since diag is 1 which would be max, must mask.
        # cheap: set diagonal to -2 in Gt copy
        # Gt shape B,N,N - setting diag for all B
        idx = np.arange(n)
        Gt[:, idx, idx] = -2.0
        pmax = Gt.max(axis=2)  # B,N
        jstar = Gt.argmax(axis=2)  # B,N
        pmean = pmax.mean(axis=1, keepdims=True)  # B,1
        coeff = 2.0 * (pmax - pmean) / float(n)  # B,N
        # gather X_jstar
        # X_jstar[b,i,:] = Xb[b, jstar[b,i], :]
        # vectorized gather:
        bidx = np.arange(B)[:, None, None]
        # expand? use advanced indexing
        # Xb: B,N,D ; jstar: B,N
        Xj = Xb[np.arange(B)[:, None], jstar]  # B,N,D ?
        # numpy: Xb[rows, cols] with rows shape B,N and cols shape B,N gives shape B,N,D? test: should broadcast.
        # fallback safe loop if shape wrong
        try:
            if Xj.shape != (B, n, d):
                raise ValueError("shape")
        except Exception:
            Xj = np.empty_like(Xb)
            for b in range(B):
                Xj[b] = Xb[b][jstar[b]]
        Grad_var1 = coeff[:, :, None] * Xj
        Grad = Grad + lam2 * Grad_var1
        # tangent projection
        dot = (Grad * Xb).sum(axis=2, keepdims=True)
        Gtang = Grad - dot * Xb
        # Adam
        Mb = b1 * Mb + (1.0 - b1) * Gtang
        Vb = b2 * Vb + (1.0 - b2) * (Gtang * Gtang)
        bc1 = 1.0 - b1 ** (step + 1)
        bc2 = 1.0 - b2 ** (step + 1)
        Mh = Mb / bc1
        Vh = Vb / bc2
        upd = lr * Mh / (np.sqrt(Vh) + eps)
        # keep update tangent (project again to reduce normal drift)
        udot = (upd * Xb).sum(axis=2, keepdims=True)
        upd = upd - udot * Xb
        # clip step to avoid blowup
        un = np.linalg.norm(upd, axis=2, keepdims=True)
        # clip per-row to 0.05
        clip = np.minimum(1.0, 0.05 / np.maximum(un, 1e-12))
        upd = upd * clip
        Xb = Xb - upd
        Xb /= np.linalg.norm(Xb, axis=2, keepdims=True)

    # ---- evaluate replicas ----
    scores = np.empty(B)
    eig_spread = np.empty(B)
    deg_var = np.empty(B)
    tops = np.empty((B, 200))
    for b in range(B):
        X = Xb[b] / np.linalg.norm(Xb[b], axis=1, keepdims=True)
        Xb[b] = X
        scores[b] = _max_cos(X)
        try:
            C = (X.T @ X) / float(n)
            ev = np.linalg.eigvalsh(C)
            ev = np.maximum(ev, 1e-12)
            eig_spread[b] = float(ev.max() / ev.min())
        except Exception:
            eig_spread[b] = 1e9
        try:
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            mm = G.max()
            degs = (G > mm - 1e-4).sum(axis=1)
            deg_var[b] = float(degs.var())
        except Exception:
            deg_var[b] = 1e9
        try:
            tops[b] = _topk_sorted(X, 200)
        except Exception:
            tops[b] = 1.0

    order = np.argsort(scores, kind="stable")
    # fingerprint distance to record
    def is_dup_record(b):
        if rec_top is None:
            return False
        diff = tops[b] - rec_top
        # L2 and max
        if np.linalg.norm(diff) < 0.004 and np.max(np.abs(diff)) < 0.0015:
            return True
        # also if max within 2e-7 and very close top200 -> duplicate basin
        if abs(scores[b] - champV) < 3e-7 and np.linalg.norm(diff) < 0.01:
            return True
        return False

    selected = []
    for b in order:
        if len(selected) >= 3:
            break
        if is_dup_record(int(b)):
            continue
        # distinct from already selected
        ok = True
        for s in selected:
            dd = np.linalg.norm(tops[int(b)] - tops[int(s)])
            if dd < 0.004:
                ok = False
                break
        if not ok:
            continue
        # prefer balanced: skip extreme eigen spread / deg var? allow but penalize
        # accept if not crazy: eig_spread < 8 and deg_var < 500, else still accept if score very good
        selected.append(int(b))
    # if all were duplicates (basin not escaped), take best non-record-duplicate by balanced score
    if len(selected) == 0:
        # balanced score: maxcos + small isotropy terms
        bal = scores + 1e-4 * np.log10(np.maximum(eig_spread, 1.0)) + 1e-6 * deg_var
        ob = np.argsort(bal, kind="stable")
        for b in ob[:3]:
            if not is_dup_record(int(b)):
                selected.append(int(b))
            if len(selected) >= 2:
                break
    # fallback: if still empty, polish champion only
    # LP-polish best 2-3 distinct replicas
    cands = []
    for b in selected:
        cands.append((float(scores[b]), Xb[b].copy()))
    # also consider champion as fallback candidate (will be compared, not necessarily polished again heavily)
    # polish each candidate under cap <=15s each and POOL_END
    npol = min(len(cands), 3)
    for k in range(npol):
        if time.time() + 2.0 > POOL_END:
            break
        v0, X0 = cands[k]
        budget = min(15.0, max(2.0, (POOL_END - time.time()) / max(1, (npol - k + 1))))
        try:
            Xp, vp = _lp_polish_deadline(X0, min(time.time() + budget, POOL_END))
        except Exception:
            continue
        if vp < champV:
            champV = float(vp)
            champion = Xp.copy()
    # final safety: if champion not improved, ensure we return best known (champion holds bestX)
    # small final snap if time remains
    if time.time() + 3.0 < POOL_END:
        try:
            Xp, vp = _lp_polish_deadline(champion, min(time.time() + 4.0, POOL_END), r_init=0.002)
            if vp < champV:
                champV = float(vp)
                champion = Xp.copy()
        except Exception:
            pass
    champion = champion / np.linalg.norm(champion, axis=1, keepdims=True)
    champion = np.where(np.isnan(champion), 0.0, champion)
    nr = np.linalg.norm(champion, axis=1, keepdims=True)
    nr = np.maximum(nr, 1e-12)
    champion = champion / nr
    return champion
