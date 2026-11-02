import time
import numpy as np
from scipy.optimize import minimize, linear_sum_assignment

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _rowmax(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return G.max(axis=1), G

def solve():
    t0 = time.time()
    LIM = 175.0
    rng = np.random.default_rng(12345)
    n0, d = 74, 13

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (74, 13):
                a = _normalize(a)
                cands.append(a)
        except Exception:
            pass
    if not cands:
        cands.append(_normalize(rng.standard_normal((n0, d))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])
    # also keep record for kicks
    base = best.copy()

    def lse_polish_generic(X0, tau, maxiter):
        nn, dd = X0.shape
        IU = np.triu_indices(nn, 1)
        I0 = IU[0].astype(np.int64); J0 = IU[1].astype(np.int64)
        y0 = X0.ravel().copy()
        def fun(yf):
            Y = yf.reshape(nn, dd)
            r2 = (Y*Y).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            X = Y / r
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau*np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            np.add.at(Wx, I0, w[:, None]*X[J0])
            np.add.at(Wx, J0, w[:, None]*X[I0])
            dot = (Wx*X).sum(axis=1, keepdims=True)
            gradY = (Wx - dot*X) / r
            return float(F), gradY.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20, "gtol": 1e-6, "maxfun": int(maxiter*6)})
            Y = res.x.reshape(nn, dd)
        except Exception:
            return X0.copy()
        rn = np.linalg.norm(Y, axis=1, keepdims=True); rn[rn==0]=1.0
        X = Y / rn
        if not np.all(np.isfinite(X)):
            return X0.copy()
        return _normalize(X)

    def micro_split(X, cur, seconds=1.0, steps=400):
        st = time.time()
        step = 5e-6
        nn = X.shape[0]
        for _ in range(steps):
            if time.time()-st > seconds or time.time()-t0 > LIM-4:
                break
            G = X @ X.T
            np.fill_diagonal(G, -2.0)
            k = int(np.argmax(G))
            i = k // nn; j = k % nn
            c = float(G[i, j])
            if c < cur - 1e-12:
                cur = c
            xi = X[i]; xj = X[j]
            gi = xj - c*xi; gj = xi - c*xj
            ni = float(np.linalg.norm(gi)); nj = float(np.linalg.norm(gj))
            if ni < 1e-12 or nj < 1e-12:
                break
            gi/=ni; gj/=nj
            improved=False
            for mode in (0,1,2):
                Y = X.copy()
                if mode==0:
                    Y[i]=xi-step*gi; Y[j]=xj-step*gj
                elif mode==1:
                    Y[i]=xi-step*2.0*gi
                else:
                    Y[j]=xj-step*2.0*gj
                Y[i]/=np.linalg.norm(Y[i]); Y[j]/=np.linalg.norm(Y[j])
                G2 = Y @ Y.T; np.fill_diagonal(G2,-2.0)
                v=float(G2.max())
                if v < cur:
                    X[:]=Y; cur=v; improved=True; break
            if improved:
                step=min(step*1.2,5e-5)
            else:
                step*=0.5
                if step<5e-13:
                    break
        return X, cur

    def largest_hole(Y, n_cand=4000):
        C = _normalize(rng.standard_normal((n_cand, d)))
        S = C @ Y.T
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        return C[j]

    def coarse_polish(X):
        # time-boxed Adam-LSE coarse to ~0.209-0.215
        try:
            X = lse_polish_generic(X, 8e-4, 50)
            if time.time()-t0 > LIM-10:
                return X
            X = lse_polish_generic(X, 3e-4, 50)
        except Exception:
            pass
        return X

    def full_polish(X, budget_sec=8.0):
        s = time.time()
        remain = budget_sec
        try:
            X = lse_polish_generic(X, 5e-4, 70)
            if time.time()-s > budget_sec-1 or time.time()-t0 > LIM-5:
                return X
            X = lse_polish_generic(X, 2e-4, 80)
            if time.time()-s > budget_sec-0.5 or time.time()-t0 > LIM-5:
                return X
            X = lse_polish_generic(X, 1e-4, 60)
            if time.time()-t0 > LIM-5:
                return X
            Xc = _max_cos(X)
            X, _ = micro_split(X, Xc, seconds=max(0.5, budget_sec-(time.time()-s)-0.5), steps=300)
            if time.time()-t0 > LIM-5:
                return X
            # final tiny LSE
            if time.time()-s < budget_sec-1:
                X = lse_polish_generic(X, 6e-5, 40)
        except Exception:
            pass
        return X

    def fingerprint(X, K=300):
        G = X @ X.T
        iu = np.triu_indices(X.shape[0], 1)
        g = G[iu]
        gs = np.sort(g)[::-1][:K]
        rm = _rowmax(X)[0]
        rms = np.sort(rm)
        return gs, rms

    def fp_dist(f1, f2):
        return float(np.max(np.abs(f1[0]-f2[0]))) + float(np.max(np.abs(f1[1]-f2[1])))

    def align_B_to_A(A, B):
        # Hungarian on A@B.T then SVD rotation
        C = A @ B.T
        try:
            ri, ci = linear_sum_assignment(-C)
        except Exception:
            ci = np.arange(A.shape[0])
            ri = np.arange(A.shape[0])
        # ri should be 0..n-1
        order = np.argsort(ri)
        col = np.asarray(ci)[order]
        Bm = B[col]
        M = Bm.T @ A
        try:
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            R = U @ Vt
        except Exception:
            R = np.eye(d)
        Brot = B @ R
        Bm_rot = Bm @ R
        diff = A - Bm_rot
        rms = float(np.sqrt((diff*diff).mean()))
        return Brot, Bm_rot, col, R, rms

    # ---- build raw parents ----
    raws = []
    # multi-scale kicks
    for sig in (0.03, 0.08, 0.15, 0.30):
        for rep in range(1):
            raws.append(_normalize(base + sig*rng.standard_normal((n0,d))))
    # extra kick copies for diversity
    raws.append(_normalize(base + 0.08*rng.standard_normal((n0,d))))
    raws.append(_normalize(base + 0.15*rng.standard_normal((n0,d))))
    # homotopy undershoot 70->74
    try:
        rm0,_ = _rowmax(base)
        drop = np.argsort(-rm0)[:4]
        keep = [i for i in range(n0) if i not in set(drop.tolist())]
        Y70 = base[keep] + 0.05*rng.standard_normal((70,d))
        Y70 = _normalize(Y70)
        raws.append(('hypo70', Y70))
    except Exception:
        pass
    # homotopy overshoot 78->74 marker
    try:
        holes = []
        cur = base.copy()
        for _ in range(4):
            h = largest_hole(cur, n_cand=5000)
            cur = np.vstack([cur, h[None,:]])
        Y78 = _normalize(cur + 0.02*rng.standard_normal((78,d)))
        raws.append(('hypo78', Y78))
    except Exception:
        pass
    # random starts
    for _ in range(2):
        raws.append(_normalize(rng.standard_normal((n0,d))))

    parents = []
    # include incumbent as parent 0 (already polished)
    parents.append(base.copy())
    parent_fps = [fingerprint(base.copy())]

    for r in raws:
        if time.time()-t0 > LIM-90:
            break
        try:
            if isinstance(r, tuple):
                tag, Y = r
                if tag=='hypo70':
                    Yp = lse_polish_generic(Y, 8e-4, 40)
                    # grow to 74 by holes
                    cur = Yp.copy()
                    for _ in range(4):
                        h = largest_hole(cur, n_cand=4000)
                        cur = np.vstack([cur, h[None,:]])
                    cur = _normalize(cur)
                    cur = coarse_polish(cur)
                    v = _max_cos(cur)
                    # accept if reasonable
                    if v < 0.23:
                        parents.append(cur)
                        parent_fps.append(fingerprint(cur))
                elif tag=='hypo78':
                    Yp = lse_polish_generic(Y, 8e-4, 40)
                    # prune 4 worst
                    rm,_ = _rowmax(_normalize(Yp))
                    drop = np.argsort(-rm)[:4]
                    keep = [i for i in range(Yp.shape[0]) if i not in set(drop.tolist())]
                    cur = _normalize(Yp[keep])
                    cur = coarse_polish(cur)
                    v = _max_cos(cur)
                    if v < 0.23:
                        parents.append(cur)
                        parent_fps.append(fingerprint(cur))
            else:
                cur = coarse_polish(r.copy())
                v = _max_cos(cur)
                if v < 0.24:
                    parents.append(cur)
                    parent_fps.append(fingerprint(cur))
                if v < best_val:
                    best_val=float(v); best=cur.copy()
        except Exception:
            continue

    # ---- dedup parents via Procrustes ----
    kept = []
    kept_fps = []
    for idx, P in enumerate(parents):
        dup=False
        fp = fingerprint(P)
        for Q, qfp in zip(kept, kept_fps):
            if fp_dist(fp,qfp) < 5e-4:
                try:
                    _,_,_,_,rms = align_B_to_A(Q,P)
                    if rms < 2e-3:
                        dup=True; break
                except Exception:
                    continue
        if not dup:
            kept.append(P); kept_fps.append(fp)
        if len(kept)>=12:
            break
    parents = kept
    # ensure at least 4
    while len(parents)<4:
        parents.append(_normalize(base+0.1*rng.standard_normal((n0,d))))

    # ---- build unlike pairs ----
    npair = len(parents)
    pair_info=[]
    for i in range(npair):
        for j in range(i+1, npair):
            try:
                _,_,_,_,rms = align_B_to_A(parents[i], parents[j])
            except Exception:
                rms=1.0
            pair_info.append(( -rms, i, j, rms))
    pair_info.sort()  # most negative first => largest rms first? -rms small = large rms
    # take up to 10 most distant pairs
    pair_info = pair_info[:10]

    children=[]
    for _, pi, pj, rms in pair_info:
        if time.time()-t0 > LIM-60:
            break
        if rms < 5e-4:
            continue
        A = parents[pi]; B = parents[pj]
        try:
            Brot, Bm_rot, col, R, _ = align_B_to_A(A,B)
        except Exception:
            continue
        # child type 1: 37 best from A + complement from B
        try:
            rmA,_ = _rowmax(A)
            order = np.argsort(rmA)  # ascending best first
            S = set(order[:37].tolist())
            C1 = np.empty_like(A)
            for k in range(n0):
                if k in S:
                    C1[k]=A[k]
                else:
                    C1[k]=Bm_rot[k]
            # repair near-duplicates
            G = C1 @ C1.T; np.fill_diagonal(G,-2.0)
            # if any pair >0.95, resample
            for _ in range(4):
                G = C1 @ C1.T; np.fill_diagonal(G,-2.0)
                m=float(G.max())
                if m < 0.95:
                    break
                kk=int(np.argmax(G)); a1=kk//n0; b1=kk%n0
                # replace b1 with hole
                mask=np.ones(n0,bool); mask[b1]=False
                h=largest_hole(C1[mask], n_cand=3000)
                C1[b1]=h
            C1=_normalize(C1)
            children.append(C1)
        except Exception:
            pass
        # child type 2: hub replacement
        try:
            rmA,G_ = _rowmax(A)
            gmax=float(G_.max())
            deg=(G_>(gmax-2e-4)).sum(axis=1)
            hubs=np.argsort(-rmA)[:8]
            # also include high-degree
            C2=A.copy()
            for h in hubs:
                C2[int(h)]=Bm_rot[int(h)]
            # repair duplicates similarly
            for _ in range(4):
                G2=C2@C2.T; np.fill_diagonal(G2,-2.0)
                if float(G2.max())<0.95:
                    break
                kk=int(np.argmax(G2)); a1=kk//n0; b1=kk%n0
                mask=np.ones(n0,bool); mask[b1]=False
                h=largest_hole(C2[mask], n_cand=3000)
                C2[b1]=h
            C2=_normalize(C2)
            children.append(C2)
        except Exception:
            pass
        if len(children)>=12:
            break

    # fingerprint filter for children: only polish new
    archive_keys=set()
    for fp in kept_fps:
        archive_keys.add(tuple(np.round(fp[0][:50],5).tolist()))
    for C in children:
        if time.time()-t0 > LIM-6:
            break
        try:
            fp = fingerprint(C)
            key=tuple(np.round(fp[0][:50],5).tolist())
            is_new=True
            for q in kept_fps:
                if fp_dist(fp,q)<2e-4:
                    is_new=False; break
            v0=_max_cos(C)
            # polish only if new or promising (within 0.01 of best)
            if (not is_new) and v0 > best_val+0.008:
                continue
            # full polish
            budget = 7.0
            # adapt budget to remaining time / remaining children
            Cp = full_polish(C.copy(), budget_sec=budget)
            v=float(_max_cos(Cp))
            archive_keys.add(key)
            kept_fps.append(fingerprint(Cp))
            if v < best_val:
                best_val=v; best=Cp.copy()
        except Exception:
            continue
        if time.time()-t0 > LIM-6:
            break

    # final squeeze of global best with remaining time
    try:
        if time.time()-t0 < LIM-8:
            best = lse_polish_generic(best, 1.2e-4, 60)
            v=_max_cos(best)
            if v < best_val:
                best_val=v
            else:
                # keep previous best if worse (still keep polished if close?)
                if v < best_val+1e-9:
                    best_val=v
            if time.time()-t0 < LIM-5:
                best, best_val2 = micro_split(best, best_val, seconds=1.5, steps=400)
                best_val=float(min(best_val, best_val2))
    except Exception:
        pass

    return _normalize(best)
