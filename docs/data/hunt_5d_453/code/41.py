import time
import numpy as np

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _tangent_bases(X):
    n, d = X.shape
    B = np.empty((n, d, 4), dtype=np.float64)
    for i in range(n):
        x = X[i]
        k = int(np.argmin(np.abs(x)))
        T = np.empty((d, 4), dtype=np.float64)
        c = 0
        for j in range(d):
            if j == k:
                continue
            v = -x[j] * x
            v[j] += 1.0
            T[:, c] = v
            c += 1
        Q, _ = np.linalg.qr(T)
        B[i] = Q[:, :4]
    return B

def _softmax_polish(X, steps, beta, lr, mu, t_end):
    n = X.shape[0]
    v = np.zeros_like(X)
    for it in range(steps):
        if time.time() > t_end:
            break
        G = X @ X.T
        np.fill_diagonal(G, -np.inf)
        m = float(np.max(G))
        if not np.isfinite(m):
            break
        W = np.exp(beta * (G - m))
        np.fill_diagonal(W, 0.0)
        s = float(np.sum(W))
        if not np.isfinite(s) or s <= 0:
            break
        W /= s
        grad = W @ X
        dot = np.sum(grad * X, axis=1, keepdims=True)
        Gt = grad - dot * X
        mn = float(np.max(np.sqrt(np.sum(Gt * Gt, axis=1))))
        if not np.isfinite(mn) or mn < 1e-16:
            break
        Gt /= (mn + 1e-30)
        v = mu * v + (1.0 - mu) * Gt
        Y = X - lr * v
        nn = np.sqrt(np.sum(Y * Y, axis=1, keepdims=True))
        nn[nn == 0] = 1.0
        X = Y / nn
    return X

def _micro_shave_cap(X, t_end, bestV):
    n, d = X.shape
    eps = 5e-7
    it = 0
    while time.time() < t_end:
        it += 1
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        if m < bestV:
            bestV = m
        flat = int(np.argmax(G))
        i = flat // n; j = flat % n
        xi = X[i].copy(); xj = X[j].copy()
        cij = float(np.clip(xi @ xj, -1.0, 1.0))
        ti = xj - cij * xi
        ni = float(np.linalg.norm(ti))
        if ni < 1e-12:
            ti = np.random.randn(d); ti -= float(ti @ xi) * xi
            ni = float(np.linalg.norm(ti)) + 1e-30
        ti /= ni
        tj = xi - cij * xj
        nj = float(np.linalg.norm(tj))
        if nj < 1e-12:
            tj = np.random.randn(d); tj -= float(tj @ xj) * xj
            nj = float(np.linalg.norm(tj)) + 1e-30
        tj /= nj
        # rest max
        G[i, :] = -2.0; G[:, i] = -2.0
        G[j, :] = -2.0; G[:, j] = -2.0
        m_rest = float(np.max(G))
        improved = False
        for sc in (eps, eps * 0.5):
            vi = xi - sc * ti; vi /= float(np.linalg.norm(vi))
            vj = xj - sc * tj; vj /= float(np.linalg.norm(vj))
            di = float(np.max(X @ vi))
            # X contains old xi,xj so di may hit i/j; recompute roughly
            # use masked evaluation
            dri = X @ vi; dri[i] = -2.0; dri[j] = -2.0
            drj = X @ vj; drj[i] = -2.0; drj[j] = -2.0
            mc = m_rest
            a = float(np.max(dri))
            if a > mc: mc = a
            b = float(np.max(drj))
            if b > mc: mc = b
            c2 = float(vi @ vj)
            if c2 > mc: mc = c2
            if mc < m - 1e-15:
                X[i] = vi; X[j] = vj
                if mc < bestV: bestV = mc
                eps = min(eps * 1.02, 2e-6)
                improved = True
                break
        if not improved:
            eps *= 0.85
            if eps < 2e-10:
                k = i if (it % 2 == 0) else j
                rnd = np.random.randn(d)
                rnd -= float(rnd @ X[k]) * X[k]
                rn = float(np.linalg.norm(rnd))
                if rn > 1e-12:
                    X[k] = X[k] + 5e-10 * (rnd / rn)
                    X[k] /= float(np.linalg.norm(X[k]))
                eps = 2e-10
        if it % 100 == 0 and time.time() > t_end:
            break
    return X, bestV

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    EXPLORE_END = T_END - 30.0
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n, d = 453, 5
    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            cands.append(np.asarray(a, dtype=np.float64))
        except Exception:
            pass
    bestX = None
    bestV = float("inf")
    for cc in cands:
        try:
            if cc.shape != (453, 5):
                continue
            cc = cc / np.linalg.norm(cc, axis=1, keepdims=True)
            if not np.all(np.isfinite(cc)):
                continue
            vv = _max_cos(cc)
            if vv < bestV:
                bestV = vv; bestX = cc.copy()
        except Exception:
            continue
    if bestX is None:
        bestX = rng.standard_normal((n, d))
        bestX /= np.linalg.norm(bestX, axis=1, keepdims=True)
        bestV = _max_cos(bestX)
    # working point (may be worse than best to explore)
    Xcur = bestX.copy()
    mcur = bestV
    iu_r, iu_c = np.triu_indices(n, 1)
    K_NULL = 50
    n_explore = 0
    while time.time() < EXPLORE_END - 4.0:
        n_explore += 1
        if time.time() > EXPLORE_END - 2.0:
            break
        try:
            G = Xcur @ Xcur.T
            c_all = G[iu_r, iu_c]
            m = float(np.max(c_all))
            slack = m - c_all
            # contacts slack < 1e-9
            cidx = np.where(slack < 1e-9)[0]
            if cidx.size > 2200:
                part = np.argpartition(slack[cidx], 2200)[:2200]
                cidx = cidx[part]
            if cidx.size < 400:
                # under-contacted: loosen to get rigidity info
                cidx = np.argsort(slack)[:800]
            p = int(cidx.size)
            I = iu_r[cidx]; J = iu_c[cidx]
            # shell 5e-7..1e-4
            sidx = np.where((slack >= 5e-7) & (slack < 1e-4))[0]
            if sidx.size > 1500:
                part = np.argpartition(slack[sidx], 1500)[:1500]
                sidx = sidx[part]
            if sidx.size < 60:
                # take next tightest above contacts
                order = np.argsort(slack)
                # skip contacts
                take = []
                for o in order:
                    if slack[o] >= 5e-7:
                        take.append(o)
                        if len(take) >= 600:
                            break
                sidx = np.array(take, dtype=np.int64)
            q = int(sidx.size)
            if q == 0 or p == 0:
                # random kick
                Xcur = Xcur + 1e-4 * rng.standard_normal((n, d))
                Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
                mcur = _max_cos(Xcur)
                continue
            SI = iu_r[sidx]; SJ = iu_c[sidx]
            B = _tangent_bases(Xcur)
            Nvar = n * 4
            # build R p x Nvar
            aI = np.einsum('pdk,pd->pk', B[I], Xcur[J])
            aJ = np.einsum('pdk,pd->pk', B[J], Xcur[I])
            R = np.zeros((p, Nvar), dtype=np.float64)
            # fill via advanced indexing per contact (loop in python but p~1700 ok)
            # vectorized using flat offsets
            rows = np.repeat(np.arange(p), 4)
            colsI = (I[:, None] * 4 + np.arange(4)[None, :]).ravel()
            colsJ = (J[:, None] * 4 + np.arange(4)[None, :]).ravel()
            R[rows, colsI] = aI.ravel()
            R[rows, colsJ] = aJ.ravel()
            # smallest subspace via RtR eigh
            RtR = R.T @ R
            # regularize tiny
            w, V = np.linalg.eigh(RtR)
            # V columns ascending
            V0 = V[:, :K_NULL]  # Nvar x K
            # shell matrix S q x Nvar
            bI = np.einsum('pdk,pd->pk', B[SI], Xcur[SJ])
            bJ = np.einsum('pdk,pd->pk', B[SJ], Xcur[SI])
            # M = S V0  (q x K) without forming S
            # S V0 = bI * V0[I-part] + bJ * V0[J-part]
            V0r = V0.reshape(n, 4, K_NULL)  # n x 4 x K
            M = np.einsum('qk,qkq->qq', bI[:, :, None], V0r[SI]) if False else None
            # compute efficiently:
            # for each shell s: M[s,k] = sum_a bI[s,a]*V0r[SI[s],a,k] + sum_a bJ[s,a]*V0r[SJ[s],a,k]
            M = np.einsum('qa,qak->qk', bI, V0r[SI]) + np.einsum('qa,qak->qk', bJ, V0r[SJ])
            rnorm = np.sqrt(np.sum(M * M, axis=1))
            rnorm[~np.isfinite(rnorm)] = 0.0
            topL = min(8, q)
            top_idx = np.argpartition(-rnorm, topL - 1)[:topL] if q >= topL else np.arange(q)
            # candidate coeffs
            cand_coeffs = []
            for ti in top_idx:
                nr = float(rnorm[ti])
                if nr < 1e-14:
                    continue
                cand_coeffs.append(M[ti] / nr)
            # add random null dirs
            for _ in range(4):
                v = rng.standard_normal(K_NULL); v /= (np.linalg.norm(v) + 1e-30)
                cand_coeffs.append(v)
            # also add negative of best? no, we want increase; keep as is
            best_try_X = None
            best_try_m = m
            best_try_gain = 0
            # count current contacts
            ncon0 = int(np.sum(slack < 1e-9))
            step_sizes = (0.003, 0.008, 0.020)
            for coeff in cand_coeffs:
                if time.time() > EXPLORE_END - 2.0:
                    break
                u = V0 @ coeff  # Nvar
                U = u.reshape(n, 4)
                D = np.einsum('idk,ik->id', B, U)
                # scale so max row norm =1
                dn = np.sqrt(np.sum(D * D, axis=1))
                mx = float(np.max(dn))
                if not np.isfinite(mx) or mx < 1e-14:
                    continue
                D /= mx
                for s in step_sizes:
                    Y = Xcur + s * D
                    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                    Gy = Y @ Y.T
                    np.fill_diagonal(Gy, -2.0)
                    my = float(np.max(Gy))
                    # quick contact count via upper triangle? approx via full
                    # accept if lowers max
                    if my < m - 1e-12:
                        # prefer largest drop
                        if my < best_try_m:
                            best_try_m = my
                            best_try_X = Y.copy()
                    elif my <= m + 2e-12:
                        # check new contacts created (pairs within 1e-9 of m)
                        # count using threshold relative to my
                        ncon = int(np.sum(Gy[np.triu_indices(n, 1)] > my - 1e-9))
                        if ncon > ncon0:
                            if best_try_X is None:
                                best_try_m = my
                                best_try_X = Y.copy()
                    # curvature estimate: if my explodes, smaller s may still work; continue
                # end steps
            if best_try_X is not None:
                # accept step
                Xcur = best_try_X
                mcur = float(best_try_m)
                if mcur < bestV - 1e-14 and np.all(np.isfinite(Xcur)):
                    bestV = mcur
                    bestX = Xcur.copy()
                # short time-capped LP-free polish (softmax)
                rem = min(3.0, EXPLORE_END - time.time() - 0.5)
                if rem > 0.6:
                    Xcur = _softmax_polish(Xcur, steps=60, beta=2500.0, lr=0.0012, mu=0.85, t_end=time.time() + rem)
                    mcur = _max_cos(Xcur)
                    if mcur < bestV:
                        bestV = mcur; bestX = Xcur.copy()
                    # if polish worsened a lot, revert to best region partially
                    if mcur > bestV + 0.002:
                        Xcur = bestX.copy(); mcur = bestV
            else:
                # no null move accepted: small random hop in null space to change basin
                coeff = rng.standard_normal(K_NULL); coeff /= (np.linalg.norm(coeff) + 1e-30)
                u = V0 @ coeff
                U = u.reshape(n, 4)
                D = np.einsum('idk,ik->id', B, U)
                dn = np.sqrt(np.sum(D * D, axis=1))
                mx = float(np.max(dn))
                if np.isfinite(mx) and mx > 1e-14:
                    D /= mx
                    s = 0.004 if (n_explore % 3 == 0) else 0.001
                    # bias rattlers: boost rows with 0 contacts
                    # compute per-point contact degree
                    deg = np.zeros(n, dtype=np.int64)
                    np.add.at(deg, I, 1); np.add.at(deg, J, 1)
                    ratt = np.where(deg == 0)[0]
                    if ratt.size > 0:
                        D[ratt] *= 2.0
                    Y = Xcur + s * D
                    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
                    my = _max_cos(Y)
                    # short relax
                    Y = _softmax_polish(Y, steps=30, beta=800.0, lr=0.001, mu=0.85, t_end=time.time() + 1.5)
                    my2 = _max_cos(Y)
                    if my2 < mcur:
                        Xcur = Y; mcur = my2
                        if my2 < bestV:
                            bestV = my2; bestX = Y.copy()
                    else:
                        # occasional restart from best with tiny jitter to avoid drift
                        if n_explore % 5 == 0:
                            Xcur = bestX.copy()
                            Xcur += 3e-7 * rng.standard_normal((n, d))
                            Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
                            mcur = _max_cos(Xcur)
            # end iteration: keep Xcur near best if drifted too far
            if mcur > bestV + 0.005:
                Xcur = bestX.copy(); mcur = bestV
        except Exception:
            # on failure, jitter best
            try:
                Xcur = bestX.copy()
                Xcur += 1e-6 * rng.standard_normal((n, d))
                Xcur /= np.linalg.norm(Xcur, axis=1, keepdims=True)
                mcur = _max_cos(Xcur)
            except Exception:
                pass
            continue
    # final 30s: polish incumbent
    try:
        X = bestX.copy()
        # brief softmax anneal
        t1 = T_END - 24.0
        if time.time() < t1:
            X = _softmax_polish(X, steps=200, beta=600.0, lr=0.0015, mu=0.85, t_end=t1)
            v = _max_cos(X)
            if v < bestV:
                bestV = v; bestX = X.copy()
            else:
                X = bestX.copy()
        t2 = T_END - 1.5
        if time.time() < t2:
            X, bestV = _micro_shave_cap(X, t2, bestV) if False else (X, bestV)
            # inline shave to avoid extra call overhead: reuse function
            try:
                X, vv = _micro_shave_cap(X, t2, bestV)
                # _micro_shave_cap returns (X,bestV)
                bestX = X.copy(); bestV = float(vv) if isinstance(vv, float) else bestV
            except Exception:
                pass
    except Exception:
        pass
    # safety normalize, shrink margin tiny
    try:
        bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
        bestX = np.asarray(bestX, dtype=np.float64)
        # ensure finite
        if not np.all(np.isfinite(bestX)):
            bestX = cands[0] / np.linalg.norm(cands[0], axis=1, keepdims=True)
    except Exception:
        pass
    return bestX
