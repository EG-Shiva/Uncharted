import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.sparse import coo_matrix

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _row_max(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return np.max(G, axis=1), G

def solve():
    t0 = time.time()
    T_END = t0 + 172.0
    rng = np.random.default_rng(7)
    np.random.seed(7)
    n, d = 249, 6
    I0, J0 = np.triu_indices(n, 1)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=float)
            if a.shape == (n, d):
                a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if not cands:
        a = rng.standard_normal(size=(n, d))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cands.append(a)
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    bestX = cands[bi].copy()
    bestVal = float(vals[bi])

    def lbfgs_soft(Xinit, tau, maxiter, creg=0.08):
        iuI, iuJ = I0, J0
        def fg(xf):
            X = xf.reshape(n, d)
            r = np.sqrt(np.sum(X * X, axis=1))
            r = np.maximum(r, 1e-12)
            Xn = X / r[:, None]
            G = Xn @ Xn.T
            g = G[iuI, iuJ]
            gm = np.max(g)
            e = np.exp((g - gm) / tau)
            S = np.sum(e)
            Lsoft = gm + tau * np.log(S)
            w = e / S
            W = np.zeros((n, n))
            W[iuI, iuJ] = w
            W = W + W.T
            A = W @ Xn
            proj = np.sum(A * Xn, axis=1)
            G0 = (A - proj[:, None] * Xn) / r[:, None]
            diff = r - 1.0
            L = Lsoft + creg * np.sum(diff * diff)
            Gr = G0 + (2.0 * creg * diff)[:, None] * Xn
            return float(L), Gr.ravel()
        Xf = Xinit.ravel().copy()
        try:
            res = minimize(fg, Xf, jac=True, method='L-BFGS-B',
                           options={'maxiter': int(maxiter), 'maxfun': int(maxiter * 4),
                                    'ftol': 1e-14, 'gtol': 1e-7, 'maxcor': 8})
            Xf = res.x
        except Exception:
            pass
        Xo = Xf.reshape(n, d)
        Xo /= np.maximum(np.linalg.norm(Xo, axis=1, keepdims=True), 1e-12)
        return Xo

    def exact_polish(Xinit, deadline, step0=1.2e-3):
        cur = Xinit.copy()
        cur /= np.linalg.norm(cur, axis=1, keepdims=True)
        curVal = _max_cos(cur)
        step = step0
        while time.time() < deadline:
            if step < 5e-14:
                break
            G = cur @ cur.T
            np.fill_diagonal(G, -2.0)
            m = float(G.max())
            window = max(10.0 * step, 2e-4)
            mask = G > (m - window)
            F = mask.astype(float) @ cur
            dot = np.sum(F * cur, axis=1)
            Ft = F - dot[:, None] * cur
            trial = cur - step * Ft
            trial /= np.maximum(np.linalg.norm(trial, axis=1, keepdims=True), 1e-12)
            v = _max_cos(trial)
            if v < curVal:
                cur = trial; curVal = v
                step = min(step * 1.25, 0.01)
            else:
                step *= 0.5
                if step < 1e-7:
                    break
        return cur, curVal

    def lp_polish_short(x, t_end):
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        best = _max_cos(x)
        radius = 8e-4
        iu = (I0, J0)
        while time.time() < t_end - 0.5:
            if radius < 1e-14:
                break
            Gfull = x @ x.T
            c = Gfull[iu]
            m = float(np.max(c))
            sel = c > m - max(20 * radius, 5e-7)
            cnt = int(np.count_nonzero(sel))
            if cnt == 0:
                radius *= 0.5; continue
            I = iu[0][sel]; J = iu[1][sel]; csv = c[sel]
            if cnt > 5000:
                idx = np.argsort(-csv)[:5000]
                I = I[idx]; J = J[idx]; csv = csv[idx]; cnt = 5000
            p = cnt; nv = n * d + 1
            rows = np.repeat(np.arange(p), 2 * d + 1)
            cols = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], 1).ravel()
            valsA = np.concatenate([x[J], x[I], -np.ones((p, 1))], 1).ravel()
            try:
                A_ub = coo_matrix((valsA, (rows, cols)), shape=(p, nv)).tocsr()
                A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
                obj = np.zeros(nv); obj[-1] = 1.0
                res = linprog(obj, A_ub=A_ub, b_ub=-csv, A_eq=A_eq, b_eq=np.zeros(n),
                              bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
            except Exception:
                radius *= 0.5; continue
            if res.status != 0:
                radius *= 0.5; continue
            y = x + res.x[:-1].reshape(n, d)
            y /= np.linalg.norm(y, axis=1, keepdims=True)
            v = _max_cos(y)
            if v < best:
                x, best = y, v
                radius = min(radius * 1.5, 0.02)
            else:
                radius *= 0.5
        return x

    def forced_step(X, cluster_nodes, delta, radius, Kall=3500):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        gtri = G[I0, J0]
        # top Kall constraints
        if Kall < len(gtri):
            thr = np.sort(gtri)[-Kall]
        else:
            thr = float(np.min(gtri))
        # ensure we also keep threshold not too low: at least m-0.02
        thr = max(thr, m - 0.02)
        selm = gtri >= thr
        I = I0[selm]; J = J0[selm]; csv = gtri[selm]
        p = len(I)
        if p == 0:
            return X, m
        cset = set(int(v) for v in cluster_nodes)
        incid = np.array([(int(a) in cset) or (int(b) in cset) for a, b in zip(I, J)])
        idx_inc = np.where(incid)[0]
        # choose 100-300 forced among incident, prefer tightest
        if len(idx_inc) == 0:
            # fallback: closest 150 pairs as forced
            order = np.argsort(-csv)[:150]
            forced_idx = set(order.tolist())
        else:
            # sort incident by tightness
            o = idx_inc[np.argsort(-csv[idx_inc])]
            if len(o) > 300:
                o = o[:300]
            elif len(o) < 100:
                # pad with tightest non-incident to reach 100
                need = 100 - len(o)
                oall = np.argsort(-csv)
                extra = []
                s = set(o.tolist())
                for k in oall:
                    if k not in s:
                        extra.append(k)
                        if len(extra) >= need:
                            break
                o = np.concatenate([o, np.array(extra, dtype=int)])
            forced_idx = set(o.tolist())
        is_f = np.zeros(p, dtype=float)
        if len(forced_idx) > 0:
            is_f[list(forced_idx)] = 1.0
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        colI = (I[:, None] * d + np.arange(d)).ravel()
        colJ = (J[:, None] * d + np.arange(d)).ravel()
        # interleave correctly: for each k, first d cols are I-block? need x[J] at I block and x[I] at J block
        # Build cols in order [I-block(d), J-block(d), t]
        cols = np.empty(p * (2 * d + 1), dtype=int)
        valsA = np.empty(p * (2 * d + 1), dtype=float)
        # vectorized fill
        for k in range(0):  # placeholder
            pass
        # build row by row via concatenation (p<=~3500, fine)
        cols_list = np.concatenate([I[:, None] * d + np.arange(d), J[:, None] * d + np.arange(d), np.full((p, 1), n * d)], axis=1).ravel()
        vals_list = np.concatenate([X[J], X[I], -np.ones((p, 1))], axis=1).ravel()
        b_ub = -csv - is_f * delta
        try:
            A_ub = coo_matrix((vals_list, (rows, cols_list)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((X.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            return None, None
        if res.status != 0:
            return None, None
        dx = res.x[:-1].reshape(n, d)
        return dx, float(res.x[-1])

    def bfs_cluster(adj, start, size):
        seen = {int(start)}
        q = [int(start)]
        order = [int(start)]
        while q and len(order) < size:
            u = q.pop(0)
            nbrs = np.where(adj[u])[0]
            # shuffle-ish deterministic: sort by degree desc then take
            for v in nbrs:
                v = int(v)
                if v not in seen:
                    seen.add(v)
                    q.append(v)
                    order.append(v)
                    if len(order) >= size:
                        break
        return order

    archive = [(bestVal, bestX.copy())]

    deltas = [1e-4, 5e-4, 1e-3, 5e-4, 1e-4, 1e-3]
    radii = [0.02, 0.05, 0.01, 0.03, 0.05, 0.02]
    it = 0
    # quick initial soft polish of incumbent to have tight baseline
    if time.time() < T_END - 60:
        tmp = lbfgs_soft(bestX, 0.002, 60)
        v = _max_cos(tmp)
        if v < bestVal:
            bestVal = v; bestX = tmp.copy(); archive.append((v, tmp.copy()))

    while time.time() < T_END - 30:
        it += 1
        # choose base: mostly best, sometimes archive diverse
        if len(archive) > 1 and (it % 5 == 0):
            # pick random archive member that is not best (diversify)
            k = int(rng.integers(0, len(archive)))
            base = archive[k][1]
        else:
            base = bestX
        # compute contact graph with slack 2e-4
        Gc = base @ base.T
        np.fill_diagonal(Gc, -2.0)
        mcur = float(np.max(Gc))
        adj = Gc > (mcur - 2e-4)
        deg = np.sum(adj, axis=1)
        # hub choice: cycle among top 10 + random top 30
        if rng.random() < 0.7:
            top = np.argsort(-deg)[:10]
            hub = int(top[it % len(top)])
        else:
            top30 = np.argsort(-deg)[:30]
            hub = int(top30[int(rng.integers(0, len(top30)))])
        csize = int(rng.integers(18, 33))  # 18..32 nodes
        cluster = bfs_cluster(adj, hub, csize)
        delta = float(deltas[it % len(deltas)])
        # randomize delta a bit
        if rng.random() < 0.3:
            delta = float(rng.choice([1e-4, 5e-4, 1e-3]))
        radius = float(radii[it % len(radii)])
        if rng.random() < 0.3:
            radius = float(rng.choice([0.01, 0.02, 0.03, 0.05]))
        dx, _ = forced_step(base, cluster, delta, radius, Kall=3500)
        if dx is None:
            # fallback: small tangent shake of cluster
            Xtry = base.copy()
            sigma = 0.02
            pert = rng.standard_normal(size=(len(cluster), d)) * sigma
            for ii, idx in enumerate(cluster):
                v = pert[ii] - np.dot(pert[ii], Xtry[idx]) * Xtry[idx]
                Xtry[idx] = Xtry[idx] + v
            Xtry /= np.linalg.norm(Xtry, axis=1, keepdims=True)
            Yf = Xtry
        else:
            Yf = base + dx
            Yf /= np.maximum(np.linalg.norm(Yf, axis=1, keepdims=True), 1e-12)
        # step even if worsened: re-polish
        # short soft-max re-jam
        Yp = lbfgs_soft(Yf, 0.0012, 50)
        Yp, vp1 = exact_polish(Yp, time.time() + 0.6, step0=1.0e-3)
        vp = _max_cos(Yp)
        # a second soft at lower tau if promising
        if vp < bestVal + 0.002 and time.time() < T_END - 32:
            Yp2 = lbfgs_soft(Yp, 0.0006, 40)
            v2 = _max_cos(Yp2)
            if v2 < vp:
                Yp = Yp2; vp = v2
        if vp < bestVal - 1e-12:
            bestVal = vp; bestX = Yp.copy()
            archive.append((vp, Yp.copy()))
            if len(archive) > 12:
                archive = sorted(archive, key=lambda z: z[0])[:12]
        elif vp < bestVal + 0.003:
            # keep diverse basin
            archive.append((vp, Yp.copy()))
            if len(archive) > 12:
                # keep best + diverse: sort keep best 6 + random 6
                archive = sorted(archive, key=lambda z: z[0])
                keep = archive[:6] + [archive[int(rng.integers(6, len(archive)))] for _ in range(min(6, len(archive) - 6))]
                # dedup by value
                archive = keep
        # occasional single-hole greedy opportunism
        if it % 7 == 0 and time.time() < T_END - 32:
            rm, Gm = _row_max(bestX)
            o2 = np.argsort(-rm)
            idx = int(o2[0])
            mask = np.ones(n, bool); mask[idx] = False
            Xo = bestX[mask]
            Cc = rng.standard_normal(size=(6000, d))
            Cc /= np.linalg.norm(Cc, axis=1, keepdims=True)
            D = Cc @ Xo.T
            cm = np.max(D, axis=1)
            b2 = int(np.argmin(cm))
            glob = float(np.max(Gm))
            if float(cm[b2]) < glob - 1e-6:
                Xt = bestX.copy(); Xt[idx] = Cc[b2]
                vg = _max_cos(Xt)
                if vg < bestVal:
                    Xt2 = lbfgs_soft(Xt, 0.001, 40)
                    vg2 = _max_cos(Xt2)
                    if vg2 < bestVal:
                        bestVal = vg2; bestX = Xt2.copy()
                        archive.append((vg2, Xt2.copy()))

    # final polish best
    Xfin = bestX.copy()
    if time.time() < T_END - 28:
        Xfin = lbfgs_soft(Xfin, 0.0008, 120)
        v = _max_cos(Xfin)
        if v < bestVal:
            bestVal = v; bestX = Xfin.copy()
        else:
            Xfin = bestX.copy()
    rem = T_END - time.time() - 1.0
    if rem > 6:
        try:
            Xlp = lp_polish_short(Xfin, T_END - 1.0)
            vlp = _max_cos(Xlp)
            if vlp < bestVal:
                bestX = Xlp; bestVal = vlp
        except Exception:
            pass
    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
