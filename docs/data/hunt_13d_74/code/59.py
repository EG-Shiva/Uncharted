import time
import numpy as np
from scipy.optimize import minimize

def _max_cos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(G.max())

def _normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def solve():
    t0 = time.time()
    LIM = 172.0
    rng = np.random.default_rng(0)
    n, d = 74, 13
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
        cands.append(_normalize(rng.standard_normal((n, d))))
    vals = [_max_cos(a) for a in cands]
    bi = int(np.argmin(vals))
    best = cands[bi].copy()
    best_val = float(vals[bi])

    IU = np.triu_indices(n, 1)
    I0 = IU[0].astype(np.int64)
    J0 = IU[1].astype(np.int64)

    def lse_subset(X0, move_idx, tau, maxiter):
        move_idx = np.asarray(move_idx, dtype=np.int64)
        m = len(move_idx)
        if m == 0:
            return X0.copy()
        # map global -> local
        g2l = np.full(n, -1, dtype=np.int64)
        g2l[move_idx] = np.arange(m)
        Xfix = X0.copy()
        y0 = X0[move_idx].ravel().copy()
        def fun(yf):
            Ym = yf.reshape(m, d)
            r2 = (Ym * Ym).sum(axis=1, keepdims=True)
            r = np.sqrt(np.maximum(r2, 1e-24))
            r = np.maximum(r, 1e-12)
            Xm = Ym / r
            X = Xfix.copy()
            X[move_idx] = Xm
            G = X @ X.T
            g = G[I0, J0]
            mmax = float(g.max())
            e = np.exp((g - mmax) / tau)
            S = float(e.sum())
            if not np.isfinite(S) or S <= 0:
                return float(mmax), np.zeros_like(yf)
            F = mmax + tau * np.log(S + 1e-300)
            w = e / (S + 1e-300)
            Wx = np.zeros_like(X)
            np.add.at(Wx, I0, w[:, None] * X[J0])
            np.add.at(Wx, J0, w[:, None] * X[I0])
            # grad only for moving rows
            Wm = Wx[move_idx]
            dot = (Wm * Xm).sum(axis=1, keepdims=True)
            gradYm = (Wm - dot * Xm) / r
            return float(F), gradYm.ravel()
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": int(maxiter), "ftol": 1e-20,
                                    "gtol": 1e-7, "maxfun": int(maxiter * 8)})
            Ym = res.x.reshape(m, d)
        except Exception:
            return X0.copy()
        rn = np.linalg.norm(Ym, axis=1, keepdims=True)
        rn[rn == 0] = 1.0
        Xm = Ym / rn
        if not np.all(np.isfinite(Xm)):
            return X0.copy()
        X = X0.copy()
        X[move_idx] = Xm
        return _normalize(X)

    def fingerprint(X):
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        gmax = float(G.max())
        tri = G[I0, J0]
        # top 200 descending
        if len(tri) > 200:
            idx = np.argpartition(-tri, 200)[:200]
            top = np.sort(tri[idx])[::-1]
        else:
            top = np.sort(tri)[::-1]
        deg = (G > (gmax - 1e-4)).sum(axis=1)
        deg_sorted = np.sort(deg)
        return top.copy(), deg_sorted.copy(), float(gmax)

    def is_diverse(fp, archive):
        top, deg, _ = fp
        for (atop, adeg, _) in archive:
            # both must be close to be non-diverse
            L = float(np.max(np.abs(top - atop)))
            ddiff = int(np.max(np.abs(deg - adeg)))
            if L < 2e-5 and ddiff == 0:
                return False
        return True

    def insert_one(Y_exist):
        # 20k candidates
        C = _normalize(rng.standard_normal((20000, d)))
        S = C @ Y_exist.T
        sc = S.max(axis=1)
        j = int(np.argmin(sc))
        x = C[j].copy()
        step = 0.05
        for _ in range(40):
            dots = Y_exist @ x
            m = float(dots.max())
            w = np.exp((dots - m) / 0.02)
            sw = float(w.sum())
            if not np.finite(sw) or sw <= 0:
                break
            # weighted sum direction
            g = (w[:, None] * Y_exist).sum(axis=0) - (float((w * dots).sum())) * x
            g = g - float(g @ x) * x
            ng = float(np.linalg.norm(g))
            if not np.finite(ng) or ng < 1e-12:
                break
            g /= ng
            ok = False
            for s in (step, step * 0.5, step * 0.25):
                xt = x - s * g
                nn = float(np.linalg.norm(xt))
                if nn < 1e-12:
                    continue
                xt /= nn
                m2 = float((Y_exist @ xt).max())
                if m2 < m:
                    x = xt
                    ok = True
                    break
            if ok:
                step = min(step * 1.05, 0.05)
            else:
                step *= 0.5
                if step < 1e-4:
                    break
        return x

    archive = []
    archive.append(fingerprint(best))

    # keep list of distinct basins (X)
    basins = [(float(best_val), best.copy())]

    trial = 0
    # quick initial fine polish of best to set baseline (cheap)
    # do not spend much
    while time.time() - t0 < LIM - 12:
        trial += 1
        remain = (t0 + LIM - 12) - time.time()
        if remain < 9:
            break
        # pick base: mostly best, sometimes second basin
        if len(basins) > 1 and (trial % 5 == 0):
            # random archived basin
            bi2 = int(rng.integers(0, len(basins)))
            base = basins[bi2][1].copy()
        else:
            base = best.copy()
        Gb = base @ base.T
        np.fill_diagonal(Gb, -2.0)
        gmaxb = float(Gb.max())
        rowmax = Gb.max(axis=1)
        deg1e4 = (Gb > (gmaxb - 1e-4)).sum(axis=1)
        # hub candidates: top degrees
        order = np.argsort(-deg1e4, kind="stable")
        # randomize among top 6
        topk = order[:6]
        hub = int(topk[int(rng.integers(0, len(topk)))])
        ksize = 12 + int(rng.integers(0, 9))  # 12-20
        # BFS over contact graph at 2e-4
        thr = gmaxb - 2e-4
        Adj = Gb > thr
        D = set()
        D.add(hub)
        queue = []
        # init neighbours sorted by dot desc
        nbrs = np.where(Adj[hub])[0].tolist()
        nbrs.sort(key=lambda j: float(Gb[hub, j]), reverse=True)
        # random shuffle tail to diversify
        queue.extend(nbrs)
        # also consider rattlers nearby: rattlers
        ratt = np.where(rowmax < (gmaxb - 1e-4))[0].tolist()
        # BFS
        qpos = 0
        while len(D) < ksize and qpos < len(queue):
            u = int(queue[qpos]); qpos += 1
            if u in D:
                continue
            D.add(u)
            if len(D) >= ksize:
                break
            nbrs2 = np.where(Adj[u])[0].tolist()
            # add unseen
            for v in nbrs2:
                if v not in D and v not in queue:
                    queue.append(int(v))
        # if still short, fill by proximity to D (largest max-dot)
        if len(D) < ksize:
            Dlist = np.array(list(D), dtype=np.int64)
            # score each outside point by max dot to D
            outside = [i for i in range(n) if i not in D]
            scores = []
            for i in outside:
                s = float(Gb[i, Dlist].max()) if len(Dlist) > 0 else 0.0
                scores.append((s, i))
            scores.sort(reverse=True)
            for _, i in scores:
                D.add(int(i))
                if len(D) >= ksize:
                    break
        # force include a nearby rattler (diversity): with 60% chance
        if len(ratt) > 0 and rng.random() < 0.6:
            Dlist = list(D)
            # find rattler closest to D (max dot) not in D
            best_r = -1; best_s = -1e9
            for r_ in ratt:
                if r_ in D:
                    continue
                # max dot to D members
                s = float(Gb[r_, np.array(Dlist, dtype=np.int64)].max()) if Dlist else -1
                if s > best_s:
                    best_s = s; best_r = int(r_)
            if best_r >= 0:
                # replace farthest member (smallest max-dot to rest of D) if D full
                # simply add and drop one non-hub far member to keep size
                if len(D) >= ksize:
                    # find member with smallest rowmax or farthest from hub
                    cand_out = [v for v in D if v != hub]
                    # farthest from hub = smallest Gb[v,hub]
                    cand_out.sort(key=lambda v: float(Gb[int(v), hub]))
                    drop = int(cand_out[0])
                    D.remove(drop)
                D.add(int(best_r))
        Darr = np.array(sorted(D), dtype=np.int64)
        # time check
        if time.time() - t0 > t0 + LIM - 11:
            break
        Fmask = np.ones(n, dtype=bool); Fmask[Darr] = False
        # rebuild one at a time
        Y_exist = base[Fmask].copy()  # (n-k, d)
        order_D = Darr.copy()
        rng.shuffle(order_D)
        rebuilt = []
        abort = False
        for _ in order_D:
            if time.time() - t0 > t0 + LIM - 10:
                abort = True
                break
            xnew = insert_one(Y_exist)
            rebuilt.append(xnew)
            Y_exist = np.vstack([Y_exist, xnew.reshape(1, -1)])
        if abort:
            break
        Xt = base.copy()
        for idx_g, xv in zip(order_D, rebuilt):
            Xt[int(idx_g)] = xv
        Xt = _normalize(Xt)
        if time.time() - t0 > t0 + LIM - 9:
            # keep if better even without polish
            v0 = _max_cos(Xt)
            if v0 < best_val:
                best_val = float(v0); best = Xt.copy()
            break
        # joint LSE polish in which only rebuilt move
        Xt2 = lse_subset(Xt, order_D, 0.003, 60)
        if time.time() - t0 > t0 + LIM - 8:
            v2 = _max_cos(Xt2)
            if v2 < best_val:
                best_val = float(v2); best = Xt2.copy()
            break
        Xt2 = lse_subset(Xt2, order_D, 7e-4, 60)
        v2 = _max_cos(Xt2)
        fp2 = fingerprint(Xt2)
        # diversity gate: if not diverse, discard cheaply (but keep if outright best)
        div = is_diverse(fp2, archive)
        if v2 < best_val - 1e-12:
            # always keep outright improvement
            best_val = float(v2); best = Xt2.copy()
            if div:
                archive.append(fp2)
                basins.append((float(v2), Xt2.copy()))
                if len(basins) > 6:
                    basins.sort(key=lambda t: t[0])
                    basins = basins[:6]
            else:
                # still archive to avoid re-trying? don't add
                pass
            # also fully polish it since it's best
            div = True  # force full polish
        if not div:
            continue
        # unfreeze and fully polish
        if time.time() - t0 > t0 + LIM - 7:
            # no time for full polish, archive and continue
            archive.append(fp2)
            continue
        all_idx = np.arange(n, dtype=np.int64)
        Xt3 = lse_subset(Xt2, all_idx, 5e-4, 70)
        v3 = _max_cos(Xt3)
        # tiny extra fine stage
        if time.time() - t0 < t0 + LIM - 6:
            Xt3b = lse_subset(Xt3, all_idx, 1.5e-4, 50)
            v3b = _max_cos(Xt3b)
            if v3b < v3:
                Xt3 = Xt3b; v3 = v3b
        fp3 = fingerprint(Xt3)
        # archive distinct basin
        if is_diverse(fp3, archive):
            archive.append(fp3)
            basins.append((float(v3), Xt3.copy()))
            if len(basins) > 6:
                basins.sort(key=lambda t: t[0])
                basins = basins[:6]
            if len(archive) > 40:
                archive = archive[-40:]
        if v3 < best_val - 1e-13:
            best_val = float(v3); best = Xt3.copy()
        # cap trials to avoid endless loop; many cycles allowed
        if trial > 60:
            break

    # final intensive polish of best
    if time.time() - t0 < LIM - 0.8:
        try:
            Xf = lse_subset(best, np.arange(n, dtype=np.int64), 2e-4, 90)
            vf = _max_cos(Xf)
            if vf < best_val:
                best_val, best = float(vf), Xf.copy()
            if time.time() - t0 < LIM - 0.5:
                Xg = lse_subset(best, np.arange(n, dtype=np.int64), 8e-5, 80)
                vg = _max_cos(Xg)
                if vg < best_val:
                    best_val, best = float(vg), Xg.copy()
        except Exception:
            pass
    best = _normalize(best)
    return best
