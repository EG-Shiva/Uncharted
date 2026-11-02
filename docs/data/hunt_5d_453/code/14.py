import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

def solve():
    t0 = time.time()
    HARD_END = 172.0
    # time allocations (seconds from start)
    MICRO_END = 55.0
    LP_END = 169.0

    rng = np.random.default_rng(0)
    np.random.seed(0)

    cands = []
    for fn in ("best.npy", "record.npy"):
        try:
            a = np.load(fn)
            a = np.asarray(a, dtype=np.float64)
            if a.shape == (453, 5):
                a = a / np.linalg.norm(a, axis=1, keepdims=True)
                if np.all(np.isfinite(a)):
                    cands.append(a)
        except Exception:
            pass
    if len(cands) == 0:
        X0 = rng.standard_normal((453, 5))
        X0 /= np.linalg.norm(X0, axis=1, keepdims=True)
        cands.append(X0)

    def max_cos_full(X, iu):
        G = X @ X.T
        c = G[iu]
        return float(np.max(c)), G

    n = 453
    d = 5
    iu_r, iu_c = np.triu_indices(n, 1)
    iu = (iu_r, iu_c)
    ar = np.arange(d)

    bestX = None
    bestV = float("inf")
    for cc in cands:
        G = cc @ cc.T
        m = float(np.max(G[iu]))
        if m < bestV:
            bestV = m
            bestX = cc.copy()
    x = bestX.copy()
    cur = bestV

    # ---------- Phase A: ultra-fine worst-pair shaving ----------
    # single isolated max => moving one endpoint tangentially away
    # reduces max by ~0.53*eps; must use eps < gap (~3e-8) to avoid
    # promoting a near-tie. Try single moves before double moves.
    eps_list = [3e-07, 1e-07, 5e-08, 2e-08, 1e-08, 5e-09, 2e-09]
    it = 0
    while time.time() - t0 < min(MICRO_END, HARD_END - 60):
        it += 1
        G = x @ x.T
        c = G[iu]
        a = int(np.argmax(c))
        m = float(c[a])
        if m < bestV - 1e-15:
            bestV = m
            bestX = x.copy()
            cur = m
        i = int(iu_r[a]); j = int(iu_c[a])
        cij = float(x[i] @ x[j])
        # tangent directions away
        ti = x[j] - cij * x[i]
        ni = float(np.linalg.norm(ti))
        if ni < 1e-12:
            rz = rng.standard_normal(d)
            rz -= float(rz @ x[i]) * x[i]
            ni2 = float(np.linalg.norm(rz))
            if ni2 < 1e-12:
                continue
            ti = rz / ni2
        else:
            ti /= ni
        tj = x[i] - cij * x[j]
        nj = float(np.linalg.norm(tj))
        if nj < 1e-12:
            rz = rng.standard_normal(d)
            rz -= float(rz @ x[j]) * x[j]
            nj2 = float(np.linalg.norm(rz))
            if nj2 < 1e-12:
                continue
            tj = rz / nj2
        else:
            tj /= nj
        found = False
        # try eps large->small, modes i-only, j-only, both
        for eps in eps_list:
            if found:
                break
            # i only
            y = x.copy()
            vi = x[i] - eps * ti
            vi /= np.linalg.norm(vi)
            y[i] = vi
            # fast max: full gram (cheap)
            G2 = y @ y.T
            m2 = float(np.max(G2[iu]))
            if m2 < m - 1e-13:
                x = y
                if m2 < bestV:
                    bestV = m2
                    bestX = y.copy()
                found = True
                break
            # j only
            y2 = x.copy()
            vj = x[j] - eps * tj
            vj /= np.linalg.norm(vj)
            y2[j] = vj
            G3 = y2 @ y2.T
            m3 = float(np.max(G3[iu]))
            if m3 < m - 1e-13:
                x = y2
                if m3 < bestV:
                    bestV = m3
                    bestX = y2.copy()
                found = True
                break
            # both
            y3 = x.copy()
            y3[i] = vi
            y3[j] = vj
            G4 = y3 @ y3.T
            m4 = float(np.max(G4[iu]))
            if m4 < m - 1e-13:
                x = y3
                if m4 < bestV:
                    bestV = m4
                    bestX = y3.copy()
                found = True
                break
        if not found:
            # no tiny move improves => near-tie with >=2 contacts, switch to LP
            # tiny random tangential kick to worst point at smallest scale to probe
            # only occasionally, else break
            if it % 20 == 0:
                # check if we are stuck for a while: do one LP-style combined probe
                # jiggle both with half eps
                pass
            # if stuck many times in a row, break to LP phase
            # count consecutive failures
            if not hasattr(solve, "_fail"):
                pass
            # use a counter
            try:
                x._fail_count = getattr(x, "_fail_count", 0) + 1
            except Exception:
                pass
            # simple: after 30 consecutive fails, break
            # implement via local var
            if it > 3000:
                # avoid infinite loop: go to LP after enough iters
                # check progress in last iters via bestV
                pass
            # break if no progress possible at 2e-09 level -> need collective move
            # we detect by trying one more ultra-small eps
            # if still fail, do a few iterations then exit phase
            # use stall counter
            if 'stall' not in locals():
                stall = 0
            stall = stall + 1 if not found else 0
            if stall > 40:
                break
        else:
            stall = 0
        # safety: limit iterations
        if it > 20000:
            break
        if time.time() - t0 > min(MICRO_END, HARD_END - 60):
            break

    # adopt best
    x = bestX.copy()

    # ---------- Phase B: tiny-radius trust-region LP ----------
    radius = 2e-07
    delta = 2.0e-06
    # pre-allocate index arrays
    while time.time() - t0 < min(LP_END, HARD_END - 1.0):
        if HARD_END - (time.time() - t0) < 2.0:
            break
        Gfull = x @ x.T
        c = Gfull[iu]
        m = float(np.max(c))
        if m < bestV:
            bestV = m
            bestX = x.copy()
        # select active set
        sel = c > m - delta
        cnt = int(np.count_nonzero(sel))
        if cnt == 0:
            radius = min(radius * 1.5, 5e-07)
            continue
        if cnt > 4000:
            # keep top 4000
            idx = np.argpartition(-c, 4000)[:4000]
            mask = np.zeros_like(sel)
            mask[idx] = True
            # intersect with sel? actually keep top 4000 regardless (they are within delta if cnt>4000)
            sel = mask
            cnt = 4000
        I = iu_r[sel]; J = iu_c[sel]
        cssel = c[sel]
        p = cnt
        nv = n * d + 1
        rows = np.repeat(np.arange(p), 2 * d + 1)
        cols = np.concatenate([I[:, None] * d + ar, J[:, None] * d + ar, np.full((p, 1), n * d)], axis=1).ravel()
        vals = np.concatenate([x[J], x[I], -np.ones((p, 1))], axis=1).ravel()
        try:
            A_ub = coo_matrix((vals, (rows, cols)), shape=(p, nv)).tocsr()
            A_eq = coo_matrix((x.ravel(), (np.repeat(np.arange(n), d), np.arange(n * d))), shape=(n, nv)).tocsr()
            obj = np.zeros(nv); obj[-1] = 1.0
            res = linprog(obj, A_ub=A_ub, b_ub=-cssel, A_eq=A_eq, b_eq=np.zeros(n),
                          bounds=[(-radius, radius)] * (n * d) + [(None, None)], method="highs")
        except Exception:
            radius *= 0.5
            if radius < 1e-11:
                radius = 1e-11
            continue
        if res.status != 0:
            radius *= 0.5
            if radius < 5e-11:
                # reset to best and try slightly larger delta
                x = bestX.copy()
                radius = 1e-07
                # shrink delta to focus
                delta = max(delta * 0.7, 5e-07)
            continue
        y = x + res.x[:-1].reshape(n, d)
        yn = np.linalg.norm(y, axis=1, keepdims=True)
        yn[yn == 0] = 1.0
        y = y / yn
        Gy = y @ y.T
        v = float(np.max(Gy[iu]))
        if v < m - 1e-13:
            x = y
            if v < bestV:
                bestV = v
                bestX = y.copy()
            radius = min(radius * 1.3, 5e-07)
            # keep delta proportional but bounded
            # if many constraints, keep delta; if few, shrink
            if cnt < 500:
                delta = max(delta * 0.9, 5e-07)
        else:
            radius *= 0.5
            if radius < 5e-11:
                x = bestX.copy()
                radius = 8e-08
                # alternate delta
                delta = 1.0e-06 if delta > 1.2e-06 else 2.0e-06
        # occasional micro-shave interleaving to catch single-pair wins without LP overhead
        if int(time.time() - t0) % 7 == 0:
            # one quick greedy shave step
            Gq = x @ x.T
            cq = Gq[iu]
            aq = int(np.argmax(cq))
            iq = int(iu_r[aq]); jq = int(iu_c[aq])
            cijq = float(x[iq] @ x[jq])
            tiq = x[jq] - cijq * x[iq]
            nq = float(np.linalg.norm(tiq))
            if nq > 1e-12:
                tiq /= nq
                for epsq in (5e-08, 1e-08):
                    yq = x.copy()
                    viq = x[iq] - epsq * tiq
                    viq /= np.linalg.norm(viq)
                    yq[iq] = viq
                    vq = float(np.max((yq @ yq.T)[iu]))
                    if vq < m - 1e-13:
                        x = yq
                        if vq < bestV:
                            bestV = vq
                            bestX = yq.copy()
                        break

    bestX = bestX / np.linalg.norm(bestX, axis=1, keepdims=True)
    return bestX
