import time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

def _max_cos(x):
    g = x @ x.T
    np.fill_diagonal(g, -2.0)
    return float(np.max(g))

def _norm(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n

def solve():
    t0 = time.time()
    DEAD = t0 + 172.0
    PHASE = t0 + 138.0
    np.random.seed(7)
    n = 292; d = 4
    try:
        xb = np.load("best.npy")
    except Exception:
        try:
            xb = np.load("record.npy")
        except Exception:
            xb = np.random.randn(n, d)
    xb = np.asarray(xb, dtype=np.float64)
    if xb.shape != (n, d):
        xb = np.random.randn(n, d)
    xb = _norm(xb)

    iuI, iuJ = np.triu_indices(n, 1)

    def make_fun(alpha):
        def fun(yf):
            y = yf.reshape(n, d)
            nr = np.sqrt((y*y).sum(axis=1, keepdims=True))
            nr = np.maximum(nr, 1e-12)
            x = y / nr
            G = x @ x.T
            g = G[iuI, iuJ]
            m = np.max(g)
            a = alpha * (g - m)
            e = np.exp(a)
            S = np.sum(e)
            L = m + np.log(S) / alpha
            w = e / S
            W = np.zeros((n, n), dtype=np.float64)
            W[iuI, iuJ] = w
            W = W + W.T
            Gx = W @ x
            dot = (Gx * x).sum(axis=1, keepdims=True)
            Gy = (Gx - dot * x) / nr
            return float(L), Gy.ravel().astype(np.float64)
        return fun

    def lbfgs(x_init, alpha, maxiter):
        fun = make_fun(alpha)
        y0 = x_init.ravel().astype(np.float64)
        try:
            res = minimize(fun, y0, jac=True, method="L-BFGS-B",
                           options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-7, "maxfun": 8000})
            y = res.x.reshape(n, d)
        except Exception:
            return x_init
        nr = np.linalg.norm(y, axis=1, keepdims=True)
        nr = np.maximum(nr, 1e-12)
        x = y / nr
        if not np.all(np.isfinite(x)):
            return x_init
        return x

    def smooth2(x_, a1=80.0, m1=70, a2=260.0, m2=70):
        x_ = lbfgs(x_, a1, m1)
        if time.time() > DEAD - 3:
            return x_
        x_ = lbfgs(x_, a2, m2)
        return x_

    def contacts_info(x_):
        G = x_ @ x_.T
        np.fill_diagonal(G, -2.0)
        glo = float(np.max(G))
        mx = np.max(G, axis=1)
        cnt = (G > glo - 1e-9).sum(axis=1)
        return G, glo, mx, cnt

    def find_holes(x_keep, k, nsamp=20000):
        chunk = 5000
        cands = []
        scores = []
        for s in range(0, nsamp, chunk):
            m = min(chunk, nsamp - s)
            c = np.random.randn(m, d)
            c /= np.linalg.norm(c, axis=1, keepdims=True)
            dots = c @ x_keep.T
            sc = dots.max(axis=1)
            cands.append(c); scores.append(sc)
        cands = np.vstack(cands); scores = np.concatenate(scores)
        order = np.argsort(scores)
        picked = []
        for idx in order:
            c = cands[idx]
            ok = True
            for p in picked:
                if float(p @ c) > 0.88:
                    ok = False; break
            if ok:
                picked.append(c)
                if len(picked) >= k:
                    break
        while len(picked) < k:
            picked.append(cands[order[len(picked)]])
        return np.array(picked)

    def lp_polish(x_, deadline, radius0=1e-3):
        x_ = _norm(x_)
        best = _max_cos(x_)
        radius = radius0
        while time.time() < deadline and radius > 1e-14:
            if time.time() > DEAD - 2:
                break
            c = (x_ @ x_.T)[iuI, iuJ]
            m = float(c.max())
            sel = c > m - max(20*radius, 1e-9)
            I = iuI[sel]; J = iuJ[sel]
            cf = c[sel]
            p = len(I)
            if p == 0:
                break
            nv = n*d+1
            try:
                rows = np.repeat(np.arange(p), 2*d+1)
                cols = np.concatenate([I[:,None]*d+np.arange(d), J[:,None]*d+np.arange(d), np.full((p,1), n*d)],1).ravel()
                vals = np.concatenate([x_[J], x_[I], -np.ones((p,1))],1).ravel()
                A_ub = coo_matrix((vals,(rows,cols)),shape=(p,nv)).tocsr()
                A_eq = coo_matrix((x_.ravel(),(np.repeat(np.arange(n),d),np.arange(n*d))),shape=(n,nv)).tocsr()
                obj = np.zeros(nv); obj[-1]=1.0
                res = linprog(obj,A_ub=A_ub,b_ub=-cf,A_eq=A_eq,b_eq=np.zeros(n),
                              bounds=[(-radius,radius)]*(n*d)+[(None,None)],method="highs")
            except Exception:
                radius*=0.5; continue
            if getattr(res,"status",1)!=0:
                radius*=0.5; continue
            try:
                y = x_ + res.x[:-1].reshape(n,d)
            except Exception:
                radius*=0.5; continue
            y = _norm(y)
            v = _max_cos(y)
            if v < best:
                x_, best, radius = y, v, min(radius*1.5,0.05)
            else:
                radius*=0.5
        return x_

    best_val = _max_cos(xb)
    x_best = xb.copy()
    cur = xb.copy(); cur_val = best_val
    # ensure starting point is locally optimal-ish with quick squeeze
    # (cheap, one short pass)
    cur = smooth2(cur, 80.0, 40, 250.0, 40)
    cur_val = _max_cos(cur)
    if cur_val < best_val:
        best_val = cur_val; x_best = cur.copy()

    it = 0
    T = 0.0010
    while time.time() < PHASE:
        it += 1
        if time.time() > DEAD - 5:
            break
        if it % 15 == 0:
            cur = x_best.copy(); cur_val = best_val
        # analyse current basin occasionally
        try:
            Gg, glo, mx, cnt = contacts_info(cur)
        except Exception:
            Gg, glo, mx, cnt = None, cur_val, None, None
        r = np.random.rand()
        pert = None
        if r < 0.38:
            sigma = float(np.random.choice([0.012,0.025,0.05,0.09]))
            pert = cur + sigma*np.random.randn(n,d)
            pert = _norm(pert)
        elif r < 0.62:
            # jammed neighbourhood shake
            knear = int(np.random.choice([12,20,35]))
            sig_big = float(np.random.choice([0.10,0.18,0.30]))
            pert = cur.copy() + 0.004*np.random.randn(n,d)
            if cnt is not None and np.random.rand()<0.8:
                cen = int(np.argmax(cnt + np.random.randn(n)*0.5))
            else:
                cen = np.random.randint(n)
            dots = cur @ cur[cen]
            nn = np.argsort(-dots)[:knear]
            pert[nn] += sig_big*np.random.randn(knear,d)
            pert = _norm(pert)
        elif r < 0.78:
            # coherent cap move: pick random cap + common tangent drift
            cdir = np.random.randn(d); cdir/=np.linalg.norm(cdir)
            proj = cur @ cdir
            thr = np.sort(proj)[n-int(np.random.choice([12,20,35,50]))]
            mask = proj >= thr
            idx = np.where(mask)[0]
            # random skew matrix
            A = np.random.randn(d,d); A = A - A.T
            theta = float(np.random.choice([0.05,0.12,0.22]))
            pert = cur.copy() + 0.004*np.random.randn(n,d)
            drift = (A @ cur[idx].T).T * theta
            pert[idx] += drift
            # also translate cap slightly
            pert[idx] += 0.02*np.random.randn(1,d)
            pert = _norm(pert)
        else:
            # hole re-insertion of 1-4 points
            nmove = int(np.random.choice([1,2,3,4]))
            if cnt is not None:
                slack = glo - mx
                order_loose = np.argsort(-slack)
                order_jam = np.argsort(-cnt)
            else:
                order_loose = np.random.permutation(n)
                order_jam = order_loose
            cand_idx = []
            if np.random.rand()<0.6:
                cand_idx.append(int(order_jam[0]))
            # include a rattler if any
            ratt = np.where(cnt==0)[0] if cnt is not None else []
            if len(ratt)>0 and np.random.rand()<0.7 and len(cand_idx)<nmove:
                cand_idx.append(int(ratt[np.random.randint(len(ratt))]))
            while len(cand_idx)<nmove:
                if np.random.rand()<0.7:
                    pool = order_loose[:25]
                    cand_idx.append(int(pool[np.random.randint(len(pool))]))
                else:
                    cand_idx.append(int(np.random.randint(n)))
            cand_idx = cand_idx[:nmove]
            keep = np.ones(n,bool); keep[cand_idx]=False
            x_keep = cur[keep]
            ns = 20000 if (PHASE-time.time())>20 else 12000
            holes = find_holes(x_keep, nmove, nsamp=ns)
            pert = cur.copy()
            pert[cand_idx]=holes[:len(cand_idx)]
            pert[cand_idx]+=0.008*np.random.randn(len(cand_idx),d)
            pert=_norm(pert)

        # polish
        x1 = lbfgs(pert, 75.0, 60)
        x2 = lbfgs(x1, 230.0, 60)
        v2 = _max_cos(x2)
        if it % 6 == 0 and time.time() < PHASE - 5:
            x3 = lbfgs(x2, 450.0, 40)
            v3 = _max_cos(x3)
            if v3 < v2:
                x2, v2 = x3, v3
        if v2 < best_val - 1e-14:
            best_val = v2; x_best = x2.copy()
            cur, cur_val = x2.copy(), v2
        else:
            if v2 < cur_val or np.random.rand() < np.exp(-(v2-cur_val)/max(T,1e-9)):
                cur, cur_val = x2.copy(), v2
        # anneal T slightly
        T *= 0.9995
        T = max(T, 0.00025)

    # final squeeze: sharp L-BFGS then LP
    try:
        if time.time() < DEAD - 8:
            x_best = lbfgs(x_best, 500.0, 120)
        if time.time() < DEAD - 6:
            x_best = lbfgs(x_best, 800.0, 80)
    except Exception:
        pass
    rem_lp = DEAD - time.time() - 1.5
    if rem_lp > 4:
        try:
            x_best = lp_polish(x_best, time.time()+min(rem_lp,38))
        except Exception:
            pass
    x_best = _norm(x_best)
    return x_best
