import time
import numpy as np

def _norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

def _maxcos(X):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    return float(np.max(G))

def _contacts(X, thr=1e-9):
    G = X @ X.T
    np.fill_diagonal(G, -2.0)
    m = float(np.max(G))
    cnt = np.sum(G > m - thr, axis=1)
    return cnt, m

def _polish(X, steps, b0, b1, lr0, lr1, mu=0.0):
    n = X.shape[0]
    v = np.zeros_like(X)
    di = np.diag_indices(n)
    for s in range(steps):
        frac = s / max(1, steps - 1)
        beta = b0 + (b1 - b0) * frac
        lr = lr0 + (lr1 - lr0) * frac
        G = X @ X.T
        np.fill_diagonal(G, -2.0)
        m = float(np.max(G))
        W = np.exp(beta * (G - m))
        W[di] = 0.0
        S = float(np.sum(W)) * 0.5
        if not np.isfinite(S) or S <= 0:
            break
        Gr = (W @ X) / S
        dot = np.sum(Gr * X, axis=1, keepdims=True)
        Gt = Gr - dot * X
        if mu != 0.0:
            v = mu * v + Gt
            step = v
        else:
            step = Gt
        X = X - lr * step
        nr = np.linalg.norm(X, axis=1, keepdims=True)
        nr[nr == 0] = 1.0
        X = X / nr
    return X

def _hole_refine(v, Y, steps=200, T0=0.008, T1=0.001, lr0=0.08, lr1=0.01):
    v = v / (np.linalg.norm(v) + 1e-30)
    for k in range(steps):
        f = k / max(1, steps - 1)
        Tk = T0 * (T1 / T0) ** f if T1 > 0 else T0
        lrk = lr0 + (lr1 - lr0) * f
        s = Y @ v
        m = float(np.max(s))
        w = np.exp((s - m) / Tk)
        sw = float(np.sum(w))
        if not np.finite(sw) or sw <= 0:
            break
        w = w / sw
        g = w @ Y
        g = g - np.dot(g, v) * v
        v = v - lrk * g
        nv = np.linalg.norm(v)
        if not np.finite(nv) or nv == 0:
            break
        v = v / nv
    return v

def solve():
    t0 = time.time()
    DEADLINE = 172.0
    rng = np.random.default_rng(12345)
    try:
        x0 = np.load("best.npy")
    except Exception:
        try:
            x0 = np.load("record.npy")
        except Exception:
            x0 = rng.standard_normal((76, 15))
    x0 = np.asarray(x0, dtype=np.float64)
    if x0.shape != (76, 15):
        x0 = rng.standard_normal((76, 15))
    x0 = _norm(x0)
    best_overall = x0.copy()
    best_val = _maxcos(best_overall)

    removal_sizes = [20, 24, 16, 28]
    # time-boxed runs
    for R in removal_sizes:
        if time.time() - t0 > 152:
            break
        try:
            cnt, mcur = _contacts(best_overall, thr=1e-9)
        except Exception:
            cnt = np.zeros(76, dtype=int)
        ratt = np.where(cnt == 0)[0].tolist()
        order = list(np.argsort(-cnt))
        removed = []
        for i in ratt:
            if len(removed) < R:
                removed.append(int(i))
        for i in order:
            if len(removed) >= R:
                break
            if int(i) not in removed:
                removed.append(int(i))
        # if fewer rattlers than expected, fill anyway
        while len(removed) < R:
            for i in order:
                if int(i) not in removed:
                    removed.append(int(i))
                    break
        keep_mask = np.ones(76, dtype=bool)
        keep_mask[removed] = False
        core = best_overall[keep_mask].copy()
        # aggressive core optimization
        # stage 1 broad + stage 2 sharp
        core = _polish(core, 2200, 60.0, 600.0, 0.02, 0.0015, mu=0.5)
        if time.time() - t0 > 158:
            # still try to continue if possible else abort run
            pass
        core = _polish(core, 600, 800.0, 2500.0, 0.002, 0.0002, mu=0.0)
        cv = _maxcos(core) if core.shape[0] > 1 else 0.0
        # if core not open enough, extra anneal
        if cv > 0.15 and (time.time() - t0 < 150):
            core = _polish(core, 1200, 40.0, 400.0, 0.025, 0.002, mu=0.6)
            cv = _maxcos(core)
        # sequential reinsertion; randomize insertion order deterministically
        ins_order = removed.copy()
        # shuffle with rng (deterministic sequence)
        perm = rng.permutation(len(ins_order))
        ins_order = [ins_order[i] for i in perm]
        cur = core
        aborted = False
        for k in range(len(ins_order)):
            if time.time() - t0 > 160:
                aborted = True
                break
            Y = cur
            mcur2 = Y.shape[0]
            # sample >=50000 candidates
            NS = 50000
            # generate in one block (6MB) - ok
            C = rng.standard_normal((NS, 15))
            C = _norm(C)
            # scores: max cosine to Y
            # chunk to limit peak memory (NS x mcur2 = 3.75M)
            best_score = np.inf
            best_v = None
            CH = 10000
            for a in range(0, NS, CH):
                Cb = C[a:a+CH]
                S = Cb @ Y.T
                worst = np.max(S, axis=1)
                bi = int(np.argmin(worst))
                if float(worst[bi]) < best_score:
                    best_score = float(worst[bi])
                    best_v = Cb[bi].copy()
            v0 = best_v
            v0 = _hole_refine(v0, Y, steps=200, T0=0.008, T1=0.0012, lr0=0.08, lr1=0.01)
            # insert
            cur = np.vstack([Y, v0.reshape(1, -1)])
            cur = _norm(cur)
            # re-polish growing code 300-500 steps
            # adapt steps to size/time
            remain = 160 - (time.time() - t0)
            left = len(ins_order) - k - 1
            if remain < 2 or (left > 0 and remain / (left + 1) < 0.25):
                ps = 150
            else:
                ps = 400
            # use moderate beta schedule
            cur = _polish(cur, ps, 200.0, 1500.0, 0.008, 0.0006, mu=0.0)
        if aborted:
            continue
        # final sharpen of completed 76
        if cur.shape[0] != 76:
            continue
        cur = _polish(cur, 700, 800.0, 3000.0, 0.0015, 0.00015, mu=0.0)
        v = _maxcos(cur)
        if v < best_val - 1e-13 and np.all(np.isfinite(cur)):
            best_val = v
            best_overall = cur.copy()

    # reserve last seconds: short polish of incumbent (LP-free fast polish)
    try:
        if time.time() - t0 < DEADLINE - 3:
            nb = min(900, int((DEADLINE - 3 - (time.time() - t0)) * 300) + 100)
            nb = max(150, min(900, nb))
            cand = _polish(best_overall.copy(), nb, 900.0, 3200.0, 0.001, 0.0001, mu=0.0)
            vv = _maxcos(cand)
            if vv < best_val and np.all(np.isfinite(cand)):
                best_overall = cand
                best_val = vv
    except Exception:
        pass

    best_overall = _norm(best_overall)
    best_overall = np.where(np.isfinite(best_overall), best_overall, 0.0)
    nr = np.linalg.norm(best_overall, axis=1, keepdims=True)
    nr[nr < 1e-12] = 1.0
    best_overall = best_overall / nr
    if best_overall.shape != (76, 15) or not np.all(np.isfinite(best_overall)):
        return _norm(x0)
    return best_overall
