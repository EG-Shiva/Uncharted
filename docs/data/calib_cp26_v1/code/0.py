import numpy as np

def solve():
    n = 26
    k = int(np.ceil(np.sqrt(n)))
    r = 0.5 / k
    pts = []
    for i in range(k):
        for j in range(k):
            if len(pts) < n:
                pts.append([(2 * i + 1) * r, (2 * j + 1) * r, r * 0.999])
    return np.array(pts)
