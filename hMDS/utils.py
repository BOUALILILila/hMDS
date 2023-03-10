# Some methods were copied from https://github.com/HazyResearch/hyperbolics/blob/master/utils/distortions.py
import numpy as np
from joblib import Parallel, delayed

def entry_is_good(h, h_rec): return (not np.isnan(h_rec)) and (not np.isinf(h_rec)) and h_rec != 0 and h != 0

def distortion_entry(h,h_rec,me,mc):
    avg = abs(h_rec - h)/h
    if h_rec/h > me: me = h_rec/h
    if h/h_rec > mc: mc = h/h_rec
    return (avg,me,mc)

def distortion_row(H1, H2, n, row):
    mc, me, avg, good = 0,0,0,0
    for i in range(n):
        if i != row and entry_is_good(H1[i], H2[i]):
            (_avg,me,mc) = distortion_entry(H1[i], H2[i],me,mc)
            good        += 1
            avg         += _avg
    avg /= good if good > 0 else 1.0
    return (mc, me, avg, n-1-good)

def distortion(H1, H2, n, jobs=1):
    H1, H2 = np.array(H1), np.array(H2)
    dists = Parallel(n_jobs=jobs)(delayed(distortion_row)(H1[i,:],H2[i,:],n,i) for i in range(n))
    dists = np.vstack(dists)
    # mc = max(dists[:,0])
    # me = max(dists[:,1])
    wc = max(dists[:,0])*max(dists[:,1])
    avg = sum(dists[:,2])/n
    # bad = sum(dists[:,3])
    return {
        'avg_distortion': avg, # best 0
        'wc_distortion' : wc,  # best 1
    }


# Simple distortion metrics implementation on symmetric matrices defined in https://arxiv.org/pdf/1804.03329.pdf 
def avg_distortion(Dnew, Dold):
    n,n = Dnew.shape
    d = 0
    for i in range(n):
        for j in range(i):
            d += abs(Dnew[i,j]-Dold[i,j])/Dold[i,j]
    return 2*d/(n*(n-1))

def wc_distortion(Dnew,Dold):
    n,n = Dnew.shape
    d = 0
    d_max = 0
    d_min = 1
    for i in range(n):
        for j in range(i):
            d = Dnew[i,j]/Dold[i,j]
            if d > d_max :
                d_max = d
            if d < d_min : 
                d_min = d 
    return d_max/d_min


def set_seed(seed):
    np.random.seed(seed)

def quadratic_form(x):
    """ Minkowski qudratic form of a point in Gans Model (x0 is not in x).
    """
    x0 = np.sqrt(1 + np.linalg.norm(x)**2)
    return x0**2 - np.dot(x,x)