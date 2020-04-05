import numpy as np
from itertools import combinations
from numpy.matlib import repmat

def MonCof(d,n):
    """
    This function is used to generate monomial coefficient
    input : d = degree
            n = dimension
    """
    temp = np.arange(1,(d+n))
    c = np.array(list(combinations(temp,n-1)))
    c = c.astype(int)
    m = np.size(c,0)
    t = np.ones(shape=[m,d+n-1],dtype=int)
    tempt1 = np.transpose(np.array([np.arange(0,m,dtype=int)]))
    tempt2 =np.matlib.repmat(tempt1,1,n-1)
    tempt3 = tempt2 +(c-1)*m
    for ii in range(np.size(tempt3,0)):
        for jj in range(np.size(tempt3,1)):
            x1 = int(np.round(tempt3[ii,jj] % m))
            x2 = int(np.floor(tempt3[ii,jj]/m))
            t[x1,x2] = 0
    tempu = np.vstack((np.zeros(m),np.transpose(t)))
    u = np.vstack((tempu,np.zeros(m)))
    v = np.cumsum(u,0)
    tempreshape1 = np.array([np.transpose(v)[np.transpose(u)==0]])
    tempreshape = np.reshape(tempreshape1,(n+1,m),order='F')
    x = np.transpose(np.diff(tempreshape,1,0))
    return x

def lassopce(x, y, method='lars', stop=0, usegram=True, gram=None, trace=0):
    """
    LARS algorithm for performing LARS or LASSO.
    Reference: 'Least Angle Regression' by Bradley Efron et al, 2003.

    Args:
        - x (nparray): design variables.
        - y (nparray): corresponding response.
        - method (str):  determines whether least angle regression or lasso regression should be performed.
        - stop (int): Nonzero STOP will perform least angle or lasso regression with early stopping.
                      If STOP is negative, STOP is an integer that determines the desired number of variables.
                      If STOP is positive, it corresponds to an upper bound on the L1-norm of the BETA coefficients.
        - usegram (bool): Specifies whether the Gram matrix X'X should be calculated or not.
        - gram (nparray): Precomputed gram matrix.
        - trace (int): Nonzero TRACE will print the adding and subtracting of variables as all LARS/lasso solutions are found.

    Return:
        - beta (nparray): An array where each row contains the predictor coefficients of one iteration.
                          A suitable row is chosen using e.g. cross-validation, possibly including interpolation
                          to achieve sub-iteration accuracy.
    """
    ##### VARIABLE SETUP #####
    n, p = np.shape(x)
    nvars = np.min((n-1,p))
    maxk = 80*nvars  # max number of iterations

    if stop == 0:
        beta = np.zeros((2*nvars,p))
    elif stop < 0:
        beta = np.zeros((2*np.round(-stop),p))
    else:
        beta = np.zeros((100,p))

    mu = np.zeros((n,1))
    I = np.arange(0,p)
    A = None

    if usegram and gram is None:
        gram = np.dot(x.T, x)
    elif not usegram:
        R = None

    lassocond = False  # Lasso condition boolean
    stopcond = False  # Early stopping condition boolean
    k = 0   # Iteration count
    var_i = 0 # Current number of variables

    ##### LARS MAIN LOOP #####
    k = 0
    A = 0
    I = np.delete(I,0)
    var_i += 1

    errloo = []
    beta[k+1,A] = np.linalg.lstsq(x[:,A].reshape(-1,1),y,rcond=None)[0]
    mu = (x[:,A] * beta[k+1,A]).reshape(-1,1)
    errloo.append(loocalc(x[:,A],mu,y))

    while var_i < nvars and not stopcond and k < maxk:
        k += 1
        c = np.dot(x.T,(y-mu))

        c_max = np.max(abs(c[I]))
        j = np.argmax(abs(c[I]))
        j = I[j]

        if not lassocond:
            if not usegram:
                R = cholinsert(R,x[:,j],x[:,A])
            A = np.hstack([A,j])
            I = I[np.where(I != j)]
            var_i += 1

        si = np.sign(c[A])

        if usegram:
            s = si * np.ones((1,var_i))
            gaa = gram[repmat(A,len(A),1),repmat(A,len(A),1).T]
            ga1 = np.linalg.solve((gaa*s.T*s),np.ones((var_i,1)))
            aa = 1 / np.sqrt(np.sum(ga1))
            w = aa*ga1*si
        else:
            ga1 = np.linalg.solve(R, np.linalg.solve(R.T,si))
            aa = 1 / np.sqrt(np.sum(ga1*si))
            w = aa * ga1
        u = np.dot(x[:,A], w)

        if var_i == nvars:
            gamma = c_max/aa
        else:
            a = np.dot(x.T,u)
            temp = np.vstack(((c_max-c[I])/(aa - a[I]),(c_max+c[I])/(aa + a[I])))
            gamma = np.min(np.hstack((temp[np.where(temp>0)],c_max/aa)))

        mu += gamma*u

        if np.size(beta,0) < k+1:
            beta = np.vstack((beta,np.zeros((np.size(beta,0),p))))

        beta[k+1,A] = np.linalg.lstsq(x[:,A],y, rcond=None)[0].flatten()
        ypr = np.dot(x[:,A] , beta[k+1,A].reshape(-1,1))
        errloo.append(loocalc(x[:,A],ypr,y))

        ##### Early Stopping condition #####
        if stop < 0:
            stopcond = var_i >= -stop

        if stop > 0:
            t2 = np.sum(abs(beta[k+1,:]))
            if t2 >= stop:
                t1 = np.sum(abs(beta[k,:]))
                s = (stop - t1)/(t2 - t1)
                beta[k+1,:] = beta[k,:] + s*(beta[k+1,:] - beta[k,:])
                stopcond = 1

        if lassocond:
            if not usegram:
                raise NotImplementedError('UseGram still not implemented')
            I = np.array([I, A[j]])
            A = np.delete(A,j)
            var_i -= 1

    ##### Trim Beta #####
    if np.size(beta,0) > k+1:
        beta = beta[:k+2,:]

    return beta,errloo


def loocalc(phis,ypr,ytr):
    phis = phis.reshape(-1,1)
    phie = np.dot(phis,(np.linalg.solve((np.dot(phis.T,phis)),phis.T)))
    dp = np.diag(phie)
    errloo = (ytr-ypr) / (1 - dp[:len(ypr)].reshape(-1,1))
    ery = np.mean(abs(errloo))

    return ery

def cholinsert(R,x1,x2):
    x1 = x1.reshape(-1,1)
    x2 = x2.reshape(-1,1)
    diag_k = np.dot(x1.T,x1)
    if R is None:
        R = np.sqrt(diag_k)
    else:
        col_k = np.dot(x1.T,x2)
        R_k = np.linalg.solve(R.T,col_k)
        R_kk = np.sqrt(diag_k - np.dot(R_k.T, R_k))
        R = np.vstack((np.hstack((R,R_k)), np.hstack(np.zeros((1,np.size(R,1))), R_kk)))

    return R
