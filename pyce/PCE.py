import numpy as np
from sobolsampling.sobol_new import sobol_points
from pyce.utils.misc import MonCof, lassopce
from scipy import special as spc
import numpy.matlib
import math

class PCE:

    def __init__(self,pceinfo,disp=False):
        checkedinfo = pceinfocheck(pceinfo,disp=disp)
        for key, value in checkedinfo.items():
            exec('self.' + key + '= value')

        self.nsamp, self.nvar =  np.shape(self.x)
        self.bounds = None
        self.info = None
        self.idh = None
        self.cofo = None
        self.errloo = None
        self.II = None

    def run(self):
        self.bounds = np.hstack((-np.ones((self.nvar, 1)), np.ones((self.nvar, 1))))
        self.info = np.ones(self.nvar)
        self.idh = self.totaltrunc(self.degree, self.p_trunc)
        phi = self.calc_phi(self.x, self.idh, self.bounds, self.info)
        self.cofo, elop = lassopce(phi,self.y)
        self.errloo = np.min(elop)
        self.II = np.argmin(elop)

    def predict(self,xval):
        phi_pred = self.calc_phi(xval, self.idh, self.bounds, self.info)
        y_pred = np.dot(phi_pred, self.cofo[self.II,:].reshape(-1,1))
        return y_pred

    def calc_phi(self, xval, idh, bounds, info):
        nsamp = np.size(xval,0)
        phi = np.ones((nsamp,np.size(idh,0)))
        na = []

        for po in range(self.nvar):
            if info[po] == 1:
                tempna = 2* ((xval[:,po].reshape(-1,1) - numpy.matlib.repmat(bounds[po,0],nsamp,1))/
                             (numpy.matlib.repmat(bounds[po,1],nsamp,1) -
                              numpy.matlib.repmat(bounds[po,0],nsamp,1))) - 1
            else:
                tempna = ((xval[:,po].reshape(-1,1) - numpy.matlib.repmat(bounds[po,0],nsamp,1)) /
                          (np.sqrt(2 * numpy.matlib.repmat(bounds[po,1],nsamp,1)**2)))
            na.append(tempna)

        for ii in range(np.size(idh,0)):
            h1 = np.ones((nsamp,1))
            ids = idh[ii,:]

            for jj, ids_i in enumerate(ids):
                if info[jj] == 1:
                    h1 *= np.polyval(spc.legendre(ids_i),na[jj]) / np.sqrt(1 / (2*ids_i+1))
                elif info[jj] == 2:
                    h1 *= ((1 / (2**(ids_i/2))) * np.polyval(spc.hermite(ids_i),na[jj])/ np.sqrt(math.factorial(ids_i)))

            phi[:,ii] = h1.flatten()

        return phi

    def totaltrunc(self,nix,q):
        """
           Generate polynomial indices for the trend function by using total-order
           truncation or hyperbolic truncation (the former is if q = 1 and the latter is if q < 1)

           inputs:
             nix - Maximum polynomial order
             nvar - number of variables
             q - hyperbolic truncation parameter

           output:
             idx - Indices of polynomial bases

           Generate index for polynomial chaos expansion
           """

        idx = []
        for i in range(nix, -1, -1):
            if i == nix:
                idx = MonCof(i, self.nvar)
            else:
                idx = np.vstack((idx, MonCof(i, self.nvar)))
        idx = np.flip(idx, 0)

        # Now truncate further (if q = 1, this equals to total-order expansion)
        if q < 1:
            try:
                pow = 1 / q
            except ZeroDivisionError:
                pow = float('Inf')
            idp = np.sum(idx ** q, 1) ** (pow)
            idx = idx[idp <= (nix + 0.000001), :]

        return idx


def pceinfocheck(pceinfo,disp=False):
    if not all(key in pceinfo for key in ('x', 'y')):
        raise AssertionError('key x and y are required.')
    else:
        pass

    assert (np.ndim(pceinfo['x']) == 2), "x requires 2 dimensional array with shape = nsamp x nvar"
    assert (np.ndim(pceinfo['y']) == 2), "y requires 2 dimensional array with shape = nsamp x 1"

    if 'degree' not in pceinfo:
        pceinfo['degree'] = 5
        if disp:
            print("PCE degree is set to 5")
    else:
        pass

    if 'p_trunc' not in pceinfo:
        pceinfo['p_trunc'] = 1
        if disp:
            print("PCE hyperbolic truncation is set to 1. this equals to total-order expansion")
    else:
        pass

    return pceinfo