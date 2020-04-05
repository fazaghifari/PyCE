import sys
sys.path.insert(0, "..")
import numpy as np
from pyce.PCE import PCE
from pyce.utils.samplingplan import sampling
import matplotlib.pyplot as plt
from matplotlib import cm

def styb(x):  # Styblinski-Tang Function
    d = np.size(x,1)
    sum = 0
    for ii in range (0,d):
        xi = x[:,ii]
        new = xi**4 - 16*xi**2 + 5*xi
        sum = sum + new
    y = sum/2
    return y

def cust(x):
    x1 = x[:,0]
    x2 = x[:,1]
    y = 3 + 5*x1 + 0.2*x2 + x1**2 + 0.1*x2**2 + 0.25*x1*x2
    return y

def run():
    nvar = 2
    nsamp = 30
    ub = np.array([5, 5])
    lb = np.array([-5, -5])
    _,xsamp = sampling('sobol',nvar,nsamp,result='real',upbound=ub,lobound=lb)
    ysamp = cust(xsamp).reshape(-1,1)

    pceinfo = {}
    pceinfo['x'] = xsamp
    pceinfo['y'] = ysamp
    pceinfo['degree'] = 2

    model = PCE(pceinfo, True)
    model.run()

    neval = 10000
    xx = np.linspace(-5, 5, 100)
    yy = np.linspace(-5, 5, 100)
    Xevalx, Xevaly = np.meshgrid(xx, yy)
    Xeval = np.zeros(shape=[neval, 2])
    Xeval[:, 0] = np.reshape(Xevalx, (neval))
    Xeval[:, 1] = np.reshape(Xevaly, (neval))

    # Evaluate output
    yeval = model.predict(Xeval)
    yact = cust(Xeval).reshape(-1,1)

    subs = np.transpose((yact - yeval))
    subs1 = np.transpose((yact - yeval) / yact)
    RMSE = np.sqrt(np.sum(subs ** 2) / neval)
    RMSRE = np.sqrt(np.sum(subs1 ** 2) / neval)
    MAPE = 100 * np.sum(abs(subs1)) / neval
    print("RMSE = ", RMSE)
    print("RMSRE = ", RMSRE)
    print("MAPE = ", MAPE, "%")

    yeval1 = np.reshape(yeval, (100, 100))
    x1eval = np.reshape(Xeval[:, 0], (100, 100))
    x2eval = np.reshape(Xeval[:, 1], (100, 100))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x1eval, x2eval, yeval1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    sc = ax.scatter(xsamp[:,0],xsamp[:,1],ysamp[:])
    plt.show()

if __name__ == '__main__':
    run()