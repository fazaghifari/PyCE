import sys
sys.path.insert(0, "..")
import numpy as np
from pyce.PCE import PCE
from pyce.utils.samplingplan import sampling
import matplotlib.pyplot as plt

Xsamp,_ = sampling("halton",1,10)
Ysamp = Xsamp * np.sin(Xsamp*np.pi)

pceinfo = {}
pceinfo['x'] = Xsamp
pceinfo['y'] = Ysamp
pceinfo['degree'] = 5

model = PCE(pceinfo,True)
model.run()

xplot = np.linspace(0,1,100).reshape(-1,1)
yval = xplot * np.sin(xplot*np.pi)
ypred = model.predict(xplot)

plt.plot(xplot, yval, 'r--', label='Real')
plt.plot(xplot, ypred, 'k', label='Prediction')
plt.scatter(Xsamp, Ysamp, c='b', marker='+', label='Samples')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
