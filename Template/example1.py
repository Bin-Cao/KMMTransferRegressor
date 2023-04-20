import sys

import pandas as pd

locdir = '/Users/jacob/Documents/GitHub/KMMTransferRegressor/KMMTR'
sys.path.append(locdir)
import KMMTR

# generate source domain data
import numpy as np
rng_S = np.random.RandomState(0)
# 20 source domian data
S_X = rng_S.uniform(-5, 5, 30)[:, np.newaxis]
S_y = 2*S_X[:, 0] + 5 +  0.8*rng_S.normal(1, 2.5, S_X.shape[0])
Sdataset = pd.DataFrame(S_X)
Sdataset['y'] = S_y
print(Sdataset.head())
print('#'*20)

# generate target domain data
rng_T = np.random.RandomState(10)
# 5 target domian data
T_X = rng_T.uniform(3, 8, 10)[:, np.newaxis]
T_y = 2*T_X[:, 0] + 5 +  0.8*rng_T.normal(1, 2.5, T_X.shape[0])
Tdataset = pd.DataFrame(T_X)
Tdataset['y'] = T_y
print(Tdataset.head())
print('#'*20)

# generate test data
rng_Ts = np.random.RandomState(13)
# 5 target domian data
Ts_X = rng_Ts.uniform(3, 8, 5)[:, np.newaxis]
Ts_y = 2*Ts_X[:, 0] + 5 +  .3 * rng_Ts.normal(1, 2.5, Ts_X.shape[0])
print('#'*20)

# plot the distribution of data
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(S_X,S_y, c='k', s=20, marker='x', label='Source Domain')
ax.scatter(T_X,T_y, c='r', s=20, marker='x',label='Target Domain')
plt.legend(loc=1)
ax.set_xlabel('x',fontsize=14)
ax.set_ylabel('y', fontsize=14)
plt.legend(fontsize = 14)
plt.savefig('DataDis.png',bbox_inches = 'tight',dpi=600)
plt.savefig('DataDis.svg',bbox_inches = 'tight',dpi=600)
plt.show()

# KMM regression
Reg = KMMTR.KMMTransferReg(Regressor='LR')
"""
    Parameters
    ----------
    Regressor : str or object, default='RF'
        A regression model used to fit the mapped data. If a string is passed, it should be one of {'RF', 'LR'}
        representing RandomForestRegressor and LinearRegression, respectively. Otherwise, an object with fit and
        predict methods that implements a regression algorithm can be passed.
    UpBound : float, default=1
        The upper bound of the beta for the target dataset during kernel mean matching.
    kernel : str, default='RBF'
        Kernel function used for kernel mean matching. It should be one of {'RBF', 'DotProduct', 'WhiteKernel', 'Matern'}
        representing the Radial Basis Function kernel, Dot Product kernel, White noise kernel and Matern kernel,
        respectively.
    Targets : int, default=1
        The number of target variables to be predicted.

    Methods
    -------
    fit(source_dataset,target_dataset,test_data)
        Fit the transfer learning regression model to the source dataset and target dataset, and predict the target variable
        on the test dataset.
"""
prevalue, beta  = Reg.fit(Sdataset,Tdataset,Ts_X,tao=0.6)
print('#'*20)

# plot the distribution of data with weights
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(S_X,S_y, c='k', marker='x',s=20, label='Source Domain')
ax.scatter(S_X,S_y, c='cyan', s=40*beta, alpha = 0.5)
ax.scatter(T_X,T_y, c='r', s=20, marker='x',label='Target Domain')
plt.legend(loc=1)
ax.set_xlabel('x',fontsize=14)
ax.set_ylabel('y', fontsize=14)
plt.legend(fontsize = 14)
plt.savefig('DataDis_weight.png',bbox_inches = 'tight',dpi=600)
plt.savefig('DataDis_weight.svg',bbox_inches = 'tight',dpi=600)
plt.show()


# compare the final prediction result by LR
from sklearn.metrics import r2_score
from sklearn import linear_model
mdoel = linear_model.LinearRegression()
pre = mdoel.fit(T_X,T_y).predict(Ts_X)
print('Without Transfer',r2_score(Ts_y,pre))
print('With Transfer',r2_score(Ts_y,prevalue))
