import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from jedi.api.refactoring import inline
#%matplotlib inline
# generate random data
np.random.seed(24)
x = np.random.uniform(-5,5,25)
ϵ = 2*np.random.randn(25)
y = 2*x+ϵ
# alternate error as a function of x
ϵ2 = ϵ*(x+5)
y2 = 2*x+ϵ2
sns.regplot(x=x,y=y)
sns.regplot(x=x,y=y2)

# add a strong outlier for high x
x_high = np.append(x,5)
y_high = np.append(y2,160)
# add a strong outlier for low x
x_low = np.append(x,-4)
y_low = np.append(y2,160)

# calculate weights for sets with low and high outlier
sample_weights_low = [1/(x+5) for x in x_low]
sample_weights_high = [1/(x+5) for x in x_high]

# reshape for compatibility
X_low = x_low.reshape(-1, 1)
X_high = x_high.reshape(-1, 1)
# ---------
# import and fit an OLS model, check coefficients
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_low, y_low)
# fit WLS using sample_weights
WLS = LinearRegression()
WLS.fit(X_low, y_low, sample_weight=sample_weights_low)
print(model.intercept_, model.coef_)
print('WLS')
print(WLS.intercept_, WLS.coef_)
# run this yourself, don't trust every result you see online =)


model = LinearRegression()
model.fit(X_high, y_high)
WLS.fit(X_high, y_high, sample_weight=sample_weights_high)
print(model.intercept_, model.coef_)
print('WLS')
print(WLS.intercept_, WLS.coef_)
data_shape = "(feature_num,1)"
data_shape_var = {'feature_num': 4}
d = eval(data_shape, data_shape_var)
print(d)
_0feature_number = 5