
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import datasets, metrics
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
import seaborn as sns
from math import sqrt
import RegscorePy


# %%
#reload in the train and test datasets
train_df = pd.read_csv("E:\\Spring 2021\\Classes\\540\\Project\\Project540\\TrainingData.txt", header=None, sep='\s+')
test_df = pd.read_csv("E:\\Spring 2021\\Classes\\540\\Project\\Project540\\TestData.txt", header=None, sep="\s+")

#load headers for the data sets
train_df.columns= ['index','precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'kitchen', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist', 'deathrate' ]
test_df.columns= ['index','precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'kitchen', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist']

#instead of eliminating the values that have NaN as an entry, replace with the average
#calculate means
train_df.mean()
test_df.mean()

#replace with means
train_df = train_df.fillna(train_df.mean())
test_df = test_df.fillna(test_df.mean())

#These are the independent variables, or regressors of the training data.
var = ['precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'kitchen', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist', 'deathrate']
independent_var = ['precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist']

#training variables
x_train = train_df[independent_var]
y_train = train_df['deathrate']


# %%
# this is to plot the points against each other to see the if they have any correlation
var = ['precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'kitchen', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist', 'deathrate']
sns.pairplot(train_df[var], diag_kind="kde")


# %%
#The above plots could be summarized as a Correaltion matrix
usecols = ['precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'kitchen', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist']
mask = np.zeros_like(train_df[usecols].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)
sns.heatmap(train_df[usecols].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="Blues", linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});


# %%
#Ridge regression
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
ridge_model = RidgeCV(alphas=np.arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
ridge_model.fit(x_train, y_train)
print(ridge_model.alpha_)


# %%
#see which features were eliminated from the model
Ridgecoeff = pd.DataFrame(ridge_model.coef_)
Ridgecoeff['Features'] = ['precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist']
Ridgecoeff.head()
Ridgecoeff


# %%
ridge_y = ridge_model.predict(test_df[independent_var])
test_df['Ridge prediction'] = ridge_y
test_df.head()
test_df


# %%
#with the ridge model, let's check on how well the model predicts for the training values. 
ridge_y_train = ridge_model.predict(x_train)
train_df['Ridge prediction'] = ridge_y_train
train_df.head()
train_df


# %%
rmse1 = sqrt(mean_squared_error(y_train, ridge_y_train)) 
rmse1


# %%
from sklearn import linear_model
Lasso_model = linear_model.LassoCV()
Lasso_model.fit(x_train, y_train)
print('Intercept: {}'.format(Lasso_model.intercept_))
print('Coefficients: {}'.format(Lasso_model.coef_))


# %%
#see which features were eliminated from the model
Lassocoeff = pd.DataFrame(Lasso_model.coef_)
Lassocoeff['Features'] = ['precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist']
Lassocoeff.head()
Lassocoeff


# %%
#now predict using the Lasso model
Lasso_predict = Lasso_model.predict(test_df[Lassocoeff['Features']])
test_df['Lasso prediction'] = Lasso_predict
test_df.head()
test_df


# %%
#with the Lasso model, let's check on how well the model predicts for the training values. 
Lassoy_train = Lasso_model.predict(x_train)
train_df['Lasso prediction'] = Lassoy_train
train_df.head()
train_df


# %%
rmse2 = sqrt(mean_squared_error(y_train, Lassoy_train)) 
rmse2


# %%
#Used R and SignifReg package to get backward selected model, criterion was AIC with alpha threshold at 1
SignifReg_predict =  1.028e+03 +  1.565e+00* test_df['precipitation'] -1.424e+00*test_df['jantemp']+ -2.131e+01*test_df['schooling'] +6.473e-03*test_df['popsqmi']+3.130e+00*test_df['nonwhitepop'] -7.589e-01*test_df['HCpol'] +  1.641e+00 * test_df['NOpol'] + 1.255e+00*test_df['atmosmoist']


# %%
test_df['SignifReg prediction'] = SignifReg_predict
test_df.head()
test_df


# %%
SignifReg_predict_train = 1.028e+03 +  1.565e+00* train_df['precipitation'] -1.424e+00*train_df['jantemp']+ -2.131e+01*train_df['schooling'] +6.473e-03*train_df['popsqmi']+3.130e+00*train_df['nonwhitepop'] -7.589e-01*train_df['HCpol'] +  1.641e+00 * train_df['NOpol'] + 1.255e+00*train_df['atmosmoist']
train_df['SignifReg prediction'] = SignifReg_predict_train
train_df.head()
train_df


# %%
rmse3 = sqrt(mean_squared_error(y_train, SignifReg_predict_train)) 
rmse3


# %%
test_df


# %%
import openpyxl
test_df.to_excel(r'E:\\Spring 2021\\Classes\\540\\Project\\Project540\\testpredictions.xlsx', index = False)


# %%



