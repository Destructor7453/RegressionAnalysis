
# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import datasets, metrics
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import seaborn as seabornInstance
from math import sqrt


# %%
df = pd.read_csv(r'E:\\Spring 2021\\Classes\\540\\Project\\Project540\\Life Expectancy Data.csv')
df


# %%
import seaborn as sns

df_developing = df[df['Status'] == 'Developing']
df_developing


# %%
df_developed = df[df['Status'] =='Developed']
df_developed

# %% [markdown]
# There are some NaN values. For this data, I will NaN values with the mean of the columns. 

# %%
#developing
df_developing.mean()
df_developing = df_developing.fillna(df_developing.mean())

#developed
df_developed.mean()
df_developed = df_developed.fillna(df_developed.mean())


# %%
#rename columns for easier usage in code
df_developing.columns = ['Country', 'Year', 'Status', 'LifeExp', 'AdultMort', 'InfMort', 'Alcohol', 'pcexp', 'HepB', 'Measles', 'Bmi', 'und5Mort', 'Polio', 'TotExp', 'Diptheria', 'Hiv', 'Gdp', 'pop', 'MinThin', 'ChildThin', 'Income', 'Schooling']
df_developed.columns = ['Country', 'Year', 'Status', 'LifeExp', 'AdultMort', 'InfMort', 'Alcohol', 'pcexp', 'HepB', 'Measles', 'Bmi', 'und5Mort', 'Polio', 'TotExp', 'Diptheria', 'Hiv', 'Gdp', 'pop', 'MinThin', 'ChildThin', 'Income', 'Schooling']


# %%
df_developed


# %%
df_developing


# %%
df_developed = df_developed.drop(['Country', 'Year', 'Hiv'], axis=1)
df_developed


# %%
df_developing = df_developing.drop(['Country', 'Year', 'Hiv'], axis=1)
df_developing


# %%
ind_var = ['AdultMort', 'InfMort', 'Alcohol', 'pcexp', 'HepB', 'Measles', 'Bmi', 'und5Mort', 'Polio', 'TotExp', 'Diptheria', 'Gdp', 'pop', 'MinThin', 'ChildThin', 'Income', 'Schooling']


# %%
sns.pairplot(df_developing[ind_var], diag_kind="kde")


# %%
sns.pairplot(df_developed[ind_var], diag_kind="kde")


# %%
usecols = ['LifeExp', 'AdultMort', 'InfMort', 'Alcohol', 'pcexp', 'HepB', 'Measles', 'Bmi', 'und5Mort', 'Polio', 'TotExp', 'Diptheria', 'Gdp', 'pop', 'MinThin', 'ChildThin', 'Income', 'Schooling']
sns.pairplot(df_developed[usecols], diag_kind="kde")


# %%
mask = np.zeros_like(df_developed[usecols].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)
sns.heatmap(df_developed[usecols].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="Blues", linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});


# %%
mask = np.zeros_like(df_developing[usecols].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)
sns.heatmap(df_developing[usecols].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="Blues", linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});


# %%
Xdevd = df_developed[ind_var]
ydevd = df_developed['LifeExp']

from sklearn.model_selection import train_test_split
xtrain_dd, xtest_dd, ytrain_dd, ytest_dd = train_test_split(Xdevd, ydevd, test_size=0.67)


# %%
Xdevg = df_developing[ind_var]
Ydevg = df_developing['LifeExp']

xtrain_dg, xtest_dg, ytrain_dg, ytest_dg = train_test_split(Xdevg, Ydevg, test_size = 0.67)


# %%
#least-squares estimation for developed countries
import statsmodels.api as sm
developedlr = sm.OLS(ytrain_dd, xtrain_dd)
developedlr= developedlr.fit()
developedlr.summary()


# %%
#least-squares estimation for developing countries
import statsmodels.api as sm
developinglr = sm.OLS(ytrain_dg, xtrain_dg)
developinglr= developinglr.fit()
developinglr.summary()


# %%
#predicting using OLS - developed
yhat_developed = developedlr.predict(xtest_dd)
rmse_developed_ols = sqrt(mean_squared_error(ytest_dd, yhat_developed)) 
rmse_developed_ols


# %%
#create residual vs. predictor plot for 'AdultMort'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developedlr, 'AdultMort', fig=fig)


# %%
#create residual vs. predictor plot for 'alcohol'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developedlr, 'Alcohol', fig=fig)


# %%
#create residual vs. predictor plot for 'assists'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developedlr, 'Income', fig=fig)


# %%
#create residual vs. predictor plot for 'assists'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developedlr, 'Schooling', fig=fig)


# %%
#predicting using OLS - developed
yhat_developing = developinglr.predict(xtest_dg)
rmse_developing_ols = sqrt(mean_squared_error(ytest_dg, yhat_developing)) 
rmse_developing_ols


# %%
#create residual vs. predictor plot for 'AdultMort'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developinglr, 'AdultMort', fig=fig)


# %%
#create residual vs. predictor plot for 'AdultMort'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developinglr, 'InfMort', fig=fig)


# %%
#create residual vs. predictor plot for 'AdultMort'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developinglr, 'HepB', fig=fig)


# %%
#create residual vs. predictor plot for 'AdultMort'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developinglr, 'Measles', fig=fig)


# %%
#create residual vs. predictor plot for 'AdultMort'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developinglr, 'Bmi', fig=fig)


# %%
#create residual vs. predictor plot for 'AdultMort'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developinglr, 'und5Mort', fig=fig)


# %%
#create residual vs. predictor plot for 'AdultMort'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developinglr, 'Polio', fig=fig)


# %%
#create residual vs. predictor plot for 'AdultMort'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developinglr, 'TotExp', fig=fig)


# %%
#create residual vs. predictor plot for 'AdultMort'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developinglr, 'Diptheria', fig=fig)


# %%
#create residual vs. predictor plot for 'AdultMort'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developinglr, 'MinThin', fig=fig)


# %%
#create residual vs. predictor plot for 'AdultMort'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developinglr, 'ChildThin', fig=fig)


# %%
#create residual vs. predictor plot for 'AdultMort'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developinglr, 'Income', fig=fig)


# %%
#create residual vs. predictor plot for 'AdultMort'
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_regress_exog(developinglr, 'Schooling', fig=fig)


# %%
#Ridge regression for developed
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
ridge_model_developed = RidgeCV(alphas=np.arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
ridge_model_developed.fit(xtrain_dd, ytrain_dd)


# %%
Ridgecoeff_developed = pd.DataFrame(ridge_model_developed.coef_)
Ridgecoeff_developed['Features'] = ind_var
Ridgecoeff_developed.head()
Ridgecoeff_developed


# %%
ridge_y_developed = ridge_model_developed.predict(xtest_dd)
ridge_y_developed


# %%
rmse_ridge_developed = sqrt(mean_squared_error(ytest_dd, ridge_y_developed)) 
rmse_ridge_developed


# %%
from sklearn.metrics import r2_score
print(r2_score(ytest_dd, ridge_y_developed))


# %%
print(ridge_model_developed.alpha_)


# %%
#Ridge regression for developing
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
ridge_model_developing = RidgeCV(alphas=np.arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
ridge_model_developing.fit(xtrain_dg, ytrain_dg)


# %%
print(ridge_model_developing.alpha_)


# %%
Ridgecoeff_developing = pd.DataFrame(ridge_model_developing.coef_)
Ridgecoeff_developing['Features'] = ind_var
Ridgecoeff_developing.head()
Ridgecoeff_developing


# %%
ridge


# %%
ridge_y_developing = ridge_model_developing.predict(xtest_dg)
ridge_y_developing


# %%
rmse_ridge_developing = sqrt(mean_squared_error(ytest_dg, ridge_y_developing)) 
rmse_ridge_developing


# %%
print(r2_score(ytest_dg, ridge_y_developing))


# %%
#Lasso regression for developed
from sklearn import linear_model
Lasso_model_developed = linear_model.LassoCV()
Lasso_model_developed.fit(xtrain_dd, ytrain_dd)


# %%
Lassocoeff_developed = pd.DataFrame(Lasso_model_developed.coef_)
Lassocoeff_developed['Features'] = ind_var
Lassocoeff_developed.head()
Lassocoeff_developed


# %%
Lasso_y_developed = Lasso_model_developed.predict(xtest_dd)
Lasso_y_developed


# %%
rmse_Lasso_developed = sqrt(mean_squared_error(ytest_dd, Lasso_y_developed)) 
rmse_Lasso_developed


# %%
print(r2_score(ytest_dd, Lasso_y_developed))


# %%
Lasso_model_developing = linear_model.LassoCV()
Lasso_model_developing.fit(xtrain_dg, ytrain_dg)


# %%
Lassocoeff_developing = pd.DataFrame(Lasso_model_developing.coef_)
Lassocoeff_developing['Features'] = ind_var
Lassocoeff_developing.head()
Lassocoeff_developing


# %%
Lasso_y_developing = Lasso_model_developing.predict(xtest_dg)
Lasso_y_developing


# %%
rmse_Lasso_developing = sqrt(mean_squared_error(ytest_dg, Lasso_y_developing)) 
rmse_Lasso_developing


# %%
print(r2_score(ytest_dg, Lasso_y_developing))


# %%
#randomForest for developed
from sklearn.ensemble import RandomForestRegressor
developed_regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
developed_regressor.fit(xtrain_dd, ytrain_dd)  


# %%
rf_y_developed = developed_regressor.predict(xtest_dd)
rf_y_developed


# %%
rmse_RF_developed = sqrt(mean_squared_error(ytest_dd, rf_y_developed)) 
rmse_RF_developed


# %%
print(r2_score(ytest_dd, rf_y_developed))


# %%
developing_regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
developing_regressor.fit(xtrain_dg, ytrain_dg)  


# %%
rf_y_developing = developing_regressor.predict(xtest_dg)
rf_y_developing


# %%
rmse_RF_developing= sqrt(mean_squared_error(ytest_dg, rf_y_developing)) 
rmse_RF_developing


# %%
print(r2_score(ytest_dg, rf_y_developing))


# %%



