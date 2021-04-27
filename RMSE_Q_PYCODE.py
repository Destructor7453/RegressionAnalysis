#RMSE ChallengeQ
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
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import seaborn as seabornInstance
from math import sqrt

# %% [markdown]
# For the first part, I am fitting using Ridge regression. From a previous selection process, detailed later, there is multicolinearity present in the predictors. Thus we have to do multiple regression selection processes to get the best fit. 
# 
# Reference: https://www.statology.org/ridge-regression-in-python/

# %%
#load in the train and test datasets
df = pd.read_csv("E:\\Spring 2021\\Classes\\540\\Project\\Project540\\TrainingData.txt", header=None, sep='\s+')
test_df = pd.read_csv("E:\\Spring 2021\\Classes\\540\\Project\\Project540\\TestData.txt", header=None, sep="\s+")
#load headers for the data sets
df.columns= ['index','precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'kitchen', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist', 'deathrate' ]
test_df.columns= ['index','precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'kitchen', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist']
#eliminating the values that have NaN as an entry
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float64)
test_df = test_df[~test_df.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float64)


# %%
independent_var = ['precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist']
#These are the independent variables, or regressors of the training data. 


# %%
x_train = df[independent_var]
y_train = df['deathrate']

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model = RidgeCV(alphas=np.arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
model.fit(x_train, y_train)


# %%
print(model.alpha_)


# %%
ridge_y = model.predict(test_df[independent_var])
ridge_y

# %% [markdown]
# This next part is the code I used to detect multicolinearity in the data, as well as selecting the model using backwards selection. 

# %%
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor


# %%
ytrain, xtrain = dmatrices('deathrate ~ precipitation+jantemp+jultemp+pop65+housemem+schooling+kitchen+popsqmi+nonwhitepop+office+less3000+HCpol+NOpol+SO2pol+atmosmoist', data=df, return_type='dataframe')


# %%
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(xtrain.values, i) for i in range(xtrain.shape[1])]
vif['variable'] = xtrain.columns


# %%
vif

# %% [markdown]
# Almost all VIF values are greater than 5, however, the variables "nonwhitepop", "less300", "HCpol", "NOpol" have the highest VIF. WE can look at a covariance heatmap:

# %%
usecols = ['precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'kitchen', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist']
mask = np.zeros_like(df[usecols].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)
seabornInstance.heatmap(df[usecols].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="Blues", linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});

# %% [markdown]
# as we can see the two biggest covariances occur with (HCpol, NOpol) and (nonwhitepop, less3000). Thus we can remove these from our model. 

# %%
ytrain1, xtrain1 = dmatrices('deathrate ~ precipitation+jantemp+jultemp+pop65+housemem+schooling+kitchen+popsqmi+office+SO2pol+atmosmoist', data=df, return_type='dataframe')
vif1 = pd.DataFrame()
vif1['VIF'] = [variance_inflation_factor(xtrain1.values, i) for i in range(xtrain1.shape[1])]
vif1['variable'] = xtrain1.columns


# %%
vif1

# %% [markdown]
# Now almost all the predictors have VIFs less than 5, where 'housemem' now has the largest. We can look at a model without the predicitor: 

# %%
ytrain2, xtrain2 = dmatrices('deathrate ~ precipitation+jantemp+jultemp+pop65+schooling+kitchen+popsqmi+office+SO2pol+atmosmoist', data=df, return_type='dataframe')
vif2 = pd.DataFrame()
vif2['VIF'] = [variance_inflation_factor(xtrain2.values, i) for i in range(xtrain2.shape[1])]
vif2['variable'] = xtrain2.columns
vif2

# %% [markdown]
# last variable that has a VIF value over 5 is 'schooling'. 

# %%
ytrain3, xtrain3 = dmatrices('deathrate ~ precipitation+jantemp+jultemp+pop65+kitchen+popsqmi+office+SO2pol+atmosmoist', data=df, return_type='dataframe')
vif3 = pd.DataFrame()
vif3['VIF'] = [variance_inflation_factor(xtrain3.values, i) for i in range(xtrain3.shape[1])]
vif3['variable'] = xtrain3.columns
vif3

# %% [markdown]
# Now all the variables have values less than 5. now using these variables to create the model:

# %%
red_ind_var = ['precipitation', 'jantemp', 'jultemp', 'pop65', 'popsqmi', 'office', 'SO2pol', 'atmosmoist']
red_xtrain = df[red_ind_var]
red_ytrain = df['deathrate']

deathmodel = LinearRegression()
deathmodel.fit(red_xtrain, red_ytrain)

print('Intercept: {}'.format(deathmodel.intercept_))
print('Coefficients: {}'.format(deathmodel.coef_))


# %%
x_test = test_df[red_ind_var]
y_test= deathmodel.predict(x_test)
y_test


# %%
#comparing the ridge regression model versus the model 
rmse = np.sqrt(mean_squared_error(y_test,ridge_y))
rmse

# %% [markdown]
# The RMSE is less than 36 using the full model and removing the rows with NaN values. 
# %% [markdown]
# Now, instead of eliminating the NaN values, we'll replace them with the mean of the column they belong to. 

# %%
#reload dataset to be original data set. 
df2 = pd.read_csv("E:\\Spring 2021\\Classes\\540\\Project\\Project540\\TrainingData.txt", header=None, sep='\s+')
test_df2 = pd.read_csv("E:\\Spring 2021\\Classes\\540\\Project\\Project540\\TestData.txt", header=None, sep="\s+")

#load headers for the data sets
df2.columns= ['index','precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'kitchen', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist', 'deathrate' ]
test_df2.columns= ['index','precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'kitchen', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist']

#calculate means
df2.mean()
test_df2.mean()

#replace with means
df2 = df2.fillna(df2.mean())
test_df2 = test_df2.fillna(test_df2.mean())


# %%
#Ridge regression
x_train2 = df2[independent_var]
y_train2 = df2['deathrate']

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
model2 = RidgeCV(alphas=np.arange(0, 1, 0.01), cv=cv, scoring='neg_mean_absolute_error')
model2.fit(x_train2, y_train2)
print(model2.alpha_)


# %%
ridge_y2 = model2.predict(test_df2[independent_var])
ridge_y2

# %% [markdown]
# Since we know which variables have multicollinearity from the previous section, we can take a look at the VIFs to confirm that these are the same in this case.

# %%
ytrainm, xtrainm = dmatrices('deathrate ~ precipitation+jantemp+jultemp+pop65+housemem+schooling+kitchen+popsqmi+nonwhitepop+office+less3000+HCpol+NOpol+SO2pol+atmosmoist', data=df2, return_type='dataframe')
vifm = pd.DataFrame()
vifm['VIF'] = [variance_inflation_factor(xtrainm.values, i) for i in range(xtrainm.shape[1])]
vifm['variable'] = xtrainm.columns
vifm

# %% [markdown]
# In comparison, we can see that the values are a lot less than previous computed where we took out the NaN values. However, the same variables are still the largest. We'll eliminate the predictors we know are the largest. First, print out the correlation matirx. 

# %%
usecols = ['precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'kitchen', 'popsqmi', 'nonwhitepop', 'office', 'less3000', 'HCpol', 'NOpol', 'SO2pol', 'atmosmoist']
mask = np.zeros_like(df2[usecols].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)
seabornInstance.heatmap(df2[usecols].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="Blues", linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});

# %% [markdown]
# Again we see the same variables still with the largest correlation, but not as large as the computation previous. Removing: 'HCpol', 'NOpol', 'nonwhitepop', 'less3000'

# %%
ytrainm1, xtrainm1 = dmatrices('deathrate ~ precipitation+jantemp+jultemp+pop65+housemem+schooling+kitchen+popsqmi+office+SO2pol+atmosmoist', data=df2, return_type='dataframe')
vifm1 = pd.DataFrame()
vifm1['VIF'] = [variance_inflation_factor(xtrainm1.values, i) for i in range(xtrainm1.shape[1])]
vifm1['variable'] = xtrainm1.columns
vifm1

# %% [markdown]
# Unlike the previous removal, after removing the 4 variables, we can see that all the other predictors are less than 5. We can probably improve this model, but first let's take a look at the RMSE of this model in comparison to the Ridge regressed model. 

# %%
red_ind_varm = ['precipitation', 'jantemp', 'jultemp', 'pop65', 'housemem', 'schooling', 'kitchen', 'popsqmi', 'office', 'SO2pol', 'atmosmoist', 'deathrate']
red_xtrainm = df[red_ind_var]
red_ytrainm = df['deathrate']

deathmodelm1 = LinearRegression()
deathmodelm1.fit(red_xtrainm, red_ytrainm)

print('Intercept: {}'.format(deathmodelm1.intercept_))
print('Coefficients: {}'.format(deathmodelm1.coef_))


# %%
x_testm = test_df2[red_ind_var]
y_testm= deathmodelm1.predict(x_testm)
y_testm


# %%
#comparing the ridge regression model versus the model 
rmse = np.sqrt(mean_squared_error(y_testm,ridge_y2))
rmse

# %% [markdown]
# In comparison, this RMSE is smaller than the previous computed RMSE for the data with deleted NaN values. 
# 
# Thus, we can say that the method of replaced the mean into the NaN values gives us a more stable model than deleting rows with NaN values. 

# %%



