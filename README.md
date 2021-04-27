# RegressionAnalysis

This repository is for the projects done in Regression Analysis, Math 540. 

1) Challenge question: Train and test data set were given. In the test dataset, the response (y.tests) were with held, and the goal is to predict the best response that are equal or under the the threshold of 36.69098. Two regression model were ran: one, used the full model and yielded an RMSE of 36.69098; and the second, was estimated using the training dataset by removing the rows with NaN, and the predicted values using the test dataset by replacing the NaNs with the means of the column. The RMSE is 45.41023. 
  
 - In my analysis, I first ran a ridge regression to find the best fit to the training dataset. Then by calculating the VIFs and eliminating the variables with high VIFs (over      5). I ran this twice, one deleting the NaN values from the dataset, and the second, by replacing the NaN values with the mean of the column. 
