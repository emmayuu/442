install.packages("glmnet")
install.packages("pls")
install.packages("MASS")
library(glmnet)
library(pls)
library(MASS)

# Create simulated data 

set.seed(1234) # set seed to generate the same data every time you run the code; for different repetition, comment out this line
n = 100 									# sample size
p = 120										# p > n => Big-p Data Problem
pp = 10 									# true variables 
beta1=rbind(1,1,1,1,1) 						# 5 coefficients for first five vars as a 5 x 1 vector
beta2=rbind(1,1,1,1,1)						# 5 coeffs for for second five vars as a 5 x 1 vector
beta0 = t(t(rep(0,1,110)))					# 110 insignificant betas as 110 x 1 vector

x=matrix(runif(n*p, min=0, max=10), n, p)	# randon uniform variates from U(0, 10) arranged in matrix n x p
z1=x[,1:5] %*% beta1						# first simulated factor z1 = sum of x1 through x5
z2=x[,6:10] %*% beta2						# second factor z2
yy= z1 + 10*sqrt(z2) 						# true dependent variable
y = yy + x[,11:120] %*% beta0 + rnorm(n)	# observed dependent variable with 100 regressors insignificant vars but we don't know that ex ante

# Partition training and test data sets

m = 0.8*n									# 80% training size ==> 20% holdout sample
ts = sample(1:n,m)							# random draw of 80 out of 100 rows
x.train = x[ts,]							
y.train = y[ts]

x.test = x[-ts,]
y.test = y[-ts]

####### Lasso Regression

out.lasso = glmnet(x.train, y.train, alpha = 1)     # fits lasso becasue alpha = 1 vanishes the quadratic penalty
summary(out.lasso)
# Coeffcients plots

plot(out.lasso, xvar="lambda", label=TRUE)	      # plots estimates vs log(lambda) values. See how *all* estimates tend to zero as lambda increases. Why? Exam Q	
title("Lasso Coefficients Plot",line=2.5)

# Extract LASSO estimates for specific lambda values

est_1_lasso = coef(out.lasso, s = 0.01)			  	  # estimates at lambda = 0.01 => many vars deemed significant, yet their magnitudes differ b/w sig and nonsig vars
est_2_lasso = coef(out.lasso, s = 0.5)					  # estimates when lambda = 0.5 ==> separation b/w sig an insignifcant vars improves substantially


# Optimal Lambda value?  Select by using n-fold cross-validation 

cvlasso=cv.glmnet(x.train, y.train, type.measure="mse", nfolds = 10)    # 10-fold cross-validation
plot(cvlasso, main = " Select Best Lambda")								# plot of MSE vs Lambda
lam_est_lasso = cvlasso$lambda.min											# best lambda --> one that minimizes mse
lasso_est = coef(out.lasso, s = lam_est_lasso)								# best parameter estimates 


# Prediction Using Test Sample Data

yhat_lasso = predict(cvlasso, s = lam_est_lasso, newx=x.test)					# x.test provides data from holdout sample
sse.test_lasso = sum((y.test - yhat_lasso)^2)									# sum of square errors in holdout sample
sst.test_lasso = sum((y.test-mean(y.test))^2)								# total sum of squares at ybar in holdout sample
r2_lasso = 1-sse.test_lasso/sst.test_lasso											# R square = 1 - sum of squares errors with the model/sum of squares errors w/o the model (ie just ybar) 


####### Ridge Regression
out.Ridge_Regression = glmnet(x.train, y.train, alpha = 0)     # fits lasso becasue alpha = 1 vanishes the quadratic penalty
summary(out.Ridge_Regression)
# Coeffcients plots

plot(out.Ridge_Regression, xvar="lambda", label=TRUE)	      # plots estimates vs log(lambda) values. See how *all* estimates tend to zero as lambda increases. Why? Exam Q	
title("Ridge Regression Coefficients Plot",line=2.5)

# Extract Ridge Regression estimates for specific lambda values

est_1_RR = coef(out.Ridge_Regression, s = 0.01)			  	  # estimates at lambda = 0.01 => many vars deemed significant, yet their magnitudes differ b/w sig and nonsig vars
est_2_RR = coef(out.Ridge_Regression, s = 0.5)					  # estimates when lambda = 0.5 ==> separation b/w sig an insignifcant vars improves substantially


# Optimal Lambda value?  Select by using n-fold cross-validation 

cv_Ridge_Regression=cv.glmnet(x.train, y.train, type.measure="mse", nfolds = 10)    # 10-fold cross-validation
plot(cv_Ridge_Regression, main = " Select Best Lambda")								# plot of MSE vs Lambda
lam_est_RR = cv_Ridge_Regression$lambda.min											# best lambda --> one that minimizes mse
Ridge_Regression_est = coef(cv_Ridge_Regression, s = lam_est_RR)								# best parameter estimates 


# Prediction Using Test Sample Data

yhat_RR = predict(cv_Ridge_Regression, s = lam_est_RR, newx=x.test)					# x.test provides data from holdout sample
sse.test_RR = sum((y.test - yhat_RR)^2)									# sum of square errors in holdout sample
sst.test_RR = sum((y.test-mean(y.test))^2)								# total sum of squares at ybar in holdout sample
r2_RR = 1-sse.test_RR/sst.test_RR	

#######Elastic Net
out.EN = glmnet(x.train, y.train, alpha = 0.5)     # fits lasso becasue alpha = 1 vanishes the quadratic penalty
summary(out.EN)
#alpha:tabulate mse at lambda.min for various values of alpha from 0.1 to 0.9 to find the best alpha value
# Coeffcients plots

plot(out.EN, xvar="lambda", label=TRUE)	      # plots estimates vs log(lambda) values. See how *all* estimates tend to zero as lambda increases. Why? Exam Q	
title("Elastic Net Coefficients Plot",line=2.5)

# Extract LASSO estimates for specific lambda values

est_1_EN = coef(out.EN, s = 0.01)			  	  # estimates at lambda = 0.01 => many vars deemed significant, yet their magnitudes differ b/w sig and nonsig vars
est_2_EN = coef(out.EN, s = 0.5)					  # estimates when lambda = 0.5 ==> separation b/w sig an insignifcant vars improves substantially


# Optimal Lambda value?  Select by using n-fold cross-validation 

cvEN=cv.glmnet(x.train, y.train, type.measure="mse", nfolds = 10)    # 10-fold cross-validation
plot(cvEN, main = " Select Best Lambda")								# plot of MSE vs Lambda
lam_est_EN = cvEN$lambda.min											# best lambda --> one that minimizes mse
EN_est = coef(out.EN, s = lam_est_EN)								# best parameter estimates 


# Prediction Using Test Sample Data

yhat_EN = predict(cvEN, s = lam_est_EN, newx=x.test)					# x.test provides data from holdout sample
sse.test_EN = sum((y.test - yhat_EN)^2)									# sum of square errors in holdout sample
sst.test_EN = sum((y.test-mean(y.test))^2)								# total sum of squares at ybar in holdout sample
r2_EN = 1-sse.test_EN/sst.test_EN											# R square = 1 - sum of squares errors with the model/sum of squares errors w/o the model (ie just ybar) 

