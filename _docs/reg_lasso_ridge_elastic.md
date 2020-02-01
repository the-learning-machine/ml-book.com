---
title: Lasso, Ridge & Elastic Net
subtitle: This document will cover something cool.
author: victor
tags: [featured]
permalink:
---

# 1. Introduction to Lasso Regularization Term (L1)

LASSO - Least Absolute Shrinkage and Selection Operator - was first formulated by Robert Tibshirani in 1996. It is a powerful method that performs two main tasks: regularization and feature selection.

Let's look at the example of lasso regularization with linear models, where OLS method is used with its regularization term.

<p align="center">
    <img src="/uploads/doc/regression/lasso_1.png" height="400" width="400">
</p>

The LASSO method puts a constraint on the sum of the absolute values of the model parameters, the sum has to be less than a fixed value (upper bound, or $$t$$):

$$
\sum^{k}_{j=1} |\beta_j|<t
$$

--where t is the upper bound for the sum of the coefficients.

In order to do so, the method applies a shrinking (regularization) process where it penalizes the coefficients of the regression variables shrinking some of them to zero. During features selection process the variables that still have a non-zero coefficient after the shrinking process are selected to be part of the model. The goal of this process is to minimize the prediction error.

# 2. Parameter alpha ($$ \alpha $$)
In practice, the tuning parameter α that controls the strength of the penalty assumes great importance. Indeed, when α is sufficiently large, coefficients are forced to be exactly equal to zero. This way, dimensionality can be reduced. The larger the parameter α, the more the number of coefficients are shrunk to zero. On the other hand, if α = 0, we have just an OLS (Ordinary Least Squares) regression.


# 3. Advantages

There are many advantages of using the LASSO method.
- First of all, it can provide a very good prediction accuracy, because shrinking and removing the coefficients can reduce variance without a substantial increase of the bias, this is especially useful when you have a small number of observation and a large number of features. In terms of the tuning parameter α we know that bias increases and variance decreases when α increases, indeed a trade-off between bias and variance has to be found.
- Moreover, the LASSO helps to increase the model interpretability by eliminating irrelevant variables that are not associated with the response variable, this way also overfitting is reduced. This is the point where we are more interested in because in this paper the focus is on the feature selection task.

# 4. Introduction to Lasso Regression

Lasso with linear models is called Lasso Regression. It is the model that describes the relationship between response variable Y and explanatory variables X. In the case of one explanatory variable, Lasso Regression is called Simple Lasso Regression while the case with two or more explanatory variables is called Multiple Lasso Regression. 

Lasso Regression holds all the assumptions of the Linear Regression, such as: 
- The response variable is normally distributed
- There is a linear relationship between the response variable and the explanatory variables
- The random errors are normally distributed, have constant (equal) variances at any point in X, and are independent

To read more about Linear Regression assumptions, go to [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression).

# 5. The Model

The LASSO minimizes the sum of squared errors, with an upper bound on the sum of the absolute values of the model parameters. The lasso estimate is defined by the solution to the L1 optimization problem:






