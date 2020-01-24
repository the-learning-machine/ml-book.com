---
title: Logistic Regression
subtitle: Logistic regression and how it works.
author: andrew
tags: [naive]
permalink:
---
# 1. Introduction

Logistic Regression despite the "regression" term in its name is used in **classification** problems when the dependent (target) variable has two possible outcomes. However, this model can be extended to tackle multiclass classification problems, and we will discuss it at the end of this article. 

# 2. Key Terms

$$Odds$$ are used in Logistic Regression algorithm to model probabilities:

$$
odds(p)=\frac{p}{1-p}=\frac{Prob(Class\ 1)}{Prob(Class\ 2)}=\frac{Prob(True)}{Prob(False)}=\frac{Prob("Yes")}{Prob("No")}\quad(1)
$$

<p align="center">
    <img src="/uploads/doc/classification/lr_odd.png" height="400" width="400">
</p>

As you can see from formula (1), $$odds(p) \in [0 ; \infty]$$ given that $$p \in [0;1]$$.
However, we want our model to take a real value number from $$[-\infty ; \infty]$$ (as our features can have any values), and output a soft number in a range [0;1] to describe a probability. Logistic function (also called Sigmoid) possesses all of these traits. It can be derived as an inverse of a log-odds function which is also called **logit**.

$$
logit(p)=log(odds(p))=log(\frac{p}{1-p})\quad(2)
$$

<img src="/uploads/doc/classification/lr_logit.png" align="center" height="400" width="400">

We can achieve the required properties by reflecting the logit function about the line $$y=x$$. This transformation can be performed by calculating the inverse of expression (2) which is called a **logistic function**:

$$
logistic(y)=logit(p)^{-1}
$$


In order to calculate that we should solve the equation: 

$$
logit(p)=y \to log(\frac{p}{1-p})=y \to \frac{p}{1-p}=e^y \to p=e^y(1-p) \to p(e^y+1)=e^y \\
p=\frac{e^y}{e^y+1}=\frac{1}{1+e^{-y}}
$$


Thus, the expression for logistic function (sigmoid function) is the following:

$$
logistic(y)=\frac{1}{1+e^{-y}}\quad(3)
$$

<img src="/uploads/doc/classification/lr_logistic.png" align="middle" height="400" width="400">

# Model Training

Logistic Regression represents logit function as a linear combination of predictors plus the intercept:

$$
logit(p)=\theta_0+\theta_1X_1+\theta_2X_2+...+\theta_k X_k,\quad(4)
$$

where 
- $$X_i$$ is the value of $$i^{th}$$ predictor
- $$\theta_i$$ is the generated coefficient

Coefficients $$\theta_i$$ indicate the effect of a one-unit change in the predictor variable on the log odds of "success"

As our train data contains more than one observation, we will denote $$x$$ as a column vector of the predictors' values for the particular observation (we will also add 1 as its first element to account for an intercept term) and $$\theta$$ as a column vector of coefficients $$\theta_0...\theta_k$$:

$$
x=
\left[\begin{array}{ccc}
1\\X_{1}\\X_{2}\\...\\X_{k}
\end{array}\right];
\qquad
\theta=
\left[\begin{array}{ccc}
\theta_0\\\theta_1\\\theta_2\\...\\\theta_k
\end{array}\right]
$$


Using this notation we can rewrite the expression (4) as follows:

$$logit(p)=\theta^Tx\quad(5)$$


If we plug in $$y=\theta^Tx$$ into formula (3), we will get an expression for the probability of a random variable Y (that represents the predicted output) being 0 or 1 given experimental data $$x$$ and model parameters $$\theta$$:

$$
Pr(Y=1 | x, \theta)=\frac{1}{1+e^{-\theta^Tx}}\quad(6)
$$


As we are dealing with two class problem, the probability $$Pr(Y=0 \| x, \theta)$$ can be expressed as follows:

$$
Pr(Y=0 | x, \theta)=1-Pr(Y=1 | x, \theta)\quad(7)
$$

We can combine probabilities used in expressions (6) and (7) into one formula:

$$
Pr(Y | x, \theta)=Pr(Y=1 | x, \theta)^{Y}(1-Pr(Y=1 | x, \theta))^{1-Y}\quad(8)
$$

One can notice that:

$$
Pr(Y | x, \theta) \to Pr(Y=1 | x, \theta) \mathrm{\ given\ Y=1},\ \mathrm{and} \\
Pr(Y | x, \theta) \to 1 - Pr(Y=1 | x, \theta) = Pr(Y=0 | x, \theta) \mathrm{\ given\ Y=0}.
$$


Our goal is to determine the coefficients $$\theta=\theta_0$$...$$\theta_k\$$ from formula (4). The intuition here is that for any given train observation we want these coefficients to maximize the probability of observing a correct label. This sentence can be converted to the following formula (assuming train data is independently distributed):

$$

L(\theta | x) = Pr(Y | x,\theta) \to max, \\
\mathrm{where}\  Pr(Y | x,\theta) = \prod_{i=1}^{n}Pr(y_i | x_i,\theta)=\prod_{i=1}^{n}Pr(y_i=1 | x_i, \theta)^{y_i}(1-Pr(y_i=1 | x_i, \theta))^{1-y_i}
$$

This expression can be maximized through various optimization techniques such as Newton-Raphson algorithm or a gradient descent (which is usually applied to log-likelihood).

# Making Predictions

Now as we have the vector of model parameters $$\theta$$ we can calculate the predicted value of the logit function for any new observation $$x$$ (we will use hat symbol for predicted values):

$$
logit(p)=\hat{y}=\theta^Tx
$$

Then we plug this value into logistic function in order to determine the probability of the data belonging to Class 1 (True, "Yes", etc):

$$
\hat{p}=\hat{p(Class\ 1)}=logistic(\hat{y})=\frac{1}{1+e^{-\hat{y}}}
$$

The last step is to set up a threshold T \(\in\) [0;1] that will be used in order to make a prediction:

$$
\mathrm{Model\ Output} = 
    \begin{cases}
        \mathrm{Class\ 1\ \ if}  & \hat{p}\ge T\\
        \mathrm{Class\ 2\ \ if}  & \hat{p}<T
    \end{cases}
$$

By default the threshold is set up to 0.5, but you can adjust it based on your needs (usually based on the True Positive Rate and False Positive Rate trade-off).

<img src="/uploads/doc/classification/lr_pic1.png" align="middle" height="400" width="400">

# 5. Regularization

Regularization means making the model less complex which can allow it to generalize better (i.e. avoid overfitting) and perform better on a new data. 

As was mentioned above, the coefficients of logistic regression are usually fitted by maximizing the log-likelihood. As many optimization techniques are aimed at finding the minimum of a function we can redefine our goal as minimizing the negative log-likelihood:

$$
\hat\theta=\min\limits_{\theta}[-log(L(\theta | x))]
$$

We can penalize the model of having coefficients that are far from zero by adding a regularization term $$R(\theta)$$ multiplied by parameter $$\lambda$$ which is called regularization strength:

$$
\hat\theta=\min\limits_{\theta}[-log(L(\theta | x))+\lambda R(\theta)]
$$

The two most popular regularizations are L1 and L2:

$$
L1: R(\theta)=\sum_{i=0}^{K}|\theta_i|
L2: R(\theta)=\frac{1}{2}\sum_{i=0}^{K}\theta_i^2
$$

The factor $$\frac{1}{2}$$ in L2 regularization is used to simplify the derivative calculations. Through $$\lambda$$ we can control the impact of the regularization term. Higher values of $$\lambda$$ lead to smaller coefficients (less regularization), but too high values can lead to underfitting.

In scikit-learn package L2 regularization is used by default. Instead of regularization strength $$\lambda$$, its inverse is used: the C parameter (the default is C=1.0). Similarly to $$\lambda$$: smaller values of C leads to smaller coefficients, but too high values can lead to underfitting. 

It is important to normalize the data before performing regularized logistic regression to ensure that the regularization term $$\lambda$$ affects the coefficients in a similar manner.


# Logistic Regression For Multinomial Problems

Logistic regression can be generalized to handle problems with more than two possible outcomes. The most popular approach is called "One-vs-Rest" logistic regression where we split our multinomial problem with M classes into M binary classification problems (see Figure 5). 

<img src="/uploads/doc/classification/lr_1vsall.png" align="middle" height="400" width="400"> 

In this case we generate different coefficients $$\theta$$ for each binary classification problem (basically we train M separate Logistic Regression models). When we have to classify a new observation, we calculate the probabilities of the data belonging to each class (which are the outputs of our models) and select the class that has the highest probability.
