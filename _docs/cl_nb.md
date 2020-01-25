---
title: Naive Bayes Classifier
subtitle: This document will cover something cool.
author: victor
tags: [featured]
permalink:
---

# 1. Introduction

Naive Bayes is so ‘naive’ because it assumes that all of the features in a data set are equally important and independent. These assumptions are rarely true in real world scenario, however Naive Bayes algorithm sometimes performs surprisingly well. This is the supervised learning algorithm used for both classification and regression. Its advantage is that it requires very small computational power and as a result works fast even with large data.

# 2. Key Terms 

- Prior probability is the proportion of dependent variable (target) in the data set.
- Likelihood is the probability of particular classification a given observation in presence of some other variable.
- Marginal likelihood is the proportion of independent variable (predictor) in the data set.

These terms might not be clear to you. Let's dive into an example that shows what exactly Naive Bayes does, with an indication of these terms. 

# 3. Example with Explanation

Below I have a training data set of weather and corresponding target variable ‘Play’ (suggesting possibilities of playing). Now, we need to classify whether players will play or not based on weather condition. Let’s follow the below steps to perform Naive Bayes:

- Step 1: Convert the data set into a frequency table (also called contingency table)
- Step 2: Create Likelihood table.
- Step 3: Use Naive Bayesian equation to calculate the posterior probability for each class. The class with the highest posterior probability is the outcome of prediction.

## 3.1. Step 1 and Step 2 

Let's go over the first two steps. These steps will also help us understand prior probability, likelihood and marginal likelihood.

<p align="center">
    <img src="/uploads/doc/classification/nb_table1.png" height="800" width="800">
</p>

The terms Likelihood, Marginal Likelihood, and Prior Probability (or Class Prior Probability, as it is related to classes "Yes" or "No") that were mentioned above are shown below

<p align="center">
    <img src="/uploads/doc/classification/nb_table2.png" height="800" width="800">
</p>

So, we can now see that:
- Likelihood = P (Feature $$\|$$ Class)
- Marginal Likelihood = P (Feature)
- Prior Likelihood = P (Class)

Likelihood is just a probability of a feature within a class. For example, if we want to calculate P(Sunny $$\|$$ "Yes"), where Sunny is a feature, and "Yes" is a class, we will count all "Yes"es, or all times we went to Play, (and ignore "No"s) when we had "Sunny" weather, divided by the overall observed days in our data set. 

Marginal Likelihood is a probability of a feature. For example, if we want to calculate P(Sunny), we will count all the Sunny days divided by the overall observed days in our data set.

Prior Likelihood or Class Prior Probability is a probability of a class. For example, if we want to calculate P("No"), we will count all the "No"s, or, the days we did not go to Play, divided by the overall observed days in our data set.

Posterior probability is the revised probability of an event occurring after taking into consideration new information. It will be discussed in more details later in this article.

## 3.2. Step 3

Use Bayes' Formula to calculate the posterior probability for each class. The class with the highest posterior probability is the outcome of prediction.

<p align="center">
    <img src="/uploads/doc/classification/nb_table3.png" height="400" width="400">
</p>

In formula above ’c’ denotes class and ’x’ denotes features. Next, let’s look at P(x). As you can see, the denominator contains the only term that is a function of the data (features) - it is not a function of the class we are currently looking at. Thus, it will be the same for all the classes. Traditionally in Naive Bayes Classification, we drop this denominator as it does not impact the final outcome of the classifier in order to make the prediction:

$$
P(x &#124 x) -> P(x \ c)P(c)\quad(1)
$$

To make it more interesting, let’s assume we have an the additional feature - Wind:

<p align="center">
    <img src="/uploads/doc/classification/nb_table4.png" height="400" width="400">
</p>

Let’s assume we want to predict the class for the data with the following features:

$$
Wind = Moderate \\
Weather = Sunny
$$

In order to make a prediction we need to compare posterior probabilities for each class after observing the input data. For this purpose we will use the expression (1). Do not forget, that Naive Bayes assumes independence of features. In order not to inflate our formulas we will use the following notation: ’X1’ for ’Weather’, ’X2’ for ’Wind’ and ’C’ for ’Class’

First, we estimate the probability for going to Play (i.e. the class = "Yes") for Wind = Moderate, Weather = Sunny:

TBD....