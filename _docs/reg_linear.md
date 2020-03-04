# Linear Regression
Linear regression is a necessary tool in any data professional's toolbox. It is a widely used supervised learning approach. The goal of linear regression is to describe the relationship between input variables $$x_{1}, x_{2}, ... x_{n}$$ and an output variable $$y$$. Examples of this include predicting how height relates to weight, how age relates to income, or how the amount of time someone studies relates to their test score. 

Below is the general equation for linear regression. A linear regression model will always have an output equation is this form. This method assumes that there is approximately a linear relationship between the input variables and output varible.

>$$y=\beta_{0}+\beta_{1}x_{1}+...+\beta_{n}x_{n}$$

The $$\beta_{0}$$ term is called the intercept. In terms of a simple linear model (one $$x$$ variable), this intercept allows the line to move up or down the y-axis. A model with multiple $$x$$ variables, as seen above, is called multiple linear regression. Each successive $$\beta_{i}$$ term is called a coefficient. Each coefficient describes to what extent the associated $$x$$ variable contributes to the model.
 
# When do you use linear regression?

  1. Determine whether the input variables and the output variable have a linear relationship.
  2. Determine how good the input variables describe the output variable.
  3. Forecasting a linear trend.

# Example - Salary vs. Years of Experience
This dataset contains information about salary vs. years of experience. The data was obtained from this [Kaggle dataset]. The graph below shows a clear positive relationship between salary and years of experience. Logically this makes sense, the more years you work, the more promotions you should have, resulting in a higher salary. The dataset stops around 10.5 years of experience. What if we wanted to forecast the salary of someone with 20 years of experience? That’s where the linear regression model comes in.

![Image of Linear Regression Data](https://raw.githubusercontent.com/shuzlee/reg_linear/master/Salary%20vs.%20Years%20of%20Experience.jpg)

### Creating a Linear Regression Model
In Python, use the statsmodel package to create your linear regression model. The OLS (ordinary least squares) method was used in the example below. The OLS method minimizes the sum of the squared error.
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv('Salary_Data.csv')

X = df['YearsExperience']
y = df['Salary']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
predictions = model.predict(X)
print(model.summary())

salary_predictions = model.predict(X)
```
The model summary is shown below. Important data points to look at are the R-squared value of 0.957. An R-squared value can be between 0 and 1. The closer the value is to 1, the more the variables in the model explain the trend of the output variable. However, be careful of overfitting a model just to get a high R-squared value.

![Image of OLS Model](https://raw.githubusercontent.com/shuzlee/reg_linear/master/OLSReg.JPG)

Other important values in the regression results are the “const” and the “YearsExperience” values under the ‘coef’ and the ‘P>|t|’ columns. The ‘coef’ column are the coefficients of the model, $$\beta_{0}$$ and $$\beta_{1}$$.

> $$\beta_{0}=2.579*10^4$$
> $$\beta_{1}=9449.9623$$

Now your model’s equation can be represented as:
> $$Salary=(2.579*10^4)+9449.9623*\textrm{Years of Experience}$$

This equation is plotted as the straight line in the graph below.

![Image of Linear Regression Data w/ Trendline](https://raw.githubusercontent.com/shuzlee/reg_linear/master/Salary%20vs.%20Years%20of%20Experience%20Pred.jpg)

The ‘P>|t|’ column below shows the p-value for each variable. All of the p-values are 0 which suggests they are significant in describing the output variable, salary. A p-value < 0.05 is usually significant and should be kept in the model. A p-value > 0.05 indicates that the variable is not significant in describing the output variable and should be removed from the model.

So now that we have the $$\beta_{0}$$ and $$\beta_{1}$$ values, we can create our linear regression equation to predict the salary of someone with 20 years of experience.

> $$y=\beta_{0}+\beta_{0}x_{1}$$
> $$Salary = (2.579*10^4) + 9449.9623 * \textrm{Years  of Experience}$$
> $$Salary = (2.579*10^4) + 9449.9623 * 20\textrm{ Years}$$
> $$Salary = \textrm{\$215,000 for someone with 20 years of experience}$$

# Things to Consider
- Adding more $$x_{i}$$ variables will always result in a higher R-squared value. However, be careful not to overfit a model. If adding a variable only slightly increases the R-squared value, then it may not be a useful variable in the model and should be removed.
- The example above only showed numerical values for the input variables. However, if you want to use categorical features (i.e. male/female), each category should be assigned a numerical value (i.e. male = 0, female = 1). In this way, the model can differentiate between the two categorical features.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen.)

   [Kaggle Dataset]: <https://www.kaggle.com/karthickveerakumar/salary-data-simple-linear-regression>