---
title: Decision Tree (ID3)
subtitle: This document will cover something cool.
author: victor
tags: [mlmindmap]
permalink:
---

# 1. Introduction 
There are two types of decision trees:  
- **Classification tree** - used with categorical data: the predicted outcome is the class to which the data belongs. For example, an outcome of a loan application as ‘safe’ or ‘risky’.  

- **Regression tree** - used with continuous data: the predicted outcome is a real number. For example, a population of a state or inhabitant height in centimeters.

Thus, decision trees can handle both categorical and numerical data. This section conveys decision trees for classification problems. For Regression trees, go [here](https://ml-book.com/docs/cl_rf/).

# 2. Decision Trees in Classification

Decision tree builds classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets by learning a series of explicit if-then rules on feature values that results in predicting a target value.

A decision tree consists of the decision nodes and leaf nodes. A decision node (Outlook or Wind) has two or more branches (e.g., Sunny, Overcast and Rain). **Leaf node** (e.g., Play Golf) represents a classification (i.e. decision), and it is an endpoint (last node) of any branch (Yes, No, No, Yes). The topmost decision node in a tree which corresponds to the best predictor called **root node** (Outlook).

<p align="center">
    <img src="/uploads/doc/classification/dt_id3_2_diagrams.png" height="400" width="400">
</p>

# 3. ID3 Algorithm

There are various decision tree algorithms, namely, ID3 (Iterative Dichotomiser 3), C4.5 (successor of ID3), CART (Classification and Regression Tree), CHAID (Chi-square Automatic Interaction Detector), MARS. This article is about a classification decision tree with ID3 algorithm.

One of the core algorithms for building decision trees is ID3 by *J. R. Quinlan*. ID3 is used to generate a decision tree from a dataset commonly represented by a table. To construct a decision tree, ID3 uses a top-down, greedy search through the given columns, where each column (further called **attribute**) at every tree node is tested, and selects the attribute that is best for classification of a given set. To decide what attribute is best to select to construct a decision tree, ID3 uses **Entropy** and **Information Gain**.

# 4. Entropy & Information Gain

**Entropy (E)**

Entropy is the measure of the **amount of uncertainty** or **randomness** in data. Intuitively, it shows predictability of a certain event. If an outcome of an event has a probability of 100%, the entropy is zero (no randomness exists), and if an outcome is 50%, the entropy takes the maximum value (i.e. equals to 1 since it is the `log base 2`) as it projects perfect randomness. For example, consider a coin toss whose probability of heads is 0.5 and probability of tails is 0.5. The entropy here is the highest possible value (i.e., equals 1), since there’s no chance to precisely determine the outcome. Alternatively, consider a coin which has heads on both the sides, the outcome of such an event can be predicted perfectly since we know beforehand that it will always be heads. In other words, this event has **no randomness**, hence its entropy is zero. **ID3 follows the rule: a branch with an entropy of 0 is a leaf node (endpoint). A branch with an entropy more than 0 needs further splitting**. In case it is not possible to achieve zero entropy in the leaf nodes, the decision is made by the method of a **simple majority**.

To build a decision tree, we need to calculate two types of entropy using frequency tables as follows:

1. Entropy $$E(S)$$ using the frequency table of one attribute, where $$S$$ is a current state (existing outcomes) and $$P(x)$$ is a probability of an event $$x$$ of that state $$S$$:

$$
E(S) = \sum_{x epsilon X}^{} - P(x)log_2P(x)
$$ (1)

2. Entropy $$E(S, A)$$ using the frequency table of two attributes - $$S$$ and $$A$$, where $$S$$ is a current state with an attribute $$A$$                          (existing outcomes with an attribute A), A is a selected attribute, and P(x) is a probability of an event x of an attribute A.

$$
E(S,A) = \sum_{x epsilon X}^{}[P(x)*E(S)(2)]
$$ (2)

$$E(S)$$ is the Entropy of the entire set, while the second term $$E(S, A)$$ relates to an Entropy of an attribute $$A$$.

**Information Gain (IG)**

Information gain (also called as Kullback-Leibler divergence) denoted by $$IG(S, A)$$ for a state S is the **effective change in entropy** after deciding on a particular attribute $$A$$. It measures the relative change (decrease) in entropy with respect to the independent variables, as follows:  

$$
IG(S,A) = E(S) - E(S,A)
$$ (3)

The information gain is based on the decrease in entropy after a dataset is split on an attribute. Constructing a decision tree is all about selecting each attribute $$(A)$$ to calculate Information Gain and finding such an attribute that returns the highest IG (i.e., the most homogeneous branches). This attribute will be the next decision node for the tree.

#5.   Example

Let’s understand this with the help of an example outlined in the beginning.

Consider a piece of data collected over the course of 14 days where the features are Outlook, Temperature, Humidity, Wind and the outcome variable is whether Golf was played on the day. Now, our job is to build a predictive model which takes in above 4 parameters and predicts whether Golf will be played on the day. We’ll build a decision tree to do that using ID3 algorithm.

<p align="center">
    <img src="/uploads/doc/classification/dt_id3_table.png" height="450" width="400">
</p>

ID3 Algorithm will perform following tasks recursively:

1. Create a root node for the tree

2. If all examples are positive, return leaf node ‘positive’

3. Else if all examples are negative, return leaf node ‘negative’

4. Calculate the entropy of current state $$E(S)$$

5. For each attribute, calculate the entropy with respect to the attribute ‘$$A$$’ denoted by $$E(S, A)$$

6. Select the attribute which has the maximum value of $$IG(S, A)$$ and split the current (parent) node on the selected attribute

7. Remove the attribute that offers highest $$IG$$ from the set of attributes

8. Repeat until we run out of all attributes, or the decision tree has all leaf nodes.

Now we’ll go ahead and grow the decision tree. The initial step is to calculate $$E(S)$$, the Entropy of the current state (i.e. existing outcomes at this stage). In the above example, we can see in total there are 9 Yes’s and 5 No’s.

<p align="center">
    <img src="/uploads/doc/classification/dt_id3_part.png" height="150" width="200">
</p>

Let's calculate $$E(S)$$ using the formula (1):

$$
E(S) = \sum_{x epsilon X}^{} - P(x)log_2P(x) = -\frac{9}{14}log_2(\frac{9}{14})-\frac{5}{14}log_2(\frac{5}{14})=0.940
$$

Remember that the Entropy is 0 if all members belong to the same class, and 1 when half of them belong to one class and other half belong to other class, which is perfect randomness. Here it’s 0.94, which means the distribution is **fairly random**.

Now the next step is to choose the attribute that gives us highest possible Information Gain which we’ll choose as the root node. Let's start with *‘Wind’* attribute, calculating its $$E(S, Wind)$$ and $$IG(S, Wind)$$:

$$
IG(S, Wind) = E(S)-E(S,Wind)\rightarrow
IG(S, WInd) = E(S) - \sum_{x epsilon X}^{}[P(x)*E(S)]
$$

where *‘x’* in $$P(x)$$ are the possible values for an attribute. Here, attribute *‘Wind’* takes two possible values in the sample data.

Hence, $$x = {Weak, Strong}$$

$$
IG(S, Wind) = E(S) - P(S_weak)*E(S_weak) - P(S_strong)*E(S_strong)
$$

Thus, we have to find the following terms:

$$
E(S_weak)\\
E(S_strong)\\
P(S_weak)\\
P(S_strong)\\
E(S)=0.940
$$, which we have already calculated

Amongst all the 14 examples we have 8 places where the wind is *Weak* and 6 where the wind is *Strong*.

<p align="center">
    <img src="/uploads/doc/classification/dt_id3_part2.png" height="150" width="200">
</p>

Now out of the 8 Weak examples, 6 of them were ‘Yes’ for Play Golf and 2 of them were ‘No’ for ‘Play Golf’. So, let's calculate an entropy for *"Weak"* values of *Wind* attribute:



Similarly, out of 6 Strong examples, we have 3 examples where the outcome was ‘Yes’ for Play Golf and 3 where we had ‘No’ for Play Golf.

Wind

Weak

Strong

Yes: 6

No: 2

Yes: 3

No: 3

Remember, here half items belong to one class while other half belong to other. Hence we have perfect randomness.

 

Now we have all the pieces required to calculate the Information Gain:

That tells us the Information Gain by considering ‘Wind’ as the attribute and gives us information gain of 0.048. Now the next step is to choose the attribute that gives us highest possible Information Gain which we’ll choose as the root node. Therefore, we must similarly calculate the Information Gain for all the other attributes and pick the one with the highest score.

(calculated in a previous example)

We can clearly see that IG(S, Outlook) has the highest information gain of 0.246, hence we chose Outlook attribute as the root node. At this point, the decision tree looks like:

Yes

Outlook

Sunny

Overcast

Rain

?

?

Here we observe that whenever the outlook at Overcast, Play Golf  is always ‘Yes’. That means, the entropy is 0 and we can leave "Yes" as a leaf node. The fact that Overcast is always yes is not a coincidence by any chance, the simple tree resulted due to the highest information gain, given by the attribute Outlook.

 

Now how do we proceed from this point? We can simply apply recursion: you might want to look at the algorithm steps described earlier.

 

Now that we have used Outlook, we have got three of them remaining: Humidity, Temperature, and Wind. And, we had three possible values of Outlook: Sunny, Overcast, Rain. Where the Overcast node already ended up having leaf node ‘Yes’, so we’re left with two subtrees to compute: Sunny and Rain. Let's start with Sunny, and compute its entropy.

 

Amongst all the 5 examples the attribute value of Outlook is Sunny, 2 of them were ‘Yes’ for Play Golf and 3 of them were ‘No’ for ‘Play Golf’.

In the similar fashion, we compute the following values:

As we can see the highest Information Gain is given by Humidity. Proceeding in the same way with             will give us Wind as the one with highest information gain.

 

The final Decision Tree is going to be looked as such:

Outlook

Sunny

Rain

Overcast

Humidity

Yes

Wind

High

No

Normal

Yes

Strong

No

Weak

Yes

6.   Summary

A decision tree is built top-down from a root node and involves partitioning the data into subsets that contain instances with similar values (homogenous). ID3 algorithm uses entropy to calculate the homogeneity of a sample. If the sample is completely homogeneous, the entropy is zero and if the sample is an equally divided it has an entropy of one. 

 

The information gain is based on the decrease in entropy after a dataset is split on an attribute. Constructing a decision tree is all about finding an attribute that returns the highest information gain (i.e. the most homogeneous branches, or the lowest entropy). After that, all the outcome instances that are possible are examined whether they belong to the same class or not. For the instances of the same class, a single name class is used to denote otherwise the instances are classified on the basis of splitting attribute.

 

7.   Overfitting and Pruning

One of the most common problems with decision trees, especially the ones that have a table full of columns, is that they tend to overfit a lot. Sometimes it looks like the tree just memorizes the data. Here are the typical examples of decision trees that overfit, both for categorical and continuous data:

Categorical:

If the client is male, between 15 and 25, from the US, likes ice-cream, has a German friend, hates birds and ate pancakes on August 25th, 2012, - he is likely to download Pokemon Go.

Continuous:

Screen Shot 2019-01-05 at 14.23.32.png
There are two main ways to mitigate overfitting in Decision Trees:

Using Random Forests

Pruning Decision Trees

Random Forest prevents the problem of overfitting as it is an ensemble of (n) decision trees, not just one, with n results in the end. The final result of Random forest is the most frequent response variable (the mode) among n results (of n Decision Trees).

We will not explain this algorithm in this section. It is a separate algorithm and you can read the article on it here in detail.

Pruning involves the removal of nodes and branches in a decision tree to make it simpler so as to mitigate overfitting and improve performance. Ideally, we want the leaf nodes to be as little randomized as possible for high accuracy, but it is very easy to overfit, so much so, that in many cases, the leaf nodes may only have a single data point. We can mitigate this by pruning the decision tree by a method called cost-effective pruning.

The following algorithm takes place while applying cost-effective pruning: 

Determine the performance of the original tree, T, with the validation data

Consider a sub-tree, t(1), and remove it from the original tree, replacing a sub-tree with a leaf.

Determine the performance of a new tree, T(new).

If the delta in performance is insignificant (that is, if validation set does not have the significant difference in delta performance), consider simpler (pruned) tree (Occam’s razor) as an original, and continue to the next sub-tree.

number of leaves

Original tree T

Validation Set

Credit

Excellent

Poor

Fair

Safe

Term

Income

0.81

5 years

3 years

Low

High

Risky

0.94

Safe

0.62

Risky

0.79

Term

3 years

5 years

Risky

0.57

Safe

0.92

Sub-tree t(1)

Note that this method goes from the bottom of the tree. When you consider sub-trees to be replaced by a leaf node, this sub-tree should be the last one to a leaf node, as shown in the example.

 
8.   Pros & Cons​

Advantages of ID3  

Easily visualized and interpreted. The training data is used to create understandable prediction rules.

No feature normalization is typically needed.

The calculation time of ID3 is the linear function of the product of the characteristic number and node number.

Works well with datasets using a mixture of feature types (continuous/categorical/binary)

​

Disadvantages of ID3  

Data may be overfitted or overclassified.

For making a decision, only one attribute is tested at an instant thus consuming a lot of time.

Classifying the continuous data may prove to be expensive in terms of computation, as many trees have to be generated to see where to break the continuum.  One disadvantage of ID3 is that when given a large number of input values, it is overly sensitive to features with a large number of values.

9.   Decision Tree in Python
View/download a template of Decision Tree located in a git repository here .