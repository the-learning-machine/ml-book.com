---
title: ML Mind Map
subtitle: High-level picture of the ML world.
author: andrew
tags: [how-to-start]
permalink:
---

# Overview
Machine Learning is divided by **2** sub-fields: **supervised** and **unsupervised** learning, both related to data you use to build a model.

- **In supervised learning**, the data you use is labelled, i.e. it has a target variable $$Y$$ you need to predict,
given the response variables $$X$$. For example, predicting price of an apartment, given such variables as: squared
footage, number of rooms, district, area, number of schools nearby, etc.
- **In unsupervised learning**, the data you use is unlabelled, i.e. it does not have a target variable Y. An example of supervised learning is
grouping your customers by segment, based on their characteristics (response variables).

To get a little bit ahead, let's view on the ML Mindmap, and further dissect it!

{% include docschart.html %}
<!-- <img src="/uploads/doc/getting_started/mindmap.png" align="middle"> -->


# Supervised Learning
Supervised learning is divided into two types of algorithms:
- #### Classification algorithms:
  Algorithms that predict a **category**. Examples can be, an algorithm predicting a movie rating: "Best", "Good", "Bad", "Worst"; an algorithm predicting a fruit: 'Banana', 'Apple', 'Orange'; an algorithm predicting any simple yes-no question: "Yes" or "No", etc.

- #### Regression algorithms:
  Algorithms that predict a **continuous value**, such as “dollars” or “weight”. Examples can be, an algorithm predicting a price for the appartment; an algorithm predicting a weight of a person.

# Unsupervised Learning
Unsupervised learning is divided into two types of algorithms:
- #### Clustering algorithms:
  Algorithms that **group** data based on common characterists that the model would find in the dataset. Examples can be, an algorithm segmenting customers in a market.
- #### Generation algorithms:
  Algorithms that **generates** data, it mostly related to Natural Language Processing, e.g. generating text.