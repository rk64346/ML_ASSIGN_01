#!/usr/bin/env python
# coding: utf-8

# In[2]:


##Introduction to
#Machine Learning-1

QD- E=plain thK followin* with an K=amplK-F
C) Artificial IntKlli*KncJ
<) MachinK LKarnin,
I) DKKp LKarning
# Artificial Intelligence (AI):
# 
# Artificial Intelligence refers to the development of computer systems that can perform tasks that typically require human intelligence. This encompasses a wide range of capabilities, from understanding natural language and recognizing images to problem-solving, learning, and decision-making
# 
# Machine Learning (ML):
# 
# Machine Learning is a subset of Artificial Intelligence that focuses on the development of algorithms and statistical models that allow computers to perform a task without being explicitly programmed. In other words, it's a way to enable machines to learn from data and make decisions or predictions based on it.
# 
# Deep Learning (DL):
# 
# Deep Learning is a subfield of Machine Learning that involves the use of neural networks with many layers (hence, "deep"). These neural networks are inspired by the structure of the human brain and are designed to automatically learn features from data.

# Q2- What is supKrvisKd lKarnin*? List somK K=amplKs of supKrvisKd lKarnin*.

# supervised learning have two fundamentals:
#     classifiaction, regression
#     regression-output-continous value 
#     eg: size of house - no. ofrooms - price of house

# Q3- GWhat is unsupKrvisKd lKarnin*? List somK K=amplKs of unsupKrvisKd lKarnin*.

# unsupervised learning: no output - clusters - group of similar data 
# eg: customer segmentation

# Q4- What is thK diffKrKncK bKtwKKn AI, ML, DL, and DS?

# AI : smart intellegence that can perform tesk without any human intervention
# ML: it provide stattool to analyse , visualize, predictive models, forecasting
# DL: multi layered nural netwoprk - to mimic the human brain
# DS: covers all the aspect AI,ML,DL it is called data science
Q5- What arK thK main diffKrKncKs bKtwKKn supKrvisKd, unsupKrvisKd, and sKmi-supKrvisKd lKarnin*?
# supervised: output with continious value
# unsupervised: no output - clusters
# semi supervised: supervised+unsupervised

# Q6- What is train, tKst and validation split? E=plain thK importancK of Kach tKrm.

# training data set: we will train our model eg: books- Q&A - train
# validation dataset: hyper tuning of the model - eg: different books- hyper parameter tuning
# test dataset: module will be test- eg; question papaer- brain test - 85%

# Q7- How can unsupKrvisKd lKarnin* bK usKd in anomaly dKtKction?

# Data Preparation:
# 
# First, you collect a dataset that represents normal behavior or states. This dataset should not contain any instances of anomalies or outliers.
# Feature Engineering:
# 
# Next, you extract relevant features from the data. These features are characteristics or attributes that help describe the data points. The choice of features is crucial in effectively detecting anomalies.
# Unsupervised Learning Algorithm:
# 
# You apply an unsupervised learning algorithm to the dataset. Common techniques used for anomaly detection in unsupervised learning include:
# Clustering: Algorithms like K-Means can be used to group similar data points together. Anomalies may be the data points that do not fit well into any cluster.
# Autoencoders: These are a type of neural network that learn to compress and then reconstruct data. Anomalies tend to have higher reconstruction errors.
# Isolation Forest: It isolates anomalies by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of that feature.
# Anomaly Score Calculation:
# 
# After applying the unsupervised algorithm, each data point is assigned an anomaly score. This score represents how unusual or different a data point is compared to the rest of the dataset.
# Threshold Setting:
# 
# Based on the anomaly scores, you set a threshold above which a data point is considered an anomaly. This threshold can be determined manually, using statistical methods, or through other techniques.
# Detection and Evaluation:
# 
# Finally, you use the established threshold to detect anomalies in new, unseen data. Data points with anomaly scores above the threshold are flagged as anomalies.
# Evaluation and Fine-tuning:
# 
# The performance of the anomaly detection model is evaluated using metrics like precision, recall, F1-score, etc. The model may be fine-tuned by adjusting parameters or feature selection to improve its performance.

# Q8- List down somK commonly usKd supKrvisKd lKarnin* al*orithms and unsupKrvisKd lKarnin*
# al*orithms.

# Supervised Learning Algorithms:
# 
# Linear Regression: Predicts a continuous target variable based on one or more input features, assuming a linear relationship.
# 
# Logistic Regression: Used for binary classification problems. It models the probability that a given input point belongs to a particular class.
# 
# Decision Trees: Makes decisions by splitting the data based on features, creating a tree-like structure.
# 
# Random Forest: An ensemble method that combines multiple decision trees to improve accuracy and control overfitting.
# 
# Unsupervised Learning Algorithms:
# 
# K-Means Clustering: Used to partition data into clusters based on similarity.
# 
# Hierarchical Clustering: Builds a hierarchy of clusters by merging or splitting them based on similarity.
# 
# DBSCAN (Density-Based Spatial Clustering of Applications with Noise): Clusters data based on density and is effective at finding clusters of arbitrary shapes.
# 
# Principal Component Analysis (PCA): Used for dimensionality reduction by transforming data into a lower-dimensional space while preserving variance.

# In[ ]:




