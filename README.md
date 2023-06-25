# Data-Analysis-and-Prediction-on-Academic-Citation-Networks

Authors: Ofri Hefetz, Shai Shani Bar

## Our Task 
In this task we will use, machine learning algorithms and statistical tools for network analysis. 
Our main task will consist of the analysis, investigation, and classification of a data set describing a network of academic citations. 

## Data
The data set is a directed graph, representing a network of academic citations. 
The graph contains about 100,000 vertices where each vertex in the graph represents an article and each directed arc between vertex A and vertex B represents that vertex quoted at the top of B. 

Each article is represented by a feature vector created by averaging all word representations (created by the gram-skip model) in its abstract and title. Also, the data set contains for each article the year of its publication and a number representing the category to which it belongs.
