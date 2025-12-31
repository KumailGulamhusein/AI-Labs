# \# AI Labs – Lab 1 \& 3: Clustering, Classification, and Neural Networks

# 

# This repository contains R scripts demonstrating \*\*unsupervised clustering\*\*, \*\*supervised classification\*\*, and \*\*neural network models\*\* on various datasets. Each part explores different machine learning techniques with visualization and evaluation metrics.

# 

# ---

# 

# \## Table of Contents

# 

# 1\. \[Part 1 – Seed Dataset Clustering](#part-1---seed-dataset-clustering)

# 2\. \[Part 2 – Seed Dataset Classification](#part-2---seed-dataset-classification)

# 3\. \[Part 3 – Neural Network: OR Gate](#part-3---neural-network-or-gate)

# 4\. \[Part 3 (cont.) – Neural Network: Wine Dataset](#part-3-cont---neural-network-wine-dataset)

# 5\. \[Usage](#usage)

# 6\. \[Dependencies](#dependencies)

# 

# ---

# 

# \## Part 1 – Seed Dataset Clustering

# 

# This part demonstrates \*\*unsupervised clustering\*\* on a seed dataset using \*\*K-means\*\* and \*\*Hierarchical clustering\*\*. The clustering results are evaluated using \*\*Weighted Kappa\*\* against the true seed classes.

# 

# \### Steps:

# 

# 1\. \*\*Data preparation\*\*

# 

# &nbsp;  \* Loads the dataset and true labels.

# &nbsp;  \* Removes missing values and standardizes the data.

# 

# 2\. \*\*K-means clustering\*\*

# 

# &nbsp;  \* Runs K-means for 2–10 clusters.

# &nbsp;  \* Calculates Weighted Kappa for each cluster count.

# &nbsp;  \* Plots Weighted Kappa vs number of clusters to identify the optimal K.

# 

# 3\. \*\*Hierarchical clustering\*\*

# 

# &nbsp;  \* Computes Euclidean distances.

# &nbsp;  \* Performs clustering with \*\*average\*\*, \*\*single\*\*, and \*\*complete linkage\*\*.

# &nbsp;  \* Cuts dendrograms into 3 clusters and visualizes cluster assignments.

# &nbsp;  \* Calculates Weighted Kappa for each method.

# 

# 4\. \*\*Visualization\*\*

# 

# &nbsp;  \* Plots raw data, dendrograms, cluster assignments, and Weighted Kappa values.

# 

# \*\*Goal:\*\* Compare K-means and hierarchical methods, identifying clusters that best match real seed types.

# 

# ---

# 

# \## Part 2 – Seed Dataset Classification

# 

# This part performs \*\*supervised classification\*\* on the seed dataset using \*\*Decision Trees\*\* and \*\*K-Nearest Neighbors (KNN)\*\*.

# 

# \### Steps:

# 

# 1\. \*\*Data preparation\*\*

# 

# &nbsp;  \* Randomizes and splits the dataset into \*\*training\*\* (first 125 samples) and \*\*test\*\* sets (remaining samples).

# 

# 2\. \*\*Decision Tree\*\*

# 

# &nbsp;  \* Builds a tree with `rpart`.

# &nbsp;  \* Visualizes the tree.

# &nbsp;  \* Predicts on the test set and calculates accuracy.

# &nbsp;  \* Performs \*\*pruning\*\* to simplify the tree and evaluates pruned tree accuracy.

# 

# 3\. \*\*K-Nearest Neighbors (KNN)\*\*

# 

# &nbsp;  \* Uses `class::knn` with k = 3, 5, 7.

# &nbsp;  \* Computes accuracy for each k.

# &nbsp;  \* Compares KNN results with pruned decision tree performance.

# 

# 4\. \*\*Visualization\*\*

# 

# &nbsp;  \* Plots feature scatterplots colored by predicted classes.

# 

# \*\*Goal:\*\* Evaluate and compare the performance of decision trees and KNN classifiers.

# 

# ---

# 

# \## Part 3 – Neural Network: OR Gate

# 

# This part demonstrates a \*\*simple neural network\*\* modeling a logical OR gate.

# 

# \### Steps:

# 

# 1\. \*\*Data preparation\*\*

# 

# &nbsp;  \* Defines input combinations for OR gate: `(-1, -1), (-1, 1), (1, -1), (1, 1)`.

# &nbsp;  \* Sets corresponding output values.

# 

# 2\. \*\*Neural network training\*\*

# 

# &nbsp;  \* Trains a network with \*\*two hidden layers\*\*, 3 neurons each.

# &nbsp;  \* Uses a threshold of 0.001 and stepmax = 1e5.

# 

# 3\. \*\*Testing\*\*

# 

# &nbsp;  \* Predicts outputs for all input combinations.

# &nbsp;  \* Converts outputs to binary using a 0.5 threshold.

# &nbsp;  \* Displays network structure, neuron activations, and predictions.

# 

# \*\*Goal:\*\* Demonstrate neural network learning on a basic logical function.

# 

# ---

# 

# \## Part 3 (cont.) – Neural Network: Wine Dataset

# 

# This part trains a neural network for \*\*binary classification\*\* of wine types based on two features.

# 

# \### Steps:

# 

# 1\. \*\*Data preparation\*\*

# 

# &nbsp;  \* Reads wine dataset.

# &nbsp;  \* Splits into training (first half) and test (second half).

# &nbsp;  \* Converts class labels to binary (0,1) and normalizes features.

# 

# 2\. \*\*Neural network training\*\*

# 

# &nbsp;  \* Combines features and output into a single dataset.

# &nbsp;  \* Trains a network with two hidden layers (3 neurons each).

# 

# 3\. \*\*Testing\*\*

# 

# &nbsp;  \* Predicts on test data.

# &nbsp;  \* Converts outputs to binary classes using 0.5 threshold.

# &nbsp;  \* Calculates accuracy against true labels.

# 

# 4\. \*\*Visualization\*\*

# 

# &nbsp;  \* Plots the neural network structure.

# 

# \*\*Goal:\*\* Train and evaluate a neural network on a real-world dataset for binary classification.

# 

# ---

# 

# \## Usage

# 

# 1\. Clone this repository:

# 

# &nbsp;  ```bash

# &nbsp;  git clone https://github.com/yourusername/AI-Labs.git

# &nbsp;  ```

# 

# 2\. Open the scripts in \*\*RStudio\*\*.

# 

# 3\. Run each script section step-by-step. Visualizations and printed outputs will help evaluate results.

# 

# ---

# 

# \## Dependencies

# 

# \* R (>= 4.0.0)

# \* Packages:

# 

# &nbsp; \* `rpart`

# &nbsp; \* `class`

# &nbsp; \* `neuralnet`

# 

# Install missing packages in R using:

# 

# ```r

# install.packages(c("rpart", "class", "neuralnet"))

# ```

# 

# ---

# 

# \*\*Summary:\*\*

# This repo provides practical examples of \*\*unsupervised clustering\*\*, \*\*supervised classification\*\*, and \*\*neural networks\*\* in R. Each part includes data preprocessing, modeling, visualization, and evaluation to help understand core machine learning techniques.

# 

# ---



