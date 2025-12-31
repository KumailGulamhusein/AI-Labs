# AI Labs – Lab 1 - 3: Clustering, Classification, and Neural Networks

This repository contains R scripts demonstrating **unsupervised clustering**, **supervised classification**, and **neural network models** on various datasets. Each part explores different machine learning techniques with visualization and evaluation metrics.

---

## Table of Contents

1. [Part 1 – Seed Dataset Clustering](#part-1---seed-dataset-clustering)
2. [Part 2 – Seed Dataset Classification](#part-2---seed-dataset-classification)
3. [Part 3 – Neural Network: OR Gate](#part-3---neural-network-or-gate)
4. [Part 3 (cont.) – Neural Network: Wine Dataset](#part-3-cont---neural-network-wine-dataset)
5. [Usage](#usage)
6. [Dependencies](#dependencies)

---

## Part 1 – Seed Dataset Clustering

This part demonstrates **unsupervised clustering** on a seed dataset using **K-means** and **Hierarchical clustering**. The clustering results are evaluated using **Weighted Kappa** against the true seed classes.

### Steps:

1. **Data preparation**

   * Loads the dataset and true labels.
   * Removes missing values and standardizes the data.

2. **K-means clustering**

   * Runs K-means for 2–10 clusters.
   * Calculates Weighted Kappa for each cluster count.
   * Plots Weighted Kappa vs number of clusters to identify the optimal K.

3. **Hierarchical clustering**

   * Computes Euclidean distances.
   * Performs clustering with **average**, **single**, and **complete linkage**.
   * Cuts dendrograms into 3 clusters and visualizes cluster assignments.
   * Calculates Weighted Kappa for each method.

4. **Visualization**

   * Plots raw data, dendrograms, cluster assignments, and Weighted Kappa values.

**Goal:** Compare K-means and hierarchical methods, identifying clusters that best match real seed types.

---

## Part 2 – Seed Dataset Classification

This part performs **supervised classification** on the seed dataset using **Decision Trees** and **K-Nearest Neighbors (KNN)**.

### Steps:

1. **Data preparation**

   * Randomizes and splits the dataset into **training** (first 125 samples) and **test** sets (remaining samples).

2. **Decision Tree**

   * Builds a tree with `rpart`.
   * Visualizes the tree.
   * Predicts on the test set and calculates accuracy.
   * Performs **pruning** to simplify the tree and evaluates pruned tree accuracy.

3. **K-Nearest Neighbors (KNN)**

   * Uses `class::knn` with k = 3, 5, 7.
   * Computes accuracy for each k.
   * Compares KNN results with pruned decision tree performance.

4. **Visualization**

   * Plots feature scatterplots colored by predicted classes.

**Goal:** Evaluate and compare the performance of decision trees and KNN classifiers.

---

## Part 3 – Neural Network: OR Gate

This part demonstrates a **simple neural network** modeling a logical OR gate.

### Steps:

1. **Data preparation**

   * Defines input combinations for OR gate: `(-1, -1), (-1, 1), (1, -1), (1, 1)`.
   * Sets corresponding output values.

2. **Neural network training**

   * Trains a network with **two hidden layers**, 3 neurons each.
   * Uses a threshold of 0.001 and stepmax = 1e5.

3. **Testing**

   * Predicts outputs for all input combinations.
   * Converts outputs to binary using a 0.5 threshold.
   * Displays network structure, neuron activations, and predictions.

**Goal:** Demonstrate neural network learning on a basic logical function.

---

## Part 3 (cont.) – Neural Network: Wine Dataset

This part trains a neural network for **binary classification** of wine types based on two features.

### Steps:

1. **Data preparation**

   * Reads wine dataset.
   * Splits into training (first half) and test (second half).
   * Converts class labels to binary (0,1) and normalizes features.

2. **Neural network training**

   * Combines features and output into a single dataset.
   * Trains a network with two hidden layers (3 neurons each).

3. **Testing**

   * Predicts on test data.
   * Converts outputs to binary classes using 0.5 threshold.
   * Calculates accuracy against true labels.

4. **Visualization**

   * Plots the neural network structure.

**Goal:** Train and evaluate a neural network on a real-world dataset for binary classification.

---

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/AI-Labs.git
   ```

2. Open the scripts in **RStudio**.

3. Run each script section step-by-step. Visualizations and printed outputs will help evaluate results.

---

## Dependencies

* R (>= 4.0.0)
* Packages:

  * `rpart`
  * `class`
  * `neuralnet`

Install missing packages in R using:

```r
install.packages(c("rpart", "class", "neuralnet"))
```

---

**Summary:**
This repo provides practical examples of **unsupervised clustering**, **supervised classification**, and **neural networks** in R. Each part includes data preprocessing, modeling, visualization, and evaluation to help understand core machine learning techniques.

---