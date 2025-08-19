
# Heart Disease Detection using K-NN and Feature Selection

This repository presents a machine learning project focused on detecting heart disease using a **K-Nearest Neighbors (K-NN)** classifier. The work emphasizes the role of **feature selection** in enhancing model performance, especially in medical datasets where certain features may be irrelevant or misleading.

---

## Key Features

### K-Nearest Neighbors (K-NN) Classifier

* Supervised learning algorithm used to classify patients into two categories:

  * **Presence of heart disease**
  * **Absence of heart disease**
* Dataset is split into training and testing subsets for robust evaluation.

### Feature Selection Methods

Two optimization-based feature selection approaches are applied:

1. **Random-Restart Hill Climbing**

   * Starts with a random subset of features.
   * Iteratively flips feature inclusion/exclusion bits.
   * Tracks and retains the best solution across multiple restarts to escape local optima.

2. **Random-Restart Variable Neighbor Search (VNS)**

   * Builds on hill climbing but dynamically adjusts the neighborhood size.
   * Expands the search when improvements are discovered, enabling broader exploration of the solution space.

### Performance Evaluation

* Both feature selection algorithms are executed **10 times** to ensure consistency.
* Average accuracy is computed from multiple runs.
* Evaluation is based on the **confusion matrix** (true positives, true negatives, false positives, false negatives).

---

## Results

* Feature selection improves the predictive performance of the K-NN classifier.
* Random-Restart VNS generally outperforms standard hill climbing by avoiding premature convergence.
* Final results highlight the **importance of careful feature selection** in medical applications.

---

## Code Structure

* **Data preprocessing:** Cleaning and preparing the dataset.
* **Feature selection:** Implementation of both optimization strategies.
* **Model training and testing:** K-NN classifier applied to selected feature subsets.
* **Evaluation:** Accuracy and confusion matrix analysis.

The code is written in Python and is structured for readability and reproducibility.

---

## Context

This project demonstrates the practical application of **machine learning and optimization techniques** in healthcare. It shows how intelligent feature selection can lead to more accurate and reliable predictions in medical datasets.

