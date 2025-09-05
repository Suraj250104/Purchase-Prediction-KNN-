# K-Nearest Neighbors (K-NN) Classification Project

This repository contains a Python script implementing the K-Nearest Neighbors (K-NN) algorithm for a classification task. The goal is to predict whether a user will purchase a product based on their age and estimated salary. The model is built using Scikit-learn and the data is visualized with Matplotlib.

## Dataset

The project uses the `Social_Network_Ads.csv` dataset.
* **Features (X):** 'Age' and 'Estimated Salary'.
* **Target (y):** 'Purchased' (0 for No, 1 for Yes).

## Workflow

The project follows these key steps:

1.  **Data Preprocessing:**
    * The necessary libraries (`numpy`, `pandas`, `matplotlib.pyplot`) are imported.
    * The dataset is loaded and split into features (X) and the target variable (y).
    * The data is divided into a Training set (75%) and a Test set (25%).
    * **Feature Scaling** is applied to the 'Age' and 'Estimated Salary' columns using `StandardScaler` to normalize the data and ensure that one feature does not dominate the other.

2.  **Model Training:**
    * A K-Nearest Neighbors classifier (`KNeighborsClassifier`) is imported from Scikit-learn.
    * The classifier is initialized with the following parameters:
        * `n_neighbors = 5` (number of neighbors to use).
        * `metric = 'minkowski'` with `p = 2` (equivalent to the standard Euclidean distance).
    * The model is trained on the scaled training data (`X_train`, `y_train`).

3.  **Prediction and Evaluation:**
    * Predictions are made on the scaled test set (`X_test`).
    * The model's performance is evaluated using a **Confusion Matrix** and **Accuracy Score**.

4.  **Visualization:**
    * The decision boundaries for both the Training and Test sets are visualized.
    * The plots show how the model classifies users in the feature space, with data points colored based on their actual purchase decision ('Salmon' for not purchased, 'DodgerBlue' for purchased).

## Results

The model achieved an accuracy of **93%** on the test set.

The confusion matrix for the test set is as follows:
```
[[64  4]
 [ 3 29]]
```
* **True Negatives:** 64
* **False Positives:** 4
* **False Negatives:** 3
* **True Positives:** 29

### Visualizations

**K-NN Decision Boundary (Training Set)**
![Training Set Results](path/to/your/training_set_image.png)
*(This plot shows the model's predictions across the training data points.)*

**K-NN Decision Boundary (Test Set)**
![Test Set Results](path/to/your/test_set_image.png)
*(This plot shows how well the model generalizes to new, unseen data from the test set.)*

## How to Run

1.  Clone this repository to your local machine:
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd your-repository-name
    ```
3.  Ensure you have the required libraries installed. You can install them using pip:
    ```bash
    pip install numpy pandas scikit-learn matplotlib
    ```
4.  Open and run the `k_nearest_neighbors.ipynb` notebook in a Jupyter environment.

## Dependencies

* Python 3
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
