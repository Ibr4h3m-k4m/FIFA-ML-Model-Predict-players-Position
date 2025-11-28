# FIFA 23 Player Analysis: Clustering & Classification

## Project Overview
This project applies Machine Learning and Data Mining techniques to the FIFA 23 players dataset. The goal is to analyze player statistics, group similar players using unsupervised clustering, and predict player positions using supervised classification models.

## Dataset
The project uses the **FIFA 23 Complete Player Dataset**.
- **Source:** Kaggle (`/kaggle/input/fifa-23-complete-player-dataset/male_players (legacy).csv`)
- **Content:** The dataset includes detailed attributes for over 18,000 players, such as physical stats (age, height, weight), skill ratings (shooting, passing, defending), and financial data (value, wage).

## Project Structure & Workflow

The analysis is implemented in a Jupyter Notebook (`ml-dm-project.ipynb`) and follows these main steps:

### 1. Data Processing & EDA
- **Loading:** Filters the legacy dataset to focus specifically on **FIFA 23** players.
- **Cleaning:** Handles missing values using `SimpleImputer` and drops redundant columns (e.g., `work_rate`).
- **Visualization:** Analyzes the distribution of player positions and correlates features using libraries like `Seaborn` and `Matplotlib`.

### 2. Unsupervised Learning (Clustering)
- **Dimensionality Reduction:** Applies **Principal Component Analysis (PCA)** to reduce the feature space while retaining variance.
- **Clustering:** Implements a **Gaussian Mixture Model (GMM)** to discover latent groupings of players based on their attributes.
- **Visualization:** Plots the resulting clusters on the first two principal components.

### 3. Supervised Learning (Classification)
- **Objective:** Predict player position groups based on their skill attributes.
- **Models Trained:**
    - **Random Forest Classifier:** A bagging ensemble method.
    - **XGBoost Classifier:** A gradient boosting framework.
- **Evaluation:** Models are evaluated using Confusion Matrices and Classification Reports (Precision, Recall, F1-Score).

## Requirements

To run this notebook, you need Python and the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost