# Student Performance and Behavior Analysis with Machine Learning

*Classification, regression, clustering and anomaly detection on a 5,000-student tabular dataset, built with pandas, scikit-learn and XGBoost.*

> Original title: **Analisis Kinerja dan Perilaku Siswa/i Dengan Pendekatan Machine Learning**. The notebook's commentary is written in Bahasa Indonesia; code and variable names are in English.

📄 **Full report:** [Student_Performance_Analysis_Report.pdf](Student_Performance_Analysis_Report.pdf) (15 slides covering the problem, method, results and recommendations)

<details>
<summary><strong>Table of Contents</strong></summary>

- [Overview](#overview)
- [Team](#team)
- [Dataset](#dataset)
- [Project Tasks](#project-tasks)
- [Pipeline](#pipeline)
- [Results](#results)
- [Key Findings](#key-findings)
- [Evaluation Notes and Limitations](#evaluation-notes-and-limitations)
- [Future Work](#future-work)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)

</details>

---

## Overview

This project asks whether routine academic and lifestyle records can tell a school which students need support. Starting from raw student records, we explore the data, clean it, engineer features, and then tackle four machine learning problems:

1. **Classification:** infer a student's parent education level and fill in the 1,794 missing labels.
2. **Regression:** predict a student's total score and stress level.
3. **Clustering:** group students by academic and behavioral profile.
4. **Anomaly detection:** flag unusual students as potential dropouts.

Each task compares several models with tuned hyperparameters. The headline result is negative: across 11 models, nothing beat a naive baseline, because the features carry almost no predictive signal. Every result is reported, including the ones that did not work.

---

## Team

Built as a university machine learning course project by:

- Reuben Zachary Susanto
- Chrisustomus Boimau
- Paulina Devina Wijaya

---

## Dataset

**Students Grading Dataset** (Kaggle), saved as `Students_Grading_Dataset.csv`.

| Property | Value |
|---|---|
| Rows | 5,000 students |
| Columns | 23 |
| Academic features | Attendance (%), Midterm, Final, Assignments, Quizzes, Participation, Projects, Total Score, Grade |
| Behavioral and background features | Study hours per week, Extracurricular activities, Internet access at home, Parent education level, Family income level, Stress level (1-10), Sleep hours per night |
| Demographics | Gender, Age (18 to 24), Department (CS, Engineering, Business, Mathematics) |
| Missing values | Attendance: 516, Assignments_Avg: 517, Parent_Education_Level: 1,794 |

The dataset is not included in this repository. Download it from Kaggle and place the CSV in the project root (see [How to Run](#how-to-run)).

---

## Project Tasks

| Task | Goal | Models compared |
|---|---|---|
| Classification | Predict `Parent_Education_Level` (High School, Bachelor's, Master's, PhD) | Random Forest, XGBoost, Gaussian Naive Bayes |
| Regression | Predict `Total_Score` and `Stress_Level` | Linear Regression, Polynomial Regression (degree 1 to 4), Random Forest Regressor |
| Clustering | Group students into performance profiles | K-Means, DBSCAN, Agglomerative Clustering |
| Anomaly detection | Flag students at risk of dropping out | Isolation Forest, Local Outlier Factor |

---

## Pipeline

### 1. Exploratory Data Analysis
- Data types, null counts and descriptive statistics for every column.
- Correlation heatmap across all numeric features.
- Distribution plots with **Freedman-Diaconis bin widths** (a custom `optimal_bins` helper) and boxplots to check for outliers.
- Bivariate views: assignments by department, attendance bands vs. total score, age vs. gender, and total score vs. stress level.

### 2. Data Cleaning and Imputation
- Dropped identifier columns (`Student_ID`, `First_Name`, `Last_Name`, `Email`).
- `Assignments_Avg`: imputed with the **department-wise mean** so each department keeps its own average.
- `Attendance (%)`: imputed with the global mean, since its distribution is close to uniform.
- `Parent_Education_Level`: imputed **with a trained classifier** (Task 1) instead of the mode.

### 3. Feature Engineering and Encoding
- **Binned features:** `Participation_Score` (Low below 6, High otherwise) and `Sleep_Hours_per_Night` (Low below 6 hours, High otherwise), then one-hot encoded.
- **One-hot encoding** for Gender, Department, Grade, Extracurricular Activities, Internet Access and Family Income Level.
- **StandardScaler** for Age and Study Hours per Week.
- **MinMaxScaler** for score-based features, attendance, assignments and stress level.

### 4. Modeling
- **Classification:** `RandomizedSearchCV` (5-fold) for Random Forest and XGBoost hyperparameters, followed by **10-fold stratified cross-validation** and a 20% hold-out test set. Feature importance plotted for both tree models.
- **Regression:** MAE, MSE, RMSE and R² for every model on a 20% hold-out set, plus actual-vs-predicted plots, residual plots and residual histograms. Polynomial degree swept from 1 to 4; Random Forest swept from 100 to 400 trees.
- **Clustering:** PCA to two components, then a manual grid search over the silhouette score for each algorithm (k from 2 to 15, DBSCAN `eps` and `min_samples`), with a dendrogram for agglomerative clustering.
- **Anomaly detection:** grid search over Isolation Forest (`n_estimators`, `max_samples`, `contamination`) and LOF (`n_neighbors` from 10 to 45, `contamination` from 0.01 to 0.09) using the silhouette score.

---

## Results

### Classification: Parent Education Level (4 balanced classes, chance about 25%)

| Model | 10-fold stratified CV accuracy | Hold-out accuracy (642 students) |
|---|---|---|
| Random Forest (`n_estimators=100`, `max_depth=30`) | **25.76% ± 2.63%** | 26.32% |
| XGBoost | 23.61% ± 2.26% | 23.21% |
| Gaussian Naive Bayes | 23.64% ± 2.37% | 24.45% |

### Regression: Total Score and Stress Level (targets scaled to [0, 1])

| Model | Total Score RMSE | Total Score R² | Stress RMSE | Stress R² |
|---|---|---|---|---|
| Linear Regression | **0.2864** | -0.0155 | **0.3131** | -0.0023 |
| Polynomial Regression (degree 2) | 0.2966 | -0.0892 | 0.3205 | -0.0504 |
| Random Forest Regressor (400 trees) | 0.2957 | -0.0287 | 0.3220 | -0.0114 |

At polynomial degree 3, RMSE roughly doubles (0.5983 for total score, 0.7399 for stress).

### Clustering (on 2 PCA components)

| Model | Best setting | Silhouette score |
|---|---|---|
| K-Means | k = 4 | 0.412 |
| Agglomerative | k = 6 | 0.364 |
| DBSCAN | eps = 0.1, min_samples = 3 (7 clusters) | 0.119 |

DBSCAN was kept for the final labels despite its lower silhouette score, because it was the only method that kept each band in the PCA projection inside a single cluster. See [Key Findings](#key-findings) for what those bands most likely are.

### Anomaly Detection

| Model | Best setting | Silhouette score |
|---|---|---|
| Isolation Forest | 300 trees, max_samples = 0.8, contamination = 0.01 | **0.109** |
| Local Outlier Factor | n_neighbors = 45, contamination = 0.01 | 0.102 |

Isolation Forest flagged **50 of 5,000 students** (1%) as potential dropouts.

---

## Key Findings

- **The features carry almost no predictive signal.** Every classifier sits at chance level for parent education, and every regressor has an R² below zero for total score and stress. The correlation heatmap shows the same thing: no two numeric features correlate by more than 0.04 in absolute value. The most likely explanation is that the dataset is synthetic.
- **Simpler models held up best.** When there is no signal to learn, extra capacity only adds variance. Linear Regression beat the Polynomial and Random Forest regressors, and polynomial error rose sharply after degree 2.
- **The clusters most likely follow age, not performance.** The PCA projection shows seven parallel bands, matching the seven ages in the data (18 to 24). Age and study hours are the only standardized features, so they have the largest variance and dominate both principal components. A simulation of the same preprocessing reproduces seven DBSCAN clusters, one per age. Coloring the PCA plot by `Age` would confirm this on the real data. Either way, the original goal of three groups (high-performing, average, struggling) was not supported.
- **Imputing a label nobody can predict adds noise.** Because the classifier performs at chance, filling the 1,794 missing parent-education values with its predictions is close to random assignment. Mode imputation or an explicit "Unknown" category would be just as defensible.

---

## Evaluation Notes and Limitations

- **A data leakage bug was found and fixed.** An earlier version of the notebook reported 92.21% hold-out accuracy for Random Forest (76.32% for XGBoost). The test set had been scored by the model from the last cross-validation fold, which had already trained on about 90% of the test rows. A deep Random Forest memorizes rows it has seen, so 0.9 × 100% + 0.1 × 25% ≈ 92.5%. The notebook now scores the test set only with the model fit on the training split, and hold-out accuracy matches cross-validation.
- **Scalers are fit on the full dataset** before the train/test split, which leaks a small amount of test-set information. It has little effect here but should be moved into a `Pipeline`.
- **The anomaly labels have no ground truth.** "Potential Dropout" means statistically unusual, not a validated dropout prediction, and the 1% share is set by the `contamination` parameter.
- **Silhouette score is a weak selection criterion for anomaly detectors.** With 1% contamination it mostly rewards isolating a handful of extreme points.

---

## Future Work

- Wrap preprocessing and models in scikit-learn `Pipeline` objects and evaluate with `cross_validate` to remove all leakage by design.
- Report a label-free baseline (`DummyClassifier`, `DummyRegressor`) beside every model to make the "no signal" result explicit.
- Test feature-target dependence directly with mutual information or permutation tests before tuning.
- Use an "Unknown" category for missing parent education instead of model-based imputation.
- Repeat the analysis on a real-world student dataset where the outcome is known to depend on the inputs.

---

## Tech Stack

```text
- python 3
- pandas, numpy
- scikit-learn (RandomizedSearchCV, StratifiedKFold, RandomForest, NaiveBayes,
  LinearRegression, PolynomialFeatures, KMeans, DBSCAN, AgglomerativeClustering,
  IsolationForest, LocalOutlierFactor, PCA)
- xgboost
- scipy (hierarchical clustering dendrogram)
- matplotlib, seaborn
```

---

## Project Structure

```text
.
├── Analisis_Kinerja_dan_Perilaku_Siswa_i_Dengan_Pendekatan_Machine_Learning.ipynb
├── Student_Performance_Analysis_Report.pdf
├── README.md
└── Students_Grading_Dataset.csv   # download from Kaggle, not committed
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ReubenZSTechs/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn jupyter

# 3. Download Students_Grading_Dataset.csv from Kaggle into the project root

# 4. Launch the notebook
jupyter notebook Analisis_Kinerja_dan_Perilaku_Siswa_i_Dengan_Pendekatan_Machine_Learning.ipynb
```

The notebook also runs on Google Colab. Upload the CSV to the session first.
