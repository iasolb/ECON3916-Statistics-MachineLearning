# ECON 3916 - Assignment 3: Causal Inference

**Author:** Ian Solberg  
**Course:** ECON 3916 - Statistics & Machine Learning  
**File:** [`Econ_3916_Assignment_3_Causal_IanSolberg.ipynb`](./Econ_3916_Assignment_3_Causal_IanSolberg.ipynb)

## 📌 Overview
This repository contains my submission for Assignment 3 in ECON 3916. The notebook bridges the gap between traditional econometric techniques and modern machine learning by focusing on **Causal Inference**. Instead of purely predictive modeling, this project aims to estimate causal effects and isolate the impact of specific treatment variables on outcomes of interest while controlling for confounding factors.

## 📊 Objectives & Methodology
The primary goal of this notebook is to rigorously estimate causal treatment effects. Depending on the dataset and specific assignment prompts, the notebook covers:

* **Data Preprocessing & Exploratory Data Analysis (EDA):** Cleaning the data, handling missing values, and exploring the distributions of treatment, outcome, and confounding variables.
* **Selection on Observables:** Implementing techniques to control for observed confounders.
* **Machine Learning for Causal Inference:** (Adjust based on your actual code)
  * *Propensity Score Matching / Weighting*
  * *Double/Debiased Machine Learning (DML)* using cross-fitting to estimate unbiased treatment effects.
  * *Heterogeneous Treatment Effects (HTE)* using methods like Causal Trees or Forests.
* **Econometric Validation:** Comparing machine learning causal estimates against baseline models like Ordinary Least Squares (OLS).

## 🧰 Dependencies & Installation
To run this notebook, you will need a standard data science environment with Python 3.x. The following libraries are required:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels
