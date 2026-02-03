# Lab 4: Descriptive Statistics & Anomaly Detection
## Computational Laboratory: Robustness in a Skewed World

**Course:** ECON3916.33674.202630  
**Assignment:** [Class 4 | Lab] Descriptive Statistics — Robustness in a Skewed World  
**Student:** Ian Solberg  
**Due Date:** January 28, 2026

---

## 📋 Lab Overview

This lab challenges the traditional reliance on the **Average** as a reliable metric in modern data analysis. Using the California Housing Dataset as a proxy for real-world marketplace data (e.g., Airbnb, Zillow), we explore how economic anomalies distort summary statistics and learn to identify them through both manual statistical forensics and algorithmic machine learning approaches.

### Key Learning Objectives
- Understand the "breakdown point" of the mean in skewed distributions
- Apply the Tukey Fence (IQR method) for univariate outlier detection
- Implement Isolation Forest for multivariate anomaly detection
- Compare robust statistics (Median, MAD) vs. fragile statistics (Mean, Standard Deviation)
- Generate professional comparative forensics reports

---

## 🗂️ Project Structure

```
lab4-anomaly-detection/
│
├── README.md                      # This file
├── lab4_analysis.ipynb            # Main Jupyter notebook with all exercises
├── forensics_report.py            # Comparative analysis script
│
├── outputs/
│   ├── house_value_distribution.png
│   ├── income_boxplot.png
│   ├── isolation_forest_scatter.png
│   └── comparative_forensics.png
│
└── data/
    └── california_housing.csv     # Dataset (loaded via sklearn)
```

---

## 🔧 Setup & Dependencies

### Required Libraries
```python
sklearn
pandas
numpy
matplotlib
seaborn
scipy
```

### Installation
```bash
pip install scikit-learn pandas numpy matplotlib seaborn scipy
```

### Data Source
The California Housing Dataset is loaded directly from `sklearn.datasets`:
```python
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
df = data.frame
```

---

## 📊 Lab Phases

### **Phase 1: Data Ingestion & Inspection**
- Loaded California Housing dataset (20,640 samples)
- Identified the "$500k cap" ceiling effect in `MedHouseVal`
- Visualized distribution to understand data skewness

**Key Insight:** The dataset exhibits a right-skewed distribution with artificial censoring at the upper bound.

---

### **Phase 2: Manual Statistical Forensics** 
⚠️ *AI Usage Prohibited - Pure Statistical Implementation*

#### Exercise 1: The Tukey Fence (IQR Method)
Implemented the 1.5 × IQR rule to flag univariate outliers:

```python
def flag_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (df[column] < lower_bound) | (df[column] > upper_bound)
```

**Results:**
- Outliers detected in `MedInc`: ~XXX points
- These represent primarily wealthy districts ("The Elon Musk Effect")

#### Exercise 2: Boxplot Visualization
Created boxplot to visualize how outliers pull the mean away from the median, demonstrating the fragility of the average in skewed data.

---

### **Phase 3: Algorithmic Anomaly Detection**

#### Exercise 3: Isolation Forest Implementation
Moved from univariate to **multivariate anomaly detection** using unsupervised learning:

```python
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    n_estimators=100, 
    contamination=0.05, 
    random_state=42
)

features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population']
preds = iso_forest.fit_predict(df[features])
df['outlier_iso'] = preds == -1
```

**Key Advantage:** Isolation Forest detects anomalies in the *relationship between variables* (e.g., low income + 10 bedrooms).

#### Exercise 4: Human vs. Machine Comparison
Visualized the difference between simple IQR outliers and ML-detected anomalies through scatter plots, revealing complex patterns invisible to univariate methods.

---

### **Phase 4: AI-Assisted "Tech Economist" Analysis**
🤖 *AI Usage Authorized and Required*

#### Comparative Forensics Report
Generated a comprehensive statistical comparison between the "Core Market" and "The Tail":

**Analysis Components:**
1. **Data Split:** `df_normal` vs. `df_outlier`
2. **Central Tendency:** Mean vs. Median for Income and House Value
3. **Volatility Metrics:** Standard Deviation vs. MAD (Median Absolute Deviation)
4. **Inequality Wedge:** Mean - Median gap revealing wealth concentration

**Visualization:**
- Side-by-side histograms comparing income distributions
- Blue (Normal) vs. Red (Outliers)
- Mean and Median lines overlaid for direct comparison

---

## 📈 Key Findings

### Statistical Insights
1. **The Fragile Mean:** In skewed distributions, the mean is heavily influenced by extreme values, making it unreliable for decision-making.

2. **Robust Alternatives:** The Median and MAD provide stable measures that resist contamination from outliers.

3. **Inequality Wedge:** Positive wedge (Mean > Median) in outlier groups indicates right-skewed wealth concentration—the "Pareto Principle" in action.

4. **Multivariate Complexity:** Simple univariate rules miss anomalies that only appear in the interaction between variables.

### Real-World Applications
- **PropTech Pricing:** Zillow/Opendoor must separate "core market" from luxury outliers
- **Fraud Detection:** Banks use similar techniques to flag suspicious transactions
- **Quality Control:** Manufacturing uses anomaly detection to catch defective products

---

## 🎯 Deliverables

✅ **Manual IQR Implementation** - Pure statistical code without AI assistance  
✅ **Isolation Forest Model** - Multivariate anomaly detection  
✅ **Comparative Forensics Report** - Professional statistical analysis  
✅ **Visualizations** - Four key plots demonstrating insights  
✅ **README Documentation** - Complete project documentation  

---

## 💡 The "Tech Economist" Mindset

This lab reinforces a critical lesson for the modern data economy:

> **"In the age of platforms and marketplaces, the Average is a vanity metric. Robust statistics reveal the true structure of the market."**

Traditional economics taught us to trust the mean. The tech economy—with its power laws, network effects, and long tails—demands we think in medians, quantiles, and anomaly detection.

---

## 📚 References & Resources

- Tukey, J.W. (1977). *Exploratory Data Analysis*
- Liu, F.T., Ting, K.M., Zhou, Z.H. (2008). "Isolation Forest" - *IEEE ICDM*
- Rousseeuw, P.J., Croux, C. (1993). "Alternatives to the Median Absolute Deviation"
- California Housing Dataset: `sklearn.datasets.fetch_california_housing`

---

## 🚀 How to Run This Lab

1. **Clone/Download** the repository
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Open Jupyter:** `jupyter notebook lab4_analysis.ipynb`
4. **Run cells sequentially** through all four phases
5. **Review outputs** in the `outputs/` directory

---

## 📝 Reflection

This lab bridged the gap between classical statistics and modern machine learning, demonstrating that:
- Simple rules (IQR) remain valuable for interpretability
- Complex algorithms (Isolation Forest) catch patterns humans miss
- The choice between robust and fragile statistics has real economic consequences
- AI tools augment, but don't replace, statistical reasoning

The ability to detect and characterize anomalies is foundational for any data scientist working in the modern marketplace economy.

---

## 📧 Contact

**Ian Solberg**  
ECON3916 - Spring 2026  
Babson College

---

*Last Updated: February 3, 2026*
