# The Illusion of Growth & The Composition Effect

## Objective
Build a Python pipeline to ingest live economic data from the Federal Reserve API (FRED) to analyze wage stagnation over five decades and correct for statistical biases that can distort economic narratives. This project demonstrates how composition effects can create misleading signals in aggregate wage data, particularly during periods of structural labor market disruption.

## Methodology

### Data Acquisition & Real Wage Calculation
- **API Integration**: Connected to the Federal Reserve Economic Data (FRED) API using `fredapi` to programmatically fetch live economic time series
- **Nominal Wages**: Retrieved AHETPI (Average Hourly Earnings of Production and Nonsupervisory Employees) spanning 1964-present
- **Price Adjustment**: Fetched Consumer Price Index (CPI) data to deflate nominal wages and calculate real purchasing power
- **Real Wage Computation**: Applied the formula `Real Wage = (Nominal Wage / CPI) × 100` to reveal inflation-adjusted trends

### Anomaly Detection & Bias Correction
- **Statistical Anomaly**: Identified an unusual spike in real wages during 2020—the "Pandemic Paradox"
- **Composition Effect Hypothesis**: Recognized that when low-wage workers exit the labor force en masse, the *average* wage of remaining workers rises artificially, even if no individual worker received a raise
- **Bias Correction**: Fetched Employment Cost Index (ECIWAG) data, which holds workforce composition constant, to isolate true wage growth from compositional artifacts
- **Comparative Analysis**: Rebased both standard wage averages and ECI data to 2015=100 to directly compare biased vs. composition-adjusted measures

### Visualization
- **Tech Stack**: Python, pandas, matplotlib, seaborn
- **Multi-Series Time Series Charts**: Plotted nominal vs. real wages (1964-present) and standard wages vs. ECI (2015-present)
- **Annotation Strategy**: Used arrows and labels to highlight the 2020 divergence between biased and corrected measures

## Key Findings

### The Money Illusion (1964-Present)
- **Nominal wages** rose dramatically over 50 years (red dashed line), creating an appearance of substantial worker gains
- **Real wages** remained essentially flat (blue line), revealing that wage growth merely kept pace with inflation
- **Core Insight**: Workers experienced the "money illusion"—larger paychecks that bought roughly the same amount of goods and services

### The Pandemic Paradox (2020)
- **Artificial Spike**: Standard wage averages showed a sharp increase in 2020, suggesting workers suddenly gained significant purchasing power during an economic crisis
- **Composition Effect Revealed**: The Employment Cost Index, which controls for workforce composition, showed *no such spike*—only steady, modest growth
- **Economic Reality**: The wage "boom" was a statistical artifact. When restaurants, hotels, and retail businesses shed low-wage workers during lockdowns, the remaining workforce skewed toward higher-paid positions, mechanically inflating the average without any individual experiencing real wage gains
- **Policy Implication**: Relying on standard wage averages during the pandemic would have led to the false conclusion that labor demand surged, when in reality the labor market was in severe distress

### Methodological Contribution
This project demonstrates the critical importance of **composition-adjusted metrics** in economic analysis. Standard averages can provide deeply misleading signals during structural shocks, and analysts must use fixed-composition indices (like ECI) to separate genuine economic trends from statistical artifacts.
