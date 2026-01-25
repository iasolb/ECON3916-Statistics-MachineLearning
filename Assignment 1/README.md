# The Cost of Living Crisis: A Data-Driven Analysis
## Why the "Average" CPI Fails Students

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iasolb/ECON3916-Statistics-MachineLearning/blob/main/Assignment%201/Econ_3916_Assignment_1.ipynb)

**Author:** Ian Solberg  
**Date:** January 2026  
**Course:** ECON3916 - Statistics & Machine Learning

---

## The Problem

When policymakers and economists discuss inflation, they rely on the Consumer Price Index (CPI) as the definitive measure of price changes. However, **the CPI represents an "average" consumer basket that fundamentally misrepresents the economic reality faced by college students.** 

While the official CPI rose moderately between 2016 and 2024, students experienced dramatically different inflation rates in the categories that dominate their budgets: tuition, rent, and basic necessities. This analysis quantifies that divergence and constructs a **Student Price Index (SPI)** to reveal the hidden inflation burden on higher education.

## Methodology

This analysis employs rigorous economic index theory and data science techniques to construct a student-specific inflation measure:

### Data Sources
- **FRED API**: Official CPI and component series (tuition, rent, food away from home, cable TV) from the Federal Reserve Economic Data database
- **Custom Student Basket**: Four representative items weighted by typical student expenditure patterns

### Index Construction (Laspeyres Method)
I constructed a fixed-weight Student Price Index using the **Laspeyres price index formula**, which measures inflation by holding the base-period consumption basket constant:

```
SPI = (Σ P_current × Q_base) / (Σ P_base × Q_base) × 100
```

**Student Basket Weights:**
- Tuition: 40% (reflecting the dominant cost of education)
- Rent (1-bedroom): 30% (major fixed expense for off-campus students)
- Chipotle Burrito: 15% (proxy for food away from home)
- Coffee Beans: 15% (daily consumable staple)

### Technical Implementation
- **Language**: Python 3.12
- **Libraries**: pandas, numpy, matplotlib, seaborn, fredapi
- **Indexing**: All series rebased to January 2016 = 100 for comparability
- **Interpolation**: Linear interpolation applied to create continuous monthly SPI series from 2016-2024 anchor points

---

## Key Findings

### The Divergence Gap
**My analysis reveals a 10.4 percentage point divergence between Student Costs and National Inflation over the 2016-2024 period.**

- **Official CPI**: Rose from 100 to approximately **119.8** (19.8% cumulative inflation)
- **Student SPI**: Rose from 100 to **129.3** (29.3% cumulative inflation)

This means that while the Federal Reserve and policy makers observe inflation of roughly 20%, **students experienced nearly 50% higher inflation rates** in their actual cost of living.

### Component-Level Inflation Rates (2016-2024)

| Category | Inflation Rate | Impact on Students |
|----------|----------------|-------------------|
| **Tuition** | +28.89% | Far exceeds wage growth and official inflation targets |
| **Rent** | +50.00% | Housing costs doubled relative to some student incomes |
| **Chipotle Burrito** | +53.33% | Food away from home becoming increasingly unaffordable |
| **Coffee Beans** | +33.33% | Even basic consumables outpace CPI significantly |

### Visual Evidence
The time-series visualization clearly shows the **sustained and widening gap** between the official CPI (blue line) and the Student SPI (orange line) throughout the analysis period. The shaded area between the lines represents the cumulative "hidden inflation" that students bear but that official statistics fail to capture.

![CPI vs Student SPI Over Time](https://github.com/iasolb/ECON3916-Statistics-MachineLearning/blob/main/Assignment%201/Econ_3916_Assignment_1.ipynb)

---

## Policy Implications

This divergence has serious consequences:

1. **Financial Aid Calculations**: Federal aid formulas based on CPI systematically underestimate student need
2. **Student Loan Adequacy**: Maximum borrowing limits haven't kept pace with actual educational costs
3. **Minimum Wage Debates**: Arguments about "livable wages" miss how much faster student costs inflate
4. **Monetary Policy**: The Federal Reserve's 2% inflation target ignores localized price shocks in education and housing

---

## Technical Skills Demonstrated

- **API Integration**: Automated data retrieval from FRED using Python
- **Index Theory**: Applied Laspeyres methodology for economic measurement
- **Data Wrangling**: Time-series manipulation, rebasing, and interpolation with pandas
- **Statistical Visualization**: Clear communication of trends using matplotlib and seaborn
- **Economic Analysis**: Translated raw price data into actionable policy insights

---

## Repository Structure

```
Assignment 1/
├── Econ_3916_Assignment_1.ipynb    # Full analysis notebook
└── README.md                        # This file
```

---

## Future Extensions

Potential avenues for expanding this analysis:
- Incorporate regional variations (urban vs. suburban college towns)
- Add healthcare costs and textbook prices to the student basket
- Compare public vs. private institution inflation rates
- Develop predictive models for future student cost trajectories
- Analyze correlation with student debt default rates

---

## Conclusion

The official CPI, while valuable for broad economic policy, **masks the true inflation experienced by students**. By constructing a student-specific price index grounded in Laspeyres methodology and real FRED data, this analysis quantifies a **10.4 percentage point gap** that has profound implications for affordability, debt, and educational access.

**For recruiters:** This project showcases my ability to identify real-world economic problems, apply rigorous quantitative methods, and communicate findings clearly—skills directly applicable to data science roles in finance, policy, and research.

---

## Contact

**Ian Solberg**  
GitHub: [@iasolb](https://github.com/iasolb)  
LinkedIn: [Connect with me](https://linkedin.com/in/yourprofile)

---

*This analysis was completed as part of ECON3916: Statistics & Machine Learning. All code and visualizations are available in the accompanying Jupyter notebook.*
