# Linear Regression Analysis - Comprehensive Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Linear Regression Equation](#linear-regression-equation)
3. [Assumptions on Linear Regression](#assumptions-on-linear-regression)
4. [Finding Coefficients](#finding-coefficients)
5. [Metrics](#metrics)
6. [Housing Dataset Example](#housing-dataset-example)
7. [Statistical Inference](#statistical-inference)
8. [Student Exam Score Example](#student-exam-score-example)

---

## Introduction

Linear regression is a fundamental statistical method used to model the relationship between a dependent variable and one or more independent variables. This comprehensive guide covers all theoretical foundations, practical implementations, and statistical inference concepts essential for mastering linear regression analysis.

---

## Linear Regression Equation

### Core Equation
```
y = β₀ + β₁x + ε
```

**Alternative notation:**
```
y = P₀ + P₁x + ε
```

**Where:**
- `y` = Dependent variable (target)
- `β₀, P₀` = Intercept coefficient
- `β₁, P₁` = Slope coefficient  
- `x` = Independent variable (feature)
- `ε` = Error term

**Predicted value:**
```
ŷ = β₀ + β₁x
```

---

## Assumptions on Linear Regression

Linear regression relies on four critical assumptions:

### 1. Linear Relationship between x and y
- There exists a linear relationship between independent and dependent variables
- The relationship can be represented by a straight line

### 2. Error terms are normally distributed
- Error terms (residuals) follow a normal distribution
- Validated using Q-Q plots and distribution plots

### 3. Error terms are independent of each other
- Error terms are independent of each other
- No autocorrelation between residuals
- Checked using residual plots over time

### 4. Error terms have constant variance
- Error terms have constant variance across all levels of independent variables (homoscedasticity)
- Validated using residuals vs fitted values plots

---

## Finding Coefficients

### 1. Ordinary Least Squares (OLS)
```
β₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
β₀ = ȳ - β₁x̄
```

**Where:**
- `x̄` = Mean value of x
- `ȳ` = Mean value of y

### 2. Normal Equation
```
β = (XᵀX)⁻¹Xᵀy
```
Matrix form for multiple linear regression

### 3. Gradient Descent
```
β₀ = β₀ - α(∂J/∂β₀)
β₁ = β₁ - α(∂J/∂β₁)
```
Iterative optimization approach

---

## Metrics

### Error-Based Metrics

**Residual Sum of Squares (RSS)**
```
RSS = Σ(yᵢ - ŷᵢ)²
```
- Measures unexplained variation by the model

**Mean Squared Error (MSE)**
```
MSE = Σ(yᵢ - ŷᵢ)² / n
```

**Residual Standard Error (RSE)**
```
RSE = √[Σ(yᵢ - ŷᵢ)² / (n - k - 1)]
```
- Where k = number of predictors
- RSE is standard deviation

**Root Mean Squared Error (RMSE)**
```
RMSE = √[Σ(yᵢ - ŷᵢ)² / n]
```

**Mean Absolute Error (MAE)**
```
MAE = Σ|yᵢ - ŷᵢ| / n
```

### Variance-Based Metrics

**Total Sum of Squares (TSS)**
```
TSS = Σ(yᵢ - ȳ)²
```
- ȳ = mean of actual values
- Measures total variation in data

**R² Score**
```
R² = 1 - (RSS/TSS)
```
- Proportion of variation explained by the model

**Adjusted R²**
```
Adjusted R² = 1 - [(1-R²)(n-1)/(n-p-1)]
```
- Modified version of R² that penalizes the model for adding too many variables that don't really help explain the data
- While R² the same, always increases or stays the same when you add more predictors, even if those predictors are less useful
- Adjusted R² can decrease if those added variables don't improve the model enough
- It only goes up if the new feature significantly improves the model

### Multicollinearity Detection

**Variance Inflation Factor (VIF)**
```
VIF = 1/(1 - Rᵢ²)
```
- Measures multicollinearity between features

### Statistical Metrics

**Standard Error**
```
SE = Residual Standard Error
```

**t-value**
```
t = Coefficient (β̂) / Standard Error of β̂
```

**p-value**
```
P = 2 × P(T > |t|)
```

---

## Housing Dataset Example

This practical example demonstrates how linear regression output should be interpreted using real statistical measures.

### Output of Regression
- **R-squared = 0.831**
- **Adj. R-squared = 0.827**  
- **F-statistic = 174.3**

### Variable Analysis Table
| Variable | Coeff | Std Err | t-value | P>|t| | [0.025 | 0.975] |
|----------|-------|---------|---------|-------|--------|---------|
| temp | 0.4725 | 0.037 | 12.832 | 0.000 | 0.400 | 0.545 |
| season_summer | 0.106 | 0.022 | 4.853 | 0.000 | 0.063 | 0.149 |
| working_day | 0.0178 | 0.009 | 1.946 | 0.052 | -0.000 | 0.036 |

### Key Observations from Housing Dataset
1. **Temperature coefficient (0.4725)** is highly significant with very low p-value (0.000)
2. **Standard Error (0.037)** indicates good reliability of the temperature coefficient
3. **t-value (12.832)** shows strong statistical significance
4. **95% Confidence Interval [0.400, 0.545]** provides range of plausible coefficient values
5. **Working day variable** has p-value (0.052) close to significance threshold (0.05)

This example sets the foundation for understanding the statistical inference concepts that follow.

---

## Statistical Inference

### Why Coefficients Cannot Be 100% Correct

In linear regression, the coefficients β₀ and β₁ are calculated for the test data, so it cannot be 100% correct value. To determine how much it is correct, we use the following parameters:
- Standard Error
- t-value  
- p-value
- Confidence interval (0.025-0.975, 95% confidence interval)

### Standard Error

**Example: Housing Dataset (temp variable)**
```
temp coefficient = 0.4725
Standard Error = 0.037
```

**Interpretation:**
Standard Error tells us, on average, the coefficient value of temp will change between ±0.037.

It can be between:
- (0.4725 + 0.037) = 0.5095
- (0.4725 - 0.037) = 0.4355

**Rules:**
- If SE is small → coefficient is more reliable
- If SE is large → the value is less reliable

### t-value

**Formula:**
```
t = Coefficient (β̂) / Standard Error of β̂
```

**Example: temp (from Housing Dataset)**
```
t = 0.4725 / 0.037 = 12.832
```

We use t-value rather than z-value because SE is calculated for the test data, not for the actual data.

**Interpretation:**
- The larger the t-value, the more significant the coefficient is
- t = 12.832 means that our estimated coefficient (0.4725) is 12.832 standard errors away from zero

### p-value

P-value is all about Hypothesis Testing.

**Hypothesis Testing:**
- **H₀ (Null Hypothesis)**: β = 0
- **H₁ (Alternative Hypothesis)**: β ≠ 0

**In Linear Regression:**
- **Null Hypothesis**: The coefficients β₀ and β₁ are equal to zero
- **Alternative Hypothesis**: The coefficients β₀ and β₁ are not equal to zero

If Null Hypothesis is true, then it means the coefficients are not having any significant value for predicting the y-value.

**Formula:**
```
P = 2 × P(T > |t|)
```

**Example (using Housing Dataset):**
P(T > 12.832) tells us: Assuming the null hypothesis is true (meaning that temperature has no real effect on the outcome you are measuring), what is the probability that a t-value we would get from a different random sample would be even larger than the 12.832 that I got for my specific sample?

**Decision Rules:**
- If p-value < α → reject the null hypothesis
- If p-value ≥ α → you fail to reject the null hypothesis

**Significance Level:**
α = 0.05 (we accept 5% chance of making a Type I error - rejecting a true null hypothesis)

---

## Student Exam Score Example

**Data:**
- Scores: [40, 50, 60, 70, 80]
- Hours: [4, 5, 6, 7, 8]
- Mean score: ȳ = (40+50+60+70+80)/5 = 60

**Step 1: Calculate TSS (Total Variation)**
```
TSS = (40-60)² + (50-60)² + (60-60)² + (70-60)² + (80-60)²
TSS = 400 + 100 + 0 + 100 + 400 = 1000
```

**Step 2: Predicted Scores**
Predicted scores: [38, 46, 54, 62, 75]

**Step 3: Calculate RSS**
```
RSS = (40-38)² + (50-46)² + (60-54)² + (70-62)² + (80-75)²
RSS = 4 + 16 + 36 + 64 + 25 = 145
```

**Step 4: Calculate R²**
```
R² = 1 - (RSS/TSS) = 1 - (145/1000) = 1 - 0.145 = 0.855 = 85.5%
```

**Interpretation:**
Your model accounts for most 85.5% of the differences between individual scores.

**Meaning:**
- Each student has their own exam score (like 40, 50, 60, etc.) - these are the individual scores
- These scores vary from average score (mean = 60)
- The difference between the individual score and average represents the variation in the data
- Your model explains 85% of these differences
- 85% of the differences are predicted correctly
- Actual variation = 1000 (TSS)
- Predicted variation = 855 (explained by model)

This is what R² is telling: how much difference your model was able to predict.

---
