#  Housing Price Prediction System

An end-to-end Machine Learning web application built using **Python**, **Streamlit**, **scikit-learn**, and **statsmodels**. This project estimates the market selling price of residential houses based on their physical dimensions, layout, location advantages, and modern amenities.

Built specifically as a college Machine Learning project, this application showcases the complete academic regression lifecycle: from Exploratory Data Analysis (EDA) and dummy encoding to **Multiple Linear Regression**, **Recursive Feature Elimination (RFE)**, and **Backward Elimination using statistical *p*-values**.

---

##  Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Machine Learning Workflow](#machine-learning-workflow)
4. [Feature Selection Techniques](#feature-selection-techniques)
5. [Application Features & Pages](#application-features--pages)
6. [Technology Stack](#technology-stack)
7. [Project Structure](#project-structure)
8. [Installation & Setup Instructions](#installation--setup-instructions)
9. [Mathematical & Algorithmic Foundations](#mathematical--algorithmic-foundations)
10. [Academic Evaluation & Results](#academic-evaluation--results)

---

## 1. Project Overview

Predicting property prices is a fundamental problem in real estate economics and predictive analytics. Property valuations are typically driven by a mix of continuous measurements (square footage), discrete counts (bedrooms, bathrooms, parking), and binary or categorical conveniences (main road proximity, air conditioning, basement, furnishing status).

This system provides:
- A modern, beginner-friendly **Streamlit web dashboard**.
- Rigorous statistical modeling implementing **Multiple Linear Regression**.
- Two prominent feature selection algorithms: **Recursive Feature Elimination (RFE)** and **Backward Elimination with *p*-values ($\alpha = 0.05$)**.
- Interactive data visualizations with **Matplotlib** and **Seaborn**.
- Zero hardcoded estimates: all predictions and diagnostic metrics are dynamically calculated from the trained model and test split.

---

## 2. Dataset Description

The system uses the benchmark `Housing.csv` dataset containing **545 verified residential property records** and **13 attributes**:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `price` | Numerical (Integer) | **Target Variable**: Actual selling price of the house (in ₹) |
| `area` | Numerical (Integer) | Total lot / carpet area in square feet |
| `bedrooms` | Numerical (Integer) | Number of bedrooms (range: 1 to 6) |
| `bathrooms` | Numerical (Integer) | Number of bathrooms (range: 1 to 4) |
| `stories` | Numerical (Integer) | Number of building levels / floors (range: 1 to 4) |
| `mainroad` | Categorical (Binary) | Direct connection to the main arterial road (`yes` / `no`) |
| `guestroom` | Categorical (Binary) | Presence of a dedicated guest bedroom (`yes` / `no`) |
| `basement` | Categorical (Binary) | Usable basement foundation (`yes` / `no`) |
| `hotwaterheating`| Categorical (Binary) | Dedicated hot water heating system / geyser (`yes` / `no`) |
| `airconditioning`| Categorical (Binary) | Central or split air conditioning installed (`yes` / `no`) |
| `parking` | Numerical (Integer) | Number of private car parking spaces (range: 0 to 3) |
| `prefarea` | Categorical (Binary) | Located in a prime / preferred locality (`yes` / `no`) |
| `furnishingstatus`| Categorical (Nominal) | Interior furnishing: `furnished`, `semi-furnished`, `unfurnished` |

**Data Integrity Verification:**
- Total records: **545**
- Missing or null values: **0**
- Duplicate records: **0**
- Average property price: **₹ 47,66,729 (₹ 47.67 Lakhs)**
- Price range: **₹ 17,50,000 to ₹ 1,33,00,000 (₹ 17.5 Lakhs to ₹ 1.33 Crores)**

---

## 3. Machine Learning Workflow

The pipeline is implemented cleanly across `preprocessing.py` and `model.py` to ensure consistency and prevent data leakage:

```
Housing.csv
    │
    ▼
[Data Validation & Loading]
    │
    ▼
[Preprocessing & Categorical Encoding]
    ├── Binary Mapping (yes/no -> 1/0)
    └── Dummy Encoding for furnishingstatus (baseline: furnished)
    │
    ▼
[Train-Test Split] (80% Train, 20% Test, random_state=42)
    │
    ├─── Baseline Model (Multiple Linear Regression on all 13 features)
    │
    ├─── Feature Selection 1: Recursive Feature Elimination (RFE) via scikit-learn
    │
    └─── Feature Selection 2: Backward Elimination via statsmodels OLS (p ≤ 0.05)
            │
            ▼
     [Final Production Model]
            │
            ├── Model Evaluation (R², Adj. R², MAE, MSE, RMSE)
            ├── Residual & Diagnostic Plots
            └── Cached for Real-Time Streamlit Prediction
```

---

## 4. Feature Selection Techniques

### A. Backward Elimination (OLS Regression with *p*-values)
Backward Elimination begins with all features included in an Ordinary Least Squares (OLS) regression model (`statsmodels.api.OLS` with an explicit constant intercept).
- At each iteration, the hypothesis test for each coefficient $H_0: \beta_j = 0$ is evaluated.
- The predictor exhibiting the highest *p*-value above the significance threshold $\alpha = 0.05$ is eliminated.
- The process repeats until all remaining predictors are statistically significant ($p \le 0.05$).

**Elimination Iterations on Housing Dataset:**
1. **Step 1:** `bedrooms` ($p = 0.3090 > 0.05$) &rarr; **Eliminated**
2. **Step 2:** `furnishingstatus_semi-furnished` ($p = 0.3101 > 0.05$) &rarr; **Eliminated**
3. **Step 3:** `guestroom` ($p = 0.1057 > 0.05$) &rarr; **Eliminated**
4. **Step 4:** All remaining 10 predictors have $p \le 0.05$ &rarr; **Termination Reached**

**Selected Features (10):**
`area`, `bathrooms`, `stories`, `mainroad`, `basement`, `hotwaterheating`, `airconditioning`, `parking`, `prefarea`, `furnishingstatus_unfurnished`.

### B. Recursive Feature Elimination (RFE)
Using `sklearn.feature_selection.RFE` with a `LinearRegression` estimator:
- Features are scaled using `StandardScaler` (fitted solely on the training split) so that large numerical dimensions like `area` are not penalized relative to binary flags.
- Prunes the least impactful features recursively based on standardized coefficient magnitudes.
- When selecting 10 features, **RFE converges on the identical 10 features selected by Backward Elimination**, providing strong mutual validation.

---

## 5. Application Features & Pages

The Streamlit interface is organized into 4 distinct, beginner-friendly pages:

### Page 1:  Home / Dashboard
- Welcoming overview of the project and real-estate valuation principles.
- 4 Key Metric Cards: Total Properties, Average Price, Lowest Price, Highest Price.
- Plain-English explanation of how supervised regression estimates property values.
- Quick "Key Valuation Drivers" chips.
- Prominent **"Estimate House Price"** call-to-action button that navigates directly to the valuation page.

### Page 2:  House Price Prediction (Core Feature)
- Clean, responsive input form grouped into 4 intuitive cards:
  1. **Property Size & Layout:** Area in sq. ft., Bedrooms, Bathrooms, Stories.
  2. **Parking & Street Access:** Parking capacity (0–3 cars), Main road connection (Yes/No).
  3. **Amenities & Location:** Prime area (Yes/No), Air conditioning (Yes/No), Hot water heating (Yes/No), Basement (Yes/No), Guest room (Yes/No).
  4. **Furnishing Status:** Furnished, Semi-furnished, Unfurnished.
- Prominent **"Predict House Price"** action button.
- Generates an instant valuation card formatted in Indian Rupees:
  - Example: **₹ 76,23,270 (76.23 Lakhs)**.
  - Realistic valuation range ($\pm 1\ \text{RMSE}$).
  - Required academic disclaimer text.
  - Full property summary chips for easy review.

### Page 3:  Dataset Insights
- Collapsible raw dataset preview with summary statistics.
- 5 Polished Visualizations:
  1. **Price Distribution:** Histogram with Kernel Density Estimate (KDE), mean and median lines.
  2. **Area vs. Selling Price:** Scatter plot with OLS trendline showing positive correlation.
  3. **Price by Bedroom Count:** Box plot illustrating price distributions across 1 to 6 bedrooms.
  4. **Furnishing Status Impact:** Categorical bar chart comparing average values for furnished, semi-furnished, and unfurnished homes.
  5. **Correlation Heatmap:** Matrix displaying pairwise linear correlation among numerical variables.

### Page 4:  Model Performance & Academic Analysis
- Regression evaluation metrics: **$R^2$ Score, Adjusted $R^2$, MAE, RMSE**.
- Plain-English guide explaining what each metric measures.
- **Actual vs. Predicted Scatter Plot** with ideal 45° reference line.
- **Residual Plot** to verify homoscedasticity and absence of non-linear patterns.
- Comparative analysis table: Baseline (13 features) vs. RFE (10 features) vs. Backward Elimination (10 features).
- Step-by-step Backward Elimination iteration log.
- Full `statsmodels.api.OLS` statistical summary table.
- Feature coefficient impact table showing the exact Rupee contribution of each property feature.

---

## 6. Technology Stack

- **Programming Language:** Python 3.10+ / 3.11+
- **Frontend Framework:** Streamlit (>= 1.30.0)
- **Data Manipulation:** pandas (>= 2.0.0), NumPy (>= 1.24.0)
- **Machine Learning & RFE:** scikit-learn (>= 1.3.0)
- **Statistical Modeling & OLS:** statsmodels (>= 0.14.0)
- **Visualizations:** Matplotlib (>= 3.7.0), Seaborn (>= 0.12.0)

---

## 7. Project Structure

```
housing-price-prediction/
│
├── app.py              # Main Streamlit web application & UI router
├── model.py            # Regression pipeline, RFE, Backward Elimination & metrics
├── preprocessing.py    # Data loading, validation, dummy encoding & currency helpers
├── Housing.csv          # Real estate dataset (545 rows, 13 columns)
├── requirements.txt    # Project dependencies
└── README.md           # Comprehensive project documentation
```

---

## 8. Installation & Setup Instructions

Follow these steps to run the application on your Windows laptop.

### Step 1: Open Windows PowerShell
Press `Win + X` and select **Windows PowerShell** (or open Windows Terminal / Command Prompt).

### Step 2: Navigate to the Project Directory
```powershell
cd "d:\housing project"
```

### Step 3: (Recommended) Create and Activate a Virtual Environment
```powershell
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once).*

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 5: Run the Streamlit Application
```powershell
streamlit run app.py
```

The application will launch automatically in your default web browser at:
```
http://localhost:8501
```

To stop the application at any time, press `Ctrl + C` in the PowerShell window.

---

## 9. Mathematical & Algorithmic Foundations

### Multiple Linear Regression Model
The multiple linear regression equation relates the dependent target variable $Y$ (House Price) to $p$ independent predictors $X_1, X_2, \dots, X_p$:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \epsilon$$

Where:
- $\beta_0$: Intercept term (baseline price when all predictors are zero).
- $\beta_j$: Regression coefficient representing the change in price per unit change in $X_j$, holding all other features constant.
- $\epsilon$: Random error term, assumed $\epsilon \sim \mathcal{N}(0, \sigma^2)$.

### Ordinary Least Squares (OLS) Estimation
The vector of coefficients $\hat{\boldsymbol{\beta}}$ is determined by minimizing the Sum of Squared Residuals (SSR):

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### Hypothesis Testing & P-Values in Backward Elimination
For each predictor $X_j$, the two-tailed hypothesis test evaluates whether $X_j$ has a non-zero linear relationship with $Y$:

$$H_0: \beta_j = 0 \quad \text{vs.} \quad H_1: \beta_j \neq 0$$

The test statistic is:

$$t = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)}$$

If $p\text{-value} > \alpha$ (where $\alpha = 0.05$), we fail to reject the null hypothesis, indicating that feature $X_j$ is not statistically significant after controlling for other variables, and it is safely removed.

---

## 10. Academic Evaluation & Results

Evaluated on the unseen **20% test split (109 properties)**:

| Metric | Baseline (All 13 Features) | Backward Elimination (10 Features) |
| :--- | :--- | :--- |
| **Test $R^2$ Score** | `0.6529` (65.3%) | `0.6480` (64.8%) |
| **Adjusted $R^2$ Score** | `0.6053` | `0.6121` *(Improved parsimony)* |
| **Mean Absolute Error (MAE)** | `₹ 9,70,043` | `₹ 9,81,869` |
| **Root Mean Squared Error (RMSE)**| `₹ 13,24,507` | `₹ 13,33,823` |
| **Model Complexity** | 13 features | **10 features** *(Pruned 3 redundant variables)* |

### Key Real-World Coefficient Impacts:
- **Bathrooms:** $+₹ 11,22,000$ per additional bathroom
- **Air Conditioning:** $+₹ 8,14,000$ for AC installation
- **Hot Water Heating:** $+₹ 6,81,000$ for geyser/heating system
- **Preferred Area:** $+₹ 6,49,600$ premium for prime location
- **Basement:** $+₹ 4,81,900$ for usable basement
- **Stories:** $+₹ 4,45,100$ per additional floor level
- **Main Road Access:** $+₹ 3,65,400$ for main road connectivity
- **Parking:** $+₹ 2,25,200$ per private vehicle parking spot
- **Area:** $+₹ 244$ per square foot
- **Unfurnished Condition:** $-₹ 3,35,600$ discount relative to furnished properties

---

##  Academic Presentation Guide
When presenting this project to evaluators:
1. **Highlight Convergence:** Emphasize that statistical Backward Elimination and standardized RFE both independently selected the exact same set of 10 features, proving the robustness of the feature selection.
2. **Explain Dropped Variables:** Explain why `bedrooms` was eliminated—its explanatory power is already captured by total `area` and `bathrooms` (multicollinearity).
3. **Demonstrate Parsimony:** Point out that removing 3 redundant features preserved model accuracy ($R^2 \approx 65\%$) while reducing complexity and avoiding overfitting.
4. **Real-Time Valuation:** Use the prediction form on Page 2 to demonstrate dynamic, non-hardcoded price calculation with user-entered values.

