# 🛢️ US Gas Production Efficiency Tracker

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://gas-appuction-efficiency-bdqkqqrdudbghfmwca6ndt.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange)

An end-to-end data analytics project that monitors natural gas production efficiency, detects operational waste (flaring/venting), and forecasts future trends using Machine Learning.

## 🚀 Live Demo
**Click here to view the interactive dashboard:**
[**👉 Open Gas Efficiency Dashboard**](https://gas-appuction-efficiency-bdqkqqrdudbghfmwca6ndt.streamlit.app/)

---

## 🧐 Background & Problem Statement

### The Problem
In the Oil & Gas industry, **Flaring and Venting** (burning off or releasing natural gas) represent a significant challenge. It is not only a major environmental concern (ESG compliance) but also a direct financial loss—often referred to as **"Revenue Leakage."**

Identifying *when* and *where* these inefficiencies occur is difficult due to the massive volume of raw production reports submitted to federal agencies.

### The Objective
This project aims to transform raw government data (OGOR-B) into actionable business intelligence by:
1.  **Quantifying Waste:** calculating the exact ratio of gas sold vs. gas wasted.
2.  **Detecting Anomalies:** Using Unsupervised Learning to flag months with suspicious spikes in flaring.
3.  **Forecasting Trends:** Using Supervised Learning to predict efficiency levels for the coming month.

---

## 📊 Key Business Insights

Based on the analysis of the US Federal Oil & Gas dataset (2015-2025):

1.  **Regional Disparity:** Inefficiencies are not evenly distributed. Certain regions (e.g., *Offshore Gulf*) demonstrate high production volumes but occasional extreme spikes in waste, likely correlated with maintenance cycles or infrastructure bottlenecks.
2.  **Anomaly Patterns:** Using **Isolation Forest**, we identified specific operational periods where the "Waste Ratio" exceeded the statistical norm by significant margins. These anomalies serve as a "Target List" for operational audits.
3.  **Seasonality:** The predictive model indicates that efficiency is not random; it follows a time-series pattern that can be forecasted, allowing operators to anticipate potential waste events.

---

## 🛠️ Technical Solution

### Tech Stack
* **Language:** Python
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Isolation Forest, Random Forest Regressor)
* **Visualization:** Plotly Interactive Charts

### Methodology
1.  **Data Engineering:**
    * Cleaned raw data from the US Department of Interior (ONRR).
    * Standardized disparate disposition codes into clear categories: `Sales` (Revenue) vs. `Waste` (Loss).
    * Engineered the `Waste_Ratio` KPI feature.
2.  **Unsupervised Learning (Anomaly Detection):**
    * Implemented an **Isolation Forest** algorithm to detect data points that deviate significantly from the cluster of "Normal Operations."
3.  **Supervised Learning (Forecasting):**
    * Developed a **Random Forest Regressor** using Lag Features (t-1, t-3 months) to predict the Waste Ratio for the upcoming month.

---

## 📂 Project Structure

```text
├── app.py                     # Main Streamlit application
├── processed_gas_efficiency.csv # Preprocessed dataset used by the app
├── requirements.txt           # List of dependencies
├── notebooks/                 # Jupyter Notebooks for EDA and prototyping
│   └── analysis_model.ipynb
└── README.md                  # Project documentation
```
## ⚙️ How to Run Locally
If you wish to run this dashboard on your local machine:

1. Clone the repo
```text
git clone [https://github.com/yourusername/gas-production-efficiency.git](https://github.com/yourusername/gas-production-efficiency.git)
```
2. install dependencies
```
pip install -r requirements.txt
```
3. Run the app
```
streamlit run app.py
```

## 👤 Author

Reginaldo Ahnaf Data Analyst | UI/UX Designer

Focus: Building data-driven solutions for the Energy and Tech sectors.
