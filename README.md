# 💼 Talent Success Formula Dashboard

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Demo](https://img.shields.io/badge/Live%20Demo-Streamlit%20Cloud-blueviolet)](https://talent-success-formula-dashboard.streamlit.app/)

> **Professional • Interactive • Insightful HR Analytics**  
> A modern Streamlit dashboard designed to explore and model what drives employee success.

---

## 🌐 Live Demo

👉 [Open on Streamlit Cloud](https://talent-success-formula-dashboard.streamlit.app/)

## 📘 Overview

The **Talent Success Formula Dashboard** is an interactive data analytics application built with **Streamlit**, **Plotly**, and **scikit-learn**.  
It helps HR professionals and data analysts discover the factors that contribute most to **employee success** — based on competency scores, psychometric data, PAPI, strengths, and contextual features.

This app is part of a **Talent Match Intelligence** initiative to identify success patterns and visualize workforce performance insights.

---

## ✨ Key Features

- 📊 **Interactive Dashboard** — Explore data from multiple HR sources and visualize patterns with Plotly.
- 🧩 **Automatic Data Integration** — Merges sheets from Excel: employees, competencies, strengths, psychometrics, and performance.
- 🔍 **Exploratory Data Analysis (EDA)** — Includes box plots, heatmaps, and psychometric distributions.
- 🧠 **Success Formula Modeling** — Builds predictive models using Linear Regression or Random Forest.
- 📈 **Feature Importance Visualization** — Displays top contributing features and their relative weights.
- 🧾 **Narrative Generator** — Automatically creates business storytelling insights grouped by category (Competency, Psychometric, Behavioral, Contextual).
- 💾 **Downloadable Results** — Export your Success Formula as a CSV file.

---

## 🧱 Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend UI | Streamlit |
| Visualization | Plotly Express, Matplotlib, Seaborn |
| Data Handling | Pandas, NumPy |
| Machine Learning | scikit-learn |
| Deployment | Streamlit Cloud / Local Run |
| Styling | Custom CSS Theme (Yellow Gradient Sidebar + Blue Header Banner) |

---

## 🧩 Data Requirements

The app expects a multi-sheet Excel file (`Study Case DA.xlsx`) or equivalent structure with the following sheets:

| Sheet Name | Description |
|-------------|--------------|
| `employees` | Employee master data (employee_id, grade_id, education_id, years_of_service_months) |
| `competencies_yearly` | Yearly competency assessment scores |
| `dim_competency_pillars` | Competency pillar definitions |
| `papi_scores` | PAPI (Personality and Performance Inventory) scores |
| `profiles_psych` | Psychometric data such as IQ, GTQ, DISC, etc. |
| `strengths` | Employee strength themes and ranks |
| `performance_yearly` | Yearly performance evaluations (optional) |

If the `performance_yearly` sheet is missing, the dashboard will automatically **create a proxy target** based on the average competency score per employee.

---

## 🧪 Installation & Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/azizmuzaki4/talent-success-formula-dashboard.git
cd talent-success-formula-dashboard
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate      # (on Mac/Linux)
venv\Scripts\activate         # (on Windows)
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Dashboard
```bash
streamlit run talent_success_formula_dashboard.py
```

## 🧑‍💻 Author

**Aziz Muzaki**  
📍 Bekasi, Indonesia  
📧 azizmuzaki4@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/aziz-muzaki-986a75241/  
💻 GitHub: https://github.com/azizmuzaki4

---

## 🪪 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute with attribution.

## ⭐ If you found this project helpful, please give it a star on GitHub!