# Employee Burnout Analysis & Visualization Dashboard

##  Project Overview

This project analyzes employee burnout risk, wellness patterns, and lifestyle factors using **Machine Learning** and **Power BI visualizations**. The goal is to help HR and management identify high-risk employees, understand stress drivers, and improve workforce wellbeing.

The project consists of two main components:
- **Machine Learning & Data Processing**
- **Interactive Power BI Dashboard** for insights

##  Dataset

The dataset contains comprehensive employee information including:

- **Demographics**: Age, Gender, Country
- **Work Metrics**: JobRole, Department, YearsAtCompany, WorkHoursPerWeek, RemoteWork
- **Wellbeing Metrics**: BurnoutLevel, JobSatisfaction, StressLevel, ProductivityScore, SleepHours, PhysicalActivityHrs, CommuteTime
- **Support Metrics**: HasMentalHealthSupport, ManagerSupportScore, HasTherapyAccess, MentalHealthDaysOff
- **Career & Compensation**: SalaryRange, WorkLifeBalanceScore, TeamSize, CareerGrowthScore

The raw dataset is enhanced with **predicted burnout risk** and **wellness clusters** using machine learning.

##  Phase 1: Machine Learning & Data Enrichment

The Python script `employee_burnout_ml.py` performs the following steps:

### 1. Data Loading
```python
df = pd.read_csv("file.csv")
```
Loads raw employee data into a DataFrame.

### 2. Encoding Categorical Variables
Uses `LabelEncoder()` on columns like:
- Gender, Country, JobRole, Department
- RemoteWork, HasMentalHealthSupport, HasTherapyAccess, SalaryRange

Converts text categories to numeric values for ML modeling without overwriting original columns.

### 3. Feature Selection
Selected features for Burnout Risk prediction include:
- Age, YearsAtCompany, WorkHoursPerWeek, StressLevel, JobSatisfaction
- ProductivityScore, SleepHours, PhysicalActivityHrs, CommuteTime
- ManagerSupportScore, WorkLifeBalanceScore, CareerGrowthScore, TeamSize
- Plus encoded categorical variables

### 4. Train-Test Split & Scaling
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
Splits data into training/test sets and standardizes numeric features.

### 5. Random Forest Classification
```python
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)
df['PredictedBurnoutRisk'] = clf.predict(scaler.transform(X))
```
Predicts Burnout Risk for all employees (Low, Medium, High).

### 6. KMeans Clustering
```python
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['ClusterGroup'] = kmeans.fit_predict(X_cluster_scaled)
```
Groups employees into 3 wellness clusters based on:
- Work hours, stress, job satisfaction
- Sleep, work-life balance, manager support

**Cluster interpretation** (example):
- Cluster 0 → Low Risk / High Wellness
- Cluster 1 → Medium Risk
- Cluster 2 → High Risk

### 7. Save Enriched Dataset
```python
df.to_csv("employee_output.csv", index=False)
```
Saves the dataset with original columns, encoded columns, predicted burnout risk, and cluster groups.

##  Phase 2: Power BI Dashboard

An interactive 3-page Power BI dashboard provides actionable insights:

###  Page 1 – Overview
**Goal**: Company-wide snapshot of employee wellness  
**Visuals**:
- KPI Cards: Total Employees, Average Burnout, % High Risk, Avg Work-Life Balance
- Pie Chart: Burnout Risk Distribution (Low/Medium/High)
- Stacked Bar Chart: Burnout Levels by Department
- Optional Map: Geographic Burnout Distribution

###  Page 2 – Department & Manager Deep Dive
**Goal**: Identify stress patterns across teams and departments  
**Visuals**:
- Cluster Group Column Chart: Employee distribution by wellness cluster
- Heatmap (Matrix): Stress Level by JobRole and Department
- Scatter Plot: Manager Support vs Burnout Level (bubble size = WorkHoursPerWeek)

###  Page 3 – Lifestyle & Remote Work
**Goal**: Explore correlations between lifestyle factors and productivity  
**Visuals**:
- Line Chart: Sleep Hours vs Average Productivity
- Clustered Column Chart: Remote Work vs Job Satisfaction
- Stacked Bar Chart: Physical Activity vs Burnout Level
- Optional Map: Employee distribution by Country

###  Interactivity & Polishing
- **Filters/Slicers**: Department, JobRole, Country, RemoteWork, Gender
- **Drill-through**: View individual employee details from department-level charts
- **Color Coding**:
  - Low Risk → Green
  - Medium Risk → Orange
  - High Risk → Red
- Professional titles and subtitles for stakeholder presentation

##  Key Skills Demonstrated

- **Data Preprocessing**: Handling missing values, encoding categorical data
- **Machine Learning**: Random Forest classification, KMeans clustering
- **Data Visualization**: Power BI dashboard with interactive charts, heatmaps, scatter plots, KPIs, and maps
- **Business Insight**: Translating raw HR data into actionable insights for employee wellness and burnout prevention

## 📂 Project Structure

```
employee-burnout-analysis/
├── README.md
├── data/
│   ├── file.csv                    # Raw HR dataset
│   └── employee_output.csv         # ML-enriched dataset
├── scripts/
│   └── employee_burnout_ml.py      # ML processing and enrichment script
└── dashboard/
    └── Employee_Burnout.pbix       # Power BI dashboard
```

##  Getting Started

### Prerequisites
- Python 3.7+
- Required Python libraries:
  ```
  pandas
  scikit-learn
  numpy
  ```
- Power BI Desktop (for dashboard viewing/editing)

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/employee-burnout-analysis.git
   cd employee-burnout-analysis
   ```

2. Install required Python packages:
   ```bash
   pip install pandas scikit-learn numpy
   ```

3. Run the ML processing script:
   ```bash
   python scripts/employee_burnout_ml.py
   ```

4. Open the Power BI dashboard:
   ```
   Open dashboard/Employee_Burnout.pbix in Power BI Desktop
   ```

##  Usage

1. **Data Processing**: Run the Python script to generate ML predictions and clusters
2. **Dashboard Analysis**: Use the Power BI dashboard to explore insights and identify at-risk employees
3. **Action Planning**: Use findings to develop targeted wellness interventions


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
