# employee_burnout_ml.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

# 1. Load dataset
df = pd.read_csv("file.csv")   
print("Initial Data Shape:", df.shape)

# 2. Encode categorical variables (without overwriting original)
categorical_cols = ["Gender", "Country", "JobRole", "Department", 
                    "RemoteWork", "HasMentalHealthSupport", "HasTherapyAccess", "SalaryRange"]

encoders = {}  # store encoders for later mapping if needed

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col + "_Encoded"] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

# 3. Define features and target for BurnoutRisk prediction
features = ['Age','YearsAtCompany','WorkHoursPerWeek','StressLevel','JobSatisfaction',
            'ProductivityScore','SleepHours','PhysicalActivityHrs','CommuteTime',
            'ManagerSupportScore','WorkLifeBalanceScore','CareerGrowthScore','TeamSize']

# Use encoded columns where needed
features_encoded = features + [col + "_Encoded" for col in categorical_cols if col + "_Encoded" in df.columns]

X = df[features_encoded]
y = df['BurnoutRisk']

# 4. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Classification Model (Random Forest)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_scaled, y_train)

# Predict BurnoutRisk
df['PredictedBurnoutRisk'] = clf.predict(scaler.transform(X))

# 7. Clustering (KMeans) for wellness segmentation
clustering_features = ['WorkHoursPerWeek','StressLevel','JobSatisfaction',
                       'SleepHours','WorkLifeBalanceScore','ManagerSupportScore']

X_cluster = df[clustering_features].fillna(0)
X_cluster_scaled = scaler.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['ClusterGroup'] = kmeans.fit_predict(X_cluster_scaled)

# 8. Save enriched dataset (with both text + encoded columns)
df.to_csv("employee_output.csv", index=False)

print(" Processing complete! Enriched file saved as employee_output.csv")
