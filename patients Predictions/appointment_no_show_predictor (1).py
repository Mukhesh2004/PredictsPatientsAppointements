
import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def create_fake_appointment_data(filename='appointments.csv', count=1000):
    names = ['M', 'F']
    records = []
    for i in range(count):
        pid = random.randint(10000, 99999)
        aid = i + 1
        gender = random.choice(names)
        age = random.randint(5, 90)
        scholarship = random.randint(0, 1)
        hypertension = random.randint(0, 1)
        diabetes = random.randint(0, 1)
        alcoholism = random.randint(0, 1)
        handicap = random.randint(0, 1)
        sms_flag = random.randint(0, 1)
        sched_day = datetime(2025, 7, random.randint(1, 20))
        appt_day = sched_day + timedelta(days=random.randint(0, 10))
        status = random.choice(['Yes', 'No'])

        records.append([
            pid, aid, gender, age, scholarship, hypertension,
            diabetes, alcoholism, handicap, sms_flag,
            sched_day, appt_day, status
        ])

    df = pd.DataFrame(records, columns=[
        'PatientID', 'AppointmentID', 'Gender', 'Age', 'Scholarship',
        'Hypertension', 'Diabetes', 'Alcoholism', 'Handicap',
        'SMS_Received', 'ScheduledDate', 'AppointmentDate', 'NoShow'
    ])

    df.to_csv(filename, index=False)

if not os.path.exists("appointments.csv"):
    create_fake_appointment_data()

df = pd.read_csv("appointments.csv")
df['ScheduledDate'] = pd.to_datetime(df['ScheduledDate'])
df['AppointmentDate'] = pd.to_datetime(df['AppointmentDate'])
df['WaitDays'] = (df['AppointmentDate'] - df['ScheduledDate']).dt.days
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
df['NoShow'] = df['NoShow'].map({'No': 0, 'Yes': 1})

feature_columns = [
    'Gender', 'Age', 'Scholarship', 'Hypertension', 'Diabetes',
    'Alcoholism', 'Handicap', 'SMS_Received', 'WaitDays'
]

X = df[feature_columns]
y = df['NoShow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = RandomForestClassifier(n_estimators=120, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n Evalution")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

df['NoShowRisk'] = clf.predict_proba(X)[:, 1]

def suggest_action(risk):
    if risk >= 0.8:
        return "Call + Provide Transport"
    elif risk >= 0.6:
        return "Call Reminder"
    elif risk >= 0.4:
        return "Send SMS"
    else:
        return "No Intervention"

df['Action'] = df['NoShowRisk'].apply(suggest_action)
df[['PatientID', 'AppointmentID', 'NoShowRisk', 'Action']].to_csv("no_show_output.csv", index=False)

print("\n Risk analysis complete. Results saved to 'no_show_output.csv'")
plt.figure(figsize=(8, 5))
sns.countplot(x='NoShow', data=df)
plt.title("No-Show Distribution")
plt.xticks([0, 1], ['Showed Up', 'No Show'])
plt.xlabel("Appointment Status")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df[feature_columns + ['NoShow']].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
feat_importance = clf.feature_importances_
sns.barplot(x=feat_importance, y=feature_columns)
plt.title("Feature Importance from Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['NoShowRisk'], bins=20, kde=True)
plt.title("Distribution of No-Show Risk Scores")
plt.xlabel("Risk Probability")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(y='Action', data=df, order=df['Action'].value_counts().index)
plt.title("Distribution of Suggested Actions")
plt.xlabel("Number of Patients")
plt.tight_layout()
plt.show()
