# -----------------------------
# TITANIC DATA CLEANING & EDA
# -----------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (FILE NAME UPDATED)
df = pd.read_csv("Titanic-Dataset.csv")

# -----------------------------
# BASIC INFO
# -----------------------------
print("First 5 Rows:\n")
print(df.head())

print("\nDataset Info:\n")
print(df.info())

print("\nMissing Values:\n")
print(df.isnull().sum())

# -----------------------------
# DATA CLEANING
# -----------------------------
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

if 'Cabin' in df.columns:
    df.drop(columns=['Cabin'], inplace=True)

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())

# -----------------------------
# STATISTICS
# -----------------------------
print("\nStatistical Summary:\n")
print(df.describe())

# -----------------------------
# EDA GRAPHS
# -----------------------------

# Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# Gender vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Gender vs Survival")
plt.show()

# Class vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Passenger Class vs Survival")
plt.show()

# Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# Fare Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title("Fare Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

print("\nEDA Completed Successfully!")

input("Press Enter to exit...")