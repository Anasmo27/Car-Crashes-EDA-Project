# ================================
# EDA on car_crashes dataset
# ================================

import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
crashes = sns.load_dataset("car_crashes")

# ================================
# Basic exploration
# ================================
print("First 5 rows of the dataset:")
print(crashes.head())  # Show first 5 rows

print("\nDataset info:")
print(crashes.info())  # Show data types and missing values

print("\nStatistical summary:")
print(crashes.describe())  # Summary statistics for numerical columns

# ================================
# Univariate Analysis (single feature)
# ================================

# Histogram of total accidents
crashes['total'].hist(bins=15)
plt.title("Distribution of Total Car Crashes")
plt.xlabel("Total crashes per billion miles")
plt.ylabel("Count")
plt.show()

# ================================
# Categorical analysis
# ================================
# Compare insurance premiums across states
sns.boxplot(x="ins_premium", data=crashes)

plt.title("Boxplot of Insurance Premiums")
plt.show()

# ================================
# Bivariate Analysis (relationships)
# ================================
# Scatterplot between total crashes and speeding
sns.scatterplot(x="speeding", y="total", data=crashes)
plt.title("Total Crashes vs Speeding")
plt.show()

# ================================
# Correlation heatmap
# ================================
corr = crashes.corr(numeric_only=True)  # get only numerical columns
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Car Crashes Dataset")
plt.show()
