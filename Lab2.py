import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# ---------------- A1: PURCHASE DATA ----------------

purchase_data = pd.read_excel(
    r"C:\Users\ADMIN\Downloads\Lab Session Data.xlsx",
    sheet_name="Purchase data"
)

print(purchase_data.head())

features = purchase_data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
payment = purchase_data["Payment (Rs)"].values

matrix_rank = np.linalg.matrix_rank(features)
pinv_features = np.linalg.pinv(features)
product_cost = pinv_features.dot(payment)

print("Feature Matrix:\n", features)
print("Payment Vector:\n", payment)
print("Rank of Feature Matrix:", matrix_rank)
print("Estimated Product Cost:", product_cost)

# ---------------- A2: CUSTOMER CLASSIFICATION ----------------

payment_label = []
for amt in payment:
    if amt > 200:
        payment_label.append(1)
    else:
        payment_label.append(0)

payment_label = np.array(payment_label)

classifier_weight = pinv_features.dot(payment_label)
prediction_score = features.dot(classifier_weight)

final_label = []
for val in prediction_score:
    if val >= 0.5:
        final_label.append("RICH")
    else:
        final_label.append("POOR")

print("Actual Class:", payment_label)
print("Predicted Class:", final_label)

# ---------------- A3: IRCTC STOCK DATA ----------------

stock_data = pd.read_excel(
    r"C:\Users\ADMIN\Downloads\Lab Session Data.xlsx",
    sheet_name="IRCTC Stock Price"
)

price_values = stock_data["Price"].values

def manual_mean(data):
    total = 0
    for x in data:
        total += x
    return total / len(data)

def manual_variance(data):
    m = manual_mean(data)
    total = 0
    for x in data:
        total += (x - m) ** 2
    return total / len(data)

def avg_time_numpy(data):
    start = time.time()
    for _ in range(10):
        np.mean(data)
        np.var(data)
    return (time.time() - start) / 10

def avg_time_manual(data):
    start = time.time()
    for _ in range(10):
        manual_mean(data)
        manual_variance(data)
    return (time.time() - start) / 10

print("Mean (NumPy):", np.mean(price_values))
print("Variance (NumPy):", np.var(price_values))
print("Mean (Manual):", manual_mean(price_values))
print("Variance (Manual):", manual_variance(price_values))

print("Avg NumPy Time:", avg_time_numpy(price_values))
print("Avg Manual Time:", avg_time_manual(price_values))

wed_data = stock_data[stock_data["Day"] == "Wed"]
wed_mean = np.mean(wed_data["Price"])

apr_data = stock_data[stock_data["Month"] == "Apr"]
apr_mean = np.mean(apr_data["Price"])

loss_days = 0
for chg in stock_data["Chg%"]:
    if chg < 0:
        loss_days += 1

loss_probability = loss_days / len(stock_data)

wed_profit = 0
for chg in wed_data["Chg%"]:
    if chg > 0:
        wed_profit += 1

wed_profit_probability = wed_profit / len(wed_data)

print("Population Mean:", np.mean(price_values))
print("Wednesday Mean:", wed_mean)
print("April Mean:", apr_mean)
print("Probability of Loss:", loss_probability)
print("Probability of Profit on Wednesday:", wed_profit_probability)

plt.scatter(stock_data["Day"], stock_data["Chg%"])
plt.xlabel("Day")
plt.ylabel("Change %")
plt.show()

# ---------------- A4: THYROID DATA ----------------

thyroid = pd.read_excel(
    r"C:\Users\Tharun Kumar\Downloads\Lab Session Data.xlsx",
    sheet_name="thyroid0387_UCI"
)

print(thyroid.head())
print(thyroid.info())

num_columns = thyroid.select_dtypes(include=[np.number]).columns
cat_columns = thyroid.select_dtypes(exclude=[np.number]).columns

print("Numeric Columns:", num_columns)
print("Categorical Columns:", cat_columns)

missing_summary = thyroid.isnull().sum()
print("Missing Values:\n", missing_summary)

def iqr_outliers(series):
    series = series.dropna()
    q1 = np.percentile(series, 25)
    q3 = np.percentile(series, 75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return len(series[(series < low) | (series > high)])

for col in num_columns:
    print(col, "Outliers:", iqr_outliers(thyroid[col]))

print("Mean:\n", thyroid[num_columns].mean())
print("Variance:\n", thyroid[num_columns].var())

# ---------------- A5: JC & SMC ----------------

binary_cols = thyroid.columns[thyroid.isin(['t', 'f']).all()]
binary_data = thyroid[binary_cols].replace({'t': 1, 'f': 0})

v1 = binary_data.iloc[0].values
v2 = binary_data.iloc[1].values

f11 = f10 = f01 = f00 = 0

for a, b in zip(v1, v2):
    if a == 1 and b == 1:
        f11 += 1
    elif a == 1 and b == 0:
        f10 += 1
    elif a == 0 and b == 1:
        f01 += 1
    else:
        f00 += 1

jc = f11 / (f11 + f10 + f01)
smc = (f11 + f00) / (f11 + f10 + f01 + f00)

print("Jaccard Coefficient:", jc)
print("Simple Matching Coefficient:", smc)

# ---------------- A6: COSINE SIMILARITY ----------------

num_data = thyroid[num_columns]
vec1 = num_data.iloc[0].values
vec2 = num_data.iloc[1].values

cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
print("Cosine Similarity:", cos_sim)

# ---------------- A7: HEATMAP ----------------

subset = num_data.iloc[:20]
binary_subset = (subset > subset.mean()).astype(int)

size = len(subset)
JC = np.zeros((size, size))
SMC = np.zeros((size, size))
COS = np.zeros((size, size))

for i in range(size):
    for j in range(size):
        JC[i, j] = Jaccard = np.sum((binary_subset.iloc[i] == 1) & (binary_subset.iloc[j] == 1)) / max(
            1, np.sum(binary_subset.iloc[i] | binary_subset.iloc[j])
        )
        COS[i, j] = np.dot(subset.iloc[i], subset.iloc[j]) / (
            np.linalg.norm(subset.iloc[i]) * np.linalg.norm(subset.iloc[j])
        )

sns.heatmap(JC, cmap="Blues")
plt.title("Jaccard Heatmap")
plt.show()

sns.heatmap(COS, cmap="Reds")
plt.title("Cosine Heatmap")
plt.show()
