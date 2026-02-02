import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# ---------------- A1 : PURCHASE DATA ----------------

purchase_df = pd.read_excel(
    r"C:\Users\ADMIN\Downloads\Lab Session Data.xlsx",
    sheet_name="Purchase data"
)

print(purchase_df.head())

X_mat = purchase_df.iloc[:, 1:4].values
y_pay = purchase_df.iloc[:, 4].values

rank_val = np.linalg.matrix_rank(X_mat)
X_inverse = np.linalg.pinv(X_mat)
item_price = X_inverse @ y_pay

print("Feature Matrix:", X_mat)
print("Payment Vector:", y_pay)
print("Rank of Feature Matrix:", rank_val)
print("Estimated Item Cost:", item_price)

# ---------------- A2 : CUSTOMER CLASSIFICATION ----------------

binary_class = (purchase_df["Payment (Rs)"] > 200).astype(int)

weight_vector = X_inverse.dot(binary_class)
score_vector = X_mat.dot(weight_vector)

final_class = np.where(score_vector >= 0.5, "RICH", "POOR")

print(binary_class)
print(final_class)

# ---------------- A3 : IRCTC STOCK DATA ----------------

stock_df = pd.read_excel(
    r"C:\Users\ADMIN\Downloads\Lab Session Data.xlsx",
    sheet_name="IRCTC Stock Price"
)

price_array = stock_df.iloc[:, 3].values

def mean_calc(values):
    total = 0
    for v in values:
        total += v
    return total / len(values)

def variance_calc(values):
    m = mean_calc(values)
    temp = 0
    for v in values:
        temp += (v - m) ** 2
    return temp / len(values)

def time_pkg(values):
    start = time.time()
    for _ in range(10):
        np.mean(values)
        np.var(values)
    return (time.time() - start) / 10

def time_manual(values):
    start = time.time()
    for _ in range(10):
        mean_calc(values)
        variance_calc(values)
    return (time.time() - start) / 10

wed_df = stock_df[stock_df["Day"] == "Wed"]
wed_mean = np.mean(wed_df["Price"])
pop_mean = np.mean(price_array)

chg_values = stock_df.iloc[:, 8]
loss_prob = (chg_values < 0).sum() / len(chg_values)

wed_profit_prob = (wed_df["Chg%"] > 0).sum() / len(wed_df)

apr_df = stock_df[stock_df["Month"] == "Apr"]
apr_mean = np.mean(apr_df["Price"])

print("Mean difference:", mean_calc(price_array) - np.mean(price_array))
print("Variance difference:", variance_calc(price_array) - np.var(price_array))
print("Manual Time:", time_manual(price_array))
print("Package Time:", time_pkg(price_array))
print("Wednesday Mean:", wed_mean)
print("Population Mean:", pop_mean)
print("April Mean:", apr_mean)
print("Loss Probability:", loss_prob)
print("Profit Probability on Wednesday:", wed_profit_prob)

plt.scatter(stock_df["Day"], stock_df["Chg%"])
plt.xlabel("Day")
plt.ylabel("Change %")
plt.show()

# ---------------- A4 : THYROID DATA ----------------

thyroid_df = pd.read_excel(
    r"C:\Users\ADMIN\Downloads\Lab Session Data.xlsx",
    sheet_name="thyroid0387_UCI"
)

print(thyroid_df.head())
print(thyroid_df.info())

num_cols = thyroid_df.select_dtypes(include=[np.number]).columns
cat_cols = thyroid_df.select_dtypes(exclude=[np.number]).columns

print("Numeric Columns:", num_cols)
print("Categorical Columns:", cat_cols)

missing_vals = thyroid_df.isnull().sum()
print("Missing Values:\n", missing_vals)

def outlier_count(series):
    q1 = np.percentile(series.dropna(), 25)
    q3 = np.percentile(series.dropna(), 75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return ((series < low) | (series > high)).sum()

for col in num_cols:
    print(col, "Outliers:", outlier_count(thyroid_df[col]))

print("Mean Values:\n", thyroid_df[num_cols].mean())
print("Variance Values:\n", thyroid_df[num_cols].var())

# ---------------- A5 : JC & SMC ----------------

binary_cols = thyroid_df.columns[thyroid_df.isin(['t', 'f']).all()]
binary_df = thyroid_df[binary_cols].replace({'t': 1, 'f': 0})

vec_a = binary_df.iloc[0].values
vec_b = binary_df.iloc[1].values

f11 = f10 = f01 = f00 = 0

for x, y in zip(vec_a, vec_b):
    if x == 1 and y == 1:
        f11 += 1
    elif x == 1 and y == 0:
        f10 += 1
    elif x == 0 and y == 1:
        f01 += 1
    else:
        f00 += 1

jc_val = f11 / (f11 + f10 + f01)
smc_val = (f11 + f00) / (f11 + f10 + f01 + f00)

print("Jaccard Coefficient:", jc_val)
print("Simple Matching Coefficient:", smc_val)

# ---------------- A6 : COSINE SIMILARITY ----------------

num_data = thyroid_df[num_cols]
v1 = num_data.iloc[0].values
v2 = num_data.iloc[1].values

cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
print("Cosine Similarity:", cos_sim)

# ---------------- A7 : HEATMAP ----------------

num_subset = num_data.iloc[:20]
binary_subset = (num_subset > num_subset.mean()).astype(int)

n = len(num_subset)
JC = np.zeros((n, n))
COS = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        JC[i, j] = np.sum((binary_subset.iloc[i] == 1) &
                          (binary_subset.iloc[j] == 1)) / max(
            1, np.sum(binary_subset.iloc[i] | binary_subset.iloc[j])
        )
        COS[i, j] = np.dot(num_subset.iloc[i], num_subset.iloc[j]) / (
            np.linalg.norm(num_subset.iloc[i]) * np.linalg.norm(num_subset.iloc[j])
        )

sns.heatmap(JC, cmap="Blues")
plt.title("Jaccard Similarity")
plt.show()

sns.heatmap(COS, cmap="Reds")
plt.title("Cosine Similarity")
plt.show()
