import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

file_path = r"C:\amrita\DAA\Blog_TFIDF_Vectors.xlsx"
data = pd.read_excel(file_path)

# ---------------------------------------------------------
# Q1 – Linear Regression (Single Feature)
# ---------------------------------------------------------

target = data.iloc[:, 200]
feature_one = data.iloc[:, [0]]

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    feature_one, target,
    test_size=0.2,
    random_state=42
)

model_lr_1 = LinearRegression()
model_lr_1.fit(X_train_1, y_train_1)

train_predictions_1 = model_lr_1.predict(X_train_1)

print("Training Predictions (Single Feature):")
print(train_predictions_1)


# ---------------------------------------------------------
# Q2 – Linear Regression (Multiple Features)
# ---------------------------------------------------------

features_multi = data.iloc[:, 0:50]

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    features_multi, target,
    test_size=0.2,
    random_state=42
)

model_lr_2 = LinearRegression()
model_lr_2.fit(X_train_2, y_train_2)

train_predictions_2 = model_lr_2.predict(X_train_2)

print("\nTraining Predictions (Multiple Features):")
print(train_predictions_2)


# ---------------------------------------------------------
# Q3 – Regression Metrics
# ---------------------------------------------------------

test_predictions_2 = model_lr_2.predict(X_test_2)

mse_tr = mean_squared_error(y_train_2, train_predictions_2)
rmse_tr = np.sqrt(mse_tr)
mape_tr = np.mean(np.abs((y_train_2 - train_predictions_2) / y_train_2)) * 100
r2_tr = r2_score(y_train_2, train_predictions_2)

mse_te = mean_squared_error(y_test_2, test_predictions_2)
rmse_te = np.sqrt(mse_te)
mape_te = np.mean(np.abs((y_test_2 - test_predictions_2) / y_test_2)) * 100
r2_te = r2_score(y_test_2, test_predictions_2)

print("\n--- Training Metrics ---")
print("MSE:", mse_tr)
print("RMSE:", rmse_tr)
print("MAPE:", mape_tr)
print("R2:", r2_tr)

print("\n--- Testing Metrics ---")
print("MSE:", mse_te)
print("RMSE:", rmse_te)
print("MAPE:", mape_te)
print("R2:", r2_te)


# ---------------------------------------------------------
# Q4 – KMeans Clustering (k = 5)
# ---------------------------------------------------------

cluster_data = data.drop(columns=['Author_Label'])

X_train_cluster, _ = train_test_split(
    cluster_data,
    test_size=0.2,
    random_state=42
)

kmeans_5 = KMeans(n_clusters=5, random_state=42, n_init="auto")
kmeans_5.fit(X_train_cluster)

print("\nCluster Labels:")
print(kmeans_5.labels_)

print("\nCluster Centers:")
print(kmeans_5.cluster_centers_)


# ---------------------------------------------------------
# Q5 – Cluster Validation Metrics (k = 2)
# ---------------------------------------------------------

kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init="auto")
kmeans_2.fit(X_train_cluster)

labels_2 = kmeans_2.labels_

sil = silhouette_score(X_train_cluster, labels_2)
ch = calinski_harabasz_score(X_train_cluster, labels_2)
db = davies_bouldin_score(X_train_cluster, labels_2)

print("\nClustering Metrics (k=2)")
print("Silhouette Score:", sil)
print("Calinski-Harabasz Score:", ch)
print("Davies-Bouldin Index:", db)


# ---------------------------------------------------------
# Q6 – Metrics for Multiple k Values
# ---------------------------------------------------------

k_range = range(2, 11)

sil_list = []
ch_list = []
db_list = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(cluster_data)
    labels_k = km.labels_

    sil_list.append(silhouette_score(cluster_data, labels_k))
    ch_list.append(calinski_harabasz_score(cluster_data, labels_k))
    db_list.append(davies_bouldin_score(cluster_data, labels_k))


plt.figure()
plt.plot(k_range, sil_list, marker='o')
plt.title("Silhouette Score vs k")
plt.xlabel("k")
plt.ylabel("Score")
plt.show()


plt.figure()
plt.plot(k_range, ch_list, marker='o')
plt.title("CH Score vs k")
plt.xlabel("k")
plt.ylabel("Score")
plt.show()


plt.figure()
plt.plot(k_range, db_list, marker='o')
plt.title("DB Index vs k")
plt.xlabel("k")
plt.ylabel("Index")
plt.show()


# ---------------------------------------------------------
# Q7 – Elbow Method
# ---------------------------------------------------------

inertia_values = []

for k in range(2, 20):
    km_elbow = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km_elbow.fit(X_train_cluster)
    inertia_values.append(km_elbow.inertia_)

plt.figure()
plt.plot(range(2, 20), inertia_values, marker='o')
plt.title("Elbow Plot")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()
