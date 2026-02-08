import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# -------------------- A1 : CONFUSION MATRIX & METRICS --------------------

file_path = r"C:\Users\rvija\Desktop\amrita\Semester-4\Machine Learning\writer_identification_through_text_blogs_curated.xlsx"
df = pd.read_excel(file_path)

X_data = df.iloc[:, 0:200]
y_label = df.iloc[:, 200]

X_tr, X_te, y_tr, y_te = train_test_split(
    X_data,
    y_label,
    test_size=0.2,
    random_state=42,
    stratify=y_label
)

svc_model = LinearSVC()
svc_model.fit(X_tr, y_tr)

train_pred = svc_model.predict(X_tr)
test_pred = svc_model.predict(X_te)

train_acc = accuracy_score(y_tr, train_pred)
test_acc = accuracy_score(y_te, test_pred)

print("Training Confusion Matrix:\n", confusion_matrix(y_tr, train_pred))
print("Training Report:\n", classification_report(y_tr, train_pred))
print("Training Accuracy:", train_acc)

print("Testing Confusion Matrix:\n", confusion_matrix(y_te, test_pred))
print("Testing Report:\n", classification_report(y_te, test_pred))
print("Testing Accuracy:", test_acc)

if train_acc < 0.7 and test_acc < 0.7:
    print("Model is underfitting")
elif train_acc > 0.9 and test_acc < 0.7:
    print("Model is overfitting")
else:
    print("Model shows regular fit")


# -------------------- A2 : REGRESSION METRICS --------------------

lab_file = r"C:\Users\rvija\Desktop\amrita\Semester-4\Machine Learning\Lab Session Data.xlsx"
lab_df = pd.read_excel(lab_file)

X_price = lab_df.iloc[:, 1:4]
y_price = lab_df["Payment (Rs)"]

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_price,
    y_price,
    test_size=0.2,
    random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_tr2, y_tr2)

predicted_price = reg_model.predict(X_te2)

mse_val = mean_squared_error(y_te2, predicted_price)
rmse_val = np.sqrt(mse_val)
mape_val = mean_absolute_percentage_error(y_te2, predicted_price)
r2_val = r2_score(y_te2, predicted_price)

print("MSE:", mse_val)
print("RMSE:", rmse_val)
print("MAPE:", mape_val)
print("R2 Score:", r2_val)


# -------------------- A3 : RANDOM DATA VISUALIZATION --------------------

np.random.seed(42)
x_vals = np.random.randint(1, 11, 20)
y_vals = np.random.randint(1, 11, 20)

class_labels = np.where(x_vals + y_vals > 10, 1, 0)
color_map = ["blue" if c == 0 else "red" for c in class_labels]

plt.scatter(x_vals, y_vals, c=color_map)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# -------------------- A4 : kNN CLASSIFICATION GRID --------------------

train_points = np.column_stack((x_vals, y_vals))

grid_x = np.arange(0, 10.1, 0.1)
grid_y = np.arange(0, 10.1, 0.1)

gx, gy = np.meshgrid(grid_x, grid_y)
test_points = np.c_[gx.ravel(), gy.ravel()]

knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(train_points, class_labels)

grid_pred = knn_3.predict(test_points)

grid_colors = ["blue" if g == 0 else "red" for g in grid_pred]
train_colors = ["blue" if t == 0 else "red" for t in class_labels]

plt.scatter(test_points[:, 0], test_points[:, 1], c=grid_colors, s=10, alpha=0.4)
plt.scatter(x_vals, y_vals, c=train_colors, s=80, edgecolors="black")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# -------------------- A5 : EFFECT OF k --------------------

for k_val in range(3, 7):
    knn_model = KNeighborsClassifier(n_neighbors=k_val)
    knn_model.fit(train_points, class_labels)

    pred_grid = knn_model.predict(test_points)
    grid_colors = ["blue" if p == 0 else "red" for p in pred_grid]

    plt.scatter(test_points[:, 0], test_points[:, 1], c=grid_colors, s=10, alpha=0.4)
    plt.scatter(x_vals, y_vals, c=train_colors, s=80, edgecolors="black")
    plt.title(f"k = {k_val}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# -------------------- A7 : GRID SEARCH FOR BEST k --------------------

knn_base = KNeighborsClassifier()
k_values = {'n_neighbors': list(range(1, 16))}

grid_search = GridSearchCV(
    knn_base,
    k_values,
    cv=5,
    scoring='accuracy'
)

grid_search.fit(train_points, class_labels)

optimal_k = grid_search.best_params_['n_neighbors']
print("Optimal k value:", optimal_k)
print("Best CV Accuracy:", grid_search.best_score_)

best_knn = KNeighborsClassifier(n_neighbors=optimal_k)
best_knn.fit(train_points, class_labels)

best_pred = best_knn.predict(test_points)
best_colors = ["blue" if p == 0 else "red" for p in best_pred]

plt.figure(figsize=(8, 6))
plt.scatter(test_points[:, 0], test_points[:, 1], c=best_colors, s=10, alpha=0.4)
plt.scatter(x_vals, y_vals, c=train_colors, s=80, edgecolors="black")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
