import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

#2
df=pd.read_excel(r"C:/Users/ADMIN/Downloads/Blog_TFIDF_Vectors.xlsx")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
y=y-1


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


rf = RandomForestClassifier()

param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)
best_rf = random_search.best_estimator_

print("Best RF Parameters:", random_search.best_params_)

#3
models = {
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest (Tuned)": best_rf,
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "MLP": MLPClassifier(max_iter=300),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}


results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    results.append({
        "Model": name,
        "Train Accuracy": accuracy_score(y_train, y_train_pred),
        "Test Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
        "F1 Score": f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    })


results_df = pd.DataFrame(results)
print("\nFinal Comparison Table:\n")
print(results_df)


results_df.to_excel("model_comparison_results.xlsx", index=False)

#4
reg_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(),
    "SVR": SVR(),
    "XGBoost Regressor": XGBRegressor()
}

reg_results = []

for name, model in reg_models.items():
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    reg_results.append({
        "Model": name,
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2 Score": r2_score(y_test, y_pred)
    })

reg_df = pd.DataFrame(reg_results)
print("\nRegression Results:\n", reg_df)

#5
X_cluster = X_train
hc = AgglomerativeClustering(n_clusters=3)
hc_labels = hc.fit_predict(X_cluster)
db = DBSCAN(eps=0.5, min_samples=5)
db_labels = db.fit_predict(X_cluster)
print("\nClustering Results:")
if len(set(hc_labels)) > 1:
    print("Hierarchical Silhouette Score:",
          silhouette_score(X_cluster, hc_labels))

if len(set(db_labels)) > 1:
    print("DBSCAN Silhouette Score:",
          silhouette_score(X_cluster, db_labels))
