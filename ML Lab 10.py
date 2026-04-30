import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SequentialFeatureSelector

# LIME & SHAP
from lime.lime_tabular import LimeTabularExplainer
import shap


# --------------------------------------------------
# Load Data
# --------------------------------------------------
df = pd.read_excel(r"C:/Users/ADMIN/Downloads/Blog_TFIDF_Vectors.xlsx")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# --------------------------------------------------
# A1: Correlation Heatmap
# --------------------------------------------------
corr = X.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()


# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --------------------------------------------------
# A2: PCA (99%)
# --------------------------------------------------
pca_99 = PCA(n_components=0.99)
X_train_pca99 = pca_99.fit_transform(X_train_scaled)
X_test_pca99 = pca_99.transform(X_test_scaled)

model_99 = RandomForestClassifier()
model_99.fit(X_train_pca99, y_train)

pred_99 = model_99.predict(X_test_pca99)

print("PCA 99% Accuracy:", accuracy_score(y_test, pred_99))


# --------------------------------------------------
# A3: PCA (95%)
# --------------------------------------------------
pca_95 = PCA(n_components=0.95)
X_train_pca95 = pca_95.fit_transform(X_train_scaled)
X_test_pca95 = pca_95.transform(X_test_scaled)

model_95 = RandomForestClassifier()
model_95.fit(X_train_pca95, y_train)

pred_95 = model_95.predict(X_test_pca95)

print("PCA 95% Accuracy:", accuracy_score(y_test, pred_95))


# --------------------------------------------------
# A4: Sequential Feature Selection
# --------------------------------------------------
rf = RandomForestClassifier()

sfs = SequentialFeatureSelector(
    rf,
    n_features_to_select=20,  # adjust if needed
    direction='forward'
)

sfs.fit(X_train_scaled, y_train)

X_train_sfs = sfs.transform(X_train_scaled)
X_test_sfs = sfs.transform(X_test_scaled)

rf.fit(X_train_sfs, y_train)

pred_sfs = rf.predict(X_test_sfs)

print("SFS Accuracy:", accuracy_score(y_test, pred_sfs))


# --------------------------------------------------
# A5: LIME Explanation
# --------------------------------------------------
explainer = LimeTabularExplainer(
    training_data=np.array(X_train_scaled),
    feature_names=X.columns.tolist(),
    class_names=[str(i) for i in np.unique(y)],
    mode='classification'
)

sample = X_test_scaled[0]

exp = explainer.explain_instance(
    sample,
    model_99.predict_proba
)

print("\nLIME Explanation:")
for f, w in exp.as_list():
    print(f, ":", w)


# --------------------------------------------------
# SHAP Explanation
# --------------------------------------------------
explainer_shap = shap.TreeExplainer(model_99)
shap_values = explainer_shap.shap_values(X_test_pca99)

print("\nSHAP summary plot:")
shap.summary_plot(shap_values, X_test_pca99)
