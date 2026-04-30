import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

# LIME
from lime.lime_tabular import LimeTabularExplainer


# --------------------------------------------------
# Load Data
# --------------------------------------------------
df = pd.read_excel(r"C:/Users/ADMIN/Downloads/Blog_TFIDF_Vectors.xlsx")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]


# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --------------------------------------------------
# A1: Stacking Classifier
# --------------------------------------------------

base_models = [
    ("rf", RandomForestClassifier(n_estimators=100)),
    ("svm", SVC(probability=True))
]

meta_model = LogisticRegression()

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model
)

stack_model.fit(X_train, y_train)

y_pred_stack = stack_model.predict(X_test)

print("Stacking Accuracy:", accuracy_score(y_test, y_pred_stack))


# --------------------------------------------------
# A2: Pipeline (Scaling + Stacking)
# --------------------------------------------------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", stack_model)
])

pipeline.fit(X_train, y_train)

y_pred_pipe = pipeline.predict(X_test)

print("Pipeline Accuracy:", accuracy_score(y_test, y_pred_pipe))


# --------------------------------------------------
# A3: LIME Explanation
# --------------------------------------------------

explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns.tolist(),
    class_names=[str(i) for i in np.unique(y)],
    mode='classification'
)

# Explain one instance
sample = X_test.iloc[0].values

exp = explainer.explain_instance(
    sample,
    pipeline.predict_proba
)

print("\nLIME Explanation:")
for feature, weight in exp.as_list():
    print(feature, ":", weight)
