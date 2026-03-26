import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# -------------------------------
# Load dataset
# -------------------------------
file_path = r"C:/Users/ADMIN/Downloads/Blog_TFIDF_Vectors.xlsx"
data = pd.read_excel(file_path)


# -------------------------------
# Q1 - Equal Width Binning
# -------------------------------
def width_binning(values, num_bins=4):
    arr = np.array(values)
    minimum = arr.min()
    maximum = arr.max()

    bin_size = (maximum - minimum) / num_bins
    bin_edges = [minimum + i * bin_size for i in range(num_bins + 1)]

    return np.digitize(arr, bin_edges[1:-1])


def calc_entropy(values):
    arr = np.array(values)
    unique_vals, counts = np.unique(arr, return_counts=True)

    probabilities = counts / len(arr)
    entropy_val = -np.sum(probabilities * np.log2(probabilities))

    return entropy_val


# -------------------------------
# Q2 - Gini Index
# -------------------------------
def calc_gini(values):
    arr = np.array(values)
    unique_vals, counts = np.unique(arr, return_counts=True)

    probabilities = counts / len(arr)
    return 1 - np.sum(probabilities ** 2)


# -------------------------------
# Q3 - Information Gain
# -------------------------------
def calc_information_gain(df, feature, target_col):
    total_ent = calc_entropy(df[target_col])

    feature_values = np.unique(df[feature])
    weighted_ent = 0

    for val in feature_values:
        subset = df[df[feature] == val]
        weight = len(subset) / len(df)
        weighted_ent += weight * calc_entropy(subset[target_col])

    return total_ent - weighted_ent


def find_root_feature(df, target_col):
    feature_list = [col for col in df.columns if col != target_col]

    gain_values = []
    for f in feature_list:
        gain_values.append(calc_information_gain(df, f, target_col))

    return feature_list[np.argmax(gain_values)]


# -------------------------------
# Q4 - Equal Frequency Binning
# -------------------------------
def frequency_binning(values, num_bins=4):
    arr = np.array(values)
    quantiles = np.percentile(arr, np.linspace(0, 100, num_bins + 1))

    return np.digitize(arr, quantiles[1:-1])


def apply_binning(values, method="width", bins=4):
    if method == "width":
        return width_binning(values, bins)
    else:
        return frequency_binning(values, bins)


# -------------------------------
# Q5 - Decision Tree (Custom)
# -------------------------------
class TreeNode:
    def __init__(self, split_feature=None, label=None):
        self.split_feature = split_feature
        self.label = label
        self.children = {}


def create_tree(df, target_col):
    # If all values same → leaf node
    if len(np.unique(df[target_col])) == 1:
        return TreeNode(label=df[target_col].iloc[0])

    features = [c for c in df.columns if c != target_col]

    # If no features left
    if len(features) == 0:
        return TreeNode(label=df[target_col].mode()[0])

    gains = [calc_information_gain(df, f, target_col) for f in features]
    best_feature = features[np.argmax(gains)]

    node = TreeNode(split_feature=best_feature)

    for val in np.unique(df[best_feature]):
        subset = df[df[best_feature] == val].drop(columns=[best_feature])
        node.children[val] = create_tree(subset, target_col)

    return node


# -------------------------------
# Q6 - Visualize Decision Tree
# -------------------------------
def show_tree(X, y):
    model = DecisionTreeClassifier()
    model.fit(X, y)

    plt.figure(figsize=(10, 6))
    plot_tree(model, feature_names=X.columns, filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()


# -------------------------------
# Q7 - Decision Boundary
# -------------------------------
def decision_boundary_plot(X, y):
    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title("Decision Boundary")
    plt.show()


# -------------------------------
# Main Execution
# -------------------------------

target_column = data.columns[-1]

# Apply binning
for column in data.columns:
    if data[column].dtype != "object" and column != target_column:
        data[column] = apply_binning(data[column], method="width", bins=4)

# Print impurity measures
print("Entropy:", calc_entropy(data[target_column]))
print("Gini Index:", calc_gini(data[target_column]))

# Root node
root_feature = find_root_feature(data, target_column)
print("Root Feature:", root_feature)

# Build tree
tree = create_tree(data, target_column)

# Prepare data
X = data.drop(columns=[target_column])
y = data[target_column]

# Visualization
show_tree(X, y)

# Decision boundary (first 2 features)
X_subset = X.iloc[:, 0:2].values
decision_boundary_plot(X_subset, y)