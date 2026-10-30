# Breast-Cancer-Classification-SVM
%pip -q install scikit-learn pandas matplotlib seaborn

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)

# Load the complete dataset
raw = load_breast_cancer(as_frame=True)
X = raw.data

# Encode target: 0 = Benign, 1 = Malignant
y = (raw.target == 0).astype(int)
labels = ["Benign", "Malignant"]

print("Number of cases:", X.shape[0])
print("Number of features:", X.shape[1])
print("Feature names:", list(X.columns))
print("\nClass distribution (0 = Benign, 1 = Malignant):")
print(y.value_counts().sort_index())

# Visualization 1: Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(
    x=y.map({0: labels[0], 1: labels[1]}),
    order=labels,
    color="#4C78A8"
)
plt.title("Class Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.show()

# Visualization 2: Correlation among the 10 features most associated with the target
correlations = X.assign(target=y).corr(numeric_only=True)["target"].drop("target")
top_features = correlations.abs().sort_values(ascending=False).head(10).index

plt.figure(figsize=(10, 7))
sns.heatmap(
    X[top_features].corr(),
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    square=True
)
plt.title("Correlation Between the Top 10 Features")
plt.tight_layout()
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build the SVM pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(probability=True, random_state=42))
])

# Select the best settings using training data only
parameters = {
    "model__C": [0.1, 1, 10, 100],
    "model__gamma": ["scale", 0.01, 0.001],
    "model__kernel": ["rbf"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
    pipeline,
    parameters,
    scoring="accuracy",
    cv=cv,
    n_jobs=-1
)
search.fit(X_train, y_train)

# Evaluate the selected model on the held-out test set
model = search.best_estimator_
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nBest parameters:", search.best_params_)
print(f"Cross-validation accuracy: {search.best_score_:.4f}")
print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

print("\nClassification report (positive class = Malignant):")
print(classification_report(
    y_test,
    predictions,
    target_names=labels,
    digits=4,
    zero_division=0
))

# Visualization 3: Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=labels,
    cmap="Blues",
    values_format="d"
)
plt.title("Confusion Matrix - Test Set")
plt.xlabel("Predicted Diagnosis")
plt.ylabel("True Diagnosis")
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()

# Visualization 4: ROC curve
false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
auc_value = roc_auc_score(y_test, probabilities)

plt.figure(figsize=(6, 5))
plt.plot(
    false_positive_rate,
    true_positive_rate,
    label=f"SVM (AUC = {auc_value:.3f})"
)
plt.plot([0, 1], [0, 1], "--", color="gray", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0ه.25)
plt.tight_layout()
plt.show()


