import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
df = pd.read_csv(r"C:\Users\saraye tel\OneDrive\Desktop\ARCH_Roadmap\Phase_1\survey.csv")

# Drop irrelevant columns
df.drop(columns=["comments", "state", "no_employees", "Country", "supervisor", "anonymity"], inplace=True, errors="ignore")

# Fill missing values
for col in df.columns:
    if df[col].dtype == "O":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols.remove("treatment")
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Set X and y
X = df.drop("treatment", axis=1)
y = df["treatment"].apply(lambda x: 1 if x.lower() == "yes" else 0)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# Models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="linear")
}

accuracy_scores = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    accuracy_scores[name] = acc

    print(f"\nModel: {name}")
    print(f"Accuracy: {acc:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test, predictions))

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Treatment", "Treatment"],
                yticklabels=["No Treatment", "Treatment"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Accuracy comparison bar chart
plt.figure(figsize=(8, 5))
sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()), palette="viridis")
plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png", dpi=300)
plt.show()