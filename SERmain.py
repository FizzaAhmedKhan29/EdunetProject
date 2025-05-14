import numpy as np
import random
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import sys

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from SERfeatures import load_data, observed_emotions

# Redirect console output to file
sys.stdout = open("SERmain_output.txt", "w")

np.random.seed(42)
random.seed(42)

# Loading the dataset
x_train, x_test, y_train, y_test = load_data(test_size=0.25)

print(f"Training samples: {x_train.shape[0]}, Testing samples: {x_test.shape[0]}")
print(f"Features per sample: {x_train.shape[1]}")

# Defining models
models = {
    "MLP": MLPClassifier(alpha=0.01, batch_size=256,
                         hidden_layer_sizes=(400,), learning_rate='adaptive', max_iter=500, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=6),
    "NaiveBayes": GaussianNB()
}

# Training and evaluating the Accuracy
predictions = {}

for name, model in models.items():
    print(f"\n\n=== {name} ===")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    predictions[name] = y_pred

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Plotting confusion matrices for All Models
for name, y_pred in predictions.items():
    cm = confusion_matrix(y_test, y_pred, labels=observed_emotions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=observed_emotions, yticklabels=observed_emotions)
    plt.title(f'{name} - CONFUSION MATRIX')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()
    plt.savefig(f"SER_confusion_matrix_{name}.png")  # Save figure
    plt.close()

# Summary of Models
print("\nModel Accuracies:")
for name in predictions:
    acc = accuracy_score(y_test, predictions[name]) * 100
    print(f"{name}: {acc:.3f}%")

# Save models
print("\nSaved Models:")
for name, model in models.items():
    joblib.dump(model, f"SER_{name}.joblib")
    print(f"Saved the {name} model!")

# Restore standard output (optional)
sys.stdout.close()
sys.stdout = sys.__stdout__
