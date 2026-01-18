import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def get_adaptive_threshold(k, n_samples, num_classes, c_strictness):
    log_n = np.log(n_samples)
    base_prior = 1.0 / num_classes
    confidence_interval = c_strictness * np.sqrt(log_n / k)
    return base_prior + confidence_interval

def predict_adaptive(X_train, y_train, X_test, min_k=5, max_k=40, c_strictness=0.7):
    n_samples = X_train.shape[0]
    num_classes = len(np.unique(y_train))
    predictions = []

    for row in X_test:
        dists = np.sqrt(np.sum((X_train - row)**2, axis=1))
        sorted_indices = np.argsort(dists)
        final_prediction = None

        for k in range(min_k, max_k + 1):
            k_indices = sorted_indices[:k]
            k_labels = y_train[k_indices]

            vote_counts = Counter(k_labels)
            most_common = vote_counts.most_common(1)[0]
            majority_class = most_common[0]
            majority_count = most_common[1]

            observed_bias = majority_count / k
            threshold = get_adaptive_threshold(k, n_samples, num_classes, c_strictness)

            if observed_bias > threshold:
                final_prediction = majority_class
                break

        if final_prediction is None:
            k_labels = y_train[sorted_indices[:max_k]]
            final_prediction = Counter(k_labels).most_common(1)[0][0]

        predictions.append(final_prediction)

    return np.array(predictions)

def run_single_trial(exp_name, X, y, test_size=0.3):
    print(f"\nExperiment: {exp_name}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=97, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    std_knn = KNeighborsClassifier(n_neighbors=5)
    std_knn.fit(X_train, y_train)
    std_acc = accuracy_score(y_test, std_knn.predict(X_test))

    ada_pred = predict_adaptive(X_train, y_train, X_test, min_k=5, max_k=40, c_strictness=0.7)
    ada_acc = accuracy_score(y_test, ada_pred)

    print()
    print(f"Standard KNN (Fixed k=5) Accuracy: {std_acc:.4f}")
    print(f"Adaptive KNN (Min k=5)   Accuracy: {ada_acc:.4f}")

    diff = ada_acc - std_acc
    if diff > 0:
        print(f"WINNER: Adaptive (+{diff*100:.2f}%)")
    elif diff < 0:
        print(f"WINNER: Standard (+{abs(diff)*100:.2f}%)")
    else:
        print("RESULT: Performance Equal (Tie)")

np.random.seed(97)

data = load_wine()
X_base, y_base = data.data, data.target

run_single_trial("1. Baseline (Clean Data)", X_base, y_base)

X_noisy = X_base + np.random.normal(0, 3.0, X_base.shape)
run_single_trial("2. Noisy Features (Gaussian)", X_noisy, y_base)

noise_cols = np.random.rand(X_base.shape[0], 8) * 10
X_irrelevant = np.hstack((X_base, noise_cols))
run_single_trial("3. Irrelevant Features (Dimensionality)", X_irrelevant, y_base)

X_outliers = X_base.copy()
num_outliers = int(0.2 * len(X_outliers))
indices = np.random.choice(len(X_outliers), num_outliers, replace=False)
X_outliers[indices, 6] *= 25.0
run_single_trial("4. Feature Outliers (Distorted Geometry)", X_outliers, y_base)

mask_keep = np.ones(len(y_base), dtype=bool)
class_1_indices = np.where(y_base == 1)[0]
drop_indices = np.random.choice(class_1_indices, size=int(0.75 * len(class_1_indices)), replace=False)
mask_keep[drop_indices] = False
X_imbalance, y_imbalance = X_base[mask_keep], y_base[mask_keep]
run_single_trial("5. High Class Imbalance (Sparse Data)", X_imbalance, y_imbalance)
