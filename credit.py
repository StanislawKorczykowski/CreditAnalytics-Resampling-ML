import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from sklearn.model_selection import GridSearchCV


data = pd.read_csv("Credit_card.csv")

X = data.drop("default payment next month", axis=1)
y = data["default payment next month"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=10
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
len(y_train[y_train == 0])
len(y_train[y_train == 1])


# Parametry do przeszukania dla drzewa decyzyjnego
dt_params = {
    'max_depth': [30, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 4],
}

# Parametry do przeszukania dla lasu losowego
rf_params = {
    'n_estimators': [100, 300],
    'max_depth': [30, None],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 4],
}

# Grid search dla drzewa decyzyjnego
dt_grid_search = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=2)
dt_grid_search.fit(X_train_scaled, y_train)
best_dt = dt_grid_search.best_params_

# Grid search dla lasu losowego
rf_grid_search = GridSearchCV(RandomForestClassifier(), rf_params, cv=2)
rf_grid_search.fit(X_train_scaled, y_train)
best_rf = rf_grid_search.best_params_


# Trenowanie modeli na oryginalnych danych
dt_original = DecisionTreeClassifier(**best_dt).fit(X_train_scaled, y_train)
rf_original = RandomForestClassifier(**best_rf).fit(X_train_scaled, y_train)


# Under-sampling
rus = RandomUnderSampler(random_state=10)
X_train_rus, y_train_rus = rus.fit_resample(X_train_scaled, y_train)

dt_rus = DecisionTreeClassifier(**best_dt).fit(X_train_rus, y_train_rus)
rf_rus = RandomForestClassifier(**best_rf).fit(X_train_rus, y_train_rus)


# Over-sampling
minority_class = X_train_scaled[y_train == 1]
minority_class_oversampled = resample(
    minority_class,
    replace=True,
    n_samples=len(X_train_scaled[y_train == 0]),
    random_state=10,
)
X_train_over = np.vstack((X_train_scaled[y_train == 0], minority_class_oversampled))
y_train_over = np.hstack(
    (y_train[y_train == 0], np.ones(len(minority_class_oversampled)))
)
len(y_train_over[y_train_over == 0])
len(y_train_over[y_train_over == 1])

dt_over = DecisionTreeClassifier(**best_dt).fit(X_train_over, y_train_over)
rf_over = RandomForestClassifier(**best_rf).fit(X_train_over, y_train_over)


# SMOTE
smote = SMOTE(random_state=10)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

dt_smote = DecisionTreeClassifier(**best_dt).fit(X_train_smote, y_train_smote)
rf_smote = RandomForestClassifier(**best_rf).fit(X_train_smote, y_train_smote)


# Funkcja do ewaluacji modelu
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return (
        accuracy_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_pred),
    )


# Ewaluacja wszystkich modeli
models = [
    dt_original,
    rf_original,
    dt_rus,
    rf_rus,
    dt_over,
    rf_over,
    dt_smote,
    rf_smote,
]
model_names = [
    "DT Original",
    "RF Original",
    "DT RUS",
    "RF RUS",
    "DT Over",
    "RF Over",
    "DT SMOTE",
    "RF SMOTE",
]

for model, name in zip(models, model_names):
    accuracy, f1, roc = evaluate_model(model, X_test_scaled, y_test)
    print(f"{name} - Accuracy: {accuracy}, F1 Score: {f1}, ROC AUC: {roc}")

print("\n", best_dt, best_rf)