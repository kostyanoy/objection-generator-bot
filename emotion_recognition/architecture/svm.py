import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def build_svm_model():
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            random_state=42,
            verbose=False
        ))
    ])
    return svm_pipeline

def svm_parameter_search(X_train, y_train):
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],  # Сила регуляризации
        'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1],  # Влияние отдельных образцов
        'svm__kernel': ['linear', 'rbf', 'poly'],  # Тип ядра
        'svm__degree': [2, 3]  # Актуально только для poly
    }

    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            class_weight='balanced',
            random_state=42,
        ))
    ])

    y_train = np.argmax(y_train, axis=1)

    grid_search = GridSearchCV(
        estimator=svm_pipeline,
        param_grid=param_grid,
        scoring='f1_macro',  # Для мультикласса
        cv=3,
        n_jobs=6,  # Используем все ядра CPU
        verbose=3  # Логирование процесса
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_params_


def train_svm(X_train, y_train, X_val, y_val, config):
    model = config["model_builder"]()
    model.fit(X_train, y_train)
    return model, {}