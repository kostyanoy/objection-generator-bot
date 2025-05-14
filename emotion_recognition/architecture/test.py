from sklearn.metrics import classification_report


def test_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    return classification_report(y_test, y_pred)