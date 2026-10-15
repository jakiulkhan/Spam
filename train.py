import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from preprocess import clean_text

DATA_PATH = os.path.join("data", "spam.csv")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH, encoding="latin-1")[["v1", "v2"]]
    df.columns = ["label", "message"]
    df = df.drop_duplicates().reset_index(drop=True)

    df["clean_message"] = df["message"].apply(clean_text)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"], df["label"],
        test_size=0.20, random_state=42, stratify=df["label"],
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("svm", LinearSVC()),
    ])

    grid = GridSearchCV(pipeline, {"svm__C": [0.1, 1, 10]}, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best params:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    best = grid.best_estimator_
    preds = best.predict(X_test)
    print("Test accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    cv = cross_val_score(best, df["clean_message"], df["label"], cv=5)
    print("5-fold CV mean:", cv.mean())

    joblib.dump(best, os.path.join(MODEL_DIR, "spam_svm_pipeline.joblib"))
    print("Model saved to model/spam_svm_pipeline.joblib")


if __name__ == "__main__":
    main()