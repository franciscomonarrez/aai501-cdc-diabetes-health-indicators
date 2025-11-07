import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

from src.aai501_diabetes.config import DATA_RAW, MODELS_DIR

RANDOM_STATE = 42
TARGET = "Diabetes_binary"  # update if the column name differs

def load() -> pd.DataFrame:
    if not DATA_RAW.exists():
        raise FileNotFoundError(f"Missing CSV at {DATA_RAW}")
    return pd.read_csv(DATA_RAW)


def split(df: pd.DataFrame):
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def preprocessor(X: pd.DataFrame):
    num = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num),
        ("cat", cat_pipe, cat),
    ])


def train_and_eval(name: str, clf, X_train, X_test, y_train, y_test, pre):
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, digits=4))
    if hasattr(pipe, "predict_proba"):
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        print(f"ROC AUC: {auc:.4f}")
    out = MODELS_DIR / f"{name.replace(' ', '_').lower()}.joblib"
    joblib.dump(pipe, out)
    print(f"Saved: {out}")


def main():
    df = load()
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. Update TARGET.")
    X_train, X_test, y_train, y_test = split(df)
    pre = preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
    }

    for name, clf in models.items():
        train_and_eval(name, clf, X_train, X_test, y_train, y_test, pre)


if __name__ == "__main__":
    main()
