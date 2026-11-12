"""Train several candidate models, pick the champion by cross-validated ROC-AUC,
and save it to model/champion.joblib with its metadata."""
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
MODEL_DIR = Path(__file__).parent / "model"


def main() -> None:
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=SEED, stratify=data.target
    )

    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)
        ),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }

    cv_scores = {}
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        cv_scores[name] = float(scores.mean())
        print(f"{name:22s} CV ROC-AUC = {scores.mean():.4f}")

    champion_name = max(cv_scores, key=cv_scores.get)
    champion = candidates[champion_name].fit(X_train, y_train)
    test_auc = float(roc_auc_score(y_test, champion.predict_proba(X_test)[:, 1]))
    print(f"Champion: {champion_name} | held-out ROC-AUC = {test_auc:.4f}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(champion, MODEL_DIR / "champion.joblib")
    meta = {
        "champion": champion_name,
        "cv_roc_auc": cv_scores,
        "test_roc_auc": test_auc,
        "feature_names": list(data.feature_names),
        "class_names": list(data.target_names),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
