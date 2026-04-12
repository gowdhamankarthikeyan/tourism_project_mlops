import pandas as pd
import joblib
import os
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from huggingface_hub import HfApi

mlflow.set_experiment("Wellness_Tourism_Experiment")

repo_id = "gowdhamankarthikeyan/tourism-dataset"
model_repo_id = "gowdhamankarthikeyan/wellness-tourism-model"

print("Loading prepared data...")
Xtrain = pd.read_csv(f"hf://datasets/{repo_id}/Xtrain.csv")
ytrain = pd.read_csv(f"hf://datasets/{repo_id}/ytrain.csv").squeeze()
Xtest = pd.read_csv(f"hf://datasets/{repo_id}/Xtest.csv")
ytest = pd.read_csv(f"hf://datasets/{repo_id}/ytest.csv").squeeze()

with mlflow.start_run():
    # UPGRADE 1: A rigorous, academic-level hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }

    # UPGRADE 2: Added class_weight='balanced' for imbalanced target data
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')

    # UPGRADE 3: Optimizing for F1-score rather than just raw accuracy
    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    ytrain_pred = best_model.predict(Xtrain)
    ytest_pred = best_model.predict(Xtest)

    # Generate probabilities for ROC-AUC
    ytest_proba = best_model.predict_proba(Xtest)[:, 1]
    auc_score = roc_auc_score(ytest, ytest_proba)

    train_report = classification_report(ytrain, ytrain_pred, output_dict=True)
    test_report = classification_report(ytest, ytest_pred, output_dict=True)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score'],
        "test_roc_auc": auc_score # Added AUC metric
    })

    joblib.dump(best_model, "rf_model.joblib")

    api = HfApi(token=os.getenv("HF_TOKEN"))
    try:
        api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)
    except:
        pass

    api.upload_file(
        path_or_fileobj="rf_model.joblib",
        path_in_repo="rf_model.joblib",
        repo_id=model_repo_id,
        repo_type="model"
    )
print("Advanced Model Training Complete.")
