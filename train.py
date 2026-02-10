import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("CW1_train.csv")

# Split features and target
X = df.drop(columns=["outcome"])
y = df["outcome"]

# Training and test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Column types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns

# Pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols),
])

# Models

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ),
}

# Create pipeline
def make_pipeline(model):
    return Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])


# Evaluate R2 Scores
def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    print(f"{name} R²: {r2:.4f}")

pipelines = {}

for name, model in models.items():
    pipe = make_pipeline(model)
    pipe.fit(X_train, y_train)
    evaluate(pipe, X_test, y_test, name)
    pipelines[name] = pipe

param_grid = {
    "model__n_estimators": [100, 300],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5],
}

# Hyperparameter Tuning
gbm_grid = GridSearchCV(
    pipelines["Gradient Boosting"],
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)

gbm_grid.fit(X_train, y_train)

best_model = gbm_grid.best_estimator_

print("Tuned GBM")
evaluate(best_model, X_test, y_test, "Tuned GBM")
print("Best parameters:", gbm_grid.best_params_)

importances = best_model["model"].feature_importances_
print("Feature importances:")
print(importances)

feature_names = best_model.named_steps["prep"].get_feature_names_out()

importances = best_model.named_steps["model"].feature_importances_

# Print most important features
feature_importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nTop 10 features:")
print(feature_importance_df.head(5))

cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=5,
    scoring="r2",
)

print(f"CV R²: {cv_scores.mean():.3f}")

# Submission
X_submission = pd.read_csv("CW1_test.csv")

y_pred = best_model.predict(X_submission)

submission = pd.DataFrame({
    "outcome": y_pred
})

# Save submission
submission.to_csv("CW1_submission_K23169531.csv", index=False)
