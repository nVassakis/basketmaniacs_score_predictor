import pandas as pd
import sqlite3
import xgboost as xgb
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np

def train_model():
    db_path = 'data/basketball.db'
    conn = sqlite3.connect(db_path)

    # ---------------------------------------------------------
    # STEP 1: Load Data 
    # ---------------------------------------------------------
    print("Loading engineered features...")
    df = pd.read_sql("SELECT * FROM ml_team_features", conn)
    conn.close()

    # Drop any rows where features couldn't calculate
    df = df.dropna().reset_index(drop=True)

    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)

    # ---------------------------------------------------------
    # STEP 2: Define the Target (y) and Features (X)
    # ---------------------------------------------------------
    target = 'TEAM_PTS'

    features = [
        'MONTH', 'DAY_OF_WEEK',
        'SOLID_PLAYER_COUNT', 'BIG_3_EFF_SUM',
        'ACTIVE_ROSTER_EFF', 'ACTIVE_ROSTER_STL', 'ACTIVE_ROSTER_BLK',
        'TEAM_PTS_last_5', 'TEAM_AST_last_5', 'TEAM_REB_last_5',
        'TEAM_3PT_PCT_last_5', 'TEAM_PTS_ALLOWED_last_5',
        # Opponent
         'OPP_BIG_3_EFF_SUM',
        'OPP_ROSTER_SCORING_VARIANCE', 'OPP_ACTIVE_ROSTER_STL', 'OPP_ACTIVE_ROSTER_BLK',
        'OPP_TEAM_PTS_last_5', 'OPP_TEAM_3PT_PCT_last_5',
        'OPP_PTS_ALLOWED_last_5', 'OPP_REB_ALLOWED_last_5',
        'OPP_3PT_PCT_ALLOWED_season',
    ]

    # Ensure all features exist in df, otherwise print a warning
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns in database: {missing_cols}")
        features = [col for col in features if col in df.columns]

    X = df[features]
    y = df[target]

    print(f"Total historical games loaded: {len(df)}")

    # Best params from RandomizedSearchCV
    best_params = {
        'subsample': 0.8,
        'reg_lambda': 5,
        'reg_alpha': 0.1,
        'n_estimators': 75,
        'min_child_weight': 3,
        'max_depth': 3,
        'learning_rate': 0.05,
        'gamma': 1.0,
        'colsample_bytree': 0.2
    }

    tscv = TimeSeriesSplit(n_splits=5)

# ####################################################################
# #### Uncomment to re-run hyperparameter search ####
#
# from sklearn.model_selection import RandomizedSearchCV
#
# param_grid = {
#     'learning_rate': [0.005, 0.0075, 0.01, 0.05],
#     'max_depth': [2, 3, 5],
#     'n_estimators': [25, 50, 75, 100],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.2, 0.3, 0.5, 0.6, 0.8, 1.0],
#     'min_child_weight': [1, 3, 5, 7],
#     'gamma': [0, 0.1, 0.5, 0.8, 1.0, 2.0],
#     'reg_alpha': [0, 0.1, 1, 2.0],
#     'reg_lambda': [1, 1.5, 5, 8, 10, 15, 20]
# }
# search = RandomizedSearchCV(
#     estimator=xgb.XGBRegressor(random_state=234),
#     param_distributions=param_grid,
#     cv=tscv,
#     scoring='neg_mean_absolute_error',
#     n_iter=50,
#     verbose=1
# )
# search.fit(X, y)
# best_params = search.best_params_
# print(f"Best parameters found: {best_params}")
#
# ####################################################################

    # ---------------------------------------------------------
    # STEP 3: Walk-Forward Validation
    # ---------------------------------------------------------
    print(f"\n Starting Walk-Forward Validation ({tscv.n_splits} Splits)...")
    fold_mae_scores = []
    fold_rmse_scores = []
    fold_r2_scores = []
    fold_baseline_mae_scores = []
    fold_dummy_mae_scores = []

    for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Dummy model: always predicts global mean
        dummy = DummyRegressor(strategy='mean')
        dummy.fit(X_train, y_train)
        dummy_mae = mean_absolute_error(y_test, dummy.predict(X_test))
        fold_dummy_mae_scores.append(dummy_mae)

        # Baseline: predict each team's mean score from training data
        team_means = df.iloc[train_index].groupby('TEAM')['TEAM_PTS'].mean()
        global_mean = y_train.mean()
        team_baseline_preds = df.iloc[test_index]['TEAM'].map(team_means).fillna(global_mean)
        baseline_mae = mean_absolute_error(y_test, team_baseline_preds)
        fold_baseline_mae_scores.append(baseline_mae)

        # Model for this fold
        fold_model = xgb.XGBRegressor(**best_params, random_state=234)
        fold_model.fit(X_train, y_train)

        test_preds = fold_model.predict(X_test)

        # Calculate scores
        test_mae = mean_absolute_error(y_test, test_preds)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        test_r2 = r2_score(y_test, test_preds)

        fold_mae_scores.append(test_mae)
        fold_rmse_scores.append(test_rmse)
        fold_r2_scores.append(test_r2)
        print(f"   Fold {fold}: Dummy MAE: {dummy_mae:.2f} | Per-team MAE: {baseline_mae:.2f} | Model MAE: {test_mae:.2f} | R²: {test_r2:.3f}")

    # Mean scores across all folds
    true_mae = sum(fold_mae_scores) / len(fold_mae_scores)
    true_rmse = sum(fold_rmse_scores) / len(fold_rmse_scores)
    true_r2 = sum(fold_r2_scores) / len(fold_r2_scores)
    true_baseline_mae = sum(fold_baseline_mae_scores) / len(fold_baseline_mae_scores)
    true_dummy_mae = sum(fold_dummy_mae_scores) / len(fold_dummy_mae_scores)
    print("\n--- RESULTS ---")
    print(f"Dummy (global mean) MAE:      {true_dummy_mae:.2f} points")
    print(f"Baseline (per-team mean) MAE: {true_baseline_mae:.2f} points")
    print(f"Model MAE:                    {true_mae:.2f} points")
    print(f"True Average RMSE across all periods: {true_rmse:.2f} points")
    print(f"True Average R²: {true_r2:.3f}")

    # ---------------------------------------------------------
    # STEP 5: Training
    # ---------------------------------------------------------
    print("\nTraining final  model on 100% of data...")

    mlflow.xgboost.autolog()
    with mlflow.start_run(run_name="Tuned_Team_Regression"):
        final_model = xgb.XGBRegressor(**best_params, random_state=234)
        final_model.fit(X, y) 
        mlflow.log_metric("cv_per_team_baseline_mae", true_baseline_mae)
        mlflow.log_metric("cv_true_mae", true_mae)
        mlflow.log_metric("cv_true_rmse", true_rmse)
        mlflow.log_metric("cv_true_r2", true_r2)

        # Generate predictions on the same data it just learned from
        final_train_preds = final_model.predict(X)
        final_train_mae = mean_absolute_error(y, final_train_preds)
        print(f"Final Model Train MAE: {final_train_mae:.2f}")
    
    # ---------------------------------------------------------
    # STEP 6: Save and Autopsy
    # ---------------------------------------------------------
    final_model.save_model("model/team_xgb_model_walk_forward.json")
    print("Model saved successfully as xgb_model.json!")

    # Generate Feature Importance Chart from the Final Model
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': final_model.feature_importances_
    }).sort_values(by='Importance', ascending=True)

    importance_df.plot(kind='barh', x='Feature', y='Importance', figsize=(8, 5), legend=False, color='steelblue')
    plt.title("Which Stats Drove the Predictions?")
    plt.xlabel("Relative Importance Score")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_model()