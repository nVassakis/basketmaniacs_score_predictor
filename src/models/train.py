import pandas as pd
import sqlite3
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

import matplotlib.pyplot as plt

def train_model():
    db_path = 'data/basketball.db'
    conn = sqlite3.connect(db_path)

    # Load Data & Sort Chronologically
    print("Loading engineered features...")
    df = pd.read_sql("SELECT * FROM ml_features", conn)
    conn.close()

    df['DATE'] = pd.to_datetime(df['DATE'])
    df = df.sort_values('DATE').reset_index(drop=True)

    # Define target and features
    target = 'PTS'

    # Features: calculated in logic.py
    features = [
        'PTS_last_3', 'EFF_last_3', 'USG_last_3',
        'PTS_season_avg', 'REB_season_avg', 'EFF_season_avg',
        'PTS_vs_OPP_hist', 'EFF_vs_OPP_hist',
        'OPP_PTS_ALLOWED_PER_PLAYER', 'OPP_REB_ALLOWED_PER_PLAYER', 
        'SCORING_VACUUM', 'AST_last_3', 'BLK_last_3', 'STL_last_3', 'REB_last_3',
        'AST_season_avg', 'BLK_season_avg', 'STL_season_avg','TS_PCT_last_3', 'TS_PCT_season_avg','MONTH', 'TEAM_USG_RANK'
    ]

    X = df[features].apply(pd.to_numeric, errors='coerce')
    y = df[target]

    # Chronological Train / Test Split
    split_index = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Training on {len(X_train)} historical games...")
    print(f"Testing on {len(X_test)} future games...")

    # Train the model
    print("Training the XGBoost model...") 
    
    # Optimized by the Grid Search
    model = xgb.XGBRegressor(
        learning_rate=0.01, 
        max_depth=3, 
        n_estimators=500, 
        subsample=0.8, 
        random_state=234
    )
    
    # Train
    model.fit(X_train, y_train)

    # Train & Tune the XGBoost Model
    # from sklearn.model_selection import GridSearchCV
    
    # param_grid = {
    #     'max_depth': [2, 3],              
    #     'learning_rate': [0.005, 0.01],     
    #     'n_estimators': [500, 750, 1000],  
    #     'subsample': [0.7, 0.8, 0.9]   
    # }

    #  Create a blank factory-default model
    # base_model = xgb.XGBRegressor(random_state=234)

    # # Set up the automated Grid Search
    # # cv=3 means it double-checks its homework 3 times per combination
    # grid_search = GridSearchCV(
    #     estimator=base_model, 
    #     param_grid=param_grid, 
    #     scoring='neg_mean_absolute_error', 
    #     cv=3, 
    #     verbose=1
    # )

    # print("Testing all combinations... this may take a few minutes...")
    # grid_search.fit(X_train, y_train)

    # # Overwrite the blank model with the absolute best one it found
    # model = grid_search.best_estimator_

    # print(f"Tuning Complete!")
    # print(f"Best Settings Found: {grid_search.best_params_}\n")
    # ---------------------------------------------------------

    # Predict
    train_predictions = model.predict(X_train)
    test_predictions  = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae  = mean_absolute_error(y_test,  test_predictions)

    print("\n--- OVERFITTING CHECK ---")
    print(f"Train MAE: {train_mae:.2f} pts")
    print(f"Test MAE:  {test_mae:.2f} pts")
    print(f"Gap:       {test_mae - train_mae:+.2f} pts")

    naive_mask  = X_test['PTS_last_3'].notna()
    naive_preds = X_test.loc[naive_mask, 'PTS_last_3']

    train_errors = y_train.values - train_predictions
    test_errors  = y_test.values  - test_predictions
    naive_errors = y_test[naive_mask].values - naive_preds.values

    def tier_mae(y_true, y_pred):
        df = pd.DataFrame({'actual': y_true, 'pred': y_pred})
        df['tier'] = pd.cut(df['actual'], bins=[-1, 5, 15, 100], labels=['Bench', 'Role', 'Star'])
        return {t: mean_absolute_error(g['actual'], g['pred']) for t, g in df.groupby('tier', observed=True)}

    train_tiers = tier_mae(y_train.values, train_predictions)
    test_tiers  = tier_mae(y_test.values,  test_predictions)
    naive_tiers = tier_mae(y_test[naive_mask].values, naive_preds.values)

    print(f"\n{'Metric':<20} {'TRAIN':>10} {'TEST':>10} {'NAIVE':>10}")
    print("-" * 52)
    print(f"{'MAE':<20} {mean_absolute_error(y_train, train_predictions):>10.2f} {mean_absolute_error(y_test, test_predictions):>10.2f} {mean_absolute_error(y_test[naive_mask], naive_preds):>10.2f}")
    print(f"{'RMSE':<20} {root_mean_squared_error(y_train, train_predictions):>10.2f} {root_mean_squared_error(y_test, test_predictions):>10.2f} {root_mean_squared_error(y_test[naive_mask], naive_preds):>10.2f}")
    print(f"{'R²':<20} {r2_score(y_train, train_predictions):>10.3f} {r2_score(y_test, test_predictions):>10.3f} {'—':>10}")
    print(f"{'Bias':<20} {train_errors.mean():>+10.2f} {test_errors.mean():>+10.2f} {naive_errors.mean():>+10.2f}")
    print(f"{'Within 3 pts':<20} {(np.abs(train_errors) <= 3).mean():>10.1%} {(np.abs(test_errors) <= 3).mean():>10.1%} {(np.abs(naive_errors) <= 3).mean():>10.1%}")
    print(f"{'Within 5 pts':<20} {(np.abs(train_errors) <= 5).mean():>10.1%} {(np.abs(test_errors) <= 5).mean():>10.1%} {(np.abs(naive_errors) <= 5).mean():>10.1%}")
    print("-" * 52)
    for tier in ['Bench', 'Role', 'Star']:
        print(f"{'MAE ' + tier:<20} {train_tiers[tier]:>10.2f} {test_tiers[tier]:>10.2f} {naive_tiers[tier]:>10.2f}")


    # Feature Importance Chart
    print("Generating Feature Importance Chart...")

    # Pair each feature name with its importance score from the model
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)

    # Save the trained model to use later
    model.save_model("model/xgb_model.json")
    print("Model saved successfully as xgb_model.json!")

    # Draw a horizontal bar chart
    importance_df.plot(kind='barh', x='Feature', y='Importance', figsize=(8, 5), legend=False, color='steelblue')
    plt.title("Which Stats Drove the Predictions?")
    plt.xlabel("Relative Importance Score")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    train_model()