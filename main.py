import os
import sys
import warnings
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
plt.rcParams['figure.dpi'] = 120

from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
import lightgbm as lgb
import optuna
import shap
import joblib

# -------------------------
# Helpers & output folders
# -------------------------
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figs"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)

def save_fig(name):
    path = os.path.join(OUT_DIR, "figs", name)
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved figure: {path}")

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

# -------------------------
# Data loading: flexible
# -------------------------
def load_merged_df():
    """
    Priority:
      1) If the variable `merged_df` exists in the global namespace (e.g., run in notebook and import),
         use that.
      2) Else attempt to load './merged_df.csv' (should be exported from your EDA).
      3) Else, raise an informative error and show the Alpaca fetch snippet you can enable.
    """
    csv_path = "data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['timestamp', 'date'], infer_datetime_format=True)
        return df
    # 3. error + instructions
    raise FileNotFoundError(
        "No merged_df in globals and './merged_df.csv' not found.\n"
        "If you want to fetch from Alpaca directly, uncomment and fill the Alpaca block in this file.\n"
        "Alternatively export your merged DataFrame to ./merged_df.csv and re-run."
    )

# --- Optionally, you can copy/paste your Alpaca fetching logic here and call it if CSV absent.
# I left it commented intentionally so running this file won't accidentally attempt API calls.
"""
# # Uncomment and add your Alpaca keys if you want the script to fetch directly:
# from alpaca.data.historical import StockHistoricalDataClient
# from alpaca.data.requests import StockBarsRequest, NewsRequest
# from alpaca.data.timeframe import TimeFrame
# from alpaca.data.historical.news import NewsClient
# from datetime import datetime
#
# def fetch_from_alpaca():
#     client = StockHistoricalDataClient('<KEY_ID>', '<SECRET>')
#     request_params = StockBarsRequest(
#         symbol_or_symbols=["AAPL"],
#         timeframe=TimeFrame.Day,
#         start=datetime.strptime("2022-01-01", '%Y-%m-%d'),
#         end=datetime.strptime("2023-12-31", '%Y-%m-%d')
#     )
#     bars = client.get_stock_bars(request_params)
#     bars_df = bars.df.copy().reset_index()
#     # fetch news similarly and merge like in your EDA
#     return merged_df
"""

def preprocess_and_engineer(df, debug=False):
    """
    Input: merged_df (as produced by your EDA)
    Steps:
      - Ensure timestamp/date columns
      - Create target = next-day high
      - Lag features, rolling stats
      - Handle missing values
      - Standardize features (scaler fitted on training set later)
    Returns: processed dataframe (no scaling applied), feature list
    """
    df = df.copy()
    # ensure datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'date' in df.columns and not np.issubdtype(df['date'].dtype, np.datetime64):
        try:
            df['date'] = pd.to_datetime(df['date'])
        except:
            df['date'] = pd.to_datetime(df['timestamp'].dt.date)
    else:
        df['date'] = pd.to_datetime(df['timestamp'].dt.date)

    # Sort by time
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Target: tomorrow's high
    df['target_high'] = df['high'].shift(-1)

    # lag features
    lags = [1,2,3]
    for lag in lags:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'return_lag_{lag}'] = df['daily_return'].shift(lag)
        df[f'vol_lag_{lag}'] = df['volume'].shift(lag)

    # Rolling stats
    df['rolling_mean_5'] = df['close'].rolling(5).mean()
    df['rolling_std_5'] = df['close'].rolling(5).std()
    df['rolling_mean_10'] = df['close'].rolling(10).mean()
    df['rolling_std_10'] = df['close'].rolling(10).std()

    # Sentiment lags
    df['sentiment_lag_1'] = df['avg_sentiment'].shift(1)

    # Time features
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

    # Drop rows with NA in target
    df = df.dropna(subset=['target_high']).reset_index(drop=True)

    # Fill remaining missing feature values with median (simple imputer later)
    # Keep columns list
    feature_cols = [
        'open','high','low','close','volume','vwap','trade_count','daily_return',
        'rolling_mean_5','rolling_std_5','rolling_mean_10','rolling_std_10',
        'close_lag_1','close_lag_2','close_lag_3',
        'return_lag_1','return_lag_2','return_lag_3',
        'vol_lag_1','vol_lag_2','vol_lag_3',
        'sentiment_lag_1','dayofweek','month'
    ]
    # Some columns on user data may not exist (e.g., trade_count)
    feature_cols = [c for c in feature_cols if c in df.columns]
    if debug:
        print("Feature columns:", feature_cols)

    # We'll not scale here; scaling is part of a sklearn pipeline fitted on train.
    return df, feature_cols

def training_only_eda(X_train_df, y_train, out_prefix="train"):
    """
    Perform exploration ONLY on the training set per assignment.
    Save descriptive stats, correlation matrix, and plots.
    """
    print("=== Training-only EDA ===")
    stats = X_train_df.describe().T
    stats.to_csv(os.path.join(OUT_DIR, f"{out_prefix}_descriptive_stats.csv"))
    print(f"Saved descriptive stats to {OUT_DIR}/{out_prefix}_descriptive_stats.csv")

    # Correlation (features + target)
    corr = X_train_df.join(y_train).corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Training-set Correlation Matrix (features + target)')
    save_fig(f"{out_prefix}_corr_matrix.png")
    plt.close()

    # Distribution plots for a handful of numeric features
    cols = X_train_df.select_dtypes(include=[np.number]).columns.tolist()[:8]
    for c in cols:
        plt.figure(figsize=(5,3))
        sns.histplot(X_train_df[c].dropna(), kde=True, bins=30)
        plt.title(f'{c} distribution (train)')
        save_fig(f"{out_prefix}_dist_{c}.png")
        plt.close()

    # Scatter: sentiment_lag_1 vs target (if present)
    if 'sentiment_lag_1' in X_train_df.columns:
        plt.figure(figsize=(5,4))
        sns.scatterplot(x=X_train_df['sentiment_lag_1'], y=y_train)
        plt.title('Sentiment (lag1) vs Target High (train)')
        save_fig(f"{out_prefix}_sentiment_vs_target.png")
        plt.close()

def evaluate_regression(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train baseline, LinearRegression, RandomForest, LightGBM, (optional) Keras.
    Return fitted models and metrics.
    """
    results = {}
    models = {}

    baseline_pred = X_test['high'].values
    results['baseline'] = evaluate_regression(y_test, baseline_pred)
    print("Baseline:", results['baseline'])

    # Pipeline to impute + scale
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_train_imputed = imputer.fit_transform(X_train)
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(imputer.transform(X_test))

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    results['linear_regression'] = evaluate_regression(y_test, lr_pred)
    models['linear_regression'] = ('lr', lr, imputer, scaler)
    print("Linear Regression:", results['linear_regression'])

    # Random Forest
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    results['random_forest'] = evaluate_regression(y_test, rf_pred)
    models['random_forest'] = ('rf', rf, imputer, scaler)
    print("Random Forest:", results['random_forest'])

    # LightGBM
    lgbm = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
    lgbm.fit(X_train_scaled, y_train)
    lgbm_pred = lgbm.predict(X_test_scaled)
    results['lightgbm'] = evaluate_regression(y_test, lgbm_pred)
    models['lightgbm'] = ('lgbm', lgbm, imputer, scaler)
    print("LightGBM:", results['lightgbm'])

    # Save models
    for name, tup in models.items():
        tag, m, imputer_obj, scaler_obj = tup
        joblib.dump(tup, os.path.join(OUT_DIR, "models", f"{name}.joblib"))
    # also save results
    res_df = pd.DataFrame(results).T
    res_df.to_csv(os.path.join(OUT_DIR, "model_results.csv"))
    print("Saved model results to outputs/model_results.csv")
    return models, results

def unsupervised_analysis(df, feature_cols):
    """
    Run KMeans, PCA, Agglomerative, and IsolationForest
    """
    print("=== Unsupervised analysis ===")
    X_unsup = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_unsup)

    # PCA (2 components for visualization)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(Xs)
    pca_df = pd.DataFrame(pcs, columns=['PC1','PC2'])
    pca_df.to_csv(os.path.join(OUT_DIR, "pca_components.csv"))
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'])
    plt.title('PCA (2 components)')
    save_fig("unsup_pca.png")
    plt.close()

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=0)
    labels = kmeans.fit_predict(Xs)
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], hue=labels, palette='tab10')
    plt.title('KMeans clusters (k=3) on PCA')
    save_fig("unsup_kmeans_pca.png")
    plt.close()

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=3)
    agg_labels = agg.fit_predict(Xs)
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], hue=agg_labels, palette='deep')
    plt.title('Agglomerative clusters on PCA')
    save_fig("unsup_agg_pca.png")
    plt.close()

    # IsolationForest for anomalies
    iso = IsolationForest(random_state=0)
    iso_pred = iso.fit_predict(Xs)
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=pca_df['PC1'], y=pca_df['PC2'], hue=iso_pred, palette='Set1')
    plt.title('IsolationForest (anomaly detection)')
    save_fig("unsup_iso_pca.png")
    plt.close()

    # Save clustering models
    joblib.dump((kmeans, scaler), os.path.join(OUT_DIR, "models", "kmeans.joblib"))
    joblib.dump((agg, scaler), os.path.join(OUT_DIR, "models", "agg.joblib"))
    joblib.dump((iso, scaler), os.path.join(OUT_DIR, "models", "iso.joblib"))

    return {
        'pca': pca,
        'kmeans': kmeans,
        'agg': agg,
        'isolation_forest': iso
    }

def time_series_validation(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_index, val_index in tscv.split(X):
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
        # simple pipeline: impute -> scale -> fit model
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(imputer.fit_transform(X_tr))
        X_val_scaled = scaler.transform(imputer.transform(X_val))
        model.fit(X_tr_scaled, y_tr)
        pred = model.predict(X_val_scaled)
        scores.append(rmse(y_val, pred))
    return np.mean(scores), np.std(scores)

def optuna_tune_rf(X_train, y_train, n_trials=30):
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 500)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        max_features = trial.suggest_categorical('max_features', ['auto','sqrt','log2', 0.5, None])

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            random_state=42,
            n_jobs=-1
        )
        # cross-validate using simple 3-fold (not time series CV for speed)
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        Xs = scaler.fit_transform(imputer.fit_transform(X_train))
        scores = -cross_val_score(model, Xs, y_train, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print("RF best params:", study.best_params)
    return study.best_params

def optuna_tune_lgb(X_train, y_train, n_trials=30):
    def objective(trial):
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 16, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': 1
        }
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        Xs = scaler.fit_transform(imputer.fit_transform(X_train))
        dtrain = lgb.Dataset(Xs, label=y_train)
        cvres = lgb.cv(param, dtrain, nfold=3, metrics='rmse', early_stopping_rounds=20, verbose_eval=False)
        return min(cvres['rmse-mean'])

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print("LightGBM best params:", study.best_params)
    return study.best_params

def shap_explain(model_tuple, X_train, X_test, feature_names, model_name="model"):
    """
    model_tuple: (tag, model, imputer, scaler) saved earlier
    """
    tag, model, imputer, scaler = model_tuple
    # preprocess
    X_train_pre = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_pre = scaler.transform(imputer.transform(X_test))

    # SHAP for tree models (RandomForest, LightGBM)
    explainer = None
    if hasattr(model, 'predict'):
        try:
            if 'LGBM' in str(type(model)) or 'lightgbm' in str(type(model)).lower():
                explainer = shap.TreeExplainer(model)
            elif 'RandomForest' in str(type(model)) or 'forest' in str(type(model)).lower():
                explainer = shap.TreeExplainer(model)
            else:
                # fallback to KernelExplainer (slower)
                explainer = shap.KernelExplainer(model.predict, X_train_pre[:100])
        except Exception as e:
            print("SHAP explainer construction failed:", e)
            return
    else:
        print("Model object not suitable for SHAP")
        return

    shap_values = explainer.shap_values(X_test_pre[:200])  # limit for speed
    # summary plot
    plt.figure(figsize=(6,4))
    shap.summary_plot(shap_values, X_test_pre[:200], feature_names=feature_names, show=False)
    save_fig(f"shap_summary_{model_name}.png")
    plt.close()

    # feature importance as table
    mean_abs = np.abs(shap_values).mean(axis=0)
    fi = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    fi.to_csv(os.path.join(OUT_DIR, f"shap_feature_importance_{model_name}.csv"))
    print(f"Saved SHAP feature importance for {model_name}")

def run_full_pipeline():
    df = load_merged_df()
    df, feature_cols = preprocess_and_engineer(df, debug=True)

    # train/test split: use temporal split (no shuffle)
    # Keep last 20% as test
    n = len(df)
    test_size = int(0.2 * n)
    train_df = df.iloc[:-test_size].reset_index(drop=True)
    test_df = df.iloc[-test_size:].reset_index(drop=True)

    # Use training-only EDA
    X_train = train_df[feature_cols]
    y_train = train_df['target_high']
    training_only_eda(X_train, y_train, out_prefix="train")

    # Unsupervised analysis on full dataset features (or training only)
    unsup_models = unsupervised_analysis(train_df, feature_cols)

    # Prepare X/y
    X_test = test_df[feature_cols]
    y_test = test_df['target_high']

    # Train supervised models and evaluate
    models, results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Validation (TimeSeries CV) for RandomForest baseline
    rf_tuple = models.get('random_forest')
    if rf_tuple:
        _, rf_model, rf_imputer, rf_scaler = rf_tuple
        # do a time-series cv on the entire training set for the raw estimator
        try:
            mean_rmse, std_rmse = time_series_validation(RandomForestRegressor(n_estimators=100), train_df[feature_cols], train_df['target_high'], n_splits=5)
            print(f"TimeSeriesCV (naive RF) RMSE: mean {mean_rmse:.4f}, std {std_rmse:.4f}")
        except Exception as e:
            print("TimeSeries CV failed:", e)

    # Hyperparameter tuning (Optuna) - limited trials for runtime
    print("Starting Optuna tuning (this may take a while)...")
    try:
        best_rf_params = optuna_tune_rf(X_train, y_train, n_trials=20)
        best_lgb_params = optuna_tune_lgb(X_train, y_train, n_trials=20)
    except Exception as e:
        print("Optuna tuning failed or interrupted:", e)
        best_rf_params, best_lgb_params = None, None

    # If tuning returned params, fit tuned models and evaluate on test
    tuned_results = {}
    if best_rf_params:
        rf_tuned = RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1)
        # fit using pipeline procedure
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(imputer.fit_transform(X_train))
        rf_tuned.fit(Xtr, y_train)
        Xte = scaler.transform(imputer.transform(X_test))
        rf_pred = rf_tuned.predict(Xte)
        tuned_results['rf_tuned'] = evaluate_regression(y_test, rf_pred)
        joblib.dump(('rf_tuned', rf_tuned, imputer, scaler), os.path.join(OUT_DIR, "models", "rf_tuned.joblib"))
        print("RF tuned results:", tuned_results['rf_tuned'])
    if best_lgb_params:
        lgb_tuned = lgb.LGBMRegressor(**best_lgb_params, random_state=42)
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(imputer.fit_transform(X_train))
        lgb_tuned.fit(Xtr, y_train)
        Xte = scaler.transform(imputer.transform(X_test))
        lgb_pred = lgb_tuned.predict(Xte)
        tuned_results['lgb_tuned'] = evaluate_regression(y_test, lgb_pred)
        joblib.dump(('lgb_tuned', lgb_tuned, imputer, scaler), os.path.join(OUT_DIR, "models", "lgb_tuned.joblib"))
        print("LightGBM tuned results:", tuned_results['lgb_tuned'])

    # Final evaluation: Gather all result rows and write CSV
    all_results = []
    # results dict earlier contains baseline, linear_regression, random_forest, lightgbm, maybe keras_nn
    for name, metrics in results.items():
        row = {'model': name}
        row.update(metrics)
        all_results.append(row)
    for name, metrics in tuned_results.items():
        row = {'model': name}
        row.update(metrics)
        all_results.append(row)
    res_df = pd.DataFrame(all_results).set_index('model')
    res_df.to_csv(os.path.join(OUT_DIR, "final_evaluation_results.csv"))
    print("Saved final evaluation results to outputs/final_evaluation_results.csv")

    if not res_df.empty:
        best_model_name = res_df['RMSE'].idxmin()
        print("Best model by RMSE:", best_model_name)
        model_tuple = None
        # try load saved version
        try:
            model_tuple = joblib.load(os.path.join(OUT_DIR, "models", f"{best_model_name}.joblib"))
        except Exception:
            # try tuned
            try:
                model_tuple = joblib.load(os.path.join(OUT_DIR, "models", f"{best_model_name}_tuned.joblib"))
            except Exception:
                # attempt to use models dict
                if best_model_name in models:
                    model_tuple = models[best_model_name]
        if model_tuple:
            try:
                shap_explain(model_tuple, X_train, X_test, feature_cols, model_name=best_model_name)
            except Exception as e:
                print("SHAP explanation failed:", e)
        else:
            print("Could not find model object for SHAP explanation:", best_model_name)
    print("Pipeline complete.")

if __name__ == "__main__":
    try:
        run_full_pipeline()
    except Exception as e:
        print("Pipeline error:", e)
        raise
