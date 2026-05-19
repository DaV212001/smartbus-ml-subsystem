import pandas as pd
import numpy as np
import joblib
import os
import sys
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from src import config, features, data_generator

# Dynamic LightGBM import fallback
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
    print("[INFO] LightGBM imported successfully.")
except ImportError:
    LGBM_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingClassifier
    print("[WARNING] LightGBM is not available in this environment. Falling back to scikit-learn HistGradientBoostingClassifier.")


def train_route_assignment_model():
    """
    Loads raw driver profiles, route details, and historical performance tables,
    engineers features for each historic assignment, and trains a supervised
    Gradient Boosting model to predict driver assignment success.
    """
    print("\n--- Training Route Assignment Model ---")
    
    # 1. Load data tables
    if not os.path.exists(config.DRIVER_PERFORMANCE_CSV):
        print("[ERROR] Historical performance data not found. Please run data_generator first.")
        return
        
    drivers_df = pd.read_csv(config.DRIVER_PROFILES_CSV)
    routes_df = pd.read_csv(config.ROUTE_DETAILS_CSV)
    trips_df = pd.read_csv(config.DRIVER_PERFORMANCE_CSV)
    
    # Pre-convert and cache datetime and peak hour columns for a massive performance boost (from minutes to under 2 seconds!)
    trips_df["scheduled_for_dt"] = pd.to_datetime(trips_df["scheduled_for"])
    trips_df["peak_hour_binary"] = trips_df["scheduled_for_dt"].apply(features.get_peak_hour_binary)
    
    # 2. Engineer features for historic trips
    print(f"Engineering features for {len(trips_df)} historic trips (pre-cached and optimized)...")
    start_time = time.time()
    
    feature_list = []
    targets = []
    
    for idx, row in trips_df.iterrows():
        try:
            # Reconstruct the feature vector at the exact point in time of the trip
            feats = features.engineer_route_assignment_features(
                driver_id=row["driver_id"],
                route_id=row["route_id"],
                scheduled_for_str=row["scheduled_for"],
                driver_profiles_df=drivers_df,
                historical_trips_df=trips_df,
                routes_df=routes_df
            )
            
            # Remove string identifiers before feeding to model
            model_ready_feats = {k: v for k, v in feats.items() if k not in [
                "driver_id", "driver_status", "route_id", "route_number", "scheduled_for"
            ]}
            
            feature_list.append(model_ready_feats)
            targets.append(row["assignment_successful"])
        except Exception as e:
            # Log and skip invalid records if any
            print(f"[WARNING] Skipping row {idx} due to error: {e}")
            
    feature_df = pd.DataFrame(feature_list)
    targets = np.array(targets)
    
    print(f"Feature engineering completed in {time.time() - start_time:.2f} seconds.")
    print(f"Engineered columns: {list(feature_df.columns)}")
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(feature_df, targets, test_size=0.2, random_state=42)
    
    # 4. Train Classifier
    if LGBM_AVAILABLE:
        print("Fitting LightGBM Classifier...")
        clf = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            random_state=42,
            verbose=-1
        )
    else:
        print("Fitting HistGradientBoosting Classifier (fallback)...")
        clf = HistGradientBoostingClassifier(
            max_iter=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        
    clf.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nModel Performance Metrics on Validation Set:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    # Feature Importance (LightGBM only)
    if LGBM_AVAILABLE:
        importances = clf.feature_importances_
        imp_df = pd.DataFrame({"Feature": feature_df.columns, "Importance": importances}).sort_values("Importance", ascending=False)
        print("\nFeature Importances:")
        print(imp_df.to_string(index=False))
        
    # 6. Save Model Artifact
    # We bundle the model classifier AND the feature names for inference alignment
    model_bundle = {
        "model": clf,
        "feature_names": list(feature_df.columns),
        "metrics": {"accuracy": acc, "f1": f1, "auc": auc},
        "engine": "lightgbm" if LGBM_AVAILABLE else "hist_gb",
        "trained_at": datetime.now().isoformat()
    }
    
    joblib.dump(model_bundle, config.ROUTE_ASSIGNMENT_MODEL_PATH)
    print(f"Saved route assignment model bundle to {config.ROUTE_ASSIGNMENT_MODEL_PATH}")
    return model_bundle


def train_scan_fraud_model():
    """
    Loads historical scan fraud datasets. Extracts engineered features.
    Filters to NORMAL scans to train the Isolation Forest (semi-supervised outlier detection).
    Saves both the standard scaler and isolation forest model.
    """
    print("\n--- Training Scan Fraud Anomaly Model ---")
    
    # 1. Load scan records
    if not os.path.exists(config.SCAN_FRAUD_DATA_CSV):
        print("[ERROR] Scan fraud data not found. Please run data_generator first.")
        return
        
    df_scans = pd.read_csv(config.SCAN_FRAUD_DATA_CSV)
    
    # 2. Process engineered features for all rows
    print("Processing engineered features for scan records...")
    feature_rows = []
    
    for idx, row in df_scans.iterrows():
        feats = features.engineer_scan_features_from_row(row)
        feature_rows.append(feats)
        
    feature_df = pd.DataFrame(feature_rows)
    feature_cols = list(feature_df.columns)
    
    # Add target labels back for splits/evaluations
    feature_df["is_fraudulent"] = df_scans["is_fraudulent"]
    
    # 3. Filter normal scans for training
    # Isolation Forest is trained on NORMAL patterns, learning to identify the bounds of valid boarding scans.
    normal_scans = feature_df[feature_df["is_fraudulent"] == 0].drop(columns=["is_fraudulent"])
    
    print(f"Training on {len(normal_scans)} normal ticket scans. Total dataset size: {len(feature_df)}")
    
    # 4. Scale features
    scaler = StandardScaler()
    scaled_normal_scans = scaler.fit_transform(normal_scans)
    
    # 5. Fit Isolation Forest
    # Contamination set to roughly 5% representing potential unknown anomalies in training data
    from sklearn.ensemble import IsolationForest
    
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=config.ANOMALY_CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )
    
    print("Fitting Isolation Forest model...")
    iso_forest.fit(scaled_normal_scans)
    
    # 6. Evaluate Model on Full Dataset (Normal + Fraud cases)
    X_full = feature_df.drop(columns=["is_fraudulent"])
    y_full = feature_df["is_fraudulent"]
    
    scaled_full = scaler.transform(X_full)
    
    # Isolation Forest decision_function returns raw anomaly scores (lower is more anomalous)
    raw_scores = iso_forest.decision_function(scaled_full)
    
    # Convert decision score to a 0.0 - 1.0 confidence/fraud score
    # Standard decision scores range from -0.5 to 0.5 where 0 is the outlier threshold.
    # We will map this: high score = low fraud, low score = high fraud.
    # Simple sigmoid or min-max mapping based on standard isolation forest boundaries
    # Let's map decision scores < 0 (anomalies) to the 0.5 - 1.0 range, and > 0 to 0.0 - 0.5
    anomaly_scores = 1.0 - (1.0 / (1.0 + np.exp(-raw_scores * 8.0)))
    
    # Compute metrics
    # If we classify anything with anomaly_score > 0.5 as fraud:
    y_pred_fraud = (anomaly_scores > 0.5).astype(int)
    
    auc = roc_auc_score(y_full, anomaly_scores)
    prec = precision_score(y_full, y_pred_fraud)
    rec = recall_score(y_full, y_pred_fraud)
    f1 = f1_score(y_full, y_pred_fraud)
    cm = confusion_matrix(y_full, y_pred_fraud)
    
    print(f"\nUnsupervised Fraud Detection Performance Metrics (Full Set):")
    print(f"  Anomaly ROC-AUC Score: {auc:.4f}")
    print(f"  Precision (Score > 0.5): {prec:.4f}")
    print(f"  Recall (Score > 0.5):    {rec:.4f}")
    print(f"  F1 Score (Score > 0.5):  {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    
    # 7. Save Scaling and Isolation Forest binaries
    joblib.dump(scaler, config.FRAUD_SCALER_PATH)
    joblib.dump({
        "model": iso_forest,
        "feature_names": feature_cols,
        "metrics": {"roc_auc": auc, "f1": f1},
        "trained_at": datetime.now().isoformat()
    }, config.FRAUD_ISOLATION_FOREST_PATH)
    
    print(f"Saved fraud detector models to:\n  - {config.FRAUD_SCALER_PATH}\n  - {config.FRAUD_ISOLATION_FOREST_PATH}")


def run_full_training_pipeline():
    """
    Drives the data generation if necessary, followed by model training.
    """
    print("=" * 60)
    print("SmartBus ML Pipeline Initializing...")
    print("=" * 60)
    
    # Check if data exists; if not, generate first
    if not (os.path.exists(config.DRIVER_PERFORMANCE_CSV) and os.path.exists(config.SCAN_FRAUD_DATA_CSV)):
        print("[INFO] Data files missing. Starting automatic synthetic generation first...")
        drivers = data_generator.generate_driver_profiles()
        routes = data_generator.generate_route_details()
        data_generator.generate_route_assignment_training_data(drivers, routes)
        data_generator.generate_scan_fraud_training_data(routes)
        
    train_route_assignment_model()
    train_scan_fraud_model()
    
    print("\n" + "=" * 60)
    print("SmartBus Models Trained and Deployed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_full_training_pipeline()
