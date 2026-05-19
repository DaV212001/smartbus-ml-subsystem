import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from src import config, features, train

# Initialize FastAPI application
app = FastAPI(
    title="SmartBus ML Subsystem",
    description="Academically complete ML microservice providing Route Driver Suggestions and Ticketing Scan Anomaly/Fraud Detection.",
    version="1.0.0"
)

# ─── Shared-secret authentication ─────────────────────────────────────────────
# When ML_INTERNAL_TOKEN is set, every request (except /health) must carry a
# matching X-Internal-Token header. Used to keep the public Render URL closed
# to anyone but the SmartBus NestJS backend.
_INTERNAL_TOKEN = os.environ.get("ML_INTERNAL_TOKEN", "").strip()

@app.middleware("http")
async def require_internal_token(request: Request, call_next):
    # Health checks must remain open so Render's load balancer can probe.
    if not _INTERNAL_TOKEN or request.url.path == "/health":
        return await call_next(request)
    provided = request.headers.get("X-Internal-Token", "")
    if provided != _INTERNAL_TOKEN:
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid X-Internal-Token header"},
        )
    return await call_next(request)

# Global variables for models and data tables loaded in-memory
models_and_data: Dict[str, Any] = {}

def calculate_haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """
    Computes the great-circle distance between two GPS coordinates in meters.
    Used for geographic deviation anomaly detection.
    """
    # Convert latitude and longitude to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000.0  # Radius of earth in meters
    return round(c * r, 2)


@app.on_event("startup")
def startup_event():
    """
    FastAPI startup event. Loads model artifacts, scaler, and baseline database csv files
    in-memory. Caches datetime variables for faster real-time feature engineering.
    """
    print("[STARTUP] Loading model artifacts and operational databases...")
    
    # 1. Load Route Assignment model bundle
    if os.path.exists(config.ROUTE_ASSIGNMENT_MODEL_PATH):
        models_and_data["route_model_bundle"] = joblib.load(config.ROUTE_ASSIGNMENT_MODEL_PATH)
        print(f"[STARTUP] Loaded Route Assignment model ({models_and_data['route_model_bundle']['engine']}).")
    else:
        print("[STARTUP] [WARNING] Route Assignment model binary not found. Run /api/v1/ml/train to generate.")
        
    # 2. Load Fraud Detection Isolation Forest and standard scaler
    if os.path.exists(config.FRAUD_ISOLATION_FOREST_PATH) and os.path.exists(config.FRAUD_SCALER_PATH):
        models_and_data["fraud_model_bundle"] = joblib.load(config.FRAUD_ISOLATION_FOREST_PATH)
        models_and_data["fraud_scaler"] = joblib.load(config.FRAUD_SCALER_PATH)
        print("[STARTUP] Loaded Fraud Anomaly Detection models (Scaler + Isolation Forest).")
    else:
        print("[STARTUP] [WARNING] Fraud detection model binary not found. Run /api/v1/ml/train to generate.")

    # 3. Load historical operational records in memory for feature lookup
    if os.path.exists(config.DRIVER_PROFILES_CSV) and os.path.exists(config.DRIVER_PERFORMANCE_CSV) and os.path.exists(config.ROUTE_DETAILS_CSV):
        models_and_data["drivers_df"] = pd.read_csv(config.DRIVER_PROFILES_CSV)
        models_and_data["routes_df"] = pd.read_csv(config.ROUTE_DETAILS_CSV)
        
        # Pre-load and cache datetime columns in memory to ensure sub-millisecond lookups during inference
        trips_df = pd.read_csv(config.DRIVER_PERFORMANCE_CSV)
        trips_df["scheduled_for_dt"] = pd.to_datetime(trips_df["scheduled_for"])
        trips_df["peak_hour_binary"] = trips_df["scheduled_for_dt"].apply(features.get_peak_hour_binary)
        models_and_data["trips_df"] = trips_df
        
        print(f"[STARTUP] Cached {len(trips_df)} historic trips and operational tables in-memory.")
    else:
        print("[STARTUP] [WARNING] Operational CSV databases not found. Retraining will generate them.")

    # 4. In-Memory state for sliding scan frequency window (simulating real-time ticketing streams)
    models_and_data["realtime_scan_history"] = []  # list of dicts: {'timestamp': datetime, 'passenger_id': str, 'device_id': str}


# ==========================================
# PYDANTIC SCHEMAS (Request/Response Validation)
# ==========================================

class RouteAssignmentRequest(BaseModel):
    routeId: str = Field(..., description="Target route identifier", example="R03")
    scheduledFor: str = Field(..., description="ISO 8601 Datetime string of scheduled trip", example="2026-05-20T08:30:00")
    candidateDriverIds: List[str] = Field(..., description="List of drivers available for assignment", example=["D001", "D005", "D012", "D025"])

class DriverSuggestion(BaseModel):
    driverId: str = Field(..., description="Driver identifier")
    driverName: str = Field(..., description="Driver full name")
    confidence: float = Field(..., description="Confidence probability score (0.0 to 1.0) of trip success")
    reasons: List[str] = Field(..., description="Explainable AI justifications derived from engineered features")

class RouteAssignmentResponse(BaseModel):
    routeId: str = Field(..., description="Target route identifier")
    suggestions: List[DriverSuggestion] = Field(..., description="Ranked suggestions with justifications")

class ScanMetadata(BaseModel):
    latitude: float = Field(..., description="GPS Latitude of mobile scan", example=9.025)
    longitude: float = Field(..., description="GPS Longitude of mobile scan", example=38.765)
    deviceId: str = Field(..., description="Mobile bus ticketing device unique identifier", example="DEV-5021")

class TicketContext(BaseModel):
    ticketId: str = Field(..., description="Ticket unique identifier", example="TKT-100234")
    passengerId: str = Field(..., description="Passenger account identifier", example="P-45091")
    fareAmount: float = Field(..., description="Fare price paid in local currency", example=35.0)
    purchasedAt: str = Field(..., description="ISO datetime of ticket purchase", example="2026-05-19T17:10:00")
    expiresAt: str = Field(..., description="ISO datetime of ticket expiration", example="2026-05-19T19:10:00")
    qrSignatureValid: bool = Field(..., description="Cryptographic QR validity flag from public key check", example=True)

class StopContext(BaseModel):
    id: str = Field(..., description="Route stop identifier", example="BS-R03-05")
    latitude: float = Field(..., description="Stop latitude", example=9.024)
    longitude: float = Field(..., description="Stop longitude", example=38.764)

class ScanAnomalyRequest(BaseModel):
    eventId: str = Field(..., description="Ticketing transaction unique event identifier", example="EV-90234")
    result: str = Field(..., description="Local validation outcome: VALID | EXPIRED | ALREADY_USED | INVALID_SIGNATURE", example="VALID")
    isOffline: bool = Field(..., description="True if transaction occurred when device was offline", example=False)
    scannedAt: str = Field(..., description="ISO datetime of ticketing scan", example="2026-05-19T17:20:00")
    syncedAt: str = Field(..., description="ISO datetime of API server synchronization", example="2026-05-19T17:20:15")
    syncDelaySeconds: float = Field(..., description="Elapsed sync time", example=15.0)
    scanMetadata: ScanMetadata = Field(..., description="Geographic and physical device details")
    ticketContext: TicketContext = Field(..., description="Cryptographic ticket metadata details")
    boardingStop: StopContext = Field(..., description="Expected route stop geographic position")

class ScanAnomalyResponse(BaseModel):
    eventId: str = Field(..., description="Scan event unique identifier")
    anomalyScore: float = Field(..., description="Normalized fraud risk confidence score (0.0 to 1.0)")
    severity: str = Field(..., description="Threat response classification: LOW | MEDIUM | HIGH")
    reasons: List[str] = Field(..., description="Flagged rule violations and Isolation Forest outlier reasons")


# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Exposes microservice operational health, loaded model engines, and accuracy metrics
    supporting university thesis verification.
    """
    status_info = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": config.ENV,
        "models_loaded": {
            "route_assignment_model": "route_model_bundle" in models_and_data,
            "scan_fraud_model": "fraud_model_bundle" in models_and_data
        }
    }
    
    if "route_model_bundle" in models_and_data:
        bundle = models_and_data["route_model_bundle"]
        status_info["route_assignment_metadata"] = {
            "engine": bundle["engine"],
            "trained_at": bundle["trained_at"],
            "validation_accuracy": round(bundle["metrics"]["accuracy"], 4),
            "validation_f1": round(bundle["metrics"]["f1"], 4),
            "validation_auc": round(bundle["metrics"]["auc"], 4)
        }
        
    if "fraud_model_bundle" in models_and_data:
        bundle = models_and_data["fraud_model_bundle"]
        status_info["scan_fraud_metadata"] = {
            "engine": "IsolationForest",
            "trained_at": bundle["trained_at"],
            "contamination": config.ANOMALY_CONTAMINATION,
            "validation_roc_auc": round(bundle["metrics"]["roc_auc"], 4),
            "validation_f1": round(bundle["metrics"]["f1"], 4)
        }
        
    return status_info


@app.post("/api/v1/ml/route-assignment", response_model=RouteAssignmentResponse)
def get_route_assignment_suggestions(request: RouteAssignmentRequest):
    """
    Supervised predictive route assignment recommendation endpoint.
    Takes candidate drivers, dynamically engineers features without data leakage,
    predicts successful trip probabilities using LightGBM, and justifies rankings.
    """
    # Validate model loading
    if "route_model_bundle" not in models_and_data:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Route assignment ML model is currently not loaded. Trigger training first."
        )
        
    route_bundle = models_and_data["route_model_bundle"]
    clf = route_bundle["model"]
    feature_names = route_bundle["feature_names"]
    
    drivers_df = models_and_data.get("drivers_df")
    routes_df = models_and_data.get("routes_df")
    trips_df = models_and_data.get("trips_df")
    
    if drivers_df is None or routes_df is None or trips_df is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Historical databases are missing in memory."
        )

    suggestions = []
    
    for driver_id in request.candidateDriverIds:
        # Check driver profile validity
        d_profile = drivers_df[drivers_df["driver_id"] == driver_id]
        if d_profile.empty:
            # Downrank missing drivers with a neutral prior
            print(f"[INFERENCE] [WARNING] Candidate driver {driver_id} not found in database profiles.")
            continue
            
        driver_name = d_profile.iloc[0]["full_name"]
        driver_status = d_profile.iloc[0]["status"]
        
        # Ineligible status check
        if driver_status in ["DISABLED", "PENDING_VERIFICATION"]:
            suggestions.append(DriverSuggestion(
                driverId=driver_id,
                driverName=driver_name,
                confidence=0.0,
                reasons=[f"Driver status is currently {driver_status}"]
            ))
            continue

        try:
            # 1. Dynamic Feature Engineering
            feats = features.engineer_route_assignment_features(
                driver_id=driver_id,
                route_id=request.routeId,
                scheduled_for_str=request.scheduledFor,
                driver_profiles_df=drivers_df,
                historical_trips_df=trips_df,
                routes_df=routes_df
            )
            
            # 2. Vector alignment (ensure columns match training order)
            vector = [feats[col] for col in feature_names]
            X = pd.DataFrame([vector], columns=feature_names)
            
            # 3. Model Inference
            # predict_proba returns probability for [0 (failure), 1 (success)]
            prob_success = float(clf.predict_proba(X)[0][1])
            
            # 4. Reason Generator (Explainability Engine)
            reasons = []
            exp_route = feats["driver_experience_on_route"]
            comp_rate = feats["trip_completion_rate"]
            delay = feats["average_trip_delay_minutes"]
            anomaly = feats["anomaly_rate"]
            recent_act = feats["trips_completed_last_30_days"]
            is_peak = feats["peak_hour_binary"]
            
            if exp_route >= 15:
                reasons.append(f"High route familiarity (+{exp_route:.0f} completed trips)")
            elif exp_route >= 5:
                reasons.append(f"Moderate route familiarity ({exp_route:.0f} completed trips)")
                
            if comp_rate >= 0.94:
                reasons.append(f"Strong historical completion rate ({comp_rate*100:.1f}%)")
                
            if anomaly <= 0.05:
                reasons.append("Exceptionally low operational anomaly rate")
                
            if delay < 4.0:
                reasons.append("Excellent schedule adherence (low average delay)")
                
            if recent_act >= 12:
                reasons.append("High recent driver activity and availability")
                
            if is_peak == 1 and anomaly < 0.08:
                reasons.append("Proven reliable peak-hour performance")
                
            # Default fallback reason if driver is new
            if not reasons:
                reasons.append("Eligible active driver with standard operational profile")
                
            suggestions.append(DriverSuggestion(
                driverId=driver_id,
                driverName=driver_name,
                confidence=round(prob_success, 3),
                reasons=reasons[:4]  # Return top 4 distinct reasons
            ))
            
        except Exception as e:
            # Handle unexpected lookup failures gracefully in UI
            print(f"[INFERENCE] [ERROR] Feature building failed for driver {driver_id}: {e}")
            suggestions.append(DriverSuggestion(
                driverId=driver_id,
                driverName=driver_name,
                confidence=0.5,
                reasons=["Baseline performance index (historical metrics processing error)"]
            ))

    # Sort candidates by confidence score in descending order
    suggestions.sort(key=lambda s: s.confidence, reverse=True)
    
    return RouteAssignmentResponse(
        routeId=request.routeId,
        suggestions=suggestions
    )


@app.post("/api/v1/ml/detect-anomaly", response_model=ScanAnomalyResponse)
def detect_scan_anomaly(request: ScanAnomalyRequest):
    """
    Scan ticket fraud and anomaly detection endpoint.
    Merges deterministic business rule checks (duplicate use, expired validity, cryptographic signature fail)
    with unsupervised Isolation Forest outlier analysis for zero-day/burst fraud vector discovery.
    """
    if "fraud_model_bundle" not in models_and_data or "fraud_scaler" not in models_and_data:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Fraud Anomaly detection models not loaded. Trigger training first."
        )

    fraud_bundle = models_and_data["fraud_model_bundle"]
    scaler = models_and_data["fraud_scaler"]
    clf = fraud_bundle["model"]
    feature_names = fraud_bundle["feature_names"]
    
    # --- Geolocation Distance Deviation Calculation ---
    geo_deviation = calculate_haversine_distance(
        lat1=request.scanMetadata.latitude,
        lon1=request.scanMetadata.longitude,
        lat2=request.boardingStop.latitude,
        lon2=request.boardingStop.longitude
    )

    # --- Stateful sliding window check for Real-Time frequencies ---
    # Register this scan in the memory stream
    now = datetime.now()
    models_and_data["realtime_scan_history"].append({
        "timestamp": now,
        "passenger_id": request.ticketContext.passengerId,
        "device_id": request.scanMetadata.deviceId
    })
    
    # Purge scans older than 5 minutes to maintain efficient cache memory
    five_mins_ago = now - pd.Timedelta(minutes=5)
    models_and_data["realtime_scan_history"] = [
        s for s in models_and_data["realtime_scan_history"] if s["timestamp"] >= five_mins_ago
    ]
    
    # Calculate rolling sliding window counts
    passenger_scan_freq = sum(1 for s in models_and_data["realtime_scan_history"] if s["passenger_id"] == request.ticketContext.passengerId)
    device_scan_freq = sum(1 for s in models_and_data["realtime_scan_history"] if s["device_id"] == request.scanMetadata.deviceId)
    
    # --- Construct unified row dictionary for feature engineering ---
    scan_row = {
        "result": request.result,
        "is_offline": int(request.isOffline),
        "sync_delay_seconds": request.syncDelaySeconds,
        "qr_signature_valid": int(request.ticketContext.qrSignatureValid),
        "geo_distance_deviation_meters": geo_deviation,
        "passenger_scan_frequency": passenger_scan_freq,
        "device_scan_frequency": device_scan_freq,
        "scanned_at": request.scannedAt
    }
    
    # 1. Feature Engineering
    engineered_feats = features.engineer_scan_features_from_row(scan_row)
    
    # 2. Vector formatting and Scaling
    vector = [engineered_feats[col] for col in feature_names]
    scaled_vector = scaler.transform([vector])
    
    # 3. Isolation Forest scoring
    # raw_decision_score ranges roughly from -0.5 (heavy anomaly) to 0.5 (completely normal)
    raw_score = float(clf.decision_function(scaled_vector)[0])
    
    # Sigmoid calibration mapping raw outlier scoring into a robust 0.0 - 1.0 scale
    iso_anomaly_score = 1.0 - (1.0 / (1.0 + np.exp(-raw_score * 8.0)))
    
    # --- MERGED DEFENSE LAYER (Deterministic Rules + ML) ---
    reasons = []
    rule_fraud_score = 0.0
    
    # Rule A: Cryptographic integrity failure (CRITICAL)
    if not request.ticketContext.qrSignatureValid or request.result == "INVALID_SIGNATURE":
        reasons.append("Invalid cryptographic ticket signature (forged ticket attempt)")
        rule_fraud_score = max(rule_fraud_score, 1.0)
        
    # Rule B: Direct Ticket Reuse (CRITICAL)
    if request.result == "ALREADY_USED":
        reasons.append("Duplicate ticket scan detected (ticket has already been sync-validated)")
        rule_fraud_score = max(rule_fraud_score, 1.0)
        
    # Rule C: Expired Ticket boarding
    if request.result == "EXPIRED":
        reasons.append("Expired ticket presented at boarding gate")
        rule_fraud_score = max(rule_fraud_score, 0.95)
        
    # Rule D: Extreme GPS boarding distance deviation
    if geo_deviation > 600.0:
        reasons.append(f"Impossible geo-location deviation ({geo_deviation:.0f}m) from scheduled boarding stop")
        rule_fraud_score = max(rule_fraud_score, 0.85)
    elif geo_deviation > 200.0:
        reasons.append(f"Minor geographic boarding deviation ({geo_deviation:.0f}m)")
        rule_fraud_score = max(rule_fraud_score, 0.40)

    # Rule E: Rapid Ticket Sharing bursts (Ticketing Machine attacks)
    if passenger_scan_freq >= 4:
        reasons.append(f"Abnormal passenger scanning frequency ({passenger_scan_freq} scans in last 5m)")
        rule_fraud_score = max(rule_fraud_score, 0.75)
        
    if device_scan_freq >= 20:
        reasons.append(f"Suspicious boarding terminal scan burst ({device_scan_freq} scans in last 5m)")
        rule_fraud_score = max(rule_fraud_score, 0.70)
        
    # Rule F: Manipulation of offline sync uploads
    if request.isOffline and request.syncDelaySeconds > 172800:  # > 48 hours delay
        reasons.append(f"Extreme offline sync upload delay ({request.syncDelaySeconds/3600:.1f} hours)")
        rule_fraud_score = max(rule_fraud_score, 0.80)
        
    # ML Outlier trigger (Zero-day fraud vectors like device cloning / complex timing bursts)
    # Trigger if Isolation Forest detects a statistically abnormal vector not fully covered by basic rules
    if iso_anomaly_score > 0.65:
        reasons.append(f"Unusual scan context cluster (Outlier Factor: {iso_anomaly_score:.2f})")
        
    # Consolidate fraud scores
    final_anomaly_score = float(max(rule_fraud_score, iso_anomaly_score))
    
    # Severity classification mapping
    if final_anomaly_score >= 0.80:
        severity = "HIGH"
    elif final_anomaly_score >= 0.40:
        severity = "MEDIUM"
    else:
        severity = "LOW"
        
    # Standard fallback justification
    if not reasons:
        reasons.append("Standard commuter boarding check pass")
        
    return ScanAnomalyResponse(
        eventId=request.eventId,
        anomalyScore=round(final_anomaly_score, 3),
        severity=severity,
        reasons=reasons
    )


def background_retrain_task():
    """
    Background worker function that runs the complete ML training pipeline
    and refreshes the model binaries and cached databases in memory.
    """
    print("[BACKGROUND-WORKER] Launching model retraining pipeline...")
    try:
        train.run_full_training_pipeline()
        
        # Refresh in-memory state
        startup_event()
        print("[BACKGROUND-WORKER] Model retraining and memory reload completed successfully!")
    except Exception as e:
        print(f"[BACKGROUND-WORKER] [ERROR] Model retraining pipeline failed: {e}")


@app.post("/api/v1/ml/train", status_code=status.HTTP_202_ACCEPTED)
def trigger_model_retraining(background_tasks: BackgroundTasks):
    """
    Asynchronous training endpoint.
    Launches retraining pipeline in a background thread to prevent API blocking
    and return instant feedback.
    """
    background_tasks.add_task(background_retrain_task)
    return {
        "status": "training_triggered",
        "message": "Retraining pipeline initiated in the background. System health metrics will update upon completion.",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    # When run directly, start the uvicorn development server
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
