import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_peak_hour_binary(dt):
    """
    Peak hour check: 07:00-09:00 and 16:00-19:00.
    """
    hour = dt.hour
    return 1 if (7 <= hour <= 9 or 16 <= hour <= 19) else 0

def engineer_route_assignment_features(
    driver_id: str,
    route_id: str,
    scheduled_for_str: str,
    driver_profiles_df: pd.DataFrame,
    historical_trips_df: pd.DataFrame,
    routes_df: pd.DataFrame
) -> dict:
    """
    Dynamically engineers predictive features for a driver-route assignment candidate at a given scheduled time.
    Separates raw features from highly engineered variables.
    
    Returns a dictionary of raw and engineered features.
    """
    scheduled_time = pd.to_datetime(scheduled_for_str)
    
    # ------------------ RAW FEATURES ------------------
    # Basic lookup features from metadata
    driver_row = driver_profiles_df[driver_profiles_df["driver_id"] == driver_id]
    route_row = routes_df[routes_df["route_id"] == route_id]
    
    if driver_row.empty or route_row.empty:
        raise ValueError(f"Invalid driver_id ({driver_id}) or route_id ({route_id}) provided.")
        
    driver_status = driver_row.iloc[0]["status"]
    
    route_num = route_row.iloc[0]["route_number"]
    estimated_duration = float(route_row.iloc[0]["estimated_duration"])
    estimated_distance = float(route_row.iloc[0]["estimated_distance"])
    total_stops = int(route_row.iloc[0]["total_stops"])
    
    # Make sure datetime columns are pre-processed for speed
    # We check if they exist, otherwise we fallback to converting on the fly
    if "scheduled_for_dt" in historical_trips_df.columns:
        # Fast path
        trips_history = historical_trips_df[
            (historical_trips_df["driver_id"] == driver_id) & 
            (historical_trips_df["scheduled_for_dt"] < scheduled_time)
        ]
        
        route_trips = historical_trips_df[
            (historical_trips_df["route_id"] == route_id) &
            (historical_trips_df["scheduled_for_dt"] < scheduled_time)
        ]
    else:
        # Slow fallback path
        historical_trips_df_dt = pd.to_datetime(historical_trips_df["scheduled_for"])
        trips_history = historical_trips_df[
            (historical_trips_df["driver_id"] == driver_id) & 
            (historical_trips_df_dt < scheduled_time)
        ]
        
        route_trips = historical_trips_df[
            (historical_trips_df["route_id"] == route_id) &
            (historical_trips_df_dt < scheduled_time)
        ]
        # Create a local copy with dt to avoid multiple conversions below
        trips_history = trips_history.copy()
        trips_history["scheduled_for_dt"] = pd.to_datetime(trips_history["scheduled_for"])
        route_trips = route_trips.copy()
        route_trips["scheduled_for_dt"] = pd.to_datetime(route_trips["scheduled_for"])

    # ------------------ ENGINEERED FEATURES ------------------
    
    # Feature 1: driver_experience_on_route
    driver_experience_on_route = len(trips_history[
        (trips_history["route_id"] == route_id) & 
        (trips_history["status"] == "COMPLETED")
    ])
    
    # Feature 2: trip_completion_rate
    total_assigned_trips = len(trips_history)
    if total_assigned_trips > 0:
        trip_completion_rate = len(trips_history[trips_history["status"] == "COMPLETED"]) / total_assigned_trips
    else:
        trip_completion_rate = 0.85  # global naive prior
        
    # Feature 3: average_trip_delay_minutes
    completed_trips = trips_history[trips_history["status"] == "COMPLETED"]
    if len(completed_trips) > 0:
        average_trip_delay_minutes = completed_trips["delay_minutes"].mean()
    else:
        average_trip_delay_minutes = 5.0
        
    # Feature 4: route_complexity_score
    route_complexity_score = (total_stops / (estimated_distance + 0.1)) * (estimated_duration / 60.0)
    
    # Feature 5: average_passenger_load
    completed_route_trips = route_trips[route_trips["status"] == "COMPLETED"]
    if len(completed_route_trips) > 0:
        average_passenger_load = completed_route_trips["total_valid_scans"].mean()
    else:
        average_passenger_load = estimated_distance * 2.0
        
    # Feature 6: peak_hour_binary
    peak_hour_binary = get_peak_hour_binary(scheduled_time)
    
    # Feature 7: demand_by_time_window
    if "peak_hour_binary" in completed_route_trips.columns:
        matching_peak_trips = completed_route_trips[
            completed_route_trips["peak_hour_binary"] == peak_hour_binary
        ]
    else:
        matching_peak_trips = completed_route_trips[
            completed_route_trips["scheduled_for_dt"].apply(get_peak_hour_binary) == peak_hour_binary
        ]
        
    if len(matching_peak_trips) > 0:
        demand_by_time_window = matching_peak_trips["total_valid_scans"].mean()
    else:
        demand_by_time_window = average_passenger_load
        
    # Feature 8: recent_assignment_count (Last 7 days)
    seven_days_ago = scheduled_time - timedelta(days=7)
    recent_trips = trips_history[
        trips_history["scheduled_for_dt"] >= seven_days_ago
    ]
    recent_assignment_count = len(recent_trips)
    
    # Feature 9: trips_completed_last_30_days
    thirty_days_ago = scheduled_time - timedelta(days=30)
    trips_completed_last_30_days = len(trips_history[
        (trips_history["scheduled_for_dt"] >= thirty_days_ago) &
        (trips_history["status"] == "COMPLETED")
    ])
    
    # Feature 10: assignment_recency_days
    if len(trips_history) > 0:
        last_trip_time = trips_history["scheduled_for_dt"].max()
        assignment_recency_days = (scheduled_time - last_trip_time).total_seconds() / 86400.0
    else:
        assignment_recency_days = 90.0
        
    # Feature 11: anomaly_rate
    if len(trips_history) > 0:
        anomalous_trips = trips_history[
            (trips_history["status"] == "CANCELLED") | 
            ((trips_history["status"] == "COMPLETED") & (trips_history["delay_minutes"] > 15.0))
        ]
        anomaly_rate = len(anomalous_trips) / len(trips_history)
    else:
        anomaly_rate = 0.05
        
    return {
        # Raw Reference Features
        "driver_id": driver_id,
        "driver_status": driver_status,
        "route_id": route_id,
        "route_number": route_num,
        "scheduled_for": scheduled_for_str,
        
        # Engineered Features
        "driver_experience_on_route": float(driver_experience_on_route),
        "trip_completion_rate": float(trip_completion_rate),
        "average_trip_delay_minutes": float(average_trip_delay_minutes),
        "route_complexity_score": float(route_complexity_score),
        "average_passenger_load": float(average_passenger_load),
        "demand_by_time_window": float(demand_by_time_window),
        "recent_assignment_count": float(recent_assignment_count),
        "trips_completed_last_30_days": float(trips_completed_last_30_days),
        "assignment_recency_days": float(assignment_recency_days),
        "anomaly_rate": float(anomaly_rate),
        "peak_hour_binary": int(peak_hour_binary)
    }


def engineer_scan_features_from_row(scan_row: dict) -> dict:
    """
    Extracts and engineers numerical/categorical features from a single scan log row
    for feed-in to the Isolation Forest anomaly detector.
    
    Returns a dictionary of cleaned, model-ready features.
    """
    # Raw characteristics
    result = scan_row.get("result", "VALID")
    is_offline = int(scan_row.get("is_offline", 0))
    sync_delay = float(scan_row.get("sync_delay_seconds", 0.0))
    qr_sig_valid = int(scan_row.get("qr_signature_valid", 1))
    
    # Geographic dev
    geo_dev = float(scan_row.get("geo_distance_deviation_meters", 0.0))
    
    # Scan frequencies (from stateful sliding-window counters)
    p_freq = float(scan_row.get("passenger_scan_frequency", 1.0))
    d_freq = float(scan_row.get("device_scan_frequency", 1.0))
    
    # Timing
    try:
        scanned_at = pd.to_datetime(scan_row.get("scanned_at", datetime.now().isoformat()))
        scan_hour = scanned_at.hour
    except Exception:
        scan_hour = 12
        
    # Derived binaries
    duplicate_ticket = 1 if result == "ALREADY_USED" else 0
    ticket_expired = 1 if result == "EXPIRED" else 0
    sig_invalid = 1 if (not qr_sig_valid or result == "INVALID_SIGNATURE") else 0
    
    # ------------------ FEATURE SEGREGATION ------------------
    # Model-ready features must be strictly numerical
    return {
        "sync_delay_seconds": float(sync_delay),
        "is_offline": int(is_offline),
        "geo_distance_deviation_meters": float(geo_dev),
        "passenger_scan_frequency": float(p_freq),
        "device_scan_frequency": float(d_freq),
        "scan_hour": float(scan_hour),
        "duplicate_ticket_attempt": int(duplicate_ticket),
        "ticket_expired": int(ticket_expired),
        "qr_signature_invalid": int(sig_invalid),
        "scans_per_minute": float(d_freq / 5.0)  # simple density proxy
    }
