import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from src import config

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_driver_profiles(num_drivers=60):
    """
    Generates a list of driver profiles with stable, moderate, or poor hidden profiles
    to drive realistic correlations in historical operational data.
    """
    first_names = ["Abel", "Sarah", "Daniel", "Michael", "Elena", "Kebede", "Aster", "David", "James", "Sophia",
                   "Yonas", "Li", "Amina", "Robert", "Tariq", "Fatima", "Carlos", "Maria", "Kenji", "Jane"]
    last_names = ["Seyoum", "Smith", "Johnson", "Bekele", "Silva", "Tadesse", "Wang", "Davis", "Miller", "Jones",
                  "Rodriguez", "Sato", "Ali", "Okonkwo", "Müller", "Dubois", "Devi", "Kim", "Patel", "García"]
    
    drivers = []
    statuses = ["ACTIVE"] * 50 + ["PENDING_VERIFICATION"] * 5 + ["DISABLED"] * 5
    
    # Hidden behavior types for generation:
    # 0 = Excellent, 1 = Average, 2 = Prone to delays, 3 = Prone to cancellations
    hidden_types = [0] * 15 + [1] * 30 + [2] * 10 + [3] * 5
    random.shuffle(hidden_types)
    
    for i in range(num_drivers):
        driver_id = f"D{i+1:03d}"
        full_name = f"{random.choice(first_names)} {random.choice(last_names)}"
        status = statuses[i] if i < len(statuses) else "ACTIVE"
        
        # Each driver has 1 or 2 routes they are highly familiar with
        favorite_routes = [f"R{random.randint(1, 10):02d}", f"R{random.randint(1, 10):02d}"]
        favorite_routes = list(set(favorite_routes))  # remove duplicates
        
        # Driver base metrics
        h_type = hidden_types[i % len(hidden_types)]
        if h_type == 0:  # Excellent
            base_cancellation_rate = 0.01
            base_delay_minutes = 2.0
            base_delay_std = 2.0
        elif h_type == 1:  # Average
            base_cancellation_rate = 0.03
            base_delay_minutes = 5.0
            base_delay_std = 4.0
        elif h_type == 2:  # Prone to delays
            base_cancellation_rate = 0.04
            base_delay_minutes = 12.0
            base_delay_std = 8.0
        else:  # Prone to cancellations
            base_cancellation_rate = 0.18
            base_delay_minutes = 6.0
            base_delay_std = 5.0

        drivers.append({
            "driver_id": driver_id,
            "full_name": full_name,
            "status": status,
            "hidden_type": h_type,
            "base_cancellation_rate": base_cancellation_rate,
            "base_delay_minutes": base_delay_minutes,
            "base_delay_std": base_delay_std,
            "favorite_routes": favorite_routes
        })
        
    df = pd.DataFrame(drivers)
    # Drop hidden generation helpers before saving, but return the full dict for further generation
    df_save = df.drop(columns=["hidden_type", "base_cancellation_rate", "base_delay_minutes", "base_delay_std", "favorite_routes"])
    df_save.to_csv(config.DRIVER_PROFILES_CSV, index=False)
    print(f"Generated {len(df_save)} driver profiles at {config.DRIVER_PROFILES_CSV}")
    return drivers


def generate_route_details(num_routes=10):
    """
    Generates realistic route topologies: short urban, long highway, high-complexity, etc.
    """
    routes = []
    route_numbers = ["10", "22B", "45", "101", "72", "99A", "5", "88", "12", "300"]
    route_names = [
        "Stadium - Airport Express",
        "Megenagna - Bole Loop",
        "Piazza - Ayat Residential",
        "Arat Kilo - Wingate Ring Road",
        "Sar Bet - Jemo Urban Connect",
        "Mercato - Kaliti Commercial",
        "Lebu - Kotebe Commuter Link",
        "CMC - Churchill Ave Route",
        "Bole - Jackros suburban Line",
        "Addis Ketema - Akaki Outer Bypass"
    ]
    
    for i in range(num_routes):
        route_id = f"R{i+1:02d}"
        route_num = route_numbers[i % len(route_numbers)]
        name = route_names[i % len(route_names)]
        
        # Route styles: short (urban, many stops), medium, long (express, fewer stops)
        if i % 3 == 0:  # Short urban
            est_duration = random.randint(25, 40)
            est_distance = round(random.uniform(5.0, 9.0), 1)
            total_stops = random.randint(15, 25)
            complexity = 1.3  # multipliers for operational delays
        elif i % 3 == 1:  # Medium commuter
            est_duration = random.randint(45, 70)
            est_distance = round(random.uniform(10.0, 18.0), 1)
            total_stops = random.randint(20, 35)
            complexity = 1.0
        else:  # Long express
            est_duration = random.randint(75, 110)
            est_distance = round(random.uniform(22.0, 38.0), 1)
            total_stops = random.randint(8, 15)  # express has fewer stops
            complexity = 1.1
            
        # Geographic bounding box center for route stops simulation
        lat_center = 9.01 + (i * 0.015)
        lon_center = 38.75 + (i * 0.015)
        
        stops = []
        for seq in range(1, total_stops + 1):
            stops.append({
                "sequence": seq,
                "latitude": round(lat_center + random.uniform(-0.02, 0.02), 5),
                "longitude": round(lon_center + random.uniform(-0.02, 0.02), 5)
            })

        routes.append({
            "route_id": route_id,
            "route_number": route_num,
            "name": name,
            "estimated_duration": est_duration,
            "estimated_distance": est_distance,
            "total_stops": total_stops,
            "complexity_factor": complexity,
            "stops": stops
        })
        
    df = pd.DataFrame(routes)
    df_save = df.drop(columns=["stops", "complexity_factor"])
    df_save.to_csv(config.ROUTE_DETAILS_CSV, index=False)
    print(f"Generated {len(df_save)} route details at {config.ROUTE_DETAILS_CSV}")
    return routes


def generate_route_assignment_training_data(drivers, routes, sample_size=config.TRAINING_SAMPLE_SIZE):
    """
    Generates historical trip records matching drivers to routes.
    Includes realistic performance, delay, and cancellation correlations.
    Creates a target label 'assignment_successful' based on these correlations.
    """
    data = []
    active_drivers = [d for d in drivers if d["status"] == "ACTIVE"]
    
    # We will step backwards in time to create historical trips over the last 90 days
    base_date = datetime.now() - timedelta(days=90)
    
    for idx in range(sample_size):
        trip_id = f"T{idx+10001:05d}"
        driver = random.choice(active_drivers)
        route = random.choice(routes)
        
        # Schedule time logic (hour and day of week)
        days_offset = random.randint(0, 90)
        hour = random.choice([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
        minute = random.choice([0, 15, 30, 45])
        scheduled_for = base_date + timedelta(days=days_offset)
        scheduled_for = scheduled_for.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        is_peak = 1 if (7 <= hour <= 9 or 16 <= hour <= 19) else 0
        is_weekend = 1 if scheduled_for.weekday() >= 5 else 0
        
        # Determine cancellation probability
        cancel_prob = driver["base_cancellation_rate"]
        # Peak hours or weekend increases cancellation slightly due to traffic/logistics
        if is_peak:
            cancel_prob += 0.03
        if route["complexity_factor"] > 1.2:
            cancel_prob += 0.02
            
        # Is driver familiar? Familiarity reduces cancellation & delay
        is_familiar = route["route_id"] in driver["favorite_routes"]
        if is_familiar:
            cancel_prob = max(0.005, cancel_prob - 0.04)
            
        trip_status = "CANCELLED" if random.random() < cancel_prob else "COMPLETED"
        
        # Compute delay and duration
        est_dur = route["estimated_duration"]
        if trip_status == "COMPLETED":
            # Delay parameters
            mean_delay = driver["base_delay_minutes"]
            std_delay = driver["base_delay_std"]
            
            # Multipliers based on route complexity and peak hour
            multiplier = route["complexity_factor"]
            if is_peak:
                multiplier *= 1.4
            if is_familiar:
                multiplier *= 0.6  # experienced driver saves time
                
            actual_delay = max(-5.0, np.random.normal(mean_delay * multiplier, std_delay * multiplier))
            actual_duration = round(est_dur + actual_delay, 1)
            
            # Scans (passengers). Peak hours have higher passenger count
            base_passengers = int(route["estimated_distance"] * 2.5)  # average passenger density
            if is_peak:
                passenger_scans = int(base_passengers * random.uniform(1.5, 2.5))
            else:
                passenger_scans = int(base_passengers * random.uniform(0.5, 1.2))
                
            started_at = scheduled_for + timedelta(minutes=int(max(0, actual_delay - 2)))
            ended_at = started_at + timedelta(minutes=int(actual_duration))
            started_str = started_at.isoformat()
            ended_str = ended_at.isoformat()
        else:
            actual_delay = 0.0
            actual_duration = 0.0
            passenger_scans = 0
            started_str = "NULL"
            ended_str = "NULL"
            
        bus_identifier = f"BUS-{random.randint(100, 999)}"
        
        # Target assignment success labeling (0 or 1)
        # Criteria for success:
        # 1. Trip is COMPLETED (cancellations are automatically 0 success).
        # 2. Delay is within reasonable bounds (e.g. less than 15 minutes or delay < 20% of estimated duration).
        # 3. Passenger scans are non-zero (unless very late night).
        assignment_successful = 1
        if trip_status == "CANCELLED":
            assignment_successful = 0
        elif actual_delay > 18.0:
            assignment_successful = 0
        elif actual_delay > 12.0 and random.random() < 0.6:  # moderate delays sometimes count as failure
            assignment_successful = 0
            
        # Introduce a touch of pure real-world noise (5%)
        if random.random() < 0.05:
            assignment_successful = 1 - assignment_successful
            
        # We need historical features: we can calculate these dynamically or pre-inject for training.
        # Let's save the exact raw tables so the feature builder can process them,
        # but also write the structured dataset for model training directly.
        data.append({
            "trip_id": trip_id,
            "driver_id": driver["driver_id"],
            "route_id": route["route_id"],
            "status": trip_status,
            "scheduled_for": scheduled_for.isoformat(),
            "started_at": started_str,
            "ended_at": ended_str,
            "actual_duration_minutes": actual_duration,
            "bus_identifier": bus_identifier,
            "total_valid_scans": passenger_scans,
            "delay_minutes": round(actual_delay, 1),
            "assignment_successful": assignment_successful
        })
        
    df_trips = pd.DataFrame(data)
    df_trips.to_csv(config.DRIVER_PERFORMANCE_CSV, index=False)
    print(f"Generated {len(df_trips)} historical trips at {config.DRIVER_PERFORMANCE_CSV}")
    return df_trips


def generate_scan_fraud_training_data(routes, sample_size=config.FRAUD_SAMPLE_SIZE):
    """
    Generates synthetic scans with realistic timestamps (e.g. rush-hour spikes), offline, and inspection events.
    Injects 6 different types of fraudulent/anomalous scan scenarios.
    """
    scans = []
    
    # Devices list
    devices = [f"DEV-{random.randint(1000, 9999)}" for _ in range(80)]
    passengers = [f"P-{random.randint(10000, 99999)}" for _ in range(300)]
    
    base_date = datetime.now() - timedelta(days=15)
    
    for idx in range(sample_size):
        scan_id = f"S{idx+100001:06d}"
        passenger_id = random.choice(passengers)
        device_id = random.choice(devices)
        route = random.choice(routes)
        
        # Timing (rush-hours spike)
        day_offset = random.randint(0, 15)
        # Bimodal distribution for hour (rush hours at 8am and 5pm)
        if random.random() < 0.65:
            hour = int(np.random.normal(8, 1)) if random.random() < 0.5 else int(np.random.normal(17, 1))
            hour = max(6, min(21, hour))
        else:
            hour = random.randint(6, 22)
            
        minute = random.randint(0, 59)
        scanned_at = base_date + timedelta(days=day_offset, hours=hour, minutes=minute)
        
        # Standard variables
        is_offline = 1 if random.random() < 0.12 else 0
        sync_delay = random.randint(2, 25) if not is_offline else random.randint(100, 3600)
        synced_at = scanned_at + timedelta(seconds=sync_delay)
        
        # Ticket features
        fare_amount = random.choice([20.0, 35.0, 50.0])
        purchased_at = scanned_at - timedelta(minutes=random.randint(2, 60))
        expires_at = purchased_at + timedelta(hours=2)
        
        # Geographic
        # Draw a stop along the route
        target_stop = random.choice(route["stops"])
        boarding_stop_id = f"BS-{route['route_id']}-{target_stop['sequence']}"
        
        # Normally very close to target stop (GPS noise)
        lat_dev = np.random.normal(0, 0.0002)
        lon_dev = np.random.normal(0, 0.0002)
        scan_lat = round(target_stop["latitude"] + lat_dev, 5)
        scan_lon = round(target_stop["longitude"] + lon_dev, 5)
        
        # Compute GPS distance in meters roughly
        lat1, lon1 = np.radians(target_stop["latitude"]), np.radians(target_stop["longitude"])
        lat2, lon2 = np.radians(scan_lat), np.radians(scan_lon)
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        geo_deviation = round(c * 6371000, 1)  # radius of earth in meters
        
        # Base variables
        result = "VALID"
        qr_signature_valid = 1
        passenger_scan_freq = random.randint(1, 2)
        device_scan_freq = random.randint(1, 4)
        is_fraud = 0
        fraud_scenario = "NONE"
        
        # Inject fraud anomalies (8% rate overall)
        fraud_roll = random.random()
        if fraud_roll < 0.08:
            is_fraud = 1
            scenario_type = random.choice(["DUPLICATE", "GEO_DEVIATION", "DELAY_UPLOAD", "EXPIRED", "SIGNATURE", "BURST"])
            
            if scenario_type == "DUPLICATE":
                # Scenario A: Ticket sharing / QR cloning
                result = "ALREADY_USED"
                passenger_scan_freq = random.randint(6, 12)
                device_scan_freq = random.randint(8, 15)
                fraud_scenario = "Duplicate ticket scan within short time"
                
            elif scenario_type == "GEO_DEVIATION":
                # Scenario B: GPS deviation (impossible scanning location)
                scan_lat = round(scan_lat + random.uniform(0.015, 0.04), 5)  # offset considerably
                scan_lon = round(scan_lon + random.uniform(0.015, 0.04), 5)
                # recalculate deviation
                lat2, lon2 = np.radians(scan_lat), np.radians(scan_lon)
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                geo_deviation = round(c * 6371000, 1)
                fraud_scenario = "Large geo-location distance deviation"
                
            elif scenario_type == "DELAY_UPLOAD":
                # Scenario C: sync_delay manipulation (offline uploads of expired or stale transactions)
                is_offline = 1
                sync_delay = random.randint(86400 * 2, 86400 * 4)  # 2 to 4 days delay
                synced_at = scanned_at + timedelta(seconds=sync_delay)
                purchased_at = scanned_at - timedelta(hours=5)
                expires_at = purchased_at + timedelta(hours=2)
                result = "EXPIRED"
                fraud_scenario = "Suspicious delayed sync upload of expired ticket"
                
            elif scenario_type == "EXPIRED":
                # Scenario D: Expired tickets
                purchased_at = scanned_at - timedelta(hours=4)
                expires_at = purchased_at + timedelta(hours=2)
                result = "EXPIRED"
                fraud_scenario = "Expired ticket presented at boarding"
                
            elif scenario_type == "SIGNATURE":
                # Scenario E: Forged signature attempt
                qr_signature_valid = 0
                result = "INVALID_SIGNATURE"
                fraud_scenario = "Invalid cryptographic ticket signature"
                
            elif scenario_type == "BURST":
                # Scenario F: Abnormal device burst
                device_scan_freq = random.randint(25, 45)
                passenger_scan_freq = random.randint(5, 10)
                fraud_scenario = "High-frequency device boarding burst"
        
        # Format scan metadata
        scan_metadata = f'{{"latitude": {scan_lat}, "longitude": {scan_lon}, "deviceId": "{device_id}"}}'
        
        scans.append({
            "scan_id": scan_id,
            "passenger_id": passenger_id,
            "ticket_id": f"TKT-{random.randint(100000, 999999)}",
            "fare_amount": fare_amount,
            "purchased_at": purchased_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "scanned_at": scanned_at.isoformat(),
            "synced_at": synced_at.isoformat(),
            "result": result,
            "is_offline": is_offline,
            "sync_delay_seconds": sync_delay,
            "scan_metadata": scan_metadata,
            "boarding_stop_id": boarding_stop_id,
            "stop_latitude": target_stop["latitude"],
            "stop_longitude": target_stop["longitude"],
            "scan_latitude": scan_lat,
            "scan_longitude": scan_lon,
            "geo_distance_deviation_meters": geo_deviation,
            "qr_signature_valid": qr_signature_valid,
            "passenger_scan_frequency": passenger_scan_freq,
            "device_scan_frequency": device_scan_freq,
            "is_fraudulent": is_fraud,
            "fraud_scenario_description": fraud_scenario
        })

    df_scans = pd.DataFrame(scans)
    
    # Save partitioned raw context tables and direct consolidated training table
    df_scans.to_csv(config.SCAN_FRAUD_DATA_CSV, index=False)
    
    # Also save separate tables to replicate SQL exports
    df_scan_context = df_scans[["scan_id", "result", "scanned_at", "synced_at", "is_offline", "sync_delay_seconds", "scan_metadata"]]
    df_scan_context.to_csv(config.SCAN_CONTEXT_CSV, index=False)
    
    df_ticket_context = df_scans[["ticket_id", "passenger_id", "fare_amount", "purchased_at", "expires_at", "qr_signature_valid"]]
    df_ticket_context.to_csv(config.TICKET_CONTEXT_CSV, index=False)
    
    print(f"Generated {len(df_scans)} scan logs at {config.SCAN_FRAUD_DATA_CSV}")
    return df_scans

if __name__ == "__main__":
    print("Starting synthetic data generation...")
    drivers = generate_driver_profiles()
    routes = generate_route_details()
    generate_route_assignment_training_data(drivers, routes)
    generate_scan_fraud_training_data(routes)
    print("Data generation complete!")
