import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Storage Directories
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Paths
DRIVER_PROFILES_CSV = DATA_DIR / "driver_profiles.csv"
DRIVER_PERFORMANCE_CSV = DATA_DIR / "driver_historical_performance.csv"
ROUTE_DETAILS_CSV = DATA_DIR / "target_route_details.csv"
ROUTE_ASSIGNMENT_DATA_CSV = DATA_DIR / "route_assignment_training_data.csv"
SCAN_CONTEXT_CSV = DATA_DIR / "scan_event_context.csv"
TICKET_CONTEXT_CSV = DATA_DIR / "ticket_context.csv"
SCAN_FRAUD_DATA_CSV = DATA_DIR / "scan_fraud_training_data.csv"

# Model Paths
ROUTE_ASSIGNMENT_MODEL_PATH = MODELS_DIR / "route_assignment_model.joblib"
FRAUD_ISOLATION_FOREST_PATH = MODELS_DIR / "fraud_isolation_forest.joblib"
FRAUD_SCALER_PATH = MODELS_DIR / "fraud_scaler.joblib"

# Training Configurations
TRAINING_SAMPLE_SIZE = int(os.getenv("TRAINING_SAMPLE_SIZE", 2000))
FRAUD_SAMPLE_SIZE = int(os.getenv("FRAUD_SAMPLE_SIZE", 5000))
ANOMALY_CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", 0.05))

# Server Configurations
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
ENV = os.getenv("ENV", "development")
