# Real-time Sensor Prediction & Online Learning System

## Project Description
This project implements a complete Machine Learning pipeline for real-time sensor monitoring, traffic prediction, and online learning. It consists of two main components:

1.  **ML Backend (Python/FastAPI):** 
    *   Hosts a PyTorch ASTGCN model
    *   Provides endpoints for real-time inference, data ingestion, and model health status
    *   Online Learning: Capable of updating weights in the background using a mixed-batch strategy (combining recent data with historical replay) without stopping the service

2.  **Autonomous Agent (Golang):** 
    *   Continuously fetches raw sensor data and requests predictions from the API
    *   Monitors performance by comparing predictions against actual readings in real-time
    *   Automatically aggregates data and triggers the backend's online training routines when a specific batch size is reached

## Configuration

### 1. Python Backend Configuration
The backend relies on environment variables and specific file placements.

**Environment Variables:**
Example of `.env` file:

```env
HOST=127.0.0.1 # The API host
PORT=8000 # The API port
```

### 2. Golang Client Configuration
Example of `.env` file:

```env
API_URL=http://localhost:8000 # URL where the API is running
SENSOR_DATA_URL=http://localhost:8000 # URL for getting sensor measurments
```

**Client Constants (`main.go`):**
Modify these hardcoded constants in the Go file if the model architecture changes:
*   `SENZOR_NUM`: (Default `228`) Number of nodes/sensors per reading.
*   `SENZOR_DATA_BATCH`: (Default `12`) Readings fetched per loop.
*   `RETRAIN_DATA_NUM`: (Default `48`) Data points collected before triggering online training.

## Usage

### Prerequisites
*   **Python:** 3.8+ with `torch`, `fastapi`, `uvicorn`, `pandas`, `numpy`.
*   **Go:** 1.16+.

### Step 1: Start the ML REST API
Run the Python server entry point. On the first run, if no saved model is found, the system will perform an initial training on fixed dataset.

```bash
pip install -r requirements
python run.py
```

### Step 2: Start the Go Client
Once the Python server is running, start the Go client. It will automatically wait for the server to report a "Ready" status before processing data.

```bash
go run main.go
```

## API Reference (Backend)

The Python Backend exposes the following REST endpoints:

*   **`GET /health`**: Returns model status (`Ready`, `Training`, `Initializing`) and device type (CPU/CUDA).
*   **`POST /predict`**: Accepts normalized sensor data and returns future predictions.
*   **`POST /add-data`**: Appends new measurements to the dataset for future training.
*   **`POST /train-online`**: Triggers a partial retraining session using the most recent data mixed with historical data.
*   **`GET /sensor-data`**: Simulates a sensor reading by returning the next line from the source CSV file.