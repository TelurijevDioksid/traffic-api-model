from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import PlainTextResponse
import numpy as np
import pandas as pd

from ml_model.config import *
from .service import model_manager
from .schemas import *

app = FastAPI()

@app.on_event("startup")
def startup_event():
    loaded = model_manager.load_weights()

    if not os.path.exists(DATA_PATH):
        p = os.path.join(DATA_DIR, "vel_full.csv")
        with open(p, "r", encoding="utf-8") as infile, open(DATA_PATH, "w", encoding="utf-8") as outfile:
            for i, line in enumerate(infile):
                if i == END_OF_INITIAL_DATA_INDEX:
                    break
                if i == 0:
                    outfile.write(line.strip())
                else:
                    outfile.write("\n" + line.strip())

    if not loaded:
        print("No existing model found. Training initial model...")
        model_manager.train()


@app.get("/health")
def health_check():
    return {"status": model_manager.status, "device": str(DEVICE) }


@app.post("/add-data")
def add_measurement(body: DataInput):
    for measurement in body.measurements:
        if len(measurement) != NUM_NODES:
            raise HTTPException(status_code=400, detail=f"Each measurement must have {NUM_NODES} values")

    try:
        model_manager.append_data(body.measurements)
        return {"message": "Data appended successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
def retrain_model(background_tasks: BackgroundTasks):
    if model_manager.status == "Initializing":
        raise HTTPException(status_code=500, detail="Model not available")

    if model_manager.status == "Training":
        raise HTTPException(status_code=500, detail="Training already in progress")

    background_tasks.add_task(model_manager.train, WINDOW_SIZE)
    return {"message": "Retraining started in background"}


@app.post("/train-online")
def retrain_online_model(body: OnlineTrain, background_tasks: BackgroundTasks):
    if model_manager.status == "Initializing":
        raise HTTPException(status_code=503, detail="Model not available")

    if model_manager.status != "Ready":
        raise HTTPException(status_code=503, detail="Model not available")

    background_tasks.add_task(model_manager.online_train, body.data_step)
    return { "message": "Online training task started" }


@app.post("/predict")
def get_prediction(body: PredictionBody):
    if model_manager.status == "Initializing":
        raise HTTPException(status_code=503, detail="Model not available")

    try:
        input_data = np.array(body.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid input data format")

    if (input_data.shape != (LEN_INPUT, NUM_NODES)):
        raise HTTPException(status_code=400, detail="Invalid input data format")

    pred = model_manager.predict(input_data)
    if pred is None:
        raise HTTPException(status_code=500, detail="Prediction failed")

    return {"predictions": pred}


csv_idx = 5001
@app.get("/sensor-data", response_class=PlainTextResponse)
def get_measurement():
    global csv_idx

    p = os.path.join(DATA_DIR, "vel_full.csv")
    try:
        with open(p, "r", encoding="utf-8") as infile:
            for i, line in enumerate(infile):
                if i == csv_idx:
                    csv_idx += 1
                    return line.strip()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSV file not found")

    raise HTTPException(status_code=404, detail="CSV index out of range")
