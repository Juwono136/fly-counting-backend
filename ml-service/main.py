from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import shutil
import os
import uuid

app = FastAPI(title="Fly Detection ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model YOLO
MODEL_PATH = "best.pt"
model = torch.hub.load(
    "./yolov5",
    "custom",
    path=MODEL_PATH,
    source="local"
)
model.conf = 0.25 

@app.get("/")
def health_check():
    return {"status": "ML Fly Detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_filename = f"/tmp/temp_{uuid.uuid4()}.jpg" # folder /tmp/ untuk keamanan di environment serverless/container

    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = model(temp_filename)
        predictions = results.pandas().xyxy[0]

        total_agas = len(predictions[predictions["class"] == 0])
        total_lalat_hijau = len(predictions[predictions["class"] == 1])

        bounding_boxes = [
            {
                "xmin": float(row["xmin"]),
                "ymin": float(row["ymin"]),
                "xmax": float(row["xmax"]),
                "ymax": float(row["ymax"]),
                "confidence": float(row["confidence"]),
                "class_id": int(row["class"]),
                "class_name": row["name"]
            }
            for _, row in predictions.iterrows()
        ]

        return {
            "total": int(len(predictions)),
            "total_agas": int(total_agas),
            "total_lalat_hijau": int(total_lalat_hijau),
            "bounding_boxes": bounding_boxes
        }

    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)