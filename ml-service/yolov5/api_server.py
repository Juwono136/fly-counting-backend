from fastapi import FastAPI, File, UploadFile
import torch
import cv2
import numpy as np
import base64

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox

app = FastAPI()

# =========================
# LOAD MODEL (LOAD SEKALI SAJA)
# =========================
device = select_device('')
model = DetectMultiBackend("models/lalat_agas_best_v1.pt", device=device)
model.eval()
names = model.names


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # =========================
    # READ IMAGE
    # =========================
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # =========================
    # PREPROCESS
    # =========================
    img = letterbox(img0, 640, stride=32, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    img = img.unsqueeze(0)

    # =========================
    # INFERENCE
    # =========================
    with torch.no_grad():
        pred = model(img)

    pred = non_max_suppression(pred, 0.25, 0.45)

    count = {"Agas": 0, "Lalat Hijau": 0}

    # =========================
    # PROCESS RESULT
    # =========================
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(
                img.shape[2:], det[:, :4], img0.shape
            ).round()

            for *xyxy, conf, cls in det:
                label = names[int(cls)]

                if label in count:
                    count[label] += 1

                x1, y1, x2, y2 = map(int, xyxy)

                # Warna berbeda tiap kelas
                if label == "Agas":
                    color = (255, 0, 0)  # Biru
                else:
                    color = (0, 255, 0)  # Hijau

                cv2.rectangle(img0, (x1, y1), (x2, y2), color, 3)

                text = f"{label} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2
                )

                cv2.rectangle(
                    img0,
                    (x1, y1 - h - 10),
                    (x1 + w + 8, y1),
                    color,
                    -1
                )

                cv2.putText(
                    img0,
                    text,
                    (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

    # =========================
    # RESIZE OUTPUT (SUPAYA RINGAN)
    # =========================
    height, width = img0.shape[:2]
    max_size = 1024

    if max(height, width) > max_size:
        scale = max_size / max(height, width)
        img0 = cv2.resize(
            img0,
            (int(width * scale), int(height * scale))
        )

    # =========================
    # ENCODE JPEG (COMPRESS)
    # =========================
    _, buffer = cv2.imencode(
        '.jpg',
        img0,
        [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    )

    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # =========================
    # RESPONSE
    # =========================
    return {
        "Agas": count["Agas"],
        "Lalat Hijau": count["Lalat Hijau"],
        "image": encoded_image
    }
