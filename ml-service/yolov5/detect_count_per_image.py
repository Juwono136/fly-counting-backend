import torch
import cv2
import os
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# ==============================
# CONFIG
# ==============================
weights_path = "runs/train/Counting_lalat_agas_v1/weights/best.pt"
source_folder = "Dataset_Roboflow/test/images"
output_folder = "runs/detect/per_image_count_color"
imgsz = 640
conf_thres = 0.25
iou_thres = 0.45

# ==============================
# SETUP
# ==============================
device = select_device('')
model = DetectMultiBackend(weights_path, device=device)
names = model.names

os.makedirs(output_folder, exist_ok=True)

print("\n🚀 Mulai deteksi dengan warna berbeda...\n")

# ==============================
# DETECTION LOOP
# ==============================
for img_path in Path(source_folder).glob("*.*"):

    img0 = cv2.imread(str(img_path))
    if img0 is None:
        continue

    img = cv2.resize(img0, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).copy()
    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    # Inference
    pred = model(img)
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    agas_count = 0
    lalat_count = 0

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in det:
                cls = int(cls)
                class_name = names[cls]

                # ==============================
                # WARNA BERBEDA PER KELAS
                # ==============================
                if class_name == "Agas":
                    color = (255, 0, 0)      # 🔵 Biru (BGR)
                    agas_count += 1
                elif class_name == "Lalat Hijau":
                    color = (0, 255, 0)      # 🟢 Hijau
                    lalat_count += 1
                else:
                    color = (0, 255, 255)    # Kuning fallback

                label = f"{class_name} {conf:.2f}"

                cv2.rectangle(
                    img0,
                    (int(xyxy[0]), int(xyxy[1])),
                    (int(xyxy[2]), int(xyxy[3])),
                    color,
                    2
                )

                cv2.putText(
                    img0,
                    label,
                    (int(xyxy[0]), int(xyxy[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

    # ==============================
    # TOTAL DI ATAS GAMBAR
    # ==============================
    cv2.putText(
        img0,
        f"Lalat Hijau: {lalat_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        3
    )

    cv2.putText(
        img0,
        f"Agas: {agas_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        3
    )

    save_path = os.path.join(output_folder, img_path.name)
    cv2.imwrite(save_path, img0)

    print(f"{img_path.name} → Lalat Hijau: {lalat_count} | Agas: {agas_count}")

print("\n✅ Selesai! Hasil ada di:")
print(output_folder)
