import cv2
import os
from pathlib import Path
import albumentations as A
import shutil

# =============================
# PATH CONFIG
# =============================
SRC_DATASET = Path("Dataset_Lalat_2class")
DST_DATASET = Path("Dataset_Lalat_2class_aug")

SRC_IMG = SRC_DATASET / "images" / "train"
SRC_LBL = SRC_DATASET / "labels" / "train"

DST_IMG = DST_DATASET / "images" / "train"
DST_LBL = DST_DATASET / "labels" / "train"

# =============================
# CREATE OUTPUT FOLDERS
# =============================
DST_IMG.mkdir(parents=True, exist_ok=True)
DST_LBL.mkdir(parents=True, exist_ok=True)

# Copy validation set WITHOUT augmentation
if not (DST_DATASET / "images" / "val").exists():
    shutil.copytree(SRC_DATASET / "images" / "val", DST_DATASET / "images" / "val")
    shutil.copytree(SRC_DATASET / "labels" / "val", DST_DATASET / "labels" / "val")

# =============================
# AUGMENTATION PIPELINE (LIGHT)
# =============================
transform = A.Compose(
    [
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.08,
            contrast_limit=0.08,
            p=0.5
        ),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.3),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3
    )
)

# =============================
# COPY ORIGINAL FILES
# =============================
for img_file in SRC_IMG.glob("*.jpg"):
    shutil.copy(img_file, DST_IMG / img_file.name)
    shutil.copy(SRC_LBL / (img_file.stem + ".txt"),
                DST_LBL / (img_file.stem + ".txt"))

# =============================
# APPLY AUGMENTATION
# =============================
AUG_PER_IMAGE = 2  # <= aman untuk small object

for img_path in SRC_IMG.glob("*.jpg"):
    label_path = SRC_LBL / (img_path.stem + ".txt")
    if not label_path.exists():
        continue

    image = cv2.imread(str(img_path))
    h, w, _ = image.shape

    bboxes = []
    class_labels = []

    with open(label_path, "r") as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.strip().split())
            bboxes.append([x, y, bw, bh])
            class_labels.append(int(cls))

    for i in range(AUG_PER_IMAGE):
        augmented = transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )

        aug_img = augmented["image"]
        aug_boxes = augmented["bboxes"]
        aug_labels = augmented["class_labels"]

        if len(aug_boxes) == 0:
            continue

        out_img_name = f"{img_path.stem}_aug{i}.jpg"
        out_lbl_name = f"{img_path.stem}_aug{i}.txt"

        cv2.imwrite(str(DST_IMG / out_img_name), aug_img)

        with open(DST_LBL / out_lbl_name, "w") as f:
            for cls, box in zip(aug_labels, aug_boxes):
                f.write(f"{cls} {' '.join(map(str, box))}\n")

print("✅ Augmentasi ringan selesai dengan sukses")
