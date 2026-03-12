import os
import random
import shutil

# =============================
# CONFIG
# =============================
SRC_IMAGES = "labelImg_CLEAN/images"
SRC_LABELS = "labelImg_CLEAN/labels"

OUT_BASE = "labelImg_SPLIT"
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

# =============================
# CREATE FOLDERS
# =============================
for p in [
    f"{OUT_BASE}/images/train",
    f"{OUT_BASE}/images/val",
    f"{OUT_BASE}/labels/train",
    f"{OUT_BASE}/labels/val",
]:
    os.makedirs(p, exist_ok=True)

# =============================
# COLLECT FILES
# =============================
images = [f for f in os.listdir(SRC_IMAGES)
          if f.lower().endswith((".jpg", ".png", ".jpeg"))]

print(f"📸 Total images: {len(images)}")

random.shuffle(images)

split_idx = int(len(images) * TRAIN_RATIO)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

# =============================
# COPY FUNCTION
# =============================
def copy_pair(img_list, split):
    for img in img_list:
        name, _ = os.path.splitext(img)
        lbl = name + ".txt"

        shutil.copy(
            os.path.join(SRC_IMAGES, img),
            f"{OUT_BASE}/images/{split}/{img}"
        )

        label_src = os.path.join(SRC_LABELS, lbl)
        if os.path.exists(label_src):
            shutil.copy(
                label_src,
                f"{OUT_BASE}/labels/{split}/{lbl}"
            )

# =============================
# EXECUTE
# =============================
copy_pair(train_imgs, "train")
copy_pair(val_imgs, "val")

print("✅ Split selesai!")
print(f"📦 Train images: {len(train_imgs)}")
print(f"📦 Val images  : {len(val_imgs)}")
