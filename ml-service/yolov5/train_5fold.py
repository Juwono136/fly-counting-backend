import os
import shutil
from sklearn.model_selection import KFold

# ===============================
# CONFIG
# ===============================
DATASET_PATH = "dataset_kfold"
IMG_SIZE = 480
EPOCHS = 150
BATCH_SIZE = 32
N_SPLITS = 5

images_path = os.path.join(DATASET_PATH, "images")
labels_path = os.path.join(DATASET_PATH, "labels")

image_files = [f for f in os.listdir(images_path) if f.endswith(".jpg") or f.endswith(".png")]

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):

    print(f"\n========== FOLD {fold+1} ==========\n")

    fold_dir = f"fold_{fold+1}"

    train_img_dir = os.path.join(fold_dir, "images/train")
    val_img_dir = os.path.join(fold_dir, "images/val")
    train_lbl_dir = os.path.join(fold_dir, "labels/train")
    val_lbl_dir = os.path.join(fold_dir, "labels/val")

    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)

    os.makedirs(train_img_dir)
    os.makedirs(val_img_dir)
    os.makedirs(train_lbl_dir)
    os.makedirs(val_lbl_dir)

    for i in train_idx:
        img = image_files[i]
        shutil.copy(os.path.join(images_path, img), train_img_dir)
        shutil.copy(os.path.join(labels_path, img.rsplit(".",1)[0] + ".txt"), train_lbl_dir)

    for i in val_idx:
        img = image_files[i]
        shutil.copy(os.path.join(images_path, img), val_img_dir)
        shutil.copy(os.path.join(labels_path, img.rsplit(".",1)[0] + ".txt"), val_lbl_dir)

    yaml_content = f"""
path: {os.path.abspath(fold_dir)}
train: images/train
val: images/val

nc: 2
names: ['Agas', 'Lalat Hijau']
"""

    yaml_path = os.path.join(fold_dir, "data.yaml")

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    os.system(
        f"python train.py --img {IMG_SIZE} "
        f"--batch {BATCH_SIZE} "
        f"--epochs {EPOCHS} "
        f"--data {yaml_path} "
        f"--weights yolov5s.pt "
        f"--name fly_fold_{fold+1}"
    )

print("===== SELESAI 5-FOLD TRAINING =====")