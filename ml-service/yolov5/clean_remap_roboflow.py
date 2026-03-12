import os

BASE_DIR = "Dataset_Roboflow"
LABEL_DIRS = [
    "train/labels",
    "valid/labels",
    "test/labels",
]

# mapping lama → baru
# lama 1 = agas → baru 0
# lama 3 = lalat_hijau → baru 1
MAP = {1: 0, 3: 1}

deleted_files = 0
converted_labels = 0

for split in LABEL_DIRS:
    label_path = os.path.join(BASE_DIR, split)
    image_path = label_path.replace("labels", "images")

    for file in os.listdir(label_path):
        if not file.endswith(".txt"):
            continue

        full_label = os.path.join(label_path, file)
        full_image = os.path.join(image_path, file.replace(".txt", ".jpg"))

        new_lines = []

        with open(full_label, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls = int(float(parts[0]))

                if cls in MAP:
                    parts[0] = str(MAP[cls])
                    new_lines.append(" ".join(parts))
                    converted_labels += 1

        # kalau TIDAK ADA label agas / lalat hijau → hapus file
        if not new_lines:
            os.remove(full_label)
            if os.path.exists(full_image):
                os.remove(full_image)
            deleted_files += 1
        else:
            with open(full_label, "w") as f:
                f.write("\n".join(new_lines) + "\n")

print("✅ Dataset Roboflow berhasil dibersihkan & diremap")
print(f"🗑️ File dihapus: {deleted_files}")
print(f"🔁 Label dikonversi: {converted_labels}")
print("📌 Final mapping: 0=agas, 1=lalat_hijau")
