import os

LABEL_DIR = "labelImg_CLEAN/labels"
REMOVED = 0

for file in os.listdir(LABEL_DIR):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(LABEL_DIR, file)
    new_lines = []

    with open(path, "r") as f:
        for line in f:
            cls = int(float(line.strip().split()[0]))
            if cls != 2:
                new_lines.append(line)
            else:
                REMOVED += 1

    if new_lines:
        with open(path, "w") as f:
            f.writelines(new_lines)
    else:
        os.remove(path)  # hapus file kalau kosong

print(f"✅ Class 2 dihapus: {REMOVED} instance")
