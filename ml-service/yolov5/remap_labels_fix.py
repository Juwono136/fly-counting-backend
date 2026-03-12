import glob

LABEL_DIR = "Dataset_Lalat_2class_aug/labels"

mapping = {
    "1": "0",  # agas
    "3": "1",  # lalat_hijau
}

for file in glob.glob(f"{LABEL_DIR}/**/*.txt", recursive=True):
    with open(file, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts[0] in mapping:
            parts[0] = mapping[parts[0]]
            new_lines.append(" ".join(parts))

    with open(file, "w") as f:
        f.write("\n".join(new_lines))

print("✅ Remapping selesai: 0=agas, 1=lalat_hijau")
