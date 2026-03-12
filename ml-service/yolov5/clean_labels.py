import os

LABEL_DIR = "../Dataset_Lalat/labels"
MAX_CLASS_ID = 4   # 5 kelas: 0–4

for root, _, files in os.walk(LABEL_DIR):
    for f in files:
        if f.endswith(".txt"):
            path = os.path.join(root, f)
            new_lines = []

            with open(path) as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    if not parts[0].isdigit():
                        continue
                    if int(parts[0]) > MAX_CLASS_ID:
                        continue
                    new_lines.append(line)

            with open(path, "w") as file:
                file.writelines(new_lines)
